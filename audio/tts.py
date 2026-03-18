"""
audio/tts.py — TTS engine (Kokoro via onnxruntime-directml or XTTS-v2).

onnxruntime-directml automatically uses AMD GPU via DirectML.
All playback goes through audio.device so beeps and TTS never overlap.
"""

import logging
import queue
import re
import threading
import unicodedata

import numpy as np
from audio.device import play_array, stop_playback

logger = logging.getLogger("buddy.tts")


# ── Markdown stripper ─────────────────────────────────────────────────────────

_MD_STRIP = [
    (re.compile(r'\*\*(.+?)\*\*'),   r'\1'),
    (re.compile(r'\*(.+?)\*'),        r'\1'),
    (re.compile(r'__(.+?)__'),        r'\1'),
    (re.compile(r'_(.+?)_'),          r'\1'),
    (re.compile(r'`+(.+?)`+'),        r'\1'),
    (re.compile(r'#+\s*'),            r''),
    (re.compile(r'^\s*[-*•]\s+', re.MULTILINE), r''),
    (re.compile(r'^\s*\d+\.\s+', re.MULTILINE), r''),
    (re.compile(r'\[(.+?)\]\(.+?\)'), r'\1'),
    (re.compile(r'[~^|]'),            r''),
    (re.compile(r'—'),                r', '),
    (re.compile(r'–'),                r', '),
    (re.compile(r'\s{2,}'),           r' '),
]


def clean_for_tts(text: str) -> str:
    """Strip markdown and special characters before sending to Kokoro."""
    for pattern, replacement in _MD_STRIP:
        text = pattern.sub(replacement, text)
    text = unicodedata.normalize("NFKC", text)
    return text.strip()


# ── Sentence splitter ─────────────────────────────────────────────────────────

_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
_MIN_CHUNK = 30


def split_sentences(text: str) -> list[str]:
    """
    Split text into speakable chunks.
    Merges short fragments together so Kokoro always gets a
    meaningful amount of text.
    """
    raw = [p.strip() for p in _SPLIT_RE.split(text.strip()) if p.strip()]
    chunks = []
    buf = ""
    for part in raw:
        buf = (buf + " " + part).strip() if buf else part
        if len(buf) >= _MIN_CHUNK:
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks


class TTSEngine:

    def __init__(self, engine="kokoro", voice="af_heart", speed=2,
                 lang="en-us", xtts_model="tts_models/multilingual/multi-dataset/xtts_v2",
                 xtts_speaker_wav="", xtts_language="en", use_gpu=True):
        self.engine           = engine
        self.voice            = voice
        self.speed            = speed
        self.lang             = lang
        self.xtts_model       = xtts_model
        self.xtts_speaker_wav = xtts_speaker_wav
        self.xtts_language    = xtts_language
        self.use_gpu          = use_gpu
        self._kokoro  = None
        self._xtts    = None
        self._lock    = threading.Lock()
        self._playing = threading.Event()

    def load(self):
        if self.engine == "kokoro":
            self._load_kokoro()
        elif self.engine == "xtts":
            self._load_xtts()
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine!r}")

    def _load_kokoro(self):
        try:
            from kokoro_onnx import Kokoro
            try:
                import onnxruntime as ort
                available = ort.get_available_providers()
                if "DmlExecutionProvider" in available:
                    logger.info("DirectML provider available — Kokoro will use AMD GPU.")
                else:
                    logger.warning("DmlExecutionProvider not found in %s — Kokoro on CPU.", available)
            except Exception:
                pass
            self._kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
            logger.info("Kokoro TTS loaded.")
        except FileNotFoundError:
            logger.error(
                "kokoro-v1.0.onnx / voices-v1.0.bin not found in project root.\n"
                "Run: python -c \"from huggingface_hub import hf_hub_download; "
                "hf_hub_download('hexgrad/Kokoro-82M','kokoro-v1.0.onnx',local_dir='.')\""
            )
            raise
        except Exception as e:
            logger.error("Failed to load Kokoro: %s", e)
            raise

    def _load_xtts(self):
        from TTS.api import TTS
        self._xtts = TTS(self.xtts_model, gpu=self.use_gpu)
        logger.info("XTTS-v2 loaded.")

    def synthesise(self, text):
        with self._lock:
            if self.engine == "kokoro":
                samples, sr = self._kokoro.create(text, voice=self.voice, speed=self.speed, lang=self.lang)
                return samples.astype(np.float32), sr
            else:
                wav = self._xtts.tts(text=text, speaker_wav=self.xtts_speaker_wav or None, language=self.xtts_language)
                return np.array(wav, dtype=np.float32), 24_000

    def speak(self, text):
        if not text.strip():
            return
        try:
            text = clean_for_tts(text)   # ✅ added cleaning
            if not text:
                return
            audio, sr = self.synthesise(text)
            self._playing.set()
            try:
                play_array(audio, sr, blocking=False)
            finally:
                self._playing.clear()
        except Exception as e:
            logger.error("TTS speak error: %s", e)

    def speak_async(self, text):
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()
        return t

    def stop(self):
        stop_playback()

    @property
    def is_playing(self):
        return self._playing.is_set()


class SentenceStreamer:

    def __init__(self, tts: TTSEngine):
        self.tts     = tts
        self._buffer = ""
        self._q      = queue.Queue()
        self._thread = None
        self._done   = threading.Event()

    def start(self):
        self._done.clear()
        self._buffer = ""
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def push(self, token):
        self._buffer += token
        sentences = split_sentences(self._buffer)
        if len(sentences) > 1:
            for sent in sentences[:-1]:
                if sent:
                    self._q.put(sent)
            self._buffer = sentences[-1]

    def finish(self):
        if self._buffer.strip():
            self._q.put(self._buffer.strip())
        self._q.put(None)

    def wait(self):
        self._done.wait()

    def stop(self):
        self.tts.stop()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._q.put(None)

    def _worker(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            self.tts.speak(item)
        self._done.set()