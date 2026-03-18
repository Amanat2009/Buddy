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
    for pattern, replacement in _MD_STRIP:
        text = pattern.sub(replacement, text)
    text = unicodedata.normalize("NFKC", text)
    return text.strip()


# ── Sentence splitter ─────────────────────────────────────────────────────────

_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
_MIN_CHUNK = 40   # 🔥 reduced from 60 → faster response


def split_sentences(text: str) -> list[str]:
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

    def __init__(self, engine="kokoro", voice="af_heart", speed=1.3,
                 lang="en-us"):
        self.engine = engine
        self.voice = voice
        self.speed = speed
        self.lang = lang

        self._kokoro = None
        self._lock = threading.Lock()
        self._playing = threading.Event()

    def load(self):
        from kokoro_onnx import Kokoro
        self._kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        logger.info("Kokoro TTS loaded.")

    def synthesise(self, text):
        with self._lock:
            samples, sr = self._kokoro.create(
                text,
                voice=self.voice,
                speed=self.speed,
                lang=self.lang
            )
            return samples.astype(np.float32), sr

    def speak(self, text):
        if not text.strip():
            return
        try:
            text = clean_for_tts(text)
            if not text:
                return

            audio, sr = self.synthesise(text)

            self._playing.set()
            try:
                play_array(audio, sr, blocking=True)  # ✅ KEEP TRUE (no skipping)
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


# ── STREAMER (FIXED) ─────────────────────────────────────────────────────────

class SentenceStreamer:

    def __init__(self, tts: TTSEngine):
        self.tts = tts
        self._buffer = ""
        self._q = queue.Queue()
        self._thread = None
        self._done = threading.Event()

    def start(self):
        self._done.clear()
        self._buffer = ""
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def push(self, token):
        self._buffer += token
        sentences = split_sentences(self._buffer)

        # ✅ Speak completed sentences
        for sent in sentences[:-1]:
            if sent.strip():
                self._q.put(sent)

        # keep last unfinished
        self._buffer = sentences[-1] if sentences else ""

        # 🔥 NEW: speak immediately if sentence ends
        if self._buffer.strip().endswith(('.', '!', '?')):
            self._q.put(self._buffer.strip())
            self._buffer = ""

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