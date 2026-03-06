"""
audio/tts.py — Text-to-Speech engine.

Supports two backends:
  • Kokoro-82M (onnxruntime) — fast, low VRAM, great quality   [DEFAULT]
  • Coqui XTTS-v2            — higher quality, ~2 GB VRAM

Sentence-level streaming:
  The LLM generates tokens in real-time; we break on sentence boundaries
  and begin speaking before the full response is ready.
  Use the SentenceStreamer class for this.

Public API:
  tts = TTSEngine(engine="kokoro")
  tts.load()
  tts.speak("Hello there, friend.")          # blocking
  tts.speak_async("Hello there, friend.")    # non-blocking thread
"""

import io
import logging
import queue
import re
import threading

import numpy as np
import sounddevice as sd

logger = logging.getLogger("buddy.tts")


# ── Helpers ───────────────────────────────────────────────────────────────────

SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\'])|(?<=[.!?])$")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS."""
    parts = SENTENCE_END_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ── TTS Engine ────────────────────────────────────────────────────────────────

class TTSEngine:

    def __init__(self, engine: str = "kokoro",
                 voice: str = "af_heart",
                 speed: float = 1.1,
                 lang: str = "en-us",
                 xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2",
                 xtts_speaker_wav: str = "",
                 xtts_language: str = "en",
                 use_gpu: bool = True):

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

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def load(self):
        """Load the TTS model into memory / VRAM."""
        if self.engine == "kokoro":
            self._load_kokoro()
        elif self.engine == "xtts":
            self._load_xtts()
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine}")

    def _load_kokoro(self):
        try:
            from kokoro_onnx import Kokoro
            self._kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
            logger.info("Kokoro TTS loaded.")
        except Exception as e:
            logger.error("Failed to load Kokoro: %s", e)
            raise

    def _load_xtts(self):
        try:
            from TTS.api import TTS
            self._xtts = TTS(self.xtts_model, gpu=self.use_gpu)
            logger.info("XTTS-v2 loaded.")
        except Exception as e:
            logger.error("Failed to load XTTS-v2: %s", e)
            raise

    # ── Synthesis ──────────────────────────────────────────────────────────

    def synthesise(self, text: str) -> np.ndarray:
        """Return audio as float32 numpy array (24 kHz for Kokoro, 24 kHz for XTTS)."""
        with self._lock:
            if self.engine == "kokoro":
                return self._synth_kokoro(text)
            else:
                return self._synth_xtts(text)

    def _synth_kokoro(self, text: str) -> np.ndarray:
        samples, sample_rate = self._kokoro.create(
            text, voice=self.voice, speed=self.speed, lang=self.lang
        )
        return samples.astype(np.float32), sample_rate

    def _synth_xtts(self, text: str) -> np.ndarray:
        wav = self._xtts.tts(
            text=text,
            speaker_wav=self.xtts_speaker_wav or None,
            language=self.xtts_language,
        )
        sample_rate = 24000
        return np.array(wav, dtype=np.float32), sample_rate

    # ── Playback ───────────────────────────────────────────────────────────

    def speak(self, text: str):
        """Synthesise and play synchronously."""
        if not text.strip():
            return
        try:
            audio, sr = self.synthesise(text)
            self._play(audio, sr)
        except Exception as e:
            logger.error("TTS speak error: %s", e)

    def speak_async(self, text: str):
        """Synthesise and play in a background thread."""
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()
        return t

    def _play(self, audio: np.ndarray, sample_rate: int):
        """Play audio via sounddevice, blocking until done."""
        self._playing.set()
        try:
            sd.play(audio, samplerate=sample_rate, blocking=True)
        finally:
            self._playing.clear()

    def stop(self):
        """Stop any current playback."""
        sd.stop()

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()


# ── Sentence Streamer (for LLM streaming) ─────────────────────────────────────

class SentenceStreamer:
    """
    Feeds partial LLM tokens in, plays TTS sentence-by-sentence.

    Usage:
        streamer = SentenceStreamer(tts_engine)
        streamer.start()
        for token in llm.stream(...):
            streamer.push(token)
        streamer.finish()
        streamer.wait()
    """

    MIN_CHARS = 40   # don't bother speaking until we have at least N chars

    def __init__(self, tts: TTSEngine):
        self.tts      = tts
        self._buffer  = ""
        self._q       = queue.Queue()
        self._thread  = None
        self._done    = threading.Event()

    def start(self):
        self._done.clear()
        self._buffer = ""
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def push(self, token: str):
        """Feed a new token from the LLM."""
        self._buffer += token
        # Flush on sentence boundary if buffer is long enough
        sentences = split_sentences(self._buffer)
        if len(sentences) > 1:
            # Keep the last (incomplete) sentence in buffer
            for sent in sentences[:-1]:
                if sent:
                    self._q.put(sent)
            self._buffer = sentences[-1]

    def finish(self):
        """Signal that the LLM is done. Flush remaining buffer."""
        if self._buffer.strip():
            self._q.put(self._buffer.strip())
        self._q.put(None)   # sentinel

    def wait(self):
        """Block until all queued sentences have been spoken."""
        self._done.wait()

    def stop(self):
        """Abort immediately."""
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
            logger.debug("TTS speaking: %s", item[:60])
            self.tts.speak(item)
        self._done.set()
