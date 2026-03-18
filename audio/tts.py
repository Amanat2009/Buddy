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


def clean_for_tts(text: str) -> str:
    """
    Strip markdown and symbols that TTS engines speak literally.

    Handles:
      **bold**, *italic*, ***bold-italic***
      __underline__, _italic_
      `inline code`, ```code blocks```
      # Headers
      - bullet points, > blockquotes
      [link text](url)  →  keeps the label, drops the URL
      Excessive whitespace
    """
    # Code blocks (multi-line) — drop entirely, they're unreadable as speech
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Inline code
    text = re.sub(r'`[^`]*`', '', text)
    # Bold + italic (order matters: *** before ** before *)
    text = re.sub(r'\*{3}(.*?)\*{3}', r'\1', text)
    text = re.sub(r'\*{2}(.*?)\*{2}', r'\1', text)
    text = re.sub(r'\*(.*?)\*',        r'\1', text)
    # Underscore emphasis
    text = re.sub(r'_{2}(.*?)_{2}', r'\1', text)
    text = re.sub(r'_(.*?)_',       r'\1', text)
    # Headers (# ## ### etc.)
    text = re.sub(r'^\s*#{1,6}\s*', '', text, flags=re.MULTILINE)
    # Bullet points and blockquotes at line start
    text = re.sub(r'^\s*[-*>•]\s+', '', text, flags=re.MULTILINE)
    # Numbered list markers  "1. " "12. "
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Links — keep the label
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Horizontal rules
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Collapse multiple blank lines / spaces
    text = re.sub(r'\n{2,}', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


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

    def synthesise(self, text: str) -> tuple[np.ndarray, int]:
        """
        Return audio as (float32 numpy array, sample_rate).
        Input text should already be cleaned via clean_for_tts().
        """
        with self._lock:
            if self.engine == "kokoro":
                return self._synth_kokoro(text)
            else:
                return self._synth_xtts(text)

    def _synth_kokoro(self, text: str) -> tuple[np.ndarray, int]:
        samples, sample_rate = self._kokoro.create(
            text, voice=self.voice, speed=self.speed, lang=self.lang
        )
        return samples.astype(np.float32), sample_rate

    def _synth_xtts(self, text: str) -> tuple[np.ndarray, int]:
        wav = self._xtts.tts(
            text=text,
            speaker_wav=self.xtts_speaker_wav or None,
            language=self.xtts_language,
        )
        sample_rate = 24000
        return np.array(wav, dtype=np.float32), sample_rate

    # ── Playback ───────────────────────────────────────────────────────────

    def speak(self, text: str):
        """Clean, synthesise, and play synchronously."""
        if not text.strip():
            return
        try:
            text = clean_for_tts(text)
            if not text:
                return
            audio, sr = self.synthesise(text)
            self._play(audio, sr)
        except Exception as e:
            logger.error("TTS speak error: %s", e)

    def speak_async(self, text: str) -> threading.Thread:
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
    Feeds partial LLM tokens in, plays TTS sentence-by-sentence with no gaps.

    Architecture — two worker threads running in parallel:

      push(token) → _synth_q → [synth thread] → _play_q → [play thread]

    The synth thread stays one sentence ahead of the play thread.
    While sentence N is playing, sentence N+1 is already being synthesised,
    so there is zero silence between sentences.

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
        self.tts        = tts
        self._buffer    = ""
        self._synth_q   = queue.Queue()   # str  → synth worker
        self._play_q    = queue.Queue()   # (ndarray, int) → play worker
        self._synth_t   = None
        self._play_t    = None
        self._done      = threading.Event()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        self._done.clear()
        self._buffer = ""
        self._synth_t = threading.Thread(
            target=self._synth_worker, daemon=True, name="TTS-Synth"
        )
        self._play_t = threading.Thread(
            target=self._play_worker, daemon=True, name="TTS-Play"
        )
        self._synth_t.start()
        self._play_t.start()

    def push(self, token: str):
        """Feed a new token from the LLM."""
        self._buffer += token
        sentences = split_sentences(self._buffer)
        if len(sentences) > 1:
            # All complete sentences go to synthesis immediately
            for sent in sentences[:-1]:
                cleaned = clean_for_tts(sent)
                if cleaned:
                    self._synth_q.put(cleaned)
            # Keep the trailing incomplete sentence in the buffer
            self._buffer = sentences[-1]

    def finish(self):
        """Signal that the LLM is done streaming. Flush remaining buffer."""
        if self._buffer.strip():
            cleaned = clean_for_tts(self._buffer.strip())
            if cleaned:
                self._synth_q.put(cleaned)
        self._synth_q.put(None)   # sentinel — tells synth thread to stop

    def wait(self):
        """Block until all queued sentences have been spoken."""
        self._done.wait()

    def stop(self):
        """Abort immediately — drain queues and stop audio."""
        self.tts.stop()
        for q in (self._synth_q, self._play_q):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        # Send sentinels so both workers exit cleanly
        self._synth_q.put(None)
        self._play_q.put(None)

    # ── Workers ────────────────────────────────────────────────────────────

    def _synth_worker(self):
        """
        Synthesises audio as fast as possible.
        Runs ahead of playback — keeps _play_q pre-loaded so the play
        worker never has to wait.
        """
        while True:
            item = self._synth_q.get()
            if item is None:
                # Forward the sentinel so the play worker also exits
                self._play_q.put(None)
                break
            try:
                logger.debug("TTS synthesising: %.60s", item)
                audio, sr = self.tts.synthesise(item)
                self._play_q.put((audio, sr))
            except Exception as e:
                logger.error("TTS synth error: %s", e)
                # Don't crash — skip this sentence and keep going

    def _play_worker(self):
        """
        Plays pre-synthesised audio back-to-back with zero gap.
        Blocks on each chunk until playback is complete, then immediately
        grabs the next one (which is usually already ready in the queue).
        """
        while True:
            item = self._play_q.get()
            if item is None:
                break
            audio, sr = item
            try:
                self.tts._play(audio, sr)
            except Exception as e:
                logger.error("TTS playback error: %s", e)
        # Signal that all speech is done
        self._done.set()
