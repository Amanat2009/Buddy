"""
audio/tts.py — Text-to-Speech engine (FIXED).

Supports two backends:
  • Kokoro-82M (onnxruntime) — fast, low VRAM, great quality   [DEFAULT]
  • Coqui XTTS-v2            — higher quality, ~2 GB VRAM

Sentence-level streaming:
  The LLM generates tokens in real-time; we break on sentence boundaries
  and begin speaking before the full response is ready.
  Use the SentenceStreamer class for this.

FIXES:
  • Strip markdown formatting (**bold**, *italic*, etc.)
  • Don't speak incomplete sentences or formatting
  • Better sentence boundary detection

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

# Markdown patterns to strip
MARKDOWN_BOLD = re.compile(r"\*\*(.+?)\*\*")
MARKDOWN_ITALIC = re.compile(r"\*(.+?)\*")
MARKDOWN_CODE = re.compile(r"`(.+?)`")
MARKDOWN_HEADER = re.compile(r"^#+\s+", re.MULTILINE)
MARKDOWN_LINK = re.compile(r"\[(.+?)\]\(.+?\)")
STANDALONE_ASTERISK = re.compile(r"(\*{1,3}|_{1,3})(?!\w)")  # Formatting markers


def clean_markdown(text: str) -> str:
    """Remove markdown formatting from text before TTS."""
    # Extract content from markdown, remove markers
    text = MARKDOWN_BOLD.sub(r"\1", text)      # **bold** → bold
    text = MARKDOWN_ITALIC.sub(r"\1", text)    # *italic* → italic
    text = MARKDOWN_CODE.sub(r"\1", text)      # `code` → code
    text = MARKDOWN_HEADER.sub("", text)       # Remove headers
    text = MARKDOWN_LINK.sub(r"\1", text)      # [link](url) → link
    text = STANDALONE_ASTERISK.sub("", text)   # Remove stray asterisks
    text = re.sub(r"[-]{2,}", "—", text)       # -- → em dash
    text = re.sub(r"#+", "", text)             # Remove stray # symbols
    text = text.strip()
    return text


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS."""
    # First clean markdown
    text = clean_markdown(text)
    
    parts = SENTENCE_END_RE.split(text.strip())
    sentences = [p.strip() for p in parts if p.strip()]
    
    # Filter out very short fragments (< 3 chars) that aren't complete
    result = []
    for s in sentences:
        # Don't include fragments like "," or "." or short prefixes
        if len(s) > 2 or s.endswith(("!", "?")):
            result.append(s)
    
    return result


# ── TTS Engine ────────────────────────────────────────────────────────────

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

    # ── Lifecycle ──────────────────────────────────────────────────────

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
            logger.info("✅ Kokoro TTS loaded.")
        except Exception as e:
            logger.error("❌ Failed to load Kokoro: %s", e)
            raise

    def _load_xtts(self):
        try:
            from TTS.api import TTS
            self._xtts = TTS(self.xtts_model, gpu=self.use_gpu)
            logger.info("✅ XTTS-v2 loaded.")
        except Exception as e:
            logger.error("❌ Failed to load XTTS-v2: %s", e)
            raise

    # ── Synthesis ──────────────────────────────────────────────────────

    def synthesise(self, text: str) -> tuple[np.ndarray, int]:
        """Return audio as float32 numpy array (24 kHz for Kokoro, 24 kHz for XTTS)."""
        # Clean markdown before synthesis
        text = clean_markdown(text)
        
        if not text.strip():
            logger.warning("⚠️ Empty text after cleaning, skipping synthesis")
            return np.array([], dtype=np.float32), 24000
        
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

    # ── Playback ───────────────────────────────────────────────────────

    def speak(self, text: str):
        """Synthesise and play synchronously."""
        if not text.strip():
            logger.warning("⚠️ Empty text, skipping speak")
            return
        try:
            audio, sr = self.synthesise(text)
            if len(audio) > 0:
                self._play(audio, sr)
        except Exception as e:
            logger.error("❌ TTS speak error: %s", e)

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

    FIXED:
    - Strips markdown before queuing
    - Doesn't speak incomplete/malformed sentences
    - Better buffer management

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
        # Accumulate tokens
        self._buffer += token
        
        # Only check for sentence boundaries if buffer is substantial
        if len(self._buffer) > self.MIN_CHARS:
            sentences = split_sentences(self._buffer)
            
            if len(sentences) > 1:
                # We have at least one complete sentence
                # Queue all but the last (incomplete) sentence
                for sent in sentences[:-1]:
                    if sent and len(sent) > 3:  # Skip very short fragments
                        self._q.put(sent)
                        logger.debug("📤 Queued for TTS: %s", sent[:60])
                
                # Keep the last incomplete sentence in buffer
                self._buffer = sentences[-1] if sentences[-1] else ""

    def finish(self):
        """Signal that the LLM is done. Flush remaining buffer."""
        if self._buffer.strip() and len(self._buffer.strip()) > 3:
            cleaned = clean_markdown(self._buffer.strip())
            if cleaned:  # Only queue if something remains after cleaning
                self._q.put(cleaned)
                logger.debug("📤 Queued final: %s", cleaned[:60])
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
        """Process sentences from queue."""
        while True:
            item = self._q.get()
            if item is None:
                break
            
            # Final cleanup before speaking
            item = clean_markdown(item.strip())
            
            if not item or len(item) < 2:
                logger.debug("⏭️ Skipping empty sentence")
                continue
            
            logger.debug("🔊 TTS speaking: %s", item[:80])
            self.tts.speak(item)
        
        self._done.set()
