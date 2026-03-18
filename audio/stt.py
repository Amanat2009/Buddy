"""
audio/stt.py — Speech-to-Text via faster-whisper.

Changes vs original:
  • download_root points to data/whisper_cache — model downloaded once,
    never hits HuggingFace again after first run
  • local_files_only=True once cache exists — zero network latency on load
  • transcribe() auto-loads if somehow called before load() — defensive fallback
  • Cleaner logging so boot sequence shows whisper status clearly
"""

import io
import logging
import threading
import wave as wv
from pathlib import Path

logger = logging.getLogger("buddy.stt")

# Model cache lives next to the data/ directory so it survives restarts
_CACHE_DIR = Path(__file__).parent.parent / "data" / "whisper_cache"


class WhisperSTT:

    def __init__(self, model: str = "small.en",
                 language: str = "en",
                 whisper_bin: str = "",
                 n_threads: int = 4,
                 use_vulkan: bool = True):
        self.model    = model
        self.language = language
        self.threads  = n_threads
        self._engine  = None
        self._lock    = threading.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def load(self):
        """
        Load the faster-whisper model.

        First run:  downloads model to data/whisper_cache/ (~150 MB for small.en)
        After that: loads from disk with no network call (local_files_only=True)
        """
        from faster_whisper import WhisperModel

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Detect if model is already on disk to skip HuggingFace check
        model_slug      = self.model.replace("/", "--")
        cache_marker    = _CACHE_DIR / f"models--Systran--faster-whisper-{model_slug}"
        local_only      = cache_marker.exists()

        if local_only:
            logger.info("Whisper '%s' found in cache — loading offline.", self.model)
        else:
            logger.info("Whisper '%s' not cached — downloading (one-time)…", self.model)

        # Try GPU first, fall back to CPU
        try:
            self._engine = WhisperModel(
                self.model,
                device           = "cuda",
                compute_type     = "float16",
                download_root    = str(_CACHE_DIR),
                local_files_only = local_only,
                num_workers      = 1,
            )
            logger.info("faster-whisper '%s' loaded on GPU.", self.model)
        except Exception as gpu_err:
            logger.debug("GPU load failed (%s) — falling back to CPU.", gpu_err)
            try:
                self._engine = WhisperModel(
                    self.model,
                    device           = "cpu",
                    compute_type     = "int8",
                    cpu_threads      = self.threads,
                    download_root    = str(_CACHE_DIR),
                    local_files_only = local_only,
                    num_workers      = 1,
                )
                logger.info("faster-whisper '%s' loaded on CPU (int8).", self.model)
            except Exception as cpu_err:
                logger.error("faster-whisper failed to load: %s", cpu_err)
                raise

    # ── Transcription ──────────────────────────────────────────────────────

    def transcribe(self, wav_bytes: bytes) -> str:
        """
        Transcribe WAV bytes → text string.
        Auto-loads the model if load() was never called (defensive fallback).
        """
        if not self._engine:
            logger.warning("STT engine not loaded — calling load() now.")
            self.load()
        if not self._engine:
            return ""

        with self._lock:
            try:
                import numpy as np

                with wv.open(io.BytesIO(wav_bytes)) as wf:
                    raw = wf.readframes(wf.getnframes())

                pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

                # Strip the ".en" suffix if present — faster-whisper wants "en" not "en-us"
                lang = self.language.split(".")[0].split("-")[0]

                segments, _ = self._engine.transcribe(
                    pcm,
                    language   = lang,
                    beam_size  = 5,
                    vad_filter = True,
                )
                return " ".join(s.text.strip() for s in segments).strip()

            except Exception as e:
                logger.error("Transcription error: %s", e)
                return ""
