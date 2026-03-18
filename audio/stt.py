"""
audio/stt.py — Speech-to-Text via faster-whisper.

FIXES:
  - Removed CUDA fallback (AMD has no CUDA). Now CPU int8 directly.
  - Lazy loading: STT loads on first transcription call, not at boot.
  - Saves ~500 MB RAM until the first voice interaction.
"""

import io
import logging
import threading
import wave as wv

import numpy as np

logger = logging.getLogger("buddy.stt")


class WhisperSTT:

    def __init__(self, model="small.en", language="en",
                 whisper_bin="", n_threads=4, use_vulkan=False):
        self.model    = model
        self.language = language.split(".")[0]
        self.threads  = n_threads
        self._engine  = None
        self._lock    = threading.Lock()

    @property
    def is_loaded(self):
        return self._engine is not None

    def load(self):
        if self.is_loaded:
            return
        from faster_whisper import WhisperModel
        logger.info("Loading faster-whisper '%s' on CPU (int8)…", self.model)
        self._engine = WhisperModel(
            self.model,
            device="cpu",
            compute_type="int8",
            cpu_threads=self.threads,
            num_workers=1,
        )
        logger.info("faster-whisper ready.")

    def ensure_loaded(self):
        if not self.is_loaded:
            self.load()

    def transcribe(self, wav_bytes: bytes) -> str:
        self.ensure_loaded()
        with self._lock:
            try:
                pcm = self._wav_to_float32(wav_bytes)
                if pcm is None or len(pcm) == 0:
                    return ""
                segments, info = self._engine.transcribe(
                    pcm,
                    language=self.language if self.language != "en" else None,
                    beam_size=3,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    condition_on_previous_text=False,
                )
                text = " ".join(s.text.strip() for s in segments).strip()
                logger.debug("STT: %r (%.2fs)", text, info.duration)
                return text
            except Exception as e:
                logger.error("Transcription error: %s", e)
                return ""

    @staticmethod
    def _wav_to_float32(wav_bytes):
        try:
            with wv.open(io.BytesIO(wav_bytes)) as wf:
                raw = wf.readframes(wf.getnframes())
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:
            logger.warning("WAV decode error: %s", e)
            return None