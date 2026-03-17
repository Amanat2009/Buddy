"""
audio/stt.py — Speech-to-Text via faster-whisper.
"""
import io
import logging
import threading
import wave as wv

logger = logging.getLogger("buddy.stt")


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

    def load(self):
        from faster_whisper import WhisperModel
        logger.info("Loading faster-whisper model '%s'...", self.model)
        try:
            self._engine = WhisperModel(self.model, device="cuda", compute_type="float16")
            logger.info("faster-whisper loaded on GPU.")
        except Exception:
            self._engine = WhisperModel(self.model, device="cpu", compute_type="int8", cpu_threads=self.threads)
            logger.info("faster-whisper loaded on CPU.")

    def transcribe(self, wav_bytes: bytes) -> str:
        if not self._engine:
            return ""
        with self._lock:
            try:
                import numpy as np
                with wv.open(io.BytesIO(wav_bytes)) as wf:
                    raw = wf.readframes(wf.getnframes())
                pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                lang = self.language.split(".")[0]
                segments, _ = self._engine.transcribe(pcm, language=lang, beam_size=5, vad_filter=False)
                return " ".join(s.text.strip() for s in segments).strip()
            except Exception as e:
                logger.error("Transcription error: %s", e)
                return ""