"""
audio/wake_word.py — Always-on Porcupine wake-word detector.

Runs in its own thread. When the wake word is heard it:
  1. Plays a confirmation beep
  2. Calls the provided callback so the STT pipeline can kick in
"""

import logging
import threading
import struct
import pvporcupine
import pyaudio

from audio.beep import play_wake_confirm

logger = logging.getLogger("buddy.wake_word")


class WakeWordDetector:
    """Wraps Porcupine + PyAudio for always-on detection."""

    def __init__(self, access_key: str, keyword: str = "jarvis",
                 sensitivity: float = 0.6, device_index=None,
                 on_wake=None):
        self.access_key   = access_key
        self.keyword      = keyword
        self.sensitivity  = sensitivity
        self.device_index = device_index
        self.on_wake      = on_wake or (lambda: None)

        self._porcupine   = None
        self._audio       = None
        self._stream      = None
        self._thread      = None
        self._running     = False

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self):
        """Initialise hardware and begin detection in background thread."""
        self._porcupine = pvporcupine.create(
            access_key  = self.access_key,
            keywords    = [self.keyword],
            sensitivities = [self.sensitivity],
        )

        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            rate            = self._porcupine.sample_rate,
            channels        = 1,
            format          = pyaudio.paInt16,
            input           = True,
            frames_per_buffer = self._porcupine.frame_length,
            input_device_index = self.device_index,
        )

        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True,
                                         name="WakeWordDetector")
        self._thread.start()
        logger.info("Wake-word detector started. Listening for '%s'…",
                    self.keyword)

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._audio:
            self._audio.terminate()
        if self._porcupine:
            self._porcupine.delete()
        logger.info("Wake-word detector stopped.")

    # ── Internal ───────────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            try:
                raw = self._stream.read(self._porcupine.frame_length,
                                        exception_on_overflow=False)
                pcm = struct.unpack_from(
                    f"{self._porcupine.frame_length}h", raw)
                result = self._porcupine.process(pcm)
                if result >= 0:
                    logger.info("Wake word detected!")
                    play_wake_confirm()
                    # Pause the stream so the mic doesn't bleed into recording
                    self._stream.stop_stream()
                    try:
                        self.on_wake()
                    finally:
                        if self._running:
                            self._stream.start_stream()
            except OSError as e:
                if self._running:
                    logger.warning("Audio read error: %s", e)
