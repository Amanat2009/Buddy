"""
audio/wake_word.py — Always-on Porcupine wake-word detector (FIXED).

Runs in its own thread. When the wake word is heard it:
  1. Plays a confirmation beep
  2. Pauses the stream temporarily
  3. Calls the provided callback so the STT pipeline can kick in
  4. Resumes stream after callback completes
"""

import logging
import threading
import struct
import time
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

    # ── Public API ─────────────────────────────────────────────────────

    def start(self):
        """Initialise hardware and begin detection in background thread."""
        try:
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
            logger.info("🎙️ Wake-word detector started. Listening for '%s'…",
                        self.keyword.upper())
        except Exception as e:
            logger.error("❌ Failed to start wake-word detector: %s", e)
            raise

    def stop(self):
        """Stop the detector and clean up resources."""
        self._running = False
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                logger.warning("Stream close error: %s", e)
        if self._audio:
            self._audio.terminate()
        if self._porcupine:
            self._porcupine.delete()
        logger.info("🛑 Wake-word detector stopped.")

    # ── Internal ───────────────────────────────────────────────────────

    def _loop(self):
        """Main detection loop."""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self._running:
            try:
                raw = self._stream.read(self._porcupine.frame_length,
                                        exception_on_overflow=False)
                pcm = struct.unpack_from(
                    f"{self._porcupine.frame_length}h", raw)
                result = self._porcupine.process(pcm)
                
                # Reset error counter on successful read
                consecutive_errors = 0
                
                if result >= 0:
                    logger.info("✅ 🔔 WAKE WORD DETECTED: '%s'", self.keyword.upper())
                    play_wake_confirm()
                    
                    # Pause stream briefly to prevent mic bleed
                    try:
                        self._stream.stop_stream()
                    except Exception as e:
                        logger.warning("Stream pause error: %s", e)
                    
                    # Give a small delay for beep to finish
                    time.sleep(0.5)
                    
                    # Call the callback (this triggers listen_once)
                    try:
                        self.on_wake()
                    except Exception as e:
                        logger.error("Wake callback error: %s", e)
                    
                    # Resume stream after callback completes
                    try:
                        if self._running and self._stream:
                            self._stream.start_stream()
                    except Exception as e:
                        logger.warning("Stream resume error: %s", e)
                        
            except OSError as e:
                consecutive_errors += 1
                if self._running:
                    logger.warning("⚠️ Audio read error (%d/%d): %s", 
                                 consecutive_errors, max_consecutive_errors, e)
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error("❌ Too many consecutive errors, stopping detector")
                        self._running = False
                        break
            except Exception as e:
                consecutive_errors += 1
                if self._running:
                    logger.warning("⚠️ Unexpected error: %s", e)
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error("❌ Too many errors, stopping detector")
                        self._running = False
                        break
