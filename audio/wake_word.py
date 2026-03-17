"""
audio/wake_word.py — Always-on Porcupine wake-word detector (FIXED v2).

FIXES:
  • Properly pause stream BEFORE callback
  • Clear audio buffer to prevent wake word being captured in STT
  • Longer delay to ensure clean listening slate
  • Better error handling
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
                    
                    # ════════════════════════════════════════════════════════
                    # CRITICAL: Pause stream to stop audio buffering
                    # ════════════════════════════════════════════════════════
                    try:
                        self._stream.stop_stream()
                        logger.debug("🛑 Stream paused - blocking new audio")
                    except Exception as e:
                        logger.warning("Stream pause error: %s", e)
                    
                    # Give time for beep to finish + user to speak
                    # This delay is CRITICAL - it clears the audio buffer
                    logger.debug("⏳ Waiting 800ms for audio buffer to clear...")
                    time.sleep(0.8)
                    
                    # Call the callback (this triggers listen_once → STT)
                    try:
                        logger.debug("📞 Calling on_wake callback...")
                        self.on_wake()
                        logger.debug("✅ Callback completed")
                    except Exception as e:
                        logger.error("Wake callback error: %s", e)
                    
                    # Resume stream after callback completes
                    try:
                        if self._running and self._stream:
                            # Small delay before resuming to ensure STT finished
                            time.sleep(0.2)
                            self._stream.start_stream()
                            logger.debug("▶️ Stream resumed - back to wake word detection")
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
