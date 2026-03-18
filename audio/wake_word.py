"""
audio/wake_word.py — Wake-word detector (FIXED v3 - Complete).

CRITICAL FIXES:
  ✅ Proper stream pause/resume
  ✅ Extensive debugging
  ✅ Proper callback invocation
  ✅ Error recovery
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
        logger.info("🔧 Initializing WakeWordDetector for keyword: %s", keyword)
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
            logger.info("📡 Creating Porcupine instance...")
            self._porcupine = pvporcupine.create(
                access_key  = self.access_key,
                keywords    = [self.keyword],
                sensitivities = [self.sensitivity],
            )
            logger.info("✅ Porcupine created (sample_rate=%d, frame_length=%d)",
                       self._porcupine.sample_rate, self._porcupine.frame_length)

            logger.info("🎙️ Opening audio stream...")
            self._audio = pyaudio.PyAudio()
            self._stream = self._audio.open(
                rate            = self._porcupine.sample_rate,
                channels        = 1,
                format          = pyaudio.paInt16,
                input           = True,
                frames_per_buffer = self._porcupine.frame_length,
                input_device_index = self.device_index,
            )
            logger.info("✅ Audio stream opened")

            self._running = True
            self._thread  = threading.Thread(target=self._loop, daemon=True,
                                             name="WakeWordDetector")
            self._thread.start()
            logger.info("✅ Wake-word detection thread started. Listening for '%s'...",
                        self.keyword.upper())
        except Exception as e:
            logger.error("❌ Failed to start wake-word detector: %s", e, exc_info=True)
            raise

    def stop(self):
        """Stop the detector and clean up resources."""
        logger.info("🛑 Stopping wake-word detector...")
        self._running = False
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
                logger.debug("✅ Stream closed")
            except Exception as e:
                logger.warning("Stream close error: %s", e)
        if self._audio:
            self._audio.terminate()
            logger.debug("✅ PyAudio terminated")
        if self._porcupine:
            self._porcupine.delete()
            logger.debug("✅ Porcupine deleted")
        logger.info("✅ Wake-word detector stopped")

    # ── Internal ───────────────────────────────────────────────────────

    def _loop(self):
        """Main detection loop with error recovery."""
        logger.info("▶️ Wake-word detection loop started")
        consecutive_errors = 0
        max_consecutive_errors = 10
        detection_count = 0
        
        while self._running:
            try:
                # Read audio frame
                raw = self._stream.read(self._porcupine.frame_length,
                                        exception_on_overflow=False)
                pcm = struct.unpack_from(
                    f"{self._porcupine.frame_length}h", raw)
                
                # Process audio
                result = self._porcupine.process(pcm)
                
                # Reset error counter on success
                consecutive_errors = 0
                
                # Check if wake word detected
                if result >= 0:
                    detection_count += 1
                    logger.warning("╔════════════════════════════════════════╗")
                    logger.warning("║  🔔 WAKE WORD DETECTED! #%d           ║", detection_count)
                    logger.warning("║  '%s'", self.keyword.upper())
                    logger.warning("╚════════════════════════════════════════╝")
                    
                    # Play confirmation beep
                    logger.info("🔊 Playing wake confirmation beep...")
                    play_wake_confirm()
                    
                    # CRITICAL: Pause stream BEFORE callback
                    logger.info("🛑 PAUSING STREAM to clear audio buffer...")
                    try:
                        self._stream.stop_stream()
                        logger.debug("✅ Stream stopped")
                    except Exception as e:
                        logger.warning("⚠️ Stream pause error: %s", e)
                    
                    # Wait for beep to finish AND audio buffer to clear
                    logger.info("⏳ Waiting 800ms for buffer clear + beep finish...")
                    time.sleep(0.8)
                    logger.debug("✅ Wait complete")
                    
                    # Invoke callback (this should call listen_once)
                    logger.info("📞 CALLING on_wake() CALLBACK...")
                    try:
                        self.on_wake()
                        logger.info("✅ on_wake() callback completed")
                    except Exception as e:
                        logger.error("❌ on_wake() callback failed: %s", e, exc_info=True)
                    
                    # Wait for STT to finish
                    logger.info("⏳ Waiting 200ms for STT to start...")
                    time.sleep(0.2)
                    
                    # Resume stream
                    logger.info("▶️ RESUMING STREAM for next wake word detection...")
                    try:
                        if self._running and self._stream:
                            self._stream.start_stream()
                            logger.info("✅ Stream resumed - back to detection mode")
                    except Exception as e:
                        logger.warning("⚠️ Stream resume error: %s", e)
                        
            except OSError as e:
                consecutive_errors += 1
                logger.warning("⚠️ Audio read error (%d/%d): %s", 
                             consecutive_errors, max_consecutive_errors, e)
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("❌ Too many consecutive errors, stopping detector")
                    self._running = False
                    break
                    
            except Exception as e:
                consecutive_errors += 1
                logger.error("❌ Unexpected error in loop (%d/%d): %s", 
                           consecutive_errors, max_consecutive_errors, e, exc_info=True)
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("❌ Too many errors, stopping detector")
                    self._running = False
                    break
        
        logger.info("⏹️ Wake-word detection loop stopped")
