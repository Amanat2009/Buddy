"""
audio/mic_input.py — Records from mic using WebRTC VAD for smart endpoint detection.

Algorithm:
  1. Open mic at 16 kHz
  2. Feed 30 ms frames into webrtcvad
  3. Speech detected  → accumulate audio frames
  4. Silence after speech → flush buffer as WAV bytes
  5. Return raw 16-bit PCM bytes (ready for whisper.cpp)

Also exposes a live RMS level for the web UI meter.
"""

import collections
import io
import logging
import struct
import threading
import time
import wave

import pyaudio
import webrtcvad

logger = logging.getLogger("buddy.mic")


class MicRecorder:
    """
    Context-manager-style recorder.

    with MicRecorder(...) as rec:
        pcm_bytes = rec.record()
    """

    FRAME_DURATION_MS = 30           # webrtcvad accepts 10, 20, or 30 ms
    SAMPLE_RATE       = 16000
    CHANNELS          = 1
    SAMPLE_WIDTH      = 2            # int16

    def __init__(self,
                 vad_aggressiveness: int = 2,
                 silence_ms: int = 900,
                 min_speech_ms: int = 300,
                 device_index=None,
                 on_level=None):       # callback(float 0-1) for UI meter
        self.vad_aggressiveness = vad_aggressiveness
        self.silence_ms         = silence_ms
        self.min_speech_ms      = min_speech_ms
        self.device_index       = device_index
        self.on_level           = on_level or (lambda x: None)

        self._frame_bytes = int(self.SAMPLE_RATE * self.FRAME_DURATION_MS
                                / 1000 * self.SAMPLE_WIDTH)
        self._vad    = webrtcvad.Vad(vad_aggressiveness)
        self._pa     = None
        self._stream = None

        # Live level (updated by the recording loop)
        self.current_level: float = 0.0

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self):
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format            = pyaudio.paInt16,
            channels          = self.CHANNELS,
            rate              = self.SAMPLE_RATE,
            input             = True,
            frames_per_buffer = self._frame_bytes // self.SAMPLE_WIDTH,
            input_device_index = self.device_index,
        )
        return self

    def __exit__(self, *_):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()

    # ── Public ─────────────────────────────────────────────────────────────

    def record(self, max_seconds: float = 30.0) -> bytes | None:
        """
        Block until speech is detected and finishes.
        Returns raw WAV bytes (16-bit / 16 kHz / mono), or None on timeout.
        """
        frames_per_silence = int(self.silence_ms / self.FRAME_DURATION_MS)
        frames_per_min_speech = int(self.min_speech_ms / self.FRAME_DURATION_MS)
        max_frames = int(max_seconds * 1000 / self.FRAME_DURATION_MS)

        # Ring buffer of recent "speech/silence" judgements for pre-roll
        ring = collections.deque(maxlen=frames_per_silence)

        speech_frames  = []
        in_speech      = False
        silence_count  = 0
        total_frames   = 0

        logger.debug("Listening…")

        while total_frames < max_frames:
            try:
                raw = self._stream.read(
                    self._frame_bytes // self.SAMPLE_WIDTH,
                    exception_on_overflow=False,
                )
            except OSError as e:
                logger.warning("Mic read error: %s", e)
                break

            total_frames += 1
            self._update_level(raw)

            is_speech = False
            try:
                is_speech = self._vad.is_speech(raw, self.SAMPLE_RATE)
            except Exception:
                pass

            ring.append((raw, is_speech))

            if not in_speech:
                if is_speech:
                    # Include the ring-buffer as pre-roll (captures leading phonemes)
                    speech_frames = [f for f, _ in ring]
                    in_speech    = True
                    silence_count = 0
                    logger.debug("Speech started.")
            else:
                speech_frames.append(raw)
                if not is_speech:
                    silence_count += 1
                    if silence_count >= frames_per_silence:
                        logger.debug("Silence end-point detected.")
                        break
                else:
                    silence_count = 0

        if len(speech_frames) < frames_per_min_speech:
            logger.debug("Recording too short, discarding.")
            return None

        self.current_level = 0.0
        self.on_level(0.0)
        return self._to_wav(b"".join(speech_frames))

    # ── Internal ───────────────────────────────────────────────────────────

    def _update_level(self, raw: bytes):
        """Compute normalised RMS and fire the UI callback."""
        samples = struct.unpack(f"{len(raw)//2}h", raw)
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        level = min(rms / 8000.0, 1.0)
        self.current_level = level
        self.on_level(level)

    def _to_wav(self, pcm: bytes) -> bytes:
        """Wrap raw PCM in a WAV container (in-memory)."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.SAMPLE_WIDTH)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(pcm)
        return buf.getvalue()
