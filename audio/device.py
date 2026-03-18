"""
audio/device.py — Shared, thread-safe audio output manager.

Single sounddevice output stream shared by beep.py and tts.py.
One threading.Lock prevents them from ever overlapping or corrupting
each other on the Windows WASAPI audio device.
"""

import threading
import logging

import numpy as np
import sounddevice as sd

logger = logging.getLogger("buddy.device")

_lock    = threading.Lock()
_playing = threading.Event()


def play_array(samples: np.ndarray, sample_rate: int, blocking: bool = True) -> None:
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)

    with _lock:
        _playing.set()
        try:
            sd.play(samples, samplerate=sample_rate)
            if blocking:
                sd.wait()
        except Exception as e:
            logger.warning("Audio playback error: %s", e)
        finally:
            _playing.clear()


def stop_playback() -> None:
    sd.stop()


def is_playing() -> bool:
    return _playing.is_set()