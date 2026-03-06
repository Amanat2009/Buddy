"""
audio/beep.py — Generates confirmation / error beep tones in-code.
No audio files needed — everything is synthesised with numpy.
"""

import numpy as np
import simpleaudio as sa
import threading


def _play_wave(samples: np.ndarray, sample_rate: int = 44100):
    """Play a numpy array as audio (non-blocking)."""
    audio = (samples * 32767).astype(np.int16)
    play_obj = sa.play_buffer(audio, 1, 2, sample_rate)
    play_obj.wait_done()


def _tone(freq: float, duration: float, volume: float = 0.4,
          sample_rate: int = 44100, fade_ms: int = 10) -> np.ndarray:
    """Synthesise a sine-wave tone with a short fade in/out to avoid clicks."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t) * volume

    fade_samples = int(sample_rate * fade_ms / 1000)
    fade_in  = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    wave[:fade_samples]  *= fade_in
    wave[-fade_samples:] *= fade_out
    return wave


def play_wake_confirm(sample_rate: int = 44100):
    """Two ascending tones — heard when wake word is detected."""
    def _run():
        silence = np.zeros(int(sample_rate * 0.04))
        wave = np.concatenate([
            _tone(880, 0.07, volume=0.35, sample_rate=sample_rate),
            silence,
            _tone(1320, 0.10, volume=0.35, sample_rate=sample_rate),
        ])
        _play_wave(wave, sample_rate)

    threading.Thread(target=_run, daemon=True).start()


def play_listening_start(sample_rate: int = 44100):
    """Single soft bloop — recording started."""
    def _run():
        _play_wave(_tone(660, 0.06, volume=0.25, sample_rate=sample_rate),
                   sample_rate)

    threading.Thread(target=_run, daemon=True).start()


def play_processing(sample_rate: int = 44100):
    """Subtle descending blip — STT done, LLM thinking."""
    def _run():
        silence = np.zeros(int(sample_rate * 0.03))
        wave = np.concatenate([
            _tone(1000, 0.05, volume=0.2, sample_rate=sample_rate),
            silence,
            _tone(800, 0.05, volume=0.2, sample_rate=sample_rate),
        ])
        _play_wave(wave, sample_rate)

    threading.Thread(target=_run, daemon=True).start()


def play_error(sample_rate: int = 44100):
    """Harsh low buzz — something went wrong."""
    def _run():
        t = np.linspace(0, 0.18, int(sample_rate * 0.18), endpoint=False)
        wave = (
            np.sin(2 * np.pi * 220 * t) * 0.3
            + np.sin(2 * np.pi * 180 * t) * 0.2
        )
        _play_wave(wave, sample_rate)

    threading.Thread(target=_run, daemon=True).start()


def play_timer_done(sample_rate: int = 44100):
    """Three ascending chimes — timer finished."""
    def _run():
        silence = np.zeros(int(sample_rate * 0.08))
        wave = np.concatenate([
            _tone(523, 0.15, volume=0.4, sample_rate=sample_rate), silence,
            _tone(659, 0.15, volume=0.4, sample_rate=sample_rate), silence,
            _tone(784, 0.25, volume=0.4, sample_rate=sample_rate),
        ])
        _play_wave(wave, sample_rate)

    threading.Thread(target=_run, daemon=True).start()
