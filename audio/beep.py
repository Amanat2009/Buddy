"""
audio/beep.py — Synthesised confirmation / error tones.
Uses audio.device (sounddevice) instead of simpleaudio.
All play_* functions are non-blocking daemon threads.
"""

import threading
import numpy as np
from audio.device import play_array


def _tone(freq, duration, volume=0.35, sample_rate=44100, fade_ms=12):
    t    = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t) * volume
    fade_n = int(sample_rate * fade_ms / 1000)
    wave[:fade_n]  *= np.linspace(0.0, 1.0, fade_n)
    wave[-fade_n:] *= np.linspace(1.0, 0.0, fade_n)
    return wave


def _silence(ms, sample_rate=44100):
    return np.zeros(int(sample_rate * ms / 1000))


def _async_play(wave, sample_rate=44100):
    threading.Thread(target=play_array, args=(wave, sample_rate, True), daemon=True).start()


def play_wake_confirm(sample_rate=44100):
    wave = np.concatenate([
        _tone(880,  0.07, 0.32, sample_rate),
        _silence(40, sample_rate),
        _tone(1320, 0.10, 0.32, sample_rate),
    ])
    _async_play(wave, sample_rate)


def play_listening_start(sample_rate=44100):
    _async_play(_tone(660, 0.06, 0.22, sample_rate), sample_rate)


def play_processing(sample_rate=44100):
    wave = np.concatenate([
        _tone(1000, 0.05, 0.18, sample_rate),
        _silence(30, sample_rate),
        _tone(800,  0.05, 0.18, sample_rate),
    ])
    _async_play(wave, sample_rate)


def play_error(sample_rate=44100):
    t    = np.linspace(0, 0.20, int(sample_rate * 0.20), endpoint=False)
    wave = np.sin(2 * np.pi * 220 * t) * 0.28 + np.sin(2 * np.pi * 180 * t) * 0.18
    _async_play(wave, sample_rate)


def play_timer_done(sample_rate=44100):
    wave = np.concatenate([
        _tone(523, 0.15, 0.38, sample_rate),
        _silence(80, sample_rate),
        _tone(659, 0.15, 0.38, sample_rate),
        _silence(80, sample_rate),
        _tone(784, 0.25, 0.40, sample_rate),
    ])
    _async_play(wave, sample_rate)