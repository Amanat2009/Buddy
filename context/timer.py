"""
context/timer.py — Natural language timer system.

Detects timer intent from user speech using regex (no LLM call).
Non-blocking: uses threading.Timer.
On completion: plays a chime and speaks an alert via TTS.

Supported patterns:
  "set a timer for 20 minutes"
  "remind me in an hour"
  "give me 45 minutes"
  "timer 5 mins"
  "wake me up in 30"
  "30 second timer"
  etc.
"""

import logging
import re
import threading
import time
import uuid
from typing import Callable

logger = logging.getLogger("buddy.timer")


# ── Intent parsing ────────────────────────────────────────────────────────────

_PATTERNS = [
    # "X hours Y minutes"
    (re.compile(
        r"(\d+)\s*h(?:our)?s?\s+(?:and\s+)?(\d+)\s*m(?:in(?:ute)?s?)?",
        re.IGNORECASE
    ), lambda m: int(m.group(1)) * 3600 + int(m.group(2)) * 60),

    # "X hours"
    (re.compile(r"(\d+)\s*h(?:our)?s?", re.IGNORECASE),
     lambda m: int(m.group(1)) * 3600),

    # "X minutes" / "X mins"
    (re.compile(r"(\d+)\s*m(?:in(?:ute)?s?)?(?!\w)", re.IGNORECASE),
     lambda m: int(m.group(1)) * 60),

    # "X seconds" / "X secs"
    (re.compile(r"(\d+)\s*s(?:ec(?:ond)?s?)?(?!\w)", re.IGNORECASE),
     lambda m: int(m.group(1))),

    # bare number "in 30" or "for 30"
    (re.compile(r"(?:in|for)\s+(\d+)\b(?!\s*s)", re.IGNORECASE),
     lambda m: int(m.group(1)) * 60),
]

_TRIGGER = re.compile(
    r"\b(set|start|create|give me|remind|wake me|timer|alarm|countdown)\b",
    re.IGNORECASE
)


def parse_timer_intent(text: str) -> int | None:
    """
    Returns duration in seconds if a timer intent is detected, else None.
    """
    if not _TRIGGER.search(text):
        return None
    for pattern, extractor in _PATTERNS:
        m = pattern.search(text)
        if m:
            seconds = extractor(m)
            if 1 <= seconds <= 86400:   # sanity: 1 sec to 24 hours
                return seconds
    return None


def extract_timer_label(text: str) -> str:
    """Best-effort extraction of what the timer is for."""
    for kw in ("for my", "for the", "for a", "to", "called"):
        idx = text.lower().find(kw)
        if idx != -1:
            tail = text[idx + len(kw):].strip()
            # Grab first 3 words
            words = tail.split()[:3]
            label = " ".join(words).strip(".,!?")
            if label:
                return label.capitalize()
    return "Timer"


# ── Timer manager ─────────────────────────────────────────────────────────────

class TimerManager:
    """Manages multiple concurrent countdown timers."""

    def __init__(self, on_done: Callable[[str, str], None] = None):
        """
        on_done(label, message) — called when a timer fires.
        """
        self._timers: dict[str, dict] = {}   # id → {label, end_time, thread}
        self._lock   = threading.Lock()
        self._on_done = on_done or (lambda label, msg: None)

    def set(self, seconds: int, label: str = "Timer") -> str:
        """Create a new timer. Returns the timer ID."""
        timer_id = str(uuid.uuid4())[:8]
        end_time = time.time() + seconds

        def _fire():
            with self._lock:
                self._timers.pop(timer_id, None)
            mins, secs = divmod(seconds, 60)
            if mins:
                duration_str = f"{mins} minute{'s' if mins != 1 else ''}"
            else:
                duration_str = f"{secs} second{'s' if secs != 1 else ''}"
            msg = f"Time's up! Your {label.lower()} timer for {duration_str} is done."
            logger.info("Timer '%s' fired: %s", label, msg)
            self._on_done(label, msg)

        t = threading.Timer(seconds, _fire)
        t.daemon = True
        t.start()

        with self._lock:
            self._timers[timer_id] = {
                "label":    label,
                "end_time": end_time,
                "thread":   t,
            }

        logger.info("Timer set: '%s' for %ds (id=%s)", label, seconds, timer_id)
        return timer_id

    def cancel(self, timer_id: str) -> bool:
        with self._lock:
            info = self._timers.pop(timer_id, None)
        if info:
            info["thread"].cancel()
            return True
        return False

    def cancel_all(self):
        with self._lock:
            for info in self._timers.values():
                info["thread"].cancel()
            self._timers.clear()

    def active_timers(self) -> list[dict]:
        """Returns list of {id, label, seconds_left} for all active timers."""
        now = time.time()
        with self._lock:
            return [
                {
                    "id":           tid,
                    "label":        info["label"],
                    "seconds_left": max(0, info["end_time"] - now),
                }
                for tid, info in self._timers.items()
            ]
