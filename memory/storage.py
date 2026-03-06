"""
memory/storage.py — Simple JSON-backed persistent storage for:
  • events.json  — calendar/reminder events extracted from conversation
  • context.json — goals, open loops, people, life phase
  • mood_log.json — timestamped sentiment scores
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pytz

logger = logging.getLogger("buddy.storage")

_lock = threading.Lock()


def _read(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Storage read error (%s): %s", path, e)
        return None


def _write(path: Path, data: Any):
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    except Exception as e:
        logger.error("Storage write error (%s): %s", path, e)


# ── Events ────────────────────────────────────────────────────────────────────

def load_events(path: Path) -> list[dict]:
    data = _read(path)
    return data if isinstance(data, list) else []


def save_event(path: Path, event: dict):
    """Append a new event to events.json."""
    with _lock:
        events = load_events(path)
        events.append(event)
        _write(path, events)


def get_todays_events(path: Path, timezone: str = "Asia/Kolkata") -> list[dict]:
    tz    = pytz.timezone(timezone)
    today = datetime.now(tz).strftime("%Y-%m-%d")
    return [e for e in load_events(path) if e.get("date") == today]


def remove_event(path: Path, event_id: str):
    with _lock:
        events = [e for e in load_events(path) if e.get("id") != event_id]
        _write(path, events)


# ── Context ───────────────────────────────────────────────────────────────────

_DEFAULT_CONTEXT = {
    "goals":      [],   # list of strings
    "open_loops": [],   # list of strings
    "people":     {},   # {name: description}
    "life_phase": "",   # free-form description
}


def load_context(path: Path) -> dict:
    data = _read(path)
    if not isinstance(data, dict):
        return dict(_DEFAULT_CONTEXT)
    # Fill missing keys
    for k, v in _DEFAULT_CONTEXT.items():
        data.setdefault(k, v)
    return data


def update_context(path: Path, updates: dict):
    """Merge updates into existing context."""
    with _lock:
        ctx = load_context(path)
        for key, val in updates.items():
            if key in ("goals", "open_loops") and isinstance(val, list):
                existing = set(ctx.get(key, []))
                ctx[key] = list(existing | set(val))
            elif key == "people" and isinstance(val, dict):
                ctx["people"].update(val)
            else:
                ctx[key] = val
        _write(path, ctx)


# ── Mood Log ──────────────────────────────────────────────────────────────────

def append_mood(path: Path, score: float, timezone: str = "Asia/Kolkata"):
    """Append a timestamped sentiment score."""
    tz  = pytz.timezone(timezone)
    now = datetime.now(tz).isoformat()
    with _lock:
        log = _read(path)
        if not isinstance(log, list):
            log = []
        log.append({"ts": now, "score": score})
        # Rolling window: keep last 60 entries (~30 days at 2/day)
        _write(path, log[-60:])


def recent_mood(path: Path, n: int = 6) -> list[dict]:
    log = _read(path)
    if not isinstance(log, list):
        return []
    return log[-n:]
