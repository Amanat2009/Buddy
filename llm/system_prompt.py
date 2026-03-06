"""
llm/system_prompt.py — Builds the live system prompt for every LLM call.

Injected every single message (no latency cost):
  • Current date / time / timezone
  • Active timers
  • Buddy personality core
  • User context (goals, open loops, life phase)
  • Mood awareness hints
  • Memory summary
"""

import datetime
import json
import logging
from pathlib import Path

import pytz

logger = logging.getLogger("buddy.prompt")


class SystemPromptBuilder:

    def __init__(self,
                 personality_core: str,
                 buddy_name: str = "Buddy",
                 user_name: str = "Boss",
                 timezone: str = "Asia/Kolkata",
                 context_file: Path = None,
                 mood_log_file: Path = None):

        self.personality_core = personality_core
        self.buddy_name       = buddy_name
        self.user_name        = user_name
        self.tz               = pytz.timezone(timezone)
        self.context_file     = context_file
        self.mood_log_file    = mood_log_file

        # Updated externally by subsystems
        self.active_timers: list[dict]  = []    # [{"label": str, "seconds_left": int}]
        self.recent_memories: list[str] = []    # from Mem0

    # ── Public ─────────────────────────────────────────────────────────────

    def build(self) -> str:
        sections = [
            self.personality_core.strip(),
            self._time_section(),
            self._timer_section(),
            self._context_section(),
            self._mood_section(),
            self._memory_section(),
        ]
        return "\n\n".join(s for s in sections if s)

    # ── Sections ───────────────────────────────────────────────────────────

    def _time_section(self) -> str:
        now = datetime.datetime.now(self.tz)
        return (
            f"CURRENT DATE & TIME:\n"
            f"  {now.strftime('%A, %B %d %Y - %I:%M %p')} ({self.tz.zone})"
        )

    def _timer_section(self) -> str:
        if not self.active_timers:
            return ""
        lines = ["ACTIVE TIMERS:"]
        for t in self.active_timers:
            mins, secs = divmod(int(t["seconds_left"]), 60)
            label = t.get("label", "Timer")
            lines.append(f"  • {label}: {mins}m {secs}s remaining")
        return "\n".join(lines)

    def _context_section(self) -> str:
        if not self.context_file or not self.context_file.exists():
            return ""
        try:
            ctx = json.loads(self.context_file.read_text())
        except Exception:
            return ""

        parts = []
        if ctx.get("goals"):
            parts.append("ACTIVE GOALS (reference naturally, don't be weird about it):")
            for g in ctx["goals"][:5]:
                parts.append(f"  • {g}")

        if ctx.get("open_loops"):
            parts.append("OPEN LOOPS (unresolved things — follow up when appropriate):")
            for ol in ctx["open_loops"][:5]:
                parts.append(f"  • {ol}")

        if ctx.get("people"):
            parts.append("PEOPLE IN THEIR LIFE:")
            for name, desc in list(ctx["people"].items())[:8]:
                parts.append(f"  • {name}: {desc}")

        if ctx.get("life_phase"):
            parts.append(f"CURRENT LIFE PHASE: {ctx['life_phase']}")

        return "\n".join(parts) if parts else ""

    def _mood_section(self) -> str:
        if not self.mood_log_file or not self.mood_log_file.exists():
            return ""
        try:
            log = json.loads(self.mood_log_file.read_text())
            if not log:
                return ""
            # Last 3 entries
            recent = log[-3:]
            avg    = sum(e["score"] for e in recent) / len(recent)
        except Exception:
            return ""

        if avg < -0.3:
            hint = f"User has been in a low mood lately (avg score {avg:.2f}). " \
                   "Be warmer and more supportive. Don't push hard topics."
        elif avg > 0.4:
            hint = "User is in a good mood. Match the energy — be more playful."
        else:
            hint = ""

        return f"MOOD CONTEXT:\n  {hint}" if hint else ""

    def _memory_section(self) -> str:
        if not self.recent_memories:
            return ""
        lines = ["RELEVANT MEMORIES (use naturally, don't announce them):"]
        for m in self.recent_memories[:8]:
            lines.append(f"  • {m}")
        return "\n".join(lines)
