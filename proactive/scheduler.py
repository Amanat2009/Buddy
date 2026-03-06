"""
proactive/scheduler.py — Proactive engagement engine.

Buddy talks on its own, unprompted. This module runs as a background thread
and checks multiple signals to decide when and what to say.

Signals checked:
  1. Morning/evening schedule (daily check-ins)
  2. events.json — today's events needing a reminder
  3. Mood pattern — concern trigger if user has been low for days
  4. Silence detection — if user hasn't talked in a long time (optional)

When triggered: generates an opening via LLM and speaks it unprompted.
"""

import datetime
import logging
import random
import threading
import time
from typing import Callable

import pytz
import schedule

logger = logging.getLogger("buddy.proactive")


class ProactiveScheduler:

    def __init__(self,
                 llm_client,
                 mem_client,
                 personality_engine,
                 tts_engine,
                 context_file,
                 events_file,
                 buddy_name: str = "Buddy",
                 user_name: str = "Boss",
                 timezone: str = "Asia/Kolkata",
                 morning_hour: int = 9,
                 evening_hour: int = 20,
                 on_speak: Callable[[str], None] = None):

        self.llm         = llm_client
        self.mem         = mem_client
        self.personality = personality_engine
        self.tts         = tts_engine
        self.ctx_file    = context_file
        self.events_file = events_file
        self.buddy_name  = buddy_name
        self.user_name   = user_name
        self.tz          = pytz.timezone(timezone)
        self.morning_hour = morning_hour
        self.evening_hour = evening_hour
        self.on_speak    = on_speak or (lambda s: None)

        self._thread  = None
        self._running = False

        # Track last proactive message time (don't spam)
        self._last_proactive: float = 0
        self._min_interval = 60 * 30   # minimum 30 min between proactive messages

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        schedule.every().day.at(f"{self.morning_hour:02d}:00").do(
            self._morning_check_in)
        schedule.every().day.at(f"{self.evening_hour:02d}:00").do(
            self._evening_check_in)
        schedule.every(15).minutes.do(self._event_check)
        schedule.every(60).minutes.do(self._mood_check)

        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="ProactiveScheduler"
        )
        self._thread.start()
        logger.info("Proactive scheduler started.")

    def stop(self):
        self._running = False
        schedule.clear()

    # ── Scheduler loop ─────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            schedule.run_pending()
            time.sleep(15)

    # ── Check-ins ──────────────────────────────────────────────────────────

    def _morning_check_in(self):
        if not self._can_speak():
            return
        today_events = self._get_todays_events()
        memories     = self.mem.search("morning routine goals plans") if self.mem else []

        prompt = self._build_opener_prompt(
            occasion="morning check-in",
            extra_context={
                "today_events": today_events,
                "memories": memories[:4],
            }
        )
        self._generate_and_speak(prompt)

    def _evening_check_in(self):
        if not self._can_speak():
            return
        memories = self.mem.search("evening reflection day") if self.mem else []

        prompt = self._build_opener_prompt(
            occasion="evening check-in",
            extra_context={"memories": memories[:4]}
        )
        self._generate_and_speak(prompt)

    def _event_check(self):
        """Remind about events happening soon (within 30 min)."""
        if not self._can_speak():
            return
        now    = datetime.datetime.now(self.tz)
        events = self._get_todays_events()
        for ev in events:
            ev_time = ev.get("time")
            if not ev_time:
                continue
            try:
                hour, minute = map(int, ev_time.split(":"))
                ev_dt = now.replace(hour=hour, minute=minute, second=0)
                delta = (ev_dt - now).total_seconds()
                if 0 < delta <= 30 * 60:
                    msg = (
                        f"Hey, heads-up — you've got '{ev['description']}' "
                        f"in {int(delta//60)} minutes."
                    )
                    self._speak(msg)
                    return
            except Exception:
                pass

    def _mood_check(self):
        """Proactively check in if mood has been low."""
        if not self._can_speak():
            return
        if self.personality.should_express_concern():
            prompt = self._build_opener_prompt(
                occasion="gentle concern check-in",
                extra_context={"mood": "user has seemed down lately"}
            )
            self._generate_and_speak(prompt)

    # ── LLM opener generation ──────────────────────────────────────────────

    def _build_opener_prompt(self, occasion: str, extra_context: dict) -> str:
        ctx_parts = [f"Occasion: {occasion}"]
        for k, v in extra_context.items():
            if v:
                ctx_parts.append(f"{k}: {v}")

        return f"""Generate a SHORT, natural, unprompted opening from {self.buddy_name} to {self.user_name}.
This is a {occasion}.

Context:
{chr(10).join(ctx_parts)}

Rules:
- 1-2 sentences MAX
- Sound completely natural, like a friend starting a conversation
- Don't announce that you're checking in — just start talking
- Match the personality: dry wit, warm, direct
- Don't ask multiple questions — one thing at most
- Don't say "Good morning" robotically

Just write the opening line(s), nothing else:"""

    def _generate_and_speak(self, prompt: str):
        try:
            response = self.llm.stream(
                messages=[{"role": "user", "content": prompt}],
            )
            if response.strip():
                self._speak(response.strip())
        except Exception as e:
            logger.error("Proactive generation error: %s", e)

    def _speak(self, text: str):
        logger.info("Proactive speak: %s", text[:80])
        self._last_proactive = time.time()
        self.on_speak(text)
        self.tts.speak_async(text)

    # ── Utilities ──────────────────────────────────────────────────────────

    def _can_speak(self) -> bool:
        return (time.time() - self._last_proactive) >= self._min_interval

    def _get_todays_events(self) -> list[dict]:
        from memory.storage import get_todays_events
        try:
            return get_todays_events(self.events_file, str(self.tz))
        except Exception:
            return []
