"""
personality/engine.py — Personality engine.

Responsibilities:
  1. Mood awareness — detects user mood and adjusts system prompt hints
  2. Context extractor — pulls goals, events, people, open loops from conversation
     using a lightweight Ollama call
  3. Concern trigger — checks if mood has been low for multiple days
"""

import json
import logging
import threading

logger = logging.getLogger("buddy.personality")


class PersonalityEngine:

    def __init__(self, llm_client, sentiment_analyser,
                 context_file, mood_log_file,
                 user_name: str = "Boss",
                 concern_threshold: float = -0.4,
                 concern_days: int = 2,
                 timezone: str = "Asia/Kolkata"):

        self.llm         = llm_client
        self.sentiment   = sentiment_analyser
        self.ctx_file    = context_file
        self.mood_file   = mood_log_file
        self.user_name   = user_name
        self.concern_threshold = concern_threshold
        self.concern_days      = concern_days
        self.timezone          = timezone

    # ── Per-message processing ─────────────────────────────────────────────

    def process_user_message(self, text: str) -> dict:
        """
        Called immediately when we receive user speech.
        Returns dict with:
          score:         float  — sentiment score
          label:         str    — positive/negative/neutral
          energy:        str    — low/medium/high
          stressed:      bool
        """
        from memory.storage import append_mood
        score   = self.sentiment.score(text)
        label   = self.sentiment.label(text)
        energy  = self.sentiment.energy_level(text)
        stressed = self.sentiment.detect_stress_keywords(text)

        append_mood(self.mood_file, score, self.timezone)

        return {
            "score":    score,
            "label":    label,
            "energy":   energy,
            "stressed": stressed,
        }

    def should_express_concern(self) -> bool:
        """
        Return True if the user has been low-mood for `concern_days` in a row.
        Trigger: proactive check-in from the scheduler.
        """
        from memory.storage import recent_mood
        recent = recent_mood(self.mood_file, n=self.concern_days * 2)
        if len(recent) < self.concern_days:
            return False
        return all(e["score"] < self.concern_threshold for e in recent[-self.concern_days:])

    # ── Background context extraction ──────────────────────────────────────

    def extract_context_async(self, user_text: str, assistant_text: str):
        """
        Fire-and-forget: ask the LLM to extract structured context from
        the conversation turn and persist it to context.json.
        """
        t = threading.Thread(
            target=self._extract_context,
            args=(user_text, assistant_text),
            daemon=True,
            name="ContextExtract"
        )
        t.start()

    def _extract_context(self, user_text: str, assistant_text: str):
        prompt = f"""Extract structured information from this conversation snippet.
Return ONLY a valid JSON object with these optional keys:
{{
  "goals": ["list of goals or intentions mentioned"],
  "open_loops": ["unresolved things that should be followed up later"],
  "people": {{"name": "relationship/description"}},
  "events": [{{"date": "YYYY-MM-DD or null", "time": "HH:MM or null", "description": "..."}}],
  "life_phase": "optional free-text description of current life context"
}}
Only include keys where you actually found something. Return {{}} if nothing notable.

User said: "{user_text}"
Assistant said: "{assistant_text}"

JSON:"""

        try:
            response = self.llm.stream(
                messages=[{"role": "user", "content": prompt}],
                on_token=None,
            )
            # Strip markdown fences if present
            clean = response.strip().strip("```json").strip("```").strip()
            data  = json.loads(clean)

            from memory.storage import update_context, save_event
            import uuid, datetime

            update_dict = {}
            if data.get("goals"):
                update_dict["goals"] = data["goals"]
            if data.get("open_loops"):
                update_dict["open_loops"] = data["open_loops"]
            if data.get("people"):
                update_dict["people"] = data["people"]
            if data.get("life_phase"):
                update_dict["life_phase"] = data["life_phase"]
            if update_dict:
                update_context(self.ctx_file, update_dict)

            for event in data.get("events", []):
                if event.get("description"):
                    event.setdefault("id", str(uuid.uuid4())[:8])
                    save_event(self.mood_file.parent / "events.json", event)

        except (json.JSONDecodeError, Exception) as e:
            logger.debug("Context extraction parse error: %s", e)
