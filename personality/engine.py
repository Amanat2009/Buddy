"""
personality/engine.py — Personality & mood engine.

FIXES:
  - Semaphore prevents concurrent context extractions piling up in RAM.
  - Strips <think>...</think> tokens that qwen3-abliterated emits.
  - Handles empty LLM responses without crashing.
"""

import json
import logging
import re
import threading

logger = logging.getLogger("buddy.personality")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class PersonalityEngine:

    def __init__(self, llm_client, sentiment_analyser, context_file,
                 mood_log_file, user_name="Boss", concern_threshold=-0.4,
                 concern_days=2, timezone="Asia/Kolkata"):
        self.llm               = llm_client
        self.sentiment         = sentiment_analyser
        self.ctx_file          = context_file
        self.mood_file         = mood_log_file
        self.user_name         = user_name
        self.concern_threshold = concern_threshold
        self.concern_days      = concern_days
        self.timezone          = timezone
        self._extract_sem      = threading.Semaphore(1)

    def process_user_message(self, text):
        from memory.storage import append_mood
        score    = self.sentiment.score(text)
        label    = self.sentiment.label(text)
        energy   = self.sentiment.energy_level(text)
        stressed = self.sentiment.detect_stress_keywords(text)
        append_mood(self.mood_file, score, self.timezone)
        return {"score": score, "label": label, "energy": energy, "stressed": stressed}

    def should_express_concern(self):
        from memory.storage import recent_mood
        recent = recent_mood(self.mood_file, n=self.concern_days * 2)
        if len(recent) < self.concern_days:
            return False
        return all(e["score"] < self.concern_threshold for e in recent[-self.concern_days:])

    def extract_context_async(self, user_text, assistant_text):
        threading.Thread(
            target=self._extract_context_guarded,
            args=(user_text, assistant_text),
            daemon=True, name="ContextExtract"
        ).start()

    def _extract_context_guarded(self, user_text, assistant_text):
        acquired = self._extract_sem.acquire(blocking=False)
        if not acquired:
            return
        try:
            self._extract_context(user_text, assistant_text)
        finally:
            self._extract_sem.release()

    def _extract_context(self, user_text, assistant_text):
        prompt = f"""Extract structured information from this conversation snippet.
Return ONLY a valid JSON object with these optional keys:
{{
  "goals": ["list of goals or intentions mentioned"],
  "open_loops": ["unresolved things to follow up later"],
  "people": {{"name": "relationship/description"}},
  "events": [{{"date": "YYYY-MM-DD or null", "time": "HH:MM or null", "description": "..."}}],
  "life_phase": "optional free-text description"
}}
Only include keys where you found something. Return {{}} if nothing notable.

User said: "{user_text}"
Assistant said: "{assistant_text}"

JSON:"""

        try:
            raw = self.llm.stream(messages=[{"role": "user", "content": prompt}])
            if not raw or not raw.strip():
                return
            clean = _THINK_RE.sub("", raw).strip().strip("```json").strip("```").strip()
            if not clean or clean == "{}":
                return
            data = json.loads(clean)

            from memory.storage import update_context, save_event
            import uuid

            update_dict = {}
            if data.get("goals"):       update_dict["goals"]      = data["goals"]
            if data.get("open_loops"):  update_dict["open_loops"] = data["open_loops"]
            if data.get("people"):      update_dict["people"]     = data["people"]
            if data.get("life_phase"):  update_dict["life_phase"] = data["life_phase"]
            if update_dict:
                update_context(self.ctx_file, update_dict)

            for event in data.get("events", []):
                if event.get("description"):
                    event.setdefault("id", str(uuid.uuid4())[:8])
                    save_event(self.mood_file.parent / "events.json", event)

        except json.JSONDecodeError as e:
            logger.debug("Context JSON parse error: %s", e)
        except Exception as e:
            logger.warning("Context extraction error: %s", e)