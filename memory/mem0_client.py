"""
memory/mem0_client.py — Long-term memory via Mem0 (fully local, Ollama backend).

Mem0 extracts facts, preferences, and relationship info from conversations
and stores them in a local Chroma vector DB.

At the start of each response, we query Mem0 for memories relevant to
the current user message and inject them into the system prompt.
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger("buddy.memory")


class MemoryClient:

    def __init__(self, config: dict, user_id: str = "buddy_user"):
        self.config  = config
        self.user_id = user_id
        self._mem0   = None
        self._lock   = threading.Lock()
        self._ready  = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def load(self):
        """Initialise Mem0 (may take a few seconds on first run)."""
        try:
            from mem0 import Memory
            self._mem0  = Memory.from_config(self.config)
            self._ready = True
            logger.info("Mem0 memory initialised.")
        except ImportError:
            logger.warning("mem0ai not installed — memory disabled.")
        except Exception as e:
            logger.error("Mem0 init failed: %s", e)

    # ── Public API ─────────────────────────────────────────────────────────

    def add(self, user_message: str, assistant_response: str):
        """
        Extract and store facts from a conversation turn.
        Runs in a background thread so it doesn't block the response.
        """
        if not self._ready:
            return
        messages = [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": assistant_response},
        ]
        t = threading.Thread(
            target=self._safe_add, args=(messages,), daemon=True,
            name="Mem0Add"
        )
        t.start()

    def search(self, query: str, limit: int = 8) -> list[str]:
        """
        Search memory for facts relevant to the query.
        Returns a list of plain-text memory strings.
        """
        if not self._ready:
            return []
        try:
            with self._lock:
                results = self._mem0.search(
                    query, user_id=self.user_id, limit=limit
                )
            # mem0 returns list of dicts with "memory" key
            return [r.get("memory", str(r)) for r in results]
        except Exception as e:
            logger.warning("Mem0 search failed: %s", e)
            return []

    def get_all(self) -> list[str]:
        """Return all stored memories (for debug / UI display)."""
        if not self._ready:
            return []
        try:
            with self._lock:
                results = self._mem0.get_all(user_id=self.user_id)
            return [r.get("memory", str(r)) for r in results]
        except Exception as e:
            logger.warning("Mem0 get_all failed: %s", e)
            return []

    # ── Internal ───────────────────────────────────────────────────────────

    def _safe_add(self, messages: list[dict]):
        try:
            with self._lock:
                self._mem0.add(messages, user_id=self.user_id)
            logger.debug("Memories updated.")
        except Exception as e:
            logger.warning("Mem0 add failed: %s", e)
