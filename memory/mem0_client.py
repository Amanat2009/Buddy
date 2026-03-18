"""
memory/mem0_client.py — Long-term memory via Mem0 (fully local, Ollama backend).

Mem0 extracts facts, preferences, and relationship info from conversations
and stores them in a local Chroma vector DB that persists across restarts.

Fixes vs original:
  • Handles mem0ai >=0.1.x API change: search() now returns
    {"results": [...]} instead of a flat list.
  • Handles both old dict format {"memory": "..."} and new format
    {"memory": "...", "id": "...", ...} defensively.
  • get_all() similarly updated for new API shape.
  • Path sanity-check on startup — logs where Chroma is persisting data
    so you can confirm it's actually writing to disk.
  • All result parsing goes through a single _extract_memories() helper
    so there's one place to fix if mem0ai changes again.
"""

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("buddy.memory")


def _extract_memories(raw: Any) -> list[str]:
    """
    Safely pull memory strings out of whatever mem0ai returns.

    mem0ai has changed its return format across versions:

      v0.0.x  →  [{"memory": "...", ...}, ...]          (flat list of dicts)
      v0.1.x  →  {"results": [{"memory": "...", ...}]}  (dict with "results" key)
      unknown →  ["string", ...]                         (list of strings, fallback)

    This function handles all three without crashing.
    """
    # Unwrap {"results": [...]} envelope (mem0ai >= 0.1.x)
    if isinstance(raw, dict):
        raw = raw.get("results", raw.get("memories", []))

    if not isinstance(raw, list):
        logger.debug("Unexpected mem0 response type: %s", type(raw))
        return []

    out = []
    for item in raw:
        if isinstance(item, dict):
            # Standard shape — "memory" key holds the text
            text = item.get("memory") or item.get("text") or item.get("content")
            if text:
                out.append(str(text))
            else:
                # Unknown dict shape — stringify the whole thing as fallback
                out.append(str(item))
        elif isinstance(item, str):
            if item.strip():
                out.append(item)
        else:
            out.append(str(item))

    return out


class MemoryClient:

    def __init__(self, config: dict, user_id: str = "buddy_user"):
        self.config  = config
        self.user_id = user_id
        self._mem0   = None
        self._lock   = threading.Lock()
        self._ready  = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def load(self):
        """
        Initialise Mem0 with a persistent local Chroma store.

        The Chroma data directory is logged at startup so you can confirm
        memories are being written to disk and will survive restarts.
        """
        try:
            from mem0 import Memory

            # Log the persist path so it's obvious where data lives
            vec_cfg  = self.config.get("vector_store", {}).get("config", {})
            chroma_path = vec_cfg.get("path", "<not set>")
            logger.info("Mem0 Chroma store → %s", chroma_path)

            # Make sure the directory actually exists before Chroma tries to use it
            if chroma_path != "<not set>":
                Path(chroma_path).mkdir(parents=True, exist_ok=True)

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
            target=self._safe_add,
            args=(messages,),
            daemon=True,
            name="Mem0Add",
        )
        t.start()

    def search(self, query: str, limit: int = 8) -> list[str]:
        """
        Search memory for facts relevant to the query.
        Returns a list of plain-text memory strings.
        Never raises — returns [] on any failure.
        """
        if not self._ready:
            return []
        try:
            with self._lock:
                raw = self._mem0.search(query, user_id=self.user_id, limit=limit)
            memories = _extract_memories(raw)
            logger.debug("Mem0 search: %d results for %r", len(memories), query[:40])
            return memories
        except Exception as e:
            logger.warning("Mem0 search failed: %s", e)
            return []

    def get_all(self) -> list[str]:
        """
        Return all stored memories (for debug / UI memory panel).
        Never raises — returns [] on any failure.
        """
        if not self._ready:
            return []
        try:
            with self._lock:
                raw = self._mem0.get_all(user_id=self.user_id)
            return _extract_memories(raw)
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
