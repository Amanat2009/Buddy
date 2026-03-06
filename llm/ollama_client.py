"""
llm/ollama_client.py — Streaming Ollama client.

Supports:
  • Streaming token callbacks (for real-time TTS + UI)
  • Full response accumulation
  • Automatic keep-alive pinging
  • Conversation history management
"""

import json
import logging
import threading
import time
from typing import Callable, Generator, Iterator

import requests

logger = logging.getLogger("buddy.llm")


class OllamaClient:

    def __init__(self,
                 host: str = "http://localhost:11434",
                 model: str = "huihui_ai/qwen3-abliterated:4b",
                 keep_alive: str = "30m",
                 options: dict = None):
        self.host       = host.rstrip("/")
        self.model      = model
        self.keep_alive = keep_alive
        self.options    = options or {}
        self._session   = requests.Session()

    # ── Streaming generation ───────────────────────────────────────────────

    def stream(self,
               messages: list[dict],
               system: str = "",
               on_token: Callable[[str], None] = None,
               on_done: Callable[[str], None] = None) -> str:
        """
        Stream a completion.

        Args:
            messages:  list of {role, content} dicts (conversation history)
            system:    system prompt injected before messages
            on_token:  called with each new token string
            on_done:   called with the complete accumulated response

        Returns:
            Complete response string.
        """
        payload = {
            "model":      self.model,
            "messages":   messages,
            "stream":     True,
            "keep_alive": self.keep_alive,
            "options":    self.options,
        }
        if system:
            payload["system"] = system

        full_text = ""
        try:
            resp = self._session.post(
                f"{self.host}/api/chat",
                json=payload,
                stream=True,
                timeout=(5, 120),
            )
            resp.raise_for_status()

            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_text += token
                    if on_token:
                        on_token(token)

                if chunk.get("done"):
                    break

        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama at %s", self.host)
            full_text = "Sorry, I can't reach my brain right now. Is Ollama running?"
        except requests.exceptions.Timeout:
            logger.error("Ollama timed out.")
            full_text = "I took too long to think. Try again?"
        except Exception as e:
            logger.error("Ollama error: %s", e)
            full_text = "Something went wrong on my end."

        if on_done:
            on_done(full_text)
        return full_text

    def ping(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            r = self._session.get(f"{self.host}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def warm_up(self):
        """Send a dummy request to load the model into VRAM."""
        logger.info("Warming up model '%s'…", self.model)
        self._session.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": " ", "keep_alive": self.keep_alive},
            timeout=60,
        )
        logger.info("Model warm-up complete.")


# ── Conversation History ───────────────────────────────────────────────────────

class ConversationHistory:
    """Rolling conversation history with max-token trimming."""

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self._messages: list[dict] = []

    def add_user(self, text: str):
        self._messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str):
        self._messages.append({"role": "assistant", "content": text})
        self._trim()

    def get(self) -> list[dict]:
        return list(self._messages)

    def clear(self):
        self._messages.clear()

    def _trim(self):
        # Keep system coherence: always drop oldest user+assistant pair
        while len(self._messages) > self.max_turns * 2:
            self._messages.pop(0)
            if self._messages and self._messages[0]["role"] == "assistant":
                self._messages.pop(0)
