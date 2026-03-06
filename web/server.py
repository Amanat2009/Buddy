"""
web/server.py — Real-time Web UI via Flask-SocketIO.

Events emitted to client:
  state_change   {state: "idle"|"listening"|"thinking"|"speaking"}
  token          {text: str}          — streaming LLM token
  response_done  {text: str}          — full response
  mic_level      {level: float 0-1}
  proactive      {text: str}          — buddy initiated message
  timer_update   {timers: [...]}
  memory_update  {memories: [...]}

Events received from client:
  start_listening   — UI button pressed
  stop             — abort current action
  text_input       {text: str}        — typed message (for testing)
"""

import logging
import threading

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

logger = logging.getLogger("buddy.web")

app     = Flask(__name__, template_folder="templates",
                static_folder="static")
app.config["SECRET_KEY"] = "buddy-local-secret-2024"

socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*",
                    logger=False, engineio_logger=False)


# ── Shared state (set by main.py after creating Buddy) ────────────────────────
_buddy = None   # reference to the Buddy orchestrator


def set_buddy(buddy):
    global _buddy
    _buddy = buddy


# ── HTTP routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── SocketIO events ───────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    logger.debug("UI client connected.")
    if _buddy:
        emit("state_change", {"state": _buddy.state})


@socketio.on("disconnect")
def on_disconnect():
    logger.debug("UI client disconnected.")


@socketio.on("start_listening")
def on_start_listening():
    if _buddy:
        threading.Thread(target=_buddy.listen_once, daemon=True).start()


@socketio.on("stop")
def on_stop():
    if _buddy:
        _buddy.abort()


@socketio.on("text_input")
def on_text_input(data):
    text = data.get("text", "").strip()
    if text and _buddy:
        threading.Thread(
            target=_buddy.handle_text, args=(text,), daemon=True
        ).start()


# ── Broadcast helpers (called by Buddy) ──────────────────────────────────────

def broadcast_state(state: str):
    socketio.emit("state_change", {"state": state})


def broadcast_token(token: str):
    socketio.emit("token", {"text": token})


def broadcast_response(text: str):
    socketio.emit("response_done", {"text": text})


def broadcast_level(level: float):
    socketio.emit("mic_level", {"level": round(level, 3)})


def broadcast_proactive(text: str):
    socketio.emit("proactive", {"text": text})


def broadcast_timers(timers: list):
    socketio.emit("timer_update", {"timers": timers})


def broadcast_memories(memories: list):
    socketio.emit("memory_update", {"memories": memories})


# ── Runner ────────────────────────────────────────────────────────────────────

def run(host: str = "127.0.0.1", port: int = 7860, debug: bool = False):
    logger.info("Web UI at http://%s:%d", host, port)
    socketio.run(app, host=host, port=port, debug=debug, use_reloader=False)
