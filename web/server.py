"""
web/server.py — Flask-SocketIO web UI.

FIX: async_mode changed from "eventlet" to "threading".
eventlet is broken on Windows Python 3.10+ and causes UI freezes.
All broadcast helpers include namespace="/" for background thread safety.
"""

import logging
import threading

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

logger = logging.getLogger("buddy.web")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "buddy-local-secret-2024"

socketio = SocketIO(
    app,
    async_mode="threading",   # <-- eventlet removed, this works on Windows
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
)

_buddy = None

def set_buddy(buddy):
    global _buddy
    _buddy = buddy

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("connect")
def on_connect():
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
        threading.Thread(target=_buddy.handle_text, args=(text,), daemon=True).start()

def broadcast_state(state):
    socketio.emit("state_change", {"state": state}, namespace="/")

def broadcast_token(token):
    socketio.emit("token", {"text": token}, namespace="/")

def broadcast_response(text):
    socketio.emit("response_done", {"text": text}, namespace="/")

def broadcast_level(level):
    socketio.emit("mic_level", {"level": round(level, 3)}, namespace="/")

def broadcast_proactive(text):
    socketio.emit("proactive", {"text": text}, namespace="/")

def broadcast_timers(timers):
    socketio.emit("timer_update", {"timers": timers}, namespace="/")

def broadcast_memories(memories):
    socketio.emit("memory_update", {"memories": memories}, namespace="/")

def run(host="127.0.0.1", port=7860, debug=False):
    logger.info("Web UI → http://%s:%d", host, port)
    socketio.run(app, host=host, port=port, debug=debug,
                 use_reloader=False, allow_unsafe_werkzeug=True)