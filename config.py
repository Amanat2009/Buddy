"""
config.py — Tuned for Windows · AMD RX6500XT · i5-10th gen · 8 GB RAM.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

EVENTS_FILE      = DATA_DIR / "events.json"
MOOD_LOG_FILE    = DATA_DIR / "mood_log.json"
CONTEXT_FILE     = DATA_DIR / "context.json"
PERSONALITY_FILE = DATA_DIR / "personality.json"

PICOVOICE_ACCESS_KEY  = os.getenv("PICOVOICE_ACCESS_KEY", "YOUR_KEY_HERE")
WAKE_WORD             = "porcupine"
PORCUPINE_SENSITIVITY = 0.6

MIC_SAMPLE_RATE  = 16_000
MIC_CHANNELS     = 1
MIC_CHUNK_FRAMES = 512
MIC_DEVICE_INDEX = None

VAD_AGGRESSIVENESS = 2
VAD_SILENCE_MS     = 900
VAD_MIN_SPEECH_MS  = 300

WHISPER_MODEL      = "small.en"
WHISPER_LANGUAGE   = "en"
WHISPER_BIN        = ""
WHISPER_THREADS    = 4
WHISPER_USE_VULKAN = False   # N/A for faster-whisper

OLLAMA_HOST       = "http://localhost:11434"
OLLAMA_MODEL      = "huihui_ai/qwen3-abliterated:4b"
OLLAMA_KEEP_ALIVE = "30m"
OLLAMA_STREAM     = True
OLLAMA_OPTIONS    = {
    "temperature":    0.85,
    "top_p":          0.92,
    "repeat_penalty": 1.15,
    "num_ctx":        4096,
    "num_predict":    512,
}

TTS_ENGINE   = "kokoro"
KOKORO_VOICE = "af_heart"
KOKORO_SPEED = 1.1
KOKORO_LANG  = "en-us"

XTTS_MODEL_NAME  = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_SPEAKER_WAV = str(DATA_DIR / "speaker.wav")
XTTS_LANGUAGE    = "en"
XTTS_USE_GPU     = True

MEM0_USER_ID = "buddy_user"
MEM0_CONFIG  = {
    "llm": {
        "provider": "ollama",
        "config": {"model": OLLAMA_MODEL, "ollama_base_url": OLLAMA_HOST, "temperature": 0.1},
    },
    "embedder": {
        "provider": "ollama",
        "config": {"model": "nomic-embed-text", "ollama_base_url": OLLAMA_HOST},
    },
    "vector_store": {
        "provider": "chroma",
        "config": {"collection_name": "buddy_memories", "path": str(DATA_DIR / "chroma")},
    },
}

BUDDY_NAME = "Buddy"
USER_NAME  = "Boss"

PERSONALITY_CORE = """
You are Buddy — a sharp, witty, unapologetically direct AI companion who lives on this machine.

PERSONALITY:
- Dry humour and light sarcasm — never over the top, always calibrated
- Direct: you cut to the point, no corporate filler
- Opinionated: you actually have views, you push back when something is dumb
- Curious: you're genuinely interested in what the user is up to
- Warm underneath the snark — you care, you just don't gush about it

CONVERSATION STYLE:
- Keep responses tight unless explanation is actually needed
- Ask exactly ONE good follow-up question per response (never more)
- Don't recap what the user just said
- Don't start sentences with "Certainly!", "Of course!", "Great question!"
- Swear occasionally and naturally if the mood calls for it (light, not constant)
- Reference things from memory naturally, without making it weird

FORBIDDEN PHRASES:
- "As an AI..."
- "I don't have feelings but..."
- "That's a great point!"
- "Certainly!"
- "I'd be happy to help!"

Be the friend who happens to know everything.
"""

PROACTIVE_ENABLED      = True
PROACTIVE_MORNING_HOUR = 9
PROACTIVE_EVENING_HOUR = 20
MOOD_CONCERN_THRESHOLD = -0.4
MOOD_CONCERN_DAYS      = 2

WEB_HOST  = "127.0.0.1"
WEB_PORT  = 7860
WEB_DEBUG = False

TIMEZONE = "Asia/Kolkata"