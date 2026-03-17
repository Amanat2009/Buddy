"""
config.py — Central configuration for Buddy (ENGLISH VERSION)

CRITICAL: This config forces English for:
  • STT (Whisper)
  • LLM (Ollama/Qwen)
  • TTS (Kokoro)
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

EVENTS_FILE      = DATA_DIR / "events.json"
MOOD_LOG_FILE    = DATA_DIR / "mood_log.json"
CONTEXT_FILE     = DATA_DIR / "context.json"
PERSONALITY_FILE = DATA_DIR / "personality.json"

# ── Wake-Word (Porcupine) ─────────────────────────────────────────────────────
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY", "YOUR_KEY_HERE")
WAKE_WORD           = "porcupine"       # Built-in keyword
PORCUPINE_SENSITIVITY = 0.6          # 0.0–1.0; higher = more sensitive

# ── Microphone ────────────────────────────────────────────────────────────────
MIC_SAMPLE_RATE      = 16000
MIC_CHANNELS         = 1
MIC_CHUNK_FRAMES     = 512
MIC_DEVICE_INDEX     = None          # None = system default

# ── VAD (WebRTC) ──────────────────────────────────────────────────────────────
VAD_AGGRESSIVENESS   = 2             # 0–3; 3 = most aggressive
VAD_SILENCE_MS       = 900           # ms of silence before STT triggers
VAD_MIN_SPEECH_MS    = 300           # ignore clips shorter than this

# ── STT (whisper.cpp) - ENGLISH ONLY ───────────────────────────────────────────
WHISPER_MODEL        = "small.en"    # .en suffix = English only (faster!)
WHISPER_LANGUAGE     = "en"          # Force English language
WHISPER_BIN          = os.getenv("WHISPER_BIN", "/usr/local/bin/whisper-cpp")
WHISPER_THREADS      = 4
WHISPER_USE_VULKAN   = True

# ── LLM (Ollama) - ENGLISH ONLY ────────────────────────────────────────────────
OLLAMA_HOST          = "http://localhost:11434"
OLLAMA_MODEL         = "qwen-nothink"    # Without thinking delay (recommended)
# Or use: "mistral" for guaranteed instant English responses
OLLAMA_KEEP_ALIVE    = "10m"
OLLAMA_STREAM        = True
OLLAMA_OPTIONS = {
    "temperature":    0.7,
    "top_p":          0.9,
    "repeat_penalty": 1.15,
    "num_ctx":        2048,
    "num_predict":    512,
}

# ── TTS ────────────────────────────────────────────────────────────────────────
# "kokoro" (fast, default) or "xtts" (higher quality)
TTS_ENGINE           = "kokoro"

# Kokoro settings
KOKORO_VOICE         = "af_heart"    # English voice
KOKORO_SPEED         = 1.1
KOKORO_LANG          = "en-us"       # ENGLISH - en-us only

# XTTS-v2 settings (if using xtts)
XTTS_MODEL_NAME      = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_SPEAKER_WAV     = str(DATA_DIR / "speaker.wav")
XTTS_LANGUAGE        = "en"          # ENGLISH
XTTS_USE_GPU         = True

# ── Memory (Mem0) ─────────────────────────────────────────────────────────────
MEM0_USER_ID         = "buddy_user"
MEM0_CONFIG = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "qwen-nothink",
            "ollama_base_url": "http://localhost:11434",
            "temperature": 0.1,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "buddy_memories",
            "path": str(DATA_DIR / "chroma"),
        },
    },
}

# ── Personality ───────────────────────────────────────────────────────────────
BUDDY_NAME           = "Buddy"
USER_NAME            = "Boss"

PERSONALITY_CORE = """
You are Buddy — a sharp, witty, unapologetically direct AI companion who lives on this machine.

🚨 CRITICAL - LANGUAGE & RESPONSE:
- ALWAYS respond in English. NEVER switch languages.
- Even if the user speaks another language, respond in English.
- Do NOT respond in Hindi, Telugu, Tamil, or any other language.
- Do NOT translate. Just answer in English.
- Do NOT generate thinking or reasoning blocks.
- Do NOT include <think> tags.
- Answer DIRECTLY and IMMEDIATELY.

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

Be the friend who happens to know everything. Respond in English only.
"""

# ── Proactive Engagement ──────────────────────────────────────────────────────
PROACTIVE_ENABLED        = True
PROACTIVE_MORNING_HOUR   = 9
PROACTIVE_EVENING_HOUR   = 20
MOOD_CONCERN_THRESHOLD   = -0.4
MOOD_CONCERN_DAYS        = 2

# ── Web UI ────────────────────────────────────────────────────────────────────
WEB_HOST             = "127.0.0.1"
WEB_PORT             = 7860
WEB_DEBUG            = False

# ── Timer / Time ──────────────────────────────────────────────────────────────
TIMEZONE             = "Asia/Kolkata"   # Your timezone
