"""
config.py — Complete configuration with thinking DISABLED and full debugging
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
WAKE_WORD           = "porcupine"
PORCUPINE_SENSITIVITY = 0.6

# ── Microphone ────────────────────────────────────────────────────────────────
MIC_SAMPLE_RATE      = 16000
MIC_CHANNELS         = 1
MIC_CHUNK_FRAMES     = 512
MIC_DEVICE_INDEX     = None

# ── VAD (WebRTC) ──────────────────────────────────────────────────────────────
VAD_AGGRESSIVENESS   = 2
VAD_SILENCE_MS       = 900
VAD_MIN_SPEECH_MS    = 300

# ── STT (whisper.cpp) - ENGLISH ONLY ───────────────────────────────────────────
WHISPER_MODEL        = "small.en"    # English only
WHISPER_LANGUAGE     = "en"          # Force English
WHISPER_BIN          = os.getenv("WHISPER_BIN", "/usr/local/bin/whisper-cpp")
WHISPER_THREADS      = 4
WHISPER_USE_VULKAN   = True

# ── LLM (Ollama) - THINKING DISABLED ───────────────────────────────────────────
OLLAMA_HOST          = "http://localhost:11434"
OLLAMA_MODEL         = "qwen-nothink"  # ← THINKING DISABLED by default
# Alternative (if qwen-nothink not available):
# OLLAMA_MODEL       = "mistral"

OLLAMA_KEEP_ALIVE    = "10m"
OLLAMA_STREAM        = True

# ✅ OPTIMIZED for speed and instant responses (NO THINKING)
OLLAMA_OPTIONS = {
    "temperature":    0.7,
    "top_p":          0.9,
    "repeat_penalty": 1.15,
    "num_ctx":        2048,
    "num_predict":    512,
}

# ── TTS ────────────────────────────────────────────────────────────────────────
TTS_ENGINE           = "kokoro"

# Kokoro settings
KOKORO_VOICE         = "af_heart"
KOKORO_SPEED         = 1.1
KOKORO_LANG          = "en-us"    # ENGLISH

# XTTS-v2 settings (optional)
XTTS_MODEL_NAME      = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_SPEAKER_WAV     = str(DATA_DIR / "speaker.wav")
XTTS_LANGUAGE        = "en"       # ENGLISH
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
You are Buddy — a sharp, witty, unapologetically direct AI companion.

🚨 CRITICAL - LANGUAGE & RESPONSE:
- ALWAYS respond in English. NEVER switch languages.
- Even if user speaks another language, respond in English.
- Do NOT respond in Hindi, Telugu, Tamil, or any other language.
- Answer INSTANTLY without internal thinking or reasoning.
- Do NOT include <think> tags or internal monologues.
- No "Let me think about this..." or similar delays.

PERSONALITY:
- Dry humour and light sarcasm — never over the top
- Direct: cut to the point, no corporate filler
- Opinionated: push back when something is dumb
- Curious: genuinely interested in what the user is up to
- Warm underneath the snark — you care

CONVERSATION STYLE:
- Keep responses tight unless explanation is needed
- Ask exactly ONE good follow-up question per response
- Don't recap what the user just said
- Don't start sentences with "Certainly!", "Of course!", "Great question!"
- Swear occasionally if the mood calls for it (light, natural)
- Reference memory naturally, without making it weird

FORBIDDEN PHRASES:
- "As an AI..."
- "I don't have feelings but..."
- "That's a great point!"
- "Certainly!"
- "I'd be happy to help!"

Be the friend who happens to know everything. Respond instantly in English.
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
TIMEZONE             = "Asia/Kolkata"

# ── DEBUGGING ─────────────────────────────────────────────────────────────────
# Set to True for verbose logging
DEBUG_ENABLED        = True
DEBUG_LOG_FILE       = str(DATA_DIR / "buddy_debug.log")

print("""
╔════════════════════════════════════════════════════════════════╗
║  BUDDY CONFIG LOADED                                           ║
╠════════════════════════════════════════════════════════════════╣
║  LLM Model:     %s
║  Thinking:      DISABLED (uses qwen-nothink)
║  Language:      ENGLISH ONLY (en)
║  STT Model:     %s
║  TTS Engine:    %s
║  Debug Log:     buddy_debug.log
╚════════════════════════════════════════════════════════════════╝
""" % (OLLAMA_MODEL, WHISPER_MODEL, TTS_ENGINE))
