# 🤖 Buddy — Local AI Companion

Always-on, wake-word-activated AI buddy. Runs 100% locally on your machine.
Built for: **8 GB RAM + RX 6500 XT 8 GB VRAM (Vulkan)**.

---

## Stack

| Layer | Component |
|-------|-----------|
| Wake Word | Porcupine (pvporcupine) |
| STT | whisper.cpp + large-v3-turbo-q8_0 — Vulkan/GPU |
| VAD | WebRTC VAD (aggressiveness=2) |
| LLM | Ollama → huihui_ai/qwen3-abliterated:4b |
| TTS | Kokoro-82M (default) or Coqui XTTS-v2 |
| Memory | Mem0 → local Chroma + Ollama embeddings |
| Web UI | Flask-SocketIO, dark terminal UI |
| Sentiment | VADER (instant, no GPU) |

---

## Quick Start

### 1. Install Python deps

```bash
cd buddy
pip install -r requirements.txt
```

### 2. Install whisper.cpp (Vulkan build for RX 6500 XT)

```bash
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp

# Build with Vulkan support
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j4

# Download the model
./models/download-ggml-model.sh large-v3-turbo-q8_0

# Optional: install globally
sudo cp build/bin/whisper-cli /usr/local/bin/whisper-cpp
cd ..
```

### 3. Install Ollama and pull the model

```bash
# Install Ollama: https://ollama.com/download
ollama pull huihui_ai/qwen3-abliterated:4b
ollama pull nomic-embed-text       # for Mem0 embeddings
```

### 4. Get a free Porcupine access key

- Sign up at https://picovoice.ai (free tier = unlimited local use)
- Copy your access key

### 5. Configure Buddy

Edit `config.py`:

```python
PICOVOICE_ACCESS_KEY = "your-key-here"
WAKE_WORD            = "jarvis"       # or: alexa, computer, bumblebee, etc.
USER_NAME            = "Boss"         # how buddy addresses you
TIMEZONE             = "Asia/Kolkata"
```

### 6. Install Kokoro TTS model files

```bash
# Download from HuggingFace
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('hexgrad/Kokoro-82M', 'kokoro-v1.0.onnx', local_dir='.')
hf_hub_download('hexgrad/Kokoro-82M', 'voices-v1.0.bin', local_dir='.')
"
```

### 7. Run

```bash
python main.py
```

Open your browser: **http://127.0.0.1:7860**

---

## XTTS-v2 (Optional — Higher Quality TTS)

If you want XTTS-v2 instead of Kokoro:

```bash
pip install TTS
```

In `config.py`:
```python
TTS_ENGINE = "xtts"
XTTS_SPEAKER_WAV = "data/speaker.wav"  # record ~30s of your voice
```

Place a clean 30-second WAV of someone speaking at `data/speaker.wav` for voice cloning.

---

## Wake Words Available (Built-in, Free)

`alexa`, `americano`, `blueberry`, `bumblebee`, `computer`,
`grapefruit`, `grasshopper`, `hey google`, `hey siri`, `jarvis`,
`ok google`, `picovoice`, `porcupine`, `terminator`

---

## Vulkan Setup for RX 6500 XT

whisper.cpp uses Vulkan automatically when built with `-DGGML_VULKAN=ON`.

For Ollama GPU acceleration on AMD with ROCm:
```bash
# Check: https://rocm.docs.amd.com/
# RX 6500 XT = gfx1032 — supported in ROCm 5.5+
HSA_OVERRIDE_GFX_VERSION=10.3.0 ollama serve
```

---

## File Structure

```
buddy/
├── main.py              # Orchestrator + entry point
├── config.py            # All settings (edit this)
├── requirements.txt
│
├── audio/
│   ├── beep.py          # Synthesised beep sounds (no audio files)
│   ├── wake_word.py     # Porcupine wake word detector
│   ├── mic_input.py     # WebRTC VAD + silence detection recorder
│   ├── stt.py           # whisper.cpp STT (Vulkan)
│   └── tts.py           # Kokoro / XTTS-v2 TTS + sentence streamer
│
├── llm/
│   ├── ollama_client.py # Streaming Ollama client + history
│   └── system_prompt.py # Dynamic system prompt (time, context, mood, memories)
│
├── memory/
│   ├── mem0_client.py   # Mem0 long-term memory
│   └── storage.py       # JSON persistence (events, context, mood log)
│
├── personality/
│   ├── sentiment.py     # VADER sentiment analyser
│   └── engine.py        # Mood tracking + context extraction
│
├── proactive/
│   └── scheduler.py     # Morning/evening check-ins, event reminders
│
├── context/
│   └── timer.py         # Natural language timers (regex, non-blocking)
│
├── web/
│   ├── server.py        # Flask-SocketIO real-time UI server
│   └── templates/
│       └── index.html   # Dark terminal dashboard
│
└── data/                # Auto-created — stores all persistent data
    ├── events.json
    ├── context.json
    ├── mood_log.json
    └── chroma/          # Mem0 vector store
```

---

## Timer Examples

All detected by regex — zero LLM calls:

```
"set a timer for 20 minutes"
"remind me in an hour"
"give me 45 minutes"
"timer 5 mins"
"wake me up in 30"
"30 second timer for the pasta"
"2 hours 30 minutes"
```

---

## Tuning for Your Hardware

**RX 6500 XT 8 GB VRAM** is enough for:
- Kokoro TTS: ~200 MB VRAM
- qwen3:4b: ~3.5 GB VRAM
- whisper large-v3-turbo-q8_0: ~1.6 GB VRAM
- **Total: ~5.3 GB VRAM** — fits comfortably

If VRAM is tight, switch to `whisper-medium.en` (~1 GB) or `qwen3:4b` → `phi3-mini`.

---

## Troubleshooting

**Wake word not triggering:**
- Increase `PORCUPINE_SENSITIVITY` (max 1.0)
- Check mic index: `python -c "import pyaudio; p=pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"`

**Ollama not found:**
- Make sure `ollama serve` is running
- Check `OLLAMA_HOST` in config.py

**No sound output:**
- Check `sounddevice` output device: `python -c "import sounddevice as sd; print(sd.query_devices())"`

**Mem0 slow:**
- First run is slow (model load). Subsequent calls are fast.
- Uses `nomic-embed-text` for embeddings — make sure it's pulled.

Required models:

Download kokoro-v1.0.onnx and place it in the project root.

Download voices-v1.0.bin and place it in the project root.
