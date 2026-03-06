\*\*🤖 Buddy — Local AI Companion

Buddy is an always-on, wake-word-activated AI assistant that runs entirely on your local machine.

No cloud APIs. No subscriptions. Your data stays local.

Designed for consumer GPUs and CPUs with optional GPU acceleration.

✨ Features

Wake-word activated assistant (jarvis, alexa, etc.)

Fully local LLM (Ollama)

Real-time speech recognition

Streaming text-to-speech

Long-term memory with Mem0

Sentiment-aware responses

Proactive reminders and check-ins

Real-time web dashboard

🧠 Architecture
Layer	Technology
Wake Word	Porcupine
Voice Activity Detection	WebRTC VAD
Speech-to-Text	whisper.cpp
LLM	Ollama
TTS	Kokoro-82M or XTTS-v2
Memory	Mem0 + Chroma
Sentiment	VADER
Web Interface	Flask-SocketIO
📦 Requirements

Minimum recommended:

8 GB RAM

Python 3.10+

Ollama installed

Microphone + speakers

Optional GPU acceleration:

NVIDIA / AMD GPU supported by whisper.cpp

Vulkan or CUDA support improves STT speed

🚀 Installation
1️⃣ Clone the repository
git clone https://github.com/Amanat2009/Buddy.git
cd Buddy
2️⃣ Install Python dependencies
pip install -r requirements.txt
3️⃣ Install Ollama and pull models

Install Ollama:

https://ollama.com/download

Then pull the required models:

ollama pull huihui\_ai/qwen3-abliterated:4b
ollama pull nomic-embed-text
4️⃣ Install whisper.cpp
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp

Build:

cmake -B build
cmake --build build --config Release

Download STT model:

./models/download-ggml-model.sh large-v3-turbo-q8\_0

Return to Buddy:

cd ..
5️⃣ Get a Porcupine Access Key

Sign up for a free key:

https://picovoice.ai

Then edit:

config.py

Example:

PICOVOICE\_ACCESS\_KEY = "your-key"
WAKE\_WORD = "jarvis"
USER\_NAME = "Boss"
TIMEZONE = "Asia/Kolkata"
🔊 Install Kokoro TTS Models

The TTS model files are not included in the repository because they are large.

Download them automatically using:

pip install huggingface\_hub

python -c "
from huggingface\_hub import hf\_hub\_download
hf\_hub\_download('hexgrad/Kokoro-82M', 'kokoro-v1.0.onnx', local\_dir='.')
hf\_hub\_download('hexgrad/Kokoro-82M', 'voices-v1.0.bin', local\_dir='.')
"

This will download the required files into the project root.

▶️ Run Buddy
python main.py

Open the dashboard:

http://127.0.0.1:7860
🧪 Optional: XTTS-v2 Voice Cloning

Install XTTS:

pip install TTS

Edit config.py:

TTS\_ENGINE = "xtts"
XTTS\_SPEAKER\_WAV = "data/speaker.wav"

Record a \~30 second clean voice sample and place it at:

data/speaker.wav
⏰ Timer Examples

Buddy understands natural timer commands without LLM calls:

set a timer for 20 minutes
remind me in an hour
timer 5 mins
wake me up in 30
30 second timer
2 hours 30 minutes
📁 Project Structure
buddy/
├ main.py
├ config.py
├ requirements.txt
├ README.md
│
├ audio/
├ llm/
├ memory/
├ personality/
├ proactive/
├ context/
├ web/
│
└ data/

The data/ directory stores persistent memory and logs.

⚠️ Files Not Included in Repo

The following files are ignored by Git:

kokoro-v1.0.onnx
voices-v1.0.bin
buddy\_env/
**pycache**/
.cache/

These are downloaded or generated locally.

🛠 Troubleshooting
Wake word not triggering

Increase sensitivity in config.py.

Ollama not responding

Make sure Ollama is running:

ollama serve
No microphone input

Check device list:

python -c "import sounddevice as sd; print(sd.query\_devices())"
📜 License

MIT License

⭐ Contributing

Contributions, improvements, and ideas are welcome.

If you build new modules (skills, tools, or agents), feel free to open a pull request.

🙌 Acknowledgements

Porcupine

whisper.cpp

Ollama

Mem0

Chroma

Kokoro TTS\*\*\*

