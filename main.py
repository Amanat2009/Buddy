"""
main.py — Buddy Orchestrator.

Wires together every subsystem and runs the main event loop.

State machine:
  idle → [wake word / UI button] → listening → [silence] → thinking → speaking → idle

Boot order matters:
  1. STT  — loads/caches whisper model (slow on first run, fast after)
  2. TTS  — loads Kokoro ONNX
  3. Memory — initialises Mem0 + Chroma
  4. Sentiment — loads VADER (instant)
  5. Ollama warm-up — async, doesn't block boot
  6. Wake-word detector — starts listening thread
  7. Proactive scheduler — starts background thread
"""

import logging
import threading
import time
from pathlib import Path

import config as cfg

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("buddy.main")


class Buddy:
    """Central orchestrator. One instance, runs forever."""

    STATES = ("idle", "listening", "thinking", "speaking")

    def __init__(self):
        self._state      = "idle"
        self._abort_flag = threading.Event()

        # ── Imports ────────────────────────────────────────────────────────
        from audio.beep      import (play_wake_confirm, play_listening_start,
                                     play_processing, play_error, play_timer_done)
        from audio.stt       import WhisperSTT
        from audio.tts       import TTSEngine, SentenceStreamer
        from audio.wake_word import WakeWordDetector
        from audio.mic_input import MicRecorder

        from llm.ollama_client import OllamaClient, ConversationHistory
        from llm.system_prompt import SystemPromptBuilder

        from memory.mem0_client import MemoryClient

        from personality.sentiment import SentimentAnalyser
        from personality.engine    import PersonalityEngine

        from context.timer import TimerManager, parse_timer_intent, extract_timer_label

        from proactive.scheduler import ProactiveScheduler

        import web.server as web_server
        self._web = web_server

        # ── STT ────────────────────────────────────────────────────────────
        self.stt = WhisperSTT(
            model       = cfg.WHISPER_MODEL,
            language    = cfg.WHISPER_LANGUAGE,
            whisper_bin = cfg.WHISPER_BIN,
            n_threads   = cfg.WHISPER_THREADS,
            use_vulkan  = cfg.WHISPER_USE_VULKAN,
        )

        # ── TTS ────────────────────────────────────────────────────────────
        self.tts = TTSEngine(
            engine           = cfg.TTS_ENGINE,
            voice            = cfg.KOKORO_VOICE,
            speed            = cfg.KOKORO_SPEED,
            lang             = cfg.KOKORO_LANG,
            xtts_model       = cfg.XTTS_MODEL_NAME,
            xtts_speaker_wav = cfg.XTTS_SPEAKER_WAV,
            xtts_language    = cfg.XTTS_LANGUAGE,
            use_gpu          = cfg.XTTS_USE_GPU,
        )

        # ── LLM ────────────────────────────────────────────────────────────
        self.llm = OllamaClient(
            host       = cfg.OLLAMA_HOST,
            model      = cfg.OLLAMA_MODEL,
            keep_alive = cfg.OLLAMA_KEEP_ALIVE,
            options    = cfg.OLLAMA_OPTIONS,
        )
        self.history = ConversationHistory(max_turns=20)

        # ── Memory ─────────────────────────────────────────────────────────
        self.mem = MemoryClient(cfg.MEM0_CONFIG, user_id=cfg.MEM0_USER_ID)

        # ── Sentiment ──────────────────────────────────────────────────────
        self.sentiment = SentimentAnalyser()

        # ── System prompt ──────────────────────────────────────────────────
        self.prompt_builder = SystemPromptBuilder(
            personality_core = cfg.PERSONALITY_CORE,
            buddy_name       = cfg.BUDDY_NAME,
            user_name        = cfg.USER_NAME,
            timezone         = cfg.TIMEZONE,
            context_file     = cfg.CONTEXT_FILE,
            mood_log_file    = cfg.MOOD_LOG_FILE,
        )

        # ── Personality engine ─────────────────────────────────────────────
        self.personality = PersonalityEngine(
            llm_client         = self.llm,
            sentiment_analyser = self.sentiment,
            context_file       = cfg.CONTEXT_FILE,
            mood_log_file      = cfg.MOOD_LOG_FILE,
            user_name          = cfg.USER_NAME,
            concern_threshold  = cfg.MOOD_CONCERN_THRESHOLD,
            concern_days       = cfg.MOOD_CONCERN_DAYS,
            timezone           = cfg.TIMEZONE,
        )

        # ── Timer ──────────────────────────────────────────────────────────
        self._parse_timer = parse_timer_intent
        self._timer_label = extract_timer_label

        def _on_timer_done(label, message):
            from audio.beep import play_timer_done
            play_timer_done()
            self._speak_response(message)
            self._web.broadcast_proactive(message)
            self._broadcast_timers()

        self.timers = TimerManager(on_done=_on_timer_done)

        # ── Wake word ──────────────────────────────────────────────────────
        self.wake_detector = WakeWordDetector(
            access_key   = cfg.PICOVOICE_ACCESS_KEY,
            keyword      = cfg.WAKE_WORD,
            sensitivity  = cfg.PORCUPINE_SENSITIVITY,
            device_index = cfg.MIC_DEVICE_INDEX,
            on_wake      = self.listen_once,
        )

        # ── Proactive scheduler ────────────────────────────────────────────
        self.proactive = ProactiveScheduler(
            llm_client         = self.llm,
            mem_client         = self.mem,
            personality_engine = self.personality,
            tts_engine         = self.tts,
            context_file       = cfg.CONTEXT_FILE,
            events_file        = cfg.EVENTS_FILE,
            buddy_name         = cfg.BUDDY_NAME,
            user_name          = cfg.USER_NAME,
            timezone           = cfg.TIMEZONE,
            morning_hour       = cfg.PROACTIVE_MORNING_HOUR,
            evening_hour       = cfg.PROACTIVE_EVENING_HOUR,
            on_speak           = lambda t: self._web.broadcast_proactive(t),
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    def _set_state(self, state: str):
        self._state = state
        self._web.broadcast_state(state)
        logger.info("State → %s", state)

    # ── Startup ───────────────────────────────────────────────────────────────

    def boot(self):
        """
        Load all models and start background services.

        STT is loaded FIRST and SYNCHRONOUSLY so the first user query
        has zero extra delay. Without this, whisper loads on first use
        causing a 7+ second freeze mid-conversation.
        """
        logger.info("═══ Buddy booting ═══")

        # ① STT — must be first and synchronous
        #   First run:  downloads model to data/whisper_cache/ (~150 MB)
        #   After that: loads from disk in ~2–3 s, no network call
        logger.info("Loading Whisper STT ('%s')…", cfg.WHISPER_MODEL)
        self.stt.load()

        # ② TTS
        logger.info("Loading TTS ('%s')…", cfg.TTS_ENGINE)
        self.tts.load()

        # ③ Memory
        logger.info("Loading memory (Mem0)…")
        self.mem.load()

        # ④ Sentiment (instant — VADER is rule-based, no model file)
        logger.info("Loading sentiment analyser…")
        self.sentiment.load()

        # ⑤ Ollama warm-up — async so it doesn't delay boot
        logger.info("Pinging Ollama…")
        if self.llm.ping():
            threading.Thread(
                target=self.llm.warm_up, daemon=True, name="OllamaWarmup"
            ).start()
        else:
            logger.warning("Ollama not reachable at %s — is it running?", cfg.OLLAMA_HOST)

        # ⑥ Wake-word detector
        if cfg.PICOVOICE_ACCESS_KEY not in ("YOUR_KEY_HERE", ""):
            logger.info("Starting wake-word detector ('%s')…", cfg.WAKE_WORD)
            self.wake_detector.start()
        else:
            logger.warning(
                "No Picovoice key set — wake word disabled. "
                "Use the Listen button or text input instead."
            )

        # ⑦ Proactive scheduler
        if cfg.PROACTIVE_ENABLED:
            logger.info("Starting proactive scheduler…")
            self.proactive.start()

        logger.info("═══ Ready ═══")

    def shutdown(self):
        self.wake_detector.stop()
        self.proactive.stop()
        self.timers.cancel_all()
        self.tts.stop()

    # ── Main interaction flow ─────────────────────────────────────────────────

    def listen_once(self):
        """Record one utterance, transcribe, generate, speak."""
        if self._state != "idle":
            logger.debug("Busy (%s) — ignoring listen request.", self._state)
            return

        self._abort_flag.clear()
        self._set_state("listening")

        from audio.beep      import play_listening_start, play_processing, play_error
        from audio.mic_input import MicRecorder

        play_listening_start()

        wav_bytes = None
        try:
            with MicRecorder(
                vad_aggressiveness = cfg.VAD_AGGRESSIVENESS,
                silence_ms         = cfg.VAD_SILENCE_MS,
                min_speech_ms      = cfg.VAD_MIN_SPEECH_MS,
                device_index       = cfg.MIC_DEVICE_INDEX,
                on_level           = self._web.broadcast_level,
            ) as mic:
                wav_bytes = mic.record()
        except Exception as e:
            logger.error("Mic error: %s", e)
            play_error()
            self._set_state("idle")
            return

        if not wav_bytes or self._abort_flag.is_set():
            self._set_state("idle")
            return

        self._set_state("thinking")
        play_processing()

        text = self.stt.transcribe(wav_bytes)
        logger.info("STT: %r", text)

        if not text.strip():
            self._set_state("idle")
            return

        self.handle_text(text, from_voice=True)

    def handle_text(self, text: str, from_voice: bool = False):
        """Process a user message — from voice or UI text input."""
        if self._state not in ("idle", "thinking"):
            self.abort()
            time.sleep(0.1)

        self._abort_flag.clear()

        if not from_voice:
            self._set_state("thinking")

        # ── Timer intent (regex — no LLM call needed) ──────────────────────
        duration = self._parse_timer(text)
        if duration is not None:
            label = self._timer_label(text)
            self.timers.set(duration, label)
            mins, secs = divmod(duration, 60)
            if mins:
                duration_str = f"{mins} minute{'s' if mins != 1 else ''}"
            else:
                duration_str = f"{secs} second{'s' if secs != 1 else ''}"
            ack = (
                f"Set a {duration_str} {label.lower()} timer. "
                f"I'll let you know when it's done."
            )
            self.history.add_user(text)
            self.history.add_assistant(ack)
            self._speak_response(ack)
            self._broadcast_timers()
            return

        # ── Mood + personality ─────────────────────────────────────────────
        mood_info = self.personality.process_user_message(text)
        logger.debug("Mood: %s (score=%.2f)", mood_info["label"], mood_info["score"])

        # ── Memory search ──────────────────────────────────────────────────
        memories = self.mem.search(text, limit=8)
        self.prompt_builder.recent_memories = memories

        # ── System prompt ──────────────────────────────────────────────────
        self.prompt_builder.active_timers = self.timers.active_timers()
        system = self.prompt_builder.build()

        self.history.add_user(text)

        # ── Stream LLM → TTS ───────────────────────────────────────────────
        from audio.tts import SentenceStreamer
        streamer = SentenceStreamer(self.tts)
        streamer.start()

        self._set_state("thinking")
        full_response = ""

        def on_token(token: str):
            nonlocal full_response
            if self._abort_flag.is_set():
                return
            full_response += token
            self._web.broadcast_token(token)
            streamer.push(token)
            if self._state == "thinking":
                self._set_state("speaking")

        def on_done(response: str):
            streamer.finish()
            self._web.broadcast_response(response)

        self.llm.stream(
            messages = self.history.get(),
            system   = system,
            on_token = on_token,
            on_done  = on_done,
        )

        streamer.wait()

        if not self._abort_flag.is_set():
            self.history.add_assistant(full_response)
            self.mem.add(text, full_response)
            self.personality.extract_context_async(text, full_response)
            threading.Thread(
                target=self._refresh_memories, daemon=True
            ).start()

        self._set_state("idle")

    def _speak_response(self, text: str):
        """Speak a short canned response without streaming."""
        self._set_state("speaking")
        self.tts.speak(text)
        self._set_state("idle")

    def abort(self):
        """Interrupt whatever Buddy is doing right now."""
        logger.info("Abort requested.")
        self._abort_flag.set()
        self.tts.stop()
        self._set_state("idle")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _broadcast_timers(self):
        self._web.broadcast_timers(self.timers.active_timers())

    def _refresh_memories(self):
        memories = self.mem.get_all()
        self._web.broadcast_memories(memories)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import web.server as web_server

    buddy = Buddy()
    buddy.boot()

    web_server.set_buddy(buddy)

    web_thread = threading.Thread(
        target = web_server.run,
        kwargs = {
            "host":  cfg.WEB_HOST,
            "port":  cfg.WEB_PORT,
            "debug": cfg.WEB_DEBUG,
        },
        daemon = True,
        name   = "WebServer",
    )
    web_thread.start()

    logger.info("Open your browser at http://%s:%d", cfg.WEB_HOST, cfg.WEB_PORT)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down…")
        buddy.shutdown()
