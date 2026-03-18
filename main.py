"""
main.py — Buddy Orchestrator (FINAL FIXED, Kokoro-only).
"""

import logging
import threading
import time

import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("buddy.main")


class Buddy:

    def __init__(self):
        self._state       = "idle"
        self._abort_flag  = threading.Event()
        self._listen_lock = threading.Lock()

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

        # STT
        self.stt = WhisperSTT(
            model=cfg.WHISPER_MODEL,
            language=cfg.WHISPER_LANGUAGE,
            n_threads=cfg.WHISPER_THREADS,
        )

        # ✅ TTS (KOKORO ONLY)
        self.tts = TTSEngine(
            engine="kokoro",
            voice=cfg.KOKORO_VOICE,
            speed=cfg.KOKORO_SPEED,
            lang=cfg.KOKORO_LANG,
        )

        # LLM
        self.llm = OllamaClient(
            host=cfg.OLLAMA_HOST,
            model=cfg.OLLAMA_MODEL,
            keep_alive=cfg.OLLAMA_KEEP_ALIVE,
            options=cfg.OLLAMA_OPTIONS,
        )
        self.history = ConversationHistory(max_turns=20)

        # Memory + personality
        self.mem       = MemoryClient(cfg.MEM0_CONFIG, user_id=cfg.MEM0_USER_ID)
        self.sentiment = SentimentAnalyser()

        self.prompt_builder = SystemPromptBuilder(
            personality_core=cfg.PERSONALITY_CORE,
            buddy_name=cfg.BUDDY_NAME,
            user_name=cfg.USER_NAME,
            timezone=cfg.TIMEZONE,
            context_file=cfg.CONTEXT_FILE,
            mood_log_file=cfg.MOOD_LOG_FILE,
        )

        self.personality = PersonalityEngine(
            llm_client=self.llm,
            sentiment_analyser=self.sentiment,
            context_file=cfg.CONTEXT_FILE,
            mood_log_file=cfg.MOOD_LOG_FILE,
            user_name=cfg.USER_NAME,
            concern_threshold=cfg.MOOD_CONCERN_THRESHOLD,
            concern_days=cfg.MOOD_CONCERN_DAYS,
            timezone=cfg.TIMEZONE,
        )

        self._parse_timer = parse_timer_intent
        self._timer_label = extract_timer_label

        def _on_timer_done(label, message):
            play_timer_done()
            time.sleep(0.6)
            self._speak_response(message)
            self._web.broadcast_proactive(message)
            self._broadcast_timers()

        self.timers = TimerManager(on_done=_on_timer_done)

        self.wake_detector = WakeWordDetector(
            access_key=cfg.PICOVOICE_ACCESS_KEY,
            keyword=cfg.WAKE_WORD,
            sensitivity=cfg.PORCUPINE_SENSITIVITY,
            device_index=cfg.MIC_DEVICE_INDEX,
            on_wake=self._on_wake_word,
        )

        self.proactive = ProactiveScheduler(
            llm_client=self.llm,
            mem_client=self.mem,
            personality_engine=self.personality,
            tts_engine=self.tts,
            context_file=cfg.CONTEXT_FILE,
            events_file=cfg.EVENTS_FILE,
            buddy_name=cfg.BUDDY_NAME,
            user_name=cfg.USER_NAME,
            timezone=cfg.TIMEZONE,
            morning_hour=cfg.PROACTIVE_MORNING_HOUR,
            evening_hour=cfg.PROACTIVE_EVENING_HOUR,
            on_speak=lambda t: self._web.broadcast_proactive(t),
        )

    @property
    def state(self):
        return self._state

    def _set_state(self, state):
        self._state = state
        self._web.broadcast_state(state)
        logger.info("State → %s", state)

    def boot(self):
        logger.info("═══ Buddy booting ═══")

        self.tts.load()
        self.mem.load()
        self.sentiment.load()

        if self.llm.ping():
            threading.Thread(target=self.llm.warm_up, daemon=True).start()

        if cfg.PICOVOICE_ACCESS_KEY != "YOUR_KEY_HERE":
            self.wake_detector.start()

        if cfg.PROACTIVE_ENABLED:
            self.proactive.start()

        logger.info("═══ Ready ═══")

    def shutdown(self):
        self.wake_detector.stop()
        self.proactive.stop()
        self.timers.cancel_all()
        self.tts.stop()

    def _on_wake_word(self):
        if self._state in ("speaking", "thinking"):
            self.abort()
            time.sleep(0.25)
        self.listen_once()

    def listen_once(self):
        if not self._listen_lock.acquire(blocking=False):
            return
        try:
            if self._state != "idle":
                return

            from audio.beep import play_listening_start, play_processing, play_error
            from audio.mic_input import MicRecorder

            self._abort_flag.clear()
            self._set_state("listening")
            play_listening_start()

            with MicRecorder(
                vad_aggressiveness=cfg.VAD_AGGRESSIVENESS,
                silence_ms=cfg.VAD_SILENCE_MS,
                min_speech_ms=cfg.VAD_MIN_SPEECH_MS,
                device_index=cfg.MIC_DEVICE_INDEX,
            ) as mic:
                wav_bytes = mic.record()

            if not wav_bytes:
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

        finally:
            self._listen_lock.release()

    def handle_text(self, text, from_voice=False):
        if self._state not in ("idle", "thinking"):
            self.abort()
            time.sleep(0.15)

        self._abort_flag.clear()

        if not from_voice:
            self._set_state("thinking")

        # ✅ CRITICAL FIX
        self.history.add_user(text)

        from audio.tts import SentenceStreamer
        streamer = SentenceStreamer(self.tts)
        streamer.start()

        full_response = ""

        def on_token(token):
            nonlocal full_response
            if self._abort_flag.is_set():
                return
            full_response += token
            self._web.broadcast_token(token)
            streamer.push(token)
            if self._state == "thinking":
                self._set_state("speaking")

        def on_done(accumulated):
            streamer.finish()
            self._web.broadcast_response(accumulated)

        self.llm.stream(
            messages=self.history.get(),
            system=self.prompt_builder.build(),
            on_token=on_token,
            on_done=on_done,
        )

        streamer.wait()

        if not self._abort_flag.is_set():
            self.history.add_assistant(full_response)
            self.mem.add(text, full_response)

        self._set_state("idle")

    def _speak_response(self, text):
        self._set_state("speaking")
        self.tts.speak(text)
        self._set_state("idle")

    def abort(self):
        self._abort_flag.set()
        self.tts.stop()
        self._set_state("idle")

    def _broadcast_timers(self):
        self._web.broadcast_timers(self.timers.active_timers())


if __name__ == "__main__":
    import web.server as web_server

    buddy = Buddy()
    buddy.boot()
    web_server.set_buddy(buddy)

    threading.Thread(
        target=web_server.run,
        kwargs={"host": cfg.WEB_HOST, "port": cfg.WEB_PORT},
        daemon=True,
    ).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        buddy.shutdown()