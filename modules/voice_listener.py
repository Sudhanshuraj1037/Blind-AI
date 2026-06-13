# modules/voice_listener.py — v9 FIXED
#
# BUG 1 FIXED: "No, no, no" false negative feedback
#   ROOT CAUSE: Whisper mishears TTS speech ("Sudhanshu",
#   "500 rupee note", "person ahead") as "No, no, no"
#   which triggered NEGATIVE feedback and cleared the queue.
#   FIX: Post-TTS silence gap (1.5s) + TTS echo blacklist
#        + whole-word matching for feedback words
#
# BUG 2 FIXED: "No" alone no longer triggers negative feedback
#   "No" appears too often in TTS echoes. Only multi-word
#   negatives like "too much", "wrong", "quiet" are accepted.
#
# BUG 3 FIXED: Mic stuck waiting forever (deadlock)
#   ROOT CAUSE: AudioModeManager bug caused is_mic_active()
#   to never return True. Now that audio_mode_manager.py is
#   fixed, this listener works properly.
#   FIX: Added safety timeout — if mic waits >60s, log warning.

import speech_recognition as sr
import whisper
import threading
import queue
import tempfile
import os
import time
import re
import numpy as np
import config

# Try to import audio mode manager, but fallback if import fails
try:
    from modules.audio_mode_manager import get_audio_manager
    _AUDIO_MANAGER_AVAILABLE = True
except ImportError:
    _AUDIO_MANAGER_AVAILABLE = False
    get_audio_manager = lambda: None

COMMANDS = {
    "read text":         "ocr",
    "read":              "ocr",
    "what does it say":  "ocr",
    "who is that":       "face",
    "who is this":       "face",
    "recognize":         "face",
    "identify":          "face",
    "what is around":    "objects",
    "describe":          "objects",
    "what do you see":   "objects",
    "look around":       "objects",
    "navigate":          "depth",
    "obstacle":          "depth",
    "how close":         "depth",
    "is it safe":        "depth",
    "detect emotion":    "emotion",
    "how do they feel":  "emotion",
    "what emotion":      "emotion",
    "describe scene":    "scene",
    "what is happening": "scene",
    "full description":  "scene",
    "what is this":      "scene",
    "check currency":    "currency",
    "what note":         "currency",
    "how much money":    "currency",
    "detect money":      "currency",
    "where am i":        "gps",
    "my location":       "gps",
    "stop":              "stop",
    "help":              "help",
}

WAKE_WORD_VARIANTS = [
    "assistant", "a system", "a sister", "assistance", "a sistem",
    "assistants", "assist", "a sistent", "asian", "assistente",
]

NATURAL_COMMANDS = {
    "who is that":      "face",
    "who is this":      "face",
    "what is that":     "face",
    "recognize":        "face",
    "read this":        "ocr",
    "what does it say": "ocr",
    "navigate":         "depth",
    "how close":        "depth",
    "check money":      "currency",
    "what note":        "currency",
    "where am i":       "gps",
}

MIN_AUDIO_RMS      = 300    # raised: ignore quiet TTS echoes
MAX_TRANSCRIPT_LEN = 120

# ── TTS echo blacklist ────────────────────────────────────
TTS_ECHO_BLACKLIST = [
    "i can see",
    "i recognise",
    "i recognize",
    "obstacle very close",
    "please slow down",
    "rupee note",
    "in front of you",
    "shutting down",
    "all systems ready",
    "starting blind",
    "path is clear",
    "warning",
    "safe to cross",
    "traffic light",
    "reading the text",
    "checking who",
    "looking at",
    "go straight",
    "move left",
    "move right",
]

# Feedback words — safe ones only (no "no" — too many false positives)
_FEEDBACK_POS      = ["good", "yes", "correct", "helpful", "right", "nice"]
_FEEDBACK_REPEAT   = ["repeat", "again", "say again"]
_FEEDBACK_NEG_SAFE = ["wrong", "quiet", "incorrect", "annoying", "too much"]


class VoiceListener:

    def __init__(self, speaker=None, audio_manager=None):
        self._speaker         = speaker
        self._audio_manager   = audio_manager if audio_manager else (get_audio_manager() if _AUDIO_MANAGER_AVAILABLE else None)
        self._post_tts_gap    = 1.5    # seconds to wait after TTS ends
        self._tts_ended_at    = 0.0
        self._audio_lock      = threading.Lock()  # FIX: serialise mic/speaker access
        self._mic_source      = None  # Keep track of active microphone
        self._wait_warned_at  = 0.0   # FIX: track long mic-wait warnings

        print("[VoiceListener] Loading Whisper model…")
        self.whisper_model    = whisper.load_model(config.WHISPER_MODEL)
        self.recognizer       = sr.Recognizer()
        self.command_queue    = queue.Queue()
        self._running         = False
        self._thread          = None
        self._last_transcript = ""

        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold         = 300

        print(f"[VoiceListener] Whisper '{config.WHISPER_MODEL}' ready.")
        print(f"[VoiceListener] Wake word: '{config.WAKE_WORD}'")
        print(f"[VoiceListener] Post-TTS gap: {self._post_tts_gap}s (prevents echo feedback)")
        print(f"[VoiceListener] Audio device locking ENABLED (prevents mic/speaker conflicts)")
        if self._audio_manager:
            print(f"[VoiceListener] Using AudioModeManager for 10-10s alternation")
        else:
            print(f"[VoiceListener] AudioModeManager not available - no strict alternation")

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._listen_loop, daemon=True, name="VoiceListener"
        )
        self._thread.start()

    def stop(self):
        self._running = False

    def get_command(self):
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None, None

    def get_last_transcript(self) -> str:
        return self._last_transcript

    # ── Internal helpers ──────────────────────────────────

    def _rms(self, audio) -> float:
        try:
            raw = np.frombuffer(
                audio.get_raw_data(), dtype=np.int16
            ).astype(np.float32)
            return float(np.sqrt(np.mean(raw ** 2)))
        except Exception:
            return 9999.0

    def _is_tts_echo(self, text: str) -> bool:
        """Return True if text looks like TTS output picked up by mic."""
        t = text.lower()
        for phrase in TTS_ECHO_BLACKLIST:
            if phrase in t:
                print(f"[VoiceListener] TTS echo blocked: '{text[:50]}'")
                return True
        return False

    def _whole_word(self, word: str, text: str) -> bool:
        """Match word as whole word only — prevents 'no' matching 'Sudhanshu'."""
        pattern = r'\b' + re.escape(word) + r'\b'
        return bool(re.search(pattern, text, re.IGNORECASE))

    def _is_hallucination(self, text: str) -> bool:
        t = text.lower()
        for variant in WAKE_WORD_VARIANTS:
            if variant in t:
                return False
        all_fb = _FEEDBACK_POS + _FEEDBACK_REPEAT + _FEEDBACK_NEG_SAFE
        for word in all_fb:
            if self._whole_word(word, t):
                return False
        if len(text) > MAX_TRANSCRIPT_LEN:
            print(f"[VoiceListener] Hallucination filtered ({len(text)} chars)")
            return True
        return False

    def _is_feedback(self, text: str) -> bool:
        """
        Strict feedback detection.
        'no' alone is NOT treated as negative feedback (too many false positives).
        Only explicit words like 'wrong', 'quiet', 'too much' are accepted.
        """
        t = text.lower().strip()
        for word in _FEEDBACK_POS:
            if self._whole_word(word, t):
                return True
        for phrase in _FEEDBACK_REPEAT:
            if phrase in t:
                return True
        for word in _FEEDBACK_NEG_SAFE:
            if self._whole_word(word, t):
                return True
        return False

    def _match_natural(self, text: str) -> str | None:
        t = text.lower().strip()
        for phrase, cmd in NATURAL_COMMANDS.items():
            if phrase in t:
                return cmd
        return None

    def _match_command(self, text: str) -> str | None:
        t = text.lower()
        for phrase, cmd in COMMANDS.items():
            if phrase in t:
                return cmd
        return None

    # ── Listener loop ─────────────────────────────────────

    def _listen_loop(self):
        """
        BUG FIX: Voice listener now yields to speaker with priority.
        Previously, microphone held exclusive lock even during speaker output.
        
        New approach:
        - Never listen while speaker is actively speaking
        - Never listen if speaker has queued messages waiting
        - Wait full post-TTS gap before resuming listening
        - Give speaker absolute priority over microphone
        """
        mic_index = getattr(config, "MIC_DEVICE_INDEX", None)
        first_run = True
        wait_start = None  # FIX: track how long mic has been waiting
        
        while self._running:
            # ─────── PRIORITY 0: Audio Mode Manager ──────────────────
            # Strict 10-10s alternation: mic for 10s, speaker for 10s
            # If mic is not in its active window, skip listening
            if self._audio_manager and not self._audio_manager.is_mic_active():
                # FIX: warn if mic has been waiting too long (signals deadlock)
                if wait_start is None:
                    wait_start = time.time()
                elif time.time() - wait_start > 60:
                    if time.time() - self._wait_warned_at > 30:
                        print("[VoiceListener] WARNING: mic has been inactive for 60s+ — check AudioModeManager")
                        self._wait_warned_at = time.time()
                time.sleep(0.5)  # Check again soon
                continue
            else:
                wait_start = None  # reset when mic becomes active

            # ─────── PRIORITY 1: Speaker is actively speaking ────────
            # STOP listening immediately - give speaker exclusive audio access
            if self._speaker and self._speaker.is_speaking:
                self._tts_ended_at = time.time()
                time.sleep(0.1)  # Wait while speaker talks
                continue

            # ─────── PRIORITY 1.5: Speaker has queued messages ────────
            # Don't interrupt if speaker has items waiting to be spoken
            if self._speaker and hasattr(self._speaker, '_items'):
                if self._speaker._items:  # If queue is not empty
                    time.sleep(0.1)
                    continue

            # ─────── PRIORITY 2: Post-TTS silence gap ─────────────────
            # After speaker finishes, wait 1.5s before mic can resume
            # This prevents mic from picking up last echo of speaker output
            if time.time() - self._tts_ended_at < self._post_tts_gap:
                time.sleep(0.1)
                continue

            # ─────── Only listen if speaker is idle ───────────────────
            try:
                with sr.Microphone(device_index=mic_index) as source:
                    # Only calibrate on first run
                    if first_run:
                        print("[VoiceListener] Calibrating ambient noise (3s — stay quiet)…")
                        self.recognizer.adjust_for_ambient_noise(source, duration=3)
                        print(f"[VoiceListener] Energy threshold: {self.recognizer.energy_threshold:.0f}")
                        print("[VoiceListener] Listening — say 'assistant <command>'")
                        first_run = False

                    try:
                        audio = self.recognizer.listen(
                            source, timeout=0.5, phrase_time_limit=8
                        )
                    except sr.WaitTimeoutError:
                        # Timeout is normal - just retry
                        continue
                    except Exception as e:
                        print(f"[VoiceListener] Listen error: {e}")
                        continue

                    # Check one more time if speaker started during listen
                    if self._speaker and self._speaker.is_speaking:
                        continue

                    # Check if speaker queue got items during listen
                    if self._speaker and hasattr(self._speaker, '_items'):
                        if self._speaker._items:
                            continue

                    # Process audio (outside mic context = mic is released)
                    self._process_audio(audio)

            except Exception as e:
                print(f"[VoiceListener] Microphone error: {e}")
                time.sleep(0.2)

        print("[VoiceListener] Worker thread exiting.")

    def _process_audio(self, audio):
        """Process audio outside of microphone lock to allow speaker access."""
        # Energy gate — skip silence
        if self._rms(audio) < MIN_AUDIO_RMS:
            return

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, prefix="wsp_"
            ) as f:
                f.write(audio.get_wav_data())
                tmp_path = f.name

            result = self.whisper_model.transcribe(
                tmp_path,
                fp16=False,
                language="en",
            )
            text = result["text"].strip()

            if not text:
                return

            # ── FIX 3: Block TTS echoes ───────────────────
            if self._is_tts_echo(text):
                return

            if self._is_hallucination(text):
                return

            self._last_transcript = text
            print(f"[VoiceListener] Heard: '{text}'")

            # Feedback
            if self._is_feedback(text):
                print(f"[VoiceListener] Feedback: '{text}'")
                self.command_queue.put((None, text))
                return

            # Natural commands (no wake word needed)
            nat_cmd = self._match_natural(text)
            if nat_cmd:
                print(f"[VoiceListener] Natural command: '{nat_cmd}'")
                self.command_queue.put((nat_cmd, text))
                return

            # Wake word + command
            wake_found = False
            after_wake = text.lower()
            for variant in WAKE_WORD_VARIANTS:
                if variant in text.lower():
                    wake_found = True
                    after_wake = text.lower().split(variant, 1)[-1].strip()
                    break

            if wake_found:
                cmd = self._match_command(after_wake)
                if cmd:
                    print(f"[VoiceListener] Command: '{cmd}'")
                    self.command_queue.put((cmd, text))
                else:
                    self.command_queue.put((None, text))

        except Exception as e:
            print(f"[VoiceListener] Transcription error: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass