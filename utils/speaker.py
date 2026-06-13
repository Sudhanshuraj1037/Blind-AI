# utils/speaker.py — v11 FIXED (Windows pyttsx3 stuck-state bug)
#
# CRITICAL FIX: pyttsx3 on Windows gets stuck after first utterance.
#   Only "Starting blind assistant" plays, then silent forever.
#   ROOT CAUSE: SAPI5 COM event loop enters bad state after first
#   runAndWait() in a threaded environment. Subsequent calls return
#   immediately without producing audio.
#
# SOLUTION: Create a NEW pyttsx3 engine for EVERY utterance.
#   Slightly slower (~50ms) but 100% reliable on Windows.
#   This is the recommended pattern in pyttsx3 GitHub issues.

import threading
import time
import config

# Try to import audio mode manager, but fallback if import fails
try:
    from modules.audio_mode_manager import get_audio_manager
    _AUDIO_MANAGER_AVAILABLE = True
except ImportError:
    _AUDIO_MANAGER_AVAILABLE = False
    print("[Speaker] WARNING: AudioModeManager not available")


class Speaker:

    def __init__(self):
        self._items        = []
        self._lock         = threading.Lock()
        self._running      = False
        self._thread       = None
        self._speaking     = False
        self._last_spoken  = []
        self._fail_count   = 0
        self._audio_lock   = None
        self._audio_manager = get_audio_manager() if _AUDIO_MANAGER_AVAILABLE else None
        self._use_powershell_fallback = False  # set True if pyttsx3 fully fails

    def set_audio_lock(self, lock):
        self._audio_lock = lock
        print("[Speaker] Audio device lock acquired for mic/speaker coordination.")

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._worker, daemon=True, name="Speaker"
        )
        self._thread.start()
        print("[Speaker] TTS engine started.")

    def stop(self):
        self._running = False

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def say(self, text: str, priority: int = 5):
        if not text:
            return
        text = text.strip()

        with self._lock:
            # Drop exact duplicate already in queue
            for _, _, existing in self._items:
                if existing == text:
                    return

            # skip if same as any of last 3 spoken AND queue is empty
            if text in self._last_spoken and not self._items:
                return

            # Queue full — drop lowest priority item
            max_q = getattr(config, "TTS_QUEUE_MAX", 5)
            if len(self._items) >= max_q:
                min_item = min(self._items, key=lambda x: x[0], default=None)
                if min_item and min_item[0] < priority:
                    self._items.remove(min_item)
                else:
                    return

            self._items.append((priority, time.time(), text))
            self._items.sort(key=lambda x: (-x[0], x[1]))

    def say_now(self, text: str):
        if not text:
            return
        with self._lock:
            danger_prio   = config.PRIORITY["danger"]
            self._items   = [i for i in self._items if i[0] >= danger_prio]
            self._items.insert(0, (danger_prio, time.time(), text))

    def clear_queue(self):
        with self._lock:
            n = len(self._items)
            self._items.clear()
        if n:
            print(f"[Speaker] Cleared {n} message(s).")

    # ── Worker thread ─────────────────────────────────────

    def _worker(self):
        """
        Polling-based worker.
        Creates a NEW pyttsx3 engine for each utterance to avoid
        Windows SAPI5 stuck-state bug.
        """
        # Initialize COM for this thread (Windows SAPI5 requires it)
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass

        # Test once if pyttsx3 is available at all
        try:
            import pyttsx3
            test_engine = pyttsx3.init("sapi5")
            del test_engine
            print("[Speaker] pyttsx3 SAPI5 available — will create fresh engine per utterance.")
        except Exception as e:
            print(f"[Speaker] pyttsx3 unavailable ({e}) — using PowerShell fallback.")
            self._use_powershell_fallback = True

        while self._running:
            # Check if speaker is allowed in this window
            if self._audio_manager and not self._audio_manager.is_speaker_active():
                time.sleep(0.1)
                continue

            # Pop next item from queue
            text = None
            with self._lock:
                if self._items:
                    _, _, text = self._items.pop(0)
                    self._last_spoken.append(text)
                    if len(self._last_spoken) > 3:
                        self._last_spoken.pop(0)

            if text is None:
                time.sleep(0.1)
                continue

            # Sanitize text for TTS
            safe = (
                text
                .replace("'",  "")
                .replace('"',  "")
                .replace("\n", " ")
                .replace("\\", "")
            )
            print(f"[Speaker] Speaking: {safe}")

            self._speaking = True
            try:
                self._do_speak_fresh(safe)
            finally:
                self._speaking = False
                time.sleep(0.2)

        # Cleanup COM
        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except Exception:
            pass

        print("[Speaker] Worker thread exiting.")

    def _do_speak_fresh(self, text: str):
        """
        WINDOWS FIX: Create a fresh pyttsx3 engine for this utterance.
        This avoids the SAPI5 COM stuck-state bug where subsequent
        runAndWait() calls return immediately without playing audio.
        """
        if self._use_powershell_fallback:
            self._speak_powershell(text)
            return

        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init("sapi5")
            engine.setProperty("rate",   config.TTS_RATE)
            engine.setProperty("volume", config.TTS_VOLUME)
            voices = engine.getProperty("voices")
            if voices and config.TTS_VOICE_INDEX < len(voices):
                engine.setProperty("voice", voices[config.TTS_VOICE_INDEX].id)

            engine.say(text)
            engine.runAndWait()
            self._fail_count = 0

        except Exception as e:
            print(f"[Speaker] pyttsx3 error: {e}")
            self._fail_count += 1
            # After 3 failures, permanently switch to PowerShell
            if self._fail_count >= 3:
                print("[Speaker] Switching permanently to PowerShell TTS.")
                self._use_powershell_fallback = True
            # Fallback for this utterance
            try:
                self._speak_powershell(text)
            except Exception as e2:
                print(f"[Speaker] PowerShell fallback failed: {e2}")

        finally:
            # CRITICAL: properly destroy the engine to release COM resources
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
                try:
                    del engine
                except Exception:
                    pass

    @staticmethod
    def _speak_powershell(text: str):
        import subprocess
        # Escape single quotes for PowerShell
        text_safe = text.replace("'", "''")
        ps_cmd = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = 2; $s.Speak('{text_safe}')"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
            timeout=20,
            capture_output=True,
        )