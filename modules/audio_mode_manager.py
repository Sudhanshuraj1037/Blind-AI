# modules/audio_mode_manager.py — Audio Mode Manager v2 FIXED
#
# CRITICAL BUG FIX:
#   Old code only switched modes when checking the OPPOSITE state.
#   e.g. is_mic_active() only switched mic→speaker, never speaker→mic.
#   Result: deadlock — system stuck in starting mode forever.
#
# FIX: Unified _update_mode() method called on EVERY check.
#      Mode transitions now happen regardless of which getter is called.

import threading
import time


class AudioModeManager:
    """
    Manages alternating microphone and speaker access.
    Prevents mic from blocking speaker during output.
    """

    def __init__(self, mic_duration=10, speaker_duration=10):
        """
        Initialize audio mode manager.

        Args:
            mic_duration (float): Seconds to keep microphone active
            speaker_duration (float): Seconds to keep speaker active
        """
        self.mic_duration = mic_duration
        self.speaker_duration = speaker_duration
        self._mode = "speaker"  # Start with speaker for initial greeting
        self._cycle_start = time.time()
        self._lock = threading.Lock()
        self._running = True

        print(
            f"[AudioModeManager] Initialized with {mic_duration}s mic / {speaker_duration}s speaker (starting in SPEAKER mode)"
        )

    def _update_mode(self):
        """
        Internal: advance mode if current window expired.
        MUST be called with self._lock already held.
        
        FIX: This is the core fix. Previously, mode transitions only
        happened inside is_mic_active() or is_speaker_active() when
        the current mode matched. Now we always check and transition.
        """
        elapsed = time.time() - self._cycle_start
        if self._mode == "mic" and elapsed >= self.mic_duration:
            self._mode = "speaker"
            self._cycle_start = time.time()
            print(
                f"[AudioModeManager] Switching to SPEAKER mode (mic was active for {elapsed:.1f}s)"
            )
        elif self._mode == "speaker" and elapsed >= self.speaker_duration:
            self._mode = "mic"
            self._cycle_start = time.time()
            print(
                f"[AudioModeManager] Switching to MIC mode (speaker was active for {elapsed:.1f}s)"
            )

    def is_mic_active(self) -> bool:
        """Check if microphone should be active."""
        with self._lock:
            self._update_mode()
            return self._mode == "mic"

    def is_speaker_active(self) -> bool:
        """Check if speaker should be active."""
        with self._lock:
            self._update_mode()
            return self._mode == "speaker"

    def force_speaker_mode(self):
        """Immediately switch to speaker mode (e.g., for urgent announcements)."""
        with self._lock:
            if self._mode != "speaker":
                print("[AudioModeManager] Force switching to SPEAKER mode")
                self._mode = "speaker"
                self._cycle_start = time.time()

    def force_mic_mode(self):
        """Immediately switch to microphone mode."""
        with self._lock:
            if self._mode != "mic":
                print("[AudioModeManager] Force switching to MIC mode")
                self._mode = "mic"
                self._cycle_start = time.time()

    def get_remaining_time(self) -> float:
        """Get remaining time in current mode."""
        with self._lock:
            self._update_mode()
            elapsed = time.time() - self._cycle_start
            if self._mode == "mic":
                remaining = self.mic_duration - elapsed
            else:
                remaining = self.speaker_duration - elapsed
            return max(0, remaining)

    def get_current_mode(self) -> str:
        """Get current active mode: 'mic' or 'speaker'."""
        with self._lock:
            self._update_mode()
            return self._mode

    def stop(self):
        """Stop the mode manager."""
        self._running = False
        print("[AudioModeManager] Stopped")


# ── Global instance (optional helper) ──────────────────
_manager = None


def get_audio_manager() -> AudioModeManager:
    """Get or create global audio mode manager."""
    global _manager
    if _manager is None:
        _manager = AudioModeManager()
    return _manager


def init_audio_manager(mic_duration=10, speaker_duration=10) -> AudioModeManager:
    """Initialize global audio mode manager."""
    global _manager
    _manager = AudioModeManager(mic_duration=mic_duration, speaker_duration=speaker_duration)
    return _manager