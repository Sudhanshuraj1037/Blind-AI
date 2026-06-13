#!/usr/bin/env python3
# test_audio_alternation.py — Test the 10-10s mic/speaker alternation
#
# This script demonstrates the AudioModeManager controlling
# microphone and speaker access with 10-second intervals.
#
# Usage:
#   python test_audio_alternation.py
#

import sys
import os
import time
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.audio_mode_manager import AudioModeManager


def test_basic_alternation():
    """Test the basic 10-10 second alternation."""
    print("\n" + "="*60)
    print("  Audio Mode Manager — 10-10s Alternation Test")
    print("="*60)
    
    manager = AudioModeManager(mic_duration=10, speaker_duration=10)
    
    print("\nTesting for 45 seconds (should see ~2 complete cycles)...")
    print("- Cycle 1: MIC (0-10s) → SPEAKER (10-20s)")
    print("- Cycle 2: MIC (20-30s) → SPEAKER (30-40s)")
    print()
    
    start_time = time.time()
    last_mode = None
    
    while time.time() - start_time < 45:
        mic_active = manager.is_mic_active()
        speaker_active = manager.is_speaker_active()
        current_mode = manager.get_current_mode()
        remaining = manager.get_remaining_time()
        elapsed = time.time() - start_time
        
        # Only print when mode changes
        if current_mode != last_mode:
            last_mode = current_mode
            print(f"[{elapsed:5.1f}s] → {current_mode.upper():6} mode active")
        
        # Detailed status every 5 seconds
        if int(elapsed) % 5 == 0 and int(elapsed) % 1 == 0:
            print(f"[{elapsed:5.1f}s] Mic: {mic_active}  Speaker: {speaker_active}  "
                  f"Remaining: {remaining:.1f}s  Mode: {current_mode}")
        
        time.sleep(0.1)
    
    print("\n✓ Test complete!")


def test_with_simulated_listeners():
    """Test with simulated mic and speaker listeners."""
    print("\n" + "="*60)
    print("  Audio Mode Manager — Simulated Mic/Speaker Test")
    print("="*60)
    
    manager = AudioModeManager(mic_duration=5, speaker_duration=5)  # Shorter for demo
    
    print("\nSimulating mic and speaker usage (5-5s cycles for 30s)...")
    print()
    
    def mic_simulator():
        """Simulate microphone listening."""
        while True:
            if manager.is_mic_active():
                elapsed = time.time() - start_time
                print(f"[{elapsed:5.1f}s] 🎤 MIC: Listening...")
                time.sleep(1)
            else:
                time.sleep(0.1)
    
    def speaker_simulator():
        """Simulate speaker output."""
        while True:
            if manager.is_speaker_active():
                elapsed = time.time() - start_time
                print(f"[{elapsed:5.1f}s] 🔊 SPEAKER: Speaking...")
                time.sleep(1)
            else:
                time.sleep(0.1)
    
    start_time = time.time()
    
    # Start threads
    mic_thread = threading.Thread(target=mic_simulator, daemon=True)
    speaker_thread = threading.Thread(target=speaker_simulator, daemon=True)
    mic_thread.start()
    speaker_thread.start()
    
    # Run for 30 seconds
    while time.time() - start_time < 30:
        time.sleep(0.5)
    
    print("\n✓ Simulated test complete!")


def test_force_modes():
    """Test force_mic_mode() and force_speaker_mode() methods."""
    print("\n" + "="*60)
    print("  Audio Mode Manager — Force Mode Test")
    print("="*60)
    
    manager = AudioModeManager(mic_duration=10, speaker_duration=10)
    
    print("\nStarting in MIC mode...")
    print(f"Current mode: {manager.get_current_mode()}")
    
    print("\nForcing SPEAKER mode...")
    manager.force_speaker_mode()
    print(f"Current mode: {manager.get_current_mode()}")
    
    print("\nForcing MIC mode...")
    manager.force_mic_mode()
    print(f"Current mode: {manager.get_current_mode()}")
    
    print("\n✓ Force mode test complete!")


if __name__ == "__main__":
    try:
        # Run all tests
        test_basic_alternation()
        test_with_simulated_listeners()
        test_force_modes()
        
        print("\n" + "="*60)
        print("  All tests completed successfully! ✓")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
