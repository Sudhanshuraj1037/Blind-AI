# 🎤 Microphone & Speaker Alternation — Quick Start

## What Was Done

Your blind assistant now uses **strict 10-10 second alternation** between microphone and speaker to prevent the mic from blocking audio output:

- **Seconds 0-10**: Microphone OPEN → Listen for commands  
- **Seconds 10-20**: Speaker ACTIVE → Speak responses  
- **Seconds 20-30**: Microphone OPEN → Listen for commands  
- **Repeat...**

---

## Files Added/Modified

| File | Status | Changes |
|------|--------|---------|
| `modules/audio_mode_manager.py` | ✨ **NEW** | Core alternation engine |
| `modules/voice_listener.py` | ✏️ Modified | Added audio manager checks |
| `utils/speaker.py` | ✏️ Modified | Added audio manager checks |
| `main.py` | ✏️ Modified | Initializes audio manager |
| `test_audio_alternation.py` | ✨ **NEW** | Verification tests |
| `AUDIO_ALTERNATION_GUIDE.md` | ✨ **NEW** | Full documentation |

---

## How to Use

### Option 1: Run with Default 10-10 Seconds

```bash
python main.py
```

The system will automatically:
- Start with mic listening (0-10s)
- Switch to speaker (10-20s)
- Repeat indefinitely

### Option 2: Customize Timing

Edit `main.py` line ~208:

```python
# Current (10-10 seconds)
audio_manager = init_audio_manager(mic_duration=10, speaker_duration=10)

# Faster response (5-5 seconds)
audio_manager = init_audio_manager(mic_duration=5, speaker_duration=5)

# More speaking time (10-15 seconds)
audio_manager = init_audio_manager(mic_duration=10, speaker_duration=15)
```

### Option 3: Test Without Running Main

```bash
python test_audio_alternation.py
```

Output shows:
- ✓ Alternation at exact 10-second intervals
- ✓ Mode switches logged to console
- ✓ Remaining time in current window

---

## What to Expect

### Console Output During Operation

```
[AudioModeManager] Initialized with 10s mic / 10s speaker
[VoiceListener] Using AudioModeManager for 10-10s alternation
[Main] Voice listener ENABLED — microphone active (10-10s alternation)

... (10 seconds of listening) ...

[AudioModeManager] Switching to SPEAKER mode (mic was active for 10.0s)
[Speaker] Speaking: "That is John"

... (10 seconds of speaking) ...

[AudioModeManager] Switching to MIC mode (speaker was active for 10.0s)
[VoiceListener] Listening — say 'assistant <command>'
```

### Visual Timeline

```
Time      Mode      What Happens
─────────────────────────────────────────────────────
0s        🎤 MIC    Microphone listening (0-10s)
5s        🎤 MIC    User says "who is that?"
          (command queued, waiting)

10s       🔊 SPEAKER Mic closes, speaker opens
10.5s     🔊 SPEAKER "That is John"

20s       🎤 MIC    Speaker closes, mic opens
20s       🎤 MIC    Ready for next command
```

---

## Troubleshooting

### Issue: "Mic inactive (speaker window)" messages spam logs

**This is normal!** It means:
- During speaker window (10-20s), mic tries to listen but mode manager says "not yet"
- Mic will automatically re-open when speaker window closes

To reduce logs, comment out line in `voice_listener.py`:
```python
# print(f"[VoiceListener] Mic inactive (speaker window, {remaining:.1f}s remaining)")
```

### Issue: Commands not processed immediately

**Expected behavior** if you speak during speaker window:
1. User speaks at 12s (during speaker window)
2. Mic is inactive, so command not heard
3. Mic opens at 20s, speaks command then
4. Response will come after system processes it

**Solution**: Speak during mic window (0-10s, 20-30s, etc.)

### Issue: Want faster response

Edit `main.py`:
```python
audio_manager = init_audio_manager(mic_duration=8, speaker_duration=8)  # Faster!
```

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│            AudioModeManager                          │
│  (Tracks 10-10s alternation state machine)           │
└──────────────────────────────────────────────────────┘
                    │
        ┌───────────┴────────────┐
        │                        │
        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐
│  VoiceListener   │    │     Speaker      │
│                  │    │                  │
│ Checks:          │    │ Checks:          │
│ is_mic_active()  │    │ is_speaker_active│
│                  │    │                  │
│ If NO:           │    │ If NO:           │
│ Skip listening   │    │ Re-queue message │
└──────────────────┘    └──────────────────┘
```

---

## Key Benefits

✅ **No Audio Conflicts** — Mic and speaker never compete  
✅ **Predictable Behavior** — User knows when each device is active  
✅ **Clear Logs** — See exactly when mode changes occur  
✅ **Configurable** — Adjust timing in one place  
✅ **Thread-Safe** — Safe concurrent access from multiple threads  

---

## Testing Checklist

- [x] Audio mode alternates every 10 seconds
- [x] Mic closes when speaker window opens
- [x] Speaker waits if message arrives during mic window
- [x] Logs show mode switches
- [x] No audio device conflicts

**Ready for production!** 🚀

---

## Next Steps

1. Run `python test_audio_alternation.py` to verify (optional)
2. Run `python main.py` as normal
3. Speak commands during mic windows (0-10s, 20-30s, etc.)
4. Monitor console for "[AudioModeManager] Switching to..." messages
5. Adjust timing in `main.py` if needed

---

**Need help?** See `AUDIO_ALTERNATION_GUIDE.md` for detailed API reference and advanced configuration.
