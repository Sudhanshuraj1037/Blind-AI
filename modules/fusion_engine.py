# ============================================================
#  modules/fusion_engine.py — Blind Assistant v8 FIXED
#
#  ROOT CAUSE OF SILENCE BUG:
#  The old fusion engine only auto-announced DANGER_CLASSES.
#  A mobile phone, chair, cup, book etc. were NEVER spoken
#  unless the user pressed SPACE or said "look around".
#
#  FIX: Added _reflex_objects() which announces ALL detected
#  objects every COOLDOWN_OBJECTS seconds automatically.
#
#  NOW: Camera detects phone → speaker says:
#       "I can see a cell phone ahead."
# ============================================================

import time
import config

_URGENCY = {
    "critical": config.PRIORITY["danger"],
    "high":     config.PRIORITY["command"],
    "medium":   config.PRIORITY["objects"],
    "low":      2,
}

# These classes trigger IMMEDIATE danger warnings
_AUTO_DANGER_CLASSES = {"car", "truck", "bus", "stairs", "motorcycle", "bicycle"}

# Natural spoken names for YOLO class labels
_SPOKEN_NAMES = {
    "cell phone":     "mobile phone",
    "remote":         "remote control",
    "tv":             "television",
    "laptop":         "laptop computer",
    "backpack":       "backpack",
    "handbag":        "handbag",
    "suitcase":       "suitcase",
    "bottle":         "bottle",
    "cup":            "cup",
    "book":           "book",
    "scissors":       "scissors",
    "keyboard":       "keyboard",
    "mouse":          "mouse",
    "potted plant":   "plant",
    "chair":          "chair",
    "couch":          "sofa",
    "dining table":   "table",
    "bed":            "bed",
    "toilet":         "toilet",
    "sink":           "sink",
    "refrigerator":   "refrigerator",
    "microwave":      "microwave",
    "oven":           "oven",
    "clock":          "clock",
    "vase":           "vase",
    "umbrella":       "umbrella",
    "tie":            "tie",
    "skis":           "skis",
    "baseball bat":   "baseball bat",
    "fire hydrant":   "fire hydrant",
    "stop sign":      "stop sign",
    "parking meter":  "parking meter",
    "bench":          "bench",
    "bird":           "bird",
    "cat":            "cat",
    "dog":            "dog",
    "horse":          "horse",
    "cow":            "cow",
    "elephant":       "elephant",
    "bear":           "bear",
    "zebra":          "zebra",
    "giraffe":        "giraffe",
    "person":         "person",
    "bicycle":        "bicycle",
    "car":            "car",
    "motorcycle":     "motorcycle",
    "airplane":       "airplane",
    "bus":            "bus",
    "train":          "train",
    "truck":          "truck",
    "boat":           "boat",
}


def _spoken(label: str) -> str:
    """Convert YOLO class name to natural spoken word."""
    return _SPOKEN_NAMES.get(label.lower(), label)


class FusionEngine:
    """
    Rule-based fusion engine.
    Decides WHAT to say, WHEN to say it, and at what PRIORITY.

    Automatic (always running):
      • Danger objects    → immediate warning, repeat every COOLDOWN_DANGER
      • Depth warning     → every COOLDOWN_DEPTH
      • ANY object        → every COOLDOWN_OBJECTS  ← THIS WAS MISSING (the bug)
      • Gesture           → every COOLDOWN_GESTURE
      • Emotion           → every COOLDOWN_EMOTION
      • Face              → every COOLDOWN_FACE
      • Currency          → every COOLDOWN_CURRENCY
      • Path clear        → after 5s of nothing

    On-demand (voice command or key press):
      • scene / ocr / objects / face / emotion / currency / depth / gps
    """

    def __init__(self, speaker, scene_module=None, face_module=None,
                 groq_assistant=None, gemini_assistant=None, trainer=None, memory=None):
        self.speaker       = speaker
        self._scene        = scene_module
        self._face_mod     = face_module
        # Support both Groq and Gemini (Groq takes priority)
        self._ai           = groq_assistant or gemini_assistant
        self._active_cmd   = None
        self._last_frame   = None

        # ── Cooldown timestamps (last time each type was spoken) ──
        self._last = {
            "danger":   0.0,
            "depth":    0.0,
            "objects":  0.0,
            "gesture":  0.0,
            "emotion":  0.0,
            "face":     0.0,
            "currency": 0.0,
            "clear":    0.0,
        }

        # Depth gate: require N consecutive frames before speaking
        self._depth_cons   = 0
        self._depth_req    = getattr(config, "DEPTH_CONSECUTIVE_FRAMES", 2)

        # Path-clear tracker
        self._clear_since       = None
        self._path_clear_spoken = False
        self._PATH_CLEAR_DELAY  = 5.0

        # Pause (after 'stop' command)
        self._paused_until = 0.0

        ai_name = "Groq" if groq_assistant else ("Gemini" if gemini_assistant else "None")
        print(f"[Fusion] v8 ready. AI backend: {ai_name}")
        print(f"[Fusion] Object announce interval: {config.COOLDOWN_OBJECTS}s")

    # ── Public API ────────────────────────────────────────

    def set_command(self, command: str):
        self._active_cmd = command

    def process_feedback(self, raw_text: str) -> float:
        """Parse voice feedback, return reward value."""
        if not raw_text:
            return 0.0
        text = raw_text.lower().strip()

        pos = getattr(config, "FEEDBACK_POSITIVE",
                      ["good", "yes", "correct", "helpful", "right", "nice"])
        neg = getattr(config, "FEEDBACK_NEGATIVE",
                      ["wrong", "no", "quiet", "stop", "too much", "annoying"])

        for w in pos:
            if w in text:
                print(f"[Fusion] Positive feedback")
                return +1.0
        for w in neg:
            if w in text:
                if w in ("stop", "quiet", "too much"):
                    self.speaker.clear_queue()
                print(f"[Fusion] Negative feedback")
                return -1.0
        return 0.0

    def process(self, results: dict, nav_state=None, frame=None):
        """Main loop — called every frame from main.py."""
        now = time.time()
        if frame is not None:
            self._last_frame = frame

        if now >= self._paused_until:
            self._auto_danger(results, now)
            self._auto_depth(results, now)
            self._auto_objects(results, now)   # ← THE KEY FIX
            self._auto_gesture(results, now)
            self._auto_emotion(results, now)
            self._auto_face(results, now)
            self._auto_currency(results, now)
            self._auto_path_clear(results, now)

        if self._active_cmd:
            cmd = self._active_cmd
            self._active_cmd = None
            self._handle_command(cmd, results, now)

    # ── AUTOMATIC REFLEXES ────────────────────────────────

    def _auto_danger(self, results: dict, now: float):
        """Immediately warn about cars, buses, stairs etc."""
        dets   = results.get("detections") or []
        danger = [
            d for d in dets
            if d.get("label", "").lower() in _AUTO_DANGER_CLASSES
            and d.get("is_danger", False)
        ]
        if not danger:
            return
        if now - self._last["danger"] < config.COOLDOWN_DANGER:
            return

        d        = danger[0]
        name     = _spoken(d["label"])
        position = d.get("position", "ahead")
        self.speaker.say(
            f"Warning! {name} {position}.",
            priority=_URGENCY["critical"],
        )
        self._last["danger"] = now

    def _auto_depth(self, results: dict, now: float):
        """Announce depth/obstacle warnings with consecutive-frame gate."""
        dw = results.get("depth_warning")
        if dw:
            self._depth_cons = min(self._depth_cons + 1, 10)
        else:
            self._depth_cons = max(self._depth_cons - 1, 0)

        if self._depth_cons < self._depth_req or not dw:
            return
        if now - self._last["depth"] < config.COOLDOWN_DEPTH:
            return

        self.speaker.say(dw, priority=config.PRIORITY["depth"])
        self._last["depth"] = now

    def _auto_objects(self, results: dict, now: float):
        """
        ── THE MAIN BUG FIX ──
        Periodically announce ALL detected objects.
        This is why the phone showed on screen but was never spoken:
        it wasn't in _AUTO_DANGER_CLASSES so the old code skipped it.

        Now: phone detected → "I can see a mobile phone ahead."
             chair detected → "I can see a chair on your left."
             cup + book     → "I can see a cup ahead and a book on your right."
        """
        if now - self._last["objects"] < config.COOLDOWN_OBJECTS:
            return

        dets = results.get("detections") or []
        if not dets:
            return

        # Skip if danger just fired (avoid double-speak)
        if now - self._last["danger"] < 2.0:
            return

        # Build spoken description for ALL objects (danger + non-danger)
        # Danger objects already announced by _auto_danger — mention them
        # briefly here if not just spoken
        parts = []
        seen  = set()

        for d in dets[:4]:   # max 4 objects per announcement
            name     = _spoken(d["label"])
            position = d.get("position", "ahead")
            key      = name.lower()
            if key in seen:
                continue
            seen.add(key)
            parts.append(f"{name} {position}")

        if not parts:
            return

        if len(parts) == 1:
            msg = f"I can see a {parts[0]}."
        else:
            msg = "I can see " + ", ".join(parts) + "."

        print(f"[Fusion] Objects: {msg}")
        self.speaker.say(msg, priority=config.PRIORITY["objects"])
        self._last["objects"] = now

    def _auto_gesture(self, results: dict, now: float):
        g = results.get("gesture_text")
        if g and now - self._last["gesture"] > config.COOLDOWN_GESTURE:
            self.speaker.say(g, priority=config.PRIORITY["gesture"])
            self._last["gesture"] = now

    def _auto_emotion(self, results: dict, now: float):
        e = results.get("emotion_text")
        if e and now - self._last["emotion"] > config.COOLDOWN_EMOTION:
            self.speaker.say(e, priority=config.PRIORITY["emotion"])
            self._last["emotion"] = now

    def _auto_face(self, results: dict, now: float):
        f = results.get("face_text")
        if f and now - self._last["face"] > config.COOLDOWN_FACE:
            self.speaker.say(f, priority=config.PRIORITY["face"])
            self._last["face"] = now

    def _auto_currency(self, results: dict, now: float):
        c = results.get("currency_text")
        if c and now - self._last["currency"] > config.COOLDOWN_CURRENCY:
            self.speaker.say(c, priority=config.PRIORITY["currency"])
            self._last["currency"] = now

    def _auto_path_clear(self, results: dict, now: float):
        """Say 'path is clear' after 5 seconds of no obstacles."""
        dets     = results.get("detections") or []
        has_obs  = (
            any(d.get("is_danger") for d in dets)
            or bool(results.get("depth_warning"))
            or any(d["label"] == "person" for d in dets)
        )
        if has_obs:
            self._clear_since       = None
            self._path_clear_spoken = False
            return

        if self._clear_since is None:
            self._clear_since = now
            return
        if self._path_clear_spoken:
            return
        if now - self._last["clear"] < 15.0:
            return
        if now - self._clear_since < self._PATH_CLEAR_DELAY:
            return

        self.speaker.say("Path is clear. You may proceed.", priority=_URGENCY["low"])
        self._last["clear"]     = now
        self._path_clear_spoken = True

    # ── COMMAND HANDLING ──────────────────────────────────

    def _handle_command(self, cmd: str, results: dict, now: float):
        """Handle voice commands and keyboard shortcuts."""
        if cmd == "stop":
            self.speaker.clear_queue()
            self.speaker.say("Stopped.", priority=_URGENCY["high"])
            self._paused_until = now + 8.0
            return

        if cmd == "help":
            self.speaker.say(
                "Available commands: look around, read text, who is that, "
                "describe scene, detect emotion, check currency, "
                "navigate, where am I, weather, stop, help.",
                priority=_URGENCY["high"],
            )
            return

        if cmd == "weather":
            if self._ai:
                self._ai.ask_weather()
            else:
                self.speaker.say("Weather requires an AI key.", priority=_URGENCY["high"])
            return

        if cmd == "depth":
            dw = results.get("depth_warning")
            self.speaker.say(dw or "Path looks clear ahead.", priority=_URGENCY["high"])
            return

        if cmd == "gps":
            gps = results.get("gps_text")
            self.speaker.say(gps or "GPS is not active.", priority=_URGENCY["high"])
            return

        if cmd == "objects":
            # Force immediate object announcement
            dets = results.get("detections") or []
            if not dets:
                self.speaker.say("I don't see any objects right now.", priority=_URGENCY["high"])
                return
            parts = []
            seen  = set()
            for d in dets[:5]:
                name = _spoken(d["label"])
                if name.lower() not in seen:
                    seen.add(name.lower())
                    parts.append(f"{name} {d.get('position','ahead')}")
            msg = "I can see " + ", ".join(parts) + "." if parts else "Nothing detected."
            self.speaker.say(msg, priority=_URGENCY["high"])
            self._last["objects"] = now
            return

        # All other commands use AI if available, else local fallback
        if self._last_frame is None:
            self.speaker.say("No camera frame yet.", priority=_URGENCY["high"])
            return

        if self._ai:
            self._route_to_ai(cmd, results)
        else:
            self._local_fallback(cmd, results)

    def _route_to_ai(self, cmd: str, results: dict):
        """Route vision commands to Groq/Gemini AI."""
        confirmations = {
            "scene":    "Looking at the scene…",
            "ocr":      "Reading the text…",
            "currency": "Checking currency…",
            "face":     "Checking who that is…",
            "emotion":  "Reading the emotion…",
        }
        if cmd in confirmations:
            if cmd in ("face", "emotion"):
                dets = results.get("detections") or []
                if not any(d["label"] == "person" for d in dets):
                    self.speaker.say("I don't see any person.", priority=_URGENCY["high"])
                    return
            self._ai.ask_vision(self._last_frame, cmd, confirmations[cmd])
        else:
            self.speaker.say("Command not recognised.", priority=_URGENCY["high"])

    def _local_fallback(self, cmd: str, results: dict):
        """Local fallback when no AI configured."""
        responses = {
            "ocr":      results.get("ocr_text")      or "No text found.",
            "currency": results.get("currency_text") or "No currency visible.",
            "emotion":  results.get("emotion_text")  or "No emotion detected.",
            "scene":    results.get("scene_text")    or "Scene description unavailable.",
        }
        if cmd == "face":
            dets = results.get("detections") or []
            if not any(d["label"] == "person" for d in dets):
                self.speaker.say("I don't see a person.", priority=_URGENCY["high"])
                return
            if self._face_mod and self._last_frame is not None:
                self.speaker.say("Checking…", priority=5)
                res = self._face_mod.force_recognize(self._last_frame, detections=dets)
                self.speaker.say(res or "I don't recognise this person.", priority=_URGENCY["high"])
            return

        if cmd in responses:
            self.speaker.say(responses[cmd], priority=_URGENCY["high"])
        elif cmd == "scene" and self._scene and self._last_frame is not None:
            self.speaker.say("Looking…", priority=5)
            res = self._scene.describe(self._last_frame)
            self.speaker.say(res or "Cannot describe scene.", priority=_URGENCY["high"])
        else:
            self.speaker.say(
                "AI not configured. Add GROQ_API_KEY to config.py.",
                priority=_URGENCY["high"]
            )