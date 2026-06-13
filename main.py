#!/usr/bin/env python3
# ============================================================
#  main.py — Blind Assistant v8  FIXED + OPTIMIZED
#
#  KEY FIXES:
#  1. Uses GroqAssistant instead of GeminiAssistant
#  2. fusion_engine now announces ALL objects (phone, chair etc)
#  3. FaceRecognizer no longer spams terminal
#  4. Speaker is properly wired through all modules
#
#  Threads:
#    1. Camera capture   (30 fps, background)
#    2. ML Worker        (YOLO, MiDaS, OCR, Face, Gesture …)
#    3. Speaker / TTS    (priority-queue, non-blocking)
#    4. VoiceListener    (Whisper, muted during TTS)
#
#  Controls:
#    Q     = quit
#    SPACE = describe objects now
#    D     = toggle debug overlay
#    S     = screenshot
# ============================================================

import sys
import os
import cv2
import time
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ── Dependency check ─────────────────────────────────────
def _check_deps():
    required = {
        "cv2":                "opencv-python",
        "pyttsx3":            "pyttsx3",
        "ultralytics":        "ultralytics",
        "easyocr":            "easyocr",
        "torch":              "torch torchvision",
        "whisper":            "openai-whisper",
        "speech_recognition": "SpeechRecognition",
        "deepface":           "deepface",
        "timm":               "timm",
        "PIL":                "Pillow",
        "numpy":              "numpy",
        "requests":           "requests",
    }
    missing = [pkg for mod, pkg in required.items() if not _ok(mod)]
    if missing:
        print(f"\n[ERROR] Missing packages:\n  pip install {' '.join(missing)}\n")
        sys.exit(1)

def _ok(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False

_check_deps()

import config

print("\n" + "=" * 60)
print("  Blind Assistant v8  —  Fixed + Optimized")
print(f"  Device: {config.DEVICE.upper()}  |  RAM mode: {'LOW' if config.LOW_RAM_MODE else 'FULL'}")
groq_key = getattr(config, "GROQ_API_KEY", "")
print(f"  Groq AI: {'CONFIGURED' if groq_key and 'your_key' not in groq_key else 'NOT SET (add GROQ_API_KEY to config.py)'}")
print("=" * 60)

from utils.camera              import Camera
from utils.speaker             import Speaker
from modules.object_detector   import ObjectDetector
from modules.ocr_reader        import OCRReader
from modules.navigator         import Navigator
from modules.audio_mode_manager import init_audio_manager

# FaceRecognizer is optional (requires TensorFlow/DeepFace)
_face = None
try:
    from modules.face_recognizer   import FaceRecognizer
    _face_module = FaceRecognizer
except ImportError as e:
    print(f"[Main] WARNING: FaceRecognizer skipped ({e})")
    _face_module = None

# EmotionDetector is optional (requires DeepFace/TensorFlow)
_emotion = None
try:
    from modules.emotion_detector  import EmotionDetector
    _emotion_module = EmotionDetector
except ImportError as e:
    print(f"[Main] WARNING: EmotionDetector skipped ({e})")
    _emotion_module = None

from modules.gesture_detector  import GestureDetector
from modules.currency_detector import CurrencyDetector
from modules.gps_navigator     import GPSNavigator
from modules.voice_listener    import VoiceListener
from modules.nav_state         import NavState
from modules.fusion_engine     import FusionEngine
from modules.groq_assistant    import GroqAssistant   # ← Groq replaces Gemini

# Scene module only when RAM allows
_scene = None
if not config.LOW_RAM_MODE:
    try:
        from modules.llava_describer import LLaVADescriber
        _scene = LLaVADescriber()
        print("[Main] LLaVA/BLIP scene module loaded.")
    except Exception as e:
        print(f"[Main] Scene module failed: {e}")


# ── Thread-safe shared results ───────────────────────────
_lock    = threading.Lock()
_frame   = [None]
_results = {
    "detections":    [],
    "depth_warning": None,
    "ocr_text":      None,
    "face_text":     None,
    "emotion_text":  None,
    "gesture_text":  None,
    "scene_text":    None,
    "currency_text": None,
    "gps_text":      None,
}

def _set(k, v):
    with _lock:
        _results[k] = v

def _get():
    with _lock:
        return dict(_results)

def _set_frame(f):
    with _lock:
        _frame[0] = f

def _get_frame():
    with _lock:
        f = _frame[0]
        return f.copy() if f is not None else None


# ── ML Worker — ALL inference runs here ──────────────────
def ml_worker(detector, ocr, nav, face, emotion, gesture, currency, gps):
    """
    Background thread: runs all heavy ML.
    Never blocks the camera/display loop.
    Each module runs every N frames (configured in config.py).
    """
    fc = 0
    while True:
        frame = _get_frame()
        if frame is None:
            time.sleep(0.005)
            continue

        fc += 1
        try:
            if fc % config.DETECT_EVERY_N_FRAMES   == 0:
                _set("detections",    detector.detect(frame))

            if fc % config.DEPTH_EVERY_N_FRAMES    == 0:
                dm = nav.estimate(frame)
                _set("depth_warning", nav.describe(dm))

            if fc % config.OCR_EVERY_N_FRAMES      == 0:
                _set("ocr_text",      ocr.read(frame))

            if fc % config.FACE_EVERY_N_FRAMES     == 0:
                # Pass current detections so face module only runs when person visible
                if face:
                    _set("face_text",     face.recognize(
                        frame, detections=_get().get("detections")
                    ))
                else:
                    _set("face_text", None)

            if fc % config.EMOTION_EVERY_N_FRAMES  == 0:
                if emotion:
                    _set("emotion_text",  emotion.detect(frame))
                else:
                    _set("emotion_text", None)

            if fc % config.GESTURE_EVERY_N_FRAMES  == 0:
                _set("gesture_text",  gesture.detect(frame))

            if fc % config.CURRENCY_EVERY_N_FRAMES == 0:
                _set("currency_text", currency.detect(frame))

            if fc % config.GPS_EVERY_N_FRAMES      == 0:
                _set("gps_text",      gps.get_location_description())

        except Exception as e:
            print(f"[Worker] Error: {e}")

        if fc >= 100_000:
            fc = 0

        time.sleep(0.001)   # yield CPU briefly


# ── Main ─────────────────────────────────────────────────
def main():
    # Initialize audio mode manager for 10-10s mic/speaker alternation
    audio_manager = init_audio_manager(mic_duration=10, speaker_duration=10)
    print("[Main] AudioModeManager initialized: 10s mic / 10s speaker")
    
    # Speaker first — all modules need it
    speaker = Speaker()
    speaker.start()
    speaker.say("Starting blind assistant.")

    # Perception modules
    camera   = Camera()
    detector = ObjectDetector()
    ocr      = OCRReader()
    nav      = Navigator()
    
    # FaceRecognizer is optional (requires TensorFlow/DeepFace)
    face = None
    if _face_module:
        try:
            face = _face_module()
        except Exception as e:
            print(f"[Main] FaceRecognizer failed to initialize: {e}")
    
    emotion = None
    if _emotion_module:
        try:
            emotion = _emotion_module()
        except Exception as e:
            print(f"[Main] EmotionDetector failed to initialize: {e}")
    
    gesture  = GestureDetector()
    currency = CurrencyDetector()
    gps      = GPSNavigator()
    gps.start()

    # Voice listener with mic-muting during TTS (optional)
    voice = None
    if getattr(config, 'VOICE_LISTENER_ENABLED', True):
        voice = VoiceListener(speaker=speaker, audio_manager=audio_manager)
        
        # BUG FIX: Share audio device lock between speaker and listener
        # This prevents the microphone from blocking speaker output on Windows
        speaker.set_audio_lock(voice._audio_lock)
        
        voice.start()
        print("[Main] Voice listener ENABLED — microphone active (10-10s alternation)")
    else:
        print("[Main] Voice listener DISABLED — speaker only mode")

    # Groq AI assistant (vision + weather)
    ai = GroqAssistant(speaker=speaker)

    # Fusion engine — decides what to say and when
    fusion = FusionEngine(
        speaker        = speaker,
        scene_module   = _scene,
        face_module    = face,
        groq_assistant = ai,
    )

    nav_state = NavState()

    # Start camera and ML worker
    camera.start()
    threading.Thread(
        target   = ml_worker,
        args     = (detector, ocr, nav, face, emotion, gesture, currency, gps),
        daemon   = True,
        name     = "MLWorker",
    ).start()

    speaker.say("All systems ready. I will describe your surroundings.")
    print(f"\n[Main] Wake word: '{config.WAKE_WORD}'")
    print("[Main] Q=quit | SPACE=describe | D=debug | S=screenshot\n")

    show_debug = True
    fps_count  = 0
    fps_start  = time.time()
    fps_val    = 0.0

    # ── MAIN LOOP ────────────────────────────────────────
    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            time.sleep(0.005)
            continue

        _set_frame(frame)
        results = _get()

        # Update navigation state
        nav_state.update(results, results.get("depth_warning"))

        # Process voice commands (if voice listener is enabled)
        if voice:
            cmd, raw_text = voice.get_command()
            if raw_text:
                reward = fusion.process_feedback(raw_text)
                if reward == 0.0 and cmd is not None:
                    fusion.set_command(cmd)
            elif cmd is not None:
                fusion.set_command(cmd)

        # Fusion decides what to speak
        fusion.process(results, nav_state=nav_state, frame=frame)

        # FPS counter
        fps_count += 1
        if time.time() - fps_start >= 1.0:
            fps_val   = fps_count / (time.time() - fps_start)
            fps_count = 0
            fps_start = time.time()

        # Display
        display = frame.copy()
        if show_debug:
            _draw_debug(display, results, nav_state, fps_val)

        cv2.imshow("Blind Assistant v8  [Q | SPACE | D | S]", display)
        key = cv2.waitKey(1) & 0xFF

        if   key == ord("q"):  break
        elif key == ord(" "):  fusion.set_command("objects")
        elif key == ord("d"):  show_debug = not show_debug
        elif key == ord("s"):
            fn = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fn, frame)
            print(f"[Main] Screenshot saved: {fn}")

    # Shutdown
    speaker.say("Shutting down. Goodbye.")
    time.sleep(1.5)
    camera.stop()
    gps.stop()
    if voice:
        voice.stop()
    speaker.stop()
    cv2.destroyAllWindows()
    print("[Main] Done.")


# ── Debug overlay ─────────────────────────────────────────
def _draw_debug(frame, results, nav_state, fps_val: float):
    # Draw bounding boxes
    for d in results.get("detections", []) or []:
        x1, y1, x2, y2 = d["box"]
        col = (0, 0, 255) if d.get("is_danger") else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(
            frame,
            f"{d['label']} {d['confidence']:.0%}",
            (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1,
        )

    # Proximity bars (bottom)
    h, w = frame.shape[:2]
    bar_y = h - 28
    for i, zone in enumerate(("left", "center", "right")):
        prox = nav_state.proximity[zone]
        bx   = 8 + i * (w // 3)
        bw   = w // 3 - 6
        r    = int(255 * prox)
        g    = int(255 * (1 - prox))
        cv2.rectangle(frame, (bx, bar_y), (bx + bw, bar_y + 8), (40, 40, 40), -1)
        cv2.rectangle(frame, (bx, bar_y), (bx + int(bw * prox), bar_y + 8), (r, g, 0), -1)
        cv2.putText(frame, zone, (bx + 2, bar_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)

    # Status text overlay
    y = 16
    def put(text: str, col=(210, 210, 210)):
        nonlocal y
        cv2.putText(frame, text[:66], (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)
        y += 14

    for key, col, lbl in [
        ("depth_warning", (60,  100, 255), "DEPTH"),
        ("ocr_text",      (0,   210, 255), "OCR"),
        ("face_text",     (255, 100, 255), "FACE"),
        ("emotion_text",  (255, 200,  50), "EMOTION"),
        ("gesture_text",  (0,   255, 200), "GESTURE"),
        ("currency_text", (50,  230, 100), "CURRENCY"),
        ("gps_text",      (200, 200, 255), "GPS"),
    ]:
        v = results.get(key)
        if v:
            put(f"{lbl}: {v}", col)

    if nav_state.any_danger_active or nav_state.proximity["center"] > 0.4:
        put(f"SAFE DIR: {nav_state.safest_direction.upper()}", (0, 255, 255))

    cv2.putText(
        frame,
        f"FPS:{fps_val:.0f} | v8 | Groq:{'ON' if getattr(config,'GROQ_API_KEY','').startswith('gsk') else 'OFF'}",
        (8, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 0), 1,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Main] Stopped by user.")
        sys.exit(0)