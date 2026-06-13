# ============================================================
#  config.py — Blind Assistant v8 FIXED
# ============================================================

LOW_RAM_MODE = True     # True = faster (disables BLIP/LLaVA)

DEVICE  = "cpu"
USE_GPU = False

# ── Camera ───────────────────────────────────────────────
CAMERA_INDEX   = 0
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
FPS_TARGET     = 30
PROCESS_WIDTH  = 320
PROCESS_HEIGHT = 240

# ── Frame skipping ───────────────────────────────────────
DETECT_EVERY_N_FRAMES   = 3
DEPTH_EVERY_N_FRAMES    = 10
OCR_EVERY_N_FRAMES      = 60
FACE_EVERY_N_FRAMES     = 30
EMOTION_EVERY_N_FRAMES  = 30
GESTURE_EVERY_N_FRAMES  = 5
CURRENCY_EVERY_N_FRAMES = 20
SCENE_EVERY_N_FRAMES    = 9999
GPS_EVERY_N_FRAMES      = 300

# ── YOLO ─────────────────────────────────────────────────
YOLO_MODEL       = "yolov8n.pt"
YOLO_CONFIDENCE  = 0.50
YOLO_DANGER_ZONE = 0.28
YOLO_IMG_SIZE    = 256
DANGER_CLASSES   = [
    "person", "car", "truck", "bus", "bicycle",
    "motorcycle", "dog", "stairs", "stop sign",
]

# ── Currency ─────────────────────────────────────────────
CURRENCY_MODEL_PATH   = "models/currency_yolo.pt"
CURRENCY_CONFIDENCE   = 0.55
CURRENCY_FALLBACK_HSV = True

# ── Navigation / Depth ───────────────────────────────────
OBSTACLE_NEAR_M          = 1.0
OBSTACLE_WARN_M          = 2.5
OBSTACLE_CLEAR_M         = 4.0
NAV_SPATIAL_MEMORY_SEC   = 5.0
NAV_SAFE_CORRIDOR_WIDTH  = 0.40
DEPTH_MODEL              = "midas"
DEPTH_ENCODER            = "vits"
DEPTH_CONSECUTIVE_FRAMES = 2

# ── OCR ──────────────────────────────────────────────────
OCR_LANGUAGES      = ["en"]
OCR_MIN_CONFIDENCE = 0.65
OCR_GPU            = False

# ── Face ─────────────────────────────────────────────────
KNOWN_FACES_DIR         = "data/known_faces"
FACE_MODEL              = "Facenet"
FACE_DISTANCE_THRESHOLD = 0.55
FACE_COOLDOWN_SEC       = 15

# ── Emotion ──────────────────────────────────────────────
EMOTION_MIN_CONFIDENCE = 0.55
EMOTIONS_TO_ANNOUNCE   = ["happy", "angry", "surprised"]

# ── Gesture ──────────────────────────────────────────────
GESTURE_CONFIDENCE   = 0.80
GESTURE_HOLD_FRAMES  = 7
GESTURES_TO_ANNOUNCE = [
    "handshake", "wave", "stop",
    "pointing_left", "pointing_right", "thumbs_up",
]

# ── LLaVA / Scene ────────────────────────────────────────
LLAVA_MODEL          = "llava"
LLAVA_ENDPOINT       = "http://localhost:11434/api/generate"
LLAVA_TIMEOUT        = 20
LLAVA_PROMPT         = (
    "You are helping a blind person. In ONE concise sentence, "
    "describe what you see: obstacles, people, text, or hazards."
)
SCENE_MODEL          = "Salesforce/blip-image-captioning-base"
SCENE_FALLBACK_MODEL = "Salesforce/blip-image-captioning-base"
SCENE_MAX_TOKENS     = 40

# ── GPS ──────────────────────────────────────────────────
GPS_ENABLED         = False
GPS_PORT            = "COM3"
GPS_BAUDRATE        = 9600
MAPS_ANNOUNCE_EVERY = 10

# ── Voice / Whisper ──────────────────────────────────────
VOICE_LISTENER_ENABLED = True    # FIX: Set to True to enable microphone/voice commands. False = speaker only
WHISPER_MODEL    = "tiny"
WAKE_WORD        = "assistant"
LISTEN_TIMEOUT   = 5
MIC_DEVICE_INDEX = None

# ── TTS ──────────────────────────────────────────────────
TTS_RATE        = 165
TTS_VOLUME      = 1.0
TTS_VOICE_INDEX = 0
TTS_QUEUE_MAX   = 5     # FIX: was 3, raised to 5 — less aggressive dropping

# ── ANNOUNCE COOLDOWNS (seconds) ─────────────────────────
COOLDOWN_DANGER   = 5.0
COOLDOWN_DEPTH    = 8.0
COOLDOWN_OBJECTS  = 6.0
COOLDOWN_FACE     = 15.0
COOLDOWN_EMOTION  = 10.0
COOLDOWN_GESTURE  = 6.0
COOLDOWN_OCR      = 8.0
COOLDOWN_CURRENCY = 6.0
COOLDOWN_SCENE    = 0.0
COOLDOWN_GPS      = 15.0

# ── Priority (higher = spoken first) ────────────────────
PRIORITY = {
    "danger":   10,
    "depth":     9,
    "command":   8,
    "gesture":   7,
    "face":      5,
    "emotion":   5,
    "ocr":       4,
    "scene":     4,
    "currency":  4,
    "objects":   3,
    "gps":       2,
}

# ── Feedback words ────────────────────────────────────────
FEEDBACK_POSITIVE = ["good", "yes", "correct", "helpful", "right", "nice"]
FEEDBACK_REPEAT   = ["repeat", "again", "say again"]
FEEDBACK_NEGATIVE = ["wrong", "quiet", "incorrect", "annoying", "too much"]

# ╔══════════════════════════════════════════════════════════╗
# ║  GROQ API KEY — PASTE YOUR KEY HERE                      ║
# ║  Get free key: https://console.groq.com                  ║
# ╚══════════════════════════════════════════════════════════╝
# GROQ_API_KEY = " "   # ← paste your key

import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ── Location (for weather command) ───────────────────────
LOCATION_LAT  = 31.2554
LOCATION_LON  = 75.7049
LOCATION_NAME = "Lovely Professional University"

# ── Other settings ────────────────────────────────────────
PATH_PLAN_INTERVAL       = 3.0
TRAFFIC_LIGHT_COOLDOWN   = 5.0
OBJECT_ANNOUNCE_INTERVAL = 6.0
RL_STATE_DIM             = 32