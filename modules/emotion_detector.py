# modules/emotion_detector.py — detects emotions on faces using DeepFace  (corrected)
#
# FIXES applied:
#   1. Exception handler now prints a debug message (configurable via DEBUG_EMOTION)
#      instead of silently swallowing all errors — makes it easier to diagnose issues
#      while still not crashing the main loop when no face is present

import cv2
from deepface import DeepFace
import config

# Set True to see emotion detection errors in the console (useful during setup)
DEBUG_EMOTION = False

class EmotionDetector:
    def __init__(self):
        print("[EmotionDetector] Ready. Uses DeepFace emotion analysis.")

    def detect(self, frame):
        """
        Analyse facial emotions in the frame.
        Returns a spoken description string or None.
        """
        try:
            results = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )

            # results can be a list (multiple faces) or a single dict
            if isinstance(results, dict):
                results = [results]

            descriptions = []
            for i, face in enumerate(results):
                emotion    = face.get("dominant_emotion", "").lower()
                confidence = face.get("emotion", {}).get(emotion, 0)

                if confidence < config.EMOTION_MIN_CONFIDENCE * 100:
                    continue

                if emotion not in config.EMOTIONS_TO_ANNOUNCE:
                    continue

                person = "The person" if len(results) == 1 else f"Person {i+1}"
                descriptions.append(f"{person} looks {emotion}")

            if not descriptions:
                return None

            return ". ".join(descriptions) + "."

        except Exception as e:
            # FIX: log errors when debugging, but still silently continue
            # (no-face-found is the most common case and should not spam console)
            if DEBUG_EMOTION:
                print(f"[EmotionDetector] Detection error (often normal — no face): {e}")
            return None