# modules/face_recognizer.py — v8 FIXED
#
# FIX: Removed the print() that fired on EVERY frame check.
# The old code printed "Sudhanshu: dist=0.978 verified=False"
# 10+ times per second, flooding terminal and wasting CPU.
# Now it ONLY prints when there is an actual match.

import os
import time
import tempfile
import cv2
from deepface import DeepFace
import config


class FaceRecognizer:

    def __init__(self):
        self.known_faces_dir = config.KNOWN_FACES_DIR
        self.model_name      = "Facenet"
        self._last_seen      = {}
        print(f"[FaceRecognizer] Ready. Model: Facenet | Folder: {self.known_faces_dir}")
        self._check_faces_dir()

    def _check_faces_dir(self):
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
        faces = [
            f for f in os.listdir(self.known_faces_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if faces:
            print(f"[FaceRecognizer] Known faces: {faces}")
        else:
            print("[FaceRecognizer] No known faces. Add photos to data/known_faces/")

    def recognize(self, frame, detections=None) -> str | None:
        """Background worker — only runs when YOLO sees a person."""
        if detections is not None:
            if not any(d.get("label") == "person" for d in detections):
                return None
        return self._run_verify(frame, bypass_cooldown=False)

    def force_recognize(self, frame, detections=None) -> str | None:
        """On-demand call from 'who is that' command."""
        if detections is not None:
            if not any(d.get("label") == "person" for d in detections):
                return "I don't see any person in front of me right now."
        return self._run_verify(frame, bypass_cooldown=True)

    def _run_verify(self, frame, bypass_cooldown: bool) -> str | None:
        known_images = [
            f for f in os.listdir(self.known_faces_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not known_images:
            return None

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False, prefix="face_"
            ) as f:
                tmp_path = f.name

            cv2.imwrite(tmp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            now   = time.time()
            found = []

            for img_file in known_images:
                img_path = os.path.join(self.known_faces_dir, img_file)
                try:
                    result = DeepFace.verify(
                        img1_path         = tmp_path,
                        img2_path         = img_path,
                        model_name        = self.model_name,
                        enforce_detection = False,
                        distance_metric   = "cosine",
                    )
                    dist     = result.get("distance", 999)
                    name_raw = os.path.splitext(img_file)[0].replace("_", " ").title()

                    # ── FIX: only print on actual match ──────────────
                    if dist < config.FACE_DISTANCE_THRESHOLD:
                        print(f"[FaceRecognizer] MATCH: {name_raw} (dist={dist:.3f})")
                        if not bypass_cooldown:
                            last = self._last_seen.get(name_raw, 0)
                            if now - last < config.FACE_COOLDOWN_SEC:
                                continue
                        self._last_seen[name_raw] = now
                        found.append(name_raw)
                    # Non-matches: NO print (was causing spam)

                except Exception as e:
                    print(f"[FaceRecognizer] Error on {img_file}: {e}")
                    continue

            if found:
                names = " and ".join(found)
                return f"I recognise {names} in front of you."

            if bypass_cooldown:
                return "I can see a person but I don't recognise them."

            return None

        except Exception as e:
            print(f"[FaceRecognizer] Error: {e}")
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass