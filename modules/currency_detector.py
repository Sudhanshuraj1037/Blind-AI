# modules/currency_detector.py — YOLO-based Indian banknote detector (v5)
#
# Primary:  custom YOLOv8n trained in Google Colab on your photos
# Fallback: HSV color heuristic (if .pt model not yet trained)
#
# To get the YOLO model:
#   1. Run collect_currency_photos.py  (get 80+ photos per denomination)
#   2. Label on Roboflow → export YOLOv8 format
#   3. Train in Colab (Cell 8 of Blind_Assistant_Training.ipynb)
#   4. Download currency_yolo.pt → put in models/

import os
import cv2
import numpy as np
import config

# ── HSV fallback profiles ─────────────────────────────────
# Used ONLY when models/currency_yolo.pt is not present.
# Closest-center matching fixes the ₹10/₹20 overlap bug.
_HSV_PROFILES = {
    "10 rupee note":   {"range": (15, 28),  "center": 21},
    "20 rupee note":   {"range": (29, 45),  "center": 37},
    "50 rupee note":   {"range": (95, 130), "center": 112},
    "100 rupee note":  {"range": (120,145), "center": 132},
    "200 rupee note":  {"range": (22, 35),  "center": 28},
    "500 rupee note":  {"range": (55, 85),  "center": 70},
    "2000 rupee note": {"range": (140,165), "center": 152},
}
_MIN_CONTOUR_PX = 3000


class CurrencyDetector:
    def __init__(self):
        self._model      = None
        self._use_yolo   = False
        self._model_path = config.CURRENCY_MODEL_PATH

        if os.path.exists(self._model_path):
            self._load_yolo()
        else:
            print(
                f"[CurrencyDetector] {self._model_path} not found. "
                "Using HSV fallback. Train in Colab to get YOLO model."
            )

    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            self._model    = YOLO(self._model_path)
            self._use_yolo = True
            print(f"[CurrencyDetector] YOLO model loaded: {self._model_path}")
        except Exception as e:
            print(f"[CurrencyDetector] YOLO load failed ({e}), using HSV fallback.")
            self._use_yolo = False

    # ── Public API ────────────────────────────────────────

    def detect(self, frame) -> str | None:
        """Returns spoken description or None."""
        if self._use_yolo:
            return self._detect_yolo(frame)
        return self._detect_hsv(frame)

    # ── YOLO detection ────────────────────────────────────

    def _detect_yolo(self, frame) -> str | None:
        try:
            results = self._model(
                frame,
                imgsz=256,
                verbose=False,
                device=config.DEVICE,
            )[0]

            found = []
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < config.CURRENCY_CONFIDENCE:
                    continue
                cls_id = int(box.cls[0])
                label  = self._model.names[cls_id]   # e.g. "500"
                # Normalize label to spoken form
                spoken = self._normalize_label(label)
                if spoken and spoken not in found:
                    found.append(spoken)

            if not found:
                return None

            return "I can see currency: " + " and ".join(found) + "."

        except Exception as e:
            print(f"[CurrencyDetector] YOLO error: {e}")
            return None

    @staticmethod
    def _normalize_label(label: str) -> str:
        """Convert model class name to spoken text.
        Handles formats like '500', 'rs500', '500_rupee', etc.
        """
        digits = "".join(c for c in label if c.isdigit())
        if not digits:
            return label
        return f"{digits} rupee note"

    # ── HSV fallback ──────────────────────────────────────

    def _detect_hsv(self, frame) -> str | None:
        try:
            hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges   = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            found = []
            for cnt in contours:
                if cv2.contourArea(cnt) < _MIN_CONTOUR_PX:
                    continue

                peri   = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) != 4:
                    continue

                x, y, rw, rh = cv2.boundingRect(approx)
                aspect = max(rw, rh) / max(min(rw, rh), 1)
                if not (1.5 < aspect < 3.5):
                    continue

                roi        = hsv[y:y+rh, x:x+rw]
                median_hue = int(np.median(roi[:, :, 0]))
                match      = self._match_hue(median_hue)
                if match and match not in found:
                    found.append(match)

            if not found:
                return None

            return "I can see currency: " + " and ".join(found) + "."

        except Exception as e:
            print(f"[CurrencyDetector] HSV error: {e}")
            return None

    @staticmethod
    def _match_hue(hue: int) -> str | None:
        """Closest-center matching — eliminates the ₹10/₹20 overlap bug."""
        best_name = None
        best_dist = float("inf")
        for name, profile in _HSV_PROFILES.items():
            lo, hi = profile["range"]
            if lo <= hue <= hi:
                dist = abs(hue - profile["center"])
                if dist < best_dist:
                    best_dist = dist
                    best_name = name
        return best_name