# modules/traffic_light.py — Traffic light color detection
#
# YOLO detects the "traffic light" bounding box.
# This module crops that region and classifies:
#   RED → "Red traffic light ahead. Please stop."
#   GREEN → "Green traffic light. Safe to cross."
#   YELLOW → "Yellow traffic light. Prepare to stop."

import cv2
import numpy as np
import time
import config

_MESSAGES = {
    "red":     "Red traffic light ahead. Please stop and wait.",
    "green":   "Green traffic light ahead. Safe to cross. Proceed carefully.",
    "yellow":  "Yellow traffic light. Prepare to stop.",
    "unknown": "Traffic light detected. Proceed with caution.",
}


class TrafficLightDetector:
    def __init__(self):
        self._last_color    = None
        self._last_time     = 0.0
        self._COOLDOWN      = getattr(config, "TRAFFIC_LIGHT_COOLDOWN", 5.0)
        print("[TrafficLight] Ready.")

    def detect(self, frame, detections) -> str | None:
        """Detect traffic light color from YOLO bounding boxes."""
        if not detections:
            return None

        tl_dets = [
            d for d in detections
            if d.get("label", "").lower() in ("traffic light", "traffic_light")
        ]
        if not tl_dets:
            return None

        now = time.time()
        # Largest bounding box = most prominent light
        tl_dets.sort(key=lambda d: _area(d["box"]), reverse=True)
        best  = tl_dets[0]
        color = self._classify(frame, best["box"])
        msg   = _MESSAGES.get(color, _MESSAGES["unknown"])

        if color == self._last_color and now - self._last_time < self._COOLDOWN:
            return None

        self._last_color = color
        self._last_time  = now
        position = best.get("position", "ahead")
        print(f"[TrafficLight] {color.upper()} ({position})")
        return msg.replace("ahead", position)

    def _classify(self, frame, box) -> str:
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        pad  = 4
        x1   = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2   = min(w, x2 + pad); y2 = min(h, y2 + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        ch  = hsv.shape[0]

        # Split into top/middle/bottom thirds (red/yellow/green bulb positions)
        regions = {
            "red":    hsv[:ch//3,     :, :],
            "yellow": hsv[ch//3:2*ch//3, :, :],
            "green":  hsv[2*ch//3:,   :, :],
        }

        # HSV ranges
        ranges = {
            "red":    [((0,100,100),(10,255,255)), ((160,100,100),(180,255,255))],
            "yellow": [((15,100,100),(40,255,255))],
            "green":  [((40,80,80),(90,255,255))],
        }

        scores = {}
        for color_name, color_ranges in ranges.items():
            region = regions[color_name]
            total  = region.shape[0] * region.shape[1]
            if total == 0:
                scores[color_name] = 0.0
                continue
            count = 0
            for lo, hi in color_ranges:
                mask  = cv2.inRange(region, np.array(lo), np.array(hi))
                count += cv2.countNonZero(mask)
            scores[color_name] = count / total

        best_color = max(scores, key=scores.get)
        if scores[best_color] < 0.04:
            return "unknown"
        return best_color


def _area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)