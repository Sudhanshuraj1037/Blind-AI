# modules/path_planner.py — Directional navigation
#
# Analyses depth map + YOLO detections to generate:
#   "move left", "move right", "go straight", "stop"
#
# Called by fusion_engine on "navigate" voice command.

import numpy as np
import config


class PathPlanner:
    """
    Converts depth map + detections into spoken navigation directions.

    Usage:
        planner = PathPlanner()
        direction, message = planner.plan(depth_map, detections)
        speaker.say(message)
    """

    def __init__(self):
        self._mode = getattr(config, "DEPTH_MODEL", "midas")
        print("[PathPlanner] Ready.")

    def plan(self, depth_map, detections=None):
        """
        Returns (direction_str, spoken_message).
        direction_str: "left" | "right" | "straight" | "stop"
        """
        if depth_map is None:
            return "unknown", "Cannot navigate — depth sensor unavailable."

        h, w = depth_map.shape
        left_zone   = depth_map[:, :w//3]
        center_zone = depth_map[:, w//3 : 2*w//3]
        right_zone  = depth_map[:, 2*w//3:]

        left_clear   = self._clearance(left_zone)
        center_clear = self._clearance(center_zone)
        right_clear  = self._clearance(right_zone)

        # Apply YOLO danger penalty
        if detections:
            left_clear, center_clear, right_clear = self._yolo_penalty(
                left_clear, center_clear, right_clear, detections, w
            )

        direction, message = self._decide(left_clear, center_clear, right_clear)
        print(f"[PathPlanner] L={left_clear:.2f} C={center_clear:.2f} R={right_clear:.2f} → {direction}")
        return direction, message

    def _clearance(self, zone) -> float:
        """Returns clearance 0–1. Higher = safer/clearer."""
        if zone.size == 0:
            return 1.0
        if self._mode == "depth_anything_v2":
            closest = float(np.percentile(zone, 5))
            return min(closest / 5.0, 1.0)
        else:
            max_depth = float(np.percentile(zone, 95))
            return 1.0 - max_depth

    def _yolo_penalty(self, lc, cc, rc, detections, frame_width):
        for d in detections:
            if not d.get("is_danger"):
                continue
            x1, _, x2, _ = d["box"]
            center_x = (x1 + x2) / 2
            penalty  = 0.6
            if center_x < frame_width * 0.33:
                lc = max(0.0, lc - penalty)
            elif center_x > frame_width * 0.66:
                rc = max(0.0, rc - penalty)
            else:
                cc = max(0.0, cc - penalty)
        return lc, cc, rc

    def _decide(self, lc, cc, rc):
        threshold = 0.45
        cs = cc >= threshold
        ls = lc >= threshold
        rs = rc >= threshold

        if not cs and not ls and not rs:
            return "stop", "Stop! Obstacles on all sides. Please wait for assistance."

        if cs and cc >= max(lc, rc) - 0.1:
            return "straight", f"Go straight. {self._dist_desc(cc)}."

        if lc >= rc and ls:
            return "left", f"Move left. {self._steps(lc)} Then check ahead."

        if rs:
            return "right", f"Move right. {self._steps(rc)} Then check ahead."

        if lc > rc:
            return "left", "Cautiously move left. Obstacle ahead."
        return "right", "Cautiously move right. Obstacle ahead."

    def _dist_desc(self, clearance: float) -> str:
        if clearance > 0.7: return "Path is clear"
        if clearance > 0.5: return "Path is mostly clear"
        return "Proceed slowly"

    def _steps(self, clearance: float) -> str:
        if clearance > 0.7: return "Take 3 to 4 steps."
        if clearance > 0.5: return "Take 2 steps."
        return "Take 1 careful step."