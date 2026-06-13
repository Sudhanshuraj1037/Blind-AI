# ============================================================
#  modules/navigator.py — Blind Assistant v7
#
#  KEY FIXES vs v6:
#  1. ROUNDED DISTANCES — no exact floats in output strings.
#     "obstacle 1.13 metres" changes every frame → bypasses TTS
#     deduplication → spam. Now outputs bucketed phrases:
#       < 1.0 m  → "very close"
#       1–2 m    → "about 1 metre"
#       2–3 m    → "about 2 metres"
#     Identical phrasing every frame → TTS dedup works correctly.
#  2. All-3-zones MiDaS false-positive filter (indoor rooms)
#  3. Consecutive-frame requirement before announcing
# ============================================================

import cv2
import torch
import numpy as np
import config


class Navigator:

    def __init__(self):
        self._prev_depths  = {}
        self._model_type   = None
        self._warn_counter = {"left": 0, "center": 0, "right": 0}

        if config.DEPTH_MODEL == "depth_anything_v2":
            self._load_depth_anything()
        else:
            self._load_midas()

    # ── Model loading ─────────────────────────────────────

    def _load_depth_anything(self):
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
            cfgs = {
                "vits": {"encoder": "vits", "features": 64,
                         "out_channels": [48, 96, 192, 384]},
                "vitb": {"encoder": "vitb", "features": 128,
                         "out_channels": [96, 192, 384, 768]},
            }
            enc          = config.DEPTH_ENCODER
            self._model  = DepthAnythingV2(**cfgs[enc])
            self._model.load_pretrained_weights(
                f"models/depth_anything_v2_{enc}.pth"
            )
            self._model      = self._model.to(config.DEVICE).eval()
            self._model_type = "depth_anything_v2"
            print(f"[Navigator] Depth Anything V2 ({enc}) ready.")
        except Exception as e:
            print(f"[Navigator] Depth Anything V2 unavailable ({e}). Using MiDaS.")
            self._load_midas()

    def _load_midas(self):
        print("[Navigator] Loading MiDaS small…")
        self._model = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
        )
        self._model.eval().to(config.DEVICE)
        tf = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self._transform  = tf.small_transform
        self._model_type = "midas"
        print(f"[Navigator] MiDaS ready on {config.DEVICE}.")

    # ── Public API ────────────────────────────────────────

    def estimate(self, frame):
        if self._model_type == "depth_anything_v2":
            return self._estimate_dav2(frame)
        return self._estimate_midas(frame)

    def describe(self, depth_map) -> str | None:
        if self._model_type == "depth_anything_v2":
            return self._describe_metric(depth_map)
        return self._describe_relative(depth_map)

    # ── Depth Anything V2 (metric, metres) ───────────────

    def _estimate_dav2(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            depth = self._model.infer_image(rgb)
        return depth

    def _describe_metric(self, depth_map) -> str | None:
        h, w = depth_map.shape
        zones = {
            "left":   depth_map[:, :w//3],
            "center": depth_map[:, w//3:2*w//3],
            "right":  depth_map[:, 2*w//3:],
        }
        warnings = []
        info     = []
        for name, zone in zones.items():
            dist_m   = float(np.percentile(zone, 5))
            prev     = self._prev_depths.get(name)
            if prev is not None and prev - dist_m > 0.3:
                warnings.append(f"object approaching fast from {name}")
            self._prev_depths[name] = dist_m

            if dist_m < config.OBSTACLE_NEAR_M:
                warnings.append(f"obstacle {self._bucket(dist_m)} on your {name}")
            elif dist_m < config.OBSTACLE_WARN_M:
                info.append(f"object {self._bucket(dist_m)} on your {name}")

        if warnings:
            return "Warning! " + ", ".join(warnings) + ". Please slow down."
        if info:
            return "Heads up: " + ", ".join(info) + "."
        return None

    # ── MiDaS relative depth ──────────────────────────────

    def _estimate_midas(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = self._transform(rgb).to(config.DEVICE)
        with torch.no_grad():
            pred = self._model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        d  = pred.cpu().numpy()
        mn, mx = d.min(), d.max()
        return (d - mn) / (mx - mn) if mx > mn else d

    def _describe_relative(self, depth_map) -> str | None:
        """
        FIX 1: All-3-zones active simultaneously = MiDaS indoor false positive.
                Skip that case; only warn for 1–2 localised zones.
        FIX 2: Require DEPTH_CONSECUTIVE_FRAMES consecutive frames before speaking.
        FIX 3: Threshold raised 0.75 → 0.85 for extra indoor margin.
        """
        THRESHOLD     = 0.85
        REQUIRED_CONS = getattr(config, "DEPTH_CONSECUTIVE_FRAMES", 3)

        h, w = depth_map.shape
        zones = {
            "left":   depth_map[:, :w//3],
            "center": depth_map[:, w//3:2*w//3],
            "right":  depth_map[:, 2*w//3:],
        }

        triggered = {z for z, v in zones.items() if v.max() > THRESHOLD}

        # Update consecutive counters
        for z in ("left", "center", "right"):
            if z in triggered:
                self._warn_counter[z] = min(self._warn_counter[z] + 1, 10)
            else:
                self._warn_counter[z] = max(self._warn_counter[z] - 1, 0)

        confirmed = {
            z for z in triggered
            if self._warn_counter[z] >= REQUIRED_CONS
        }

        # All 3 confirmed simultaneously = indoor false positive, skip
        if len(confirmed) == 3:
            return None

        if not confirmed:
            return None

        # 1 or 2 zones = real localised obstacle
        zones_str = " and ".join(sorted(confirmed))
        return f"Obstacle very close on your {zones_str}. Please slow down."

    # ── Helper: bucket distance into stable spoken phrase ─

    @staticmethod
    def _bucket(metres: float) -> str:
        """
        Convert a float distance to a stable, rounded phrase.

        Without this, "obstacle 1.13 metres" on frame N and
        "obstacle 1.17 metres" on frame N+1 are different strings,
        so TTS deduplication never fires → every frame gets spoken.

        With bucketing, both become "about 1 metre" → dedup works.
        """
        if metres < 0.8:
            return "very close"
        elif metres < 1.5:
            return "about 1 metre away"
        elif metres < 2.5:
            return "about 2 metres away"
        elif metres < 3.5:
            return "about 3 metres away"
        else:
            return "several metres away"