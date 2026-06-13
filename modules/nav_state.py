# modules/nav_state.py — Real-time navigation state tracker
#
# Maintains a persistent spatial picture of the environment across frames:
#   - Obstacle proximity per zone (left / center / right)
#   - Danger memory with decay
#   - Safe corridor detection
#   - Movement hints (which way is clearest)
#
# Used to build the RL state vector and drive navigation decisions.

import time
import numpy as np
import config


class NavState:
    """
    Aggregates perception results into a coherent navigation picture.
    Updated every frame by main.py, read by rl_env.py and fusion_engine.py.
    """

    def __init__(self):
        # Obstacle proximity per zone: 0.0 = clear, 1.0 = very close
        self.proximity = {"left": 0.0, "center": 0.0, "right": 0.0}

        # Danger memory: zone → expiry timestamp
        self._danger_memory: dict[str, float] = {}

        # Object positions this frame: list of {"label", "position", "is_danger"}
        self.detections: list[dict] = []

        # Depth map values per zone (raw metres from navigator)
        self.depth_m = {"left": 9.9, "center": 9.9, "right": 9.9}

        # Flags
        self.has_person    = False
        self.has_vehicle   = False
        self.has_gesture   = False
        self.has_face      = False
        self.has_ocr       = False
        self.has_currency  = False
        self.has_emotion   = False
        self.has_scene     = False

        # Time trackers
        self.last_danger_time   = 0.0
        self.last_speech_time   = 0.0
        self.steps_since_danger = 0
        self._reward_momentum   = 0.0   # exponential moving avg of recent rewards

        # Cumulative step counter
        self.step = 0

    # ── Update from perception results ───────────────────

    def update(self, results: dict, depth_warning: str | None = None):
        """Call every frame with the latest results dict from ml_worker."""
        now = time.time()
        self.step += 1

        dets = results.get("detections", [])
        self.detections = dets

        # Flags from detections
        self.has_person  = any(d["label"] == "person"  for d in dets)
        self.has_vehicle = any(d["label"] in ("car", "truck", "bus", "motorcycle", "bicycle")
                               for d in dets)

        # Danger memory: zones where danger was seen recently
        for d in dets:
            if d["is_danger"]:
                pos = d["position"].replace("on your ", "").replace(" ", "_")
                # Normalize position string to left/center/right
                if "left"   in pos: zone = "left"
                elif "right" in pos: zone = "right"
                else:                zone = "center"
                self._danger_memory[zone] = now + config.NAV_SPATIAL_MEMORY_SEC
                self.last_danger_time = now

        # Expire old danger memories
        self._danger_memory = {
            z: t for z, t in self._danger_memory.items() if t > now
        }

        # Proximity per zone from depth_warning text
        if depth_warning:
            for zone in ("left", "center", "right"):
                if zone in depth_warning:
                    # Extract distance if possible
                    for word in depth_warning.split():
                        try:
                            m = float(word)
                            self.depth_m[zone] = m
                            # Normalize: 0m=1.0, OBSTACLE_CLEAR_M=0.0
                            self.proximity[zone] = max(0.0, min(1.0,
                                1.0 - m / config.OBSTACLE_CLEAR_M
                            ))
                        except ValueError:
                            pass
        else:
            # No warning = decay proximity toward 0 slowly
            for zone in self.proximity:
                self.proximity[zone] = max(0.0, self.proximity[zone] - 0.05)

        # Object density per zone
        self._zone_counts = {"left": 0, "center": 0, "right": 0}
        for d in dets:
            pos = d["position"]
            if "left"   in pos: self._zone_counts["left"]   += 1
            elif "right" in pos: self._zone_counts["right"]  += 1
            else:                self._zone_counts["center"] += 1

        # Modality flags
        self.has_gesture  = results.get("gesture_text")  is not None
        self.has_face     = results.get("face_text")      is not None
        self.has_ocr      = results.get("ocr_text")       is not None
        self.has_currency = results.get("currency_text")  is not None
        self.has_emotion  = results.get("emotion_text")   is not None
        self.has_scene    = results.get("scene_text")     is not None

        # Steps since last danger
        if self.last_danger_time > 0 and now - self.last_danger_time < 0.1:
            self.steps_since_danger = 0
        else:
            self.steps_since_danger = min(self.steps_since_danger + 1, 300)

    def on_speech(self):
        """Call when the speaker says something."""
        self.last_speech_time = time.time()

    def add_reward_signal(self, reward: float):
        """Exponential moving average of recent user feedback rewards."""
        self._reward_momentum = 0.9 * self._reward_momentum + 0.1 * reward

    # ── Navigation helpers ────────────────────────────────

    @property
    def safest_direction(self) -> str:
        """Returns 'left', 'center', or 'right' — the least obstructed path."""
        return min(self.proximity, key=lambda z: self.proximity[z])

    @property
    def is_path_clear(self) -> bool:
        return self.proximity["center"] < 0.3

    @property
    def any_danger_active(self) -> bool:
        return bool(self._danger_memory) or any(
            d["is_danger"] for d in self.detections
        )

    @property
    def zone_counts(self) -> dict:
        return getattr(self, "_zone_counts", {"left": 0, "center": 0, "right": 0})

    @property
    def time_since_speech(self) -> float:
        if self.last_speech_time == 0:
            return 999.0
        return time.time() - self.last_speech_time

    @property
    def reward_momentum(self) -> float:
        return self._reward_momentum

    # ── Encode to RL state vector ─────────────────────────

    def encode(self) -> np.ndarray:
        """
        Encode current nav state into a 32-dim float32 vector for the RL agent.
        All values normalized to [0, 1].
        """
        obs = np.zeros(config.RL_STATE_DIM, dtype=np.float32)

        # [0–2] Obstacle proximity per zone
        obs[0] = self.proximity["left"]
        obs[1] = self.proximity["center"]
        obs[2] = self.proximity["right"]

        # [3–5] Danger flags (current + memory)
        mem = self._danger_memory
        obs[3] = 1.0 if "left"   in mem or any(
            d["is_danger"] and "left"  in d["position"] for d in self.detections) else 0.0
        obs[4] = 1.0 if "center" in mem or any(
            d["is_danger"] and "ahead" in d["position"] for d in self.detections) else 0.0
        obs[5] = 1.0 if "right"  in mem or any(
            d["is_danger"] and "right" in d["position"] for d in self.detections) else 0.0

        # [6–8] Object density per zone (normalized to max 5)
        zc = self.zone_counts
        obs[6] = min(zc["left"],   5) / 5.0
        obs[7] = min(zc["center"], 5) / 5.0
        obs[8] = min(zc["right"],  5) / 5.0

        # [9] Has person in frame
        obs[9]  = float(self.has_person)
        # [10] Has vehicle in frame
        obs[10] = float(self.has_vehicle)
        # [11] Gesture detected
        obs[11] = float(self.has_gesture)
        # [12] Face recognized
        obs[12] = float(self.has_face)
        # [13] OCR text present
        obs[13] = float(self.has_ocr)
        # [14] Currency present
        obs[14] = float(self.has_currency)
        # [15] Emotion detected
        obs[15] = float(self.has_emotion)

        # [16] Time since last speech (normalized: 0=just spoke, 1=10s+ ago)
        obs[16] = min(self.time_since_speech / 10.0, 1.0)

        # [17] Steps since danger (normalized: 0=just saw danger, 1=50+ steps ago)
        obs[17] = min(self.steps_since_danger / 50.0, 1.0)

        # [18] Reward momentum (normalized: -1..+1 → 0..1)
        obs[18] = (self._reward_momentum + 1.0) / 2.0

        # [19] Path clear flag
        obs[19] = float(self.is_path_clear)

        # [20] Safest direction: 0=left, 0.5=center, 1=right
        obs[20] = {"left": 0.0, "center": 0.5, "right": 1.0}[self.safest_direction]

        # [21] Normalized step count (progress through session)
        obs[21] = min(self.step / 10000.0, 1.0)

        # [22–31] Reserved for action history (filled by rl_env)
        # (left as zeros here — rl_env fills positions 22–31)

        return obs