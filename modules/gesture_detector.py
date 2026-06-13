# modules/gesture_detector.py — MediaPipe hand gesture detection  (corrected)
#
# FIXES applied:
#   1. thumb_up heuristic now accounts for hand side — on a LEFT hand the
#      thumb extends to the RIGHT (higher x), opposite of right hand
#   2. "fist" description softened — removed alarming "Be cautious" message
#      since users often close their hand naturally without threatening intent

import cv2
import numpy as np
import config

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[GestureDetector] mediapipe not installed. Run: pip install mediapipe")


class GestureDetector:

    # Landmark indices (MediaPipe hand model)
    WRIST       = 0
    THUMB_TIP   = 4
    INDEX_TIP   = 8
    INDEX_MCP   = 5
    MIDDLE_TIP  = 12
    MIDDLE_MCP  = 9
    RING_TIP    = 16
    RING_MCP    = 13
    PINKY_TIP   = 20
    PINKY_MCP   = 17

    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            self._ready = False
            return

        self._mp_hands = mp.solutions.hands
        self._hands    = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self._ready          = True
        self._last_gesture   = None
        self._hold_count     = 0
        print("[GestureDetector] MediaPipe Hands ready.")

    def detect(self, frame):
        """
        Detect hand gestures in a frame.
        Returns spoken description string or None.
        Gesture must be held for GESTURE_HOLD_FRAMES before speaking.
        """
        if not self._ready:
            return None

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            self._hold_count   = 0
            self._last_gesture = None
            return None

        gestures_found = []

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            lm        = hand_landmarks.landmark
            hand_side = handedness.classification[0].label  # "Left" or "Right"
            gesture   = self._classify(lm, frame.shape, hand_side)

            if gesture:
                gestures_found.append((gesture, hand_side))

            # Draw landmarks on frame for debug
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, self._mp_hands.HAND_CONNECTIONS
            )

        if not gestures_found:
            self._hold_count = 0
            return None

        # Use the first detected gesture
        gesture, hand_side = gestures_found[0]

        # Stability filter: must hold same gesture for N frames
        if gesture == self._last_gesture:
            self._hold_count += 1
        else:
            self._last_gesture = gesture
            self._hold_count   = 1

        if self._hold_count < config.GESTURE_HOLD_FRAMES:
            return None

        if gesture not in config.GESTURES_TO_ANNOUNCE:
            return None

        return self._describe(gesture, hand_side)

    def _classify(self, lm, shape, hand_side):
        """
        Classify hand landmarks into a gesture name.
        Uses geometric rules on landmark positions.
        hand_side: "Left" or "Right" — needed to correctly detect thumb direction.
        """
        def finger_up(tip, mcp):
            return lm[tip].y < lm[mcp].y

        # FIX: thumb direction depends on which hand it is.
        # Right hand: thumb extends LEFT (lower x than index knuckle)
        # Left hand:  thumb extends RIGHT (higher x than index knuckle)
        if hand_side == "Right":
            thumb_up = lm[self.THUMB_TIP].x < lm[self.INDEX_MCP].x
        else:
            thumb_up = lm[self.THUMB_TIP].x > lm[self.INDEX_MCP].x

        index_up  = finger_up(self.INDEX_TIP,  self.INDEX_MCP)
        middle_up = finger_up(self.MIDDLE_TIP, self.MIDDLE_MCP)
        ring_up   = finger_up(self.RING_TIP,   self.RING_MCP)
        pinky_up  = finger_up(self.PINKY_TIP,  self.PINKY_MCP)

        fingers_up = sum([index_up, middle_up, ring_up, pinky_up])

        # ── STOP: all 4 fingers up, palm open ────────────────
        if index_up and middle_up and ring_up and pinky_up:
            wrist_y  = lm[self.WRIST].y
            middle_y = lm[self.MIDDLE_TIP].y
            if middle_y < wrist_y - 0.2:
                return "stop"

        # ── THUMBS UP: only thumb extended, fist otherwise ───
        if thumb_up and fingers_up == 0:
            if lm[self.THUMB_TIP].y < lm[self.WRIST].y - 0.1:
                return "thumbs_up"

        # ── FIST: all fingers and thumb closed ────────────────
        if fingers_up == 0 and not thumb_up:
            return "fist"

        # ── POINTING: only index finger extended ──────────────
        if index_up and not middle_up and not ring_up and not pinky_up:
            tip_x = lm[self.INDEX_TIP].x
            mcp_x = lm[self.INDEX_MCP].x
            if tip_x < mcp_x - 0.05:
                return "pointing_left"
            elif tip_x > mcp_x + 0.05:
                return "pointing_right"
            return "pointing_forward"

        # ── HANDSHAKE: index + middle up, wrist centered ──────
        if index_up and middle_up and not ring_up and not pinky_up:
            wrist_x = lm[self.WRIST].x
            if 0.3 < wrist_x < 0.7:
                return "handshake"

        # ── WAVE: all 4 fingers up ────────────────────────────
        if fingers_up == 4:
            return "wave"

        return None

    def _describe(self, gesture, hand_side):
        """Convert gesture + hand side to a natural spoken string."""
        hand_str = f"{hand_side.lower()} hand"
        descriptions = {
            "handshake":        f"Someone is extending their {hand_str} for a handshake.",
            "wave":             f"Someone is waving their {hand_str} at you.",
            "stop":             f"Someone is signalling stop with their {hand_str}. Please pause.",
            "pointing_left":    f"Someone is pointing to your left.",
            "pointing_right":   f"Someone is pointing to your right.",
            "pointing_forward": f"Someone is pointing ahead of you.",
            "thumbs_up":        f"Someone is giving a thumbs up.",
            # FIX: removed alarming "Be cautious" — fist is often natural and non-threatening
            "fist":             f"A closed fist detected with the {hand_str}.",
        }
        return descriptions.get(gesture, f"Gesture detected: {gesture}.")