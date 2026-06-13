# utils/camera.py — threaded, non-blocking camera  (corrected)
#
# FIXES applied:
#   1. Resize condition now checks BOTH width AND height (was width-only)
#   2. Added reconnect safety: _ok is set False when cap.read() fails
#   3. CAP_DSHOW kept — correct for Windows (Dell laptop)
#   4. Added is_open property for health checks from main loop

import cv2
import threading
import time
import config

class Camera:
    def __init__(self):
        self.cap        = None
        self._frame     = None
        self._ok        = False
        self._running   = False
        self._thread    = None
        self._lock      = threading.Lock()
        self._fail_count = 0          # FIX: track consecutive failures
        self._MAX_FAILS  = 30         # FIX: after 30 consecutive failures → mark stale

    def start(self):
        # CAP_DSHOW = DirectShow backend — correct and faster on Windows
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          config.FPS_TARGET)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # keep buffer at 1 — drop stale frames

        if not self.cap.isOpened():
            raise RuntimeError(
                "Cannot open camera. Check CAMERA_INDEX in config.py "
                "and ensure no other app is using the webcam."
            )

        # Warm up — grab frames to stabilise exposure
        for _ in range(5):
            self.cap.read()

        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[Camera] Started in threaded mode on index {config.CAMERA_INDEX} (CAP_DSHOW/Windows).")

    def _capture_loop(self):
        """Runs in background — continuously reads frames and stores latest."""
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                self._fail_count = 0

                # FIX: check BOTH dimensions, not just width
                if (config.PROCESS_WIDTH  != config.FRAME_WIDTH or
                        config.PROCESS_HEIGHT != config.FRAME_HEIGHT):
                    frame = cv2.resize(
                        frame,
                        (config.PROCESS_WIDTH, config.PROCESS_HEIGHT),
                        interpolation=cv2.INTER_LINEAR
                    )

                with self._lock:
                    self._frame = frame
                    self._ok    = True
            else:
                self._fail_count += 1
                # FIX: mark stale after repeated failures so main loop can react
                if self._fail_count >= self._MAX_FAILS:
                    with self._lock:
                        self._ok = False
                    print("[Camera] WARNING: camera read failing repeatedly — frame may be stale.")
                time.sleep(0.01)

    def read(self):
        """Non-blocking. Returns (success, latest_frame). Never waits."""
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ok, self._frame.copy()

    @property
    def is_open(self):
        """True if camera is running and returning valid frames."""
        with self._lock:
            return self._ok

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        print("[Camera] Stopped.")