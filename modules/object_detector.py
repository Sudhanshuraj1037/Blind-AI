# modules/object_detector.py — YOLOv8 (optimized for speed)

from ultralytics import YOLO
import config

class ObjectDetector:
    def __init__(self):
        print("[ObjectDetector] Loading YOLOv8 model...")
        self.model = YOLO(config.YOLO_MODEL)
        # Move to GPU if available
        if config.USE_GPU:
            self.model.to(config.DEVICE)
        print(f"[ObjectDetector] Ready on {config.DEVICE}. Image size: {config.YOLO_IMG_SIZE}px")

    def detect(self, frame):
        """
        Run detection on a frame.
        Returns list of dicts: {label, confidence, box, is_danger, position}
        Uses smaller image size (320) for 2x speed boost.
        """
        results = self.model(
            frame,
            imgsz=config.YOLO_IMG_SIZE,   # 320 instead of 640 = much faster
            verbose=False,
            device=config.DEVICE,
        )[0]

        detections = []
        frame_w = frame.shape[1]

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < config.YOLO_CONFIDENCE:
                continue

            cls_id = int(box.cls[0])
            label  = self.model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            center_x  = (x1 + x2) / 2
            box_width  = (x2 - x1) / frame_w

            if center_x < frame_w * 0.33:
                position = "on your left"
            elif center_x > frame_w * 0.66:
                position = "on your right"
            else:
                position = "ahead"

            is_danger = (
                label in config.DANGER_CLASSES and
                box_width > config.YOLO_DANGER_ZONE
            )

            detections.append({
                "label":      label,
                "confidence": round(conf, 2),
                "box":        (x1, y1, x2, y2),
                "position":   position,
                "is_danger":  is_danger,
            })

        return detections

    def describe(self, detections):
        """Convert detections into a natural spoken sentence."""
        if not detections:
            return None
        danger = [d for d in detections if d["is_danger"]]
        normal = [d for d in detections if not d["is_danger"]]
        parts  = []
        if danger:
            for d in danger:
                parts.append(f"Warning! {d['label']} {d['position']}")
        if normal:
            names = [f"{d['label']} {d['position']}" for d in normal[:3]]
            parts.append("I see " + ", ".join(names))
        return ". ".join(parts) if parts else None