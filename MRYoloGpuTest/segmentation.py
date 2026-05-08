# ============================================
# YOLOv8 SEGMENTATION MODULE
# ============================================

import cv2
import numpy as np
from ultralytics import YOLO


class YOLOSegmentation:
    def __init__(
        self,
        model_path="yolov8n-seg.pt",
        device="cpu",
        conf=0.4
    ):

        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf

    def process(self, frame):

        results = self.model(
            frame,
            imgsz=320,
            verbose=False
        )[0]

        h, w = frame.shape[:2]

        overlay = frame.copy()

        combined_mask = np.zeros((h, w), dtype=np.float32)

        detections = []

        if results.masks is not None:

            masks = results.masks.data.cpu().numpy()

            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()

            for i, mask in enumerate(masks):

                mask = cv2.resize(mask, (w, h))

                binary_mask = (mask > 0.5).astype(np.uint8)

                combined_mask = np.maximum(
                    combined_mask,
                    binary_mask.astype(np.float32)
                )

                color = np.random.randint(0, 255, (3,), dtype=np.uint8)

                overlay[binary_mask == 1] = (
                    0.5 * overlay[binary_mask == 1]
                    + 0.5 * color
                )

                x1, y1, x2, y2 = boxes[i].astype(int)

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "class": int(classes[i]),
                    "score": float(scores[i])
                })

                cv2.rectangle(
                    overlay,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

        combined_mask = np.stack(
            [combined_mask] * 3,
            axis=-1
        )

        return overlay, combined_mask, detections