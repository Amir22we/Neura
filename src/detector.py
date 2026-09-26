from __future__ import annotations

import numpy as np

VEHICLE_IDS = {2, 3, 5, 7}
PEDESTRIAN_ID = 0
TRAFFIC_LIGHT_ID = 9


class Detector:
    def __init__(self, weights="weights/yolov8n.pt", device=None, conf=0.35, imgsz=640):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        if device is None:
            try:
                import torch

                device = "0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def infer(self, frame):
        result = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.zeros((0, 4)), np.zeros((0,), dtype=int), np.zeros((0,))
        boxes = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        conf = result.boxes.conf.cpu().numpy()
        return boxes, cls, conf
