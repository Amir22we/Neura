from __future__ import annotations

import cv2
import numpy as np


def _iou(a, b):
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class ObstacleDetector:
    def __init__(self, min_area=250, persist_frames=6):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=45, detectShadows=False)
        self.min_area = min_area
        self.persist_frames = persist_frames
        self.blob_streak = {}

    def update(self, frame, scene, tracks):
        fg = self.bg.apply(frame, learningRate=0.003)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        if scene.road_mask.any():
            fg = cv2.bitwise_and(fg, fg, mask=scene.road_mask)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        occupied = [tr.bbox for tr in tracks if tr.time_since_update == 0]
        current_keys = set()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            box = (x, y, x + w, y + h)
            if any(_iou(box, ob) > 0.1 for ob in occupied):
                continue
            key = (x // 20, y // 20)
            current_keys.add(key)
            self.blob_streak[key] = self.blob_streak.get(key, 0) + 1
        for key in list(self.blob_streak.keys()):
            if key not in current_keys:
                self.blob_streak[key] = max(0, self.blob_streak[key] - 2)
                if self.blob_streak[key] == 0:
                    del self.blob_streak[key]
        return any(v >= self.persist_frames for v in self.blob_streak.values())


class FireSmokeDetector:
    def __init__(self, fire_ratio_th=0.02, smoke_ratio_th=0.08, streak_needed=3):
        self.fire_ratio_th = fire_ratio_th
        self.smoke_ratio_th = smoke_ratio_th
        self.streak_needed = streak_needed
        self.hits = 0

    def update(self, frame, scene):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        fire_mask = cv2.inRange(hsv, (5, 120, 150), (30, 255, 255))
        smoke_mask = cv2.inRange(hsv, (0, 0, 120), (180, 60, 220))
        if scene.road_mask.any():
            fire_mask = cv2.bitwise_and(fire_mask, fire_mask, mask=scene.road_mask)
            smoke_mask = cv2.bitwise_and(smoke_mask, smoke_mask, mask=scene.road_mask)
        fire_ratio = float(fire_mask.mean()) / 255.0
        smoke_ratio = float(smoke_mask.mean()) / 255.0
        active = fire_ratio > self.fire_ratio_th or smoke_ratio > self.smoke_ratio_th
        self.hits = self.hits + 1 if active else max(0, self.hits - 1)
        return self.hits >= self.streak_needed
