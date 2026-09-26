from __future__ import annotations

import cv2
import numpy as np


class TrafficLightState:
    def __init__(self, memory=5):
        self.memory = memory
        self.recent = []
        self.state = "unknown"

    def update(self, frame, boxes):
        votes = []
        for box in boxes:
            x1, y1, x2, y2 = (int(v) for v in box)
            x1, y1 = max(0, x1), max(0, y1)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            red_mask = cv2.inRange(hsv, (0, 90, 90), (10, 255, 255)) | cv2.inRange(
                hsv, (170, 90, 90), (180, 255, 255)
            )
            green_mask = cv2.inRange(hsv, (45, 90, 90), (90, 255, 255))
            red_ratio = float(red_mask.mean())
            green_ratio = float(green_mask.mean())
            if red_ratio > green_ratio and red_ratio > 4.0:
                votes.append("red")
            elif green_ratio > red_ratio and green_ratio > 4.0:
                votes.append("green")
        if votes:
            self.recent.append(max(set(votes), key=votes.count))
        elif boxes:
            self.recent.append("unknown")
        self.recent = self.recent[-self.memory:]
        if self.recent:
            self.state = max(set(self.recent), key=self.recent.count)
        return self.state
