from __future__ import annotations

import cv2
import numpy as np


class SceneModel:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.directions = []
        self.stop_points = []
        self.road_mask = np.zeros((height, width), dtype=np.uint8)
        self.crossing_mask = np.zeros((height, width), dtype=np.uint8)
        self.flow_clusters = None
        self.stop_line_y = None
        self.stop_line_forward = 1
        self.calibrated = False

    def observe(self, track):
        if len(track.history) < 2:
            return
        t0, b0 = track.history[-2]
        t1, b1 = track.history[-1]
        c0 = ((b0[0] + b0[2]) / 2.0, (b0[1] + b0[3]) / 2.0)
        c1 = ((b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0)
        dx, dy = c1[0] - c0[0], c1[1] - c0[1]
        dist = (dx ** 2 + dy ** 2) ** 0.5
        cx, cy = int(c1[0]), int(c1[1])
        in_bounds = 0 <= cx < self.width and 0 <= cy < self.height
        if track.cls == "vehicle":
            if in_bounds:
                cv2.circle(self.road_mask, (cx, cy), 12, 255, -1)
            if dist > 1.2:
                self.directions.append((dx, dy))
            speed = dist / max(t1 - t0, 1e-3)
            if speed < 2.0 and in_bounds:
                self.stop_points.append(c1)
        elif track.cls == "pedestrian" and in_bounds:
            cv2.circle(self.crossing_mask, (cx, cy), 15, 255, -1)
            cv2.circle(self.road_mask, (cx, cy), 8, 255, -1)

    def calibrate(self, k=2):
        if len(self.directions) < 20:
            return
        dirs = np.array(self.directions, dtype=np.float32)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = dirs / norms
        angles = np.arctan2(unit[:, 1], unit[:, 0])
        feat = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
        k_use = min(k, len(feat))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, _, centers = cv2.kmeans(feat, k_use, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        self.flow_clusters = [float(np.arctan2(c[1], c[0])) for c in centers]
        kernel = np.ones((25, 25), np.uint8)
        self.road_mask = cv2.morphologyEx(self.road_mask, cv2.MORPH_CLOSE, kernel)
        self.crossing_mask = cv2.morphologyEx(self.crossing_mask, cv2.MORPH_CLOSE, kernel)
        if len(self.stop_points) >= 15:
            ys = np.array([p[1] for p in self.stop_points], dtype=np.float32)
            hist, edges = np.histogram(ys, bins=20)
            peak = int(np.argmax(hist))
            self.stop_line_y = float((edges[peak] + edges[peak + 1]) / 2.0)
            vy_signs = [np.sin(a) for a in self.flow_clusters]
            self.stop_line_forward = 1 if float(np.mean(vy_signs)) >= 0 else -1
        self.calibrated = True

    def on_road(self, x, y):
        xi, yi = int(x), int(y)
        if 0 <= xi < self.width and 0 <= yi < self.height:
            return bool(self.road_mask[yi, xi] > 0)
        return False

    def on_crossing(self, x, y):
        xi, yi = int(x), int(y)
        if 0 <= xi < self.width and 0 <= yi < self.height:
            return bool(self.crossing_mask[yi, xi] > 0)
        return False

    def nearest_flow_angle(self, angle):
        if not self.flow_clusters:
            return None
        diffs = [self.angle_diff(angle, c) for c in self.flow_clusters]
        idx = int(np.argmin(diffs))
        return self.flow_clusters[idx]

    @staticmethod
    def angle_diff(a, b):
        return abs(((a - b + np.pi) % (2 * np.pi)) - np.pi)
