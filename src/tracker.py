from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b):
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


class Track:
    def __init__(self, track_id, bbox, cls, t_sec):
        self.id = track_id
        self.cls = cls
        self.bbox = tuple(float(v) for v in bbox)
        self.history = [(t_sec, self.bbox)]
        self.time_since_update = 0.0
        self.hits = 1

    def predict(self, dt=0.12):
        if len(self.history) >= 2:
            (t0, b0), (t1, b1) = self.history[-2], self.history[-1]
            span = max(t1 - t0, 1e-3)
            vx1 = (b1[0] - b0[0]) / span
            vy1 = (b1[1] - b0[1]) / span
            vx2 = (b1[2] - b0[2]) / span
            vy2 = (b1[3] - b0[3]) / span
            return (
                b1[0] + vx1 * dt,
                b1[1] + vy1 * dt,
                b1[2] + vx2 * dt,
                b1[3] + vy2 * dt,
            )
        return self.bbox

    def update(self, bbox, t_sec):
        self.bbox = tuple(float(v) for v in bbox)
        self.history.append((t_sec, self.bbox))
        self.time_since_update = 0.0
        self.hits += 1
        if len(self.history) > 300:
            self.history = self.history[-300:]

    def centroid(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def speed(self, window_sec=1.0):
        if len(self.history) < 2:
            return 0.0, (0.0, 0.0)
        t_now = self.history[-1][0]
        idx = len(self.history) - 1
        while idx > 0 and t_now - self.history[idx][0] < window_sec:
            idx -= 1
        t0, b0 = self.history[idx]
        t1, b1 = self.history[-1]
        dt = max(t1 - t0, 1e-3)
        c0 = ((b0[0] + b0[2]) / 2.0, (b0[1] + b0[3]) / 2.0)
        c1 = ((b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0)
        dx = c1[0] - c0[0]
        dy = c1[1] - c0[1]
        dist = (dx ** 2 + dy ** 2) ** 0.5
        return dist / dt, (dx / dt, dy / dt)


class Tracker:
    def __init__(self, iou_threshold=0.3, max_age=1.0):
        self.tracks: list[Track] = []
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def update(self, boxes, classes, t_sec):
        preds = [tr.predict() for tr in self.tracks]
        matched, unmatched_dets = self._associate(preds, boxes, classes)
        for ti, di in matched:
            self.tracks[ti].update(boxes[di], t_sec)
        for di in unmatched_dets:
            tr = Track(self.next_id, boxes[di], classes[di], t_sec)
            self.next_id += 1
            self.tracks.append(tr)
        for tr in self.tracks:
            tr.time_since_update = t_sec - tr.history[-1][0]
        self.tracks = [tr for tr in self.tracks if tr.time_since_update <= self.max_age]
        return self.tracks

    def _associate(self, preds, boxes, classes):
        if len(preds) == 0 or len(boxes) == 0:
            return [], list(range(len(boxes)))
        cost = np.ones((len(preds), len(boxes)))
        for i, p in enumerate(preds):
            for j, b in enumerate(boxes):
                if self.tracks[i].cls == classes[j]:
                    cost[i, j] = 1.0 - iou(p, b)
        row, col = linear_sum_assignment(cost)
        matched = []
        used_dets = set()
        for r, c in zip(row, col):
            if cost[r, c] < (1.0 - self.iou_threshold):
                matched.append((r, c))
                used_dets.add(c)
        unmatched_dets = [j for j in range(len(boxes)) if j not in used_dets]
        return matched, unmatched_dets
