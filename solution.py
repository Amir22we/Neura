from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from src.detector import Detector, VEHICLE_IDS, PEDESTRIAN_ID, TRAFFIC_LIGHT_ID
from src.tracker import Tracker
from src.geometry import SceneModel
from src.signals import TrafficLightState
from src.obstacles import ObstacleDetector, FireSmokeDetector
from src.rules import (
    _deceleration,
    track_flags,
    pair_flags,
    crossing_flag,
    congestion_flag,
    stopline_flags,
)
from src.segments import flags_to_segments, clip_to_duration

CLASSES: list[str] = [
    "accident",
    "near_miss",
    "red_light",
    "wrong_way",
    "illegal_u_turn",
    "stopped_vehicle",
    "jaywalking",
    "failure_to_yield",
    "illegal_turn",
    "solid_line_crossing",
    "stop_line",
    "congestion",
    "road_obstacle",
    "fire_smoke",
]

RISK_HORIZON_SEC = 5.0

SAMPLE_STRIDE = int(os.environ.get("EVENT_SAMPLE_STRIDE", "3"))
CALIB_SEC = float(os.environ.get("SCENE_CALIB_SEC", "15.0"))
RECALIB_EVERY_SEC = 10.0
# Keep the submission on the standard pretrained detector. Experimental
# training artifacts may remain in the repository but are not used here.
YOLO_WEIGHTS = "weights/yolov8n.pt"
YOLO_DEVICE = os.environ.get("YOLO_DEVICE") or None

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        weights = Path(YOLO_WEIGHTS)
        if not weights.is_absolute():
            weights = Path(__file__).resolve().parent / weights
        if not weights.exists():
            fallback = Path(__file__).resolve().parent / "weights" / "yolov8n.pt"
            weights = fallback
        _detector = Detector(weights=str(weights), device=YOLO_DEVICE)
    return _detector


def _split_detections(boxes, cls_ids):
    det_boxes, det_cls, tl_boxes = [], [], []
    for box, c in zip(boxes, cls_ids):
        if c in VEHICLE_IDS:
            det_boxes.append(box)
            det_cls.append("vehicle")
        elif c == PEDESTRIAN_ID:
            det_boxes.append(box)
            det_cls.append("pedestrian")
        elif c == TRAFFIC_LIGHT_ID:
            tl_boxes.append(box)
    return det_boxes, det_cls, tl_boxes


def detect_events(video_path: str) -> list[list]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    duration = frame_count / float(fps) if frame_count > 0 else 0.0

    detector = _get_detector()
    tracker = Tracker()
    scene = SceneModel(width, height)
    tls = TrafficLightState()
    obstacle_det = ObstacleDetector()
    fire_det = FireSmokeDetector()

    times: list[float] = []
    per_class_flags = {c: [] for c in CLASSES}

    frame_idx = 0
    calibrated_once = False
    last_duration = duration

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_sec = frame_idx / fps
        last_duration = t_sec

        if frame_idx % SAMPLE_STRIDE != 0:
            frame_idx += 1
            continue

        boxes, cls_ids, _ = detector.infer(frame)
        det_boxes, det_cls, tl_boxes = _split_detections(boxes, cls_ids)
        tracks = tracker.update(det_boxes, det_cls, t_sec)

        for tr in tracks:
            scene.observe(tr)

        if not calibrated_once and t_sec >= CALIB_SEC:
            scene.calibrate()
            calibrated_once = True
        elif calibrated_once and frame_idx % max(1, int(fps * RECALIB_EVERY_SEC / SAMPLE_STRIDE)) == 0:
            scene.calibrate()

        tl_state = tls.update(frame, tl_boxes)

        active = [t for t in tracks if t.time_since_update == 0]
        frame_flags = {c: False for c in CLASSES}

        for tr in active:
            for k, v in track_flags(tr, scene).items():
                frame_flags[k] = frame_flags[k] or v

        accident, near_miss = pair_flags(active)
        frame_flags["accident"] = accident
        frame_flags["near_miss"] = near_miss
        frame_flags["failure_to_yield"] = crossing_flag(active, scene)
        frame_flags["congestion"] = congestion_flag(active, scene)

        red_light, stop_line = stopline_flags(active, scene, tl_state)
        frame_flags["red_light"] = red_light
        frame_flags["stop_line"] = stop_line

        frame_flags["road_obstacle"] = obstacle_det.update(frame, scene, active)
        frame_flags["fire_smoke"] = fire_det.update(frame, scene)

        times.append(t_sec)
        for c in CLASSES:
            per_class_flags[c].append(frame_flags[c])

        frame_idx += 1

    cap.release()

    events: list[list] = []
    for c in CLASSES:
        events.extend(
            flags_to_segments(
                per_class_flags[c],
                times,
                c,
                end_time=last_duration,
            )
        )
    events = clip_to_duration(events, last_duration)
    return events


def _time_to_collision(a, b):
    ca, cb = a.centroid(), b.centroid()
    _, va = a.speed(0.6)
    _, vb = b.speed(0.6)
    rx, ry = cb[0] - ca[0], cb[1] - ca[1]
    rvx, rvy = vb[0] - va[0], vb[1] - va[1]
    rel_speed_sq = rvx ** 2 + rvy ** 2
    if rel_speed_sq < 1e-6:
        return None
    t_closest = -(rx * rvx + ry * rvy) / rel_speed_sq
    if t_closest <= 0:
        return None
    closest_dist = ((rx + rvx * t_closest) ** 2 + (ry + rvy * t_closest) ** 2) ** 0.5
    scale = max(a.bbox[2] - a.bbox[0], b.bbox[2] - b.bbox[0], 1.0)
    if closest_dist < scale * 1.5:
        return t_closest
    return None


def _risk_from_tracks(tracks, scene=None):
    """Return a causal accident risk using only current track histories."""
    active = [t for t in tracks if t.time_since_update == 0]
    min_ttc = None
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            first, second = active[i], active[j]
            if first.cls == "pedestrian" and second.cls == "pedestrian":
                continue
            if first.cls == "pedestrian" or second.cls == "pedestrian":
                pedestrian = first if first.cls == "pedestrian" else second
                vehicle = second if first.cls == "pedestrian" else first
                if scene is not None and not scene.on_road(*pedestrian.centroid()):
                    continue
                distance = np.linalg.norm(
                    np.subtract(pedestrian.centroid(), vehicle.centroid())
                )
                vehicle_width = max(vehicle.bbox[2] - vehicle.bbox[0], 1.0)
                if distance > vehicle_width * 3.0:
                    continue
            ttc = _time_to_collision(first, second)
            if ttc is not None and (min_ttc is None or ttc < min_ttc):
                min_ttc = ttc

    ttc_score = 0.0
    if min_ttc is not None:
        ttc_score = max(0.0, min(1.0, 1.0 - min_ttc / RISK_HORIZON_SEC))

    braking_score = 0.0
    for track in active:
        if track.cls != "vehicle":
            continue
        braking_score = max(braking_score, min(1.0, _deceleration(track) / 60.0))

    # TTC is the strongest signal; braking adds sensitivity before contact.
    return max(ttc_score, 0.35 * braking_score)


class RiskEstimator:
    def reset(self, meta: dict) -> None:
        self.fps = float(meta.get("fps") or 25.0)
        self.width = int(meta.get("width") or 1)
        self.height = int(meta.get("height") or 1)
        self.detector = _get_detector()
        self.tracker = Tracker()
        self.scene = SceneModel(self.width, self.height)
        self.frame_idx = 0
        self.last_score = 0.0
        self.stride = SAMPLE_STRIDE

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        if self.frame_idx % self.stride != 0:
            self.frame_idx += 1
            return self.last_score

        boxes, cls_ids, _ = self.detector.infer(frame)
        det_boxes, det_cls, _ = _split_detections(boxes, cls_ids)
        return self.step_detections(det_boxes, det_cls, t_sec)

    def step_detections(self, det_boxes, det_cls, t_sec: float) -> float:
        """Update risk a from detections already computed for the current frame."""
        if self.frame_idx % self.stride != 0:
            self.frame_idx += 1
            return self.last_score

        tracks = self.tracker.update(det_boxes, det_cls, t_sec)
        for tr in tracks:
            self.scene.observe(tr)
        if self.frame_idx > 0 and self.frame_idx % max(1, int(self.fps * RECALIB_EVERY_SEC / self.stride)) == 0:
            self.scene.calibrate()

        active = [t for t in tracks if t.time_since_update == 0]
        raw_score = _risk_from_tracks(active, self.scene)

        self.last_score = max(raw_score, self.last_score * 0.85)
        self.frame_idx += 1
        return self.last_score
