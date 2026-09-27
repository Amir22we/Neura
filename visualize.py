from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from solution import (
    RISK_HORIZON_SEC,
    RiskEstimator,
    _get_detector,
    _split_detections,
    _time_to_collision,
    SAMPLE_STRIDE,
)
from src.tracker import Tracker


COLORS = {
    "vehicle": (0, 220, 0),
    "pedestrian": (255, 180, 0),
}


def _active_events(events, timestamp):
    return [label for start, end, label in events if start <= timestamp <= end]


def _min_ttc(tracks):
    active = [track for track in tracks if track.time_since_update == 0]
    result = None
    for index, first in enumerate(active):
        for second in active[index + 1 :]:
            ttc = _time_to_collision(first, second)
            if ttc is not None and (result is None or ttc < result):
                result = ttc
    return result


def _event_subject_ids(label, tracks):
    active = [track for track in tracks if track.time_since_update == 0]
    vehicles = [track for track in active if track.cls == "vehicle"]
    pedestrians = [track for track in active if track.cls == "pedestrian"]
    if label in {"accident", "near_miss"}:
        ranked = []
        for index, first in enumerate(active):
            for second in active[index + 1 :]:
                distance = cv2.norm(
                    (
                        first.centroid()[0] - second.centroid()[0],
                        first.centroid()[1] - second.centroid()[1],
                    )
                )
                ranked.append((distance, first.id, second.id))
        if ranked:
            _, first_id, second_id = min(ranked)
            return [first_id, second_id]
    if label in {"jaywalking", "failure_to_yield"} and pedestrians:
        if label == "jaywalking":
            return [min(pedestrians, key=lambda track: track.centroid()[1]).id]
        return [track.id for track in pedestrians[:1] + vehicles[:1]]
    if label in {
        "wrong_way",
        "illegal_u_turn",
        "illegal_turn",
        "solid_line_crossing",
        "red_light",
        "stop_line",
        "stopped_vehicle",
    } and vehicles:
        return [vehicles[0].id]
    return []


def render(video_path: str, output_path: str) -> list[list]:
    source = Path(video_path)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1

    detector = _get_detector()
    tracker = Tracker()
    risk_estimator = RiskEstimator()
    risk_estimator.reset(
        {
            "video_id": source.name,
            "fps": float(fps),
            "width": width,
            "height": height,
            "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    )
    frame_idx = 0
    last_tracks = []
    events = []
    risk_curve = []
    event_subjects = {}

    # Run the official event detector first, then draw its time segments below.
    from solution import detect_events

    events = detect_events(str(source))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_idx / fps

        if frame_idx % max(1, SAMPLE_STRIDE) == 0:
            boxes, class_ids, _ = detector.infer(frame)
            det_boxes, det_classes, _ = _split_detections(boxes, class_ids)
            last_tracks = tracker.update(det_boxes, det_classes, timestamp)

        if frame_idx % max(1, SAMPLE_STRIDE) == 0:
            risk_score = float(
                risk_estimator.step_detections(det_boxes, det_classes, timestamp)
            )
        else:
            risk_estimator.frame_idx += 1
            risk_score = float(risk_estimator.last_score)
        risk_curve.append([round(timestamp, 4), round(risk_score, 4)])
        ttc = _min_ttc(last_tracks)

        for track in last_tracks:
            if track.time_since_update > 0.5:
                continue
            x1, y1, x2, y2 = (int(value) for value in track.bbox)
            color = COLORS.get(track.cls, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cx, cy = (int(value) for value in track.centroid())
            cv2.circle(frame, (cx, cy), 5, color, -1)
            text = f"{track.cls} #{track.id}"
            cv2.putText(
                frame,
                text,
                (x1, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        risk_color = (0, 220, 0)
        if risk_score >= 0.5:
            risk_color = (0, 0, 255)
        elif risk_score >= 0.2:
            risk_color = (0, 180, 255)
        risk_text = f"ACCIDENT RISK: {risk_score * 100:.0f}%"
        if risk_score >= 0.2:
            if ttc is not None and 0.0 < ttc <= RISK_HORIZON_SEC:
                risk_text += f" | possible in ~{ttc:.1f}s"
            else:
                risk_text += f" | possible within {RISK_HORIZON_SEC:.0f}s"
        panel_width = min(430, max(300, width // 3))
        panel_x = width - panel_width
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, height - 58),
            (width, height),
            (25, 25, 25),
            -1,
        )
        frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)
        cv2.putText(
            frame,
            risk_text,
            (panel_x + 14, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            risk_color,
            2,
            cv2.LINE_AA,
        )

        active = _active_events(events, timestamp)
        if active:
            line_height = 29
            header_height = 42
            max_visible = max(1, (height - header_height - 20) // line_height)
            visible_events = active[:max_visible]
            panel_height = header_height + line_height * len(visible_events) + 12
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (panel_x, 12),
                (width - 10, 12 + panel_height),
                (25, 25, 25),
                -1,
            )
            frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)
            cv2.putText(
                frame,
                f"EVENTS @ {timestamp:.2f}s",
                (panel_x + 14, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            for index, label in enumerate(visible_events):
                ids = _event_subject_ids(label, last_tracks)
                event_key = f"{label}:{next((event[0] for event in events if event[2] == label and event[0] <= timestamp <= event[1]), timestamp):.3f}"
                event_subjects[event_key] = ids
                suffix = f" [IDs: {', '.join('#' + str(track_id) for track_id in ids)}]" if ids else ""
                text = label + suffix
                cv2.putText(
                    frame,
                    text[:42],
                    (panel_x + 14, 70 + index * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 80, 255),
                    2,
                    cv2.LINE_AA,
                )
            if len(active) > len(visible_events):
                cv2.putText(
                    frame,
                    f"+{len(active) - len(visible_events)} more",
                    (panel_x + 14, 70 + len(visible_events) * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    json_path = output.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "video": source.name,
                "events": events,
                "event_subjects": event_subjects,
                "risk": risk_curve,
                "risk_horizon_sec": RISK_HORIZON_SEC,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an annotated traffic-event video.")
    parser.add_argument("--video", required=True, help="Input .mp4 file")
    parser.add_argument("--out", default="outputs/annotated.mp4", help="Output .mp4 file")
    args = parser.parse_args()

    events = render(args.video, args.out)
    print(f"Saved video: {args.out}")
    print(f"Saved events: {Path(args.out).with_suffix('.json')}")
    print(f"Events found: {len(events)}")
    for event in events:
        print(event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
