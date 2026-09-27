from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
from ultralytics import YOLO
import torch


CLASSES = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "pedestrian",
    "traffic_light",
]

# COCO ids used by the YOLO model.
COCO_TO_LOCAL = {
    0: 4,   # person
    2: 0,   # car
    3: 1,   # motorcycle is remapped below for compatibility with project names
    5: 2,   # bus
    7: 1,   # truck
    9: 5,   # traffic light
}


def _local_class(coco_id: int) -> int | None:
    if coco_id == 3:
        return 3
    return COCO_TO_LOCAL.get(coco_id)


def _write_label(path: Path, boxes, class_ids, width: int, height: int) -> int:
    rows = []
    for box, coco_id in zip(boxes, class_ids):
        local_id = _local_class(int(coco_id))
        if local_id is None:
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        center_x = ((x1 + x2) / 2.0) / width
        center_y = ((y1 + y2) / 2.0) / height
        box_width = max(0.0, x2 - x1) / width
        box_height = max(0.0, y2 - y1) / height
        rows.append(
            f"{local_id} {center_x:.6f} {center_y:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )
    path.write_text("\n".join(rows), encoding="utf-8")
    return len(rows)


def annotate(videos: list[Path], output: Path, stride: int, val_ratio: float) -> None:
    model = YOLO("weights/yolov8n.pt")
    device = 0 if torch.cuda.is_available() else "cpu"
    rng = random.Random(42)
    output = output.resolve()

    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_frames = 0
    total_boxes = 0
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"skip: cannot open {video}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_index = 0
        video_split = "val" if rng.random() < val_ratio else "train"
        print(f"{video.name}: {video_split}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride == 0:
                result = model.predict(
                    frame,
                    conf=0.35,
                    imgsz=640,
                    device=device,
                    verbose=False,
                )[0]
                if result.boxes is not None and len(result.boxes):
                    boxes = result.boxes.xyxy.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                else:
                    boxes, class_ids = [], []

                stem = f"{video.stem}_{frame_index:07d}"
                image_path = output / "images" / video_split / f"{stem}.jpg"
                label_path = output / "labels" / video_split / f"{stem}.txt"
                cv2.imwrite(str(image_path), frame)
                total_boxes += _write_label(
                    label_path,
                    boxes,
                    class_ids,
                    frame.shape[1],
                    frame.shape[0],
                )
                total_frames += 1
            frame_index += 1
        cap.release()
        print(f"  frames: {frame_index}, sampled: {(frame_index + stride - 1) // stride}")

    yaml_path = output / "data.yaml"
    yaml_path.write_text(
        "path: " + output.as_posix() + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES)),
        encoding="utf-8",
    )
    print(f"saved dataset: {output}")
    print(f"sampled frames: {total_frames}, boxes: {total_boxes}")
    print(f"data yaml: {yaml_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create pseudo-labeled YOLO data from traffic videos.")
    parser.add_argument("--videos", nargs="+", required=True, help="Input .mp4 files")
    parser.add_argument("--out", default="datasets/traffic_auto", help="Output dataset folder")
    parser.add_argument("--stride", type=int, default=10, help="Use every Nth frame")
    parser.add_argument("--val-ratio", type=float, default=0.25)
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    annotate([Path(video) for video in args.videos], Path(args.out), args.stride, args.val_ratio)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
