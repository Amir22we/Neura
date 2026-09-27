"""Run the detector on the newest video file in a folder.

The input is normalized to MP4 with OpenCV before the official harness runs.
This keeps the local command independent of the original filename/extension.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {
    ".avi", ".dav", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg",
    ".mpg", ".ts", ".webm", ".wmv",
}


def newest_video(folder: Path) -> Path:
    candidates = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not candidates:
        raise FileNotFoundError(f"no supported video files found in {folder}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def convert_to_mp4(source: Path, destination: Path) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(
            f"cannot open {source}; install FFmpeg or use a codec supported by OpenCV"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"cannot determine video dimensions for {source}")

    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("OpenCV cannot create MP4 output on this system")

    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            frames += 1
    finally:
        capture.release()
        writer.release()
    if frames == 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"video contains no readable frames: {source}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="incoming", help="folder where the new video is uploaded")
    parser.add_argument("--out", default="predictions_latest.json")
    parser.add_argument("--team", default="unnamed-team")
    parser.add_argument("--keep-mp4", action="store_true", help="do not delete temporary normalized video")
    args = parser.parse_args()

    source = newest_video(Path(args.input))
    with tempfile.TemporaryDirectory(prefix="wiut_video_") as temp_dir:
        normalized = Path(temp_dir) / "latest.mp4"
        print(f"input: {source.name}")
        convert_to_mp4(source, normalized)
        command = [
            sys.executable,
            "run_submission.py",
            "--videos",
            str(normalized),
            "--out",
            args.out,
            "--team",
            args.team,
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            return result.returncode
        if args.keep_mp4:
            kept = Path(args.out).with_suffix(".normalized.mp4")
            shutil.copy2(normalized, kept)
            print(f"normalized video: {kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
