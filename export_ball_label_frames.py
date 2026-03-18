from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Export sampled frames from a basketball clip for ball labeling")
    parser.add_argument("--video", required=True, help="Path to local MP4 clip")
    parser.add_argument("--output", default="datasets/ball_detector/images/train", help="Output image directory")
    parser.add_argument("--every-nth-frame", type=int, default=6, help="Sample every Nth frame")
    parser.add_argument("--max-frames", type=int, default=250, help="Maximum frames to export")
    parser.add_argument("--prefix", default="ball", help="Filename prefix")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    saved = 0
    frame_index = 0
    stem = Path(args.prefix).stem

    while saved < args.max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % max(1, args.every_nth_frame) == 0:
            output_path = output_dir / f"{stem}_{saved:04d}.jpg"
            cv2.imwrite(str(output_path), frame)
            saved += 1
        frame_index += 1

    capture.release()
    print(f"Exported {saved} frames to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
