from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from basketball_analyzer.analysis_profiles import prepare_analysis_source
from basketball_analyzer.pipeline import BasketballAnalysisPipeline


def _yolo_bbox_line(frame_width: int, frame_height: int, x1: float, y1: float, x2: float, y2: float) -> str:
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    return "0 {:.6f} {:.6f} {:.6f} {:.6f}".format(
        center_x / frame_width,
        center_y / frame_height,
        width / frame_width,
        height / frame_height,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export bootstrap ball labels from the current analyzer")
    parser.add_argument("--video", required=True, help="Path to local MP4 clip")
    parser.add_argument("--analysis-profile", default="Standard Clip", choices=["Fast Preview", "Standard Clip", "Full Clip"])
    parser.add_argument("--tracking-backend", default="yolo_detect", help="Tracking backend to use for bootstrap labels")
    parser.add_argument("--images-output", default="datasets/ball_detector/images/train", help="Output image directory")
    parser.add_argument("--labels-output", default="datasets/ball_detector/labels/train", help="Output YOLO label directory")
    parser.add_argument("--prefix", default="ball_bootstrap", help="Filename prefix")
    parser.add_argument("--include-empty", action="store_true", help="Also export sampled frames without ball labels as negatives")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    analysis_path, _ = prepare_analysis_source(video_path, args.analysis_profile)
    pipeline = BasketballAnalysisPipeline()
    result = pipeline.run(analysis_path, tracking_backend=args.tracking_backend)

    images_output = Path(args.images_output)
    labels_output = Path(args.labels_output)
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)

    tracked_frames = {frame.frame_index: frame for frame in result.frames}
    capture = cv2.VideoCapture(str(analysis_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open prepared video: {analysis_path}")

    exported = 0
    positives = 0
    negatives = 0
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        tracked = tracked_frames.get(frame_index)
        if tracked is None:
            frame_index += 1
            continue

        has_ball = tracked.ball is not None
        if not has_ball and not args.include_empty:
            frame_index += 1
            continue

        stem = f"{Path(args.prefix).stem}_{exported:04d}"
        image_path = images_output / f"{stem}.jpg"
        label_path = labels_output / f"{stem}.txt"
        cv2.imwrite(str(image_path), frame)

        if tracked.ball is not None:
            bbox = tracked.ball.bbox
            label_path.write_text(
                _yolo_bbox_line(frame.shape[1], frame.shape[0], bbox.x1, bbox.y1, bbox.x2, bbox.y2) + "\n",
                encoding="utf-8",
            )
            positives += 1
        else:
            label_path.write_text("", encoding="utf-8")
            negatives += 1

        exported += 1
        frame_index += 1

    capture.release()
    print(f"Prepared clip: {analysis_path}")
    print(f"Exported {exported} frames to {images_output.resolve()}")
    print(f"Positive labels: {positives}")
    print(f"Negative labels: {negatives}")


if __name__ == "__main__":
    main()
