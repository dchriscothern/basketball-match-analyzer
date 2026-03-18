from __future__ import annotations

import argparse
import random
import shutil
import tempfile
from pathlib import Path

import cv2

from basketball_analyzer.analysis_profiles import prepare_analysis_source
from basketball_analyzer.pipeline import BasketballAnalysisPipeline


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


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


def _find_image_for_stem(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _resolve_recent_temp_sources(limit: int) -> list[Path]:
    temp_root = Path(tempfile.gettempdir())
    candidates = sorted(
        temp_root.glob("basketball_url_*\\source.mp4"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[:limit]


def _clear_split_dirs(dataset_root: Path) -> None:
    for relative in (
        "_bootstrap",
        "images\\train",
        "images\\val",
        "images\\test",
        "labels\\train",
        "labels\\val",
        "labels\\test",
    ):
        target = dataset_root / relative
        if target.exists():
            shutil.rmtree(target)


def _copy_pair(image_path: Path, label_path: Path, dataset_root: Path, split: str) -> None:
    image_target_dir = dataset_root / "images" / split
    label_target_dir = dataset_root / "labels" / split
    image_target_dir.mkdir(parents=True, exist_ok=True)
    label_target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_target_dir / image_path.name)
    shutil.copy2(label_path, label_target_dir / label_path.name)


def _validate_dataset(dataset_root: Path) -> list[str]:
    issues: list[str] = []
    for split in ("train", "val", "test"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        images = sorted(path for path in image_dir.glob("*") if path.suffix.lower() in IMAGE_EXTENSIONS) if image_dir.exists() else []
        labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
        label_map = {path.stem: path for path in labels}

        for image_path in images:
            label_path = label_map.get(image_path.stem)
            if label_path is None:
                issues.append(f"{split}: missing label for {image_path.name}")
                continue
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            for line in text.splitlines():
                parts = line.split()
                if len(parts) != 5:
                    issues.append(f"{split}: bad label format in {label_path.name}: {line}")

        for label_path in labels:
            if _find_image_for_stem(image_dir, label_path.stem) is None:
                issues.append(f"{split}: label without matching image: {label_path.name}")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a larger bootstrap ball dataset from multiple recent temp clips")
    parser.add_argument("--recent-temp-count", type=int, default=4, help="How many recent temp source clips to include")
    parser.add_argument("--dataset-root", default="datasets/ball_detector", help="Dataset root folder")
    parser.add_argument("--analysis-profile", default="Standard Clip", choices=["Fast Preview", "Standard Clip", "Full Clip"])
    parser.add_argument("--tracking-backend", default="yolo_detect", help="Tracking backend to use for bootstrap labels")
    parser.add_argument("--prefix", default="batch_bootstrap", help="Filename prefix")
    parser.add_argument("--include-empty", action="store_true", help="Also export sampled frames without ball labels as negatives")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=7, help="Shuffle seed")
    parser.add_argument("--clear-existing", action="store_true", help="Clear existing split folders before copying")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Ratios must satisfy: train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio < 1")

    video_paths = _resolve_recent_temp_sources(args.recent_temp_count)
    if not video_paths:
        raise FileNotFoundError("No recent temporary basketball source clips were found. Run the app on a clip first.")

    if args.clear_existing:
        _clear_split_dirs(dataset_root)

    bootstrap_image_dir = dataset_root / "_bootstrap" / "images"
    bootstrap_label_dir = dataset_root / "_bootstrap" / "labels"
    bootstrap_image_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_label_dir.mkdir(parents=True, exist_ok=True)

    pipeline = BasketballAnalysisPipeline()
    exported = 0
    positives = 0
    negatives = 0

    for clip_index, video_path in enumerate(video_paths):
        analysis_path, _ = prepare_analysis_source(video_path, args.analysis_profile)
        result = pipeline.run(analysis_path, tracking_backend=args.tracking_backend)
        tracked_frames = {frame.frame_index: frame for frame in result.frames}
        capture = cv2.VideoCapture(str(analysis_path))
        if not capture.isOpened():
            continue

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

            stem = f"{Path(args.prefix).stem}_{clip_index:02d}_{exported:04d}"
            image_path = bootstrap_image_dir / f"{stem}.jpg"
            label_path = bootstrap_label_dir / f"{stem}.txt"
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

    pairs: list[tuple[Path, Path]] = []
    for label_path in sorted(bootstrap_label_dir.glob("*.txt")):
        image_path = _find_image_for_stem(bootstrap_image_dir, label_path.stem)
        if image_path is not None:
            pairs.append((image_path, label_path))

    if not pairs:
        raise RuntimeError("No image/label pairs found after bootstrap export.")

    random.Random(args.seed).shuffle(pairs)
    total = len(pairs)
    train_cut = int(total * args.train_ratio)
    val_cut = train_cut + int(total * args.val_ratio)
    split_map = {
        "train": pairs[:train_cut],
        "val": pairs[train_cut:val_cut],
        "test": pairs[val_cut:],
    }

    for split, split_pairs in split_map.items():
        for image_path, label_path in split_pairs:
            _copy_pair(image_path, label_path, dataset_root, split)

    issues = _validate_dataset(dataset_root)

    print(f"Source clips used: {len(video_paths)}")
    for path in video_paths:
        print(f"- {path}")
    print(f"Bootstrap export: {bootstrap_image_dir.resolve()}")
    print(f"Exported frames: {exported}")
    print(f"Positive labels: {positives}")
    print(f"Negative labels: {negatives}")
    print(f"Dataset root: {dataset_root.resolve()}")
    for split, split_pairs in split_map.items():
        print(f"{split}: {len(split_pairs)} pairs")

    if issues:
        print("\nDataset issues")
        for issue in issues[:50]:
            print(f"- {issue}")
        if len(issues) > 50:
            print(f"- ... and {len(issues) - 50} more")
    else:
        print("\nDataset validation passed.")

    print("\nNext preview command:")
    print(
        "C:\\GitHub\\basketball-match-analyzer\\.venv\\Scripts\\python.exe "
        "preview_ball_dataset.py --dataset-root datasets\\ball_detector --split train"
    )


if __name__ == "__main__":
    main()
