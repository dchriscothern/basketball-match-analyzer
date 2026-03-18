from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _find_image_for_stem(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _copy_pair(image_path: Path, label_path: Path, dataset_root: Path, split: str) -> None:
    image_target_dir = dataset_root / "images" / split
    label_target_dir = dataset_root / "labels" / split
    image_target_dir.mkdir(parents=True, exist_ok=True)
    label_target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_target_dir / image_path.name)
    shutil.copy2(label_path, label_target_dir / label_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split bootstrap ball labels into train/val/test YOLO folders")
    parser.add_argument("--images-source", required=True, help="Flat source image folder")
    parser.add_argument("--labels-source", required=True, help="Flat source label folder")
    parser.add_argument("--dataset-root", default="datasets/ball_detector", help="Dataset root folder")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=7, help="Shuffle seed")
    parser.add_argument("--clear-existing", action="store_true", help="Clear existing split folders before copying")
    args = parser.parse_args()

    images_source = Path(args.images_source)
    labels_source = Path(args.labels_source)
    dataset_root = Path(args.dataset_root)

    if not images_source.exists():
        raise FileNotFoundError(f"Images source not found: {images_source}")
    if not labels_source.exists():
        raise FileNotFoundError(f"Labels source not found: {labels_source}")
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Ratios must satisfy: train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio < 1")

    if args.clear_existing:
        for split in ("train", "val", "test"):
            for subdir in ("images", "labels"):
                target = dataset_root / subdir / split
                if target.exists():
                    shutil.rmtree(target)

    pairs: list[tuple[Path, Path]] = []
    for label_path in sorted(labels_source.glob("*.txt")):
        image_path = _find_image_for_stem(images_source, label_path.stem)
        if image_path is None:
            continue
        pairs.append((image_path, label_path))

    if not pairs:
        raise RuntimeError("No image/label pairs found to split.")

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

    print(f"Prepared dataset at {dataset_root.resolve()}")
    for split, split_pairs in split_map.items():
        print(f"{split}: {len(split_pairs)} pairs")


if __name__ == "__main__":
    main()
