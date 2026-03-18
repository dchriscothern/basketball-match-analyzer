from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _find_image_for_stem(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _draw_labels(image: np.ndarray, label_path: Path) -> np.ndarray:
    canvas = image.copy()
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        cv2.putText(canvas, "empty", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
        return canvas

    height, width = canvas.shape[:2]
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, x_center, y_center, box_width, box_height = parts
        x_center = float(x_center) * width
        y_center = float(y_center) * height
        box_width = float(box_width) * width
        box_height = float(box_height) * height
        x1 = int(max(0, x_center - box_width / 2.0))
        y1 = int(max(0, y_center - box_height / 2.0))
        x2 = int(min(width - 1, x_center + box_width / 2.0))
        y2 = int(min(height - 1, y_center + box_height / 2.0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 215, 255), 2)
        cv2.circle(canvas, (int(x_center), int(y_center)), 4, (0, 215, 255), -1)
    return canvas


def _load_tiles(image_dir: Path, label_dir: Path, sample_count: int, seed: int) -> list[np.ndarray]:
    labels = sorted(label_dir.glob("*.txt"))
    if not labels:
        return []
    random.Random(seed).shuffle(labels)
    chosen = labels[:sample_count]

    tiles: list[np.ndarray] = []
    for label_path in chosen:
        image_path = _find_image_for_stem(image_dir, label_path.stem)
        if image_path is None:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        rendered = _draw_labels(image, label_path)
        rendered = cv2.resize(rendered, (320, 180), interpolation=cv2.INTER_AREA)
        cv2.putText(
            rendered,
            label_path.stem,
            (10, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(rendered)
    return tiles


def _make_contact_sheet(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    if not tiles:
        raise RuntimeError("No labeled image tiles available for preview.")

    tile_height, tile_width = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / columns)
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)

    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns
        y1 = row * tile_height
        y2 = y1 + tile_height
        x1 = col * tile_width
        x2 = x1 + tile_width
        sheet[y1:y2, x1:x2] = tile

    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a quick visual preview of labeled ball-detector samples")
    parser.add_argument("--dataset-root", default="datasets/ball_detector", help="Dataset root folder")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split to preview")
    parser.add_argument("--sample-count", type=int, default=12, help="Number of labeled samples to include")
    parser.add_argument("--columns", type=int, default=3, help="Columns in the preview contact sheet")
    parser.add_argument("--seed", type=int, default=7, help="Shuffle seed")
    parser.add_argument("--output", default=None, help="Optional preview image output path")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    image_dir = dataset_root / "images" / args.split
    label_dir = dataset_root / "labels" / args.split
    if not image_dir.exists():
        raise FileNotFoundError(f"Image split folder not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label split folder not found: {label_dir}")

    tiles = _load_tiles(image_dir, label_dir, args.sample_count, args.seed)
    preview = _make_contact_sheet(tiles, args.columns)

    output_path = Path(args.output) if args.output else dataset_root / f"{args.split}_preview.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), preview)
    print(f"Saved dataset preview to {output_path.resolve()}")


if __name__ == "__main__":
    main()
