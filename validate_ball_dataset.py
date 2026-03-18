from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a YOLO-style basketball ball detector dataset")
    parser.add_argument("--dataset-root", default="datasets/ball_detector", help="Dataset root folder")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    summary: dict[str, dict[str, int]] = {}
    issues: list[str] = []

    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = sorted(path for path in image_dir.glob("*") if _is_image(path)) if image_dir.exists() else []
        labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
        label_map = {path.stem: path for path in labels}
        split_summary = {
            "images": len(images),
            "labels": len(labels),
            "positive_labels": 0,
            "empty_labels": 0,
            "missing_labels": 0,
            "bad_lines": 0,
        }

        for image_path in images:
            label_path = label_map.get(image_path.stem)
            if label_path is None:
                split_summary["missing_labels"] += 1
                issues.append(f"{split}: missing label for {image_path.name}")
                continue

            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                split_summary["empty_labels"] += 1
                continue

            split_summary["positive_labels"] += 1
            for line in text.splitlines():
                parts = line.split()
                if len(parts) != 5:
                    split_summary["bad_lines"] += 1
                    issues.append(f"{split}: bad label format in {label_path.name}: {line}")
                    continue
                try:
                    class_id = int(parts[0])
                    coords = [float(value) for value in parts[1:]]
                except ValueError:
                    split_summary["bad_lines"] += 1
                    issues.append(f"{split}: non-numeric label values in {label_path.name}: {line}")
                    continue
                if class_id != 0:
                    split_summary["bad_lines"] += 1
                    issues.append(f"{split}: unexpected class id in {label_path.name}: {line}")
                if any(value < 0.0 or value > 1.0 for value in coords):
                    split_summary["bad_lines"] += 1
                    issues.append(f"{split}: normalized bbox out of range in {label_path.name}: {line}")

        for label_path in labels:
            image_exists = any((root / "images" / split / f"{label_path.stem}{ext}").exists() for ext in IMAGE_EXTENSIONS)
            if not image_exists:
                issues.append(f"{split}: label without matching image: {label_path.name}")

        summary[split] = split_summary

    print("Ball dataset summary")
    for split, values in summary.items():
        print(
            f"{split}: images={values['images']} labels={values['labels']} "
            f"positive={values['positive_labels']} empty={values['empty_labels']} "
            f"missing={values['missing_labels']} bad_lines={values['bad_lines']}"
        )

    if issues:
        print("\nIssues")
        for issue in issues[:100]:
            print(f"- {issue}")
        if len(issues) > 100:
            print(f"- ... and {len(issues) - 100} more")
    else:
        print("\nNo dataset issues found.")


if __name__ == "__main__":
    main()
