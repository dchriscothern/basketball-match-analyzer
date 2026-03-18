from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO  # type: ignore
except Exception as exc:  # pragma: no cover
    YOLO = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dedicated basketball ball detector with Ultralytics YOLO")
    parser.add_argument("--data", required=True, help="Path to YOLO dataset YAML")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model checkpoint to fine-tune")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--project", default="runs/ball_detector", help="Output project directory")
    parser.add_argument("--name", default="wnba_ball", help="Run name")
    parser.add_argument("--device", default="cpu", help="Training device, e.g. cpu or 0")
    args = parser.parse_args()

    if YOLO is None:
        raise RuntimeError(
            "Ultralytics is not available in this environment. "
            f"Import error: {IMPORT_ERROR}"
        )

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        single_cls=True,
        pretrained=True,
        patience=15,
        close_mosaic=10,
        save=True,
    )


if __name__ == "__main__":
    main()
