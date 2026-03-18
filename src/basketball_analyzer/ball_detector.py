from __future__ import annotations

import os
from pathlib import Path

from .schemas import BBox, Detection

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    repo_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault('YOLO_CONFIG_DIR', str(repo_root / '.ultralytics'))
    os.environ.setdefault('ULTRALYTICS_CONFIG_DIR', str(repo_root / '.ultralytics'))
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None


_MODEL_CACHE: dict[str, object] = {}


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class BallDetector:
    """Optional learned basketball detector.

    This is intentionally separate from the general player detector so we can
    swap in a small-ball model without rewriting the rest of the pipeline.

    Expected weights:
    - env var `BASKETBALL_BALL_MODEL`
    - or `basketball-ball.pt` / `ball-detector.pt` in the repo root
    """

    def __init__(self, conf: float = 0.03) -> None:
        self.conf = conf
        self._model = None
        self._weights_path = self._resolve_weights()

    def _resolve_weights(self) -> Path | None:
        if os.environ.get('BASKETBALL_DISABLE_LEARNED_BALL', '').strip().lower() in {'1', 'true', 'yes', 'on'}:
            return None

        explicit = os.environ.get('BASKETBALL_BALL_MODEL')
        if explicit:
            path = Path(explicit)
            if path.exists():
                return path

        repo_root = Path(__file__).resolve().parents[2]
        for candidate in (
            repo_root / 'basketball-ball.pt',
            repo_root / 'ball-detector.pt',
            repo_root / 'weights' / 'basketball-ball.pt',
        ):
            if candidate.exists():
                return candidate

        trained_candidates = sorted(
            repo_root.glob('runs/**/weights/best.pt'),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if trained_candidates:
            return trained_candidates[0]
        return None

    def is_available(self) -> bool:
        return YOLO is not None and cv2 is not None and self._weights_path is not None

    @property
    def active_weights_path(self) -> Path | None:
        return self._weights_path

    def _model_instance(self):
        if self._model is None and self._weights_path is not None:
            cache_key = str(self._weights_path.resolve())
            cached = _MODEL_CACHE.get(cache_key)
            if cached is None:
                cached = YOLO(str(self._weights_path))
                _MODEL_CACHE[cache_key] = cached
            self._model = cached
        return self._model

    def detect(
        self,
        frame,
        *,
        predicted_center: tuple[float, float] | None,
        player_boxes: list[BBox],
    ) -> Detection | None:
        if not self.is_available():
            return None

        height, width = frame.shape[:2]
        model = self._model_instance()
        if model is None:
            return None

        crops: list[tuple[object, tuple[int, int], str]] = []
        if predicted_center is not None:
            window = int(min(width, height) * 0.42)
            half = max(80, window // 2)
            cx, cy = int(predicted_center[0]), int(predicted_center[1])
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(width, cx + half)
            y2 = min(height, cy + half)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append((crop, (x1, y1), 'predicted'))

        for bbox in player_boxes[:10]:
            pad_x = int((bbox.x2 - bbox.x1) * 0.5)
            pad_y = int((bbox.y2 - bbox.y1) * 0.28)
            x1 = max(0, int(bbox.x1) - pad_x)
            y1 = max(0, int(bbox.y1) - pad_y)
            x2 = min(width, int(bbox.x2) + pad_x)
            y2 = min(height, int(bbox.y2) + pad_y)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append((crop, (x1, y1), 'player'))

        crops.append((frame, (0, 0), 'full'))
        best: tuple[float, Detection] | None = None

        for crop, offset, crop_type in crops:
            results = model.predict(
                source=crop,
                stream=False,
                conf=self.conf,
                verbose=False,
                device='cpu',
                imgsz=640,
            )
            for result in results:
                boxes = getattr(result, 'boxes', None)
                if boxes is None or len(boxes) == 0:
                    continue
                xyxy_list = boxes.xyxy.cpu().tolist()
                conf_list = boxes.conf.cpu().tolist()
                for xyxy, confidence in zip(xyxy_list, conf_list):
                    x1, y1, x2, y2 = xyxy
                    world_x1 = x1 + offset[0]
                    world_y1 = y1 + offset[1]
                    world_x2 = x2 + offset[0]
                    world_y2 = y2 + offset[1]
                    center = ((world_x1 + world_x2) / 2.0, (world_y1 + world_y2) / 2.0)
                    score = float(confidence)
                    area = max(1.0, (world_x2 - world_x1) * (world_y2 - world_y1))
                    if crop_type == 'predicted':
                        score += 0.35
                    elif crop_type == 'player':
                        score += 0.2
                    if area < (width * height) * 0.0012:
                        score += 0.2
                    if predicted_center is not None:
                        score += max(0.0, 1.0 - (_distance(center, predicted_center) / max(width, height)))
                    detection = Detection(
                        track_id='ball',
                        label='ball',
                        bbox=BBox(world_x1, world_y1, world_x2, world_y2),
                        confidence=float(confidence),
                        meta={
                            'learned_ball_detector': True,
                            'learned_ball_crop': crop_type,
                            'learned_ball_score': round(score, 4),
                        },
                    )
                    if best is None or score > best[0]:
                        best = (score, detection)

        return None if best is None else best[1]
