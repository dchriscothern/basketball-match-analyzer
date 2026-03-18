from __future__ import annotations

import os
import site
import sys
from pathlib import Path

from .court_geometry import CourtCalibration, box_is_in_playable_area, estimate_court_calibration
from .schemas import BBox, Detection, TrackingFrame

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    repo_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault('YOLO_CONFIG_DIR', str(repo_root / '.ultralytics'))
    os.environ.setdefault('ULTRALYTICS_CONFIG_DIR', str(repo_root / '.ultralytics'))
    user_site = site.getusersitepackages()
    if isinstance(user_site, str) and user_site and user_site not in sys.path:
        sys.path.append(user_site)
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None


def _center(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class YoloMOTExtractor:
    """Experimental YOLO + MOT backend.

    This keeps the same internal schema as the existing extractors so the rest
    of the pipeline can stay unchanged while we evaluate a stronger MOT path.
    """

    def __init__(self, frame_stride: int = 2, conf: float = 0.18, tracker: str = 'botsort.yaml') -> None:
        self.frame_stride = frame_stride
        self.conf = conf
        self.tracker = tracker
        self._model = None
        self._calibration: CourtCalibration | None = None
        self._available = YOLO is not None and cv2 is not None
        self._last_error: str | None = None

    def is_available(self) -> bool:
        return self._available

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _model_instance(self):
        if self._model is None:
            weights_path = Path(__file__).resolve().parents[2] / 'yolov8n.pt'
            self._model = YOLO(str(weights_path if weights_path.exists() else 'yolov8n.pt'))
        return self._model

    def _signature(self, frame, xyxy: tuple[float, float, float, float]) -> tuple[float, float, float] | None:
        x1, y1, x2, y2 = [int(value) for value in xyxy]
        width = x2 - x1
        height = y2 - y1
        x1 = max(0, x1 + int(width * 0.2))
        x2 = min(frame.shape[1], x2 - int(width * 0.2))
        y1 = max(0, y1 + int(height * 0.12))
        y2 = min(frame.shape[0], max(y1 + 1, y1 + int(height * 0.38)))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 35)
        pixels = crop[mask] if mask.any() else crop.reshape(-1, 3)
        if pixels.size == 0:
            return None
        mean_bgr = pixels.reshape(-1, 3).mean(axis=0)
        return (float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2]))

    def _is_on_court(self, frame, xyxy: tuple[float, float, float, float]) -> bool:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = xyxy
        return box_is_in_playable_area(
            width,
            height,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            calibration=self._calibration,
        )

    def _filter_people(
        self,
        frame,
        candidates: list[tuple[tuple[float, float, float, float], float, int | None]],
    ) -> list[tuple[tuple[float, float, float, float], float, int | None]]:
        if len(candidates) <= 3:
            return candidates

        height, width = frame.shape[:2]
        centers = [_center(box) for box, _, _ in candidates]
        kept: list[tuple[tuple[float, float, float, float], float, int | None]] = []

        for idx, ((x1, y1, x2, y2), confidence, track_id) in enumerate(candidates):
            distances = [
                _distance(centers[idx], other_center)
                for j, other_center in enumerate(centers)
                if j != idx
            ]
            ordered = sorted(distances)
            nearest = ordered[0] if ordered else 999999.0
            second_nearest = ordered[1] if len(ordered) > 1 else 999999.0
            box_height = y2 - y1
            is_foreground = y2 > height * 0.72
            is_deep_top_band = y2 < height * 0.6
            connected_to_cluster = nearest <= width * 0.13 or second_nearest <= width * 0.2
            strong_player_shape = box_height > height * 0.22
            if connected_to_cluster or strong_player_shape or (is_foreground and nearest <= width * 0.22):
                if is_deep_top_band and nearest > width * 0.1 and second_nearest > width * 0.16:
                    continue
                kept.append(((x1, y1, x2, y2), confidence, track_id))

        return kept or candidates

    def run(self, video_path: str | Path) -> list[TrackingFrame]:
        self._last_error = None
        if not self.is_available():
            return []

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return []
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        ok, first_frame = capture.read()
        if ok:
            self._calibration = estimate_court_calibration(first_frame)
        capture.release()

        try:
            model = self._model_instance()
            results = model.track(
                source=str(video_path),
                stream=True,
                tracker=self.tracker,
                classes=[0, 32],
                conf=self.conf,
                iou=0.45,
                persist=True,
                verbose=False,
                device='cpu',
            )
        except Exception as exc:  # pragma: no cover - runtime/backend dependent
            self._last_error = str(exc)
            return []

        frames: list[TrackingFrame] = []
        frame_index = 0

        for result in results:
            if frame_index % self.frame_stride != 0:
                frame_index += 1
                continue

            orig = result.orig_img
            players: list[Detection] = []
            ball = None
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                xyxy_list = boxes.xyxy.cpu().tolist()
                cls_list = boxes.cls.cpu().tolist()
                conf_list = boxes.conf.cpu().tolist()
                id_list = boxes.id.int().cpu().tolist() if getattr(boxes, 'id', None) is not None else [None] * len(xyxy_list)
                player_candidates: list[tuple[tuple[float, float, float, float], float, int | None]] = []
                ball_candidates: list[tuple[tuple[float, float, float, float], float]] = []

                for xyxy, cls_id, confidence, track_id in zip(xyxy_list, cls_list, conf_list, id_list):
                    x1, y1, x2, y2 = xyxy
                    if int(cls_id) == 0:
                        if not self._is_on_court(orig, (x1, y1, x2, y2)):
                            continue
                        player_candidates.append(((x1, y1, x2, y2), float(confidence), track_id))
                    elif int(cls_id) == 32:
                        if not self._is_on_court(orig, (x1, y1, x2, y2)):
                            continue
                        ball_candidates.append(((x1, y1, x2, y2), float(confidence)))

                player_candidates = self._filter_people(orig, player_candidates)
                for (x1, y1, x2, y2), confidence, track_id in player_candidates:
                    track_label = f'mot_p{track_id}' if track_id is not None else f'mot_f{frame_index}_{int(x1)}_{int(y1)}'
                    players.append(
                        Detection(
                            track_id=track_label,
                            label='player',
                            bbox=BBox(x1, y1, x2, y2),
                            confidence=float(confidence),
                            team_id=None,
                            meta={'color_signature': self._signature(orig, (x1, y1, x2, y2))},
                        )
                    )

                for (x1, y1, x2, y2), confidence in ball_candidates:
                    candidate = Detection(
                        track_id='ball',
                        label='ball',
                        bbox=BBox(x1, y1, x2, y2),
                        confidence=float(confidence),
                    )
                    if ball is None or candidate.confidence > ball.confidence:
                        ball = candidate

            frames.append(
                TrackingFrame(
                    frame_index=frame_index,
                    timestamp_s=round(frame_index / fps, 2),
                    players=players,
                    ball=ball,
                )
            )
            frame_index += 1

        return frames
