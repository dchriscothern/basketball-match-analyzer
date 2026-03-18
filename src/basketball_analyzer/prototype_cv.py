from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .court_geometry import CourtCalibration, bbox_tuple_is_in_playable_area, estimate_court_calibration
from .schemas import BBox, Detection, TrackingFrame

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    cv2 = None
    np = None


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


@dataclass(slots=True)
class TrackState:
    track_id: str
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    last_seen_frame: int
    misses: int = 0
    color_signature: tuple[float, float, float] | None = None


class PrototypeCVExtractor:
    """Improved-but-still-MVP CV extractor for raw MP4 inputs.

    Strategy:
    - sample frames at a moderate stride
    - fuse motion blobs with OpenCV's built-in HOG people detector
    - track players greedily across frames
    - estimate jersey color from the torso crop
    - detect the ball using orange-color and circularity heuristics
    """

    def __init__(self, frame_stride: int = 2, max_players: int = 10) -> None:
        self.frame_stride = frame_stride
        self.max_players = max_players
        self._next_track_id = 1
        self._tracks: list[TrackState] = []
        self._calibration: CourtCalibration | None = None
        self._last_ball_center: tuple[float, float] | None = None
        self._hog = None
        if cv2 is not None:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def is_available(self) -> bool:
        return cv2 is not None and np is not None

    def _upper_body_signature(self, frame, box: tuple[int, int, int, int]) -> tuple[float, float, float] | None:
        x, y, w, h = box
        x1 = max(0, x + int(w * 0.2))
        y1 = max(0, y + int(h * 0.12))
        x2 = min(frame.shape[1], x + w - int(w * 0.2))
        y2 = min(frame.shape[0], y1 + max(10, int(h * 0.38)))
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

    def _motion_boxes(self, frame, subtractor) -> list[tuple[int, int, int, int]]:
        fg_mask = subtractor.apply(frame)
        kernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        clean = cv2.dilate(clean, kernel, iterations=2)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 180 or area > 15000:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if h < 24 or w < 10:
                continue
            if h < w:
                continue
            if not bbox_tuple_is_in_playable_area(frame.shape, (x, y, w, h), calibration=self._calibration):
                continue
            boxes.append((x, y, w, h))
        return boxes

    def _hog_boxes(self, frame) -> list[tuple[int, int, int, int]]:
        if self._hog is None:
            return []
        resized = cv2.resize(frame, None, fx=0.75, fy=0.75)
        rects, _ = self._hog.detectMultiScale(
            resized,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        boxes: list[tuple[int, int, int, int]] = []
        for x, y, w, h in rects:
            box = (int(x / 0.75), int(y / 0.75), int(w / 0.75), int(h / 0.75))
            if bbox_tuple_is_in_playable_area(frame.shape, box, calibration=self._calibration):
                boxes.append(box)
        return boxes

    def _merge_boxes(self, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda item: item[2] * item[3], reverse=True)
        merged: list[tuple[int, int, int, int]] = []
        for candidate in boxes:
            if any(_iou(candidate, existing) > 0.35 for existing in merged):
                continue
            merged.append(candidate)
        return merged[: self.max_players]

    def _filter_isolated_boxes(self, frame, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        if len(boxes) <= 3:
            return boxes

        height, width = frame.shape[:2]
        centers = [_center(box) for box in boxes]
        kept: list[tuple[int, int, int, int]] = []

        for idx, box in enumerate(boxes):
            x, y, w, h = box
            cx, cy = centers[idx]
            distances = [
                _distance((cx, cy), other_center)
                for j, other_center in enumerate(centers)
                if j != idx
            ]
            ordered = sorted(distances)
            nearest = ordered[0] if ordered else 999999.0
            second_nearest = ordered[1] if len(ordered) > 1 else 999999.0
            is_foreground = (y + h) > height * 0.72
            is_deep_top_band = (y + h) < height * 0.6
            connected_to_cluster = nearest <= width * 0.13 or second_nearest <= width * 0.2
            strong_player_shape = h > height * 0.22
            if connected_to_cluster or strong_player_shape or (is_foreground and nearest <= width * 0.22):
                if is_deep_top_band and nearest > width * 0.1 and second_nearest > width * 0.16:
                    continue
                kept.append(box)

        return kept or boxes

    def _update_tracks(self, frame, boxes: list[tuple[int, int, int, int]], frame_index: int) -> list[Detection]:
        detections: list[Detection] = []
        unmatched_tracks = list(range(len(self._tracks)))

        for box in boxes:
            center = _center(box)
            best_track_idx = None
            best_distance = 999999.0

            for idx in unmatched_tracks:
                track = self._tracks[idx]
                dist = _distance(track.center, center)
                overlap = _iou(track.bbox, box)
                if dist < 90 and overlap > 0.05 and dist < best_distance:
                    best_distance = dist
                    best_track_idx = idx

            color_signature = self._upper_body_signature(frame, box)
            if best_track_idx is None:
                track_id = f'cv_p{self._next_track_id}'
                self._next_track_id += 1
                track = TrackState(
                    track_id=track_id,
                    center=center,
                    bbox=box,
                    last_seen_frame=frame_index,
                    misses=0,
                    color_signature=color_signature,
                )
                self._tracks.append(track)
            else:
                track = self._tracks[best_track_idx]
                track.center = center
                track.bbox = box
                track.last_seen_frame = frame_index
                track.misses = 0
                if color_signature is not None:
                    track.color_signature = color_signature
                unmatched_tracks.remove(best_track_idx)

            x, y, w, h = box
            detections.append(
                Detection(
                    track_id=track.track_id,
                    label='player',
                    bbox=BBox(x, y, x + w, y + h),
                    confidence=0.55,
                    team_id=None,
                    meta={'color_signature': track.color_signature},
                )
            )

        kept_tracks: list[TrackState] = []
        seen_ids = {detection.track_id for detection in detections}
        for track in self._tracks:
            if track.track_id not in seen_ids:
                track.misses += 1
            if track.misses <= 5:
                kept_tracks.append(track)
        self._tracks = kept_tracks[-25:]
        return detections

    def _distance_to_player_box(self, point: tuple[float, float], player: Detection) -> float:
        px, py = point
        bbox = player.bbox
        dx = max(bbox.x1 - px, 0.0, px - bbox.x2)
        dy = max(bbox.y1 - py, 0.0, py - bbox.y2)
        return (dx * dx + dy * dy) ** 0.5

    def _nearest_player_distance(self, point: tuple[float, float], players: list[Detection]) -> float:
        if not players:
            return 999999.0
        return min(self._distance_to_player_box(point, player) for player in players)

    def _detect_ball(self, frame, players: list[Detection]) -> Detection | None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([4, 100, 90])
        upper = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 8 or area > 900:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = 0.0 if perimeter == 0 else (4 * 3.14159 * area) / (perimeter * perimeter)
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0 or w == 0:
                continue
            aspect = w / h
            if not 0.7 <= aspect <= 1.35:
                continue
            if not bbox_tuple_is_in_playable_area(frame.shape, (x, y, w, h), calibration=self._calibration):
                continue
            center = (x + w / 2.0, y + h / 2.0)
            nearest_player_distance = self._nearest_player_distance(center, players)
            continuity_distance = _distance(center, self._last_ball_center) if self._last_ball_center is not None else None

            if nearest_player_distance > frame.shape[1] * 0.22 and continuity_distance is None:
                continue

            score = circularity + min(area / 180.0, 1.0)
            if nearest_player_distance <= frame.shape[1] * 0.08:
                score += 1.0
            elif nearest_player_distance <= frame.shape[1] * 0.16:
                score += 0.45
            else:
                score -= 0.6

            if continuity_distance is not None:
                if continuity_distance <= frame.shape[1] * 0.06:
                    score += 1.1
                elif continuity_distance <= frame.shape[1] * 0.12:
                    score += 0.45
                else:
                    score -= 0.8

            candidates.append((score, x, y, w, h))
        if not candidates:
            self._last_ball_center = None
            return None
        _, x, y, w, h = max(candidates, key=lambda item: item[0])
        self._last_ball_center = (x + w / 2.0, y + h / 2.0)
        return Detection(
            track_id='ball',
            label='ball',
            bbox=BBox(x, y, x + w, y + h),
            confidence=0.55,
        )

    def run(self, video_path: str | Path) -> list[TrackingFrame]:
        if not self.is_available():
            return []

        self._tracks = []
        self._next_track_id = 1
        self._calibration = None
        self._last_ball_center = None
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return []

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        ok, first_frame = capture.read()
        if ok:
            self._calibration = estimate_court_calibration(first_frame)
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=24, detectShadows=False)
        frames: list[TrackingFrame] = []
        frame_index = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % self.frame_stride != 0:
                frame_index += 1
                continue

            motion_boxes = self._motion_boxes(frame, subtractor)
            hog_boxes = self._hog_boxes(frame)
            boxes = self._merge_boxes(motion_boxes + hog_boxes)
            boxes = self._filter_isolated_boxes(frame, boxes)
            players = self._update_tracks(frame, boxes, frame_index)
            ball = self._detect_ball(frame, players)
            frames.append(
                TrackingFrame(
                    frame_index=frame_index,
                    timestamp_s=round(frame_index / fps, 2),
                    players=players,
                    ball=ball,
                )
            )
            frame_index += 1

        capture.release()
        return frames
