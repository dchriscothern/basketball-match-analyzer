from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .ball_detector import BallDetector
from .court_geometry import CourtCalibration, box_is_in_playable_area, estimate_court_calibration, estimate_hoop_anchors
from .schemas import BBox, Detection, TrackingFrame

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


def _center(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


@dataclass(slots=True)
class TrackState:
    track_id: str
    center: tuple[float, float]
    misses: int = 0


class YoloCVExtractor:
    def __init__(self, frame_stride: int = 2, conf: float = 0.1) -> None:
        self.frame_stride = frame_stride
        self.conf = conf
        self._model = None
        self._ball_detector = BallDetector()
        self._tracks: list[TrackState] = []
        self._next_track_id = 1
        self._calibration: CourtCalibration | None = None
        self._last_ball_center: tuple[float, float] | None = None
        self._hoop_anchors: dict[str, tuple[float, float]] | None = None
        self._previous_frame = None
        self._ball_missing_streak = 0
        self._consecutive_estimated_frames = 0
        self._frames_since_observed_ball = 0
        self._estimated_window: list[bool] = []
        self._last_ball_owner_track_id: str | None = None
        self._last_ball_relative_offset: tuple[float, float] | None = None
        self._last_ball_velocity: tuple[float, float] | None = None

    def is_available(self) -> bool:
        return YOLO is not None and cv2 is not None

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

    def _assign_track(self, xyxy: tuple[float, float, float, float]) -> str:
        center = _center(xyxy)
        best_track = None
        best_distance = 999999.0
        for track in self._tracks:
            dist = _distance(track.center, center)
            if dist < 100 and dist < best_distance:
                best_distance = dist
                best_track = track
        if best_track is None:
            track_id = f'yt_p{self._next_track_id}'
            self._next_track_id += 1
            self._tracks.append(TrackState(track_id=track_id, center=center, misses=0))
            return track_id
        best_track.center = center
        best_track.misses = 0
        return best_track.track_id

    def _filter_isolated_people(
        self,
        frame,
        candidates: list[tuple[tuple[float, float, float, float], float]],
    ) -> list[tuple[tuple[float, float, float, float], float]]:
        if len(candidates) <= 3:
            return candidates

        height, width = frame.shape[:2]
        kept: list[tuple[tuple[float, float, float, float], float]] = []
        centers = [_center(box) for box, _ in candidates]

        for idx, ((x1, y1, x2, y2), confidence) in enumerate(candidates):
            cx, cy = centers[idx]
            distances = [
                _distance((cx, cy), other_center)
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
                kept.append(((x1, y1, x2, y2), confidence))

        return kept or candidates

    def _rescue_player_candidates(
        self,
        frame,
        original_candidates: list[tuple[tuple[float, float, float, float], float]],
        filtered_candidates: list[tuple[tuple[float, float, float, float], float]],
    ) -> list[tuple[tuple[float, float, float, float], float]]:
        if len(filtered_candidates) >= 7:
            return filtered_candidates

        rescued = list(filtered_candidates)
        rescued_boxes = {tuple(round(value, 1) for value in box) for box, _ in rescued}
        height, width = frame.shape[:2]
        for box, confidence in sorted(original_candidates, key=lambda item: item[1], reverse=True):
            box_key = tuple(round(value, 1) for value in box)
            if box_key in rescued_boxes:
                continue
            x1, y1, x2, y2 = box
            box_height = y2 - y1
            box_width = x2 - x1
            if box_height < height * 0.09:
                continue
            if box_width > box_height * 1.18:
                continue
            if y2 < height * 0.42:
                continue
            if confidence < max(0.08, self.conf * 0.72):
                continue
            rescued.append((box, confidence))
            rescued_boxes.add(box_key)
            if len(rescued) >= 9:
                break
        return rescued

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

    def _bbox_motion_intensity(self, frame, x: int, y: int, w: int, h: int) -> float:
        if self._previous_frame is None:
            return 0.0
        pad = 3
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        prev_gray = cv2.cvtColor(self._previous_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        if prev_gray.size == 0 or curr_gray.size == 0:
            return 0.0
        diff = cv2.absdiff(prev_gray, curr_gray)
        return float(diff.mean() / 255.0)

    def _player_ball_proximity_score(self, player: Detection, point: tuple[float, float]) -> float:
        bbox = player.bbox
        width = bbox.x2 - bbox.x1
        height = bbox.y2 - bbox.y1
        control_anchor = ((bbox.x1 + bbox.x2) / 2.0, bbox.y1 + (height * 0.42))
        anchor_distance = _distance(control_anchor, point)
        bbox_distance = self._distance_to_player_box(point, player)
        contains_ball = (
            bbox.x1 - width * 0.08 <= point[0] <= bbox.x2 + width * 0.08
            and bbox.y1 - height * 0.08 <= point[1] <= bbox.y2 + height * 0.12
        )
        score = min(anchor_distance, bbox_distance + 8.0)
        if contains_ball:
            score -= 16.0
        return score

    def _predicted_ball_center(self) -> tuple[float, float] | None:
        if self._last_ball_center is None:
            return None
        if self._last_ball_velocity is None:
            return self._last_ball_center
        return (
            self._last_ball_center[0] + self._last_ball_velocity[0],
            self._last_ball_center[1] + self._last_ball_velocity[1],
        )

    def _enrich_ball_metadata(self, ball: Detection) -> Detection:
        meta = dict(ball.meta or {})
        if self._hoop_anchors is not None:
            center = ball.bbox.center
            left_distance = _distance(center, self._hoop_anchors['left'])
            right_distance = _distance(center, self._hoop_anchors['right'])
            target_hoop = 'left' if left_distance <= right_distance else 'right'
            meta.update(
                {
                    'target_hoop': target_hoop,
                    'target_hoop_xy': self._hoop_anchors[target_hoop],
                    'hoop_distance_px': min(left_distance, right_distance),
                }
            )
        ball.meta = meta
        return ball

    def _record_ball_context(self, ball: Detection, players: list[Detection]) -> None:
        previous_center = self._last_ball_center
        self._last_ball_center = ball.bbox.center
        self._ball_missing_streak = 0
        if previous_center is not None:
            velocity = (
                self._last_ball_center[0] - previous_center[0],
                self._last_ball_center[1] - previous_center[1],
            )
            if self._last_ball_velocity is None:
                self._last_ball_velocity = velocity
            else:
                self._last_ball_velocity = (
                    self._last_ball_velocity[0] * 0.45 + velocity[0] * 0.55,
                    self._last_ball_velocity[1] * 0.45 + velocity[1] * 0.55,
                )
        if not players:
            return
        owner = min(players, key=lambda player: self._player_ball_proximity_score(player, ball.bbox.center))
        if self._player_ball_proximity_score(owner, ball.bbox.center) > max(48.0, (owner.bbox.x2 - owner.bbox.x1) * 0.9):
            return
        owner_center = ((owner.bbox.x1 + owner.bbox.x2) / 2.0, owner.bbox.y1 + (owner.bbox.y2 - owner.bbox.y1) * 0.58)
        self._last_ball_owner_track_id = owner.track_id
        self._last_ball_relative_offset = (
            ball.bbox.center[0] - owner_center[0],
            ball.bbox.center[1] - owner_center[1],
        )

    def _ball_fallback(self, frame, players: list[Detection]) -> Detection | None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = (4, 100, 90)
        upper = (25, 255, 255)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, float, float, float, float, float, float]] = []
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
            if not self._is_on_court(frame, (x, y, x + w, y + h)):
                continue
            center = (x + w / 2.0, y + h / 2.0)
            nearest_player_distance = self._nearest_player_distance(center, players)
            continuity_distance = _distance(center, self._last_ball_center) if self._last_ball_center is not None else None
            predicted_center = self._predicted_ball_center()
            predicted_distance = _distance(center, predicted_center) if predicted_center is not None else None
            motion_intensity = self._bbox_motion_intensity(frame, x, y, w, h)
            if nearest_player_distance > frame.shape[1] * 0.22 and continuity_distance is None:
                continue
            if motion_intensity < 0.018 and continuity_distance is None:
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
            if predicted_distance is not None:
                if predicted_distance <= frame.shape[1] * 0.05:
                    score += 1.0
                elif predicted_distance <= frame.shape[1] * 0.1:
                    score += 0.4
                else:
                    score -= 0.55
            if motion_intensity >= 0.08:
                score += 0.9
            elif motion_intensity >= 0.04:
                score += 0.45
            else:
                score -= 0.45

            candidates.append((score, x, y, w, h, nearest_player_distance, motion_intensity))

        if not candidates:
            self._ball_missing_streak += 1
            if self._ball_missing_streak > 4:
                self._last_ball_center = None
            return None

        score, x, y, w, h, _, motion_intensity = max(candidates, key=lambda item: item[0])
        self._last_ball_center = (x + w / 2.0, y + h / 2.0)
        self._ball_missing_streak = 0
        confidence = max(0.24, min(0.58, 0.3 + (score * 0.08)))
        meta = {
            'fallback_ball': True,
            'ball_score': round(score, 3),
            'motion_intensity': round(motion_intensity, 3),
        }
        if self._hoop_anchors is not None:
            center = self._last_ball_center
            left_distance = _distance(center, self._hoop_anchors['left'])
            right_distance = _distance(center, self._hoop_anchors['right'])
            target_hoop = 'left' if left_distance <= right_distance else 'right'
            meta.update(
                {
                    'target_hoop': target_hoop,
                    'target_hoop_xy': self._hoop_anchors[target_hoop],
                    'hoop_distance_px': min(left_distance, right_distance),
                }
            )
        return Detection(
            track_id='ball',
            label='ball',
            bbox=BBox(x, y, x + w, y + h),
            confidence=confidence,
            meta=meta,
        )

    def _motion_ball_fallback(self, frame, players: list[Detection]) -> Detection | None:
        if self._previous_frame is None:
            return None

        prev_gray = cv2.cvtColor(self._previous_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, mask = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, int, int, int, int]] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 6 or area > 240:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0 or w == 0:
                continue
            aspect = w / h
            if not 0.5 <= aspect <= 1.6:
                continue
            if not self._is_on_court(frame, (x, y, x + w, y + h)):
                continue

            center = (x + w / 2.0, y + h / 2.0)
            nearest_player_distance = self._nearest_player_distance(center, players)
            continuity_distance = _distance(center, self._last_ball_center) if self._last_ball_center is not None else None
            predicted_center = self._predicted_ball_center()
            predicted_distance = _distance(center, predicted_center) if predicted_center is not None else None
            score = min(area / 60.0, 1.6)
            if nearest_player_distance <= frame.shape[1] * 0.08:
                score += 0.8
            elif nearest_player_distance <= frame.shape[1] * 0.16:
                score += 0.35
            else:
                score -= 0.35

            if continuity_distance is not None:
                if continuity_distance <= frame.shape[1] * 0.05:
                    score += 1.2
                elif continuity_distance <= frame.shape[1] * 0.12:
                    score += 0.45
                else:
                    score -= 0.9
            if predicted_distance is not None:
                if predicted_distance <= frame.shape[1] * 0.05:
                    score += 0.9
                elif predicted_distance <= frame.shape[1] * 0.1:
                    score += 0.35
                else:
                    score -= 0.5

            candidates.append((score, x, y, w, h))

        if not candidates:
            return None

        _, x, y, w, h = max(candidates, key=lambda item: item[0])
        self._last_ball_center = (x + w / 2.0, y + h / 2.0)
        self._ball_missing_streak = 0
        meta = {'motion_ball': True}
        if self._hoop_anchors is not None:
            center = self._last_ball_center
            left_distance = _distance(center, self._hoop_anchors['left'])
            right_distance = _distance(center, self._hoop_anchors['right'])
            target_hoop = 'left' if left_distance <= right_distance else 'right'
            meta.update(
                {
                    'target_hoop': target_hoop,
                    'target_hoop_xy': self._hoop_anchors[target_hoop],
                    'hoop_distance_px': min(left_distance, right_distance),
                }
            )
        return Detection(
            track_id='ball',
            label='ball',
            bbox=BBox(x, y, x + w, y + h),
            confidence=0.45,
            meta=meta,
        )

    def _player_ball_estimate(self, frame, players: list[Detection]) -> Detection | None:
        if self._last_ball_center is None or not players:
            return None

        target_player = None
        if self._last_ball_owner_track_id is not None:
            for player in players:
                if player.track_id == self._last_ball_owner_track_id:
                    target_player = player
                    break
        if target_player is None:
            target_player = min(
                players,
                key=lambda player: self._player_ball_proximity_score(player, self._last_ball_center),
            )
        player_score = self._player_ball_proximity_score(target_player, self._last_ball_center)
        if player_score > frame.shape[1] * 0.08:
            return None

        bbox = target_player.bbox
        ball_size = max(8.0, min((bbox.x2 - bbox.x1) * 0.18, 18.0))
        owner_anchor = ((bbox.x1 + bbox.x2) / 2.0, bbox.y1 + (bbox.y2 - bbox.y1) * 0.58)
        if self._last_ball_relative_offset is not None:
            offset_x, offset_y = self._last_ball_relative_offset
            offset_x = max(-(bbox.x2 - bbox.x1) * 0.45, min((bbox.x2 - bbox.x1) * 0.45, offset_x))
            offset_y = max(-(bbox.y2 - bbox.y1) * 0.18, min((bbox.y2 - bbox.y1) * 0.18, offset_y))
            center_x = owner_anchor[0] + offset_x
            center_y = owner_anchor[1] + offset_y
        else:
            side = -1.0 if self._last_ball_center[0] < owner_anchor[0] else 1.0
            center_x = owner_anchor[0] + side * max(ball_size * 0.9, (bbox.x2 - bbox.x1) * 0.22)
            center_y = owner_anchor[1]
        predicted_center = self._predicted_ball_center()
        if predicted_center is not None:
            center_x = center_x * 0.62 + predicted_center[0] * 0.38
            center_y = center_y * 0.62 + predicted_center[1] * 0.38
        x1 = max(0.0, center_x - ball_size / 2.0)
        y1 = max(0.0, center_y - ball_size / 2.0)
        x2 = min(frame.shape[1] - 1.0, center_x + ball_size / 2.0)
        y2 = min(frame.shape[0] - 1.0, center_y + ball_size / 2.0)
        self._last_ball_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self._ball_missing_streak = 0

        meta = {
            'estimated_ball': True,
            'estimated_from_player': target_player.track_id,
        }
        if self._hoop_anchors is not None:
            center = self._last_ball_center
            left_distance = _distance(center, self._hoop_anchors['left'])
            right_distance = _distance(center, self._hoop_anchors['right'])
            target_hoop = 'left' if left_distance <= right_distance else 'right'
            meta.update(
                {
                    'target_hoop': target_hoop,
                    'target_hoop_xy': self._hoop_anchors[target_hoop],
                    'hoop_distance_px': min(left_distance, right_distance),
                }
            )
        return Detection(
            track_id='ball',
            label='ball',
            bbox=BBox(x1, y1, x2, y2),
            confidence=0.38,
            meta=meta,
        )

    def _choose_ball_candidate(
        self,
        *,
        frame,
        players: list[Detection],
        primary: Detection | None,
        alternatives: list[Detection | None],
    ) -> Detection | None:
        if primary is None:
            best_alternative = None
            best_score = -999999.0
            predicted_center = self._predicted_ball_center()
            for alternative in alternatives:
                if alternative is not None:
                    distance_to_player = self._nearest_player_distance(alternative.bbox.center, players)
                    predicted_distance = (
                        _distance(alternative.bbox.center, predicted_center)
                        if predicted_center is not None
                        else 0.0
                    )
                    score = -distance_to_player - predicted_distance * 0.35 + alternative.confidence * 45.0
                    if score > best_score:
                        best_score = score
                        best_alternative = alternative
            return best_alternative

        primary_distance = self._nearest_player_distance(primary.bbox.center, players)
        continuity_distance = (
            _distance(primary.bbox.center, self._last_ball_center)
            if self._last_ball_center is not None
            else 0.0
        )
        primary_meta = primary.meta or {}
        primary_quality = float(primary_meta.get('ball_score', primary.confidence))
        primary_motion = float(primary_meta.get('motion_intensity', 0.0))
        predicted_center = self._predicted_ball_center()
        primary_predicted_distance = (
            _distance(primary.bbox.center, predicted_center)
            if predicted_center is not None
            else 0.0
        )
        for alternative in alternatives:
            if alternative is None:
                continue
            alternative_distance = self._nearest_player_distance(alternative.bbox.center, players)
            alternative_continuity = (
                _distance(alternative.bbox.center, self._last_ball_center)
                if self._last_ball_center is not None
                else 0.0
            )
            alternative_predicted_distance = (
                _distance(alternative.bbox.center, predicted_center)
                if predicted_center is not None
                else 0.0
            )
            if (
                alternative_distance + 8.0 < primary_distance
                and alternative_continuity <= continuity_distance + frame.shape[1] * 0.04
                and alternative_predicted_distance <= primary_predicted_distance + frame.shape[1] * 0.02
            ):
                return alternative
            if (
                primary_meta.get('fallback_ball')
                and primary_motion < 0.035
                and alternative.meta.get('estimated_ball')
                and alternative_distance <= primary_distance + 18.0
            ):
                return alternative
            if (
                primary_meta.get('fallback_ball')
                and primary_quality < 1.2
                and alternative.meta.get('estimated_ball')
                and alternative_predicted_distance <= primary_predicted_distance + frame.shape[1] * 0.06
            ):
                return alternative
            if (
                primary_meta.get('fallback_ball')
                and alternative.meta.get('estimated_ball')
                and self._last_ball_owner_track_id is not None
                and alternative.meta.get('estimated_from_player') == self._last_ball_owner_track_id
                and primary_distance > frame.shape[1] * 0.07
                and alternative_distance <= primary_distance + 12.0
            ):
                return alternative
        return primary

    def _decay_tracks(self, seen_ids: set[str]) -> None:
        kept: list[TrackState] = []
        for track in self._tracks:
            if track.track_id not in seen_ids:
                track.misses += 1
            if track.misses <= 6:
                kept.append(track)
        self._tracks = kept[-30:]

    def run(self, video_path: str | Path) -> list[TrackingFrame]:
        if not self.is_available():
            return []

        self._tracks = []
        self._next_track_id = 1
        self._calibration = None
        self._last_ball_center = None
        self._hoop_anchors = None
        self._previous_frame = None
        self._ball_missing_streak = 0
        self._consecutive_estimated_frames = 0
        self._frames_since_observed_ball = 0
        self._estimated_window = []
        self._last_ball_owner_track_id = None
        self._last_ball_relative_offset = None
        self._last_ball_velocity = None
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return []
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        ok, first_frame = capture.read()
        if ok:
            self._calibration = estimate_court_calibration(first_frame)
            self._hoop_anchors = estimate_hoop_anchors(first_frame.shape[1], first_frame.shape[0], self._calibration)
        capture.release()

        model = self._model_instance()
        frames: list[TrackingFrame] = []
        frame_index = 0

        results = model.predict(
            source=str(video_path),
            stream=True,
            classes=[0, 32],
            conf=self.conf,
            iou=0.35,
            imgsz=1280,
            verbose=False,
            device='cpu',
        )

        for result in results:
            if frame_index % self.frame_stride != 0:
                frame_index += 1
                continue

            orig = result.orig_img
            players: list[Detection] = []
            ball = None
            seen_ids: set[str] = set()
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                xyxy_list = boxes.xyxy.cpu().tolist()
                cls_list = boxes.cls.cpu().tolist()
                conf_list = boxes.conf.cpu().tolist()
                player_candidates: list[tuple[tuple[float, float, float, float], float]] = []
                ball_candidates: list[tuple[tuple[float, float, float, float], float]] = []

                for xyxy, cls_id, confidence in zip(xyxy_list, cls_list, conf_list):
                    x1, y1, x2, y2 = xyxy
                    if int(cls_id) == 0:
                        if not self._is_on_court(orig, (x1, y1, x2, y2)):
                            continue
                        player_candidates.append(((x1, y1, x2, y2), float(confidence)))
                    elif int(cls_id) == 32:
                        # Ball can be in flight above the playable area or near hoop edges.
                        # Use a relaxed frame-boundary check instead of full court membership.
                        _bh, _bw = orig.shape[:2]
                        _bcx = (x1 + x2) / 2.0
                        _bcy = (y1 + y2) / 2.0
                        if _bcy < _bh * 0.12 or _bcy > _bh * 0.97:
                            continue
                        if _bcx < _bw * 0.02 or _bcx > _bw * 0.98:
                            continue
                        ball_candidates.append(((x1, y1, x2, y2), float(confidence)))

                original_player_candidates = list(player_candidates)
                player_candidates = self._filter_isolated_people(orig, player_candidates)
                player_candidates = self._rescue_player_candidates(orig, original_player_candidates, player_candidates)
                for (x1, y1, x2, y2), confidence in player_candidates:
                    track_id = self._assign_track((x1, y1, x2, y2))
                    seen_ids.add(track_id)
                    players.append(
                        Detection(
                            track_id=track_id,
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
                        meta={},
                    )
                    candidate = self._enrich_ball_metadata(candidate)
                    if ball is None or candidate.confidence > ball.confidence:
                        ball = candidate

                learned_ball = self._ball_detector.detect(
                    orig,
                    predicted_center=self._predicted_ball_center(),
                    player_boxes=[player.bbox for player in players],
                )
                if learned_ball is not None:
                    learned_ball = self._enrich_ball_metadata(learned_ball)
                fallback_ball = self._ball_fallback(orig, players) if ball is None else None
                motion_ball = self._motion_ball_fallback(orig, players) if ball is None else None
                estimated_ball = (
                    self._player_ball_estimate(orig, players)
                    if ball is None and len(players) >= 4
                    else None
                )
                ball = self._choose_ball_candidate(
                    frame=orig,
                    players=players,
                    primary=learned_ball or ball or fallback_ball,
                    alternatives=[motion_ball, estimated_ball, fallback_ball if learned_ball is not None else None],
                )
                if ball is not None:
                    ball_meta = ball.meta or {}
                    is_estimated = ball_meta.get('estimated_ball') or ball_meta.get('fallback_ball')
                    is_learned = ball_meta.get('learned_ball_detector')
                    if not is_learned and not is_estimated and ball.confidence < 0.18:
                        ball = None
                if ball is not None:
                    ball_meta = ball.meta or {}
                    if ball_meta.get('estimated_ball'):
                        self._consecutive_estimated_frames += 1
                        print(f"ESTIMATED_BALL frame={frame_index} consecutive={self._consecutive_estimated_frames}")
                        self._estimated_window.append(True)
                        if len(self._estimated_window) > 20:
                            self._estimated_window.pop(0)
                        if self._consecutive_estimated_frames > 4 or sum(self._estimated_window) > 8:
                            ball = None
                            self._estimated_window[-1] = False
                            self._consecutive_estimated_frames = 0
                    else:
                        self._consecutive_estimated_frames = 0
                        self._estimated_window.append(False)
                        if len(self._estimated_window) > 20:
                            self._estimated_window.pop(0)
                else:
                    self._consecutive_estimated_frames = 0
                    self._estimated_window.append(False)
                    if len(self._estimated_window) > 20:
                        self._estimated_window.pop(0)
                if ball is not None:
                    self._record_ball_context(ball, players)

            self._decay_tracks(seen_ids)
            frames.append(
                TrackingFrame(
                    frame_index=frame_index,
                    timestamp_s=round(frame_index / fps, 2),
                    players=players,
                    ball=ball,
                )
            )
            self._previous_frame = orig.copy()
            frame_index += 1

        _ball_count = sum(1 for _f in frames if _f.ball is not None)
        _learned_ball_count = sum(
            1
            for _f in frames
            if _f.ball is not None and (_f.ball.meta or {}).get('learned_ball_detector')
        )
        _fallback_ball_count = sum(
            1
            for _f in frames
            if _f.ball is not None and (_f.ball.meta or {}).get('fallback_ball')
        )
        _motion_ball_count = sum(
            1
            for _f in frames
            if _f.ball is not None and (_f.ball.meta or {}).get('motion_ball')
        )
        _estimated_ball_count = sum(
            1
            for _f in frames
            if _f.ball is not None and (_f.ball.meta or {}).get('estimated_ball')
        )
        _generic_ball_count = max(
            0,
            _ball_count - _learned_ball_count - _fallback_ball_count - _motion_ball_count - _estimated_ball_count,
        )
        print(
            f"YoloCVExtractor: {len(frames)} frames processed | "
            f"final ball present in {_ball_count} "
            f"({100.0 * _ball_count / len(frames) if frames else 0.0:.1f}%) | "
            f"learned={_learned_ball_count}, generic_yolo={_generic_ball_count}, "
            f"fallback={_fallback_ball_count}, motion={_motion_ball_count}, "
            f"estimated={_estimated_ball_count}"
        )
        return frames
