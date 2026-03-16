from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .schemas import BBox, Detection, TrackingFrame

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    cv2 = None
    np = None


@dataclass(slots=True)
class TrackState:
    track_id: str
    center: tuple[float, float]


class PrototypeCVExtractor:
    """Very early-stage CV extractor for raw MP4 inputs.

    This is not production tracking. It is a best-effort prototype that:
    - samples frames sparsely
    - detects an orange-ish ball candidate via HSV thresholding
    - detects moving player blobs via background subtraction
    - assigns simple track IDs via nearest-neighbor matching
    - uses left/right court halves as a temporary team proxy
    """

    def __init__(self, frame_stride: int = 5, max_players: int = 10) -> None:
        self.frame_stride = frame_stride
        self.max_players = max_players
        self._next_track_id = 1
        self._previous_tracks: list[TrackState] = []

    def is_available(self) -> bool:
        return cv2 is not None and np is not None

    def _assign_track_id(self, center: tuple[float, float]) -> str:
        best_idx = None
        best_distance = 999999.0
        for idx, track in enumerate(self._previous_tracks):
            dist = ((track.center[0] - center[0]) ** 2 + (track.center[1] - center[1]) ** 2) ** 0.5
            if dist < best_distance:
                best_distance = dist
                best_idx = idx

        if best_idx is not None and best_distance < 65:
            track = self._previous_tracks.pop(best_idx)
            self._previous_tracks.append(TrackState(track.track_id, center))
            return track.track_id

        track_id = f'cv_p{self._next_track_id}'
        self._next_track_id += 1
        self._previous_tracks.append(TrackState(track_id, center))
        self._previous_tracks = self._previous_tracks[-25:]
        return track_id

    def _detect_ball(self, frame) -> Detection | None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([5, 120, 120])
        upper = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if 20 <= cv2.contourArea(c) <= 2000]
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        return Detection(track_id='ball', label='ball', bbox=BBox(x, y, x + w, y + h), confidence=0.5)

    def _detect_players(self, frame, fg_mask) -> list[Detection]:
        kernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = frame.shape[:2]
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 250 or area > 8000:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if h < 20 or w < 8:
                continue
            center = (x + w / 2.0, y + h / 2.0)
            track_id = self._assign_track_id(center)
            team_id = 'home' if center[0] < (width / 2.0) else 'away'
            candidates.append(
                Detection(
                    track_id=track_id,
                    label='player',
                    bbox=BBox(x, y, x + w, y + h),
                    confidence=0.35,
                    team_id=team_id,
                )
            )
        candidates.sort(key=lambda player: player.bbox.y2 - player.bbox.y1, reverse=True)
        return candidates[: self.max_players]

    def run(self, video_path: str | Path) -> list[TrackingFrame]:
        if not self.is_available():
            return []

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return []

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=False)
        frames: list[TrackingFrame] = []
        frame_index = 0
        sampled_index = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % self.frame_stride != 0:
                frame_index += 1
                continue

            fg_mask = subtractor.apply(frame)
            players = self._detect_players(frame, fg_mask)
            ball = self._detect_ball(frame)
            frames.append(
                TrackingFrame(
                    frame_index=sampled_index,
                    timestamp_s=round(frame_index / fps, 2),
                    players=players,
                    ball=ball,
                )
            )
            sampled_index += 1
            frame_index += 1

        capture.release()
        return frames
