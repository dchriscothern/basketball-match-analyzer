from __future__ import annotations

import math
from copy import deepcopy

from .base import BaseAgent
from ..schemas import BBox, Detection, TrackingFrame


class BallInterpolator(BaseAgent):
    name = "ball_interpolator"

    def _distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.dist(a, b)

    def run(self, frames: list[TrackingFrame]) -> list[TrackingFrame]:
        filtered_known: list[tuple[int, Detection]] = []
        max_jump = 140.0
        for idx, frame in enumerate(frames):
            if frame.ball is None:
                continue
            if filtered_known:
                _, prev_ball = filtered_known[-1]
                if self._distance(prev_ball.bbox.center, frame.ball.bbox.center) > max_jump:
                    frame.ball = None
                    continue
            filtered_known.append((idx, frame.ball))

        missing = [idx for idx, frame in enumerate(frames) if frame.ball is None]
        if not missing:
            return frames

        known = [(idx, frame.ball) for idx, frame in enumerate(frames) if frame.ball is not None]
        if len(known) < 2:
            return frames

        for gap_idx in missing:
            left = max((item for item in known if item[0] < gap_idx), default=None, key=lambda item: item[0])
            right = min((item for item in known if item[0] > gap_idx), default=None, key=lambda item: item[0])
            if not left or not right:
                continue

            left_idx, left_ball = left
            right_idx, right_ball = right
            span = right_idx - left_idx
            if span <= 1:
                continue
            alpha = (gap_idx - left_idx) / span
            lb = left_ball.bbox
            rb = right_ball.bbox
            bbox = BBox(
                x1=lb.x1 + (rb.x1 - lb.x1) * alpha,
                y1=lb.y1 + (rb.y1 - lb.y1) * alpha,
                x2=lb.x2 + (rb.x2 - lb.x2) * alpha,
                y2=lb.y2 + (rb.y2 - lb.y2) * alpha,
            )
            frames[gap_idx].ball = Detection(
                track_id=left_ball.track_id,
                label=left_ball.label,
                bbox=bbox,
                confidence=min(left_ball.confidence, right_ball.confidence),
                team_id=left_ball.team_id or right_ball.team_id,
                meta={
                    **deepcopy(left_ball.meta),
                    **deepcopy(right_ball.meta),
                    "interpolated": True,
                },
            )
        return frames
