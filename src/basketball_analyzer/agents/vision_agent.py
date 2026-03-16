from __future__ import annotations

import json
from pathlib import Path

from .base import BaseAgent
from ..demo_data import generate_demo_frames
from ..prototype_cv import PrototypeCVExtractor
from ..schemas import BBox, Detection, TrackingFrame


class VisionAgent(BaseAgent):
    name = 'vision_agent'

    def __init__(self) -> None:
        self.prototype_cv = PrototypeCVExtractor()

    def _from_json(self, path: Path) -> list[TrackingFrame]:
        raw = json.loads(path.read_text(encoding='utf-8'))
        frames: list[TrackingFrame] = []
        for item in raw:
            players = [
                Detection(
                    track_id=player['track_id'],
                    label=player.get('label', 'player'),
                    team_id=player.get('team_id'),
                    confidence=player.get('confidence', 1.0),
                    bbox=BBox(**player['bbox']),
                    meta=player.get('meta', {}),
                )
                for player in item.get('players', [])
            ]
            ball_raw = item.get('ball')
            ball = None
            if ball_raw:
                ball = Detection(
                    track_id=ball_raw.get('track_id', 'ball'),
                    label=ball_raw.get('label', 'ball'),
                    confidence=ball_raw.get('confidence', 1.0),
                    team_id=ball_raw.get('team_id'),
                    bbox=BBox(**ball_raw['bbox']),
                    meta=ball_raw.get('meta', {}),
                )
            frames.append(
                TrackingFrame(
                    frame_index=item['frame_index'],
                    timestamp_s=item['timestamp_s'],
                    players=players,
                    ball=ball,
                )
            )
        return frames

    def run(self, video_path: str | Path) -> list[TrackingFrame]:
        path = Path(video_path)
        token = str(video_path).lower()

        if token in {'demo', '__demo__', 'sample.mp4'}:
            return generate_demo_frames()
        if path.suffix.lower() == '.json' and path.exists():
            return self._from_json(path)
        if path.exists() and path.suffix.lower() == '.mp4':
            frames = self.prototype_cv.run(path)
            if frames:
                return frames
            return generate_demo_frames()
        return []
