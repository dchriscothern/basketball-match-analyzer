from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(slots=True)
class Detection:
    track_id: str
    label: str
    bbox: BBox
    confidence: float = 1.0
    team_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrackingFrame:
    frame_index: int
    timestamp_s: float
    players: list[Detection] = field(default_factory=list)
    ball: Detection | None = None


@dataclass(slots=True)
class PossessionFrame:
    frame_index: int
    timestamp_s: float
    player_id: str | None
    team_id: str | None


@dataclass(slots=True)
class Event:
    event_type: str
    timestamp_s: float
    team_id: str | None
    player_id: str | None
    secondary_player_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnalysisOutput:
    frames: list[TrackingFrame]
    possession_timeline: list[PossessionFrame]
    events: list[Event]
    player_stats: dict[str, dict[str, float | int]]
    team_stats: dict[str, dict[str, float | int]]
    report_text: str
    rendered_media_path: str | None = None
    rendered_media_kind: str | None = None
    rendered_media_note: str | None = None
