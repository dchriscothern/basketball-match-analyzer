from __future__ import annotations

from .base import BaseAgent
from ..schemas import TrackingFrame


class TeamClassifier(BaseAgent):
    name = "team_classifier"

    def run(self, frames: list[TrackingFrame]) -> list[TrackingFrame]:
        """
        Placeholder for jersey / embedding-based team assignment.

        Assumes team labels may already be present from upstream tracking.
        """
        return frames
