from __future__ import annotations

from pathlib import Path

from .agents import (
    AnalyticsAgent,
    BallInterpolator,
    EventsAgent,
    ReportingAgent,
    TeamClassifier,
    VisionAgent,
)
from .schemas import AnalysisOutput


class BasketballAnalysisPipeline:
    def __init__(self) -> None:
        self.vision_agent = VisionAgent()
        self.ball_interpolator = BallInterpolator()
        self.team_classifier = TeamClassifier()
        self.events_agent = EventsAgent()
        self.analytics_agent = AnalyticsAgent()
        self.reporting_agent = ReportingAgent()

    def run(self, video_path: str | Path = 'demo') -> AnalysisOutput:
        frames = self.vision_agent.run(video_path)
        frames = self.ball_interpolator.run(frames)
        frames = self.team_classifier.run(frames)
        possession_timeline, events = self.events_agent.run(frames)
        player_stats, team_stats = self.analytics_agent.run(possession_timeline, events)
        report_text = self.reporting_agent.run(player_stats, team_stats)
        return AnalysisOutput(
            frames=frames,
            possession_timeline=possession_timeline,
            events=events,
            player_stats=player_stats,
            team_stats=team_stats,
            report_text=report_text,
        )
