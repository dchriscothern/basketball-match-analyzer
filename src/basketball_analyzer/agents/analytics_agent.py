from __future__ import annotations

from collections import Counter, defaultdict

from .base import BaseAgent
from ..schemas import Event, PossessionFrame


class AnalyticsAgent(BaseAgent):
    name = 'analytics_agent'

    def run(
        self,
        possession_timeline: list[PossessionFrame],
        events: list[Event],
    ) -> tuple[dict[str, dict[str, float | int]], dict[str, dict[str, float | int]]]:
        player_stats: dict[str, dict[str, float | int]] = defaultdict(lambda: defaultdict(int))
        team_stats: dict[str, dict[str, float | int]] = defaultdict(lambda: defaultdict(int))

        possession_counts = Counter(frame.player_id for frame in possession_timeline if frame.player_id)
        team_possession_counts = Counter(frame.team_id for frame in possession_timeline if frame.team_id)
        total_controlled_frames = sum(team_possession_counts.values())

        for player_id, frames in possession_counts.items():
            player_stats[player_id]['possession_frames'] = frames

        for team_id, frames in team_possession_counts.items():
            team_stats[team_id]['possession_frames'] = frames
            team_stats[team_id]['possession_pct'] = round((frames / total_controlled_frames) * 100, 1) if total_controlled_frames else 0.0

        for event in events:
            if event.player_id:
                player_stats[event.player_id][event.event_type] += 1
            if event.secondary_player_id and event.event_type == 'pass':
                player_stats[event.secondary_player_id]['passes_received'] += 1
            if event.team_id:
                team_stats[event.team_id][event.event_type] += 1

        for stats in player_stats.values():
            stats.setdefault('pass', 0)
            stats.setdefault('steal', 0)
            stats.setdefault('turnover', 0)
            stats.setdefault('shot_attempt', 0)
            stats.setdefault('offensive_rebound', 0)
            stats.setdefault('defensive_rebound', 0)

        for stats in team_stats.values():
            stats.setdefault('pass', 0)
            stats.setdefault('steal', 0)
            stats.setdefault('turnover', 0)
            stats.setdefault('shot_attempt', 0)
            stats.setdefault('offensive_rebound', 0)
            stats.setdefault('defensive_rebound', 0)

        return {k: dict(v) for k, v in player_stats.items()}, {k: dict(v) for k, v in team_stats.items()}
