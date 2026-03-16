from __future__ import annotations

from .base import BaseAgent


class ReportingAgent(BaseAgent):
    name = 'reporting_agent'

    def run(
        self,
        player_stats: dict[str, dict[str, float | int]],
        team_stats: dict[str, dict[str, float | int]],
    ) -> str:
        if not team_stats:
            return 'No tracked events were available, so the report is limited. Wire up the vision and tracking layer to generate basketball possession analysis.'

        teams = sorted(team_stats.items(), key=lambda item: item[0])
        lead_team_id, lead_team_stats = max(
            teams,
            key=lambda item: (
                item[1].get('pass', 0) + item[1].get('steal', 0) + item[1].get('defensive_rebound', 0),
                item[1].get('possession_pct', 0),
            ),
        )
        other_team = next((item for item in teams if item[0] != lead_team_id), (lead_team_id, lead_team_stats))
        other_team_id, other_team_stats = other_team

        standout = sorted(
            player_stats.items(),
            key=lambda item: (
                item[1].get('pass', 0) + item[1].get('steal', 0) + item[1].get('shot_attempt', 0),
                item[1].get('possession_frames', 0),
            ),
            reverse=True,
        )[:3]
        standout_text = ', '.join(
            f"{player_id} ({stats.get('pass', 0)} passes, {stats.get('steal', 0)} steals, {stats.get('shot_attempt', 0)} shots)"
            for player_id, stats in standout
        ) or 'No standout performers identified yet'

        return (
            f"{lead_team_id.title()} won the key stretch with {lead_team_stats.get('pass', 0)} completed passes, "
            f"{lead_team_stats.get('steal', 0)} steals and {lead_team_stats.get('possession_pct', 0)}% of tracked possession. "
            f"{other_team_id.title()} generated {other_team_stats.get('shot_attempt', 0)} shot attempts but were hurt by {other_team_stats.get('turnover', 0)} turnovers. "
            f"The demo shows the agent stack segmenting possessions, spotting a shot sequence, tagging the rebound, and catching a live-ball turnover swing. "
            f"Standouts: {standout_text}."
        )
