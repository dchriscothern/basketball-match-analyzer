from __future__ import annotations

import math
from collections import deque

from .base import BaseAgent
from ..schemas import Event, PossessionFrame, TrackingFrame


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.dist(a, b)


class EventsAgent(BaseAgent):
    name = 'events_agent'

    def __init__(self, control_radius_px: float = 55.0, turnover_smoothing_frames: int = 3) -> None:
        self.control_radius_px = control_radius_px
        self.turnover_smoothing_frames = turnover_smoothing_frames

    def _controller(self, frame: TrackingFrame) -> tuple[str | None, str | None]:
        if frame.ball is None or not frame.players:
            return None, None

        ball_center = frame.ball.bbox.center
        nearest = min(frame.players, key=lambda player: _distance(player.bbox.center, ball_center))
        if _distance(nearest.bbox.center, ball_center) > self.control_radius_px:
            return None, None
        return nearest.track_id, nearest.team_id

    def run(self, frames: list[TrackingFrame]) -> tuple[list[PossessionFrame], list[Event]]:
        possession_timeline: list[PossessionFrame] = []
        events: list[Event] = []
        pending_change: deque[tuple[str | None, str | None]] = deque(maxlen=self.turnover_smoothing_frames)
        stable_controller: tuple[str | None, str | None] = (None, None)
        previous_stable: tuple[str | None, str | None] = (None, None)
        last_shot_team: str | None = None
        shot_active = False

        for frame in frames:
            controller = self._controller(frame)
            pending_change.append(controller)

            if pending_change and len(set(pending_change)) == 1:
                stable_controller = pending_change[-1]

            player_id, team_id = stable_controller
            possession_timeline.append(
                PossessionFrame(
                    frame_index=frame.frame_index,
                    timestamp_s=frame.timestamp_s,
                    player_id=player_id,
                    team_id=team_id,
                )
            )

            ball_meta = frame.ball.meta if frame.ball else {}
            if ball_meta.get('ball_state') == 'shot_attempt' and not shot_active:
                last_shot_team = ball_meta.get('shot_team_id') or team_id
                shot_active = True
                events.append(
                    Event(
                        event_type='shot_attempt',
                        timestamp_s=frame.timestamp_s,
                        team_id=last_shot_team,
                        player_id=previous_stable[0] or player_id,
                        metadata={'target_hoop': ball_meta.get('target_hoop')},
                    )
                )

            if stable_controller == previous_stable:
                continue

            prev_player, prev_team = previous_stable
            curr_player, curr_team = stable_controller
            if prev_player and curr_player and prev_team and curr_team:
                if prev_team == curr_team and prev_player != curr_player:
                    events.append(
                        Event(
                            event_type='pass',
                            timestamp_s=frame.timestamp_s,
                            team_id=curr_team,
                            player_id=prev_player,
                            secondary_player_id=curr_player,
                        )
                    )
                elif prev_team != curr_team:
                    events.append(
                        Event(
                            event_type='turnover',
                            timestamp_s=frame.timestamp_s,
                            team_id=prev_team,
                            player_id=prev_player,
                            secondary_player_id=curr_player,
                            metadata={'new_team_id': curr_team},
                        )
                    )
                    events.append(
                        Event(
                            event_type='steal',
                            timestamp_s=frame.timestamp_s,
                            team_id=curr_team,
                            player_id=curr_player,
                            secondary_player_id=prev_player,
                        )
                    )

            if shot_active and curr_player and curr_team:
                rebound_type = 'offensive_rebound' if curr_team == last_shot_team else 'defensive_rebound'
                events.append(
                    Event(
                        event_type=rebound_type,
                        timestamp_s=frame.timestamp_s,
                        team_id=curr_team,
                        player_id=curr_player,
                        metadata={'shot_team_id': last_shot_team},
                    )
                )
                shot_active = False
                last_shot_team = None

            previous_stable = stable_controller

        return possession_timeline, events
