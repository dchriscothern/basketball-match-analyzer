from __future__ import annotations

import math
from collections import deque
from .base import BaseAgent
from ..schemas import Event, PossessionFrame, TrackingFrame


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.dist(a, b)


class EventsAgent(BaseAgent):
    name = 'events_agent'

    def __init__(
        self,
        control_radius_px: float = 55.0,
        turnover_smoothing_frames: int = 3,
        possession_grace_frames: int = 2,
        takeover_confirmation_frames: int = 2,
        shot_window_frames: int = 4,
    ) -> None:
        self.control_radius_px = control_radius_px
        self.turnover_smoothing_frames = turnover_smoothing_frames
        self.possession_grace_frames = possession_grace_frames
        self.takeover_confirmation_frames = takeover_confirmation_frames
        self.shot_window_frames = shot_window_frames

    def _distance_to_bbox(self, point: tuple[float, float], bbox) -> float:
        px, py = point
        dx = max(bbox.x1 - px, 0.0, px - bbox.x2)
        dy = max(bbox.y1 - py, 0.0, py - bbox.y2)
        return math.hypot(dx, dy)

    def _player_control_score(self, player, ball_center: tuple[float, float]) -> float:
        bbox = player.bbox
        width = bbox.x2 - bbox.x1
        height = bbox.y2 - bbox.y1
        control_anchor = ((bbox.x1 + bbox.x2) / 2.0, bbox.y1 + (height * 0.38))
        anchor_distance = _distance(control_anchor, ball_center)
        bbox_distance = self._distance_to_bbox(ball_center, bbox)
        contains_ball = (
            bbox.x1 - width * 0.08 <= ball_center[0] <= bbox.x2 + width * 0.08
            and bbox.y1 - height * 0.12 <= ball_center[1] <= bbox.y2 + height * 0.12
        )
        score = min(anchor_distance, bbox_distance + 8.0)
        if contains_ball:
            score -= 18.0
        return score

    def _controller(self, frame: TrackingFrame) -> tuple[str | None, str | None]:
        if frame.ball is None or not frame.players:
            return None, None

        ball_center = frame.ball.bbox.center
        nearest = min(frame.players, key=lambda player: self._player_control_score(player, ball_center))
        if self._player_control_score(nearest, ball_center) > self.control_radius_px:
            return None, None
        return nearest.track_id, nearest.team_id

    def _player_lookup(self, frame: TrackingFrame) -> dict[str, object]:
        return {player.track_id: player for player in frame.players}

    def _shot_candidate(
        self,
        *,
        subject_frame: TrackingFrame,
        controller_before_gap: tuple[str | None, str | None],
        recent_ball_samples: deque[tuple[tuple[float, float], dict]],
    ) -> tuple[bool, dict]:
        if controller_before_gap == (None, None) or not recent_ball_samples:
            return False, {}
        shooter_id, shooter_team = controller_before_gap
        if shooter_id is None or shooter_team is None:
            return False, {}

        shooter = self._player_lookup(subject_frame).get(shooter_id)
        if shooter is None:
            return False, {}

        ball_center, ball_meta = recent_ball_samples[-1]
        bbox = shooter.bbox
        shoulder_y = bbox.y1 + (bbox.y2 - bbox.y1) * 0.36
        horizontal_gap = abs(ball_center[0] - ((bbox.x1 + bbox.x2) / 2.0))
        vertical_clearance = shoulder_y - ball_center[1]

        if vertical_clearance < 12.0:
            return False, {}
        if horizontal_gap > max(55.0, (bbox.x2 - bbox.x1) * 1.25):
            return False, {}

        if len(recent_ball_samples) < 2:
            return False, {}
        centers = [sample[0] for sample in recent_ball_samples]
        upward_travel = max(center[1] for center in centers) - min(center[1] for center in centers)
        if upward_travel < 14.0:
            return False, {}

        hoop_xy = ball_meta.get('target_hoop_xy')
        if not (
            isinstance(hoop_xy, (tuple, list))
            and len(hoop_xy) == 2
        ):
            return False, {}

        first_center = centers[0]
        current_distance_to_hoop = _distance(ball_center, (float(hoop_xy[0]), float(hoop_xy[1])))
        initial_distance_to_hoop = _distance(first_center, (float(hoop_xy[0]), float(hoop_xy[1])))
        hoop_gain = initial_distance_to_hoop - current_distance_to_hoop
        if hoop_gain < 10.0:
            return False, {}

        return True, {
            'shooter_id': shooter_id,
            'shooter_team_id': shooter_team,
            'target_hoop': ball_meta.get('target_hoop'),
            'vertical_clearance': round(vertical_clearance, 1),
            'upward_travel': round(upward_travel, 1),
            'hoop_gain': round(hoop_gain, 1),
        }

    def run(self, frames: list[TrackingFrame]) -> tuple[list[PossessionFrame], list[Event]]:
        possession_timeline: list[PossessionFrame] = []
        events: list[Event] = []
        stable_controller: tuple[str | None, str | None] = (None, None)
        previous_stable: tuple[str | None, str | None] = (None, None)
        candidate_controller: tuple[str | None, str | None] = (None, None)
        candidate_count = 0
        last_shot_team: str | None = None
        shot_active = False
        possession_gap = 0
        recent_ball_samples: deque[tuple[tuple[float, float], dict]] = deque(maxlen=self.shot_window_frames)
        previous_frame: TrackingFrame | None = None

        for frame in frames:
            if frame.ball is not None:
                recent_ball_samples.append((frame.ball.bbox.center, dict(frame.ball.meta or {})))
            controller = self._controller(frame)
            stable_before_update = stable_controller

            if controller != (None, None):
                if stable_controller == (None, None):
                    stable_controller = controller
                    candidate_controller = (None, None)
                    candidate_count = 0
                elif controller == stable_controller:
                    candidate_controller = (None, None)
                    candidate_count = 0
                else:
                    if controller == candidate_controller:
                        candidate_count += 1
                    else:
                        candidate_controller = controller
                        candidate_count = 1

                    if candidate_count >= self.takeover_confirmation_frames:
                        stable_controller = candidate_controller
                        candidate_controller = (None, None)
                        candidate_count = 0

            if controller == (None, None) and stable_controller != (None, None):
                possession_gap += 1
                if possession_gap <= self.possession_grace_frames:
                    controller = stable_controller
                else:
                    stable_controller = (None, None)
                    candidate_controller = (None, None)
                    candidate_count = 0
            else:
                possession_gap = 0

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

            if (
                not shot_active
                and controller == (None, None)
                and stable_before_update != (None, None)
                and recent_ball_samples
            ):
                subject_frame = frame if stable_before_update[0] in self._player_lookup(frame) else (previous_frame or frame)
                is_shot, shot_meta = self._shot_candidate(
                    subject_frame=subject_frame,
                    controller_before_gap=stable_before_update,
                    recent_ball_samples=recent_ball_samples,
                )
                if is_shot:
                    last_shot_team = shot_meta.get('shooter_team_id') or stable_before_update[1]
                    shot_active = True
                    events.append(
                        Event(
                            event_type='shot_attempt',
                            timestamp_s=frame.timestamp_s,
                            team_id=last_shot_team,
                            player_id=shot_meta.get('shooter_id') or stable_before_update[0],
                            metadata=shot_meta,
                        )
                    )

            if stable_controller == previous_stable:
                previous_frame = frame
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
            previous_frame = frame

        return possession_timeline, events
