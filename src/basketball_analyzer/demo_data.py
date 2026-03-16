from __future__ import annotations

from dataclasses import dataclass

from .schemas import BBox, Detection, TrackingFrame

COURT_WIDTH = 940
COURT_HEIGHT = 500
LEFT_HOOP = (60, 250)
RIGHT_HOOP = (880, 250)


@dataclass(slots=True)
class PlayerState:
    track_id: str
    team_id: str
    x: float
    y: float


def _player_detection(player: PlayerState) -> Detection:
    return Detection(
        track_id=player.track_id,
        label='player',
        team_id=player.team_id,
        bbox=BBox(player.x - 14, player.y - 30, player.x + 14, player.y + 30),
    )


def _ball_detection(x: float, y: float, *, meta: dict | None = None) -> Detection:
    return Detection(
        track_id='ball',
        label='ball',
        bbox=BBox(x - 5, y - 5, x + 5, y + 5),
        meta=meta or {},
    )


def generate_demo_frames() -> list[TrackingFrame]:
    home = [
        PlayerState('home_g1', 'home', 180, 250),
        PlayerState('home_g2', 'home', 260, 180),
        PlayerState('home_g3', 'home', 260, 320),
        PlayerState('home_g4', 'home', 360, 210),
        PlayerState('home_g5', 'home', 360, 300),
    ]
    away = [
        PlayerState('away_g1', 'away', 620, 250),
        PlayerState('away_g2', 'away', 700, 180),
        PlayerState('away_g3', 'away', 700, 320),
        PlayerState('away_g4', 'away', 560, 210),
        PlayerState('away_g5', 'away', 560, 300),
    ]

    frames: list[TrackingFrame] = []
    fps = 5.0

    def add_frame(index: int, ball_xy: tuple[float, float], ball_meta: dict | None = None):
        all_players = [_player_detection(player) for player in home + away]
        frames.append(
            TrackingFrame(
                frame_index=index,
                timestamp_s=round(index / fps, 2),
                players=all_players,
                ball=_ball_detection(ball_xy[0], ball_xy[1], meta=ball_meta),
            )
        )

    sequence = [
        ((180, 250), None),
        ((180, 250), None),
        ((180, 250), None),
        ((220, 215), None),
        ((260, 180), None),
        ((260, 180), None),
        ((260, 180), None),
        ((310, 195), None),
        ((360, 210), None),
        ((360, 210), None),
        ((360, 210), None),
        ((420, 215), {'ball_state': 'shot_attempt', 'shot_team_id': 'home', 'target_hoop': 'right'}),
        ((560, 220), {'ball_state': 'flight', 'shot_team_id': 'home', 'target_hoop': 'right'}),
        ((700, 210), {'ball_state': 'rebound_window', 'shot_team_id': 'home', 'target_hoop': 'right'}),
        ((700, 180), None),
        ((700, 180), None),
        ((700, 180), None),
        ((650, 215), None),
        ((620, 250), None),
        ((620, 250), None),
        ((620, 250), None),
        ((580, 275), None),
        ((560, 300), None),
        ((560, 300), None),
        ((560, 300), None),
        ((470, 290), None),
        ((360, 300), None),
        ((360, 300), None),
        ((360, 300), None),
    ]

    for idx, (ball_xy, meta) in enumerate(sequence):
        add_frame(idx, ball_xy, meta)

    return frames
