from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .base import BaseAgent
from ..schemas import Event, PossessionFrame, TrackingFrame

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    cv2 = None


COURT_WIDTH = 940
COURT_HEIGHT = 500
HOME_COLOR = '#2563eb'
AWAY_COLOR = '#dc2626'
BALL_COLOR = '#f59e0b'
LINE_COLOR = '#f8fafc'
COURT_COLOR = '#9a3412'
PAINT_COLOR = '#7c2d12'


class VideoRendererAgent(BaseAgent):
    name = 'video_renderer_agent'

    def _draw_court(self, draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle((0, 0, COURT_WIDTH - 1, COURT_HEIGHT - 1), outline=LINE_COLOR, width=4)
        draw.line((COURT_WIDTH / 2, 0, COURT_WIDTH / 2, COURT_HEIGHT), fill=LINE_COLOR, width=3)
        draw.ellipse((COURT_WIDTH / 2 - 60, COURT_HEIGHT / 2 - 60, COURT_WIDTH / 2 + 60, COURT_HEIGHT / 2 + 60), outline=LINE_COLOR, width=3)
        draw.rectangle((0, 170, 160, 330), outline=LINE_COLOR, fill=PAINT_COLOR, width=3)
        draw.rectangle((COURT_WIDTH - 160, 170, COURT_WIDTH, 330), outline=LINE_COLOR, fill=PAINT_COLOR, width=3)
        draw.arc((20, 170, 140, 330), start=270, end=90, fill=LINE_COLOR, width=3)
        draw.arc((COURT_WIDTH - 140, 170, COURT_WIDTH - 20, 330), start=90, end=270, fill=LINE_COLOR, width=3)
        draw.ellipse((45, 235, 75, 265), outline=LINE_COLOR, width=3)
        draw.ellipse((COURT_WIDTH - 75, 235, COURT_WIDTH - 45, 265), outline=LINE_COLOR, width=3)

    def _recent_event(self, timestamp_s: float, events: list[Event]) -> Event | None:
        latest = None
        for event in events:
            if event.timestamp_s <= timestamp_s and timestamp_s - event.timestamp_s <= 1.2:
                latest = event
        return latest

    def _possession_lookup(self, frame_index: int, timeline: list[PossessionFrame]) -> tuple[str | None, str | None]:
        for item in timeline:
            if item.frame_index == frame_index:
                return item.player_id, item.team_id
        return None, None

    def _render_gif(
        self,
        frames: list[TrackingFrame],
        possession_timeline: list[PossessionFrame],
        events: list[Event],
        team_stats: dict[str, dict[str, float | int]],
        output_path: Path,
    ) -> tuple[str, str, str]:
        images: list[Image.Image] = []

        for frame in frames:
            image = Image.new('RGB', (COURT_WIDTH, COURT_HEIGHT + 110), '#111827')
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, COURT_WIDTH, COURT_HEIGHT), fill=COURT_COLOR)
            self._draw_court(draw)

            poss_player, poss_team = self._possession_lookup(frame.frame_index, possession_timeline)
            recent_event = self._recent_event(frame.timestamp_s, events)

            for player in frame.players:
                color = HOME_COLOR if player.team_id == 'home' else AWAY_COLOR
                x1 = int(player.bbox.x1)
                y1 = int(player.bbox.y1)
                x2 = int(player.bbox.x2)
                y2 = int(player.bbox.y2)
                width = 5 if player.track_id == poss_player else 3
                draw.rounded_rectangle((x1, y1, x2, y2), radius=8, outline=color, width=width)
                draw.text((x1, max(0, y1 - 16)), player.track_id, fill='white')

            if frame.ball is not None:
                bx1 = int(frame.ball.bbox.x1)
                by1 = int(frame.ball.bbox.y1)
                bx2 = int(frame.ball.bbox.x2)
                by2 = int(frame.ball.bbox.y2)
                draw.ellipse((bx1, by1, bx2, by2), fill=BALL_COLOR, outline='#111827', width=2)

            draw.rectangle((0, COURT_HEIGHT, COURT_WIDTH, COURT_HEIGHT + 110), fill='#0f172a')
            draw.text((20, COURT_HEIGHT + 12), f'Time {frame.timestamp_s:0.1f}s', fill='white')
            draw.text((160, COURT_HEIGHT + 12), f'Possession: {poss_team or "none"}', fill='white')
            draw.text((20, COURT_HEIGHT + 42), f"Home: {team_stats.get('home', {}).get('pass', 0)} passes | {team_stats.get('home', {}).get('steal', 0)} steals | {team_stats.get('home', {}).get('turnover', 0)} turnovers", fill='#bfdbfe')
            draw.text((20, COURT_HEIGHT + 68), f"Away: {team_stats.get('away', {}).get('pass', 0)} passes | {team_stats.get('away', {}).get('steal', 0)} steals | {team_stats.get('away', {}).get('turnover', 0)} turnovers", fill='#fecaca')
            if recent_event:
                label = recent_event.event_type.replace('_', ' ').title()
                actor = recent_event.player_id or recent_event.team_id or 'unknown'
                draw.text((500, COURT_HEIGHT + 12), f'Latest event: {label} by {actor}', fill='#fde68a')

            images.append(image)

        gif_path = output_path.with_suffix('.gif')
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=180,
            loop=0,
            format='GIF',
        )
        return (
            str(gif_path),
            'gif',
            'Rendered as a synthetic court animation. Install OpenCV to enable source-video overlays for MP4 uploads.',
        )

    def _render_source_video_overlay(
        self,
        frames: list[TrackingFrame],
        possession_timeline: list[PossessionFrame],
        events: list[Event],
        team_stats: dict[str, dict[str, float | int]],
        output_path: Path,
        source_video_path: Path,
    ) -> tuple[str, str, str] | None:
        if cv2 is None:
            return None

        capture = cv2.VideoCapture(str(source_video_path))
        if not capture.isOpened():
            return None

        fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        output_file = output_path.with_suffix('.mp4')
        writer = cv2.VideoWriter(
            str(output_file),
            cv2.VideoWriter_fourcc(*'mp4v'),
            max(1.0, min(fps, 12.0)),
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            return None

        for frame in frames:
            capture.set(cv2.CAP_PROP_POS_MSEC, frame.timestamp_s * 1000.0)
            ok, raw_frame = capture.read()
            if not ok:
                continue

            overlay = raw_frame.copy()
            poss_player, poss_team = self._possession_lookup(frame.frame_index, possession_timeline)
            recent_event = self._recent_event(frame.timestamp_s, events)

            for player in frame.players:
                color = (235, 99, 37) if player.team_id == 'home' else (38, 38, 220)
                x1 = int(player.bbox.x1)
                y1 = int(player.bbox.y1)
                x2 = int(player.bbox.x2)
                y2 = int(player.bbox.y2)
                thickness = 4 if player.track_id == poss_player else 2
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(
                    overlay,
                    player.track_id,
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            if frame.ball is not None:
                center = tuple(int(value) for value in frame.ball.bbox.center)
                radius = max(6, int((frame.ball.bbox.x2 - frame.ball.bbox.x1) / 2))
                cv2.circle(overlay, center, radius, (11, 158, 245), -1)
                cv2.circle(overlay, center, radius, (17, 24, 39), 2)

            cv2.rectangle(overlay, (0, 0), (width, 86), (15, 23, 42), -1)
            cv2.putText(overlay, f'Time {frame.timestamp_s:0.1f}s', (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, f'Possession: {poss_team or "none"}', (220, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                overlay,
                f'Home P/S/T: {team_stats.get("home", {}).get("pass", 0)}/{team_stats.get("home", {}).get("steal", 0)}/{team_stats.get("home", {}).get("turnover", 0)}',
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (191, 219, 254),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f'Away P/S/T: {team_stats.get("away", {}).get("pass", 0)}/{team_stats.get("away", {}).get("steal", 0)}/{team_stats.get("away", {}).get("turnover", 0)}',
                (360, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (254, 202, 202),
                2,
                cv2.LINE_AA,
            )
            if recent_event:
                label = recent_event.event_type.replace('_', ' ').title()
                actor = recent_event.player_id or recent_event.team_id or 'unknown'
                cv2.putText(
                    overlay,
                    f'Latest event: {label} by {actor}',
                    (720, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (253, 230, 138),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(overlay)

        capture.release()
        writer.release()
        return (
            str(output_file),
            'mp4',
            'Rendered as an overlay on sampled source-video frames from the uploaded MP4.',
        )

    def render(
        self,
        frames: list[TrackingFrame],
        possession_timeline: list[PossessionFrame],
        events: list[Event],
        team_stats: dict[str, dict[str, float | int]],
        output_path: str | Path,
        source_video_path: str | Path = 'demo',
    ) -> tuple[str, str, str] | None:
        if not frames:
            return None

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        source_path = Path(source_video_path)

        if source_path.exists() and source_path.suffix.lower() == '.mp4':
            overlay_result = self._render_source_video_overlay(
                frames,
                possession_timeline,
                events,
                team_stats,
                output,
                source_path,
            )
            if overlay_result is not None:
                return overlay_result

        return self._render_gif(frames, possession_timeline, events, team_stats, output)
