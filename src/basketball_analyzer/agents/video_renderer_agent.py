from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .base import BaseAgent
from ..court_profile import WNBA_COURT
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
PX_PER_FOOT_X = COURT_WIDTH / WNBA_COURT.length_ft
PX_PER_FOOT_Y = COURT_HEIGHT / WNBA_COURT.width_ft


class VideoRendererAgent(BaseAgent):
    name = 'video_renderer_agent'

    def _ft_x(self, feet: float) -> float:
        return feet * PX_PER_FOOT_X

    def _ft_y(self, feet: float) -> float:
        return feet * PX_PER_FOOT_Y

    def _preview_frame(self, overlay, preview_images: list[Image.Image], index: int) -> None:
        if index % 2 != 0:
            return
        rgb_frame = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        image.thumbnail((960, 540))
        preview_images.append(image)

    def _ball_radius(self, ball_bbox, frame_height: int) -> int:
        box_width = max(1.0, ball_bbox.x2 - ball_bbox.x1)
        box_height = max(1.0, ball_bbox.y2 - ball_bbox.y1)
        estimated = int(min(box_width, box_height) * 0.28)
        return max(4, min(max(8, int(frame_height * 0.018)), estimated))

    def _draw_court(self, draw: ImageDraw.ImageDraw) -> None:
        center_x = COURT_WIDTH / 2
        center_y = COURT_HEIGHT / 2
        lane_half = self._ft_y(WNBA_COURT.lane_width_ft / 2)
        lane_depth = self._ft_x(WNBA_COURT.free_throw_line_ft)
        restricted_radius_x = self._ft_x(WNBA_COURT.restricted_area_radius_ft)
        restricted_radius_y = self._ft_y(WNBA_COURT.restricted_area_radius_ft)
        hoop_offset = self._ft_x(5.25)
        corner_y1 = self._ft_y(WNBA_COURT.corner_three_sideline_in)
        corner_y2 = COURT_HEIGHT - corner_y1
        corner_depth = self._ft_x(WNBA_COURT.corner_three_baseline_in)
        arc_radius_x = self._ft_x(WNBA_COURT.three_point_radius_ft)
        arc_radius_y = self._ft_y(WNBA_COURT.three_point_radius_ft)

        draw.rectangle((0, 0, COURT_WIDTH - 1, COURT_HEIGHT - 1), outline=LINE_COLOR, width=4)
        draw.line((center_x, 0, center_x, COURT_HEIGHT), fill=LINE_COLOR, width=3)
        draw.ellipse(
            (center_x - self._ft_x(6), center_y - self._ft_y(6), center_x + self._ft_x(6), center_y + self._ft_y(6)),
            outline=LINE_COLOR,
            width=3,
        )

        draw.rectangle((0, center_y - lane_half, lane_depth, center_y + lane_half), outline=LINE_COLOR, fill=PAINT_COLOR, width=3)
        draw.rectangle((COURT_WIDTH - lane_depth, center_y - lane_half, COURT_WIDTH, center_y + lane_half), outline=LINE_COLOR, fill=PAINT_COLOR, width=3)

        draw.arc(
            (lane_depth - self._ft_x(6), center_y - self._ft_y(6), lane_depth + self._ft_x(6), center_y + self._ft_y(6)),
            start=270,
            end=90,
            fill=LINE_COLOR,
            width=3,
        )
        draw.arc(
            (COURT_WIDTH - lane_depth - self._ft_x(6), center_y - self._ft_y(6), COURT_WIDTH - lane_depth + self._ft_x(6), center_y + self._ft_y(6)),
            start=90,
            end=270,
            fill=LINE_COLOR,
            width=3,
        )

        draw.ellipse(
            (hoop_offset - 8, center_y - 8, hoop_offset + 8, center_y + 8),
            outline=LINE_COLOR,
            width=3,
        )
        draw.ellipse(
            (COURT_WIDTH - hoop_offset - 8, center_y - 8, COURT_WIDTH - hoop_offset + 8, center_y + 8),
            outline=LINE_COLOR,
            width=3,
        )

        draw.arc(
            (hoop_offset - restricted_radius_x, center_y - restricted_radius_y, hoop_offset + restricted_radius_x, center_y + restricted_radius_y),
            start=270,
            end=90,
            fill=LINE_COLOR,
            width=3,
        )
        draw.arc(
            (COURT_WIDTH - hoop_offset - restricted_radius_x, center_y - restricted_radius_y, COURT_WIDTH - hoop_offset + restricted_radius_x, center_y + restricted_radius_y),
            start=90,
            end=270,
            fill=LINE_COLOR,
            width=3,
        )

        draw.line((0, corner_y1, corner_depth, corner_y1), fill=LINE_COLOR, width=3)
        draw.line((0, corner_y2, corner_depth, corner_y2), fill=LINE_COLOR, width=3)
        draw.line((COURT_WIDTH - corner_depth, corner_y1, COURT_WIDTH, corner_y1), fill=LINE_COLOR, width=3)
        draw.line((COURT_WIDTH - corner_depth, corner_y2, COURT_WIDTH, corner_y2), fill=LINE_COLOR, width=3)
        draw.arc(
            (hoop_offset - arc_radius_x, center_y - arc_radius_y, hoop_offset + arc_radius_x, center_y + arc_radius_y),
            start=302,
            end=58,
            fill=LINE_COLOR,
            width=3,
        )
        draw.arc(
            (COURT_WIDTH - hoop_offset - arc_radius_x, center_y - arc_radius_y, COURT_WIDTH - hoop_offset + arc_radius_x, center_y + arc_radius_y),
            start=122,
            end=238,
            fill=LINE_COLOR,
            width=3,
        )

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
    ) -> tuple[str, str, str, str]:
        images: list[Image.Image] = []

        for frame in frames:
            image = Image.new('RGB', (COURT_WIDTH, COURT_HEIGHT + 110), '#111827')
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, COURT_WIDTH, COURT_HEIGHT), fill=COURT_COLOR)
            self._draw_court(draw)

            poss_player, poss_team = self._possession_lookup(frame.frame_index, possession_timeline)
            recent_event = self._recent_event(frame.timestamp_s, events)

            for player in frame.players:
                if player.meta.get('role') == 'official':
                    continue
                color = HOME_COLOR if player.team_id == 'home' else AWAY_COLOR
                x1 = int(player.bbox.x1)
                y1 = int(player.bbox.y1)
                x2 = int(player.bbox.x2)
                y2 = int(player.bbox.y2)
                width = 5 if player.track_id == poss_player else 3
                draw.rounded_rectangle((x1, y1, x2, y2), radius=8, outline=color, width=width)
                draw.text((x1, max(0, y1 - 16)), player.track_id, fill='white')

            if frame.ball is not None:
                center_x = int(frame.ball.bbox.center[0])
                center_y = int(frame.ball.bbox.center[1])
                radius = self._ball_radius(frame.ball.bbox, COURT_HEIGHT)
                draw.ellipse(
                    (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
                    fill=BALL_COLOR,
                    outline='#111827',
                    width=2,
                )

            draw.rectangle((0, COURT_HEIGHT, COURT_WIDTH, COURT_HEIGHT + 110), fill='#0f172a')
            draw.text((20, COURT_HEIGHT + 12), f'{WNBA_COURT.name} court | Time {frame.timestamp_s:0.1f}s', fill='white')
            draw.text((160, COURT_HEIGHT + 12), f'Possession: {poss_team or "none"}', fill='white')
            draw.text((20, COURT_HEIGHT + 42), f"Home: {team_stats.get('home', {}).get('pass', 0)} passes | {team_stats.get('home', {}).get('steal', 0)} steals | {team_stats.get('home', {}).get('turnover', 0)} turnovers", fill='#bfdbfe')
            draw.text((20, COURT_HEIGHT + 68), f"Away: {team_stats.get('away', {}).get('pass', 0)} passes | {team_stats.get('away', {}).get('steal', 0)} steals | {team_stats.get('away', {}).get('turnover', 0)} turnovers", fill='#fecaca')
            if recent_event:
                label = recent_event.event_type.replace('_', ' ').title()
                actor = recent_event.player_id or recent_event.team_id or 'unknown'
                draw.text((500, COURT_HEIGHT + 12), f'Latest event: {label} by {actor}', fill='#fde68a')

            images.append(image)

        gif_path = output_path.with_suffix('.gif')
        gif_duration_ms = 180
        if len(frames) > 1:
            avg_delta_s = sum(
                max(0.001, frames[idx].timestamp_s - frames[idx - 1].timestamp_s)
                for idx in range(1, len(frames))
            ) / max(1, len(frames) - 1)
            gif_duration_ms = max(80, int(round(avg_delta_s * 1000)))
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=gif_duration_ms,
            loop=0,
            format='GIF',
        )
        return (
            str(gif_path),
            'gif',
            'Rendered as a synthetic court animation. Install OpenCV to enable source-video overlays for MP4 uploads.',
            str(gif_path),
        )

    def _render_source_video_overlay(
        self,
        frames: list[TrackingFrame],
        possession_timeline: list[PossessionFrame],
        events: list[Event],
        team_stats: dict[str, dict[str, float | int]],
        output_path: Path,
        source_video_path: Path,
    ) -> tuple[str, str, str, str] | None:
        if cv2 is None:
            return None

        capture = cv2.VideoCapture(str(source_video_path))
        if not capture.isOpened():
            return None

        fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        output_fps = max(1.0, min(fps, 12.0))
        if len(frames) > 1:
            avg_delta_s = sum(
                max(0.001, frames[idx].timestamp_s - frames[idx - 1].timestamp_s)
                for idx in range(1, len(frames))
            ) / max(1, len(frames) - 1)
            output_fps = max(1.0, min(1.0 / avg_delta_s, fps))
        output_file = output_path.with_suffix('.mp4')
        writer = cv2.VideoWriter(
            str(output_file),
            cv2.VideoWriter_fourcc(*'mp4v'),
            output_fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            return None

        preview_images: list[Image.Image] = []

        target_frames = {frame.frame_index: frame for frame in frames}
        if not target_frames:
            capture.release()
            writer.release()
            return None

        current_index = 0
        rendered_count = 0
        last_target_index = max(target_frames)

        while True:
            ok, raw_frame = capture.read()
            if not ok:
                break
            if current_index not in target_frames:
                current_index += 1
                if current_index > last_target_index:
                    break
                continue

            frame = target_frames[current_index]

            overlay = raw_frame.copy()
            poss_player, poss_team = self._possession_lookup(frame.frame_index, possession_timeline)
            recent_event = self._recent_event(frame.timestamp_s, events)

            for player in frame.players:
                if player.meta.get('role') == 'official':
                    continue
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
                radius = self._ball_radius(frame.ball.bbox, height)
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

            self._preview_frame(overlay, preview_images, rendered_count)
            rendered_count += 1
            current_index += 1
            if current_index > last_target_index:
                break

        capture.release()
        writer.release()
        preview_path = output_path.with_name(f'{output_path.stem}_preview.gif')
        if preview_images:
            preview_duration_ms = 140
            if rendered_count > 1 and len(frames) > 1:
                avg_delta_s = sum(
                    max(0.001, frames[idx].timestamp_s - frames[idx - 1].timestamp_s)
                    for idx in range(1, len(frames))
                ) / max(1, len(frames) - 1)
                preview_duration_ms = max(80, int(round(avg_delta_s * 1000 * 2)))
            preview_images[0].save(
                preview_path,
                save_all=True,
                append_images=preview_images[1:],
                duration=preview_duration_ms,
                loop=0,
                format='GIF',
            )
        return (
            str(output_file),
            'mp4',
            'Rendered as an overlay on sampled source-video frames from the uploaded MP4. The in-app preview uses a GIF for better browser compatibility.',
            str(preview_path) if preview_images else str(output_file),
        )

    def render(
        self,
        frames: list[TrackingFrame],
        possession_timeline: list[PossessionFrame],
        events: list[Event],
        team_stats: dict[str, dict[str, float | int]],
        output_path: str | Path,
        source_video_path: str | Path = 'demo',
    ) -> tuple[str, str, str, str] | None:
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
