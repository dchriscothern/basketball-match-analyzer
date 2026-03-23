from __future__ import annotations

from collections import defaultdict

from .base import BaseAgent
from ..schemas import TrackingFrame


def _mean_color(colors: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    if not colors:
        return (0.0, 0.0, 0.0)
    count = float(len(colors))
    return (
        sum(color[0] for color in colors) / count,
        sum(color[1] for color in colors) / count,
        sum(color[2] for color in colors) / count,
    )


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _color_spread(color: tuple[float, float, float]) -> float:
    return max(color) - min(color)


def _color_mean(color: tuple[float, float, float]) -> float:
    return sum(color) / 3.0


def _mean_position(samples: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    if not samples:
        return (0.0, 0.0, 0.0)
    count = float(len(samples))
    return (
        sum(sample[0] for sample in samples) / count,
        sum(sample[1] for sample in samples) / count,
        sum(sample[2] for sample in samples) / count,
    )


class TeamClassifier(BaseAgent):
    name = "team_classifier"

    def _cluster_track_colors(
        self,
        track_colors: dict[str, tuple[float, float, float]],
        track_weights: dict[str, int] | None = None,
    ) -> dict[str, str]:
        items = list(track_colors.items())
        if len(items) < 2:
            return {}

        farthest_pair = (items[0][1], items[-1][1])
        farthest_distance = -1.0
        for idx, (_, color_a) in enumerate(items):
            for _, color_b in items[idx + 1:]:
                dist = _distance(color_a, color_b)
                if dist > farthest_distance:
                    farthest_distance = dist
                    farthest_pair = (color_a, color_b)

        centroid_a, centroid_b = farthest_pair
        assignments: dict[str, str] = {}

        for _ in range(6):
            group_a: list[tuple[float, float, float]] = []
            group_b: list[tuple[float, float, float]] = []
            next_assignments: dict[str, str] = {}

            for track_id, color in items:
                if _distance(color, centroid_a) <= _distance(color, centroid_b):
                    next_assignments[track_id] = 'home'
                    group_a.append(color)
                else:
                    next_assignments[track_id] = 'away'
                    group_b.append(color)

            if group_a:
                centroid_a = _mean_color(group_a)
            if group_b:
                centroid_b = _mean_color(group_b)
            assignments = next_assignments

        if len({team for team in assignments.values()}) < 2:
            midpoint = len(items) // 2
            assignments = {
                track_id: ('home' if idx < midpoint else 'away')
                for idx, (track_id, _) in enumerate(sorted(items, key=lambda item: sum(item[1])))
            }

        weights = track_weights or {track_id: 1 for track_id, _ in items}
        home_weight = sum(weights.get(track_id, 1) for track_id, team_id in assignments.items() if team_id == 'home')
        away_weight = sum(weights.get(track_id, 1) for track_id, team_id in assignments.items() if team_id == 'away')
        total_weight = home_weight + away_weight
        imbalance_ratio = 1.0 if total_weight == 0 else min(home_weight, away_weight) / total_weight

        if len(items) >= 4 and imbalance_ratio < 0.25:
            scored = []
            for track_id, color in items:
                affinity = _distance(color, centroid_b) - _distance(color, centroid_a)
                scored.append((affinity, track_id))
            scored.sort(key=lambda item: item[0])

            balanced_assignments: dict[str, str] = {}
            running_weight = 0
            target_weight = total_weight / 2.0
            for _, track_id in scored:
                balanced_assignments[track_id] = 'home' if running_weight < target_weight else 'away'
                running_weight += weights.get(track_id, 1)
            assignments = balanced_assignments
        return assignments

    def _find_non_team_tracks(
        self,
        averaged: dict[str, tuple[float, float, float]],
        assignments: dict[str, str],
        track_positions: dict[str, tuple[float, float, float]],
        frame_width: float,
        frame_height: float,
    ) -> set[str]:
        home_colors = [averaged[track_id] for track_id, team_id in assignments.items() if team_id == 'home' and track_id in averaged]
        away_colors = [averaged[track_id] for track_id, team_id in assignments.items() if team_id == 'away' and track_id in averaged]
        if not home_colors or not away_colors:
            return set()

        home_centroid = _mean_color(home_colors)
        away_centroid = _mean_color(away_colors)
        team_separation = _distance(home_centroid, away_centroid)
        outlier_threshold = max(42.0, team_separation * 0.42)

        non_team_tracks: set[str] = set()
        for track_id, color in averaged.items():
            nearest_team_distance = min(_distance(color, home_centroid), _distance(color, away_centroid))
            spread = _color_spread(color)
            mean_value = _color_mean(color)
            neutral_dark = spread < 52.0 and mean_value < 125.0
            neutral_mid = spread < 62.0 and 85.0 <= mean_value <= 180.0
            avg_x, avg_y, avg_h = track_positions.get(track_id, (0.0, 0.0, 0.0))
            top_band = avg_y < frame_height * 0.48
            sideline_edge = avg_x < frame_width * 0.12 or avg_x > frame_width * 0.88
            small_box = avg_h < frame_height * 0.18
            midcourt_lane = frame_width * 0.28 < avg_x < frame_width * 0.72
            on_floor_but_small = avg_y > frame_height * 0.48 and small_box
            likely_official_zone = (top_band and sideline_edge) or (top_band and small_box)
            likely_floor_official = neutral_dark and on_floor_but_small and midcourt_lane
            persistent_small_sideline = small_box and sideline_edge and avg_y < frame_height * 0.72
            neutral_small_floor_track = neutral_mid and on_floor_but_small and nearest_team_distance > outlier_threshold * 0.4

            if (
                (nearest_team_distance > outlier_threshold and spread < 72.0)
                or (neutral_dark and nearest_team_distance > max(22.0, team_separation * 0.18))
                or (neutral_dark and likely_official_zone)
                or (nearest_team_distance > outlier_threshold * 0.55 and likely_floor_official)
                or (nearest_team_distance > outlier_threshold * 0.8 and likely_official_zone)
                or (persistent_small_sideline and nearest_team_distance > outlier_threshold * 0.3)
                or neutral_small_floor_track
            ):
                non_team_tracks.add(track_id)
        return non_team_tracks

    def run(self, frames: list[TrackingFrame]) -> list[TrackingFrame]:
        """
        If upstream detections already have team IDs, preserve them.
        Otherwise infer home/away from persistent torso-color signatures.
        """
        if not frames:
            return frames

        if any(player.team_id for frame in frames for player in frame.players):
            return frames

        track_colors: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        track_positions_raw: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        frame_width = 0.0
        frame_height = 0.0
        for frame in frames:
            for player in frame.players:
                signature = player.meta.get('color_signature')
                if signature is not None:
                    track_colors[player.track_id].append(tuple(float(value) for value in signature))
                bbox = player.bbox
                track_positions_raw[player.track_id].append(
                    (
                        float((bbox.x1 + bbox.x2) / 2.0),
                        float((bbox.y1 + bbox.y2) / 2.0),
                        float(bbox.y2 - bbox.y1),
                    )
                )
                frame_width = max(frame_width, float(bbox.x2))
                frame_height = max(frame_height, float(bbox.y2))

        min_appearances = 5
        track_weights = {track_id: len(colors) for track_id, colors in track_colors.items()}
        track_positions = {
            track_id: _mean_position(samples)
            for track_id, samples in track_positions_raw.items()
            if samples
        }
        track_appearance_counts = {
            track_id: len(samples)
            for track_id, samples in track_positions_raw.items()
            if samples
        }
        averaged = {
            track_id: _mean_color(colors)
            for track_id, colors in track_colors.items()
            if len(colors) >= min_appearances
        }
        assignments = self._cluster_track_colors(averaged, track_weights=track_weights)
        home_colors = [averaged[track_id] for track_id, team_id in assignments.items() if team_id == 'home' and track_id in averaged]
        away_colors = [averaged[track_id] for track_id, team_id in assignments.items() if team_id == 'away' and track_id in averaged]
        if home_colors and away_colors:
            home_centroid = _mean_color(home_colors)
            away_centroid = _mean_color(away_colors)
            team_separation = _distance(home_centroid, away_centroid)
            outlier_threshold = max(42.0, team_separation * 0.42)
            print(
                "TEAM_CENTROIDS: "
                f"home={[round(value) for value in home_centroid]} "
                f"away={[round(value) for value in away_centroid]} "
                f"separation={team_separation:.1f} threshold={outlier_threshold:.1f}"
            )
        for track_id, color in sorted(averaged.items()):
            team = assignments.get(track_id, 'unassigned')
            pos = track_positions.get(track_id, (0.0, 0.0, 0.0))
            print(
                f"TRACK {track_id}: team={team} "
                f"color=({color[0]:.0f},{color[1]:.0f},{color[2]:.0f}) "
                f"pos=({pos[0]:.0f},{pos[1]:.0f}) height={pos[2]:.0f}"
            )
        non_team_tracks = self._find_non_team_tracks(
            averaged,
            assignments,
            track_positions,
            frame_width=max(frame_width, 1.0),
            frame_height=max(frame_height, 1.0),
        )
        total_frames = len(frames)
        for track_id, count in track_appearance_counts.items():
            if count < max(3, int(total_frames * 0.08)):
                non_team_tracks.add(track_id)
        all_seen_track_ids = {
            player.track_id
            for frame in frames
            for player in frame.players
        }
        unclassified_tracks = all_seen_track_ids - set(averaged.keys())
        non_team_tracks.update(unclassified_tracks)
        surviving_classified_tracks = [
            track_id
            for track_id in assignments
            if track_id not in non_team_tracks and track_id in track_positions
        ]
        surviving_heights = sorted(track_positions[track_id][2] for track_id in surviving_classified_tracks)
        if len(surviving_classified_tracks) > 5 and surviving_heights:
            mid = len(surviving_heights) // 2
            median_height = (
                surviving_heights[mid]
                if len(surviving_heights) % 2 == 1
                else (surviving_heights[mid - 1] + surviving_heights[mid]) / 2.0
            )
            extra_official_tracks = [
                track_id
                for track_id in surviving_classified_tracks
                if track_positions[track_id][2] < median_height * 0.88
                and frame_height * 0.45 < track_positions[track_id][1] < frame_height * 0.78
            ]
            if extra_official_tracks:
                likely_official = min(extra_official_tracks, key=lambda track_id: track_positions[track_id][2])
                non_team_tracks.add(likely_official)
                print(f"SIZE_FILTER_TRACK: {likely_official}")

        for frame in frames:
            filtered_players = []
            for player in frame.players:
                if player.track_id in non_team_tracks:
                    player.meta['role'] = 'official'
                    continue
                player.team_id = assignments.get(player.track_id, player.team_id or 'unknown')
                filtered_players.append(player)
            frame.players = filtered_players
        print(f"TEAM_CLASSIFIER_RUN: filtered={len(non_team_tracks)} ids={sorted(non_team_tracks)}")
        print(f"UNCLASSIFIED_TRACKS: {sorted(unclassified_tracks)}")
        return frames
