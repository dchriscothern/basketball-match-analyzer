from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .analysis_profiles import prepare_analysis_source
from .pipeline import BasketballAnalysisPipeline
from .video_sources import download_video_url


def _load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError('Benchmark manifest must be a JSON array of benchmark case objects.')
    return data


def _count_players(frames) -> int:
    return sum(len(frame.players) for frame in frames)


def _unique_player_ids(frames) -> int:
    return len({player.track_id for frame in frames for player in frame.players})


def _ball_frames(frames) -> int:
    return sum(1 for frame in frames if frame.ball is not None)


def _possession_frames(possession_timeline) -> int:
    return sum(1 for item in possession_timeline if item.player_id is not None)


def _run_case(pipeline: BasketballAnalysisPipeline, case: dict, backend: str) -> dict:
    source = case['source']
    profile = case.get('analysis_profile', 'Standard Clip')
    prepared_source = source
    prepared_label = 'full clip'
    settings_preview_only = profile != 'Full Clip'
    if source != 'demo' and str(source).lower().startswith(('http://', 'https://')):
        try:
            source = str(download_video_url(str(source), preview_only=settings_preview_only))
        except Exception:
            source = str(download_video_url(str(source), preview_only=False, allow_partial_clip=False))
    if source != 'demo' and str(source).lower().endswith('.mp4'):
        if not Path(source).exists():
            raise FileNotFoundError(f'Clip not found: {source}')
        prepared_source, prepared_label = prepare_analysis_source(source, profile)

    start = time.perf_counter()
    vision_frames = pipeline.vision_agent.run(prepared_source, backend=backend)
    vision_runtime = time.perf_counter() - start

    raw_player_count = _count_players(vision_frames)
    raw_avg_players = raw_player_count / len(vision_frames) if vision_frames else 0.0
    raw_ball_ratio = _ball_frames(vision_frames) / len(vision_frames) if vision_frames else 0.0

    frames = pipeline.ball_interpolator.run(vision_frames)
    pre_class_player_count = _count_players(frames)
    classified_frames = pipeline.team_classifier.run(frames)
    filtered_player_count = _count_players(classified_frames)
    removed_players = max(0, pre_class_player_count - filtered_player_count)

    possession_timeline, events = pipeline.events_agent.run(classified_frames)
    player_stats, team_stats = pipeline.analytics_agent.run(possession_timeline, events)
    report_text = pipeline.reporting_agent.run(player_stats, team_stats)
    total_runtime = time.perf_counter() - start

    frame_count = len(classified_frames)
    possession_ratio = _possession_frames(possession_timeline) / frame_count if frame_count else 0.0
    post_ball_ratio = _ball_frames(classified_frames) / frame_count if frame_count else 0.0

    return {
        'case_name': case['name'],
        'source': source,
        'prepared_source': str(prepared_source),
        'prepared_label': prepared_label,
        'analysis_profile': profile,
        'tracking_backend': backend,
        'vision_runtime_s': round(vision_runtime, 3),
        'total_runtime_s': round(total_runtime, 3),
        'frames': frame_count,
        'avg_players_detected': round(raw_avg_players, 3),
        'unique_player_tracks': _unique_player_ids(classified_frames),
        'ball_frame_ratio_raw': round(raw_ball_ratio, 3),
        'ball_frame_ratio_post': round(post_ball_ratio, 3),
        'possession_frame_ratio': round(possession_ratio, 3),
        'events': len(events),
        'passes': sum(1 for event in events if event.event_type == 'pass'),
        'turnovers': sum(1 for event in events if event.event_type == 'turnover'),
        'steals': sum(1 for event in events if event.event_type == 'steal'),
        'shots': sum(1 for event in events if event.event_type == 'shot_attempt'),
        'raw_players_total': raw_player_count,
        'players_removed_post_classification': removed_players,
        'tracking_note': pipeline.vision_agent.last_run_note,
        'calibration_note': pipeline.vision_agent.last_calibration_note,
        'report_excerpt': report_text[:180],
    }


def _write_markdown_summary(results: list[dict], output_path: Path) -> Path:
    summary_path = output_path.with_suffix('.md')
    lines = ['# Benchmark Summary', '']
    grouped: dict[str, list[dict]] = {}
    for row in results:
        grouped.setdefault(row.get('case_name', 'unknown'), []).append(row)

    for case_name, rows in grouped.items():
        lines.append(f'## {case_name}')
        lines.append('')
        lines.append('| Backend | Runtime (s) | Avg Players | Ball Ratio | Possession Ratio | Events | Notes |')
        lines.append('|---|---:|---:|---:|---:|---:|---|')
        for row in rows:
            if 'error' in row:
                lines.append(f"| {row['tracking_backend']} | - | - | - | - | - | ERROR: {row['error']} |")
            else:
                note = (row.get('tracking_note') or '').replace('|', '/')
                lines.append(
                    f"| {row['tracking_backend']} | {row['total_runtime_s']} | {row['avg_players_detected']} | "
                    f"{row['ball_frame_ratio_post']} | {row['possession_frame_ratio']} | {row['events']} | {note} |"
                )
        lines.append('')

    summary_path.write_text('\n'.join(lines), encoding='utf-8')
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark basketball analyzer backends across a clip manifest')
    parser.add_argument('--manifest', required=True, help='Path to benchmark manifest JSON')
    parser.add_argument('--output', default='benchmark_results.json', help='Path to JSON output report')
    parser.add_argument(
        '--backends',
        nargs='+',
        default=['yolo_detect', 'yolo_botsort', 'yolo_bytetrack', 'prototype_cv'],
        help='Tracking backends to evaluate',
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    cases = _load_manifest(manifest_path)
    pipeline = BasketballAnalysisPipeline()
    results: list[dict] = []

    for case in cases:
        for backend in args.backends:
            try:
                results.append(_run_case(pipeline, case, backend))
            except Exception as exc:
                results.append(
                    {
                        'case_name': case.get('name', 'unknown'),
                        'source': case.get('source'),
                        'analysis_profile': case.get('analysis_profile', 'Standard Clip'),
                        'tracking_backend': backend,
                        'error': str(exc),
                    }
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    summary_path = _write_markdown_summary(results, output_path)
    print(f'Saved benchmark results to {output_path.resolve()}')
    print(f'Saved benchmark summary to {summary_path.resolve()}')


if __name__ == '__main__':
    main()
