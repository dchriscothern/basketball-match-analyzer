from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import BasketballAnalysisPipeline


def _event_to_dict(event):
    return {
        'event_type': event.event_type,
        'timestamp_s': event.timestamp_s,
        'team_id': event.team_id,
        'player_id': event.player_id,
        'secondary_player_id': event.secondary_player_id,
        'metadata': event.metadata,
    }


def _possession_to_dict(item):
    return {
        'frame_index': item.frame_index,
        'timestamp_s': item.timestamp_s,
        'player_id': item.player_id,
        'team_id': item.team_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the basketball multi-agent analysis pipeline')
    parser.add_argument('--video', default='demo', help='Path to input MP4 or JSON. Use demo for synthetic sample data.')
    parser.add_argument('--output', default='outputs', help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = BasketballAnalysisPipeline()
    result = pipeline.run(args.video, render_output_path=output_dir / 'analysis.gif')

    (output_dir / 'report.txt').write_text(result.report_text, encoding='utf-8')
    (output_dir / 'events.json').write_text(
        json.dumps([_event_to_dict(event) for event in result.events], indent=2),
        encoding='utf-8',
    )
    (output_dir / 'possession_timeline.json').write_text(
        json.dumps([_possession_to_dict(item) for item in result.possession_timeline], indent=2),
        encoding='utf-8',
    )
    (output_dir / 'player_stats.json').write_text(json.dumps(result.player_stats, indent=2), encoding='utf-8')
    (output_dir / 'team_stats.json').write_text(json.dumps(result.team_stats, indent=2), encoding='utf-8')

    print(result.report_text)
    if result.tracking_note:
        print(result.tracking_note)
    if result.rendered_media_path:
        print(f'Rendered analysis artifact: {result.rendered_media_path}')
    if result.rendered_media_note:
        print(result.rendered_media_note)
    print(f'Saved outputs to {output_dir.resolve()}')


if __name__ == '__main__':
    main()
