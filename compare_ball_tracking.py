from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from basketball_analyzer.analysis_profiles import prepare_analysis_source
from basketball_analyzer.pipeline import BasketballAnalysisPipeline


def _resolve_video_path(video_arg: str) -> Path:
    if video_arg.lower() != "latest-temp":
        return Path(video_arg)

    temp_root = Path(tempfile.gettempdir())
    candidates = sorted(
        temp_root.glob("basketball_url_*\\source.mp4"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No recent temporary basketball source clip was found. Use a real MP4 path or run the app once first."
        )
    return candidates[0]


def _summarize(result, label: str) -> dict[str, float | int | str]:
    frames = result.frames
    ball_frames = sum(1 for frame in frames if frame.ball is not None)
    raw_ball_frames = sum(
        1
        for frame in frames
        if frame.ball is not None
        and not frame.ball.meta.get("fallback_ball")
        and not frame.ball.meta.get("estimated_ball")
        and not frame.ball.meta.get("motion_ball")
    )
    learned_ball_frames = sum(
        1 for frame in frames if frame.ball is not None and frame.ball.meta.get("learned_ball_detector")
    )
    return {
        "mode": label,
        "frames": len(frames),
        "ball_ratio": round(ball_frames / len(frames), 3) if frames else 0.0,
        "raw_ball_ratio": round(raw_ball_frames / len(frames), 3) if frames else 0.0,
        "learned_ball_ratio": round(learned_ball_frames / len(frames), 3) if frames else 0.0,
        "possession_ratio": round(len(result.possession_timeline) / len(frames), 3) if frames else 0.0,
        "events": len(result.events),
        "passes": sum(1 for event in result.events if event.event_type == "pass"),
        "shots": sum(1 for event in result.events if event.event_type == "shot_attempt"),
        "turnovers": sum(1 for event in result.events if event.event_type == "turnover"),
        "tracking_note": result.tracking_note or "",
    }


def _run_once(video_path: Path, analysis_profile: str, tracking_backend: str, *, disable_learned_ball: bool):
    previous = os.environ.get("BASKETBALL_DISABLE_LEARNED_BALL")
    try:
        if disable_learned_ball:
            os.environ["BASKETBALL_DISABLE_LEARNED_BALL"] = "1"
        else:
            os.environ.pop("BASKETBALL_DISABLE_LEARNED_BALL", None)

        prepared_path, prepared_label = prepare_analysis_source(video_path, analysis_profile)
        pipeline = BasketballAnalysisPipeline()
        result = pipeline.run(prepared_path, tracking_backend=tracking_backend)
        return result, prepared_path, prepared_label
    finally:
        if previous is None:
            os.environ.pop("BASKETBALL_DISABLE_LEARNED_BALL", None)
        else:
            os.environ["BASKETBALL_DISABLE_LEARNED_BALL"] = previous


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare learned ball detector vs fallback tracking on the same clip")
    parser.add_argument("--video", required=True, help="Path to local MP4 clip, or use 'latest-temp'")
    parser.add_argument("--analysis-profile", default="Standard Clip", choices=["Fast Preview", "Standard Clip", "Full Clip"])
    parser.add_argument("--tracking-backend", default="yolo_detect", help="Tracking backend to compare")
    args = parser.parse_args()

    video_path = _resolve_video_path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}. Replace the placeholder with a real MP4 path or use --video latest-temp."
        )

    learned_result, prepared_path, prepared_label = _run_once(
        video_path, args.analysis_profile, args.tracking_backend, disable_learned_ball=False
    )
    fallback_result, _, _ = _run_once(
        video_path, args.analysis_profile, args.tracking_backend, disable_learned_ball=True
    )

    print(f"Source video: {video_path}")
    print(f"Prepared clip: {prepared_path} ({prepared_label})\n")

    for summary in (
        _summarize(learned_result, "learned_on"),
        _summarize(fallback_result, "learned_off"),
    ):
        print(summary["mode"])
        print(f"  frames: {summary['frames']}")
        print(f"  ball_ratio: {summary['ball_ratio']}")
        print(f"  raw_ball_ratio: {summary['raw_ball_ratio']}")
        print(f"  learned_ball_ratio: {summary['learned_ball_ratio']}")
        print(f"  possession_ratio: {summary['possession_ratio']}")
        print(f"  events: {summary['events']}  passes: {summary['passes']}  shots: {summary['shots']}  turnovers: {summary['turnovers']}")
        print(f"  note: {summary['tracking_note']}\n")


if __name__ == "__main__":
    main()
