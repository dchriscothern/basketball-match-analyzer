from __future__ import annotations

import tempfile
from pathlib import Path

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


ANALYSIS_PROFILES: dict[str, dict[str, float | int | bool | None]] = {
    'Fast Preview': {
        'preview_only': True,
        'max_seconds': 6.0,
        'max_width': 640,
        'target_fps': 6.0,
    },
    'Standard Clip': {
        'preview_only': True,
        'max_seconds': 15.0,
        'max_width': 960,
        'target_fps': 10.0,
    },
    'Full Clip': {
        'preview_only': False,
        'max_seconds': None,
        'max_width': None,
        'target_fps': None,
    },
}


def analysis_profile_settings(profile_name: str) -> dict[str, float | int | bool | None]:
    return ANALYSIS_PROFILES.get(profile_name, ANALYSIS_PROFILES['Standard Clip'])


def prepare_fast_analysis_clip(
    source_path: str | Path,
    max_seconds: float = 6.0,
    max_width: int = 640,
    target_fps: float = 6.0,
) -> Path:
    source = Path(source_path)
    if cv2 is None:
        return source

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return source

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 1.0:
        fps = 24.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        capture.release()
        return source

    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if source_width <= 0 or source_height <= 0:
        capture.release()
        return source

    max_frames = max(1, int(max_seconds * fps))
    frames_to_process = min(total_frames, max_frames)
    frame_step = max(1, int(round(fps / max(target_fps, 1.0))))

    scale = min(1.0, max_width / float(source_width))
    target_width = max(2, int(round(source_width * scale)))
    target_height = max(2, int(round(source_height * scale)))
    if target_width % 2:
        target_width -= 1
    if target_height % 2:
        target_height -= 1

    if (
        frames_to_process >= total_frames
        and frame_step == 1
        and target_width == source_width
        and target_height == source_height
    ):
        capture.release()
        return source

    temp_dir = Path(tempfile.mkdtemp(prefix='basketball_preview_clip_'))
    clip_path = temp_dir / 'preview.mp4'
    writer = cv2.VideoWriter(
        str(clip_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        min(fps, max(target_fps, 1.0)),
        (target_width, target_height),
    )

    frame_index = 0
    written = 0
    try:
        while frame_index < frames_to_process:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index % frame_step == 0:
                if (target_width, target_height) != (source_width, source_height):
                    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                written += 1
            frame_index += 1
    finally:
        writer.release()
        capture.release()

    if written <= 0 or not clip_path.exists():
        return source
    return clip_path


def prepare_analysis_source(source_path: str | Path, profile_name: str) -> tuple[Path, str]:
    source = Path(source_path)
    settings = analysis_profile_settings(profile_name)
    if settings['max_seconds'] is None:
        return source, 'full clip'
    prepared = prepare_fast_analysis_clip(
        source,
        max_seconds=float(settings['max_seconds']),
        max_width=int(settings['max_width']),
        target_fps=float(settings['target_fps']),
    )
    return prepared, profile_name.lower()
