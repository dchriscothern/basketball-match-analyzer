from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

import cv2

try:
    from yt_dlp import YoutubeDL  # type: ignore
    from yt_dlp.utils import download_range_func  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    YoutubeDL = None
    download_range_func = None


def _iter_binary_candidates(binary_name: str):
    seen: set[str] = set()

    resolved = shutil.which(binary_name)
    if resolved:
        normalized = str(Path(resolved))
        seen.add(normalized.lower())
        yield Path(normalized)

    local_app_data = Path(os.environ.get('LOCALAPPDATA', ''))
    search_roots = [
        local_app_data / 'Microsoft' / 'WinGet' / 'Links',
        local_app_data / 'Microsoft' / 'WinGet' / 'Packages',
        local_app_data / 'Programs',
        Path(r'C:\Program Files'),
        Path(r'C:\Program Files (x86)'),
    ]

    direct_candidates = [
        root / binary_name
        for root in search_roots
    ] + [
        root / 'bin' / binary_name
        for root in search_roots
    ]

    for candidate in direct_candidates:
        if candidate.exists():
            normalized = str(candidate)
            lowered = normalized.lower()
            if lowered not in seen:
                seen.add(lowered)
                yield candidate

    for root in search_roots:
        try:
            root_exists = root.exists()
        except OSError:
            continue
        if not root_exists:
            continue
        try:
            matches = root.rglob(binary_name)
        except OSError:
            continue
        for candidate in matches:
            normalized = str(candidate)
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            yield candidate


def _find_binary(binary_name: str) -> str | None:
    for candidate in _iter_binary_candidates(binary_name):
        return str(candidate)
    return None


def _validate_video_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return False
        ok, frame = cap.read()
        return bool(ok and frame is not None and frame.size > 0)
    finally:
        cap.release()


def _extract_video_info(url: str, ffmpeg_path: str | None) -> dict | None:
    if YoutubeDL is None:
        return None
    options = {
        'skip_download': True,
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
    }
    if ffmpeg_path:
        options['ffmpeg_location'] = str(Path(ffmpeg_path).parent)
    try:
        with YoutubeDL(options) as ydl:
            return ydl.extract_info(url, download=False)
    except Exception:
        return None


def _download_direct_media_url(media_url: str, destination: Path) -> Path:
    request = urllib.request.Request(
        media_url,
        headers={
            'User-Agent': 'Mozilla/5.0',
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response, destination.open('wb') as fh:
        shutil.copyfileobj(response, fh)
    return destination


def _trim_video_segment(source_path: Path, start_s: float | None, end_s: float | None) -> Path:
    if start_s is None or end_s is None or end_s <= start_s:
        return source_path

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        return source_path

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        return source_path

    start_frame = max(0, int(start_s * fps))
    end_frame = max(start_frame + 1, int(end_s * fps))
    target_path = source_path.with_name('clip.mp4')
    writer = cv2.VideoWriter(
        str(target_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        return source_path

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    try:
        while frame_idx < end_frame:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            writer.write(frame)
            frame_idx += 1
    finally:
        capture.release()
        writer.release()

    if _validate_video_file(target_path):
        return target_path
    return source_path


def _build_download_option_sets(
    *,
    url: str,
    output_template: str,
    ffmpeg_path: str | None,
    preview_only: bool,
    allow_partial_clip: bool,
) -> list[dict]:
    ffmpeg_available = ffmpeg_path is not None
    if preview_only:
        progressive_formats = [
            'worst[ext=mp4][height<=480][protocol!=m3u8][protocol!=m3u8_native]',
            'worst[ext=mp4][height<=720][protocol!=m3u8][protocol!=m3u8_native]',
            'best[ext=mp4][height<=480][protocol!=m3u8][protocol!=m3u8_native]',
            'best[ext=mp4][protocol!=m3u8][protocol!=m3u8_native]',
        ]
        merged_format = 'worstvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best'
    else:
        progressive_formats = [
            'best[ext=mp4][protocol!=m3u8][protocol!=m3u8_native]/best[protocol!=m3u8][protocol!=m3u8_native]',
        ]
        merged_format = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best'

    clip_ranges = None
    if allow_partial_clip:
        info = _extract_video_info(url, None)
        if info is not None:
            section_start = info.get('section_start')
            section_end = info.get('section_end')
            if section_start is not None and section_end is not None and download_range_func is not None:
                clip_ranges = download_range_func(None, None, from_info=True)

    option_sets = []
    if clip_ranges is not None and ffmpeg_available:
        option_sets.append(
            {
                'format': 'worst[ext=mp4][protocol!=m3u8][protocol!=m3u8_native]/best[ext=mp4][height<=360][protocol!=m3u8][protocol!=m3u8_native]',
                'download_ranges': clip_ranges,
                'force_keyframes_at_cuts': True,
                'ffmpeg_location': str(Path(ffmpeg_path).parent),
                'outtmpl': output_template,
                'noplaylist': True,
                'quiet': True,
                'no_warnings': True,
            }
        )
    if ffmpeg_available:
        option_sets.append(
            {
                'format': merged_format,
                'merge_output_format': 'mp4',
                'ffmpeg_location': str(Path(ffmpeg_path).parent),
                **({'download_ranges': clip_ranges, 'force_keyframes_at_cuts': True} if clip_ranges is not None else {}),
                'outtmpl': output_template,
                'noplaylist': True,
                'quiet': True,
                'no_warnings': True,
            }
        )
    for progressive_format in progressive_formats:
        option_sets.append(
            {
                'format': progressive_format,
                'outtmpl': output_template,
                'noplaylist': True,
                'quiet': True,
                'no_warnings': True,
            }
        )
    option_sets.append(
        {
            'format': 'best',
            'outtmpl': output_template,
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
        }
    )
    return option_sets


def download_video_url(url: str, preview_only: bool = True, allow_partial_clip: bool = True) -> Path:
    if YoutubeDL is None:
        raise RuntimeError('yt-dlp is not installed. Install it to enable direct WNBA video URL downloads.')

    temp_dir = Path(tempfile.mkdtemp(prefix='basketball_url_'))
    output_template = str(temp_dir / 'source.%(ext)s')
    ffmpeg_path = _find_binary('ffmpeg.exe') or _find_binary('ffmpeg')
    ffprobe_path = _find_binary('ffprobe.exe') or _find_binary('ffprobe')
    ffmpeg_available = ffmpeg_path is not None
    info = _extract_video_info(url, None)
    direct_media_url = info.get('url') if isinstance(info, dict) else None
    is_clip_url = '/clip/' in url.lower()

    if is_clip_url and isinstance(direct_media_url, str) and direct_media_url:
        direct_target = temp_dir / 'source.mp4'
        downloaded = _download_direct_media_url(direct_media_url, direct_target)
        if _validate_video_file(downloaded):
            downloaded = _trim_video_segment(
                downloaded,
                info.get('section_start') if isinstance(info, dict) else None,
                info.get('section_end') if isinstance(info, dict) else None,
            )
            return downloaded

    if not allow_partial_clip and isinstance(direct_media_url, str) and direct_media_url:
        direct_target = temp_dir / 'source.mp4'
        try:
            downloaded = _download_direct_media_url(direct_media_url, direct_target)
            if _validate_video_file(downloaded):
                if isinstance(info, dict):
                    downloaded = _trim_video_segment(
                        downloaded,
                        info.get('section_start'),
                        info.get('section_end'),
                    )
                return downloaded
        except Exception:
            pass

    option_sets = _build_download_option_sets(
        url=url,
        output_template=output_template,
        ffmpeg_path=ffmpeg_path,
        preview_only=preview_only,
        allow_partial_clip=allow_partial_clip,
    )

    last_error = None
    downloaded = None
    for options in option_sets:
        try:
            with YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded = Path(ydl.prepare_filename(info))
                if downloaded.suffix.lower() != '.mp4':
                    alt = downloaded.with_suffix('.mp4')
                    if alt.exists():
                        downloaded = alt
                if not downloaded.exists():
                    mp4_files = sorted(temp_dir.glob('*.mp4'))
                    if mp4_files:
                        downloaded = mp4_files[0]
                if downloaded.exists() and _validate_video_file(downloaded):
                    return downloaded
                last_error = RuntimeError(
                    f'The downloaded video file was not decodable: {downloaded}'
                )
        except Exception as exc:  # pragma: no cover - depends on remote source behavior
            last_error = exc

    if allow_partial_clip:
        try:
            return download_video_url(url, preview_only=preview_only, allow_partial_clip=False)
        except Exception as exc:  # pragma: no cover - only used as a fallback path
            last_error = exc
    else:
        if isinstance(direct_media_url, str) and direct_media_url:
            direct_target = temp_dir / 'source.mp4'
            try:
                downloaded = _download_direct_media_url(direct_media_url, direct_target)
                if _validate_video_file(downloaded):
                    if isinstance(info, dict):
                        downloaded = _trim_video_segment(
                            downloaded,
                            info.get('section_start'),
                            info.get('section_end'),
                        )
                    return downloaded
                last_error = RuntimeError(f'The direct media download was not decodable: {downloaded}')
            except Exception as exc:  # pragma: no cover - network/source dependent
                last_error = exc

    if downloaded is None:
        if not ffmpeg_available:
            raise RuntimeError(
                'Could not download a simple MP4 stream from that URL without ffmpeg. '
                'Try a different clip URL or use Raw MP4 Upload.'
            ) from last_error
        raise RuntimeError('The video URL download failed.') from last_error

    if downloaded.exists() and _validate_video_file(downloaded):
        return downloaded

    mp4_files = [path for path in sorted(temp_dir.glob('*.mp4')) if _validate_video_file(path)]
    if mp4_files:
        return mp4_files[0]

    if ffmpeg_available and ffprobe_path is None:
        raise RuntimeError(
            'A video file was downloaded, but it was not decodable. ffmpeg was found but ffprobe was not. '
            'Try restarting the app shell or use Raw MP4 Upload.'
        ) from last_error
    raise RuntimeError(
        'The downloaded clip was corrupt or not decodable. Try a different URL or use Raw MP4 Upload.'
    ) from last_error
