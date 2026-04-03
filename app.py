from __future__ import annotations

import base64
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from basketball_analyzer.analysis_profiles import analysis_profile_settings, prepare_analysis_source
from basketball_analyzer.pipeline import BasketballAnalysisPipeline
from basketball_analyzer.video_sources import download_video_url


st.set_page_config(page_title='Basketball Video Intelligence', layout='wide')

DEFAULT_DEMO_URL = 'https://youtube.com/clip/UgkxKugBQoEq_ZSwC5BWPceIG1gu0HSiaxQB?si=76QMLAn41-b89zJn'
DEFAULT_DEMO_MODE = 'WNBA Video URL'
DEFAULT_DEMO_PROFILE = 'Standard Clip'
DEFAULT_DEMO_BACKEND = 'yolo_detect'


@st.cache_resource(max_entries=1)
def _get_pipeline(version: int = 8) -> BasketballAnalysisPipeline:
    return BasketballAnalysisPipeline()


def _render_preview(preview_path: Path) -> None:
    suffix = preview_path.suffix.lower()
    if suffix == '.gif':
        encoded = base64.b64encode(preview_path.read_bytes()).decode('ascii')
        st.markdown(
            f"<img src='data:image/gif;base64,{encoded}' style='width:100%; border-radius:12px;' />",
            unsafe_allow_html=True,
        )
        return
    st.image(str(preview_path), caption='Rendered tracking + events animation', width='stretch')


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --page-bg: #f4efe4;
            --panel: rgba(255, 252, 246, 0.9);
            --panel-strong: #fffaf0;
            --line: rgba(93, 64, 29, 0.14);
            --ink: #1d1a17;
            --muted: #5f564c;
            --accent: #c96b28;
            --accent-deep: #8a3f16;
            --success: #1f6b4f;
            --shadow: 0 16px 44px rgba(72, 44, 18, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(239, 164, 84, 0.22), transparent 28%),
                radial-gradient(circle at top right, rgba(110, 151, 107, 0.14), transparent 26%),
                linear-gradient(180deg, #fbf7ef 0%, var(--page-bg) 100%);
            color: var(--ink);
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            font-family: Georgia, "Aptos Display", "Trebuchet MS", serif;
            color: var(--ink);
            letter-spacing: -0.02em;
        }

        .demo-hero {
            background:
                linear-gradient(135deg, rgba(255, 249, 238, 0.95), rgba(250, 239, 220, 0.92)),
                linear-gradient(120deg, rgba(201, 107, 40, 0.08), rgba(31, 107, 79, 0.04));
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 1.6rem 1.7rem;
            box-shadow: var(--shadow);
            margin-bottom: 1.2rem;
        }

        .demo-kicker {
            display: inline-block;
            margin-bottom: 0.75rem;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            background: rgba(201, 107, 40, 0.12);
            color: var(--accent-deep);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .demo-lead {
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.55;
            margin: 0.45rem 0 0;
            max-width: 62rem;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }

        .demo-pill {
            padding: 0.45rem 0.72rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(93, 64, 29, 0.10);
            color: var(--muted);
            font-size: 0.88rem;
        }

        .demo-card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 0.9rem;
            margin: 0.9rem 0 1.1rem;
        }

        .demo-card, .metric-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            box-shadow: 0 8px 24px rgba(72, 44, 18, 0.06);
        }

        .demo-card h4, .metric-card h4 {
            margin: 0 0 0.35rem;
            font-size: 0.98rem;
            color: var(--ink);
        }

        .demo-card p, .metric-card p {
            margin: 0;
            color: var(--muted);
            font-size: 0.91rem;
            line-height: 1.45;
        }

        .metric-card .value {
            font-size: 1.6rem;
            line-height: 1.05;
            color: var(--accent-deep);
            font-weight: 700;
            margin-bottom: 0.28rem;
        }

        .section-label {
            color: var(--accent-deep);
            font-size: 0.84rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .callout {
            border-radius: 20px;
            padding: 0.95rem 1rem;
            border: 1px solid var(--line);
            background: rgba(255, 252, 246, 0.86);
            color: var(--muted);
            margin: 0.5rem 0 1rem;
        }

        .callout strong {
            color: var(--ink);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_card(label: str, value: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="value">{value}</div>
            <h4>{label}</h4>
            <p>{detail}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_default_state() -> None:
    st.markdown('<div class="section-label">Purpose And Promise</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="demo-card-grid">
            <div class="demo-card">
                <h4>Purpose</h4>
                <p>Turn basketball video into possessions, events, team context, and a visual replay layer that is fast to review.</p>
            </div>
            <div class="demo-card">
                <h4>Why It Matters</h4>
                <p>Coaches, analysts, and operators should not need to tag every play by hand just to understand what happened.</p>
            </div>
            <div class="demo-card">
                <h4>What To Watch</h4>
                <p>Look for player movement, possession flow, event detection, and the rendered court replay more than perfect raw tracking.</p>
            </div>
            <div class="demo-card">
                <h4>What We Are Learning</h4>
                <p>This demo shows what already works now and where better ball tracking, benchmarking, and richer game understanding come next.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">Choose Your Demo Path</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="demo-card-grid">
            <div class="demo-card">
                <h4>Demo Sequence</h4>
                <p>Best way to introduce the concept, the flow of analysis, and the final replay experience.</p>
            </div>
            <div class="demo-card">
                <h4>Tracked JSON Upload</h4>
                <p>Best way to show the downstream value with cleaner detections and a more stable event story.</p>
            </div>
            <div class="demo-card">
                <h4>Raw MP4 / WNBA URL</h4>
                <p>Best way to show the live prototype pipeline and the future direction, even if ball tracking is still visually noisy.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info('Pick a source in the sidebar and click Run Analysis. The main thing to sell is the analysis outcome and replay layer, not every raw detection detail.')


def _render_story_cards() -> None:
    st.markdown('<div class="section-label">Purpose, Value, And Next Learnings</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="demo-card-grid">
            <div class="demo-card">
                <h4>Purpose</h4>
                <p>Convert raw basketball footage into a usable layer of possessions, events, context, and replay.</p>
            </div>
            <div class="demo-card">
                <h4>Why It Matters</h4>
                <p>Faster film review, lighter manual tagging, clearer coaching conversations, and a stronger foundation for automated analysis.</p>
            </div>
            <div class="demo-card">
                <h4>What To Watch</h4>
                <p>The best signals in this demo are the rendered play view, possession flow, and event timeline rather than perfect ball precision.</p>
            </div>
            <div class="demo-card">
                <h4>What We Learn Next</h4>
                <p>Better ball tracking, event accuracy benchmarking, richer player and team insights, and eventually more production-ready workflows.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">Future Phases</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="demo-card-grid">
            <div class="demo-card">
                <h4>Phase 1: D2 Essentials</h4>
                <p>Prioritize the lowest-cost, highest-trust outputs: possessions, event timeline, team context, replay, clip exports, and coach-ready summaries.</p>
            </div>
            <div class="demo-card">
                <h4>Phase 2: Decision Support</h4>
                <p>Add lineup views, shot maps, simple shot quality, transition and turnover patterns, plus shareable game and player reports.</p>
            </div>
            <div class="demo-card">
                <h4>Phase 3: Advanced AI Insights</h4>
                <p>Explore gravity, reaction speed, defensive synchronization, and other higher-cost metrics once the tracking layer is strong enough to support them.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _event_breakdown(events) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        counts[event.event_type] = counts.get(event.event_type, 0) + 1
    return counts


def _run_pipeline(
    source_mode: str,
    uploaded_file=None,
    video_url: str | None = None,
    analysis_profile: str = 'Standard Clip',
    tracking_backend: str = 'auto',
):
    pipeline = _get_pipeline(version=8)
    render_dir = Path(tempfile.mkdtemp(prefix='basketball_render_'))
    render_path = render_dir / 'analysis.gif'

    if source_mode == 'Demo Sequence':
        return pipeline.run('demo', render_output_path=render_path), 'demo'
    if source_mode == 'WNBA Video URL':
        if not video_url:
            raise ValueError('No video URL provided.')
        settings = analysis_profile_settings(analysis_profile)
        temp_path = download_video_url(video_url, preview_only=bool(settings['preview_only']))
        analysis_path, profile_label = prepare_analysis_source(temp_path, analysis_profile)
        return (
            pipeline.run(analysis_path, render_output_path=render_path, tracking_backend=tracking_backend),
            f'{temp_path} ({profile_label})',
        )

    if uploaded_file is None:
        raise ValueError('No file provided.')

    suffix = Path(uploaded_file.name).suffix or '.bin'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name
    analysis_path = Path(temp_path)
    if analysis_path.suffix.lower() == '.mp4':
        analysis_path, profile_label = prepare_analysis_source(analysis_path, analysis_profile)
        return (
            pipeline.run(analysis_path, render_output_path=render_path, tracking_backend=tracking_backend),
            f'{temp_path} ({profile_label})',
        )
    return pipeline.run(temp_path, render_output_path=render_path, tracking_backend=tracking_backend), temp_path


def _init_demo_state() -> None:
    st.session_state.setdefault('source_mode', DEFAULT_DEMO_MODE)
    st.session_state.setdefault('video_url_input', DEFAULT_DEMO_URL)
    st.session_state.setdefault('analysis_profile_input', DEFAULT_DEMO_PROFILE)
    st.session_state.setdefault('tracking_backend_input', DEFAULT_DEMO_BACKEND)
    st.session_state.setdefault('auto_demo_pending', True)


_inject_styles()
_init_demo_state()
st.markdown(
    """
    <section class="demo-hero">
        <div class="demo-kicker">Basketball Video Intelligence Demo</div>
        <h1>Basketball Video Intelligence</h1>
        <p class="demo-lead">
            This demo turns basketball footage into possessions, events, team context, and a replay-ready court view. The goal is to show why automated game understanding matters,
            what is already useful today, and what the next iterations can unlock for coaching, scouting, and operations.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header('Demo Controls')
    st.caption('For the smoothest walkthrough, start with Demo Sequence. Use a short real clip only when you want to show the current prototype pipeline and where it is heading.')
    mode = st.radio(
        'Choose source',
        ['Demo Sequence', 'Tracked JSON Upload', 'Raw MP4 Upload', 'WNBA Video URL'],
        key='source_mode',
    )
    upload = None
    video_url = None
    analysis_profile = st.session_state.get('analysis_profile_input', DEFAULT_DEMO_PROFILE)
    tracking_backend = st.session_state.get('tracking_backend_input', DEFAULT_DEMO_BACKEND)
    if mode == 'Tracked JSON Upload':
        upload = st.file_uploader('Upload tracking JSON', type=['json'])
    elif mode == 'Raw MP4 Upload':
        upload = st.file_uploader('Upload MP4', type=['mp4'])
        analysis_profile = st.selectbox(
            'Analysis quality',
            ['Fast Preview', 'Standard Clip', 'Full Clip'],
            index=['Fast Preview', 'Standard Clip', 'Full Clip'].index(st.session_state.get('analysis_profile_input', DEFAULT_DEMO_PROFILE)),
            key='analysis_profile_input',
        )
        tracking_backend = st.selectbox(
            'Tracking backend',
            [
                ('Auto', 'auto'),
                ('Prototype CV', 'prototype_cv'),
                ('YOLO Detect', 'yolo_detect'),
                ('YOLO + BoT-SORT (experimental)', 'yolo_botsort'),
                ('YOLO + ByteTrack (experimental)', 'yolo_bytetrack'),
            ],
            index=[item[1] for item in [
                ('Auto', 'auto'),
                ('Prototype CV', 'prototype_cv'),
                ('YOLO Detect', 'yolo_detect'),
                ('YOLO + BoT-SORT (experimental)', 'yolo_botsort'),
                ('YOLO + ByteTrack (experimental)', 'yolo_bytetrack'),
            ]].index(st.session_state.get('tracking_backend_input', DEFAULT_DEMO_BACKEND)),
            format_func=lambda item: item[0],
            key='tracking_backend_input',
        )[1]
        st.caption('Recommended for demo: Standard Clip + YOLO Detect. It is the steadiest current real-video path.')
    elif mode == 'WNBA Video URL':
        video_url = st.text_input(
            'Paste WNBA video URL',
            placeholder='https://www.youtube.com/watch?v=...',
            key='video_url_input',
        )
        analysis_profile = st.selectbox(
            'Analysis quality',
            ['Fast Preview', 'Standard Clip', 'Full Clip'],
            index=['Fast Preview', 'Standard Clip', 'Full Clip'].index(st.session_state.get('analysis_profile_input', DEFAULT_DEMO_PROFILE)),
            key='analysis_profile_input',
        )
        tracking_backend = st.selectbox(
            'Tracking backend',
            [
                ('Auto', 'auto'),
                ('Prototype CV', 'prototype_cv'),
                ('YOLO Detect', 'yolo_detect'),
                ('YOLO + BoT-SORT (experimental)', 'yolo_botsort'),
                ('YOLO + ByteTrack (experimental)', 'yolo_bytetrack'),
            ],
            index=[item[1] for item in [
                ('Auto', 'auto'),
                ('Prototype CV', 'prototype_cv'),
                ('YOLO Detect', 'yolo_detect'),
                ('YOLO + BoT-SORT (experimental)', 'yolo_botsort'),
                ('YOLO + ByteTrack (experimental)', 'yolo_bytetrack'),
            ]].index(st.session_state.get('tracking_backend_input', DEFAULT_DEMO_BACKEND)),
            format_func=lambda item: item[0],
            key='tracking_backend_input',
        )[1]
        st.caption('Use a short highlight or possession clip so the demo stays responsive.')

    run_clicked = st.button('Run Analysis', type='primary', width='stretch')
    st.caption('Standard Clip is the best balance for a live demo. Keep the controls simple unless someone specifically asks about the pipeline.')

auto_run = (
    st.session_state.get('auto_demo_pending', False)
    and mode == DEFAULT_DEMO_MODE
    and st.session_state.get('video_url_input') == DEFAULT_DEMO_URL
)
if auto_run:
    st.session_state.auto_demo_pending = False
run_requested = run_clicked or auto_run

if run_requested:
    try:
        result, source_ref = _run_pipeline(mode, upload, video_url, analysis_profile, tracking_backend)
        event_counts = _event_breakdown(result.events)
        ball_frames = sum(1 for frame in result.frames if frame.ball is not None)
        ball_ratio = (ball_frames / len(result.frames)) if result.frames else 0.0
        source_label = mode if mode != 'WNBA Video URL' else 'WNBA URL'

        if auto_run:
            st.info('Loaded the default demo clip automatically. You can switch sources or paste a different URL anytime.')
        st.success(f'Analysis complete from: {source_ref}')
        st.markdown('<div class="section-label">Run Summary</div>', unsafe_allow_html=True)
        top_metrics = st.columns(4)
        with top_metrics[0]:
            _render_metric_card('Source', source_label, f'Profile: {analysis_profile}')
        with top_metrics[1]:
            _render_metric_card('Frames', str(len(result.frames)), f'{ball_frames} frames with ball signal')
        with top_metrics[2]:
            _render_metric_card('Events', str(len(result.events)), 'Automated pass / turnover / shot timeline')
        with top_metrics[3]:
            _render_metric_card('Coverage', f'{ball_ratio:.0%}', f'{len(result.player_stats)} tracked players in final output')

        if result.tracking_note:
            if mode == 'Raw MP4 Upload':
                st.warning(f'Prototype tracking note: {result.tracking_note}')
            else:
                st.info(result.tracking_note)
        if result.calibration_note:
            st.caption(result.calibration_note)

        st.markdown('<div class="section-label">Replay View</div>', unsafe_allow_html=True)
        if result.rendered_media_path and Path(result.rendered_media_path).exists():
            media_path = Path(result.rendered_media_path)
            preview_path = Path(result.rendered_preview_path) if result.rendered_preview_path else media_path
            if preview_path.exists():
                _render_preview(preview_path)
                st.caption('Rendered tracking + event overlay preview')
            st.caption(f'Rendered artifact: {media_path}')
            if result.rendered_media_note:
                st.info(result.rendered_media_note)
            st.download_button(
                f'Download rendered {media_path.suffix.lower()}',
                data=media_path.read_bytes(),
                file_name=media_path.name,
                mime='video/mp4' if media_path.suffix.lower() == '.mp4' else 'image/gif',
                width='stretch',
            )
        else:
            st.info('Rendered animation was not available for this run.')

        summary_tab, events_tab, players_tab, teams_tab = st.tabs(['Coach Summary', 'Event Timeline', 'Player Insights', 'Team Insights'])
        with summary_tab:
            _render_story_cards()
            st.markdown('<div class="section-label">Automated Report</div>', unsafe_allow_html=True)
            st.write(result.report_text)
            summary_cols = st.columns(4)
            summary_cols[0].metric('Possession Frames', len(result.possession_timeline))
            summary_cols[1].metric('Tracked Players', len(result.player_stats))
            summary_cols[2].metric('Teams', len(result.team_stats))
            summary_cols[3].metric('Event Types', len(event_counts))
            if event_counts:
                st.markdown('<div class="section-label">Event Mix</div>', unsafe_allow_html=True)
                mix_cols = st.columns(min(4, len(event_counts)))
                for idx, (event_type, count) in enumerate(sorted(event_counts.items())):
                    mix_cols[idx % len(mix_cols)].metric(event_type.replace('_', ' ').title(), count)
            with st.expander('Possession Timeline'):
                st.dataframe([
                    {
                        'frame_index': item.frame_index,
                        'timestamp_s': item.timestamp_s,
                        'player_id': item.player_id,
                        'team_id': item.team_id,
                    }
                    for item in result.possession_timeline
                ], width='stretch')
        with events_tab:
            st.dataframe([
                {
                    'event_type': event.event_type,
                    'timestamp_s': event.timestamp_s,
                    'team_id': event.team_id,
                    'player_id': event.player_id,
                    'secondary_player_id': event.secondary_player_id,
                    'metadata': json.dumps(event.metadata),
                }
                for event in result.events
            ], width='stretch')
        with players_tab:
            st.dataframe([
                {'player_id': player_id, **stats}
                for player_id, stats in sorted(result.player_stats.items())
            ], width='stretch')
        with teams_tab:
            st.dataframe([
                {'team_id': team_id, **stats}
                for team_id, stats in sorted(result.team_stats.items())
            ], width='stretch')
    except Exception as exc:
        st.error(f'Analysis failed: {exc}')
else:
    _render_default_state()

