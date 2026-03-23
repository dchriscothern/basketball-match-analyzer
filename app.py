from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

import streamlit as st

from basketball_analyzer.analysis_profiles import analysis_profile_settings, prepare_analysis_source
from basketball_analyzer.pipeline import BasketballAnalysisPipeline
from basketball_analyzer.video_sources import download_video_url


st.set_page_config(page_title='Basketball Match Analyzer', layout='wide')


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


st.title('Basketball Match Analyzer')
st.caption('A basketball-focused multi-agent demo for possessions, passes, turnovers, steals, shots, rebounds, automated summaries, and rendered play overlays. The synthetic court renderer now uses a WNBA court preset.')

with st.sidebar:
    st.header('Input')
    mode = st.radio('Choose source', ['Demo Sequence', 'Tracked JSON Upload', 'Raw MP4 Upload', 'WNBA Video URL'])
    upload = None
    video_url = None
    analysis_profile = 'Standard Clip'
    tracking_backend = 'yolo_detect'
    if mode == 'Tracked JSON Upload':
        upload = st.file_uploader('Upload tracking JSON', type=['json'])
    elif mode == 'Raw MP4 Upload':
        upload = st.file_uploader('Upload MP4', type=['mp4'])
        analysis_profile = st.selectbox('Analysis quality', ['Fast Preview', 'Standard Clip', 'Full Clip'], index=1)
        tracking_backend = st.selectbox(
            'Tracking backend',
            [
                ('Auto', 'auto'),
                ('Prototype CV', 'prototype_cv'),
                ('YOLO Detect', 'yolo_detect'),
                ('YOLO + BoT-SORT (experimental)', 'yolo_botsort'),
                ('YOLO + ByteTrack (experimental)', 'yolo_bytetrack'),
            ],
            index=2,
            format_func=lambda item: item[0],
        )[1]
    elif mode == 'WNBA Video URL':
        video_url = st.text_input('Paste WNBA video URL', placeholder='https://www.youtube.com/watch?v=...')
        analysis_profile = st.selectbox('Analysis quality', ['Fast Preview', 'Standard Clip', 'Full Clip'], index=1)
        tracking_backend = st.selectbox(
            'Tracking backend',
            [
                ('Auto', 'auto'),
                ('Prototype CV', 'prototype_cv'),
                ('YOLO Detect', 'yolo_detect'),
                ('YOLO + BoT-SORT (experimental)', 'yolo_botsort'),
                ('YOLO + ByteTrack (experimental)', 'yolo_bytetrack'),
            ],
            index=2,
            format_func=lambda item: item[0],
        )[1]

    run_clicked = st.button('Run Analysis', type='primary', width='stretch')
    st.caption('Raw MP4 can now use Auto, Prototype CV, YOLO Detect, YOLO + BoT-SORT, or YOLO + ByteTrack. Choose Fast Preview for speed, Standard Clip for better coverage, or Full Clip for the heaviest run.')

if run_clicked:
    try:
        result, source_ref = _run_pipeline(mode, upload, video_url, analysis_profile, tracking_backend)
        st.success(f'Analysis complete from: {source_ref}')

        if result.tracking_note:
            if mode == 'Raw MP4 Upload':
                st.warning(result.tracking_note)
            else:
                st.info(result.tracking_note)
        if result.calibration_note:
            st.caption(result.calibration_note)

        st.subheader('Rendered Analysis')
        if result.rendered_media_path and Path(result.rendered_media_path).exists():
            media_path = Path(result.rendered_media_path)
            preview_path = Path(result.rendered_preview_path) if result.rendered_preview_path else media_path
            if preview_path.exists():
                _render_preview(preview_path)
                st.caption('Rendered tracking + events animation')
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

        st.subheader('Automated Report')
        st.write(result.report_text)

        metric_cols = st.columns(4)
        metric_cols[0].metric('Frames', len(result.frames))
        metric_cols[1].metric('Possession Frames', len(result.possession_timeline))
        metric_cols[2].metric('Events', len(result.events))
        metric_cols[3].metric('Players', len(result.player_stats))

        tab1, tab2, tab3 = st.tabs(['Events', 'Player Stats', 'Team Stats'])
        with tab1:
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
        with tab2:
            st.dataframe([
                {'player_id': player_id, **stats}
                for player_id, stats in sorted(result.player_stats.items())
            ], width='stretch')
        with tab3:
            st.dataframe([
                {'team_id': team_id, **stats}
                for team_id, stats in sorted(result.team_stats.items())
            ], width='stretch')

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
    except Exception as exc:
        st.error(f'Analysis failed: {exc}')
else:
    st.info('Choose a source and click Run Analysis. Demo Sequence is the fastest way to see the full basketball pipeline working.')
