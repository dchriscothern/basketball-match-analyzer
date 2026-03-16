from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from basketball_analyzer.pipeline import BasketballAnalysisPipeline


st.set_page_config(page_title='Basketball Match Analyzer', layout='wide')


def _run_pipeline(source_mode: str, uploaded_file=None) -> tuple[object, str]:
    pipeline = BasketballAnalysisPipeline()
    if source_mode == 'Demo Sequence':
        return pipeline.run('demo'), 'demo'

    if uploaded_file is None:
        raise ValueError('No file provided.')

    suffix = Path(uploaded_file.name).suffix or '.bin'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name
    return pipeline.run(temp_path), temp_path


st.title('Basketball Match Analyzer')
st.caption('A basketball-focused multi-agent demo for possessions, passes, turnovers, steals, shots, rebounds, and automated game summaries.')

with st.sidebar:
    st.header('Input')
    mode = st.radio('Choose source', ['Demo Sequence', 'Tracked JSON Upload', 'Raw MP4 Upload'])
    upload = None
    if mode == 'Tracked JSON Upload':
        upload = st.file_uploader('Upload tracking JSON', type=['json'])
    elif mode == 'Raw MP4 Upload':
        upload = st.file_uploader('Upload MP4', type=['mp4'])

    run_clicked = st.button('Run Analysis', type='primary', use_container_width=True)
    st.caption('Raw MP4 uses a crude OpenCV prototype if available; otherwise it falls back to demo mode. JSON upload is the best real-data path right now.')

if run_clicked:
    try:
        result, source_ref = _run_pipeline(mode, upload)
        st.success(f'Analysis complete from: {source_ref}')

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
            ], use_container_width=True)
        with tab2:
            st.dataframe([
                {'player_id': player_id, **stats}
                for player_id, stats in sorted(result.player_stats.items())
            ], use_container_width=True)
        with tab3:
            st.dataframe([
                {'team_id': team_id, **stats}
                for team_id, stats in sorted(result.team_stats.items())
            ], use_container_width=True)

        with st.expander('Possession Timeline'):
            st.dataframe([
                {
                    'frame_index': item.frame_index,
                    'timestamp_s': item.timestamp_s,
                    'player_id': item.player_id,
                    'team_id': item.team_id,
                }
                for item in result.possession_timeline
            ], use_container_width=True)
    except Exception as exc:
        st.error(f'Analysis failed: {exc}')
else:
    st.info('Choose a source and click Run Analysis. Demo Sequence is the fastest way to see the full basketball pipeline working.')
