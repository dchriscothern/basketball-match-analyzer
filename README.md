# Basketball Match Analyzer

A basketball-specific multi-agent video analysis repo inspired by multi-agent football analysis workflows, but redesigned for possessions, passes, turnovers, steals, shots, rebounds, automated game summaries, and rendered analysis overlays.

## System Pipeline

`MP4 -> VisionAgent -> BallInterpolator -> TeamClassifier -> EventsAgent -> AnalyticsAgent -> ReportingAgent -> VideoRendererAgent -> Game Report + Analysis GIF`

## What You Can See In Action Right Now

This repo now has both a CLI demo and a Streamlit app.

- `demo` input generates a synthetic basketball possession sequence
- `EventsAgent` detects passes, turnovers, steals, shot attempts, and rebounds
- `AnalyticsAgent` rolls those into player and team stats
- `ReportingAgent` writes a short game-style summary
- `VideoRendererAgent` creates a rendered basketball-court animation with players, ball, possession, and event overlays
- for uploaded MP4s, the renderer will attempt a source-video overlay export when OpenCV is installed
- Streamlit lets you run the demo visually and inspect events/stats in one place
- Streamlit also lets you preview and download the rendered `analysis.gif`

## Run The Demo In The Terminal

```powershell
cd C:\GitHub\basketball-match-analyzer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m basketball_analyzer.cli --video demo --output outputs
```

This now writes a rendered artifact alongside the JSON outputs and report. Today that will usually be `analysis.gif`, and for MP4 uploads it can become an overlay `analysis.mp4` when OpenCV is available.

## Run The Streamlit App

```powershell
cd C:\GitHub\basketball-match-analyzer
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
streamlit run app.py --server.port 8513
```

## Input Modes

1. `demo`
   The fastest way to see the whole system working.

2. `Tracked JSON`
   The best real-data path right now. If you already have external tracking output, upload a JSON file and let the downstream agents do eventing, stats, reporting, and rendering.

3. `Raw MP4`
   A crude first-pass OpenCV prototype is included for raw video. It is not production-quality tracking. It sparsely samples frames, guesses moving player blobs, tries to find an orange ball, and uses left/right court halves as a temporary team proxy. If that fails, the repo falls back to the demo sequence.

## Current Renderer Scope

The renderer currently creates a synthetic court animation (`analysis.gif`) from the tracked positions and detected events. That makes the multi-agent pipeline easy to validate visually.

For uploaded MP4s, the app now attempts a sampled source-video overlay export if OpenCV is installed in the environment. If OpenCV is missing, or raw tracking is weak, it falls back to the synthetic renderer instead.

This is still an MVP renderer, not a production broadcast-analysis stack. The next renderer milestone is a more robust, frame-accurate overlay on top of stronger tracking outputs.

## What Still Needs Real Model Integrations

- robust player / ball detection from raw MP4
- jersey / team classification from actual crops or embeddings
- court calibration for x/y normalization and zone logic
- made / missed shot confirmation from trajectory + hoop zone
- assists, blocks, screens, fouls, lineup analysis
- rendered overlays on the real source video rather than the synthetic court canvas
- optional LLM-backed reporting with Groq / OpenAI / LangChain
