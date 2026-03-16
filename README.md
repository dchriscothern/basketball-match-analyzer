# Basketball Match Analyzer

A basketball-specific multi-agent video analysis repo inspired by multi-agent football analysis workflows, but redesigned for possessions, passes, turnovers, steals, shots, rebounds, and automated game summaries.

## System Pipeline

`MP4 -> VisionAgent -> BallInterpolator -> TeamClassifier -> EventsAgent -> AnalyticsAgent -> ReportingAgent -> Game Report`

## What You Can See In Action Right Now

This repo now has both a CLI demo and a Streamlit app.

- `demo` input generates a synthetic basketball possession sequence
- `EventsAgent` detects passes, turnovers, steals, shot attempts, and rebounds
- `AnalyticsAgent` rolls those into player and team stats
- `ReportingAgent` writes a short game-style summary
- Streamlit lets you run the demo visually and inspect events/stats in one place

## Run The Demo In The Terminal

```powershell
cd C:\GitHubasketball-match-analyzer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m basketball_analyzer.cli --video demo --output outputs
```

## Run The Streamlit App

```powershell
cd C:\GitHubasketball-match-analyzer
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
streamlit run app.py
```

## Input Modes

1. `demo`
   The fastest way to see the whole system working.

2. `Tracked JSON`
   The best real-data path right now. If you already have external tracking output, upload a JSON file and let the downstream agents do eventing, stats, and reporting.

3. `Raw MP4`
   A crude first-pass OpenCV prototype is included for raw video. It is not production-quality tracking. It sparsely samples frames, guesses moving player blobs, tries to find an orange ball, and uses left/right court halves as a temporary team proxy. If that fails, the repo falls back to the demo sequence.

## Current Basketball Event Logic

Implemented now:

- ball-handler detection from nearest-player-to-ball distance
- temporal smoothing of control changes
- pass detection when control changes within the same team
- turnover and steal detection when control flips teams
- shot-attempt tagging from ball-state metadata in the demo feed
- rebound classification after a shot window closes

## What Still Needs Real Model Integrations

- robust player / ball detection from raw MP4
- jersey / team classification from actual crops or embeddings
- court calibration for x/y normalization and zone logic
- made / missed shot confirmation from trajectory + hoop zone
- assists, blocks, screens, fouls, lineup analysis
- optional LLM-backed reporting with Groq / OpenAI / LangChain

## Suggested Next Milestones

1. Replace the prototype MP4 extractor with YOLO / ByteTrack / court calibration.
2. Add shot zones, makes/misses, and rebounds from trajectory logic.
3. Add an Orchestrator Agent and messaging hooks.
4. Add Telegram / Slack / Discord delivery for automated reports.
