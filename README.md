# Basketball Match Analyzer

A basketball-specific multi-agent video analysis repo inspired by multi-agent football analysis workflows, but redesigned for possessions, passes, turnovers, steals, shots, rebounds, and automated game summaries.

## System Pipeline

`MP4 -> VisionAgent -> BallInterpolator -> TeamClassifier -> EventsAgent -> AnalyticsAgent -> ReportingAgent -> Game Report`

## What You Can See In Action Right Now

This repo now has a runnable demo mode.

- `demo` input generates a synthetic basketball possession sequence
- `EventsAgent` detects passes, turnovers, steals, shot attempts, and rebounds
- `AnalyticsAgent` rolls those into player and team stats
- `ReportingAgent` writes a short game-style summary

That means you can run the full pipeline today even before real video detection is wired in.

## Demo Run

```powershell
cd C:\GitHubasketball-match-analyzer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m basketball_analyzer.cli --video demo --output outputs
```

Outputs written to `outputs/`:

- `report.txt`
- `events.json`
- `possession_timeline.json`
- `player_stats.json`
- `team_stats.json`

## Current Basketball Event Logic

Implemented now:

- ball-handler detection from nearest-player-to-ball distance
- temporal smoothing of control changes
- pass detection when control changes within the same team
- turnover and steal detection when control flips teams
- shot-attempt tagging from ball-state metadata in the demo feed
- rebound classification after a shot window closes

## What Still Needs Real Model Integrations

- computer vision player / ball detection from raw MP4
- jersey / team classification from actual crops or embeddings
- court calibration for x/y normalization and zone logic
- made / missed shot confirmation from trajectory + hoop zone
- assists, blocks, screens, fouls, lineup analysis
- optional LLM-backed reporting with Groq / OpenAI / LangChain

## Suggested Next Milestones

1. Replace `VisionAgent` with YOLO / OpenCV / tracker integration.
2. Add court-aware shot and zone logic.
3. Add a true Orchestrator Agent.
4. Add Telegram / Slack / Discord delivery for automated reports.
