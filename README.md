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
- the app can also download a WNBA video URL directly and run the same pipeline
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

This now writes a rendered artifact alongside the JSON outputs and report. Today that will usually be `analysis.gif`, and for MP4 uploads it can become an overlay `analysis.mp4` when OpenCV is available. For browser preview reliability, MP4 uploads also generate a GIF preview.

## Run The Streamlit App

```powershell
cd C:\GitHub\basketball-match-analyzer
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
streamlit run app.py --server.port 8513
```

## Benchmark Backends

Use the benchmark runner to compare tracking backends on the same clips:

```powershell
cd C:\GitHub\basketball-match-analyzer
.\.venv\Scripts\Activate.ps1
python -m basketball_analyzer.benchmark --manifest benchmarks\sample_manifest.json --output benchmarks\results.json
```

This writes per-backend metrics such as:
- runtime
- average players detected
- ball frame coverage
- possession frame coverage
- event counts
- players removed after post-classification filtering

The manifest can now use either:
- a local MP4 file path
- a direct video URL such as a YouTube clip URL

The benchmark also writes a markdown summary next to the JSON output so results are easier to scan.

## Input Modes

1. `demo`
   The fastest way to see the whole system working.

2. `Tracked JSON`
   The best real-data path right now. If you already have external tracking output, upload a JSON file and let the downstream agents do eventing, stats, reporting, and rendering.

3. `Raw MP4`
   Raw MP4 mode now supports multiple backends:
   - `Auto`
   - `Prototype CV`
   - `YOLO Detect`
   - `YOLO + BoT-SORT (experimental)`
   - `YOLO + ByteTrack (experimental)`

   The default app flow still prioritizes speed for preview runs, but you can now compare stronger detector/tracker paths without changing the rest of the pipeline.

4. `WNBA Video URL`
   Paste a WNBA video URL and the app will download it with `yt-dlp`, then run the same MP4 analysis path. This is the quickest way to test official highlight clips without manually creating a local file first.

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

## Dedicated Ball Detector Path

The repo now includes an optional dedicated ball-detector slot in
[ball_detector.py](/C:/GitHub/basketball-match-analyzer/src/basketball_analyzer/ball_detector.py).

If you add custom weights at one of these locations:
- `C:\GitHub\basketball-match-analyzer\basketball-ball.pt`
- `C:\GitHub\basketball-match-analyzer\ball-detector.pt`
- `C:\GitHub\basketball-match-analyzer\weights\basketball-ball.pt`

or set:
- `BASKETBALL_BALL_MODEL`

then the YOLO pipeline will try that learned ball detector before falling back to
the current heuristic paths.

If you train a ball detector with [train_ball_detector.py](/C:/GitHub/basketball-match-analyzer/train_ball_detector.py),
the analyzer will also auto-pick the newest `runs/**/weights/best.pt` file if no
explicit model path is set.
The first run after training may still be slower while the learned ball model
loads; repeat runs in the same app session should be faster.

Training and dataset plan:
- [BALL_MODEL_PLAN.md](/C:/GitHub/basketball-match-analyzer/BALL_MODEL_PLAN.md)
- [train_ball_detector.py](/C:/GitHub/basketball-match-analyzer/train_ball_detector.py)
- [ball_detector_template.yaml](/C:/GitHub/basketball-match-analyzer/datasets/ball_detector_template.yaml)
- [export_ball_bootstrap_labels.py](/C:/GitHub/basketball-match-analyzer/export_ball_bootstrap_labels.py)
- [prepare_ball_dataset.py](/C:/GitHub/basketball-match-analyzer/prepare_ball_dataset.py)
- [preview_ball_dataset.py](/C:/GitHub/basketball-match-analyzer/preview_ball_dataset.py)
- [split_ball_dataset.py](/C:/GitHub/basketball-match-analyzer/split_ball_dataset.py)
- [validate_ball_dataset.py](/C:/GitHub/basketball-match-analyzer/validate_ball_dataset.py)

If you want one command to bootstrap a clip into `train/val/test` folders, use:

```powershell
cd C:\GitHub\basketball-match-analyzer
C:\GitHub\basketball-match-analyzer\.venv\Scripts\python.exe prepare_ball_dataset.py --video C:\path\to\clip.mp4 --tracking-backend yolo_detect --include-empty --clear-existing
```

If you just tested a clip in the app and want to reuse the newest temp source file, you can use:

```powershell
cd C:\GitHub\basketball-match-analyzer
C:\GitHub\basketball-match-analyzer\.venv\Scripts\python.exe prepare_ball_dataset.py --video latest-temp --tracking-backend yolo_detect --include-empty --clear-existing
```

To spot-check the resulting labels visually before training, use:

```powershell
cd C:\GitHub\basketball-match-analyzer
C:\GitHub\basketball-match-analyzer\.venv\Scripts\python.exe preview_ball_dataset.py --dataset-root datasets\ball_detector --split train
```

To compare learned-ball tracking against the fallback path on the same clip, use:

```powershell
cd C:\GitHub\basketball-match-analyzer
C:\GitHub\basketball-match-analyzer\.venv\Scripts\python.exe compare_ball_tracking.py --video latest-temp --tracking-backend yolo_detect
```

If one clip is too small for training, bootstrap from several recent temp clips at once:

```powershell
cd C:\GitHub\basketball-match-analyzer
C:\GitHub\basketball-match-analyzer\.venv\Scripts\python.exe prepare_ball_dataset_batch.py --recent-temp-count 4 --tracking-backend yolo_detect --include-empty --clear-existing
```
