# Ball Model Plan

## Why This Exists

The current analyzer now has a much better fallback/estimated ball path than it
did originally, but the remaining failure mode is still the same: weak
orange-blob candidates can beat the real ball in broadcast footage.

That means the next serious quality jump is not more color heuristics. It is a
dedicated learned ball detector.

## Target Architecture

Keep:
- `YOLO Detect` for players
- downstream event/stats/report/render agents

Add:
- a dedicated small-ball model through
  [ball_detector.py](/C:/GitHub/basketball-match-analyzer/src/basketball_analyzer/ball_detector.py)

Fallback order:
1. learned ball detector
2. YOLO sports-ball class
3. orange/motion fallback
4. player-anchored estimated ball

## Expected Weights Locations

The analyzer will automatically use a dedicated ball model if one exists at:

- `C:\GitHub\basketball-match-analyzer\basketball-ball.pt`
- `C:\GitHub\basketball-match-analyzer\ball-detector.pt`
- `C:\GitHub\basketball-match-analyzer\weights\basketball-ball.pt`

Or via:

- environment variable `BASKETBALL_BALL_MODEL`

## Dataset Shape

Use standard Ultralytics detection format:

- `datasets/ball_detector/images/train`
- `datasets/ball_detector/images/val`
- `datasets/ball_detector/images/test`
- matching YOLO label `.txt` files in `labels/...`

Dataset YAML template:

- [ball_detector_template.yaml](/C:/GitHub/basketball-match-analyzer/datasets/ball_detector_template.yaml)

Frame export helper:

- [export_ball_label_frames.py](/C:/GitHub/basketball-match-analyzer/export_ball_label_frames.py)
- [export_ball_bootstrap_labels.py](/C:/GitHub/basketball-match-analyzer/export_ball_bootstrap_labels.py)

Example:

```powershell
cd C:\GitHub\basketball-match-analyzer
C:\GitHub\basketball-match-analyzer\.venv\Scripts\python.exe export_ball_label_frames.py --video C:\path\to\clip.mp4 --output datasets\ball_detector\images\train --every-nth-frame 6 --max-frames 250 --prefix wnba_ball
```

Bootstrap-label example:

```powershell
cd C:\GitHub\basketball-match-analyzer
C:\GitHub\basketball-match-analyzer\.venv\Scripts\python.exe export_ball_bootstrap_labels.py --video C:\path\to\clip.mp4 --analysis-profile "Standard Clip" --tracking-backend yolo_detect --images-output datasets\ball_detector\images\train --labels-output datasets\ball_detector\labels\train --prefix wnba_bootstrap --include-empty
```

This path uses the current analyzer to create starter YOLO labels that you can
fix by hand instead of labeling every frame from scratch.

Validation helper:

- [validate_ball_dataset.py](/C:/GitHub/basketball-match-analyzer/validate_ball_dataset.py)
- [split_ball_dataset.py](/C:/GitHub/basketball-match-analyzer/split_ball_dataset.py)
- [prepare_ball_dataset.py](/C:/GitHub/basketball-match-analyzer/prepare_ball_dataset.py)
- [preview_ball_dataset.py](/C:/GitHub/basketball-match-analyzer/preview_ball_dataset.py)

One-command bootstrap example:

```powershell
cd C:\GitHub\basketball-match-analyzer
C:\GitHub\basketball-match-analyzer\.venv\Scripts\python.exe prepare_ball_dataset.py --video C:\path\to\clip.mp4 --tracking-backend yolo_detect --include-empty --clear-existing
```

Newest temp clip example:

```powershell
cd C:\GitHub\basketball-match-analyzer
C:\GitHub\basketball-match-analyzer\.venv\Scripts\python.exe prepare_ball_dataset.py --video latest-temp --tracking-backend yolo_detect --include-empty --clear-existing
```

Preview helper example:

```powershell
cd C:\GitHub\basketball-match-analyzer
C:\GitHub\basketball-match-analyzer\.venv\Scripts\python.exe preview_ball_dataset.py --dataset-root datasets\ball_detector --split train
```

Compare helper example:

```powershell
cd C:\GitHub\basketball-match-analyzer
C:\GitHub\basketball-match-analyzer\.venv\Scripts\python.exe compare_ball_tracking.py --video latest-temp --tracking-backend yolo_detect
```

Batch bootstrap example:

```powershell
cd C:\GitHub\basketball-match-analyzer
C:\GitHub\basketball-match-analyzer\.venv\Scripts\python.exe prepare_ball_dataset_batch.py --recent-temp-count 4 --tracking-backend yolo_detect --include-empty --clear-existing
```

Split helper example:

```powershell
cd C:\GitHub\basketball-match-analyzer
C:\GitHub\basketball-match-analyzer\.venv\Scripts\python.exe split_ball_dataset.py --images-source datasets\ball_detector\images\train --labels-source datasets\ball_detector\labels\train --dataset-root datasets\ball_detector --train-ratio 0.8 --val-ratio 0.15 --clear-existing
```

Example:

```powershell
cd C:\GitHub\basketball-match-analyzer
C:\GitHub\basketball-match-analyzer\.venv\Scripts\python.exe validate_ball_dataset.py --dataset-root datasets\ball_detector
```

Single class:
- `basketball`

## Training Command

Use:

```powershell
cd C:\GitHub\basketball-match-analyzer
C:\GitHub\basketball-match-analyzer\.venv\Scripts\python.exe train_ball_detector.py --data datasets\ball_detector_template.yaml --epochs 40 --imgsz 960 --batch 8
```

Training script:

- [train_ball_detector.py](/C:/GitHub/basketball-match-analyzer/train_ball_detector.py)

## Practical Labeling Advice

Focus first on:
- WNBA broadcast clips
- ball near hands
- ball in air on passes/shots
- small-ball frames where the current heuristic path fails
- orange-confuser frames like scorebug elements, logos, crowd graphics, and ads

Prioritize diversity over volume at first:
- 500-1500 labeled frames is enough to test whether the dedicated detector is
  directionally better than the heuristic path

## Success Criteria

The new ball model should beat the current heuristic path on:

- real ball follows the actual carrier more often
- fewer false positives on orange blobs
- better shot trajectory continuity
- better ball frame coverage without overestimating possession

## Next Step After Ball Model

Once the ball detector is good enough:
- add hoop/backboard detection
- then improve shot-attempt and make/miss logic
- then revisit richer basketball analytics on top of cleaner tracking
