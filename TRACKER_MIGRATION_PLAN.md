# Tracker Migration Plan

This project should stop relying on heuristic-only tracking as the main path.

The most realistic upgrade path is:

1. use a stronger open-source detector + MOT stack
2. fine-tune on sports tracking data
3. keep the current eventing, reporting, and rendering layers as downstream consumers

## Recommended Stack

### Best practical path for this repo

- detector: YOLO
- tracker: BoT-SORT or ByteTrack
- sports data for adaptation: SportsMOT and TeamTrack

Why:
- easier to integrate than a heavy research framework
- strong community support
- realistic for a local prototype that needs to improve quickly
- good bridge from MVP to something credible

### Best research framework path

- TrackLab

Why:
- modular
- supports detectors, trackers, re-id, and evaluation as separate components
- better long-term structure if this project becomes a real R&D platform

Downside:
- more setup and orchestration overhead up front

## What We Keep vs Replace

### Keep

- Streamlit app
- analysis profiles
- team classifier
- events agent
- analytics agent
- reporting agent
- video renderer

### Replace or down-rank

- prototype OpenCV player tracking as the primary path
- heuristic ball detection as the primary path
- current nearest-player possession logic as the only control model

## Near-Term Integration Plan

### Step 1

Add a new tracker backend abstraction:

- `prototype_cv`
- `yolo_botsort`
- `yolo_bytetrack`
- `tracklab` (future)

### Step 2

Make the new `yolo_mot` backend output the same internal schema:

- `TrackingFrame`
- `Detection`
- `BBox`

This lets the downstream agents keep working.

### Step 3

Use sports datasets to improve the stack:

- SportsMOT for player tracking
- TeamTrack for multi-sport MOT benchmarking

### Step 4

Add a stronger ball module:

- learned detector
- temporal smoothing
- player proximity + continuity scoring
- optional hoop-aware shot trajectory logic

### Step 5

Move possession and event logic from heuristic-only to model-assisted scoring.

## Rough Timeline

### If using existing open-source tracking well

- first upgraded prototype: 2 to 4 weeks
- strong internal beta: 1 to 3 months
- dependable WNBA-first product layer: 3 to 6 months

### If aiming for SkillCorner-like coverage

- much longer
- likely multi-person and multi-year

## Compared With SkillCorner

SkillCorner is not just tracking.

It combines:
- broadcast tracking
- full player coverage
- off-camera extrapolation
- advanced event layers
- production QA
- delivery at league scale

So the realistic goal here is:

- first become a good WNBA-first analyzer
- then become a robust basketball analytics product
- not pretend this is already a SkillCorner-equivalent platform

## Compared With a Sloan-Style Valuation Model

A Sloan-style possession value / Shapley attribution layer is more buildable than SkillCorner-level tracking.

But it needs reliable inputs first:
- possession quality
- player identity quality
- event quality
- enough games and possessions

The ideal sequence is:

1. improve tracking
2. improve possession and event quality
3. build expected possession value
4. build attribution and valuation layers
