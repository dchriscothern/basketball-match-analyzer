# Basketball Match Analyzer Roadmap

This roadmap is meant to keep the project grounded in a realistic path:

- build a focused WNBA-first product that works on real clips
- harden it for repeatable use across normal broadcasts
- only then widen toward a full platform

## Phase 1: WNBA MVP

Goal:
- upload a WNBA clip
- detect the visible court correctly
- track enough players and the ball to create useful possession-level output
- generate a rendered preview and a believable short report

Core deliverables:
- stronger WNBA court calibration from broadcast frames
- better on-court filtering for refs, camera crew, and benches
- more stable player tracking across short clips
- basic team assignment from jersey color clusters
- core events:
  - possession
  - pass
  - shot attempt
  - rebound
  - turnover
  - steal
- fast preview mode that returns results quickly for testing

Success criteria:
- short clips run in under about 30 to 60 seconds on local hardware
- overlays mostly stay on players who are actually on court
- possession and event output is directionally useful for review

## Phase 2: Broadcast Robustness

Goal:
- make the analyzer dependable across a wider set of WNBA and college clips

Core deliverables:
- improved ball detection and smoothing
- hoop and backboard detection for shot context
- handling for camera cuts and abrupt zoom changes
- stronger identity persistence across occlusions
- confidence scoring and QA flags
- a small internal evaluation set of real clips with expected outputs

Expanded event set:
- made vs missed shot heuristics
- transition vs half-court possessions
- screens and pick-and-roll triggers
- player involvement summaries by possession

Success criteria:
- works on many normal broadcast clips instead of hand-picked ones
- lower false positives on sideline staff and off-court figures
- event timeline is stable enough for practical review

## Phase 3: Platform Expansion

Goal:
- move from a strong prototype into a scalable basketball data product

Core deliverables:
- custom-trained detection and tracking models
- larger labeled dataset for WNBA, NCAA, and pro footage
- batch processing pipeline
- API and export layer
- stronger evaluation and QA tooling
- off-camera inference research
- lineup and tactical analytics layers

What this phase starts to resemble:
- the early shape of a SkillCorner-style platform for basketball

## What SkillCorner-Level Really Implies

A SkillCorner-like product is not just a better demo app. It usually requires:

- large-scale labeled data
- custom computer vision models
- robust identity tracking
- full-court calibration across many camera styles
- off-camera player position estimation
- scalable processing infrastructure
- strong QA and analyst workflows

That is a multi-year effort, not a quick feature.

## Immediate Next Steps

1. Keep preview mode fast and reliable for local testing.
2. Improve WNBA court calibration on real broadcast clips.
3. Improve ball tracking and possession smoothing.
4. Reduce off-court false positives further.
5. Add hoop-aware shot context.
6. Add a true Fast vs Full analysis mode in the Streamlit app.
