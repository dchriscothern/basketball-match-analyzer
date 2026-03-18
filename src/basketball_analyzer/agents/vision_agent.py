from __future__ import annotations

import json
from pathlib import Path

from .base import BaseAgent
from ..demo_data import generate_demo_frames
from ..prototype_cv import PrototypeCVExtractor
from ..schemas import BBox, Detection, TrackingFrame
from ..yolo_extractor import YoloCVExtractor
from ..yolo_mot_extractor import YoloMOTExtractor


class VisionAgent(BaseAgent):
    name = 'vision_agent'

    def __init__(self) -> None:
        self.yolo_botsort = YoloMOTExtractor(tracker='botsort.yaml')
        self.yolo_bytetrack = YoloMOTExtractor(tracker='bytetrack.yaml')
        self.yolo_cv = YoloCVExtractor()
        self.prototype_cv = PrototypeCVExtractor()
        self.last_run_note: str | None = None
        self.last_calibration_note: str | None = None

    def _quality_note(self, frames: list[TrackingFrame]) -> str | None:
        if not frames:
            return 'The raw MP4 tracker could not detect enough usable player/ball data from this clip yet. No demo fallback was used.'

        avg_players = sum(len(frame.players) for frame in frames) / len(frames)
        ball_ratio = sum(1 for frame in frames if frame.ball is not None) / len(frames)
        if avg_players < 4 or ball_ratio < 0.15:
            return (
                'The raw MP4 tracker produced low-signal detections. Treat this run as prototype output and prefer the JSON tracking path for cleaner analysis.'
            )
        return None

    def _signal_score(self, frames: list[TrackingFrame]) -> float:
        if not frames:
            return 0.0
        avg_players = sum(len(frame.players) for frame in frames) / len(frames)
        ball_ratio = sum(1 for frame in frames if frame.ball is not None) / len(frames)
        return avg_players + (ball_ratio * 4.0)

    def _is_good_enough(self, frames: list[TrackingFrame]) -> bool:
        if not frames:
            return False
        avg_players = sum(len(frame.players) for frame in frames) / len(frames)
        ball_ratio = sum(1 for frame in frames if frame.ball is not None) / len(frames)
        return avg_players >= 2.0 or ball_ratio >= 0.35 or self._signal_score(frames) >= 2.5

    def _from_json(self, path: Path) -> list[TrackingFrame]:
        raw = json.loads(path.read_text(encoding='utf-8'))
        frames: list[TrackingFrame] = []
        for item in raw:
            players = [
                Detection(
                    track_id=player['track_id'],
                    label=player.get('label', 'player'),
                    team_id=player.get('team_id'),
                    confidence=player.get('confidence', 1.0),
                    bbox=BBox(**player['bbox']),
                    meta=player.get('meta', {}),
                )
                for player in item.get('players', [])
            ]
            ball_raw = item.get('ball')
            ball = None
            if ball_raw:
                ball = Detection(
                    track_id=ball_raw.get('track_id', 'ball'),
                    label=ball_raw.get('label', 'ball'),
                    confidence=ball_raw.get('confidence', 1.0),
                    team_id=ball_raw.get('team_id'),
                    bbox=BBox(**ball_raw['bbox']),
                    meta=ball_raw.get('meta', {}),
                )
            frames.append(
                TrackingFrame(
                    frame_index=item['frame_index'],
                    timestamp_s=item['timestamp_s'],
                    players=players,
                    ball=ball,
                )
            )
        return frames

    def run(self, video_path: str | Path, backend: str = 'auto') -> list[TrackingFrame]:
        self.last_run_note = None
        self.last_calibration_note = None
        path = Path(video_path)
        token = str(video_path).lower()

        if token in {'demo', '__demo__', 'sample.mp4'}:
            self.last_run_note = 'Running the built-in demo sequence.'
            return generate_demo_frames()
        if path.suffix.lower() == '.json' and path.exists():
            self.last_run_note = 'Running from tracked JSON input.'
            return self._from_json(path)
        if path.exists() and path.suffix.lower() == '.mp4':
            if backend in {'yolo_mot', 'yolo_botsort'}:
                mot_frames = self.yolo_botsort.run(path) if self.yolo_botsort.is_available() else []
                if mot_frames:
                    self.last_run_note = 'Running the experimental YOLO + BoT-SORT backend for raw MP4 input.'
                    if self.yolo_botsort._calibration is not None:
                        self.last_calibration_note = f'Court calibration: {self.yolo_botsort._calibration.source}.'
                    quality_note = self._quality_note(mot_frames)
                    if quality_note:
                        self.last_run_note = f'{self.last_run_note} {quality_note}'.strip()
                    return mot_frames
                backend = 'auto'
                self.last_run_note = (
                    'The experimental YOLO + BoT-SORT backend returned no usable detections, so the app fell back to auto mode.'
                    if not self.yolo_botsort.last_error
                    else f'The experimental YOLO + BoT-SORT backend failed ({self.yolo_botsort.last_error}), so the app fell back to auto mode.'
                )

            if backend == 'yolo_bytetrack':
                mot_frames = self.yolo_bytetrack.run(path) if self.yolo_bytetrack.is_available() else []
                if mot_frames:
                    self.last_run_note = 'Running the experimental YOLO + ByteTrack backend for raw MP4 input.'
                    if self.yolo_bytetrack._calibration is not None:
                        self.last_calibration_note = f'Court calibration: {self.yolo_bytetrack._calibration.source}.'
                    quality_note = self._quality_note(mot_frames)
                    if quality_note:
                        self.last_run_note = f'{self.last_run_note} {quality_note}'.strip()
                    return mot_frames
                backend = 'auto'
                self.last_run_note = (
                    'The experimental YOLO + ByteTrack backend returned no usable detections, so the app fell back to auto mode.'
                    if not self.yolo_bytetrack.last_error
                    else f'The experimental YOLO + ByteTrack backend failed ({self.yolo_bytetrack.last_error}), so the app fell back to auto mode.'
                )

            if backend == 'yolo_detect':
                yolo_frames = self.yolo_cv.run(path) if self.yolo_cv.is_available() else []
                self.last_run_note = 'Running the YOLO detection backend for raw MP4 input.'
                learned_weights = self.yolo_cv._ball_detector.active_weights_path
                if learned_weights is not None:
                    self.last_run_note = f"{self.last_run_note} Learned ball detector active: {learned_weights.name}."
                if self.yolo_cv._calibration is not None:
                    self.last_calibration_note = f'Court calibration: {self.yolo_cv._calibration.source}.'
                quality_note = self._quality_note(yolo_frames)
                if quality_note:
                    self.last_run_note = f'{self.last_run_note} {quality_note}'.strip()
                return yolo_frames

            if backend == 'prototype_cv':
                prototype_frames = self.prototype_cv.run(path)
                self.last_run_note = 'Running the prototype OpenCV backend for raw MP4 input.'
                if self.prototype_cv._calibration is not None:
                    self.last_calibration_note = f'Court calibration: {self.prototype_cv._calibration.source}.'
                quality_note = self._quality_note(prototype_frames)
                if quality_note:
                    self.last_run_note = f'{self.last_run_note} {quality_note}'.strip()
                return prototype_frames

            yolo_detect_frames = self.yolo_cv.run(path) if self.yolo_cv.is_available() else []
            if self._is_good_enough(yolo_detect_frames):
                frames = yolo_detect_frames
                prefix = f'{self.last_run_note} ' if self.last_run_note else ''
                self.last_run_note = f'{prefix}The app used the YOLO detection backend by default for this clip.'.strip()
                learned_weights = self.yolo_cv._ball_detector.active_weights_path
                if learned_weights is not None and 'Learned ball detector active:' not in self.last_run_note:
                    self.last_run_note = f"{self.last_run_note} Learned ball detector active: {learned_weights.name}."
                if self.yolo_cv._calibration is not None:
                    self.last_calibration_note = f'Court calibration: {self.yolo_cv._calibration.source}.'
            else:
                botsort_frames = self.yolo_botsort.run(path) if self.yolo_botsort.is_available() else []
                bytetrack_frames = self.yolo_bytetrack.run(path) if self.yolo_bytetrack.is_available() else []
                prototype_frames = self.prototype_cv.run(path)
                yolo_score = self._signal_score(yolo_detect_frames)
                botsort_score = self._signal_score(botsort_frames)
                bytetrack_score = self._signal_score(bytetrack_frames)
                prototype_score = self._signal_score(prototype_frames)

                if botsort_score > yolo_score and botsort_frames and botsort_score >= max(prototype_score, bytetrack_score):
                    frames = botsort_frames
                    prefix = f'{self.last_run_note} ' if self.last_run_note else ''
                    self.last_run_note = f'{prefix}The app escalated to the experimental YOLO + BoT-SORT backend because YOLO Detect looked too weak on this clip.'.strip()
                    if self.yolo_botsort._calibration is not None:
                        self.last_calibration_note = f'Court calibration: {self.yolo_botsort._calibration.source}.'
                elif bytetrack_score > yolo_score and bytetrack_frames and bytetrack_score >= prototype_score:
                    frames = bytetrack_frames
                    prefix = f'{self.last_run_note} ' if self.last_run_note else ''
                    self.last_run_note = f'{prefix}The app escalated to the experimental YOLO + ByteTrack backend because YOLO Detect looked too weak on this clip.'.strip()
                    if self.yolo_bytetrack._calibration is not None:
                        self.last_calibration_note = f'Court calibration: {self.yolo_bytetrack._calibration.source}.'
                elif prototype_score > yolo_score and prototype_frames:
                    frames = prototype_frames
                    prefix = f'{self.last_run_note} ' if self.last_run_note else ''
                    self.last_run_note = f'{prefix}The app fell back to the prototype OpenCV tracker because it beat YOLO Detect on this clip.'.strip()
                    if self.prototype_cv._calibration is not None:
                        self.last_calibration_note = f'Court calibration: {self.prototype_cv._calibration.source}.'
                else:
                    frames = yolo_detect_frames or botsort_frames or bytetrack_frames or prototype_frames
                    prefix = f'{self.last_run_note} ' if self.last_run_note else ''
                    if yolo_detect_frames:
                        self.last_run_note = f'{prefix}The app stayed on YOLO Detect because it still matched or beat the fallback backends.'.strip()
                        learned_weights = self.yolo_cv._ball_detector.active_weights_path
                        if learned_weights is not None and 'Learned ball detector active:' not in self.last_run_note:
                            self.last_run_note = f"{self.last_run_note} Learned ball detector active: {learned_weights.name}."
                        if self.yolo_cv._calibration is not None:
                            self.last_calibration_note = f'Court calibration: {self.yolo_cv._calibration.source}.'
                    elif botsort_frames:
                        self.last_run_note = f'{prefix}The app escalated to YOLO + BoT-SORT because YOLO Detect returned no usable detections.'.strip()
                        if self.yolo_botsort._calibration is not None:
                            self.last_calibration_note = f'Court calibration: {self.yolo_botsort._calibration.source}.'
                    elif bytetrack_frames:
                        self.last_run_note = f'{prefix}The app escalated to YOLO + ByteTrack because YOLO Detect returned no usable detections.'.strip()
                        if self.yolo_bytetrack._calibration is not None:
                            self.last_calibration_note = f'Court calibration: {self.yolo_bytetrack._calibration.source}.'
                    elif prototype_frames:
                        self.last_run_note = f'{prefix}The app fell back to the prototype OpenCV tracker because the YOLO backends returned no usable detections.'.strip()
                        if self.prototype_cv._calibration is not None:
                            self.last_calibration_note = f'Court calibration: {self.prototype_cv._calibration.source}.'
            quality_note = self._quality_note(frames)
            if quality_note:
                self.last_run_note = f'{self.last_run_note} {quality_note}'.strip() if self.last_run_note else quality_note
            return frames
        return []
