from basketball_analyzer.schemas import BBox, Detection
from basketball_analyzer.yolo_extractor import YoloCVExtractor


class DummyFrame:
    shape = (360, 640, 3)


def _player(x1: float, y1: float, x2: float, y2: float, *, track_id: str = 'p1') -> Detection:
    return Detection(track_id=track_id, label='player', bbox=BBox(x1, y1, x2, y2))


def _ball(x1: float, y1: float, x2: float, y2: float, confidence: float, **meta) -> Detection:
    return Detection(track_id='ball', label='ball', bbox=BBox(x1, y1, x2, y2), confidence=confidence, meta=meta)


def test_oversized_learned_ball_loses_to_plausible_fallback():
    extractor = YoloCVExtractor()
    frame = DummyFrame()
    players = [_player(300, 110, 350, 240)]
    extractor._last_ball_center = (321.0, 170.0)
    extractor._last_ball_velocity = (2.0, 1.0)

    learned = _ball(
        375.0,
        150.0,
        420.0,
        205.0,
        0.93,
        learned_ball_detector=True,
        learned_ball_crop='full',
        learned_ball_score=1.9,
    )
    fallback = _ball(
        316.0,
        164.0,
        328.0,
        176.0,
        0.42,
        fallback_ball=True,
        ball_score=1.35,
    )

    chosen = extractor._choose_ball_candidate(frame=frame, players=players, primary=learned, alternatives=[fallback])

    assert chosen is fallback
    assert chosen.meta.get('fallback_ball') is True


def test_compact_learned_ball_beats_distant_fallback():
    extractor = YoloCVExtractor()
    frame = DummyFrame()
    players = [_player(300, 110, 350, 240)]
    extractor._last_ball_center = (321.0, 170.0)
    extractor._last_ball_velocity = (2.0, 1.0)

    learned = _ball(
        318.0,
        164.0,
        330.0,
        176.0,
        0.78,
        learned_ball_detector=True,
        learned_ball_crop='predicted',
        learned_ball_score=2.15,
    )
    fallback = _ball(
        388.0,
        146.0,
        406.0,
        164.0,
        0.45,
        fallback_ball=True,
        ball_score=1.5,
    )

    chosen = extractor._choose_ball_candidate(frame=frame, players=players, primary=learned, alternatives=[fallback])

    assert chosen is learned
    assert chosen.meta.get('learned_ball_detector') is True
