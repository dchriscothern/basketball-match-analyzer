from basketball_analyzer.agents.events_agent import EventsAgent
from basketball_analyzer.demo_data import generate_demo_frames


def test_detects_basketball_events_from_demo_frames():
    frames = generate_demo_frames()
    agent = EventsAgent()
    _, events = agent.run(frames)

    event_types = [event.event_type for event in events]
    assert 'pass' in event_types
    assert 'turnover' in event_types
    assert 'steal' in event_types
    assert 'shot_attempt' in event_types
    assert 'defensive_rebound' in event_types
