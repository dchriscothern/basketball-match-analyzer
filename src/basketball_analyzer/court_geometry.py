from __future__ import annotations

from dataclasses import dataclass

from .court_profile import WNBA_COURT

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
    np = None


DEFAULT_COURT_PROFILE = WNBA_COURT


@dataclass(frozen=True, slots=True)
class CourtCalibration:
    polygon: tuple[tuple[int, int], ...] | None = None
    source: str = 'default'


def _court_bounds(
    width: int,
    height: int,
    foot_y: float,
    *,
    top_start: float = 0.43,
    top_inset: float = 0.12,
) -> tuple[float, float] | None:
    if foot_y < height * top_start:
        return None
    progress = min(1.0, max(0.0, (foot_y - height * top_start) / (height * (1.0 - top_start))))
    left_bound = width * (top_inset - top_inset * progress)
    right_bound = width * (1.0 - top_inset + top_inset * progress)
    return left_bound, right_bound


def default_court_polygon(width: int, height: int) -> tuple[tuple[int, int], ...]:
    return (
        (int(width * 0.12), int(height * 0.43)),
        (int(width * 0.88), int(height * 0.43)),
        (width - 1, height - 1),
        (0, height - 1),
    )


def court_bounds(width: int, height: int, calibration: CourtCalibration | None = None) -> tuple[float, float, float, float]:
    polygon = calibration.polygon if calibration and calibration.polygon else default_court_polygon(width, height)
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def estimate_hoop_anchors(
    width: int,
    height: int,
    calibration: CourtCalibration | None = None,
) -> dict[str, tuple[float, float]]:
    min_x, min_y, max_x, max_y = court_bounds(width, height, calibration)
    court_width = max(1.0, max_x - min_x)
    court_height = max(1.0, max_y - min_y)
    center_y = min_y + (court_height * 0.52)
    inset_x = court_width * 0.06
    return {
        'left': (min_x + inset_x, center_y),
        'right': (max_x - inset_x, center_y),
    }


def estimate_court_calibration(frame) -> CourtCalibration:
    if cv2 is None or np is None:
        return CourtCalibration()

    height, width = frame.shape[:2]
    lower_band = frame[int(height * 0.4):, :]
    if lower_band.size == 0:
        return CourtCalibration()

    hsv = cv2.cvtColor(lower_band, cv2.COLOR_BGR2HSV)
    sample = hsv[:, int(width * 0.2): int(width * 0.8)]
    if sample.size == 0:
        return CourtCalibration()

    median_h = int(np.median(sample[:, :, 0]))
    median_s = int(np.median(sample[:, :, 1]))
    median_v = int(np.median(sample[:, :, 2]))

    lower = np.array([max(0, median_h - 12), max(0, median_s - 55), max(40, median_v - 70)])
    upper = np.array([min(179, median_h + 12), min(255, median_s + 55), min(255, median_v + 70)])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > width * height * 0.04]
    if not contours:
        return CourtCalibration()

    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    points = []
    for point in approx[:, 0, :]:
        x, y = int(point[0]), int(point[1] + height * 0.4)
        points.append((x, y))

    if len(points) < 4:
        x, y, w, h = cv2.boundingRect(contour)
        points = [
            (x, y + int(height * 0.4)),
            (x + w, y + int(height * 0.4)),
            (x + w, y + h + int(height * 0.4)),
            (x, y + h + int(height * 0.4)),
        ]

    return CourtCalibration(polygon=tuple(points), source='estimated')


def point_is_in_playable_area(
    width: int,
    height: int,
    *,
    foot_x: float,
    foot_y: float,
    calibration: CourtCalibration | None = None,
) -> bool:
    if calibration and calibration.polygon and cv2 is not None:
        polygon = np.array(calibration.polygon, dtype=np.int32)
        return cv2.pointPolygonTest(polygon, (float(foot_x), float(foot_y)), False) >= 0

    bounds = _court_bounds(width, height, foot_y)
    if bounds is None:
        return False
    left_bound, right_bound = bounds
    return left_bound <= foot_x <= right_bound


def box_is_in_playable_area(
    width: int,
    height: int,
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    calibration: CourtCalibration | None = None,
    min_height_ratio: float = 0.09,
    max_width_height_ratio: float = 1.2,
) -> bool:
    box_height = y2 - y1
    box_width = x2 - x1
    if box_height < max(42.0, height * min_height_ratio):
        return False
    if box_width > box_height * max_width_height_ratio:
        return False

    foot_x = (x1 + x2) / 2.0
    foot_y = y2
    if foot_y < height * 0.35:
        return False
    return point_is_in_playable_area(width, height, foot_x=foot_x, foot_y=foot_y, calibration=calibration)


def bbox_tuple_is_in_playable_area(
    frame_shape: tuple[int, int, int],
    box: tuple[int, int, int, int],
    calibration: CourtCalibration | None = None,
) -> bool:
    height, width = frame_shape[:2]
    x, y, w, h = box
    return box_is_in_playable_area(
        width,
        height,
        x1=x,
        y1=y,
        x2=x + w,
        y2=y + h,
        calibration=calibration,
    )
