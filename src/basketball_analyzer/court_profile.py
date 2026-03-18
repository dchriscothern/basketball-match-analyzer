from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CourtProfile:
    name: str
    length_ft: float
    width_ft: float
    lane_width_ft: float
    free_throw_line_ft: float
    three_point_radius_ft: float
    corner_three_sideline_in: float
    corner_three_baseline_in: float
    restricted_area_radius_ft: float


# Based on the 2025 WNBA Official Rule Book, Rule No. 1, Section I:
# - 94' x 50' court
# - three-point lines 36" from sidelines
# - 93 1/3" from baseline
# - 22' 1 3/4" arc
# - restricted area 4'
WNBA_COURT = CourtProfile(
    name='WNBA',
    length_ft=94.0,
    width_ft=50.0,
    lane_width_ft=16.0,
    free_throw_line_ft=19.0,
    three_point_radius_ft=22.1458333333,
    corner_three_sideline_in=3.0,
    corner_three_baseline_in=7.7777777778,
    restricted_area_radius_ft=4.0,
)
