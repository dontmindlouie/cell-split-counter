"""Tests for review.py's adaptive_radius -- the vision-review marker's neighbor-aware sizing."""

import math

from src.review import adaptive_radius, _TICK_RADIUS, _TICK_RADIUS_MIN


def test_no_neighbor_returns_default_radius():
    assert adaptive_radius(None) == _TICK_RADIUS


def test_far_neighbor_no_cell_area_returns_default_radius():
    # fraction=0.5 of a very large raw distance, capped at _TICK_RADIUS.
    assert adaptive_radius(1000.0) == _TICK_RADIUS


def test_close_neighbor_without_areas_uses_fraction_formula():
    # No cell_area_px/neighbor_area_px known -- falls back to the old fraction-based
    # proxy: floor=radius_min=15, computed = 40*0.5 - 5 = 15 -> max(15, min(55, 15)) = 15.
    assert adaptive_radius(40.0) == 15


def test_moderate_conflict_uses_neighbor_ceiling_not_own_floor():
    # own-size floor (25) exceeds the true neighbor-safe ceiling (20) -- not the extreme
    # near-zero-clearance case below, just an ordinary moderate conflict. The fixed
    # formula must still prefer the smaller, neighbor-safe value over the bigger floor.
    own_area = math.pi * (25.0 / 1.3) ** 2  # size_k * own_radius == 25
    neighbor_area = math.pi * 25.0 ** 2  # own radius 25
    radius = adaptive_radius(50.0, cell_area_px=own_area, neighbor_area_px=neighbor_area)
    assert radius == 20


def test_large_neighbor_at_safe_looking_distance_shrinks_below_own_floor():
    # Regression test for the real 2026-07-11 misattribution root cause: a candidate
    # large enough to want an 18px floor, next to a neighbor at a raw distance (24.5px)
    # that looked safe under the old fraction-based formula, but whose own body (radius
    # 20px) leaves almost no real clearance. The old formula (max(floor, computed))
    # would have returned the floor (18px) here, a box big enough to reach the
    # neighbor's mask -- the fixed formula must not exceed the neighbor-safe ceiling.
    own_area = 615.0  # own radius ~14.0, size_k=1.3 floor ~18.2
    neighbor_area = math.pi * 20.0 ** 2
    radius = adaptive_radius(24.5, cell_area_px=own_area, neighbor_area_px=neighbor_area)
    assert radius == _TICK_RADIUS_MIN  # clamped to the absolute physical minimum
    assert radius < 18  # strictly smaller than the old (floor-dominated) result


def test_same_conflict_shape_without_neighbor_area_still_uses_old_floor_priority():
    # Same numbers as above, but neighbor_area_px unknown (e.g. an event classified
    # before 2026-07-11) -- the fraction-based proxy isn't trustworthy enough to hard-cap
    # on, so the own-size floor still wins, exactly like the pre-fix formula.
    radius = adaptive_radius(24.5, cell_area_px=615.0)
    assert radius == 18


def test_neighbor_ceiling_never_exceeds_tick_radius():
    own_area = math.pi * 100.0 ** 2  # huge candidate, would want a huge floor
    neighbor_area = math.pi * 1.0 ** 2  # tiny, far neighbor
    radius = adaptive_radius(500.0, cell_area_px=own_area, neighbor_area_px=neighbor_area)
    assert radius == _TICK_RADIUS
