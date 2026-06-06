"""Unit tests for simulation/event_generator.py.

Focus on the A/B manipulation contract: experiment ("aligned") and control
("random") arms must differ ONLY in direction, never in force (magnitude). This
is the actual independent variable of the P7-H study, so it is the property most
worth pinning down with a regression test.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.event_generator import (
    NON_CRITICAL_THRESHOLD,
    design_bifurcation_event,
    event_sequence_planner,
    intensity_modulation,
    is_in_effective_zone,
    rollback_distance,
)

PV = [0.2, -0.1, 0.3, 0.0, 0.1, -0.2, 0.15, 0.05, -0.1]


def test_aligned_event_returns_unit_direction_and_displacement():
    ev = design_bifurcation_event(PV, app_calibrated=True, direction_mode="aligned")
    direction = np.array(ev["direction"])
    assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-6)
    # displacement = direction * magnitude
    disp = np.array(ev["displacement"])
    assert np.linalg.norm(disp) == pytest.approx(ev["magnitude"], abs=1e-9)
    assert len(ev["feature_deltas"]) == 9


def test_control_and_experiment_share_magnitude_differ_in_direction():
    aligned = design_bifurcation_event(PV, app_calibrated=True, direction_mode="aligned")
    control = design_bifurcation_event(
        PV, app_calibrated=True, direction_mode="random",
        rng=np.random.RandomState(0),
    )
    # Identical force...
    assert control["magnitude"] == pytest.approx(aligned["magnitude"], abs=1e-9)
    # ...but different direction (random unit vector ≠ v1).
    cos = float(np.dot(aligned["direction"], control["direction"]))
    assert abs(cos) < 0.999


def test_control_direction_is_reproducible_with_seed():
    a = design_bifurcation_event(PV, direction_mode="random", rng=np.random.RandomState(42))
    b = design_bifurcation_event(PV, direction_mode="random", rng=np.random.RandomState(42))
    assert a["direction"] == b["direction"]


def test_template_target_uses_catalogue_direction():
    ev = design_bifurcation_event(PV, target="explore_risk", direction_mode="aligned")
    assert ev["type"] == "explore_risk"
    assert np.linalg.norm(ev["direction"]) == pytest.approx(1.0, abs=1e-6)


def test_intensity_modulation_decreases_with_proximity():
    far = intensity_modulation(0.0)
    near = intensity_modulation(1.0)
    assert far > near
    # min nudge floor at proximity = 1
    assert near == pytest.approx(0.0005, abs=1e-9)


def test_intensity_modulation_clips_out_of_range_proximity():
    assert intensity_modulation(-5.0) == intensity_modulation(0.0)
    assert intensity_modulation(5.0) == intensity_modulation(1.0)


def test_event_sequence_planner_produces_n_steps():
    plan = event_sequence_planner(PV, n_steps=4, app_calibrated=True, direction_mode="aligned")
    assert plan["n_steps"] == 4
    assert len(plan["events"]) == 4
    assert [e["step"] for e in plan["events"]] == [0, 1, 2, 3]
    assert len(plan["total_displacement"]) == 9


def test_event_sequence_control_reproducible_with_seed():
    a = event_sequence_planner(PV, n_steps=3, direction_mode="random", seed=1)
    b = event_sequence_planner(PV, n_steps=3, direction_mode="random", seed=1)
    assert a["total_displacement"] == b["total_displacement"]


def test_is_in_effective_zone_threshold_semantics():
    res = is_in_effective_zone(PV, app_calibrated=True)
    assert res["threshold"] == NON_CRITICAL_THRESHOLD
    assert res["in_zone"] == (res["proximity"] >= NON_CRITICAL_THRESHOLD)
    assert res["recommendation"] in ("proceed_with_event", "pre_condition_first")


def test_rollback_distance_returns_bounded_rounds():
    res = rollback_distance(PV, app_calibrated=True)
    assert 0 <= res["rounds_to_recover"] <= 500
    assert res["recovery_threshold"] == 0.1
