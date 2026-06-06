"""Tests for the Space A ↔ Space B personality bijection."""

from __future__ import annotations

import numpy as np
import pytest

from simulation.bifurcation_detector import BASELINE_ATTRACTOR, EPSILON_C_APP
from simulation.personality_space import (
    SPACE_B_SCALE,
    a_to_b,
    b_to_a,
    displacement_a_to_b,
    displacement_b_to_a,
)

RNG = np.random.RandomState(0)
SAMPLES = [RNG.uniform(-1, 1, size=9) for _ in range(20)]


def test_scale_is_epsilon_c_app():
    assert SPACE_B_SCALE == EPSILON_C_APP


def test_baseline_maps_to_origin():
    assert np.allclose(a_to_b(BASELINE_ATTRACTOR), np.zeros(9))


@pytest.mark.parametrize("vec", SAMPLES)
def test_a_to_b_round_trip_is_exact(vec):
    assert np.allclose(b_to_a(a_to_b(vec)), vec, atol=1e-12)


@pytest.mark.parametrize("vec", SAMPLES)
def test_b_to_a_round_trip_is_exact(vec):
    assert np.allclose(a_to_b(b_to_a(vec)), vec, atol=1e-12)


def test_is_a_bijection_distinct_inputs_give_distinct_outputs():
    outs = [tuple(np.round(a_to_b(v), 9)) for v in SAMPLES]
    assert len(set(outs)) == len(SAMPLES)


def test_displacement_map_is_linear_no_recenter():
    d = np.array([0.1, -0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
    # displacement uses scale only (no baseline shift)
    assert np.allclose(displacement_a_to_b(d), d / SPACE_B_SCALE)
    assert np.allclose(displacement_b_to_a(displacement_a_to_b(d)), d, atol=1e-12)


def test_realistic_spread_is_order_one_in_space_b():
    """A baseline-distance-~ε_c_app perturbation should be magnitude ~1 in B."""
    direction = np.ones(9) / np.linalg.norm(np.ones(9))
    a = BASELINE_ATTRACTOR + EPSILON_C_APP * direction
    assert np.linalg.norm(a_to_b(a)) == pytest.approx(1.0, abs=1e-9)


def test_wrong_length_raises():
    with pytest.raises(ValueError):
        a_to_b([0.0] * 8)
    with pytest.raises(ValueError):
        b_to_a([0.0] * 10)
