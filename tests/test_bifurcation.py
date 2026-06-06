"""Unit tests for bifurcation_detector and event_generator (P7-H Phase I)."""

import numpy as np
import pytest

from simulation.bifurcation_detector import (
    BASELINE_ATTRACTOR,
    CRITICAL_PROXIMITY,
    EPSILON_C,
    EPSILON_C_APP,
    FEATURE_NAMES,
    LAMBDA_MAX,
    SENSITIVE_BASIS,
    V1,
    V2,
    compute_bifurcation_distance,
    compute_sensitive_direction,
    get_jacobian,
)
from simulation.event_generator import (
    design_bifurcation_event,
    event_sequence_planner,
    intensity_modulation,
)


# ── bifurcation_detector ─────────────────────────────────────────────────────

class TestComputeBifurcationDistance:
    def test_at_baseline_zero_distance(self):
        result = compute_bifurcation_distance(BASELINE_ATTRACTOR)
        assert result["distance"] == pytest.approx(0.0, abs=1e-12)
        assert result["bifurcation_proximity"] == pytest.approx(0.0, abs=1e-12)
        assert result["is_critical"] is False

    def test_at_epsilon_c_proximity_one(self):
        perturbed = BASELINE_ATTRACTOR.copy()
        perturbed[0] += EPSILON_C
        result = compute_bifurcation_distance(perturbed)
        assert result["bifurcation_proximity"] == pytest.approx(1.0, abs=0.01)
        assert result["is_critical"] is True

    def test_proximity_capped_at_one(self):
        far = BASELINE_ATTRACTOR + 1.0
        result = compute_bifurcation_distance(far)
        assert result["bifurcation_proximity"] <= 1.0

    def test_list_input_accepted(self):
        result = compute_bifurcation_distance(BASELINE_ATTRACTOR.tolist())
        assert "distance" in result

    def test_v1_projection_nonzero_when_displaced_along_v1(self):
        perturbed = BASELINE_ATTRACTOR + 0.002 * V1
        result = compute_bifurcation_distance(perturbed)
        assert abs(result["v1_projection"]) > 1e-6

    def test_v2_projection_nonzero_when_displaced_along_v2(self):
        perturbed = BASELINE_ATTRACTOR + 0.002 * V2
        result = compute_bifurcation_distance(perturbed)
        assert abs(result["v2_projection"]) > 1e-6


class TestComputeSensitiveDirection:
    def test_returns_required_keys(self):
        result = compute_sensitive_direction(BASELINE_ATTRACTOR)
        for key in ("direction", "secondary_direction", "eigenvalue", "sensitivity_strength"):
            assert key in result

    def test_direction_is_unit_vector(self):
        result = compute_sensitive_direction(BASELINE_ATTRACTOR)
        direction = np.array(result["direction"])
        assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-6)

    def test_eigenvalue_equals_lambda_max(self):
        result = compute_sensitive_direction(BASELINE_ATTRACTOR)
        assert result["eigenvalue"] == pytest.approx(LAMBDA_MAX, rel=1e-6)

    def test_direction_length_is_9(self):
        result = compute_sensitive_direction(BASELINE_ATTRACTOR)
        assert len(result["direction"]) == 9


class TestGetJacobian:
    def test_jacobian_shape(self):
        J = get_jacobian()
        assert J.shape == (9, 9)

    def test_jacobian_is_symmetric(self):
        J = get_jacobian()
        assert np.allclose(J, J.T, atol=1e-10)

    def test_jacobian_nonzero(self):
        J = get_jacobian()
        assert np.any(np.abs(J) > 1e-3)


# ── event_generator ──────────────────────────────────────────────────────────

class TestIntensityModulation:
    def test_zero_proximity_gives_max(self):
        mag = intensity_modulation(0.0)
        # max = EPSILON_C * 0.5 * (1 - 0) + min_magnitude(0.0005)
        assert mag == pytest.approx(EPSILON_C * 0.5 + 0.0005, rel=0.01)

    def test_full_proximity_gives_min(self):
        mag = intensity_modulation(1.0)
        assert mag == pytest.approx(0.0005, rel=0.01)

    def test_monotonically_decreasing(self):
        values = [intensity_modulation(p) for p in [0.0, 0.25, 0.5, 0.75, 1.0]]
        for a, b in zip(values, values[1:]):
            assert a >= b

    def test_clamps_below_zero(self):
        assert intensity_modulation(-0.5) == intensity_modulation(0.0)

    def test_clamps_above_one(self):
        assert intensity_modulation(1.5) == intensity_modulation(1.0)


class TestDesignBifurcationEvent:
    def test_returns_required_keys(self):
        ev = design_bifurcation_event(BASELINE_ATTRACTOR)
        for key in ("type", "direction", "magnitude", "duration_rounds",
                    "proximity", "displacement", "feature_deltas"):
            assert key in ev

    def test_direction_is_unit_vector(self):
        ev = design_bifurcation_event(BASELINE_ATTRACTOR)
        d = np.array(ev["direction"])
        assert np.linalg.norm(d) == pytest.approx(1.0, abs=1e-6)

    def test_displacement_equals_direction_times_magnitude(self):
        ev = design_bifurcation_event(BASELINE_ATTRACTOR)
        expected = np.array(ev["direction"]) * ev["magnitude"]
        assert np.allclose(ev["displacement"], expected, atol=1e-10)

    def test_feature_deltas_length(self):
        ev = design_bifurcation_event(BASELINE_ATTRACTOR)
        assert len(ev["feature_deltas"]) == 9
        assert set(ev["feature_deltas"].keys()) == set(FEATURE_NAMES)

    def test_duration_rounds_positive(self):
        ev = design_bifurcation_event(BASELINE_ATTRACTOR)
        assert ev["duration_rounds"] >= 1

    def test_list_input_accepted(self):
        ev = design_bifurcation_event(BASELINE_ATTRACTOR.tolist())
        assert ev["magnitude"] > 0

    def test_known_event_type(self):
        ev = design_bifurcation_event(BASELINE_ATTRACTOR, target="explore_risk")
        assert ev["type"] == "explore_risk"

    def test_intensity_scale_multiplies_magnitude(self):
        ev1 = design_bifurcation_event(BASELINE_ATTRACTOR, intensity_scale=1.0)
        ev2 = design_bifurcation_event(BASELINE_ATTRACTOR, intensity_scale=2.0)
        assert ev2["magnitude"] == pytest.approx(ev1["magnitude"] * 2.0, rel=1e-6)


class TestEventSequencePlanner:
    def test_returns_required_keys(self):
        seq = event_sequence_planner(BASELINE_ATTRACTOR, n_steps=3)
        for key in ("events", "total_displacement", "predicted_final_proximity", "n_steps"):
            assert key in seq

    def test_correct_number_of_steps(self):
        for n in [1, 3, 5]:
            seq = event_sequence_planner(BASELINE_ATTRACTOR, n_steps=n)
            assert len(seq["events"]) == n
            assert seq["n_steps"] == n

    def test_proximity_increases_over_sequence(self):
        seq = event_sequence_planner(BASELINE_ATTRACTOR, n_steps=5)
        proximities = [ev["proximity"] for ev in seq["events"]]
        # First step should be at or near 0; last should be higher
        assert proximities[-1] >= proximities[0]

    def test_final_proximity_approaches_one(self):
        seq = event_sequence_planner(BASELINE_ATTRACTOR, n_steps=5)
        assert seq["predicted_final_proximity"] > 0.5

    def test_total_displacement_length(self):
        seq = event_sequence_planner(BASELINE_ATTRACTOR, n_steps=3)
        assert len(seq["total_displacement"]) == 9


# ── B1 calibration: application-layer projection profile ─────────────────────

class TestApplicationProfile:
    def test_sensitive_basis_orthonormal(self):
        # SENSITIVE_BASIS rows are v1, v2 — orthonormal
        assert SENSITIVE_BASIS.shape == (2, 9)
        gram = SENSITIVE_BASIS @ SENSITIVE_BASIS.T
        assert np.allclose(gram, np.eye(2), atol=1e-6)

    def test_epsilon_app_larger_than_dynamics(self):
        # App scale must be much larger to avoid instant saturation
        assert EPSILON_C_APP > EPSILON_C
        assert EPSILON_C_APP == pytest.approx(0.11, abs=1e-6)

    def test_projection_does_not_saturate_at_realistic_distance(self):
        # A player 0.05 along v1 saturates in euclidean but not in projection
        player = BASELINE_ATTRACTOR + 0.05 * V1
        eucl = compute_bifurcation_distance(player, mode="euclidean")
        proj = compute_bifurcation_distance(player, mode="projection")
        assert eucl["bifurcation_proximity"] >= 1.0
        assert proj["bifurcation_proximity"] < 1.0

    def test_projection_proximity_one_at_epsilon_app_along_v1(self):
        player = BASELINE_ATTRACTOR + EPSILON_C_APP * V1
        proj = compute_bifurcation_distance(player, mode="projection")
        assert proj["bifurcation_proximity"] == pytest.approx(1.0, abs=0.01)

    def test_off_plane_perturbation_has_low_projection_proximity(self):
        # A perturbation orthogonal to the v1-v2 plane should barely register
        off = np.cross(V1[:3], V2[:3])  # some direction; build a 9D off-plane vec
        vec = np.zeros(9)
        vec[3] = 1.0  # raw axis; remove its v1,v2 components
        vec = vec - (vec @ V1) * V1 - (vec @ V2) * V2
        vec /= np.linalg.norm(vec)
        player = BASELINE_ATTRACTOR + 0.05 * vec
        proj = compute_bifurcation_distance(player, mode="projection")
        assert proj["bifurcation_proximity"] < 0.05

    def test_mode_key_present(self):
        r = compute_bifurcation_distance(BASELINE_ATTRACTOR, mode="projection")
        assert r["mode"] == "projection"

    def test_app_sequence_climbs_smoothly(self):
        seq = event_sequence_planner(BASELINE_ATTRACTOR, n_steps=5, app_calibrated=True)
        proximities = [ev["proximity"] for ev in seq["events"]]
        # Strictly non-saturated early steps (smooth gradient, not instant 1.0)
        assert proximities[0] < 0.1
        assert 0.3 < proximities[1] < 0.9
        assert seq["predicted_final_proximity"] > 0.8

    def test_aligned_beats_random_under_app_profile(self):
        # v1-aligned events reach far higher projection proximity than random
        rng = np.random.RandomState(0)
        # aligned
        pv = BASELINE_ATTRACTOR.copy()
        for _ in range(5):
            ev = design_bifurcation_event(pv, app_calibrated=True)
            pv = pv + np.array(ev["displacement"])
        aligned = compute_bifurcation_distance(pv, mode="projection")["bifurcation_proximity"]
        # random
        pv = BASELINE_ATTRACTOR.copy()
        for _ in range(5):
            b = compute_bifurcation_distance(pv, mode="projection")
            mag = intensity_modulation(b["bifurcation_proximity"], scale=EPSILON_C_APP)
            d = rng.randn(9); d /= np.linalg.norm(d)
            pv = pv + d * mag
        random_prox = compute_bifurcation_distance(pv, mode="projection")["bifurcation_proximity"]
        assert aligned > random_prox + 0.3
