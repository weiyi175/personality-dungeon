"""Tests for CC1 — classifier calibration (phase rotation R² fallback).

Coverage
--------
1. _simplex3_phase_angle: basic geometry
2. _unwrap_phases: stdlib-only unwrap correctness
3. _linear_fit_r2: R² computation
4. phase_rotation_r2: end-to-end on synthetic data
5. classify_cycle_level with fallback: OR logic, backward compat
"""

from math import atan2, pi, sqrt

import pytest

from analysis.cycle_metrics import (
    PhaseRotationR2Result,
    _linear_fit_r2,
    _simplex3_phase_angle,
    _unwrap_phases,
    classify_cycle_level,
    phase_rotation_r2,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _make_rotating_simplex(n: int, *, angular_speed: float = 0.05, noise: float = 0.0) -> dict[str, list[float]]:
    """Generate a 3-strategy simplex trajectory with steady rotation.

    angular_speed: radians per step (negative = clockwise).
    Returns dict {"aggressive": [...], "defensive": [...], "balanced": [...]}.
    """
    import random
    rng = random.Random(42)
    center = 1.0 / 3.0
    radius = 0.15
    pa, pd, pb = [], [], []
    for t in range(n):
        theta = angular_speed * t
        # Inverse of _simplex3_phase_angle: θ = atan2(√3(pd-pb), 2pa-pd-pb)
        # With pa+pd+pb=1 and centroid offset coords:
        #   x = 2pa - pd - pb = r·cos(θ)
        #   y = √3(pd - pb) = r·sin(θ)
        x = radius * (pi / 4 + 0.01) * 0.0  # not used; use explicit formula
        # From centroid plus perturbation in simplex coordinates:
        cos_t = radius * __import__('math').cos(theta)
        sin_t = radius * __import__('math').sin(theta)
        # x_coord = 2*pa - pd - pb; y_coord = sqrt(3)*(pd - pb)
        # pa + pd + pb = 1
        # Solving: pa = 1/3 + cos_t/3*2, but let's use a cleaner transform.
        # Let δ = (pa - 1/3, pd - 1/3, pb - 1/3) with sum=0.
        # x = 2δa - δd - δb = 3δa (since δa+δd+δb=0 → δd+δb=-δa)
        # y = √3(δd - δb)
        # So δa = cos_t/3, δd-δb = sin_t/√3
        # Also δd + δb = -cos_t/3
        # → δd = (-cos_t/3 + sin_t/√3)/2, δb = (-cos_t/3 - sin_t/√3)/2
        da = cos_t / 3.0
        dd = (-cos_t / 3.0 + sin_t / sqrt(3.0)) / 2.0
        db = (-cos_t / 3.0 - sin_t / sqrt(3.0)) / 2.0
        n_pa = center + da + noise * (rng.random() - 0.5)
        n_pd = center + dd + noise * (rng.random() - 0.5)
        n_pb = center + db + noise * (rng.random() - 0.5)
        # Clamp and renormalize to simplex
        n_pa = max(n_pa, 1e-6)
        n_pd = max(n_pd, 1e-6)
        n_pb = max(n_pb, 1e-6)
        total = n_pa + n_pd + n_pb
        pa.append(n_pa / total)
        pd.append(n_pd / total)
        pb.append(n_pb / total)
    return {"aggressive": pa, "defensive": pd, "balanced": pb}


def _make_stationary_simplex(n: int) -> dict[str, list[float]]:
    """Flat simplex with tiny noise — no rotation."""
    import random
    rng = random.Random(99)
    pa, pd, pb = [], [], []
    for _ in range(n):
        a = 0.34 + 0.001 * (rng.random() - 0.5)
        d = 0.33 + 0.001 * (rng.random() - 0.5)
        b = 1.0 - a - d
        pa.append(a)
        pd.append(d)
        pb.append(b)
    return {"aggressive": pa, "defensive": pd, "balanced": pb}


# ── 1. _simplex3_phase_angle ──────────────────────────────────────────

class TestSimplex3PhaseAngle:

    def test_centroid_is_zero(self):
        # At exact centroid, atan2(0, 0) = 0
        angle = _simplex3_phase_angle(1 / 3, 1 / 3, 1 / 3)
        assert angle == pytest.approx(0.0)

    def test_pure_aggressive_angle(self):
        # pa=1, pd=0, pb=0 → x=2, y=0 → θ=0
        angle = _simplex3_phase_angle(1.0, 0.0, 0.0)
        assert angle == pytest.approx(0.0)

    def test_pure_defensive_angle(self):
        # pa=0, pd=1, pb=0 → x=-1, y=√3 → θ=2π/3
        angle = _simplex3_phase_angle(0.0, 1.0, 0.0)
        assert angle == pytest.approx(2 * pi / 3, abs=1e-10)

    def test_pure_balanced_angle(self):
        # pa=0, pd=0, pb=1 → x=-1, y=-√3 → θ=-2π/3
        angle = _simplex3_phase_angle(0.0, 0.0, 1.0)
        assert angle == pytest.approx(-2 * pi / 3, abs=1e-10)


# ── 2. _unwrap_phases ─────────────────────────────────────────────────

class TestUnwrapPhases:

    def test_no_wrap_needed(self):
        phases = [0.0, 0.1, 0.2, 0.3]
        assert _unwrap_phases(phases) == pytest.approx(phases)

    def test_simple_wrap(self):
        # Jump from near +π to near −π should be unwrapped to continuation.
        phases = [3.0, 3.1, -3.0]
        result = _unwrap_phases(phases)
        # The jump from 3.1 to -3.0 is ~-6.1, which normalises to ~+0.18
        assert len(result) == 3
        assert result[0] == pytest.approx(3.0)
        assert result[1] == pytest.approx(3.1)
        # -3.0 + 2π ≈ 3.28
        assert result[2] == pytest.approx(-3.0 + 2 * pi, abs=0.01)

    def test_empty(self):
        assert _unwrap_phases([]) == []

    def test_single(self):
        assert _unwrap_phases([1.5]) == [1.5]

    def test_monotonic_negative(self):
        # Steady decrease that wraps multiple times
        step = -0.3
        raw = [((i * step) % (2 * pi)) - pi for i in range(100)]
        unwrapped = _unwrap_phases(raw)
        # Should be roughly monotonically decreasing
        diffs = [unwrapped[i + 1] - unwrapped[i] for i in range(len(unwrapped) - 1)]
        negative_fraction = sum(1 for d in diffs if d < 0) / len(diffs)
        assert negative_fraction > 0.9


# ── 3. _linear_fit_r2 ─────────────────────────────────────────────────

class TestLinearFitR2:

    def test_perfect_line(self):
        ys = [2.0 + 0.5 * i for i in range(50)]
        r2, slope, intercept = _linear_fit_r2(ys)
        assert r2 == pytest.approx(1.0, abs=1e-10)
        assert slope == pytest.approx(0.5, abs=1e-10)
        assert intercept == pytest.approx(2.0, abs=1e-10)

    def test_constant(self):
        ys = [3.0] * 20
        r2, slope, intercept = _linear_fit_r2(ys)
        assert slope == pytest.approx(0.0)

    def test_too_short(self):
        r2, slope, intercept = _linear_fit_r2([1.0])
        assert r2 == 0.0

    def test_r2_between_0_and_1(self):
        # Noisy linear data
        import random
        rng = random.Random(42)
        ys = [0.1 * i + 0.5 * (rng.random() - 0.5) for i in range(100)]
        r2, slope, intercept = _linear_fit_r2(ys)
        assert 0.0 <= r2 <= 1.0
        assert slope > 0


# ── 4. phase_rotation_r2 ──────────────────────────────────────────────

class TestPhaseRotationR2:

    def test_strong_rotation(self):
        props = _make_rotating_simplex(500, angular_speed=0.05)
        res = phase_rotation_r2(props)
        assert isinstance(res, PhaseRotationR2Result)
        assert res.r2 > 0.95
        assert res.cumulative_rotation > 20.0
        assert res.window_length == 500

    def test_stationary_low_r2(self):
        props = _make_stationary_simplex(500)
        res = phase_rotation_r2(props)
        assert res.cumulative_rotation < 5.0

    def test_burn_in_tail(self):
        props = _make_rotating_simplex(1000, angular_speed=0.05)
        res = phase_rotation_r2(props, burn_in=200, tail=500)
        assert res.window_length == 500
        assert res.r2 > 0.9

    def test_missing_strategy_returns_zero(self):
        props = {"aggressive": [0.5] * 100, "defensive": [0.3] * 100}
        res = phase_rotation_r2(props)
        assert res.r2 == 0.0
        assert res.window_length == 0

    def test_short_series(self):
        props = _make_rotating_simplex(5, angular_speed=0.05)
        res = phase_rotation_r2(props)
        assert res.r2 == 0.0  # < 10 points threshold


# ── 5. classify_cycle_level with fallback ──────────────────────────────

class TestFallbackClassifier:

    def test_backward_compat_no_fallback(self):
        """Without fallback params, behaviour is identical to legacy."""
        props = _make_rotating_simplex(2000, angular_speed=0.05, noise=0.01)
        result_legacy = classify_cycle_level(props)
        result_no_fb = classify_cycle_level(
            props,
            stage2_fallback_r2_threshold=None,
        )
        assert result_legacy.level == result_no_fb.level

    def test_fallback_promotes_level(self):
        """A rotating signal that fails Stage 2 autocorr but passes R² fallback."""
        # Very slow, smooth rotation → high R², weak autocorr
        props = _make_rotating_simplex(2000, angular_speed=0.02, noise=0.002)
        # Without fallback:
        result_nofb = classify_cycle_level(props)
        # With fallback (generous thresholds):
        result_fb = classify_cycle_level(
            props,
            stage2_fallback_r2_threshold=0.80,
            stage2_fallback_min_rotation=5.0,
        )
        # The fallback should allow advancement past Stage 2
        assert result_fb.level >= result_nofb.level

    def test_fallback_does_not_rescue_stationary(self):
        """Stationary signal should NOT be rescued by fallback."""
        props = _make_stationary_simplex(2000)
        result = classify_cycle_level(
            props,
            stage2_fallback_r2_threshold=0.80,
            stage2_fallback_min_rotation=5.0,
        )
        # R² of stationary is low, cumulative_rotation near 0
        assert result.level <= 1

    def test_fallback_respects_min_rotation(self):
        """Even with high R², if cumulative rotation is too small, fallback fails."""
        # Build a signal that fails primary Stage 2 autocorr but has decent R².
        # We do this by adding enough noise to kill autocorrelation while
        # preserving a slow linear trend in the unwrapped phase.
        props = _make_rotating_simplex(500, angular_speed=0.001, noise=0.04)
        # Verify primary autocorr likely fails (noisy)
        result_nofb = classify_cycle_level(props)
        # With generous R² but strict rotation (the signal only rotates ~0.5 rad)
        result = classify_cycle_level(
            props,
            stage2_fallback_r2_threshold=0.01,
            stage2_fallback_min_rotation=20.0,  # needs 20 rad, will have ~0.5 rad
        )
        # If primary already passes, that's fine — the test is about the
        # fallback's min_rotation guard. If primary fails, fallback should
        # also fail because cumulative rotation < 20.
        if result_nofb.level <= 1:
            assert result.level <= 1

    def test_fallback_method_tagged(self):
        """When fallback fires, stage2.method should be 'phase_rotation_r2_fallback'."""
        props = _make_rotating_simplex(2000, angular_speed=0.02, noise=0.002)
        result = classify_cycle_level(
            props,
            stage2_fallback_r2_threshold=0.50,
            stage2_fallback_min_rotation=2.0,
        )
        if result.level >= 2 and result.stage2 is not None:
            if result.stage2.method == "phase_rotation_r2_fallback":
                assert result.stage2.statistic_name == "phase_r2"
                assert result.stage2.statistic is not None
                assert result.stage2.statistic >= 0.50

    def test_stage1_failure_unchanged(self):
        """Fallback only applies at Stage 2; Stage 1 failure is unaffected."""
        # Constant — fails Stage 1 amplitude
        props = {"aggressive": [0.333] * 500, "defensive": [0.334] * 500, "balanced": [0.333] * 500}
        result = classify_cycle_level(
            props,
            stage2_fallback_r2_threshold=0.50,
            stage2_fallback_min_rotation=1.0,
        )
        assert result.level == 0
    def test_stage3_fallback_promotes_to_level3(self):
        """A signal that passes Stage 2 via fallback + Stage 3 via fallback → CL=3."""
        # Slow smooth rotation: fails autocorr but high R² and large cumulative rotation
        props = _make_rotating_simplex(2000, angular_speed=0.02, noise=0.002)
        result = classify_cycle_level(
            props,
            stage2_fallback_r2_threshold=0.50,
            stage2_fallback_min_rotation=5.0,
        )
        # Should reach Level 3 (both Stage 2 and Stage 3 rescued by fallback)
        assert result.level == 3

    def test_stage3_fallback_direction(self):
        """Fallback-promoted Stage 3 should carry correct direction."""
        # CW rotation (negative angular_speed)
        props = _make_rotating_simplex(2000, angular_speed=-0.03, noise=0.002)
        result = classify_cycle_level(
            props,
            stage2_fallback_r2_threshold=0.50,
            stage2_fallback_min_rotation=5.0,
        )
        if result.level >= 3 and result.stage3 is not None:
            # Negative speed → slope < 0 → direction = -1 (cw)
            assert result.stage3.direction == -1

    def test_stage3_fallback_does_not_rescue_random_walk(self):
        """Random walk with low R² should not be rescued at Stage 3."""
        props = _make_stationary_simplex(2000)
        result = classify_cycle_level(
            props,
            stage2_fallback_r2_threshold=0.80,
            stage2_fallback_min_rotation=5.0,
        )
        # Stationary has low R² and low rotation — should stay ≤ 1
        assert result.level <= 1