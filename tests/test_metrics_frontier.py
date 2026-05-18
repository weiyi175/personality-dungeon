import numpy as np
import pytest

from analysis.metrics import (
	calculate_resonance_score,
	calculate_slr,
	calculate_peak_relative_width,
	calculate_sweetspot_width,
	calculate_z_drift,
)


def test_calculate_peak_relative_width_example_plateau_stays_stable_under_trailing_smoothing():
	re = np.array([10.0, 10.0, 10.0, 10.0, 8.0])
	assert calculate_peak_relative_width(re, peak_fraction=0.95, gap_tolerance=0) == 4
	assert calculate_sweetspot_width(re, threshold=22000, gap_tolerance=0) == 4


def test_calculate_peak_relative_width_rejects_two_step_gap_when_tolerance_is_one():
	re = np.array([10.0, 10.0, 10.0, 9.0, 9.0, 10.0, 10.0, 10.0])
	assert calculate_peak_relative_width(re, peak_fraction=0.95, gap_tolerance=1) == 4
	assert calculate_peak_relative_width(re, peak_fraction=0.95, gap_tolerance=2) == 8


def test_calculate_z_drift_returns_zero_for_constant_trajectory():
	z_history = np.zeros((60, 3), dtype=float)
	assert calculate_z_drift(z_history) == (0.0, 0.0)


def test_calculate_z_drift_reports_positive_drift_for_linear_axis():
	steps = np.arange(80, dtype=float)
	z_history = np.column_stack([steps, np.zeros_like(steps), np.zeros_like(steps)])
	median_sigma, p95_sigma = calculate_z_drift(z_history)
	assert median_sigma > 0.0
	assert p95_sigma >= median_sigma


def test_calculate_resonance_score_detects_aligned_component():
	t = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
	re = np.sin(t)
	z_history = np.column_stack([
		np.sin(t),
		np.zeros_like(t),
		np.ones_like(t),
	])
	scores = calculate_resonance_score(re, z_history, max_lag=12)
	assert scores[0] > 0.99
	assert scores[1] == 0.0
	assert scores[2] == 0.0


def test_calculate_slr_handles_scalar_and_vector_inputs():
	assert calculate_slr(10.0, 7.0) == pytest.approx(0.3)
	assert calculate_slr(10.0, 12.0) == 0.0
	assert calculate_slr(0.0, 1.0) == 0.0

	base = np.array([10.0, 0.0])
	asy = np.array([7.0, 1.0])
	result = calculate_slr(base, asy)
	assert np.allclose(result, np.array([0.3, 0.0]))