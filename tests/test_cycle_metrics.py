import pytest

from analysis.cycle_metrics import (
	assess_stage1_amplitude,
	assess_stage2_frequency,
	classify_cycle_level,
	dominant_frequency_autocorr,
	has_dominant_frequency,
	phase_direction_consistency,
	peak_to_peak_amplitude,
)


def test_peak_to_peak_amplitude_basic_and_burn_in():
	s = [0.1, 0.2, 0.15, 0.3]
	assert peak_to_peak_amplitude(s) == pytest.approx(0.2)
	# burn in removes the initial 0.1
	assert peak_to_peak_amplitude(s, burn_in=1) == pytest.approx(0.15)


def test_dominant_frequency_autocorr_finds_period_4():
	# Repeat a perfect 4-step pattern with zero mean.
	pattern = [0.0, 1.0, 0.0, -1.0]
	s = pattern * 30
	res = dominant_frequency_autocorr(s, min_lag=2, max_lag=20)
	assert res.lag == 4
	assert res.corr == pytest.approx(1.0, abs=1e-12)
	assert res.freq == pytest.approx(0.25)
	assert has_dominant_frequency(s, min_lag=2, max_lag=20, corr_threshold=0.9) is True


def test_dominant_frequency_autocorr_constant_series_returns_none():
	s = [0.2] * 100
	res = dominant_frequency_autocorr(s, min_lag=2, max_lag=20)
	assert res.lag is None
	assert res.corr == pytest.approx(0.0)
	assert res.freq is None
	assert has_dominant_frequency(s, min_lag=2, max_lag=20, corr_threshold=0.1) is False


def test_dominant_frequency_autocorr_avoids_degenerate_overlap_false_positive():
	# With overlap length = 1, Pearson-like corr is ±1 by construction.
	# Ensure we do NOT allow such extreme lags to dominate the selection.
	# This series is not meaningfully periodic.
	s = [0.0] * 250 + [1.0] * 250
	res = dominant_frequency_autocorr(s, min_lag=2, max_lag=499)
	# Before the overlap guard, this could incorrectly return lag=499 corr=1.0.
	assert not (res.lag == 499 and res.corr == pytest.approx(1.0))


def test_assess_stage1_amplitude_any_vs_all():
	props = {
		"aggressive": [0.1, 0.2, 0.1, 0.2],  # amp 0.1
		"defensive": [0.33, 0.331, 0.33, 0.331],  # amp 0.001
	}
	any_res = assess_stage1_amplitude(props, threshold=0.05, aggregation="any")
	assert any_res.passed is True
	all_res = assess_stage1_amplitude(props, threshold=0.05, aggregation="all")
	assert all_res.passed is False


def test_assess_stage2_frequency_any_passes_if_one_strategy_periodic():
	periodic = [0.0, 1.0, 0.0, -1.0] * 30
	flat = [0.2] * 120
	props = {"aggressive": periodic, "defensive": flat}
	res = assess_stage2_frequency(props, min_lag=2, max_lag=20, corr_threshold=0.9, aggregation="any")
	assert res.passed is True
	res_all = assess_stage2_frequency(props, min_lag=2, max_lag=20, corr_threshold=0.9, aggregation="all")
	assert res_all.passed is False


def test_phase_direction_consistency_detects_rotation():
	# Construct a small-radius loop around centroid in (pA, pD).
	# pB is implicit: 1 - pA - pD. Pick r small to keep all proportions positive.
	import math

	r = 0.05
	n = 200
	pa = []
	pd = []
	for i in range(n):
		theta = 2.0 * math.pi * (i / n)
		pa.append((1.0 / 3.0) + r * math.cos(theta))
		pd.append((1.0 / 3.0) + r * math.sin(theta))
	props = {
		"aggressive": pa,
		"defensive": pd,
		"balanced": [1.0 - a - d for a, d in zip(pa, pd)],
	}
	res = phase_direction_consistency(props, burn_in=0, eta=0.8)
	assert res.passed is True
	assert res.direction in (1, -1)
	assert res.score >= 0.8
	assert res.turn_strength > 0.0


def test_phase_direction_consistency_turn_strength_gate_blocks_tiny_motion():
	# Very tiny loop: direction may be consistent, but strength should be near 0.
	import math

	r = 1e-9
	n = 200
	pa = []
	pd = []
	for i in range(n):
		theta = 2.0 * math.pi * (i / n)
		pa.append((1.0 / 3.0) + r * math.cos(theta))
		pd.append((1.0 / 3.0) + r * math.sin(theta))
	props = {
		"aggressive": pa,
		"defensive": pd,
		"balanced": [1.0 - a - d for a, d in zip(pa, pd)],
	}
	res = phase_direction_consistency(props, eta=0.8, min_turn_strength=1e-6)
	assert res.passed is False


def test_classify_cycle_level_progression():
	# Level 0: flat proportions -> no amplitude.
	props0 = {"aggressive": [0.33] * 120, "defensive": [0.33] * 120, "balanced": [0.34] * 120}
	res0 = classify_cycle_level(props0, amplitude_threshold=0.02, corr_threshold=0.3, eta=0.6)
	assert res0.level == 0

	# Level 2: periodic in one strategy (amplitude + dominant freq), but no structure required here.
	periodic = [0.0, 1.0, 0.0, -1.0] * 30
	props2 = {"aggressive": periodic, "defensive": [0.0] * 120, "balanced": [0.0] * 120}
	res2 = classify_cycle_level(props2, amplitude_threshold=0.5, min_lag=2, max_lag=20, corr_threshold=0.9, eta=0.99)
	assert res2.level in (1, 2)
	# It should at least pass amplitude.
	assert res2.stage1.passed is True


def test_assess_stage2_frequency_fft_power_ratio_passes_on_periodic():
	periodic = [0.0, 1.0, 0.0, -1.0] * 60
	props = {"aggressive": periodic, "defensive": periodic}
	res = assess_stage2_frequency(
		props,
		min_lag=2,
		max_lag=20,
		corr_threshold=0.8,
		aggregation="any",
		stage2_method="fft_power_ratio",
		stage2_prefilter=True,
		power_ratio_kappa=8.0,
	)
	assert res.method == "fft_power_ratio"
	assert res.statistic_name == "power_ratio"
	assert res.passed is True
	assert res.statistic is not None
	assert float(res.statistic) >= 8.0
	assert res.effective_window_n == len(periodic)


def test_assess_stage2_frequency_permutation_p_passes_on_periodic_with_seed():
	periodic = [0.0, 1.0, 0.0, -1.0] * 60
	props = {"aggressive": periodic, "defensive": periodic}
	res = assess_stage2_frequency(
		props,
		min_lag=2,
		max_lag=20,
		corr_threshold=0.8,
		aggregation="any",
		stage2_method="permutation_p",
		stage2_prefilter=True,
		permutation_alpha=0.05,
		permutation_resamples=200,
		permutation_seed=123,
	)
	assert res.method == "permutation_p"
	assert res.statistic_name == "p_value"
	assert res.passed is True
	assert res.statistic is not None
	assert float(res.statistic) <= 0.05
	assert res.effective_window_n == len(periodic)


def test_stage2_prefilter_skips_fft_work_on_flat_series(monkeypatch):
	import analysis.cycle_metrics as cm

	def _boom(*args, **kwargs):
		raise AssertionError("fft helper should not be called when prefilter fails")

	monkeypatch.setattr(cm, "_fft_power_ratio_from_autocorr_peak", _boom)
	flat = [0.2] * 240
	props = {"aggressive": flat, "defensive": flat}
	res = assess_stage2_frequency(
		props,
		min_lag=2,
		max_lag=20,
		corr_threshold=0.9,
		stage2_method="fft_power_ratio",
		stage2_prefilter=True,
	)
	assert res.method == "fft_power_ratio"
	assert res.passed is False


def test_stage2_prefilter_skips_permutation_work_on_flat_series(monkeypatch):
	import analysis.cycle_metrics as cm

	def _boom(*args, **kwargs):
		raise AssertionError("permutation helper should not be called when prefilter fails")

	monkeypatch.setattr(cm, "_permutation_p_value_autocorr", _boom)
	flat = [0.2] * 240
	props = {"aggressive": flat, "defensive": flat}
	res = assess_stage2_frequency(
		props,
		min_lag=2,
		max_lag=20,
		corr_threshold=0.9,
		stage2_method="permutation_p",
		stage2_prefilter=True,
		permutation_resamples=50,
		permutation_seed=0,
	)
	assert res.method == "permutation_p"
	assert res.passed is False

