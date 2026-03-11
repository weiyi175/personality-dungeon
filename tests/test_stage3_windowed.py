from __future__ import annotations

import math


def _synthetic_cycle_weights(n: int, *, direction: int, period: int = 40, amp: float = 0.35) -> dict[str, list[float]]:
	# Three phase-shifted cosines stay positive for amp<1.
	# Normalization happens inside metrics.
	w1: list[float] = []
	w2: list[float] = []
	w3: list[float] = []
	for t in range(n):
		theta = float(direction) * (2.0 * math.pi) * (float(t) / float(period))
		w1.append(1.0 + amp * math.cos(theta))
		w2.append(1.0 + amp * math.cos(theta + (2.0 * math.pi / 3.0)))
		w3.append(1.0 + amp * math.cos(theta + (4.0 * math.pi / 3.0)))
	return {"aggressive": w1, "defensive": w2, "balanced": w3}


def test_stage3_turning_window_matches_full_when_single_window() -> None:
	from analysis.cycle_metrics import phase_direction_consistency_turning, phase_direction_consistency_turning_windowed

	series = _synthetic_cycle_weights(120, direction=1)
	full = phase_direction_consistency_turning(series, eta=0.65, min_turn_strength=0.0, phase_smoothing=1)
	win = phase_direction_consistency_turning_windowed(
		series,
		eta=0.65,
		min_turn_strength=0.0,
		phase_smoothing=1,
		window=120,
		step=120,
		quantile=0.75,
	)
	assert abs(win.score - full.score) < 1e-12
	assert abs(win.turn_strength - full.turn_strength) < 1e-12
	assert win.direction == full.direction


def test_windowed_p75_score_is_high_on_mostly_ccw_series() -> None:
	from analysis.cycle_metrics import phase_direction_consistency_turning, phase_direction_consistency_turning_windowed

	# 80% CCW then 20% CW; full-tail turning can be dragged down by the reversal boundary,
	# while windowed quantile should stay closer to the dominant regime.
	ccw = _synthetic_cycle_weights(160, direction=1)
	cw = _synthetic_cycle_weights(40, direction=-1)
	series = {
		"aggressive": ccw["aggressive"] + cw["aggressive"],
		"defensive": ccw["defensive"] + cw["defensive"],
		"balanced": ccw["balanced"] + cw["balanced"],
	}
	full = phase_direction_consistency_turning(series, eta=0.65, min_turn_strength=0.0, phase_smoothing=1)
	win = phase_direction_consistency_turning_windowed(
		series,
		eta=0.65,
		min_turn_strength=0.0,
		phase_smoothing=1,
		window=60,
		step=20,
		quantile=0.75,
	)
	assert win.score >= full.score
	assert win.direction in (-1, 1)


def test_classify_cycle_level_uses_windowed_stage3_when_enabled() -> None:
	from analysis.cycle_metrics import classify_cycle_level

	ccw = _synthetic_cycle_weights(160, direction=1)
	cw = _synthetic_cycle_weights(40, direction=-1)
	series = {
		"aggressive": ccw["aggressive"] + cw["aggressive"],
		"defensive": ccw["defensive"] + cw["defensive"],
		"balanced": ccw["balanced"] + cw["balanced"],
	}

	# Relax stage1/2 thresholds so the test focuses on stage3 selection.
	res_full = classify_cycle_level(
		series,
		burn_in=0,
		tail=None,
		amplitude_threshold=0.01,
		min_lag=2,
		max_lag=80,
		corr_threshold=-1.0,
		eta=0.65,
		min_turn_strength=0.0,
		normalize_for_phase=True,
		stage3_method="turning",
		phase_smoothing=1,
	)
	res_win = classify_cycle_level(
		series,
		burn_in=0,
		tail=None,
		amplitude_threshold=0.01,
		min_lag=2,
		max_lag=80,
		corr_threshold=-1.0,
		eta=0.65,
		min_turn_strength=0.0,
		normalize_for_phase=True,
		stage3_method="turning",
		phase_smoothing=1,
		stage3_window=60,
		stage3_step=20,
		stage3_quantile=0.75,
	)

	assert res_full.stage3 is not None and res_win.stage3 is not None
	# With windowing enabled the Stage3 score should reflect the windowed aggregation.
	assert res_win.stage3.score >= res_full.stage3.score
