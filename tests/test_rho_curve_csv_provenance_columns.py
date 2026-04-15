import pytest

from simulation.rho_curve import SWEEP_CSV_FIELDNAMES, _run_one_seed


def test_rho_curve_sweep_csv_includes_protocol_and_stage3_provenance_columns():
	# Protocol
	for name in (
		"payoff_mode",
		"gamma",
		"epsilon",
		"seeds",
		"k_grid",
		"players_grid",
		"memory_kernel",
		"threshold_theta",
		"threshold_theta_low",
		"threshold_theta_high",
		"threshold_trigger",
		"threshold_state_alpha",
		"threshold_a_hi",
		"threshold_b_hi",
	):
		assert name in SWEEP_CSV_FIELDNAMES

	# Stage3 settings
	for name in ("stage3_method", "phase_smoothing", "stage3_window", "stage3_step", "stage3_quantile"):
		assert name in SWEEP_CSV_FIELDNAMES

	# Sanity: legacy columns remain present
	for name in ("players", "selection_strength", "series", "rounds", "burn_in", "tail"):
		assert name in SWEEP_CSV_FIELDNAMES


def test_run_one_seed_threshold_ab_noop_matches_matrix_ab_mean_field():
	metric_kwargs = {
		"amplitude_threshold": 0.0,
		"amplitude_normalization": "none",
		"amplitude_control_reference": None,
		"amplitude_threshold_factor": 3.0,
		"min_lag": 2,
		"max_lag": 8,
		"corr_threshold": 0.0,
		"stage2_method": "autocorr_threshold",
		"stage2_prefilter": True,
		"power_ratio_kappa": 8.0,
		"permutation_alpha": 0.05,
		"permutation_resamples": 32,
		"permutation_seed": 0,
		"eta": 0.0,
		"min_turn_strength": 0.0,
		"normalize_for_phase": True,
		"stage3_method": "turning",
		"phase_smoothing": 1,
		"stage3_window": None,
		"stage3_step": 1,
		"stage3_quantile": 0.75,
	}
	common = {
		"seed": 7,
		"players": 40,
		"rounds": 24,
		"popularity_mode": "expected",
		"evolution_mode": "mean_field",
		"payoff_lag": 1,
		"memory_kernel": 3,
		"matrix_cross_coupling": 0.2,
		"gamma": 0.1,
		"epsilon": 0.0,
		"a": 1.0,
		"b": 0.9,
		"init_bias": 0.12,
		"selection_strength": 0.06,
		"series": "w",
		"burn_in": 0,
		"tail": None,
		"metric_kwargs": metric_kwargs,
	}
	base = _run_one_seed(
		payoff_mode="matrix_ab",
		threshold_theta=0.40,
		threshold_theta_low=None,
		threshold_theta_high=None,
		threshold_trigger="ad_share",
		threshold_state_alpha=1.0,
		threshold_a_hi=None,
		threshold_b_hi=None,
		**common,
	)
	noop = _run_one_seed(
		payoff_mode="threshold_ab",
		threshold_theta=0.40,
		threshold_theta_low=0.40,
		threshold_theta_high=0.40,
		threshold_trigger="ad_share",
		threshold_state_alpha=1.0,
		threshold_a_hi=1.0,
		threshold_b_hi=0.9,
		**common,
	)

	assert noop.level == base.level
	assert noop.stage3_passed == base.stage3_passed
	assert noop.stage2_statistic_name == base.stage2_statistic_name
	assert noop.stage2_effective_window_n == base.stage2_effective_window_n
	assert noop.stage3_score == pytest.approx(base.stage3_score)
	assert noop.env_gamma == pytest.approx(base.env_gamma)
	assert noop.env_gamma_r2 == pytest.approx(base.env_gamma_r2)
	assert noop.max_amp == pytest.approx(base.max_amp)
	assert noop.max_corr == pytest.approx(base.max_corr)
	assert noop.stage2_statistic == pytest.approx(base.stage2_statistic)
