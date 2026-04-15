import pytest

from pathlib import Path

from simulation.run_simulation import SimConfig, simulate, simulate_series_window


@pytest.mark.parametrize("series", ["p", "w"])
def test_simulate_series_window_matches_simulate_rows(series: str):
	cfg = SimConfig(
		n_players=30,
		n_rounds=50,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=1.2,
		matrix_cross_coupling=0.0,
		init_bias=0.0,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.02,
		out_csv=Path("outputs") / "_ignored.csv",
		memory_kernel=1,
	)

	burn_in = 10
	tail = 20

	series_window = simulate_series_window(cfg, series=series, burn_in=burn_in, tail=tail)
	strategy_space, rows = simulate(cfg)

	# The window semantics should match analysis.cycle_metrics._slice_series
	begin = max(burn_in, cfg.n_rounds - tail)
	window_rows = rows[begin:]

	assert set(series_window.keys()) == set(strategy_space)
	for s in strategy_space:
		if series == "p":
			expected = [float(r[f"p_{s}"]) for r in window_rows]
		else:
			expected = [float(r[f"w_{s}"]) for r in window_rows]
		assert series_window[s] == pytest.approx(expected, abs=0.0)


@pytest.mark.parametrize("series", ["p", "w"])
def test_simulate_series_window_mean_field_matches_simulate_rows_with_memory_kernel(series: str):
	cfg = SimConfig(
		n_players=0,
		n_rounds=40,
		seed=45,
		payoff_mode="matrix_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="mean_field",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_mean_field.csv",
		memory_kernel=3,
	)

	burn_in = 5
	tail = 12

	series_window = simulate_series_window(cfg, series=series, burn_in=burn_in, tail=tail)
	strategy_space, rows = simulate(cfg)

	begin = max(burn_in, cfg.n_rounds - tail)
	window_rows = rows[begin:]

	assert set(series_window.keys()) == set(strategy_space)
	for s in strategy_space:
		if series == "p":
			expected = [float(r[f"p_{s}"]) for r in window_rows]
		else:
			expected = [float(r[f"w_{s}"]) for r in window_rows]
		assert series_window[s] == pytest.approx(expected, abs=0.0)


def test_simulate_tangential_drift_zero_matches_sampled_control_rows():
	common = dict(
		n_players=30,
		n_rounds=40,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_b5_sampled.csv",
		memory_kernel=3,
	)
	strategy_space, base_rows = simulate(SimConfig(**common))
	strategy_space_drift, drift_rows = simulate(SimConfig(tangential_drift_delta=0.0, **common))

	assert strategy_space_drift == strategy_space
	assert len(base_rows) == len(drift_rows)
	for base_row, drift_row in zip(base_rows, drift_rows, strict=True):
		for key in ["round", *[f"p_{s}" for s in strategy_space], *[f"w_{s}" for s in strategy_space]]:
			assert drift_row[key] == pytest.approx(base_row[key], abs=0.0)


def test_simulate_tangential_drift_zero_matches_mean_field_control_rows():
	common = dict(
		n_players=0,
		n_rounds=40,
		seed=45,
		payoff_mode="matrix_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="mean_field",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_b5_mean_field.csv",
		memory_kernel=3,
	)
	strategy_space, base_rows = simulate(SimConfig(**common))
	strategy_space_drift, drift_rows = simulate(SimConfig(tangential_drift_delta=0.0, **common))

	assert strategy_space_drift == strategy_space
	assert len(base_rows) == len(drift_rows)
	for base_row, drift_row in zip(base_rows, drift_rows, strict=True):
		for key in ["round", *[f"p_{s}" for s in strategy_space], *[f"w_{s}" for s in strategy_space]]:
			assert drift_row[key] == pytest.approx(base_row[key], abs=0.0)


def test_threshold_ab_noop_mean_field_matches_matrix_ab_rows():
	base_cfg = SimConfig(
		n_players=0,
		n_rounds=40,
		seed=45,
		payoff_mode="matrix_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="mean_field",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_matrix.csv",
		memory_kernel=3,
	)
	thr_cfg = SimConfig(
		n_players=0,
		n_rounds=40,
		seed=45,
		payoff_mode="threshold_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="mean_field",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_threshold.csv",
		memory_kernel=3,
		threshold_theta=0.40,
		threshold_a_hi=1.0,
		threshold_b_hi=0.9,
	)

	strategy_space, base_rows = simulate(base_cfg)
	strategy_space_thr, thr_rows = simulate(thr_cfg)

	assert strategy_space_thr == strategy_space
	assert len(base_rows) == len(thr_rows)
	for base_row, thr_row in zip(base_rows, thr_rows, strict=True):
		for key in ["round", *[f"p_{s}" for s in strategy_space], *[f"w_{s}" for s in strategy_space]]:
			assert thr_row[key] == pytest.approx(base_row[key], abs=0.0)


def test_threshold_ab_hysteresis_equal_bounds_match_plain_threshold_mean_field_rows():
	base_cfg = SimConfig(
		n_players=0,
		n_rounds=40,
		seed=45,
		payoff_mode="threshold_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="mean_field",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_threshold_plain.csv",
		memory_kernel=3,
		threshold_theta=0.55,
		threshold_a_hi=1.1,
		threshold_b_hi=1.0,
	)
	hys_cfg = SimConfig(
		n_players=0,
		n_rounds=40,
		seed=45,
		payoff_mode="threshold_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="mean_field",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_threshold_hys.csv",
		memory_kernel=3,
		threshold_theta=0.55,
		threshold_theta_low=0.55,
		threshold_theta_high=0.55,
		threshold_a_hi=1.1,
		threshold_b_hi=1.0,
	)

	strategy_space, base_rows = simulate(base_cfg)
	strategy_space_hys, hys_rows = simulate(hys_cfg)

	assert strategy_space_hys == strategy_space
	assert len(base_rows) == len(hys_rows)
	for base_row, hys_row in zip(base_rows, hys_rows, strict=True):
		for key in ["round", *[f"p_{s}" for s in strategy_space], *[f"w_{s}" for s in strategy_space]]:
			assert hys_row[key] == pytest.approx(base_row[key], abs=0.0)


def test_threshold_ab_h22_backward_compatible_with_plain_threshold_mean_field_rows():
	base_cfg = SimConfig(
		n_players=0,
		n_rounds=40,
		seed=45,
		payoff_mode="threshold_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="mean_field",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_threshold_h22_base.csv",
		memory_kernel=3,
		threshold_theta=0.55,
		threshold_a_hi=1.1,
		threshold_b_hi=1.0,
	)
	h22_cfg = SimConfig(
		n_players=0,
		n_rounds=40,
		seed=45,
		payoff_mode="threshold_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="mean_field",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_threshold_h22.csv",
		memory_kernel=3,
		threshold_theta=0.55,
		threshold_theta_low=0.55,
		threshold_theta_high=0.55,
		threshold_trigger="ad_share",
		threshold_state_alpha=1.0,
		threshold_a_hi=1.1,
		threshold_b_hi=1.0,
	)

	strategy_space, base_rows = simulate(base_cfg)
	strategy_space_h22, h22_rows = simulate(h22_cfg)

	assert strategy_space_h22 == strategy_space
	assert len(base_rows) == len(h22_rows)
	for base_row, h22_row in zip(base_rows, h22_rows, strict=True):
		for key in ["round", *[f"p_{s}" for s in strategy_space], *[f"w_{s}" for s in strategy_space]]:
			assert h22_row[key] == pytest.approx(base_row[key], abs=0.0)


def test_h4_zero_hybrid_share_matches_sampled_rows():
	base_cfg = SimConfig(
		n_players=30,
		n_rounds=20,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_h4_base.csv",
		memory_kernel=3,
	)
	h4_cfg = SimConfig(
		**{
			**base_cfg.__dict__,
			"evolution_mode": "hybrid",
			"hybrid_update_share": 0.0,
			"out_csv": Path("outputs") / "_ignored_h4_zero.csv",
		}
	)

	strategy_space, base_rows = simulate(base_cfg)
	strategy_space_h4, h4_rows = simulate(h4_cfg)

	assert strategy_space_h4 == strategy_space
	assert len(base_rows) == len(h4_rows)
	for base_row, h4_row in zip(base_rows, h4_rows, strict=True):
		for key in ["round", *[f"p_{s}" for s in strategy_space], *[f"w_{s}" for s in strategy_space]]:
			assert h4_row[key] == pytest.approx(base_row[key], abs=0.0)


def test_h4_positive_hybrid_share_changes_sampled_path():
	base_cfg = SimConfig(
		n_players=40,
		n_rounds=30,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_h4_base_sampled.csv",
		memory_kernel=3,
	)
	h4_cfg = SimConfig(
		**{
			**base_cfg.__dict__,
			"evolution_mode": "hybrid",
			"hybrid_update_share": 0.2,
			"out_csv": Path("outputs") / "_ignored_h4_hybrid.csv",
		}
	)

	strategy_space, base_rows = simulate(base_cfg)
	strategy_space_h4, h4_rows = simulate(h4_cfg)

	assert strategy_space_h4 == strategy_space
	assert len(base_rows) == len(h4_rows)
	assert any(
		h4_row[f"w_{s}"] != pytest.approx(base_row[f"w_{s}"], abs=1e-12)
		for base_row, h4_row in zip(base_rows, h4_rows, strict=True)
		for s in strategy_space
	)


def test_h41_zero_inertia_matches_h4_rows():
	h4_cfg = SimConfig(
		n_players=40,
		n_rounds=30,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="hybrid",
		payoff_lag=1,
		selection_strength=0.06,
		hybrid_update_share=0.2,
		out_csv=Path("outputs") / "_ignored_h41_base.csv",
		memory_kernel=3,
	)
	h41_cfg = SimConfig(
		**{
			**h4_cfg.__dict__,
			"hybrid_inertia": 0.0,
			"out_csv": Path("outputs") / "_ignored_h41_zero.csv",
		}
	)

	strategy_space, h4_rows = simulate(h4_cfg)
	strategy_space_h41, h41_rows = simulate(h41_cfg)

	assert strategy_space_h41 == strategy_space
	assert len(h4_rows) == len(h41_rows)
	for h4_row, h41_row in zip(h4_rows, h41_rows, strict=True):
		for key in ["round", *[f"p_{s}" for s in strategy_space], *[f"w_{s}" for s in strategy_space]]:
			assert h41_row[key] == pytest.approx(h4_row[key], abs=0.0)


def test_h41_positive_inertia_changes_h4_path():
	h4_cfg = SimConfig(
		n_players=40,
		n_rounds=30,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="hybrid",
		payoff_lag=1,
		selection_strength=0.06,
		hybrid_update_share=0.2,
		out_csv=Path("outputs") / "_ignored_h41_h4.csv",
		memory_kernel=3,
	)
	h41_cfg = SimConfig(
		**{
			**h4_cfg.__dict__,
			"hybrid_inertia": 0.4,
			"out_csv": Path("outputs") / "_ignored_h41_inertia.csv",
		}
	)

	strategy_space, h4_rows = simulate(h4_cfg)
	strategy_space_h41, h41_rows = simulate(h41_cfg)

	assert strategy_space_h41 == strategy_space
	assert len(h4_rows) == len(h41_rows)
	assert any(
		h41_row[f"w_{s}"] != pytest.approx(h4_row[f"w_{s}"], abs=1e-12)
		for h4_row, h41_row in zip(h4_rows, h41_rows, strict=True)
		for s in strategy_space
	)


def test_h51_zero_sampled_inertia_matches_sampled_rows():
	base_cfg = SimConfig(
		n_players=40,
		n_rounds=30,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_h51_base.csv",
		memory_kernel=3,
	)
	h51_cfg = SimConfig(
		**{
			**base_cfg.__dict__,
			"evolution_mode": "sampled_inertial",
			"sampled_inertia": 0.0,
			"out_csv": Path("outputs") / "_ignored_h51_zero.csv",
		}
	)

	strategy_space, base_rows = simulate(base_cfg)
	strategy_space_h51, h51_rows = simulate(h51_cfg)

	assert strategy_space_h51 == strategy_space
	assert len(base_rows) == len(h51_rows)
	for base_row, h51_row in zip(base_rows, h51_rows, strict=True):
		for key in ["round", *[f"p_{s}" for s in strategy_space], *[f"w_{s}" for s in strategy_space]]:
			assert h51_row[key] == pytest.approx(base_row[key], abs=0.0)


def test_h51_positive_sampled_inertia_changes_sampled_path():
	base_cfg = SimConfig(
		n_players=40,
		n_rounds=30,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_h51_base_sampled.csv",
		memory_kernel=3,
	)
	h51_cfg = SimConfig(
		**{
			**base_cfg.__dict__,
			"evolution_mode": "sampled_inertial",
			"sampled_inertia": 0.3,
			"out_csv": Path("outputs") / "_ignored_h51_inertia.csv",
		}
	)

	strategy_space, base_rows = simulate(base_cfg)
	strategy_space_h51, h51_rows = simulate(h51_cfg)

	assert strategy_space_h51 == strategy_space
	assert len(base_rows) == len(h51_rows)
	assert any(
		h51_row[f"w_{s}"] != pytest.approx(base_row[f"w_{s}"], abs=1e-12)
		for base_row, h51_row in zip(base_rows, h51_rows, strict=True)
		for s in strategy_space
	)


def test_h51_mean_field_zero_sampled_inertia_matches_mean_field_rows():
	base_cfg = SimConfig(
		n_players=0,
		n_rounds=40,
		seed=45,
		payoff_mode="matrix_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="mean_field",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_h51_mean_field.csv",
		memory_kernel=3,
	)
	h51_cfg = SimConfig(
		**{
			**base_cfg.__dict__,
			"sampled_inertia": 0.0,
			"out_csv": Path("outputs") / "_ignored_h51_mean_field_zero.csv",
		}
	)

	strategy_space, base_rows = simulate(base_cfg)
	strategy_space_h51, h51_rows = simulate(h51_cfg)

	assert strategy_space_h51 == strategy_space
	assert len(base_rows) == len(h51_rows)
	for base_row, h51_row in zip(base_rows, h51_rows, strict=True):
		for key in ["round", *[f"p_{s}" for s in strategy_space], *[f"w_{s}" for s in strategy_space]]:
			assert h51_row[key] == pytest.approx(base_row[key], abs=0.0)


def test_h4_requires_sampled_popularity():
	cfg = SimConfig(
		n_players=30,
		n_rounds=10,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="hybrid",
		payoff_lag=1,
		selection_strength=0.06,
		hybrid_update_share=0.2,
		out_csv=Path("outputs") / "_ignored_h4_expected_invalid.csv",
		memory_kernel=3,
	)
	with pytest.raises(ValueError):
		simulate(cfg)


def test_h4_disallows_fixed_subgroup_family():
	cfg = SimConfig(
		n_players=30,
		n_rounds=10,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="hybrid",
		payoff_lag=1,
		selection_strength=0.06,
		hybrid_update_share=0.2,
		fixed_subgroup_share=0.2,
		fixed_subgroup_weights=(0.8, 0.8, 1.4),
		out_csv=Path("outputs") / "_ignored_h4_subgroup_invalid.csv",
		memory_kernel=3,
	)
	with pytest.raises(ValueError):
		simulate(cfg)


def test_h41_inertia_requires_hybrid_mode():
	cfg = SimConfig(
		n_players=30,
		n_rounds=10,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		hybrid_inertia=0.3,
		out_csv=Path("outputs") / "_ignored_h41_invalid.csv",
		memory_kernel=3,
	)
	with pytest.raises(ValueError):
		simulate(cfg)


def test_h51_sampled_inertia_requires_sampled_inertial_or_mean_field_mode():
	cfg = SimConfig(
		n_players=30,
		n_rounds=10,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		sampled_inertia=0.3,
		out_csv=Path("outputs") / "_ignored_h51_invalid.csv",
		memory_kernel=3,
	)
	with pytest.raises(ValueError):
		simulate(cfg)


def test_h51_disallows_hybrid_or_subgroup_family():
	cfg = SimConfig(
		n_players=30,
		n_rounds=10,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled_inertial",
		payoff_lag=1,
		selection_strength=0.06,
		sampled_inertia=0.3,
		hybrid_update_share=0.2,
		out_csv=Path("outputs") / "_ignored_h51_hybrid_invalid.csv",
		memory_kernel=3,
	)
	with pytest.raises(ValueError):
		simulate(cfg)


def test_hetero_equal_strengths_match_sampled_rows():
	base_cfg = SimConfig(
		n_players=30,
		n_rounds=40,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_sampled.csv",
		memory_kernel=3,
	)
	hetero_cfg = SimConfig(
		n_players=30,
		n_rounds=40,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="hetero",
		payoff_lag=1,
		selection_strength=0.06,
		strategy_selection_strengths=(0.06, 0.06, 0.06),
		out_csv=Path("outputs") / "_ignored_hetero.csv",
		memory_kernel=3,
	)

	strategy_space, base_rows = simulate(base_cfg)
	strategy_space_hetero, hetero_rows = simulate(hetero_cfg)

	assert strategy_space_hetero == strategy_space
	assert len(base_rows) == len(hetero_rows)
	for base_row, hetero_row in zip(base_rows, hetero_rows, strict=True):
		for key in ["round", *[f"p_{s}" for s in strategy_space], *[f"w_{s}" for s in strategy_space]]:
			assert hetero_row[key] == pytest.approx(base_row[key], abs=0.0)


def test_h31_zero_fixed_share_matches_hetero_rows():
	base_cfg = SimConfig(
		n_players=30,
		n_rounds=30,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="hetero",
		payoff_lag=1,
		selection_strength=0.06,
		strategy_selection_strengths=(0.07, 0.06, 0.04),
		out_csv=Path("outputs") / "_ignored_h31_base.csv",
		memory_kernel=3,
	)
	h31_cfg = SimConfig(
		**{**base_cfg.__dict__, "fixed_subgroup_share": 0.0, "fixed_subgroup_weights": (0.8, 0.8, 1.4)}
	)

	strategy_space, base_rows = simulate(base_cfg)
	strategy_space_h31, h31_rows = simulate(h31_cfg)

	assert strategy_space_h31 == strategy_space
	assert len(base_rows) == len(h31_rows)
	for base_row, h31_row in zip(base_rows, h31_rows, strict=True):
		for key in ["round", *[f"p_{s}" for s in strategy_space], *[f"w_{s}" for s in strategy_space]]:
			assert h31_row[key] == pytest.approx(base_row[key], abs=0.0)


def test_h31_requires_fixed_weights_when_share_positive():
	cfg = SimConfig(
		n_players=30,
		n_rounds=10,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="hetero",
		payoff_lag=1,
		selection_strength=0.06,
		strategy_selection_strengths=(0.07, 0.06, 0.04),
		fixed_subgroup_share=0.2,
		fixed_subgroup_weights=None,
		out_csv=Path("outputs") / "_ignored_h31_missing_weights.csv",
		memory_kernel=3,
	)
	with pytest.raises(ValueError):
		simulate(cfg)


def test_h31_all_fixed_players_keep_fixed_weights():
	cfg = SimConfig(
		n_players=20,
		n_rounds=12,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="hetero",
		payoff_lag=1,
		selection_strength=0.06,
		strategy_selection_strengths=(0.07, 0.06, 0.04),
		fixed_subgroup_share=1.0,
		fixed_subgroup_weights=(0.8, 0.8, 1.4),
		out_csv=Path("outputs") / "_ignored_h31_all_fixed.csv",
		memory_kernel=3,
	)
	strategy_space, rows = simulate(cfg)
	fixed = {"aggressive": 0.8, "defensive": 0.8, "balanced": 1.4}
	assert strategy_space == ["aggressive", "defensive", "balanced"]
	for row in rows:
		for s in strategy_space:
			assert row[f"w_{s}"] == pytest.approx(fixed[s], abs=1e-12)


def test_h34_full_anchor_pull_matches_h31_rows():
	base_cfg = SimConfig(
		n_players=30,
		n_rounds=20,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="hetero",
		payoff_lag=1,
		selection_strength=0.06,
		strategy_selection_strengths=(0.07, 0.06, 0.04),
		fixed_subgroup_share=0.2,
		fixed_subgroup_weights=(0.8, 0.8, 1.4),
		out_csv=Path("outputs") / "_ignored_h34_full_pull.csv",
		memory_kernel=3,
	)
	h34_cfg = SimConfig(**{**base_cfg.__dict__, "fixed_subgroup_anchor_pull_strength": 1.0})

	strategy_space, base_rows = simulate(base_cfg)
	strategy_space_h34, h34_rows = simulate(h34_cfg)

	assert strategy_space_h34 == strategy_space
	assert len(base_rows) == len(h34_rows)
	for base_row, h34_row in zip(base_rows, h34_rows, strict=True):
		for key in ["round", *[f"p_{s}" for s in strategy_space], *[f"w_{s}" for s in strategy_space]]:
			assert h34_row[key] == pytest.approx(base_row[key], abs=0.0)


def test_h34_zero_anchor_pull_matches_base_sampled_rows():
	base_cfg = SimConfig(
		n_players=30,
		n_rounds=20,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="hetero",
		payoff_lag=1,
		selection_strength=0.06,
		strategy_selection_strengths=(0.07, 0.06, 0.04),
		out_csv=Path("outputs") / "_ignored_h34_base.csv",
		memory_kernel=3,
	)
	h34_cfg = SimConfig(
		**{
			**base_cfg.__dict__,
			"fixed_subgroup_share": 0.2,
			"fixed_subgroup_weights": (0.8, 0.8, 1.4),
			"fixed_subgroup_anchor_pull_strength": 0.0,
			"out_csv": Path("outputs") / "_ignored_h34_zero_pull.csv",
		}
	)

	strategy_space, base_rows = simulate(base_cfg)
	strategy_space_h34, h34_rows = simulate(h34_cfg)

	assert strategy_space_h34 == strategy_space
	assert len(base_rows) == len(h34_rows)
	for base_row, h34_row in zip(base_rows, h34_rows, strict=True):
		for key in ["round", *[f"p_{s}" for s in strategy_space], *[f"w_{s}" for s in strategy_space]]:
			assert h34_row[key] == pytest.approx(base_row[key], abs=0.0)


def test_h34_partial_anchor_pull_changes_sampled_path():
	base_cfg = SimConfig(
		n_players=40,
		n_rounds=30,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_h34_base_sampled.csv",
		memory_kernel=3,
	)
	h34_cfg = SimConfig(
		**{
			**base_cfg.__dict__,
			"fixed_subgroup_share": 0.2,
			"fixed_subgroup_weights": (0.8, 0.8, 1.4),
			"fixed_subgroup_anchor_pull_strength": 0.6,
			"out_csv": Path("outputs") / "_ignored_h34_partial_pull.csv",
		}
	)

	strategy_space, base_rows = simulate(base_cfg)
	strategy_space_h34, h34_rows = simulate(h34_cfg)

	assert strategy_space_h34 == strategy_space
	assert len(base_rows) == len(h34_rows)
	assert any(
		h34_row[f"w_{s}"] != pytest.approx(base_row[f"w_{s}"], abs=1e-12)
		for base_row, h34_row in zip(base_rows, h34_rows, strict=True)
		for s in strategy_space
	)


def test_h35_zero_bidirectional_coupling_matches_h34_rows():
	base_cfg = SimConfig(
		n_players=40,
		n_rounds=30,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		fixed_subgroup_share=0.2,
		fixed_subgroup_weights=(0.8, 0.8, 1.4),
		fixed_subgroup_anchor_pull_strength=0.6,
		fixed_subgroup_bidirectional_coupling_strength=0.0,
		out_csv=Path("outputs") / "_ignored_h35_zero.csv",
		memory_kernel=3,
	)
	h35_cfg = SimConfig(**{**base_cfg.__dict__})

	strategy_space, base_rows = simulate(base_cfg)
	strategy_space_h35, h35_rows = simulate(h35_cfg)

	assert strategy_space_h35 == strategy_space
	assert len(base_rows) == len(h35_rows)
	for base_row, h35_row in zip(base_rows, h35_rows, strict=True):
		for key in ["round", *[f"p_{s}" for s in strategy_space], *[f"w_{s}" for s in strategy_space]]:
			assert h35_row[key] == pytest.approx(base_row[key], abs=0.0)


def test_h35_positive_bidirectional_coupling_changes_sampled_path():
	base_cfg = SimConfig(
		n_players=40,
		n_rounds=30,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		fixed_subgroup_share=0.2,
		fixed_subgroup_weights=(0.8, 0.8, 1.4),
		fixed_subgroup_anchor_pull_strength=0.6,
		fixed_subgroup_bidirectional_coupling_strength=0.0,
		out_csv=Path("outputs") / "_ignored_h35_base.csv",
		memory_kernel=3,
	)
	h35_cfg = SimConfig(
		**{
			**base_cfg.__dict__,
			"fixed_subgroup_bidirectional_coupling_strength": 0.3,
			"out_csv": Path("outputs") / "_ignored_h35_bi.csv",
		}
	)

	strategy_space, base_rows = simulate(base_cfg)
	strategy_space_h35, h35_rows = simulate(h35_cfg)

	assert strategy_space_h35 == strategy_space
	assert len(base_rows) == len(h35_rows)
	assert any(
		h35_row[f"w_{s}"] != pytest.approx(base_row[f"w_{s}"], abs=1e-12)
		for base_row, h35_row in zip(base_rows, h35_rows, strict=True)
		for s in strategy_space
	)


def test_h35_requires_partial_anchor_pull():
	cfg = SimConfig(
		n_players=30,
		n_rounds=10,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		fixed_subgroup_share=0.2,
		fixed_subgroup_weights=(0.8, 0.8, 1.4),
		fixed_subgroup_anchor_pull_strength=1.0,
		fixed_subgroup_bidirectional_coupling_strength=0.3,
		out_csv=Path("outputs") / "_ignored_h35_requires_partial.csv",
		memory_kernel=3,
	)
	with pytest.raises(ValueError):
		simulate(cfg)


@pytest.mark.parametrize(
	("field", "value"),
	[
		("fixed_subgroup_coupling_strength", 0.2),
		("fixed_subgroup_state_coupling_strength", 0.3),
	],
)
def test_h35_is_mutually_exclusive_with_h32_and_h33b(field: str, value: float):
	cfg = SimConfig(
		n_players=30,
		n_rounds=10,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		fixed_subgroup_share=0.2,
		fixed_subgroup_weights=(0.8, 0.8, 1.4),
		fixed_subgroup_anchor_pull_strength=0.6,
		fixed_subgroup_bidirectional_coupling_strength=0.3,
		fixed_subgroup_coupling_strength=0.0,
		fixed_subgroup_state_coupling_strength=0.0,
		fixed_subgroup_state_coupling_beta=8.0,
		fixed_subgroup_state_coupling_theta=0.1,
		fixed_subgroup_state_signal="gap_norm",
		out_csv=Path("outputs") / "_ignored_h35_mutex.csv",
		memory_kernel=3,
	)
	cfg = SimConfig(**{**cfg.__dict__, field: value})
	with pytest.raises(ValueError):
		simulate(cfg)


def test_h34_requires_fixed_subgroup_when_anchor_pull_below_one():
	cfg = SimConfig(
		n_players=30,
		n_rounds=10,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		fixed_subgroup_share=0.0,
		fixed_subgroup_weights=None,
		fixed_subgroup_anchor_pull_strength=0.6,
		out_csv=Path("outputs") / "_ignored_h34_missing_anchor.csv",
		memory_kernel=3,
	)
	with pytest.raises(ValueError):
		simulate(cfg)


@pytest.mark.parametrize(
	("field", "value"),
	[
		("fixed_subgroup_coupling_strength", 0.2),
		("fixed_subgroup_state_coupling_strength", 0.3),
	],
)
def test_h34_partial_anchor_pull_is_mutually_exclusive_with_h32_and_h33b(field: str, value: float):
	cfg = SimConfig(
		n_players=30,
		n_rounds=10,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		fixed_subgroup_share=0.2,
		fixed_subgroup_weights=(0.8, 0.8, 1.4),
		fixed_subgroup_anchor_pull_strength=0.6,
		fixed_subgroup_coupling_strength=0.0,
		fixed_subgroup_state_coupling_strength=0.0,
		fixed_subgroup_state_coupling_beta=8.0,
		fixed_subgroup_state_coupling_theta=0.1,
		fixed_subgroup_state_signal="gap_norm",
		out_csv=Path("outputs") / "_ignored_h34_mutex.csv",
		memory_kernel=3,
	)
	cfg = SimConfig(**{**cfg.__dict__, field: value})
	with pytest.raises(ValueError):
		simulate(cfg)


def test_h32_zero_coupling_matches_h31_rows():
	base_cfg = SimConfig(
		n_players=30,
		n_rounds=20,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		fixed_subgroup_share=0.2,
		fixed_subgroup_weights=(0.8, 0.8, 1.4),
		fixed_subgroup_coupling_strength=0.0,
		out_csv=Path("outputs") / "_ignored_h32_zero.csv",
		memory_kernel=3,
	)
	h32_cfg = SimConfig(
		**{
			**base_cfg.__dict__,
			"fixed_subgroup_coupling_strength": 0.0,
		}
	)

	strategy_space, base_rows = simulate(base_cfg)
	strategy_space_h32, h32_rows = simulate(h32_cfg)

	assert strategy_space_h32 == strategy_space
	assert len(base_rows) == len(h32_rows)
	for base_row, h32_row in zip(base_rows, h32_rows, strict=True):
		for key in ["round", *[f"p_{s}" for s in strategy_space], *[f"w_{s}" for s in strategy_space]]:
			assert h32_row[key] == pytest.approx(base_row[key], abs=0.0)


def test_h32_requires_fixed_subgroup_when_coupling_positive():
	cfg = SimConfig(
		n_players=30,
		n_rounds=10,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		fixed_subgroup_share=0.0,
		fixed_subgroup_weights=None,
		fixed_subgroup_coupling_strength=0.3,
		out_csv=Path("outputs") / "_ignored_h32_missing_anchor.csv",
		memory_kernel=3,
	)
	with pytest.raises(ValueError):
		simulate(cfg)


def test_h32_positive_coupling_changes_sampled_path():
	base_cfg = SimConfig(
		n_players=40,
		n_rounds=30,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		fixed_subgroup_share=0.2,
		fixed_subgroup_weights=(0.8, 0.8, 1.4),
		fixed_subgroup_coupling_strength=0.0,
		out_csv=Path("outputs") / "_ignored_h32_base_sampled.csv",
		memory_kernel=3,
	)
	h32_cfg = SimConfig(
		**{
			**base_cfg.__dict__,
			"fixed_subgroup_coupling_strength": 0.4,
			"out_csv": Path("outputs") / "_ignored_h32_coupled_sampled.csv",
		}
	)

	strategy_space, base_rows = simulate(base_cfg)
	strategy_space_h32, h32_rows = simulate(h32_cfg)

	assert strategy_space_h32 == strategy_space
	assert len(base_rows) == len(h32_rows)
	assert any(
		h32_row[f"w_{s}"] != pytest.approx(base_row[f"w_{s}"], abs=1e-12)
		for base_row, h32_row in zip(base_rows, h32_rows, strict=True)
		for s in strategy_space
	)


def test_h33b_zero_coupling_matches_h31_rows():
	base_cfg = SimConfig(
		n_players=30,
		n_rounds=20,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		fixed_subgroup_share=0.2,
		fixed_subgroup_weights=(0.8, 0.8, 1.4),
		fixed_subgroup_state_coupling_strength=0.0,
		fixed_subgroup_state_coupling_beta=8.0,
		fixed_subgroup_state_coupling_theta=0.1,
		fixed_subgroup_state_signal="gap_norm",
		out_csv=Path("outputs") / "_ignored_h33b_zero.csv",
		memory_kernel=3,
	)
	h33b_cfg = SimConfig(**{**base_cfg.__dict__})

	strategy_space, base_rows = simulate(base_cfg)
	strategy_space_h33b, h33b_rows = simulate(h33b_cfg)

	assert strategy_space_h33b == strategy_space
	assert len(base_rows) == len(h33b_rows)
	for base_row, h33b_row in zip(base_rows, h33b_rows, strict=True):
		for key in ["round", *[f"p_{s}" for s in strategy_space], *[f"w_{s}" for s in strategy_space]]:
			assert h33b_row[key] == pytest.approx(base_row[key], abs=0.0)


def test_h33b_requires_fixed_subgroup_when_coupling_positive():
	cfg = SimConfig(
		n_players=30,
		n_rounds=10,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="expected",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		fixed_subgroup_share=0.0,
		fixed_subgroup_weights=None,
		fixed_subgroup_state_coupling_strength=0.3,
		fixed_subgroup_state_coupling_beta=8.0,
		fixed_subgroup_state_coupling_theta=0.1,
		fixed_subgroup_state_signal="gap_norm",
		out_csv=Path("outputs") / "_ignored_h33b_missing_anchor.csv",
		memory_kernel=3,
	)
	with pytest.raises(ValueError):
		simulate(cfg)


def test_h33b_positive_coupling_changes_sampled_path():
	base_cfg = SimConfig(
		n_players=40,
		n_rounds=30,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		fixed_subgroup_share=0.2,
		fixed_subgroup_weights=(0.8, 0.8, 1.4),
		fixed_subgroup_state_coupling_strength=0.0,
		fixed_subgroup_state_coupling_beta=8.0,
		fixed_subgroup_state_coupling_theta=0.05,
		fixed_subgroup_state_signal="gap_norm",
		out_csv=Path("outputs") / "_ignored_h33b_base_sampled.csv",
		memory_kernel=3,
	)
	h33b_cfg = SimConfig(
		**{
			**base_cfg.__dict__,
			"fixed_subgroup_state_coupling_strength": 0.4,
			"out_csv": Path("outputs") / "_ignored_h33b_coupled_sampled.csv",
		}
	)

	strategy_space, base_rows = simulate(base_cfg)
	strategy_space_h33b, h33b_rows = simulate(h33b_cfg)

	assert strategy_space_h33b == strategy_space
	assert len(base_rows) == len(h33b_rows)
	assert any(
		h33b_row[f"w_{s}"] != pytest.approx(base_row[f"w_{s}"], abs=1e-12)
		for base_row, h33b_row in zip(base_rows, h33b_rows, strict=True)
		for s in strategy_space
	)


def test_h33b_and_h32_are_mutually_exclusive():
	cfg = SimConfig(
		n_players=30,
		n_rounds=10,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		fixed_subgroup_share=0.2,
		fixed_subgroup_weights=(0.8, 0.8, 1.4),
		fixed_subgroup_coupling_strength=0.2,
		fixed_subgroup_state_coupling_strength=0.3,
		fixed_subgroup_state_coupling_beta=8.0,
		fixed_subgroup_state_coupling_theta=0.1,
		fixed_subgroup_state_signal="gap_norm",
		out_csv=Path("outputs") / "_ignored_h33b_h32_mutex.csv",
		memory_kernel=3,
	)
	with pytest.raises(ValueError):
		simulate(cfg)
