import csv

import pytest

from simulation import seed_stability
from simulation.seed_stability import _parse_float_grid, _parse_seeds, _run_one


def test_parse_seeds_range_inclusive():
	assert _parse_seeds("0:2") == [0, 1, 2]
	assert _parse_seeds("2:0:-1") == [2, 1, 0]


def test_parse_float_grid_list_and_range():
	assert _parse_float_grid("0.5,1.0") == [0.5, 1.0]
	vals = _parse_float_grid("0.5:1.0:0.25")
	assert vals == pytest.approx([0.5, 0.75, 1.0])


def test_parse_float_grid_default_step():
	vals = _parse_float_grid("0.0:0.2")
	# default step=0.1 => 0.0,0.1,0.2
	assert vals == pytest.approx([0.0, 0.1, 0.2])


def test_run_one_threshold_ab_noop_matches_matrix_ab_mean_field():
	metric_kwargs = {
		"amplitude_threshold": 0.0,
		"min_lag": 2,
		"max_lag": 8,
		"corr_threshold": 0.0,
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
		"popularity_mode": "expected",
		"evolution_mode": "mean_field",
		"payoff_lag": 1,
		"memory_kernel": 3,
		"gamma": 0.1,
		"epsilon": 0.0,
		"a": 1.0,
		"b": 0.9,
		"matrix_cross_coupling": 0.2,
		"init_bias": 0.12,
		"key": None,
	}
	base = _run_one(
		players=40,
		rounds=24,
		seed=7,
		payload={"payoff_mode": "matrix_ab", **common},
		series="w",
		burn_in=0,
		tail=None,
		selection_strength=0.06,
		metric_kwargs=metric_kwargs,
	)
	noop = _run_one(
		players=40,
		rounds=24,
		seed=7,
		payload={
			"payoff_mode": "threshold_ab",
			"threshold_theta": 0.40,
			"threshold_a_hi": 1.0,
			"threshold_b_hi": 0.9,
			**common,
		},
		series="w",
		burn_in=0,
		tail=None,
		selection_strength=0.06,
		metric_kwargs=metric_kwargs,
	)

	assert noop[0] == base[0]
	assert noop[1:5] == base[1:5]
	assert noop[5:] == pytest.approx(base[5:])


def test_run_one_hetero_equal_strengths_matches_sampled_path():
	metric_kwargs = {
		"amplitude_threshold": 0.0,
		"min_lag": 2,
		"max_lag": 8,
		"corr_threshold": 0.0,
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
		"payoff_mode": "matrix_ab",
		"popularity_mode": "expected",
		"evolution_mode": "sampled",
		"payoff_lag": 1,
		"memory_kernel": 3,
		"gamma": 0.1,
		"epsilon": 0.0,
		"a": 1.0,
		"b": 0.9,
		"matrix_cross_coupling": 0.2,
		"init_bias": 0.12,
		"key": None,
	}
	base = _run_one(
		players=40,
		rounds=24,
		seed=7,
		payload=common,
		series="w",
		burn_in=0,
		tail=None,
		selection_strength=0.06,
		metric_kwargs=metric_kwargs,
	)
	hetero = _run_one(
		players=40,
		rounds=24,
		seed=7,
		payload={
			**common,
			"evolution_mode": "hetero",
			"strategy_selection_strengths": "0.06,0.06,0.06",
		},
		series="w",
		burn_in=0,
		tail=None,
		selection_strength=0.06,
		metric_kwargs=metric_kwargs,
	)

	assert hetero[0] == base[0]
	assert hetero[1:5] == base[1:5]
	assert hetero[5:] == pytest.approx(base[5:])


def test_run_one_h4_zero_hybrid_share_matches_sampled_path():
	metric_kwargs = {
		"amplitude_threshold": 0.0,
		"min_lag": 2,
		"max_lag": 8,
		"corr_threshold": 0.0,
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
		"payoff_mode": "matrix_ab",
		"popularity_mode": "sampled",
		"evolution_mode": "sampled",
		"payoff_lag": 1,
		"memory_kernel": 3,
		"gamma": 0.1,
		"epsilon": 0.0,
		"a": 1.0,
		"b": 0.9,
		"matrix_cross_coupling": 0.2,
		"init_bias": 0.12,
		"key": None,
	}
	base = _run_one(
		players=40,
		rounds=24,
		seed=7,
		payload=common,
		series="w",
		burn_in=0,
		tail=None,
		selection_strength=0.06,
		metric_kwargs=metric_kwargs,
	)
	h4 = _run_one(
		players=40,
		rounds=24,
		seed=7,
		payload={
			**common,
			"evolution_mode": "hybrid",
			"hybrid_update_share": 0.0,
		},
		series="w",
		burn_in=0,
		tail=None,
		selection_strength=0.06,
		metric_kwargs=metric_kwargs,
	)

	assert h4[0] == base[0]
	assert h4[1:5] == base[1:5]
	assert h4[5:] == pytest.approx(base[5:])


def test_run_one_h41_zero_inertia_matches_h4_path():
	metric_kwargs = {
		"amplitude_threshold": 0.0,
		"min_lag": 2,
		"max_lag": 8,
		"corr_threshold": 0.0,
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
		"payoff_mode": "matrix_ab",
		"popularity_mode": "sampled",
		"evolution_mode": "hybrid",
		"payoff_lag": 1,
		"memory_kernel": 3,
		"gamma": 0.1,
		"epsilon": 0.0,
		"a": 1.0,
		"b": 0.9,
		"matrix_cross_coupling": 0.2,
		"init_bias": 0.12,
		"hybrid_update_share": 0.2,
		"key": None,
	}
	h4 = _run_one(
		players=40,
		rounds=24,
		seed=7,
		payload=common,
		series="w",
		burn_in=0,
		tail=None,
		selection_strength=0.06,
		metric_kwargs=metric_kwargs,
	)
	h41 = _run_one(
		players=40,
		rounds=24,
		seed=7,
		payload={
			**common,
			"hybrid_inertia": 0.0,
		},
		series="w",
		burn_in=0,
		tail=None,
		selection_strength=0.06,
		metric_kwargs=metric_kwargs,
	)

	assert h41[0] == h4[0]
	assert h41[1:5] == h4[1:5]
	assert h41[5:] == pytest.approx(h4[5:])


def test_run_one_h51_zero_sampled_inertia_matches_sampled_path():
	metric_kwargs = {
		"amplitude_threshold": 0.0,
		"min_lag": 2,
		"max_lag": 8,
		"corr_threshold": 0.0,
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
		"payoff_mode": "matrix_ab",
		"popularity_mode": "sampled",
		"evolution_mode": "sampled",
		"payoff_lag": 1,
		"memory_kernel": 3,
		"gamma": 0.1,
		"epsilon": 0.0,
		"a": 1.0,
		"b": 0.9,
		"matrix_cross_coupling": 0.2,
		"init_bias": 0.12,
		"key": None,
	}
	base = _run_one(
		players=40,
		rounds=24,
		seed=7,
		payload=common,
		series="w",
		burn_in=0,
		tail=None,
		selection_strength=0.06,
		metric_kwargs=metric_kwargs,
	)
	h51 = _run_one(
		players=40,
		rounds=24,
		seed=7,
		payload={
			**common,
			"evolution_mode": "sampled_inertial",
			"sampled_inertia": 0.0,
		},
		series="w",
		burn_in=0,
		tail=None,
		selection_strength=0.06,
		metric_kwargs=metric_kwargs,
	)

	assert h51[0] == base[0]
	assert h51[1:5] == base[1:5]
	assert h51[5:] == pytest.approx(base[5:])


def test_run_one_mean_field_zero_sampled_inertia_matches_mean_field_path():
	metric_kwargs = {
		"amplitude_threshold": 0.0,
		"min_lag": 2,
		"max_lag": 8,
		"corr_threshold": 0.0,
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
		"payoff_mode": "matrix_ab",
		"popularity_mode": "expected",
		"evolution_mode": "mean_field",
		"payoff_lag": 1,
		"memory_kernel": 3,
		"gamma": 0.1,
		"epsilon": 0.0,
		"a": 1.0,
		"b": 0.9,
		"matrix_cross_coupling": 0.2,
		"init_bias": 0.12,
		"key": None,
	}
	base = _run_one(
		players=40,
		rounds=24,
		seed=7,
		payload=common,
		series="w",
		burn_in=0,
		tail=None,
		selection_strength=0.06,
		metric_kwargs=metric_kwargs,
	)
	gate = _run_one(
		players=40,
		rounds=24,
		seed=7,
		payload={
			**common,
			"sampled_inertia": 0.0,
		},
		series="w",
		burn_in=0,
		tail=None,
		selection_strength=0.06,
		metric_kwargs=metric_kwargs,
	)

	assert gate[0] == base[0]
	assert gate[1:5] == base[1:5]
	assert gate[5:] == pytest.approx(base[5:])


def test_main_single_report_includes_h41_provenance(tmp_path, monkeypatch, capsys):
	out_path = tmp_path / "h41_report.csv"
	monkeypatch.setattr(
		"sys.argv",
		[
			"seed_stability",
			"--payoff-mode",
			"matrix_ab",
			"--a",
			"1.0",
			"--b",
			"0.9",
			"--matrix-cross-coupling",
			"0.2",
			"--players",
			"30",
			"--rounds",
			"18",
			"--seeds",
			"7:7",
			"--series",
			"p",
			"--burn-in",
			"0",
			"--tail",
			"12",
			"--selection-strength",
			"0.06",
			"--popularity-mode",
			"sampled",
			"--evolution-mode",
			"hybrid",
			"--hybrid-update-share",
			"0.2",
			"--hybrid-inertia",
			"0.3",
			"--memory-kernel",
			"3",
			"--init-bias",
			"0.12",
			"--amplitude-threshold",
			"0.0",
			"--corr-threshold",
			"0.0",
			"--eta",
			"0.0",
			"--min-turn-strength",
			"0.0",
			"--min-lag",
			"2",
			"--max-lag",
			"6",
			"--jobs",
			"1",
			"--out",
			str(out_path),
		],
	)

	seed_stability.main()

	stdout = capsys.readouterr().out
	assert "hybrid_update_share=0.2" in stdout
	assert "hybrid_inertia=0.3" in stdout

	with out_path.open(newline="") as fh:
		row = next(csv.DictReader(fh))

	assert row["popularity_mode"] == "sampled"
	assert row["evolution_mode"] == "hybrid"
	assert row["hybrid_update_share"] == "0.2"
	assert row["hybrid_inertia"] == "0.3"


def test_main_single_report_includes_h51_provenance(tmp_path, monkeypatch, capsys):
	out_path = tmp_path / "h51_report.csv"
	monkeypatch.setattr(
		"sys.argv",
		[
			"seed_stability",
			"--payoff-mode",
			"matrix_ab",
			"--a",
			"1.0",
			"--b",
			"0.9",
			"--matrix-cross-coupling",
			"0.2",
			"--players",
			"30",
			"--rounds",
			"18",
			"--seeds",
			"7:7",
			"--series",
			"p",
			"--burn-in",
			"0",
			"--tail",
			"12",
			"--selection-strength",
			"0.06",
			"--popularity-mode",
			"sampled",
			"--evolution-mode",
			"sampled_inertial",
			"--sampled-inertia",
			"0.3",
			"--memory-kernel",
			"3",
			"--init-bias",
			"0.12",
			"--amplitude-threshold",
			"0.0",
			"--corr-threshold",
			"0.0",
			"--eta",
			"0.0",
			"--min-turn-strength",
			"0.0",
			"--min-lag",
			"2",
			"--max-lag",
			"6",
			"--jobs",
			"1",
			"--out",
			str(out_path),
		],
	)

	seed_stability.main()

	stdout = capsys.readouterr().out
	assert "sampled_inertia=0.3" in stdout

	with out_path.open(newline="") as fh:
		row = next(csv.DictReader(fh))

	assert row["popularity_mode"] == "sampled"
	assert row["evolution_mode"] == "sampled_inertial"
	assert row["sampled_inertia"] == "0.3"


def test_run_one_h32_zero_coupling_matches_h31_path():
	metric_kwargs = {
		"amplitude_threshold": 0.0,
		"min_lag": 2,
		"max_lag": 8,
		"corr_threshold": 0.0,
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
		"payoff_mode": "matrix_ab",
		"popularity_mode": "sampled",
		"evolution_mode": "sampled",
		"payoff_lag": 1,
		"memory_kernel": 3,
		"gamma": 0.1,
		"epsilon": 0.0,
		"a": 1.0,
		"b": 0.9,
		"matrix_cross_coupling": 0.2,
		"init_bias": 0.12,
		"fixed_subgroup_share": 0.2,
		"fixed_subgroup_weights": "0.8,0.8,1.4",
		"key": None,
	}
	base = _run_one(
		players=40,
		rounds=24,
		seed=7,
		payload=common,
		series="w",
		burn_in=0,
		tail=None,
		selection_strength=0.06,
		metric_kwargs=metric_kwargs,
	)
	h32 = _run_one(
		players=40,
		rounds=24,
		seed=7,
		payload={
			**common,
			"fixed_subgroup_coupling_strength": 0.0,
		},
		series="w",
		burn_in=0,
		tail=None,
		selection_strength=0.06,
		metric_kwargs=metric_kwargs,
	)

	assert h32[0] == base[0]
	assert h32[1:5] == base[1:5]
	assert h32[5:] == pytest.approx(base[5:])


def test_run_one_h34_zero_anchor_pull_matches_base_path():
	metric_kwargs = {
		"amplitude_threshold": 0.0,
		"min_lag": 2,
		"max_lag": 8,
		"corr_threshold": 0.0,
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
		"payoff_mode": "matrix_ab",
		"popularity_mode": "sampled",
		"evolution_mode": "sampled",
		"payoff_lag": 1,
		"memory_kernel": 3,
		"gamma": 0.1,
		"epsilon": 0.0,
		"a": 1.0,
		"b": 0.9,
		"matrix_cross_coupling": 0.2,
		"init_bias": 0.12,
		"key": None,
	}
	base = _run_one(
		players=40,
		rounds=24,
		seed=7,
		payload=common,
		series="w",
		burn_in=0,
		tail=None,
		selection_strength=0.06,
		metric_kwargs=metric_kwargs,
	)
	h34 = _run_one(
		players=40,
		rounds=24,
		seed=7,
		payload={
			**common,
			"fixed_subgroup_share": 0.2,
			"fixed_subgroup_weights": "0.8,0.8,1.4",
			"fixed_subgroup_anchor_pull_strength": 0.0,
		},
		series="w",
		burn_in=0,
		tail=None,
		selection_strength=0.06,
		metric_kwargs=metric_kwargs,
	)

	assert h34[0] == base[0]
	assert h34[1:5] == base[1:5]
	assert h34[5:] == pytest.approx(base[5:])


def test_run_one_h35_zero_bidirectional_coupling_matches_h34_path():
	metric_kwargs = {
		"amplitude_threshold": 0.0,
		"min_lag": 2,
		"max_lag": 8,
		"corr_threshold": 0.0,
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
		"payoff_mode": "matrix_ab",
		"popularity_mode": "sampled",
		"evolution_mode": "sampled",
		"payoff_lag": 1,
		"memory_kernel": 3,
		"gamma": 0.1,
		"epsilon": 0.0,
		"a": 1.0,
		"b": 0.9,
		"matrix_cross_coupling": 0.2,
		"init_bias": 0.12,
		"fixed_subgroup_share": 0.2,
		"fixed_subgroup_weights": "0.8,0.8,1.4",
		"fixed_subgroup_anchor_pull_strength": 0.6,
		"key": None,
	}
	base = _run_one(
		players=40,
		rounds=24,
		seed=7,
		payload=common,
		series="w",
		burn_in=0,
		tail=None,
		selection_strength=0.06,
		metric_kwargs=metric_kwargs,
	)
	h35 = _run_one(
		players=40,
		rounds=24,
		seed=7,
		payload={
			**common,
			"fixed_subgroup_bidirectional_coupling_strength": 0.0,
		},
		series="w",
		burn_in=0,
		tail=None,
		selection_strength=0.06,
		metric_kwargs=metric_kwargs,
	)

	assert h35[0] == base[0]
	assert h35[1:5] == base[1:5]
	assert h35[5:] == pytest.approx(base[5:])


def test_run_one_h33b_zero_coupling_matches_h31_path():
	metric_kwargs = {
		"amplitude_threshold": 0.0,
		"min_lag": 2,
		"max_lag": 8,
		"corr_threshold": 0.0,
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
		"payoff_mode": "matrix_ab",
		"popularity_mode": "sampled",
		"evolution_mode": "sampled",
		"payoff_lag": 1,
		"memory_kernel": 3,
		"gamma": 0.1,
		"epsilon": 0.0,
		"a": 1.0,
		"b": 0.9,
		"matrix_cross_coupling": 0.2,
		"init_bias": 0.12,
		"fixed_subgroup_share": 0.2,
		"fixed_subgroup_weights": "0.8,0.8,1.4",
		"key": None,
	}
	base = _run_one(
		players=40,
		rounds=24,
		seed=7,
		payload=common,
		series="w",
		burn_in=0,
		tail=None,
		selection_strength=0.06,
		metric_kwargs=metric_kwargs,
	)
	h33b = _run_one(
		players=40,
		rounds=24,
		seed=7,
		payload={
			**common,
			"fixed_subgroup_state_coupling_strength": 0.0,
			"fixed_subgroup_state_coupling_beta": 8.0,
			"fixed_subgroup_state_coupling_theta": 0.1,
			"fixed_subgroup_state_signal": "gap_norm",
		},
		series="w",
		burn_in=0,
		tail=None,
		selection_strength=0.06,
		metric_kwargs=metric_kwargs,
	)

	assert h33b[0] == base[0]
	assert h33b[1:5] == base[1:5]
	assert h33b[5:] == pytest.approx(base[5:])
