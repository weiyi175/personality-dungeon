from __future__ import annotations

import csv
from pathlib import Path

import pytest

from simulation.personality_h7 import _build_player_setup_callback, _sample_h71_personalities, main as h7_main
from simulation.run_simulation import SimConfig, simulate, simulate_series_window


def _h71_setup_callback(*, n_players: int, jitter: float, seed: int):
	personalities = _sample_h71_personalities(n_players=n_players, jitter=jitter, seed=seed)
	return _build_player_setup_callback(personalities=personalities)


def test_personality_coupled_noop_matches_sampled_rows() -> None:
	common = dict(
		n_players=30,
		n_rounds=80,
		seed=45,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_h71.csv",
		memory_kernel=3,
	)
	setup = _h71_setup_callback(n_players=30, jitter=0.08, seed=45)
	control_cfg = SimConfig(evolution_mode="sampled", **common)
	noop_cfg = SimConfig(
		evolution_mode="personality_coupled",
		personality_coupling_mu_base=0.0,
		personality_coupling_lambda_mu=0.0,
		personality_coupling_lambda_k=0.0,
		**common,
	)

	strategy_space, control_rows = simulate(control_cfg, player_setup_callback=setup)
	strategy_space_noop, noop_rows = simulate(noop_cfg, player_setup_callback=setup)

	assert strategy_space_noop == strategy_space
	assert len(noop_rows) == len(control_rows)
	for control_row, noop_row in zip(control_rows, noop_rows, strict=True):
		for key in ["round", *[f"p_{s}" for s in strategy_space], *[f"w_{s}" for s in strategy_space]]:
			assert noop_row[key] == pytest.approx(control_row[key], abs=0.0)


def test_personality_coupled_noop_series_window_matches_simulate_rows() -> None:
	setup = _h71_setup_callback(n_players=30, jitter=0.08, seed=47)
	cfg = SimConfig(
		n_players=30,
		n_rounds=90,
		seed=47,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		evolution_mode="personality_coupled",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_h71_window.csv",
		memory_kernel=3,
		personality_coupling_mu_base=0.0,
		personality_coupling_lambda_mu=0.0,
		personality_coupling_lambda_k=0.0,
	)
	series_window = simulate_series_window(cfg, series="w", burn_in=20, tail=25, player_setup_callback=setup)
	strategy_space, rows = simulate(cfg, player_setup_callback=setup)

	begin = max(20, cfg.n_rounds - 25)
	window_rows = rows[begin:]
	assert set(series_window.keys()) == set(strategy_space)
	for s in strategy_space:
		expected = [float(row[f"w_{s}"]) for row in window_rows]
		assert series_window[s] == pytest.approx(expected, abs=0.0)


def test_personality_coupled_beta_state_k_changes_sampled_path() -> None:
	common = dict(
		n_players=30,
		n_rounds=80,
		seed=45,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		init_bias=0.12,
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_b4.csv",
		memory_kernel=3,
	)
	setup = _h71_setup_callback(n_players=30, jitter=0.08, seed=45)
	control_cfg = SimConfig(evolution_mode="sampled", **common)
	active_cfg = SimConfig(
		evolution_mode="personality_coupled",
		personality_coupling_mu_base=0.0,
		personality_coupling_lambda_mu=0.0,
		personality_coupling_lambda_k=0.0,
		personality_coupling_beta_state_k=0.6,
		**common,
	)

	strategy_space, control_rows = simulate(control_cfg, player_setup_callback=setup)
	_, active_rows = simulate(active_cfg, player_setup_callback=setup)

	assert len(active_rows) == len(control_rows)
	assert any(
		abs(float(active_row[f"w_{strategy_space[0]}"]) - float(control_row[f"w_{strategy_space[0]}"])) > 1e-12
		for control_row, active_row in zip(control_rows, active_rows, strict=True)
	)


def test_h7_main_writes_outputs(tmp_path: Path, monkeypatch) -> None:
	out_root = tmp_path / "h71"
	summary_tsv = tmp_path / "h71_summary.tsv"
	combined_tsv = tmp_path / "h71_combined.tsv"
	decision_md = tmp_path / "h71_decision.md"

	monkeypatch.setattr(
		"sys.argv",
		[
			"personality_h7",
			"--players",
			"30",
			"--rounds",
			"120",
			"--seeds",
			"45,47",
			"--burn-in",
			"30",
			"--tail",
			"60",
			"--out-root",
			str(out_root),
			"--summary-tsv",
			str(summary_tsv),
			"--combined-tsv",
			str(combined_tsv),
			"--decision-md",
			str(decision_md),
		],
	)

	assert h7_main() is None
	with summary_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 8
		assert {row["condition"] for row in rows} == {"control", "inertia_only", "k_only", "combined_low"}
	with combined_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 4
		assert rows[0]["condition"] == "control"
		assert any(row["condition"] == "combined_low" for row in rows)
	assert decision_md.exists()