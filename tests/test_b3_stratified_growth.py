from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from evolution.replicator_dynamics import sampled_growth_vector, stratified_growth_vector
from simulation.b3_stratified_growth import build_b3_player_setup_callback, build_b3_round_diagnostics, run_b3_scout
from simulation.run_simulation import SimConfig, simulate


def test_stratified_growth_n1_matches_sampled_growth() -> None:
	strategy_space = ["aggressive", "defensive", "balanced"]
	players = [
		SimpleNamespace(last_strategy="aggressive", last_reward=1.0, stratum=0),
		SimpleNamespace(last_strategy="aggressive", last_reward=0.5, stratum=0),
		SimpleNamespace(last_strategy="defensive", last_reward=0.2, stratum=0),
		SimpleNamespace(last_strategy="balanced", last_reward=-0.1, stratum=0),
	]

	baseline = sampled_growth_vector(players, strategy_space)
	stratified = stratified_growth_vector(players, strategy_space, n_strata=1)

	assert stratified == pytest.approx(baseline, abs=0.0)


def test_sampled_growth_n_strata_one_matches_sampled_rows() -> None:
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
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_b3_n1.csv",
		memory_kernel=3,
	)
	control_cfg = SimConfig(**common)
	stratified_cfg = SimConfig(sampled_growth_n_strata=1, **common)

	strategy_space, control_rows = simulate(control_cfg)
	strategy_space_stratified, stratified_rows = simulate(stratified_cfg)

	assert strategy_space_stratified == strategy_space
	assert len(control_rows) == len(stratified_rows)
	for control_row, stratified_row in zip(control_rows, stratified_rows, strict=True):
		for key in ["round", *[f"p_{s}" for s in strategy_space], *[f"w_{s}" for s in strategy_space]]:
			assert stratified_row[key] == pytest.approx(control_row[key], abs=0.0)


def test_sampled_growth_n_strata_three_changes_sampled_path() -> None:
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
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_b3_n3.csv",
		memory_kernel=3,
	)
	control_cfg = SimConfig(**common)
	active_cfg = SimConfig(sampled_growth_n_strata=3, **common)

	strategy_space, control_rows = simulate(control_cfg)
	_, active_rows = simulate(active_cfg)

	assert len(active_rows) == len(control_rows)
	assert any(
		abs(float(active_row[f"w_{strategy_space[0]}"]) - float(control_row[f"w_{strategy_space[0]}"])) > 1e-12
		for control_row, active_row in zip(control_rows, active_rows, strict=True)
	)


def test_personality_strata_n1_matches_sampled_rows() -> None:
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
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_b32_n1.csv",
		memory_kernel=3,
		sampled_growth_n_strata=1,
	)
	control_cfg = SimConfig(**common)
	callback = build_b3_player_setup_callback(
		strata_mode="personality",
		n_strata=1,
		personality_jitter=0.08,
		personality_seed=45,
	)

	strategy_space, control_rows = simulate(control_cfg)
	strategy_space_b32, personality_rows = simulate(control_cfg, player_setup_callback=callback)

	assert strategy_space_b32 == strategy_space
	assert len(control_rows) == len(personality_rows)
	for control_row, personality_row in zip(control_rows, personality_rows, strict=True):
		for key in ["round", *[f"p_{s}" for s in strategy_space], *[f"w_{s}" for s in strategy_space]]:
			assert personality_row[key] == pytest.approx(control_row[key], abs=0.0)


def test_personality_strata_n3_changes_sampled_path() -> None:
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
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_b32_n3.csv",
		memory_kernel=3,
		sampled_growth_n_strata=3,
	)
	control_cfg = SimConfig(**{**common, "sampled_growth_n_strata": 1})
	active_cfg = SimConfig(**common)
	callback = build_b3_player_setup_callback(
		strata_mode="personality",
		n_strata=3,
		personality_jitter=0.08,
		personality_seed=45,
	)

	strategy_space, control_rows = simulate(control_cfg)
	_, active_rows = simulate(active_cfg, player_setup_callback=callback)

	assert len(active_rows) == len(control_rows)
	assert any(
		abs(float(active_row[f"w_{strategy_space[0]}"]) - float(control_row[f"w_{strategy_space[0]}"])) > 1e-12
		for control_row, active_row in zip(control_rows, active_rows, strict=True)
	)


def test_phase_strata_setup_assigns_distinct_buckets() -> None:
	strategy_space = ["aggressive", "defensive", "balanced"]
	players = [
		SimpleNamespace(
			strategy_weights={"aggressive": 1.8, "defensive": 0.6, "balanced": 0.6},
			strategy_biases={strategy: 0.0 for strategy in strategy_space},
			stratum=0,
		),
		SimpleNamespace(
			strategy_weights={"aggressive": 0.6, "defensive": 1.8, "balanced": 0.6},
			strategy_biases={strategy: 0.0 for strategy in strategy_space},
			stratum=0,
		),
		SimpleNamespace(
			strategy_weights={"aggressive": 0.6, "defensive": 0.6, "balanced": 1.8},
			strategy_biases={strategy: 0.0 for strategy in strategy_space},
			stratum=0,
		),
	]
	callback = build_b3_player_setup_callback(
		strata_mode="phase",
		n_strata=3,
		personality_jitter=0.08,
		personality_seed=45,
	)

	callback(players, strategy_space, SimpleNamespace())

	assert len({int(getattr(player, "stratum", -1)) for player in players}) >= 2
	assert all(hasattr(player, "b3_phase_angle") for player in players)


def test_phase_rebucket_interval_must_be_positive() -> None:
	with pytest.raises(ValueError, match="phase_rebucket_interval"):
		build_b3_round_diagnostics(strata_mode="phase", phase_rebucket_interval=0)


def test_phase_strata_n1_matches_sampled_rows() -> None:
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
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_b33_n1.csv",
		memory_kernel=3,
		sampled_growth_n_strata=1,
	)
	control_cfg = SimConfig(**common)
	callback = build_b3_player_setup_callback(
		strata_mode="phase",
		n_strata=1,
		personality_jitter=0.08,
		personality_seed=45,
	)
	diagnostics = build_b3_round_diagnostics(strata_mode="phase", phase_rebucket_interval=1)

	strategy_space, control_rows = simulate(control_cfg)
	strategy_space_b33, phase_rows = simulate(control_cfg, player_setup_callback=callback, round_callback=diagnostics.callback)

	assert strategy_space_b33 == strategy_space
	assert len(control_rows) == len(phase_rows)
	for control_row, phase_row in zip(control_rows, phase_rows, strict=True):
		for key in ["round", *[f"p_{s}" for s in strategy_space], *[f"w_{s}" for s in strategy_space]]:
			assert phase_row[key] == pytest.approx(control_row[key], abs=0.0)


def test_phase_strata_n3_changes_sampled_path() -> None:
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
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		out_csv=Path("outputs") / "_ignored_b33_n3.csv",
		memory_kernel=3,
		sampled_growth_n_strata=3,
	)
	control_cfg = SimConfig(**{**common, "sampled_growth_n_strata": 1})
	active_cfg = SimConfig(**common)
	callback = build_b3_player_setup_callback(
		strata_mode="phase",
		n_strata=3,
		personality_jitter=0.08,
		personality_seed=45,
	)
	diagnostics = build_b3_round_diagnostics(strata_mode="phase", phase_rebucket_interval=1)

	strategy_space, control_rows = simulate(control_cfg)
	_, active_rows = simulate(active_cfg, player_setup_callback=callback, round_callback=diagnostics.callback)

	assert len(active_rows) == len(control_rows)
	assert any(
		abs(float(active_row[f"w_{strategy_space[0]}"]) - float(control_row[f"w_{strategy_space[0]}"])) > 1e-12
		for control_row, active_row in zip(control_rows, active_rows, strict=True)
	)


def test_phase_rebucket_updates_strata() -> None:
	active_cfg = SimConfig(
		n_players=30,
		n_rounds=40,
		seed=45,
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
		out_csv=Path("outputs") / "_ignored_b33_rebucket.csv",
		memory_kernel=3,
		sampled_growth_n_strata=3,
	)
	callback = build_b3_player_setup_callback(
		strata_mode="phase",
		n_strata=3,
		personality_jitter=0.08,
		personality_seed=45,
	)
	diagnostics = build_b3_round_diagnostics(strata_mode="phase", phase_rebucket_interval=1)

	simulate(active_cfg, player_setup_callback=callback, round_callback=diagnostics.callback)
	stats = diagnostics.finalize()

	assert float(stats["mean_phase_rebucket_churn"]) > 0.0
	assert float(stats["phase_occupancy_entropy"]) >= 0.0


def test_sampled_growth_n_strata_requires_sampled_mode() -> None:
	with pytest.raises(ValueError, match="sampled_growth_n_strata"):
		simulate(
			SimConfig(
				n_players=12,
				n_rounds=8,
				seed=45,
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
				sampled_growth_n_strata=3,
			),
		)


def test_b3_scout_writes_outputs(tmp_path: Path) -> None:
	out_root = tmp_path / "b3"
	summary_tsv = tmp_path / "b3_summary.tsv"
	combined_tsv = tmp_path / "b3_combined.tsv"
	decision_md = tmp_path / "b3_decision.md"

	result = run_b3_scout(
		seeds=[45, 47],
		n_strata_values=[1, 3],
		selection_strengths=[0.06],
		out_root=out_root,
		summary_tsv=summary_tsv,
		combined_tsv=combined_tsv,
		decision_md=decision_md,
		players=30,
		rounds=120,
		burn_in=30,
		tail=60,
		memory_kernel=3,
		enable_events=False,
	)

	assert Path(result["summary_tsv"]).exists()
	assert Path(result["combined_tsv"]).exists()
	assert Path(result["decision_md"]).exists()

	with combined_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 2
		assert {row["condition"] for row in rows} == {"strata1_k0p06", "strata3_k0p06"}
		assert any(row["is_control"] == "yes" for row in rows)
		assert any(row["verdict"] in {"pass", "weak_positive", "fail"} for row in rows if row["is_control"] == "no")
		for row in rows:
			assert Path(row["representative_simplex_png"]).exists()
			assert Path(row["representative_phase_amplitude_png"]).exists()
			assert row["mean_inter_strata_cosine"] != ""


def test_b32_scout_writes_outputs(tmp_path: Path) -> None:
	out_root = tmp_path / "b32"
	summary_tsv = tmp_path / "b32_summary.tsv"
	combined_tsv = tmp_path / "b32_combined.tsv"
	decision_md = tmp_path / "b32_decision.md"

	result = run_b3_scout(
		seeds=[45, 47],
		n_strata_values=[1, 3],
		selection_strengths=[0.06],
		strata_mode="personality",
		personality_jitter=0.08,
		out_root=out_root,
		summary_tsv=summary_tsv,
		combined_tsv=combined_tsv,
		decision_md=decision_md,
		players=30,
		rounds=120,
		burn_in=30,
		tail=60,
		memory_kernel=3,
		enable_events=False,
	)

	assert Path(result["summary_tsv"]).exists()
	assert Path(result["combined_tsv"]).exists()
	assert Path(result["decision_md"]).exists()

	with combined_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 2
		assert {row["condition"] for row in rows} == {"personality_strata1_k0p06", "personality_strata3_k0p06"}
		assert {row["strata_mode"] for row in rows} == {"personality"}
		for row in rows:
			assert Path(row["representative_simplex_png"]).exists()
			assert Path(row["representative_phase_amplitude_png"]).exists()
			assert float(row["personality_jitter"]) == pytest.approx(0.08)


def test_b33_scout_writes_outputs(tmp_path: Path) -> None:
	out_root = tmp_path / "b33"
	summary_tsv = tmp_path / "b33_summary.tsv"
	combined_tsv = tmp_path / "b33_combined.tsv"
	decision_md = tmp_path / "b33_decision.md"

	result = run_b3_scout(
		seeds=[45, 47],
		n_strata_values=[1, 3],
		selection_strengths=[0.06],
		strata_mode="phase",
		phase_rebucket_interval=1,
		out_root=out_root,
		summary_tsv=summary_tsv,
		combined_tsv=combined_tsv,
		decision_md=decision_md,
		players=30,
		rounds=120,
		burn_in=30,
		tail=60,
		memory_kernel=3,
		enable_events=False,
	)

	assert Path(result["summary_tsv"]).exists()
	assert Path(result["combined_tsv"]).exists()
	assert Path(result["decision_md"]).exists()

	with combined_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 2
		assert {row["condition"] for row in rows} == {"phase_strata1_k0p06", "phase_strata3_k0p06"}
		assert {row["strata_mode"] for row in rows} == {"phase"}
		for row in rows:
			assert Path(row["representative_simplex_png"]).exists()
			assert Path(row["representative_phase_amplitude_png"]).exists()
			assert int(row["phase_rebucket_interval"]) == 1
			assert row["mean_phase_rebucket_churn"] != ""
			assert row["mean_within_stratum_phase_spread"] != ""
			assert row["phase_occupancy_entropy"] != ""

	decision_text = decision_md.read_text(encoding="utf-8")
	assert "B5" in decision_text
	assert "best_active_seed" in decision_text
	assert "worst_active_seed" in decision_text