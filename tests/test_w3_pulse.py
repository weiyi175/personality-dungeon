from __future__ import annotations

import csv
from pathlib import Path

from core.stackelberg import dominant_strategy_from_shares, ema_step
from simulation.run_simulation import SimConfig
from simulation.w3_pulse import W3PulseAdapter, W3PulseCellConfig, main as w3_pulse_main, run_w3_pulse_cell, run_w3_pulse_scout


def test_ema_step_and_dominant_tie_break() -> None:
	assert ema_step(previous=None, current=0.30, alpha=0.15) == 0.30
	assert ema_step(previous=0.20, current=0.40, alpha=0.25) == 0.25
	assert dominant_strategy_from_shares(aggressive=0.40, defensive=0.40, balanced=0.20) == "aggressive"
	assert dominant_strategy_from_shares(aggressive=0.20, defensive=0.40, balanced=0.40) == "defensive"


def test_pulse_adapter_is_finite_and_refractory_blocks_retrigger(tmp_path: Path) -> None:
	baseline = __import__("simulation.w3_pulse", fromlist=["_default_baseline_commitment"])._default_baseline_commitment()
	pulse_commitment = __import__("core.stackelberg", fromlist=["StackelbergCommitment"]).StackelbergCommitment(
		condition="w3_pulse_crossguard",
		leader_action="cross_pulse",
		a=1.00,
		b=0.90,
		matrix_cross_coupling=0.35,
	)
	config = W3PulseCellConfig(
		condition="w3_pulse_crossguard",
		baseline_commitment=baseline,
		pulse_commitment=pulse_commitment,
		players=9,
		rounds=6,
		events_json=Path("docs/personality_dungeon_v1/02_event_templates_smoke_v1.json"),
		selection_strength=0.06,
		init_bias=0.12,
		memory_kernel=3,
		burn_in=0,
		tail=6,
		ema_alpha=1.0,
		pulse_horizon=2,
		refractory_rounds=2,
		out_dir=tmp_path / "w3_pulse_crossguard",
	)
	adapter = W3PulseAdapter(config=config, seed=45)

	class _Dungeon:
		a = 1.0
		b = 0.9
		matrix_cross_coupling = 0.2

	dungeon = _Dungeon()
	cfg = SimConfig(
		n_players=9,
		n_rounds=6,
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
	)
	rows = [
		{"p_aggressive": 0.60, "p_defensive": 0.20, "p_balanced": 0.20},
		{"p_aggressive": 0.20, "p_defensive": 0.60, "p_balanced": 0.20},
		{"p_aggressive": 0.20, "p_defensive": 0.20, "p_balanced": 0.60},
		{"p_aggressive": 0.60, "p_defensive": 0.20, "p_balanced": 0.20},
		{"p_aggressive": 0.20, "p_defensive": 0.60, "p_balanced": 0.20},
		{"p_aggressive": 0.60, "p_defensive": 0.20, "p_balanced": 0.20},
	]
	for round_index, row in enumerate(rows):
		adapter(round_index, cfg, [], dungeon, [], row)

	assert adapter.pulse_count == 2
	assert adapter.step_rows[1]["pulse_started"] == "yes"
	assert adapter.step_rows[2]["pulse_started"] == "no"
	assert adapter.step_rows[2]["pulse_active"] == "yes"
	assert adapter.step_rows[3]["pulse_active"] == "no"
	assert int(adapter.step_rows[3]["refractory_rounds_left"]) > 0
	assert adapter.step_rows[5]["pulse_started"] == "yes"


def test_run_w3_pulse_scout_close_precedence_over_weak_positive(tmp_path: Path) -> None:
	def _fake_runner(config: W3PulseCellConfig, seed: int) -> dict[str, object]:
		stage3 = 0.20
		gamma = -0.020
		pulse_count = 0
		pulse_share = 0.0
		first_pulse_round = ""
		transitions = 1
		if config.condition == "w3_pulse_crossguard":
			stage3 = 0.22
			gamma = -0.010
			pulse_count = 2
			pulse_share = 0.15
			first_pulse_round = 120
		elif config.condition == "w3_pulse_commitpush":
			stage3 = 0.25
			gamma = -0.005
			pulse_count = 3
			pulse_share = 0.20
			first_pulse_round = 90
		return {
			"protocol": "w3_pulse",
			"condition": config.condition,
			"leader_action": config.pulse_commitment.leader_action if not config.is_control else config.baseline_commitment.leader_action,
			"a": "1.050000",
			"b": "0.850000",
			"matrix_cross_coupling": "0.350000",
			"seed": int(seed),
			"cycle_level": 2,
			"mean_stage3_score": f"{stage3:.6f}",
			"mean_turn_strength": "0.050000",
			"mean_env_gamma": f"{gamma:.6f}",
			"env_gamma_r2": "0.900000",
			"env_gamma_n_peaks": 3,
			"mean_success_rate": "0.500000",
			"level3_seed_count": 0,
			"has_level3_seed": "no",
			"pulse_count": pulse_count,
			"pulse_active_share": f"{pulse_share:.6f}",
			"first_pulse_round": first_pulse_round,
			"dominant_transition_count": transitions,
			"pulse_steps_tsv": str(config.pulse_steps_path(seed)),
			"out_csv": str(config.out_csv(seed)),
			"provenance_json": str(config.provenance_path(seed)),
		}

	result = run_w3_pulse_scout(
		seeds=[45, 47],
		out_root=tmp_path / "w3_pulse",
		summary_tsv=tmp_path / "w3_pulse_summary.tsv",
		combined_tsv=tmp_path / "w3_pulse_combined.tsv",
		decision_md=tmp_path / "w3_pulse_decision.md",
		cell_runner=_fake_runner,
	)

	assert result["decision"] == "close_w3_3"
	with (tmp_path / "w3_pulse_combined.tsv").open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		best_row = next(row for row in rows if row["condition"] == "w3_pulse_commitpush")
		assert best_row["is_best_policy"] == "yes"
		assert best_row["verdict"] == "weak_positive"


def test_run_w3_pulse_scout_pass_and_weak_positive(tmp_path: Path) -> None:
	def _fake_runner(config: W3PulseCellConfig, seed: int) -> dict[str, object]:
		cycle_level = 2
		stage3 = 0.20
		gamma = -0.020
		level3_seed_count = 0
		if config.condition == "w3_pulse_crossguard":
			cycle_level = 3
			stage3 = 0.31
			gamma = 0.010
			level3_seed_count = 1
		return {
			"protocol": "w3_pulse",
			"condition": config.condition,
			"leader_action": config.pulse_commitment.leader_action if not config.is_control else config.baseline_commitment.leader_action,
			"a": "1.050000",
			"b": "0.850000",
			"matrix_cross_coupling": "0.350000",
			"seed": int(seed),
			"cycle_level": cycle_level,
			"mean_stage3_score": f"{stage3:.6f}",
			"mean_turn_strength": "0.050000",
			"mean_env_gamma": f"{gamma:.6f}",
			"env_gamma_r2": "0.900000",
			"env_gamma_n_peaks": 3,
			"mean_success_rate": "0.500000",
			"level3_seed_count": level3_seed_count,
			"has_level3_seed": "yes" if level3_seed_count else "no",
			"pulse_count": 2 if not config.is_control else 0,
			"pulse_active_share": "0.200000" if not config.is_control else "0.000000",
			"first_pulse_round": 80 if not config.is_control else "",
			"dominant_transition_count": 3,
			"pulse_steps_tsv": str(config.pulse_steps_path(seed)),
			"out_csv": str(config.out_csv(seed)),
			"provenance_json": str(config.provenance_path(seed)),
		}

	result = run_w3_pulse_scout(
		seeds=[45, 47],
		out_root=tmp_path / "w3_pulse",
		summary_tsv=tmp_path / "w3_pulse_summary.tsv",
		combined_tsv=tmp_path / "w3_pulse_combined.tsv",
		decision_md=tmp_path / "w3_pulse_decision.md",
		cell_runner=_fake_runner,
	)

	assert result["decision"] == "pass"


def test_run_w3_pulse_cell_smoke(tmp_path: Path) -> None:
	config = W3PulseCellConfig(
		condition="w3_pulse_crossguard",
		baseline_commitment=__import__("simulation.w3_pulse", fromlist=["_default_baseline_commitment"])._default_baseline_commitment(),
		pulse_commitment=__import__("core.stackelberg", fromlist=["StackelbergCommitment"]).StackelbergCommitment(
			condition="w3_pulse_crossguard",
			leader_action="cross_pulse",
			a=1.00,
			b=0.90,
			matrix_cross_coupling=0.35,
		),
		players=9,
		rounds=60,
		events_json=Path("docs/personality_dungeon_v1/02_event_templates_smoke_v1.json"),
		selection_strength=0.06,
		init_bias=0.12,
		memory_kernel=3,
		burn_in=0,
		tail=60,
		ema_alpha=0.15,
		pulse_horizon=10,
		refractory_rounds=20,
		out_dir=tmp_path / "w3_pulse_crossguard",
	)

	row = run_w3_pulse_cell(config, 45)
	assert Path(str(row["out_csv"])).exists()
	assert Path(str(row["provenance_json"])).exists()
	assert Path(str(row["pulse_steps_tsv"])).exists()


def test_w3_pulse_main_writes_outputs(tmp_path: Path, monkeypatch) -> None:
	out_root = tmp_path / "w3_pulse"
	summary_tsv = tmp_path / "w3_pulse_summary.tsv"
	combined_tsv = tmp_path / "w3_pulse_combined.tsv"
	decision_md = tmp_path / "w3_pulse_decision.md"

	monkeypatch.setattr(
		"sys.argv",
		[
			"w3_pulse",
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
			"--pulse-horizon",
			"20",
			"--refractory-rounds",
			"30",
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

	assert w3_pulse_main() == 0
	with summary_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 8
	with combined_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 4
		assert rows[0]["condition"] == "control_pulse_policy"
	assert decision_md.exists()