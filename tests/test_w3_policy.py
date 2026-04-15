from __future__ import annotations

import csv
from pathlib import Path

from core.stackelberg import dominance_gap_from_shares, hysteresis_gate
from simulation.w3_policy import W3PolicyCellConfig, main as w3_policy_main, run_w3_policy_cell, run_w3_policy_scout


def test_hysteresis_gate_and_dominance_gap() -> None:
	assert dominance_gap_from_shares(aggressive=0.50, defensive=0.30, balanced=0.20) == 0.30
	assert hysteresis_gate(value=0.13, current_active=False, theta_low=0.08, theta_high=0.12) is True
	assert hysteresis_gate(value=0.07, current_active=True, theta_low=0.08, theta_high=0.12) is False
	assert hysteresis_gate(value=0.10, current_active=True, theta_low=0.08, theta_high=0.12) is True


def test_run_w3_policy_scout_writes_outputs_and_close_decision(tmp_path: Path) -> None:
	def _fake_runner(config: W3PolicyCellConfig, seed: int) -> dict[str, object]:
		stage3 = 0.20
		gamma = -0.020
		activation = 0.0
		switches = 0
		gap = 0.05
		if config.condition == "w3_policy_crossguard":
			stage3 = 0.22
			gamma = -0.010
			activation = 0.40
			switches = 2
			gap = 0.11
		elif config.condition == "w3_policy_edgetilt":
			stage3 = 0.19
			gamma = -0.021
			activation = 0.30
			switches = 2
			gap = 0.10
		elif config.condition == "w3_policy_commitpush":
			stage3 = 0.25
			gamma = -0.005
			activation = 0.60
			switches = 3
			gap = 0.13
		return {
			"protocol": "w3_policy",
			"condition": config.condition,
			"leader_action": config.active_commitment.leader_action if not config.is_control else config.baseline_commitment.leader_action,
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
			"policy_activation_share": f"{activation:.6f}",
			"n_policy_switches": switches,
			"mean_dominance_gap": f"{gap:.6f}",
			"policy_steps_tsv": str(config.policy_steps_path(seed)),
			"out_csv": str(config.out_csv(seed)),
			"provenance_json": str(config.provenance_path(seed)),
		}

	result = run_w3_policy_scout(
		seeds=[45, 47],
		out_root=tmp_path / "w3_policy",
		summary_tsv=tmp_path / "w3_policy_summary.tsv",
		combined_tsv=tmp_path / "w3_policy_combined.tsv",
		decision_md=tmp_path / "w3_policy_decision.md",
		cell_runner=_fake_runner,
	)

	assert result["decision"] == "close_w3_2"
	with (tmp_path / "w3_policy_summary.tsv").open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 8
	with (tmp_path / "w3_policy_combined.tsv").open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 4
		best_row = next(row for row in rows if row["condition"] == "w3_policy_commitpush")
		assert best_row["is_best_policy"] == "yes"
		assert best_row["verdict"] == "weak_positive"
	assert (tmp_path / "w3_policy_decision.md").exists()


def test_run_w3_policy_scout_passes_on_level3_candidate(tmp_path: Path) -> None:
	def _fake_runner(config: W3PolicyCellConfig, seed: int) -> dict[str, object]:
		cycle_level = 2
		stage3 = 0.20
		gamma = -0.020
		level3_seed_count = 0
		activation = 0.0
		if config.condition == "w3_policy_crossguard":
			cycle_level = 3
			stage3 = 0.31
			gamma = 0.010
			level3_seed_count = 1
			activation = 0.50
		return {
			"protocol": "w3_policy",
			"condition": config.condition,
			"leader_action": config.active_commitment.leader_action if not config.is_control else config.baseline_commitment.leader_action,
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
			"policy_activation_share": f"{activation:.6f}",
			"n_policy_switches": 2,
			"mean_dominance_gap": "0.130000",
			"policy_steps_tsv": str(config.policy_steps_path(seed)),
			"out_csv": str(config.out_csv(seed)),
			"provenance_json": str(config.provenance_path(seed)),
		}

	result = run_w3_policy_scout(
		seeds=[45, 47],
		out_root=tmp_path / "w3_policy",
		summary_tsv=tmp_path / "w3_policy_summary.tsv",
		combined_tsv=tmp_path / "w3_policy_combined.tsv",
		decision_md=tmp_path / "w3_policy_decision.md",
		cell_runner=_fake_runner,
	)

	assert result["decision"] == "pass"


def test_run_w3_policy_cell_smoke(tmp_path: Path) -> None:
	config = W3PolicyCellConfig(
		condition="w3_policy_crossguard",
		baseline_commitment=__import__("simulation.w3_policy", fromlist=["_default_baseline_commitment"])._default_baseline_commitment(),
		active_commitment=__import__("core.stackelberg", fromlist=["StackelbergCommitment"]).StackelbergCommitment(
			condition="w3_policy_crossguard",
			leader_action="cross_guard_policy",
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
		policy_update_interval=20,
		theta_low=0.08,
		theta_high=0.12,
		out_dir=tmp_path / "w3_policy_crossguard",
	)

	row = run_w3_policy_cell(config, 45)
	assert Path(str(row["out_csv"])).exists()
	assert Path(str(row["provenance_json"])).exists()
	assert Path(str(row["policy_steps_tsv"])).exists()


def test_w3_policy_main_writes_outputs(tmp_path: Path, monkeypatch) -> None:
	out_root = tmp_path / "w3_policy"
	summary_tsv = tmp_path / "w3_policy_summary.tsv"
	combined_tsv = tmp_path / "w3_policy_combined.tsv"
	decision_md = tmp_path / "w3_policy_decision.md"

	monkeypatch.setattr(
		"sys.argv",
		[
			"w3_policy",
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
			"--policy-update-interval",
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

	assert w3_policy_main() == 0
	with summary_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 8
	with combined_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 4
		assert rows[0]["condition"] == "control_policy"
	assert decision_md.exists()