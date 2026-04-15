from __future__ import annotations

import csv
from pathlib import Path

from core.stackelberg import leader_prefers, select_best_commitment
from simulation.w3_stackelberg import W3CellConfig, main as w3_main, run_w3_cell, run_w3_scout


def test_stackelberg_prefers_level3_before_aggregate_uplift() -> None:
	control = {
		"condition": "control",
		"level3_seed_count": 0,
		"mean_stage3_score": 0.40,
		"mean_env_gamma": 0.020,
		"mean_turn_strength": 0.10,
	}
	level3_candidate = {
		"condition": "w3_edge_tilt",
		"level3_seed_count": 1,
		"mean_stage3_score": 0.39,
		"mean_env_gamma": -0.010,
		"mean_turn_strength": 0.05,
	}
	aggregate_candidate = {
		"condition": "w3_commit_push",
		"level3_seed_count": 0,
		"mean_stage3_score": 0.55,
		"mean_env_gamma": 0.030,
		"mean_turn_strength": 0.20,
	}

	assert leader_prefers(level3_candidate, control) is True
	assert select_best_commitment(
		[
			{"condition": "control", "is_control": "yes", **control},
			{"condition": "w3_edge_tilt", "is_control": "no", **level3_candidate},
			{"condition": "w3_commit_push", "is_control": "no", **aggregate_candidate},
		]
	) == "w3_edge_tilt"


def test_run_w3_scout_writes_outputs_and_close_decision(tmp_path: Path) -> None:
	def _fake_runner(config: W3CellConfig, seed: int) -> dict[str, object]:
		stage3 = 0.20
		gamma = -0.020
		turn = 0.040
		if config.condition == "w3_cross_strong":
			stage3 = 0.22
			gamma = -0.010
			turn = 0.050
		elif config.condition == "w3_edge_tilt":
			stage3 = 0.24
			gamma = -0.005
			turn = 0.060
		elif config.condition == "w3_commit_push":
			stage3 = 0.27
			gamma = 0.001
			turn = 0.070
		return {
			"protocol": "w3_stackelberg",
			"condition": config.condition,
			"leader_action": config.commitment.leader_action,
			"a": f"{config.commitment.a:.6f}",
			"b": f"{config.commitment.b:.6f}",
			"matrix_cross_coupling": f"{config.commitment.matrix_cross_coupling:.6f}",
			"seed": int(seed),
			"cycle_level": 2,
			"mean_stage3_score": f"{stage3:.6f}",
			"mean_turn_strength": f"{turn:.6f}",
			"mean_env_gamma": f"{gamma:.6f}",
			"env_gamma_r2": "0.900000",
			"env_gamma_n_peaks": 3,
			"mean_success_rate": "0.500000",
			"level3_seed_count": 0,
			"has_level3_seed": "no",
			"out_csv": str(config.out_csv(seed)),
			"provenance_json": str(config.provenance_path(seed)),
		}

	result = run_w3_scout(
		seeds=[45, 47],
		out_root=tmp_path / "w3",
		summary_tsv=tmp_path / "w3_summary.tsv",
		combined_tsv=tmp_path / "w3_combined.tsv",
		decision_md=tmp_path / "w3_decision.md",
		cell_runner=_fake_runner,
	)

	assert result["decision"] == "close_w3_1"
	with (tmp_path / "w3_summary.tsv").open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 8
	with (tmp_path / "w3_combined.tsv").open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 4
		control_row = next(row for row in rows if row["condition"] == "control")
		assert control_row["verdict"] == "control"
		best_row = next(row for row in rows if row["condition"] == "w3_commit_push")
		assert best_row["is_best_commitment"] == "yes"
		assert best_row["verdict"] == "weak_positive"
	assert (tmp_path / "w3_decision.md").exists()


def test_run_w3_scout_passes_on_level3_candidate(tmp_path: Path) -> None:
	def _fake_runner(config: W3CellConfig, seed: int) -> dict[str, object]:
		cycle_level = 2
		stage3 = 0.20
		gamma = -0.020
		turn = 0.040
		level3_seed_count = 0
		if config.condition == "w3_edge_tilt":
			cycle_level = 3
			stage3 = 0.31
			gamma = 0.010
			turn = 0.080
			level3_seed_count = 1
		return {
			"protocol": "w3_stackelberg",
			"condition": config.condition,
			"leader_action": config.commitment.leader_action,
			"a": f"{config.commitment.a:.6f}",
			"b": f"{config.commitment.b:.6f}",
			"matrix_cross_coupling": f"{config.commitment.matrix_cross_coupling:.6f}",
			"seed": int(seed),
			"cycle_level": cycle_level,
			"mean_stage3_score": f"{stage3:.6f}",
			"mean_turn_strength": f"{turn:.6f}",
			"mean_env_gamma": f"{gamma:.6f}",
			"env_gamma_r2": "0.900000",
			"env_gamma_n_peaks": 3,
			"mean_success_rate": "0.500000",
			"level3_seed_count": level3_seed_count,
			"has_level3_seed": "yes" if level3_seed_count else "no",
			"out_csv": str(config.out_csv(seed)),
			"provenance_json": str(config.provenance_path(seed)),
		}

	result = run_w3_scout(
		seeds=[45, 47],
		out_root=tmp_path / "w3",
		summary_tsv=tmp_path / "w3_summary.tsv",
		combined_tsv=tmp_path / "w3_combined.tsv",
		decision_md=tmp_path / "w3_decision.md",
		cell_runner=_fake_runner,
	)

	assert result["decision"] == "pass"


def test_run_w3_cell_smoke(tmp_path: Path) -> None:
	config = W3CellConfig(
		commitment=next(commitment for commitment in __import__("simulation.w3_stackelberg", fromlist=["_default_commitments"])._default_commitments() if commitment.condition == "w3_cross_strong"),
		players=9,
		rounds=20,
		events_json=Path("docs/personality_dungeon_v1/02_event_templates_smoke_v1.json"),
		selection_strength=0.06,
		init_bias=0.12,
		memory_kernel=3,
		burn_in=0,
		tail=20,
		out_dir=tmp_path / "w3_cross_strong",
	)

	row = run_w3_cell(config, 45)
	assert Path(str(row["out_csv"])).exists()
	assert Path(str(row["provenance_json"])).exists()


def test_w3_main_writes_outputs(tmp_path: Path, monkeypatch) -> None:
	out_root = tmp_path / "w3"
	summary_tsv = tmp_path / "w3_summary.tsv"
	combined_tsv = tmp_path / "w3_combined.tsv"
	decision_md = tmp_path / "w3_decision.md"

	monkeypatch.setattr(
		"sys.argv",
		[
			"w3_stackelberg",
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

	assert w3_main() == 0
	with summary_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 8
	with combined_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 4
		assert rows[0]["condition"] == "control"
	assert decision_md.exists()