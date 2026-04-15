from __future__ import annotations

import csv
import json
from pathlib import Path

from simulation.world_runtime_w1 import INITIAL_WORLD_STATE, W1CellConfig
from simulation.world_state_w1 import main as w1_main, run_w1_scout


def _fake_cell_runner(config: W1CellConfig, seed: int) -> dict[str, object]:
	condition = str(config.condition)
	lambda_world = float(config.lambda_world)
	stage3_score = 0.50
	env_gamma = 0.01
	cycle_level = 2
	if condition == "w1_low":
		stage3_score = 0.51
		env_gamma = 0.015
	elif condition == "w1_base":
		stage3_score = 0.56
		env_gamma = 0.025
		cycle_level = 3 if int(seed) == 45 else 2
	elif condition == "w1_high":
		stage3_score = 0.48
		env_gamma = -0.01
	state = dict(INITIAL_WORLD_STATE)
	if condition != "control":
		state["threat"] = min(1.0, state["threat"] + lambda_world)
	identity_json = json.dumps({"Threat": 1.0}, sort_keys=True)
	non_identity_json = json.dumps({"Threat": 1.05}, sort_keys=True)
	return {
		"seed": int(seed),
		"cycle_level": int(cycle_level),
		"stage3_score": float(stage3_score),
		"turn_strength": 0.10,
		"env_gamma": float(env_gamma),
		"env_gamma_r2": 0.80,
		"env_gamma_n_peaks": 2,
		"success_rate": 0.45,
		"out_csv": str(config.out_csv_path(int(seed))),
		"provenance_json": str(config.provenance_path(int(seed))),
		"world_update_rows": [
			{
				"seed": int(seed),
				"window_index": 0,
				"round": 200,
				"window_start_round": 0,
				"window_end_round": 199,
				"scarcity": state["scarcity"],
				"threat": state["threat"],
				"noise": state["noise"],
				"intel": state["intel"],
				"dominant_event_type": "Threat",
				"a_new": float(config.a),
				"b_new": float(config.b),
				"event_distribution": json.dumps(
					{
						"Threat": 0.40,
						"Resource": 0.20,
						"Uncertainty": 0.15,
						"Navigation": 0.15,
						"Internal": 0.10,
					},
					sort_keys=True,
				),
				"p_aggressive": 0.34,
				"p_defensive": 0.33,
				"p_balanced": 0.33,
				"mean_reward_window": 0.28,
				"event_share_threat": 0.30,
				"event_share_resource": 0.20,
				"event_share_uncertainty": 0.20,
				"event_share_navigation": 0.15,
				"event_share_internal": 0.15,
				"state_deviated": condition != "control",
				"risk_multipliers_json": non_identity_json if condition != "control" else identity_json,
				"reward_multipliers_json": non_identity_json if condition != "control" else identity_json,
				"trait_multipliers_json": non_identity_json if condition != "control" else identity_json,
			}
		],
		"world_updates_tsv": str(config.world_updates_path(int(seed))),
	}


def test_run_w1_scout_writes_outputs(tmp_path: Path) -> None:
	out_root = tmp_path / "w1"
	summary_tsv = tmp_path / "w1_summary.tsv"
	combined_tsv = tmp_path / "w1_combined.tsv"
	decision_md = tmp_path / "w1_decision.md"
	updates_manifest_tsv = tmp_path / "w1_updates.tsv"

	result = run_w1_scout(
		seeds=[45, 47],
		out_root=out_root,
		summary_tsv=summary_tsv,
		combined_tsv=combined_tsv,
		decision_md=decision_md,
		updates_tsv=updates_manifest_tsv,
		events_json=Path("docs/personality_dungeon_v1/02_event_templates_smoke_v1.json"),
		players=30,
		rounds=120,
		world_update_interval=40,
		cell_runner=_fake_cell_runner,
	)

	assert result["decision"] == "pass"
	with summary_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 8
		assert {row["condition"] for row in rows} == {"control", "w1_low", "w1_base", "w1_high"}
	with combined_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 4
		control_row = next(row for row in rows if row["condition"] == "control")
		assert control_row["world_state_deviated"] == "no"
		base_row = next(row for row in rows if row["condition"] == "w1_base")
		assert base_row["verdict"] == "pass"
		assert base_row["world_state_deviated"] == "yes"
		assert base_row["gamma_improved_20pct"] == "yes"
		assert base_row["nonisomorphic_pass"] == "yes"
	with updates_manifest_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 8
		first_path = Path(rows[0]["world_updates_tsv"])
		assert first_path.name.startswith("w1_step_world_")
		assert first_path.exists()
	with first_path.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert rows[0]["condition"] in {"control", "w1_low", "w1_base", "w1_high"}
		assert "event_distribution" in rows[0]
	assert decision_md.exists()


def test_main_accepts_injected_runner(tmp_path: Path, monkeypatch) -> None:
	out_root = tmp_path / "w1_main"
	summary_tsv = tmp_path / "w1_main_summary.tsv"
	combined_tsv = tmp_path / "w1_main_combined.tsv"
	decision_md = tmp_path / "w1_main_decision.md"
	updates_manifest_tsv = tmp_path / "w1_main_updates.tsv"

	monkeypatch.setattr(
		"sys.argv",
		[
			"world_state_w1",
			"--players",
			"30",
			"--rounds",
			"120",
			"--seeds",
			"45,47",
			"--world-update-interval",
			"40",
			"--out-root",
			str(out_root),
			"--summary-tsv",
			str(summary_tsv),
			"--combined-tsv",
			str(combined_tsv),
			"--decision-md",
			str(decision_md),
			"--updates-tsv",
			str(updates_manifest_tsv),
		],
	)

	w1_main(cell_runner=_fake_cell_runner)
	assert summary_tsv.exists()
	assert combined_tsv.exists()
	assert decision_md.exists()
	assert updates_manifest_tsv.exists()


def test_timescale_protocol_supports_per_cell_intervals_and_close_w1(tmp_path: Path) -> None:
	out_root = tmp_path / "w1_timescale"
	summary_tsv = tmp_path / "w1_timescale_summary.tsv"
	combined_tsv = tmp_path / "w1_timescale_combined.tsv"
	decision_md = tmp_path / "w1_timescale_decision.md"
	updates_manifest_tsv = tmp_path / "w1_timescale_updates.tsv"
	seen_intervals: dict[str, int] = {}

	def _timescale_runner(config: W1CellConfig, seed: int) -> dict[str, object]:
		seen_intervals[str(config.condition)] = int(config.world_update_interval)
		condition = str(config.condition)
		stage3_score = 0.50
		env_gamma = -0.020
		cycle_level = 2
		if condition == "w1_fast":
			stage3_score = 0.490
			env_gamma = -0.008
		elif condition == "w1_mid_fast":
			stage3_score = 0.485
			env_gamma = -0.004
		elif condition == "w1_high_fast":
			stage3_score = 0.470
			env_gamma = 0.001
		state = dict(INITIAL_WORLD_STATE)
		if condition != "control":
			state["noise"] = min(1.0, 0.5 + float(config.lambda_world))
		identity_json = json.dumps({"Threat": 1.0}, sort_keys=True)
		non_identity_json = json.dumps({"Threat": 1.08}, sort_keys=True)
		return {
			"seed": int(seed),
			"cycle_level": int(cycle_level),
			"stage3_score": float(stage3_score),
			"turn_strength": 0.10,
			"env_gamma": float(env_gamma),
			"env_gamma_r2": 0.80,
			"env_gamma_n_peaks": 2,
			"success_rate": 0.45,
			"out_csv": str(config.out_csv_path(int(seed))),
			"provenance_json": str(config.provenance_path(int(seed))),
			"world_update_rows": [
				{
					"seed": int(seed),
					"window_index": 0,
					"round": int(config.world_update_interval),
					"window_start_round": 0,
					"window_end_round": int(config.world_update_interval) - 1,
					"scarcity": state["scarcity"],
					"threat": state["threat"],
					"noise": state["noise"],
					"intel": state["intel"],
					"dominant_event_type": "Uncertainty",
					"a_new": float(config.a),
					"b_new": float(config.b),
					"event_distribution": json.dumps({"Uncertainty": 0.50, "Threat": 0.50}, sort_keys=True),
					"p_aggressive": 0.34,
					"p_defensive": 0.33,
					"p_balanced": 0.33,
					"mean_reward_window": 0.28,
					"event_share_threat": 0.50,
					"event_share_resource": 0.0,
					"event_share_uncertainty": 0.50,
					"event_share_navigation": 0.0,
					"event_share_internal": 0.0,
					"state_deviated": condition != "control",
					"risk_multipliers_json": non_identity_json if condition != "control" else identity_json,
					"reward_multipliers_json": non_identity_json if condition != "control" else identity_json,
					"trait_multipliers_json": non_identity_json if condition != "control" else identity_json,
				}
			],
			"world_updates_tsv": str(config.world_updates_path(int(seed))),
		}

	result = run_w1_scout(
		protocol="w1_timescale",
		seeds=[45, 47],
		out_root=out_root,
		summary_tsv=summary_tsv,
		combined_tsv=combined_tsv,
		decision_md=decision_md,
		updates_tsv=updates_manifest_tsv,
		events_json=Path("docs/personality_dungeon_v1/02_event_templates_v1.json"),
		players=30,
		rounds=120,
		cell_runner=_timescale_runner,
	)

	assert result["decision"] == "close_w1"
	assert seen_intervals == {
		"control": 100,
		"w1_fast": 100,
		"w1_mid_fast": 150,
		"w1_high_fast": 100,
	}
	with combined_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert {row["condition"] for row in rows} == {"control", "w1_fast", "w1_mid_fast", "w1_high_fast"}
		high_row = next(row for row in rows if row["condition"] == "w1_high_fast")
		assert high_row["gamma_gate_threshold_pct"] == "30"
		assert high_row["stage3_not_below_control"] == "no"
		assert high_row["promotion_gate_pass"] == "no"
	assert "close_w1" in decision_md.read_text(encoding="utf-8")


def test_default_runtime_runner_writes_seed_world_tsvs(tmp_path: Path) -> None:
	out_root = tmp_path / "w1_runtime"
	summary_tsv = tmp_path / "w1_runtime_summary.tsv"
	combined_tsv = tmp_path / "w1_runtime_combined.tsv"
	decision_md = tmp_path / "w1_runtime_decision.md"
	updates_manifest_tsv = tmp_path / "w1_runtime_updates.tsv"

	run_w1_scout(
		protocol="w1_timescale",
		seeds=[45],
		out_root=out_root,
		summary_tsv=summary_tsv,
		combined_tsv=combined_tsv,
		decision_md=decision_md,
		updates_tsv=updates_manifest_tsv,
		events_json=Path("docs/personality_dungeon_v1/02_event_templates_v1.json"),
		players=30,
		rounds=300,
		selection_strength=0.06,
		init_bias=0.12,
		memory_kernel=3,
		burn_in=30,
		tail=60,
		a=1.0,
		b=0.9,
		cross=0.20,
	)

	with updates_manifest_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 4
		for row in rows:
			path = Path(row["world_updates_tsv"])
			assert path.exists()
			with path.open(newline="", encoding="utf-8") as world_handle:
				world_rows = list(csv.DictReader(world_handle, delimiter="\t"))
				assert len(world_rows) >= 2
				assert all(step["a_new"] == "1.000000" for step in world_rows)
				assert all(step["b_new"] == "0.900000" for step in world_rows)