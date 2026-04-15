from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from simulation.personality_gate1 import gamma_improvement_flags
from simulation.world_runtime_w1 import INITIAL_WORLD_STATE, W1CellConfig, nonisomorphic_profile, run_w1_cell

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = "w1_1"
DEFAULT_EVENTS_JSON = REPO_ROOT / "docs" / "personality_dungeon_v1" / "02_event_templates_smoke_v1.json"
DEFAULT_TIMESCALE_EVENTS_JSON = REPO_ROOT / "docs" / "personality_dungeon_v1" / "02_event_templates_v1.json"
DEFAULT_PROTOCOL_PATHS = {
	"w1_1": {
		"events_json": DEFAULT_EVENTS_JSON,
		"out_root": REPO_ROOT / "outputs" / "w1_worldstate",
		"summary_tsv": REPO_ROOT / "outputs" / "w1_worldstate_summary.tsv",
		"combined_tsv": REPO_ROOT / "outputs" / "w1_worldstate_combined.tsv",
		"decision_md": REPO_ROOT / "outputs" / "w1_worldstate_decision.md",
		"updates_tsv": REPO_ROOT / "outputs" / "w1_worldstate_updates.tsv",
	},
	"w1_timescale": {
		"events_json": DEFAULT_TIMESCALE_EVENTS_JSON,
		"out_root": REPO_ROOT / "outputs" / "w1_worldstate_timescale",
		"summary_tsv": REPO_ROOT / "outputs" / "w1_worldstate_timescale_summary.tsv",
		"combined_tsv": REPO_ROOT / "outputs" / "w1_worldstate_timescale_combined.tsv",
		"decision_md": REPO_ROOT / "outputs" / "w1_worldstate_timescale_decision.md",
		"updates_tsv": REPO_ROOT / "outputs" / "w1_worldstate_timescale_updates.tsv",
	},
}
PROTOCOL_SPECS = {
	"w1_1": {
		"label": "W1.1",
		"gamma_gate_threshold_pct": 20,
		"require_stage3_not_below_control": False,
		"decision_without_pass": "weak_positive",
		"cells": [
			{"condition": "control", "lambda_world": 0.00, "world_update_interval": 200},
			{"condition": "w1_low", "lambda_world": 0.04, "world_update_interval": 200},
			{"condition": "w1_base", "lambda_world": 0.08, "world_update_interval": 200},
			{"condition": "w1_high", "lambda_world": 0.15, "world_update_interval": 200},
		],
		"stop_rule_lines": [
			"至少 1 個 non-control cell 要同時出現 >=1 個 Level 3 seed，且 mean_env_gamma 相對 control 達到 >=20% 改善，才算 W1.1 pass。",
			"如果只有 aggregate uplift 或 world-state 偏離證據，最多只能記為 weak_positive，不得直接跳到 W2 / W3。",
			"若結果只剩 time-varying event weights，而 risk/reward/trait multipliers 都維持 identity，必須標記為 nonisomorphic_fail，不得進 Gate 2。",
		],
	},
	"w1_timescale": {
		"label": "W1.2 Timescale",
		"gamma_gate_threshold_pct": 30,
		"require_stage3_not_below_control": True,
		"decision_without_pass": "close_w1",
		"cells": [
			{"condition": "control", "lambda_world": 0.00, "world_update_interval": 100},
			{"condition": "w1_fast", "lambda_world": 0.08, "world_update_interval": 100},
			{"condition": "w1_mid_fast", "lambda_world": 0.12, "world_update_interval": 150},
			{"condition": "w1_high_fast", "lambda_world": 0.15, "world_update_interval": 100},
		],
		"stop_rule_lines": [
			"Promotion Gate：至少 1 個 non-control cell 同時滿足 >=1 個 Level 3 seed、mean_env_gamma 比 control 改善 >=30%、且 mean_stage3_score 不低於 control。",
			"Closure Gate：若所有 adaptive cells 都沒有 Level 3 seed，或 mean_stage3_score 仍持續低於 control，整輪直接記為 close_w1。",
			"若 W1.2 仍然 close_w1，W1 主線正式結案，不再做 lambda_world、coupling gain、或更大 state 幅度 sweep。",
		],
	},
}

METRIC_FIELDNAMES = [
	"protocol",
	"condition",
	"world_update_lambda",
	"world_update_interval",
	"world_mode",
	"world_state_init_json",
	"seed",
	"cycle_level",
	"stage3_score",
	"turn_strength",
	"env_gamma",
	"env_gamma_r2",
	"env_gamma_n_peaks",
	"success_rate",
	"control_env_gamma",
	"control_stage3_score",
	"gamma_uplift_ratio_vs_control_seed",
	"stage3_uplift_vs_control_seed",
	"has_level3_seed",
	"out_csv",
	"provenance_json",
]

COMBINED_FIELDNAMES = [
	"protocol",
	"condition",
	"world_update_lambda",
	"world_update_interval",
	"world_mode",
	"world_state_init_json",
	"n_seeds",
	"mean_cycle_level",
	"mean_stage3_score",
	"mean_turn_strength",
	"mean_env_gamma",
	"mean_success_rate",
	"level_counts_json",
	"p_level_ge_2",
	"p_level_3",
	"level3_seed_count",
	"control_mean_env_gamma",
	"control_mean_stage3_score",
	"gamma_uplift_ratio_vs_control",
	"gamma_improve_pct",
	"stage3_uplift_vs_control",
	"gamma_improved_20pct",
	"gamma_gate_threshold_pct",
	"gamma_gate_pass",
	"stage3_not_below_control",
	"promotion_gate_pass",
	"mean_world_scarcity",
	"mean_world_threat",
	"mean_world_noise",
	"mean_world_intel",
	"world_state_deviated",
	"nonisomorphic_pass",
	"verdict",
	"events_json",
	"selection_strength",
	"memory_kernel",
	"out_dir",
]

WORLD_UPDATE_FIELDNAMES = [
	"condition",
	"world_update_lambda",
	"world_update_interval",
	"seed",
	"window_index",
	"round",
	"window_start_round",
	"window_end_round",
	"scarcity",
	"threat",
	"noise",
	"intel",
	"dominant_event_type",
	"a_new",
	"b_new",
	"event_distribution",
	"p_aggressive",
	"p_defensive",
	"p_balanced",
	"mean_reward_window",
	"event_share_threat",
	"event_share_resource",
	"event_share_uncertainty",
	"event_share_navigation",
	"event_share_internal",
	"state_deviated",
	"risk_multipliers_json",
	"reward_multipliers_json",
	"trait_multipliers_json",
]

UPDATES_MANIFEST_FIELDNAMES = ["protocol", "condition", "seed", "world_updates_tsv"]

CellRunner = Callable[[W1CellConfig, int], Mapping[str, Any]]


def _parse_seeds(spec: str) -> list[int]:
	parts = [part.strip() for part in str(spec).split(",") if part.strip()]
	if not parts:
		raise ValueError("--seeds cannot be empty")
	return [int(part) for part in parts]


def _safe_float(value: Any, default: float = 0.0) -> float:
	if value in (None, ""):
		return float(default)
	return float(value)


def _safe_int(value: Any, default: int = 0) -> int:
	if value in (None, ""):
		return int(default)
	return int(value)


def _safe_mean(values: list[float]) -> float:
	if not values:
		return 0.0
	return float(sum(values) / float(len(values)))


def _get_protocol_spec(protocol: str) -> dict[str, Any]:
	name = str(protocol)
	if name not in PROTOCOL_SPECS:
		raise ValueError(f"unsupported W1 protocol: {protocol}")
	return dict(PROTOCOL_SPECS[name])


def _resolve_protocol_path(protocol: str, key: str, explicit: Path | None) -> Path:
	if explicit is not None:
		return explicit
	return Path(DEFAULT_PROTOCOL_PATHS[str(protocol)][key])


def _format_float(value: float) -> str:
	return f"{float(value):.6f}"


def _format_ratio(value: float) -> str:
	if value == float("inf"):
		return "inf"
	if value == float("-inf"):
		return "-inf"
	return f"{float(value):.6f}"


def _gamma_improvement_pct(*, baseline_gamma: float, candidate_gamma: float) -> float:
	base = float(baseline_gamma)
	alt = float(candidate_gamma)
	if base == 0.0:
		if alt > 0.0:
			return float("inf")
		if alt < 0.0:
			return float("-inf")
		return 0.0
	return ((alt - base) / abs(base)) * 100.0


def _gamma_threshold_pass(*, baseline_gamma: float, candidate_gamma: float, threshold_pct: int) -> bool:
	base = float(baseline_gamma)
	alt = float(candidate_gamma)
	threshold = float(threshold_pct) / 100.0
	if base == 0.0:
		return alt >= 0.0
	return alt >= (base + threshold * abs(base))


def _yes_no(value: bool) -> str:
	return "yes" if bool(value) else "no"


def _write_tsv(path: Path, *, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
		writer.writeheader()
		writer.writerows(rows)


def _gamma_uplift_ratio(*, baseline_gamma: float, candidate_gamma: float) -> float:
	base = float(baseline_gamma)
	alt = float(candidate_gamma)
	if base == 0.0:
		if alt > 0.0:
			return float("inf")
		if alt < 0.0:
			return float("-inf")
		return 1.0
	return alt / base


def _closer_to_zero(*, baseline: float, candidate: float) -> bool:
	return abs(float(candidate)) < abs(float(baseline))


def _level_counts(metrics_rows: list[dict[str, Any]]) -> dict[int, int]:
	levels = [int(row["cycle_level"]) for row in metrics_rows]
	return {level: levels.count(level) for level in range(4)}


def _world_state_deviated(update_rows: list[dict[str, Any]]) -> bool:
	for row in update_rows:
		if any(abs(_safe_float(row.get(name), 0.5) - 0.5) > 1e-9 for name in INITIAL_WORLD_STATE):
			return True
	return False


def _nonisomorphic_pass(update_rows: list[dict[str, Any]]) -> bool:
	for row in update_rows:
		for key in ("risk_multipliers_json", "reward_multipliers_json", "trait_multipliers_json"):
			text = str(row.get(key, "") or "").strip()
			if not text:
				continue
			if nonisomorphic_profile({key.removesuffix("_json"): json.loads(text)}):
				return True
	return False


def _coerce_metric_row(
	*,
	protocol: str,
	condition: str,
	lambda_world: float,
	world_update_interval: int,
	row: Mapping[str, Any],
) -> dict[str, Any]:
	return {
		"protocol": str(protocol),
		"condition": condition,
		"world_update_lambda": _format_float(lambda_world),
		"world_update_interval": int(world_update_interval),
		"world_mode": "adaptive_world_w1",
		"world_state_init_json": json.dumps(INITIAL_WORLD_STATE, sort_keys=True),
		"seed": _safe_int(row.get("seed")),
		"cycle_level": _safe_int(row.get("cycle_level")),
		"stage3_score": _format_float(_safe_float(row.get("stage3_score"))),
		"turn_strength": _format_float(_safe_float(row.get("turn_strength"))),
		"env_gamma": _format_float(_safe_float(row.get("env_gamma"))),
		"env_gamma_r2": _format_float(_safe_float(row.get("env_gamma_r2"))),
		"env_gamma_n_peaks": _safe_int(row.get("env_gamma_n_peaks")),
		"success_rate": _format_float(_safe_float(row.get("success_rate"))),
		"control_env_gamma": "",
		"control_stage3_score": "",
		"gamma_uplift_ratio_vs_control_seed": "",
		"stage3_uplift_vs_control_seed": "",
		"has_level3_seed": _yes_no(_safe_int(row.get("cycle_level")) >= 3),
		"out_csv": str(row.get("out_csv", "")),
		"provenance_json": str(row.get("provenance_json", "")),
	}


def _coerce_world_update_row(
	*,
	condition: str,
	lambda_world: float,
	world_update_interval: int,
	row: Mapping[str, Any],
) -> dict[str, Any]:
	return {
		"condition": condition,
		"world_update_lambda": _format_float(lambda_world),
		"world_update_interval": int(world_update_interval),
		"seed": _safe_int(row.get("seed")),
		"window_index": _safe_int(row.get("window_index")),
		"round": _safe_int(row.get("round")),
		"window_start_round": _safe_int(row.get("window_start_round")),
		"window_end_round": _safe_int(row.get("window_end_round")),
		"scarcity": _format_float(_safe_float(row.get("scarcity"), 0.5)),
		"threat": _format_float(_safe_float(row.get("threat"), 0.5)),
		"noise": _format_float(_safe_float(row.get("noise"), 0.5)),
		"intel": _format_float(_safe_float(row.get("intel"), 0.5)),
		"dominant_event_type": str(row.get("dominant_event_type", "")),
		"a_new": _format_float(_safe_float(row.get("a_new"))),
		"b_new": _format_float(_safe_float(row.get("b_new"))),
		"event_distribution": str(row.get("event_distribution", "{}")),
		"p_aggressive": _format_float(_safe_float(row.get("p_aggressive"), 1.0 / 3.0)),
		"p_defensive": _format_float(_safe_float(row.get("p_defensive"), 1.0 / 3.0)),
		"p_balanced": _format_float(_safe_float(row.get("p_balanced"), 1.0 / 3.0)),
		"mean_reward_window": _format_float(_safe_float(row.get("mean_reward_window"))),
		"event_share_threat": _format_float(_safe_float(row.get("event_share_threat"))),
		"event_share_resource": _format_float(_safe_float(row.get("event_share_resource"))),
		"event_share_uncertainty": _format_float(_safe_float(row.get("event_share_uncertainty"))),
		"event_share_navigation": _format_float(_safe_float(row.get("event_share_navigation"))),
		"event_share_internal": _format_float(_safe_float(row.get("event_share_internal"))),
		"state_deviated": _yes_no(bool(row.get("state_deviated", False))),
		"risk_multipliers_json": str(row.get("risk_multipliers_json", "{}")),
		"reward_multipliers_json": str(row.get("reward_multipliers_json", "{}")),
		"trait_multipliers_json": str(row.get("trait_multipliers_json", "{}")),
	}


def _build_condition_summary(
	*,
	protocol: str,
	condition: str,
	lambda_world: float,
	world_update_interval: int,
	metrics_rows: list[dict[str, Any]],
	world_update_rows: list[dict[str, Any]],
	events_json: Path,
	out_dir: Path,
	selection_strength: float,
	memory_kernel: int,
) -> dict[str, Any]:
	level_counts = _level_counts(metrics_rows)
	n_seeds = len(metrics_rows)
	p_level_ge_2 = 0.0 if n_seeds == 0 else sum(int(row["cycle_level"]) >= 2 for row in metrics_rows) / float(n_seeds)
	p_level_3 = 0.0 if n_seeds == 0 else sum(int(row["cycle_level"]) >= 3 for row in metrics_rows) / float(n_seeds)
	return {
		"protocol": str(protocol),
		"condition": condition,
		"world_update_lambda": _format_float(lambda_world),
		"world_update_interval": int(world_update_interval),
		"world_mode": "adaptive_world_w1",
		"world_state_init_json": json.dumps(INITIAL_WORLD_STATE, sort_keys=True),
		"n_seeds": int(n_seeds),
		"mean_cycle_level": _format_float(_safe_mean([_safe_float(row["cycle_level"]) for row in metrics_rows])),
		"mean_stage3_score": _format_float(_safe_mean([_safe_float(row["stage3_score"]) for row in metrics_rows])),
		"mean_turn_strength": _format_float(_safe_mean([_safe_float(row["turn_strength"]) for row in metrics_rows])),
		"mean_env_gamma": _format_float(_safe_mean([_safe_float(row["env_gamma"]) for row in metrics_rows])),
		"mean_success_rate": _format_float(_safe_mean([_safe_float(row["success_rate"]) for row in metrics_rows])),
		"level_counts_json": json.dumps(level_counts, sort_keys=True),
		"p_level_ge_2": _format_float(p_level_ge_2),
		"p_level_3": _format_float(p_level_3),
		"mean_world_scarcity": _format_float(_safe_mean([_safe_float(row["scarcity"], 0.5) for row in world_update_rows])),
		"mean_world_threat": _format_float(_safe_mean([_safe_float(row["threat"], 0.5) for row in world_update_rows])),
		"mean_world_noise": _format_float(_safe_mean([_safe_float(row["noise"], 0.5) for row in world_update_rows])),
		"mean_world_intel": _format_float(_safe_mean([_safe_float(row["intel"], 0.5) for row in world_update_rows])),
		"world_state_deviated": _yes_no(_world_state_deviated(world_update_rows)),
		"nonisomorphic_pass": _yes_no(_nonisomorphic_pass(world_update_rows)),
		"events_json": str(events_json),
		"selection_strength": _format_float(selection_strength),
		"memory_kernel": int(memory_kernel),
		"out_dir": str(out_dir),
	}


def _write_decision(
	path: Path,
	*,
	protocol: str,
	decision: str,
	control_row: Mapping[str, Any],
	combined_rows: list[dict[str, Any]],
) -> None:
	protocol_spec = _get_protocol_spec(protocol)
	lines = [
		f"# {protocol_spec['label']} Decision",
		"",
		f"- protocol: {protocol}",
		f"- decision: {decision}",
		f"- control_mean_env_gamma: {control_row['mean_env_gamma']}",
		f"- control_mean_stage3_score: {control_row['mean_stage3_score']}",
		f"- control_level3_seed_count: {control_row['level3_seed_count']}",
		f"- control_world_state_deviated: {control_row['world_state_deviated']}",
		f"- control_nonisomorphic_pass: {control_row['nonisomorphic_pass']}",
		"",
		"## Cell Verdicts",
		"",
		"| Cell | Level 3 seeds | mean_stage3_score | mean_env_gamma | gamma_improve_pct | verdict |",
		"|---|---:|---:|---:|---:|---|",
	]
	for row in combined_rows:
		lines.append(
			f"| {row['condition']} | {row['level3_seed_count']} | {row['mean_stage3_score']} | {row['mean_env_gamma']} | {row['gamma_improve_pct']} | {row['verdict']} |"
		)
	lines.append("")
	lines.extend(
		[
			"## Stop Rule",
			"",
		]
	)
	for line in list(protocol_spec["stop_rule_lines"]):
		lines.append(f"- {line}")
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_cell_config(
	*,
	condition: str,
	lambda_world: float,
	out_root: Path,
	events_json: Path,
	players: int,
	rounds: int,
	selection_strength: float,
	init_bias: float,
	memory_kernel: int,
	world_update_interval: int,
	burn_in: int,
	tail: int,
	a: float,
	b: float,
	cross: float,
) -> W1CellConfig:
	return W1CellConfig(
		condition=str(condition),
		lambda_world=float(lambda_world),
		out_dir=out_root / str(condition),
		events_json=events_json,
		players=int(players),
		rounds=int(rounds),
		selection_strength=float(selection_strength),
		init_bias=float(init_bias),
		memory_kernel=int(memory_kernel),
		world_update_interval=int(world_update_interval),
		burn_in=int(burn_in),
		tail=int(tail),
		a=float(a),
		b=float(b),
		cross=float(cross),
	)


def run_w1_scout(
	*,
	protocol: str = DEFAULT_PROTOCOL,
	seeds: list[int],
	out_root: Path,
	summary_tsv: Path,
	combined_tsv: Path,
	decision_md: Path,
	updates_tsv: Path,
	events_json: Path,
	players: int = 300,
	rounds: int = 3000,
	selection_strength: float = 0.06,
	init_bias: float = 0.12,
	memory_kernel: int = 3,
	world_update_interval: int = 200,
	burn_in: int = 1000,
	tail: int = 1000,
	a: float = 1.0,
	b: float = 0.9,
	cross: float = 0.20,
	cell_runner: CellRunner | None = None,
) -> dict[str, Any]:
	protocol_spec = _get_protocol_spec(protocol)
	runner = cell_runner or run_w1_cell
	out_root.mkdir(parents=True, exist_ok=True)
	cell_metrics: dict[str, list[dict[str, Any]]] = {}
	cell_updates: dict[str, list[dict[str, Any]]] = {}
	cell_summaries: dict[str, dict[str, Any]] = {}
	control_by_seed: dict[int, dict[str, Any]] | None = None
	update_manifest_rows: list[dict[str, Any]] = []

	for spec in list(protocol_spec["cells"]):
		condition = str(spec["condition"])
		lambda_world = float(spec["lambda_world"])
		condition_world_update_interval = int(spec["world_update_interval"])
		config = _build_cell_config(
			condition=condition,
			lambda_world=lambda_world,
			out_root=out_root,
			events_json=events_json,
			players=int(players),
			rounds=int(rounds),
			selection_strength=float(selection_strength),
			init_bias=float(init_bias),
			memory_kernel=int(memory_kernel),
			world_update_interval=int(condition_world_update_interval),
			burn_in=int(burn_in),
			tail=int(tail),
			a=float(a),
			b=float(b),
			cross=float(cross),
		)
		metrics_rows: list[dict[str, Any]] = []
		world_update_rows: list[dict[str, Any]] = []
		for seed in seeds:
			result = dict(runner(config, int(seed)))
			metrics_rows.append(
				_coerce_metric_row(
					protocol=protocol,
					condition=condition,
					lambda_world=lambda_world,
					world_update_interval=int(condition_world_update_interval),
					row=result,
				)
			)
			seed_update_rows = [
				_coerce_world_update_row(
					condition=condition,
					lambda_world=lambda_world,
					world_update_interval=int(condition_world_update_interval),
					row=row,
				)
				for row in list(result.get("world_update_rows", []))
			]
			seed_updates_path = Path(str(result.get("world_updates_tsv", config.world_updates_path(int(seed)))))
			_write_tsv(seed_updates_path, fieldnames=WORLD_UPDATE_FIELDNAMES, rows=seed_update_rows)
			update_manifest_rows.append(
				{
					"protocol": str(protocol),
					"condition": condition,
					"seed": int(seed),
					"world_updates_tsv": str(seed_updates_path),
				}
			)
			world_update_rows.extend(seed_update_rows)
		condition_summary = _build_condition_summary(
			protocol=protocol,
			condition=condition,
			lambda_world=lambda_world,
			world_update_interval=int(condition_world_update_interval),
			metrics_rows=metrics_rows,
			world_update_rows=world_update_rows,
			events_json=events_json,
			out_dir=config.out_dir,
			selection_strength=float(selection_strength),
			memory_kernel=int(memory_kernel),
		)
		cell_metrics[condition] = metrics_rows
		cell_updates[condition] = world_update_rows
		cell_summaries[condition] = condition_summary
		if condition == "control":
			control_by_seed = {int(row["seed"]): row for row in metrics_rows}

	if control_by_seed is None:
		raise RuntimeError("missing W1 control metrics")

	control_summary = cell_summaries["control"]
	control_mean_env_gamma = float(control_summary["mean_env_gamma"])
	control_mean_stage3_score = float(control_summary["mean_stage3_score"])
	gamma_gate_threshold_pct = int(protocol_spec["gamma_gate_threshold_pct"])
	require_stage3_not_below_control = bool(protocol_spec["require_stage3_not_below_control"])

	summary_rows: list[dict[str, Any]] = []
	combined_rows: list[dict[str, Any]] = []
	for spec in list(protocol_spec["cells"]):
		condition = str(spec["condition"])
		metrics_rows = cell_metrics[condition]
		for row in metrics_rows:
			seed = int(row["seed"])
			if condition == "control":
				row["control_env_gamma"] = _format_float(float(row["env_gamma"]))
				row["control_stage3_score"] = _format_float(float(row["stage3_score"]))
				row["gamma_uplift_ratio_vs_control_seed"] = _format_ratio(1.0)
				row["stage3_uplift_vs_control_seed"] = _format_float(0.0)
			else:
				control_row = control_by_seed.get(seed)
				if control_row is None:
					raise RuntimeError(f"missing control metrics for seed {seed}")
				row["control_env_gamma"] = _format_float(float(control_row["env_gamma"]))
				row["control_stage3_score"] = _format_float(float(control_row["stage3_score"]))
				row["gamma_uplift_ratio_vs_control_seed"] = _format_ratio(
					_gamma_uplift_ratio(
						baseline_gamma=float(control_row["env_gamma"]),
						candidate_gamma=float(row["env_gamma"]),
					)
				)
				row["stage3_uplift_vs_control_seed"] = _format_float(
					float(row["stage3_score"]) - float(control_row["stage3_score"])
				)
			summary_rows.append(dict(row))

		level3_seed_count = sum(1 for row in metrics_rows if int(row["cycle_level"]) >= 3)
		mean_stage3_score = float(cell_summaries[condition]["mean_stage3_score"])
		mean_env_gamma = float(cell_summaries[condition]["mean_env_gamma"])
		stage3_uplift = mean_stage3_score - control_mean_stage3_score
		gamma_ratio = _gamma_uplift_ratio(
			baseline_gamma=control_mean_env_gamma,
			candidate_gamma=mean_env_gamma,
		)
		gamma_flags = gamma_improvement_flags(
			baseline_gamma=control_mean_env_gamma,
			hetero_gamma=mean_env_gamma,
		)
		gamma_improved_20pct = bool(gamma_flags["gamma_relative_20pct"])
		gamma_improve_pct = _gamma_improvement_pct(
			baseline_gamma=control_mean_env_gamma,
			candidate_gamma=mean_env_gamma,
		)
		gamma_gate_pass = _gamma_threshold_pass(
			baseline_gamma=control_mean_env_gamma,
			candidate_gamma=mean_env_gamma,
			threshold_pct=gamma_gate_threshold_pct,
		)
		stage3_not_below_control = mean_stage3_score >= control_mean_stage3_score
		promotion_gate_pass = level3_seed_count >= 1 and gamma_gate_pass and (
			not require_stage3_not_below_control or stage3_not_below_control
		)
		nonisomorphic_pass = str(cell_summaries[condition]["nonisomorphic_pass"]) == "yes"
		verdict = "control"
		if condition != "control":
			if not nonisomorphic_pass and str(cell_summaries[condition]["world_state_deviated"]) == "yes":
				verdict = "nonisomorphic_fail"
			elif promotion_gate_pass:
				verdict = "pass"
			elif stage3_uplift > 0.0 or _closer_to_zero(baseline=control_mean_env_gamma, candidate=mean_env_gamma) or str(cell_summaries[condition]["world_state_deviated"]) == "yes":
				verdict = "weak_positive"
			else:
				verdict = "fail"
		combined_rows.append(
			{
				**cell_summaries[condition],
				"level3_seed_count": int(level3_seed_count),
				"control_mean_env_gamma": _format_float(control_mean_env_gamma),
				"control_mean_stage3_score": _format_float(control_mean_stage3_score),
				"gamma_uplift_ratio_vs_control": _format_ratio(gamma_ratio),
				"gamma_improve_pct": _format_ratio(gamma_improve_pct),
				"stage3_uplift_vs_control": _format_float(stage3_uplift),
				"gamma_improved_20pct": _yes_no(gamma_improved_20pct),
				"gamma_gate_threshold_pct": int(gamma_gate_threshold_pct),
				"gamma_gate_pass": _yes_no(gamma_gate_pass),
				"stage3_not_below_control": _yes_no(stage3_not_below_control),
				"promotion_gate_pass": _yes_no(promotion_gate_pass),
				"verdict": verdict,
			}
		)

	candidate_rows = [row for row in combined_rows if str(row["condition"]) != "control"]
	decision = "fail"
	if any(str(row["verdict"]) == "pass" for row in candidate_rows):
		decision = "pass"
	elif str(protocol_spec["decision_without_pass"]) == "close_w1":
		decision = "close_w1"
	elif any(str(row["verdict"]) in {"weak_positive", "nonisomorphic_fail"} for row in candidate_rows):
		decision = "weak_positive"

	_write_tsv(summary_tsv, fieldnames=METRIC_FIELDNAMES, rows=summary_rows)
	_write_tsv(combined_tsv, fieldnames=COMBINED_FIELDNAMES, rows=combined_rows)
	_write_tsv(updates_tsv, fieldnames=UPDATES_MANIFEST_FIELDNAMES, rows=update_manifest_rows)
	control_combined_row = next(row for row in combined_rows if str(row["condition"]) == "control")
	_write_decision(
		decision_md,
		protocol=protocol,
		decision=decision,
		control_row=control_combined_row,
		combined_rows=combined_rows,
	)
	return {
		"decision": decision,
		"summary_tsv": str(summary_tsv),
		"combined_tsv": str(combined_tsv),
		"decision_md": str(decision_md),
		"updates_tsv": str(updates_tsv),
		"combined_rows": combined_rows,
	}


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="W1 in-loop adaptive world harness")
	parser.add_argument("--protocol", type=str, default=DEFAULT_PROTOCOL, choices=sorted(PROTOCOL_SPECS))
	parser.add_argument("--seeds", type=str, default="45,47,49")
	parser.add_argument("--out-root", type=Path, default=None)
	parser.add_argument("--summary-tsv", type=Path, default=None)
	parser.add_argument("--combined-tsv", type=Path, default=None)
	parser.add_argument("--decision-md", type=Path, default=None)
	parser.add_argument("--updates-tsv", type=Path, default=None)
	parser.add_argument("--events-json", type=Path, default=None)
	parser.add_argument("--players", type=int, default=300)
	parser.add_argument("--rounds", type=int, default=3000)
	parser.add_argument("--selection-strength", type=float, default=0.06)
	parser.add_argument("--init-bias", type=float, default=0.12)
	parser.add_argument("--memory-kernel", type=int, default=3)
	parser.add_argument("--world-update-interval", type=int, default=200)
	parser.add_argument("--burn-in", type=int, default=1000)
	parser.add_argument("--tail", type=int, default=1000)
	parser.add_argument("--a", type=float, default=1.0)
	parser.add_argument("--b", type=float, default=0.9)
	parser.add_argument("--cross", type=float, default=0.20)
	return parser


def main(*, cell_runner: CellRunner | None = None) -> None:
	parser = _build_parser()
	args = parser.parse_args()
	protocol = str(args.protocol)
	result = run_w1_scout(
		protocol=protocol,
		seeds=_parse_seeds(args.seeds),
		out_root=_resolve_protocol_path(protocol, "out_root", args.out_root),
		summary_tsv=_resolve_protocol_path(protocol, "summary_tsv", args.summary_tsv),
		combined_tsv=_resolve_protocol_path(protocol, "combined_tsv", args.combined_tsv),
		decision_md=_resolve_protocol_path(protocol, "decision_md", args.decision_md),
		updates_tsv=_resolve_protocol_path(protocol, "updates_tsv", args.updates_tsv),
		events_json=_resolve_protocol_path(protocol, "events_json", args.events_json),
		players=int(args.players),
		rounds=int(args.rounds),
		selection_strength=float(args.selection_strength),
		init_bias=float(args.init_bias),
		memory_kernel=int(args.memory_kernel),
		world_update_interval=int(args.world_update_interval),
		burn_in=int(args.burn_in),
		tail=int(args.tail),
		a=float(args.a),
		b=float(args.b),
		cross=float(args.cross),
		cell_runner=cell_runner,
	)
	print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()