from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path
from typing import Any, Mapping

from simulation.personality_gate0 import (
	DEFAULT_GATE0_EVENTS_JSON,
	PROTOTYPES,
	projected_initial_weights,
	sample_personality,
)
from simulation.personality_gate1 import run_condition
from simulation.run_simulation import SimConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
LITTLE_DRAGON_MODULE_PATH = REPO_ROOT / "docs" / "personality_dungeon_v1" / "05_little_dragon_v1.py"

DEFAULT_GATE1_OUT_ROOT = REPO_ROOT / "outputs" / "h6_gate1_expanded"
DEFAULT_GATE1_SUMMARY_TSV = REPO_ROOT / "outputs" / "h6_gate1_expanded_summary.tsv"
DEFAULT_GATE1_COMBINED_TSV = REPO_ROOT / "outputs" / "h6_gate1_expanded_combined.tsv"
DEFAULT_GATE1_DECISION_MD = REPO_ROOT / "outputs" / "h6_gate1_expanded_decision.md"

DEFAULT_GATE2_STEPS_TSV = REPO_ROOT / "outputs" / "h6_gate2_offline_steps.tsv"
DEFAULT_GATE2_SUMMARY_TSV = REPO_ROOT / "outputs" / "h6_gate2_offline_summary.tsv"
DEFAULT_GATE2_DECISION_MD = REPO_ROOT / "outputs" / "h6_gate2_offline_decision.md"

DEFAULT_GATE2_BASE_A = 0.4
DEFAULT_GATE2_BASE_B = 0.24254
STRATEGY_ORDER = ("aggressive", "defensive", "balanced")

GATE1_METRIC_FIELDNAMES = [
	"condition",
	"ratio_spec",
	"is_zero_control",
	"is_reference_control",
	"seed",
	"cycle_level",
	"stage3_score",
	"turn_strength",
	"env_gamma",
	"env_gamma_r2",
	"env_gamma_n_peaks",
	"success_rate",
	"out_csv",
	"provenance_json",
]

GATE1_COMBINED_FIELDNAMES = [
	"condition",
	"ratio_spec",
	"is_zero_control",
	"is_reference_control",
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
	"reference_mean_env_gamma",
	"reference_mean_stage3_score",
	"gamma_uplift_ratio_vs_reference",
	"stage3_uplift_vs_reference",
	"stage3_gate_pass",
	"level3_seed_pass",
	"gate1_pass",
	"verdict",
	"events_json",
	"personality_jitter",
	"selection_strength",
	"memory_kernel",
	"out_dir",
]

GATE2_STEP_FIELDNAMES = [
	"source_condition",
	"seed",
	"round",
	"dominant_strategy",
	"previous_dominant_strategy",
	"dominant_changed",
	"dominance_bias",
	"event_type",
	"previous_event_type",
	"a",
	"b",
	"pressure",
	"risk_drift",
	"threshold_multiplier",
	"anti_pressure_pass",
	"response_pass",
]

GATE2_SUMMARY_FIELDNAMES = [
	"source_condition",
	"seed",
	"n_snapshots",
	"n_changed_steps",
	"n_evaluable_changed_steps",
	"n_response_pass",
	"anti_pressure_share",
	"decision_basis",
]


def _parse_seeds(spec: str) -> list[int]:
	parts = [part.strip() for part in str(spec).split(",") if part.strip()]
	if not parts:
		raise ValueError("--seeds cannot be empty")
	return [int(part) for part in parts]


def _format_ratio(value: float) -> str:
	if value == float("inf"):
		return "inf"
	if value == float("-inf"):
		return "-inf"
	return f"{float(value):.6f}"


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


def _parse_ratio_spec(spec: str) -> tuple[int, int, int]:
	parts = [part.strip() for part in str(spec).split(":") if part.strip()]
	if len(parts) != 3:
		raise ValueError(f"ratio_spec must contain 3 integers: {spec!r}")
	ratio = tuple(int(part) for part in parts)
	if any(value < 0 for value in ratio):
		raise ValueError(f"ratio_spec cannot contain negative values: {spec!r}")
	if sum(ratio) <= 0:
		raise ValueError(f"ratio_spec must have positive total weight: {spec!r}")
	return ratio


def _ratio_counts(*, n_players: int, ratio_spec: str) -> dict[str, int]:
	ratio = _parse_ratio_spec(ratio_spec)
	total = sum(ratio)
	if int(n_players) % int(total) != 0:
		raise ValueError(f"n_players={n_players} must be divisible by ratio total {total}")
	unit = int(n_players) // int(total)
	return {
		strategy: int(ratio[index]) * unit
		for index, strategy in enumerate(STRATEGY_ORDER)
	}


def build_ratio_player_setup_callback(
	*,
	ratio_spec: str,
	jitter: float,
	personality_seed: int,
) -> Any:
	def _setup(players: list[object], _strategy_space: list[str], cfg: SimConfig) -> None:
		counts_local = _ratio_counts(n_players=len(players), ratio_spec=ratio_spec)
		rng = random.Random(personality_seed)
		personalities: list[dict[str, float]] = []
		for strategy in STRATEGY_ORDER:
			prototype = PROTOTYPES[strategy]
			for _ in range(int(counts_local[strategy])):
				personalities.append(sample_personality(prototype, jitter=float(jitter), rng=rng))
		if len(personalities) != len(players):
			raise RuntimeError("personality count mismatch in ratio setup")
		for player, personality in zip(players, personalities, strict=True):
			player.personality = dict(personality)
			player.update_weights(projected_initial_weights(personality, init_bias=float(cfg.init_bias)))

	return _setup


def _gate1_condition_specs() -> list[dict[str, Any]]:
	return [
		{
			"condition": "zero_control",
			"ratio_spec": "0:0:0",
			"is_zero_control": True,
			"is_reference_control": False,
		},
		{
			"condition": "ratio_111_reference",
			"ratio_spec": "1:1:1",
			"is_zero_control": False,
			"is_reference_control": True,
		},
		{
			"condition": "ratio_210_aggressive",
			"ratio_spec": "2:1:0",
			"is_zero_control": False,
			"is_reference_control": False,
		},
		{
			"condition": "ratio_120_defensive",
			"ratio_spec": "1:2:0",
			"is_zero_control": False,
			"is_reference_control": False,
		},
		{
			"condition": "ratio_102_balanced",
			"ratio_spec": "1:0:2",
			"is_zero_control": False,
			"is_reference_control": False,
		},
		{
			"condition": "ratio_300_aggressive",
			"ratio_spec": "3:0:0",
			"is_zero_control": False,
			"is_reference_control": False,
		},
		{
			"condition": "ratio_030_defensive",
			"ratio_spec": "0:3:0",
			"is_zero_control": False,
			"is_reference_control": False,
		},
		{
			"condition": "ratio_003_balanced",
			"ratio_spec": "0:0:3",
			"is_zero_control": False,
			"is_reference_control": False,
		},
	]


def _gate1_rank_key(row: Mapping[str, Any]) -> tuple[float, int, float]:
	return (
		float(row["stage3_uplift_vs_reference"]),
		int(row["level3_seed_count"]),
		float(row["_gamma_ratio_numeric"]),
	)


def _write_gate1_decision(
	path: Path,
	*,
	decision: str,
	reference_row: Mapping[str, Any],
	zero_row: Mapping[str, Any],
	combined_rows: list[dict[str, Any]],
	best_row: Mapping[str, Any] | None,
) -> None:
	lines = [
		"# H6 Gate 1 Decision",
		"",
		f"- decision: {decision}",
		f"- gate2_allowed: {_yes_no(decision == 'pass' and best_row is not None)}",
		f"- reference_condition: {reference_row['condition']}",
		f"- reference_mean_env_gamma: {reference_row['mean_env_gamma']}",
		f"- reference_mean_stage3_score: {reference_row['mean_stage3_score']}",
		f"- zero_control_mean_env_gamma: {zero_row['mean_env_gamma']}",
		f"- zero_control_mean_stage3_score: {zero_row['mean_stage3_score']}",
	]
	if best_row is None:
		lines.append("- best_condition: none")
	else:
		lines.extend(
			[
				f"- best_condition: {best_row['condition']}",
				f"- best_ratio_spec: {best_row['ratio_spec']}",
				f"- best_stage3_uplift_vs_reference: {best_row['stage3_uplift_vs_reference']}",
				f"- best_level3_seed_count: {best_row['level3_seed_count']}",
				f"- best_gamma_uplift_ratio_vs_reference: {best_row['gamma_uplift_ratio_vs_reference']}",
			]
		)
	lines.extend(["", "## Candidate Verdicts", ""])
	for row in combined_rows:
		if row["is_zero_control"] == "yes" or row["is_reference_control"] == "yes":
			continue
		lines.append(
			f"- {row['condition']}: ratio={row['ratio_spec']} stage3_uplift={row['stage3_uplift_vs_reference']} level3_seeds={row['level3_seed_count']} gamma_ratio={row['gamma_uplift_ratio_vs_reference']} verdict={row['verdict']}"
		)
	lines.extend([
		"",
		"## Stop Rule",
		"",
		"- 只有 Gate 1 整體 `pass` 才能開 Gate 2 offline Little Dragon。",
		"- 若只有 weak_positive，應把結論記為『靜態 personality/event 已逼近 replicator plateau 上限』。",
	])
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_little_dragon_module() -> Any:
	spec = importlib.util.spec_from_file_location("personality_little_dragon_v1", LITTLE_DRAGON_MODULE_PATH)
	if spec is None or spec.loader is None:
		raise RuntimeError(f"failed to load Little Dragon module from {LITTLE_DRAGON_MODULE_PATH}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def _anti_pressure_pass(*, result: Mapping[str, Any], base_a: float, base_b: float) -> bool:
	dominant = str(result["dominant_strategy"])
	event_type = str(result["event_type"])
	a_value = float(result["a"])
	b_value = float(result["b"])
	if dominant == "aggressive":
		return event_type == "Threat" and a_value >= float(base_a) and b_value >= float(base_b)
	if dominant == "defensive":
		return event_type == "Resource" and a_value <= float(base_a) and b_value >= float(base_b)
	if dominant == "balanced":
		return event_type == "Uncertainty" and a_value >= float(base_a) and b_value <= float(base_b)
	return False


def _discover_seed_csvs(input_dir: Path) -> dict[int, Path]:
	seed_csvs: dict[int, Path] = {}
	for path in sorted(input_dir.glob("seed*.csv")):
		stem = path.stem
		if not stem.startswith("seed"):
			continue
		seed = int(stem.removeprefix("seed"))
		seed_csvs[seed] = path
	if not seed_csvs:
		raise ValueError(f"no seed*.csv files found in {input_dir}")
	return seed_csvs


def _write_gate2_decision(
	path: Path,
	*,
	decision: str,
	source_condition: str,
	sampling_interval: int,
	total_changed_steps: int,
	total_evaluable_changed_steps: int,
	total_response_pass: int,
	anti_pressure_share: float,
) -> None:
	lines = [
		"# H6 Gate 2 Decision",
		"",
		f"- decision: {decision}",
		f"- source_condition: {source_condition}",
		f"- sampling_interval: {int(sampling_interval)}",
		f"- n_changed_steps: {int(total_changed_steps)}",
		f"- n_evaluable_changed_steps: {int(total_evaluable_changed_steps)}",
		f"- n_response_pass: {int(total_response_pass)}",
		f"- anti_pressure_share: {anti_pressure_share:.6f}",
		"",
		"## Stop Rule",
		"",
		"- 若沒有任何可評估 dominant-change steps，Gate 2 直接記為 fail。",
		"- 只有 anti_pressure_share >= 0.70 才能宣稱 offline Little Dragon 映射成立。",
	]
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_gate1(args: argparse.Namespace) -> dict[str, Any]:
	results: list[dict[str, Any]] = []
	seed_csvs_by_condition: dict[str, dict[int, Path]] = {}
	metric_rows: list[dict[str, Any]] = []
	for spec in _gate1_condition_specs():
		out_dir = Path(args.gate1_out_root) / str(spec["condition"])
		callback = None
		world_mode = "zero"
		if not spec["is_zero_control"]:
			world_mode = "heterogeneous"
		condition_rows, condition_summary, seed_csvs, _actions, _diagnostics = run_condition(
			condition=str(spec["condition"]),
			world_mode=world_mode,
			seeds=list(args.seeds),
			out_dir=out_dir,
			events_json=Path(args.events_json),
			players=int(args.players),
			rounds=int(args.rounds),
			selection_strength=float(args.selection_strength),
			init_bias=float(args.init_bias),
			memory_kernel=int(args.memory_kernel),
			burn_in=int(args.burn_in),
			tail=int(args.tail),
			jitter=float(args.jitter),
			a=float(args.a),
			b=float(args.b),
			cross=float(args.cross),
			player_setup_callback=callback,
			player_setup_callback_factory=(
				None
				if spec["is_zero_control"]
				else lambda seed, ratio_spec=str(spec["ratio_spec"]), jitter=float(args.jitter): build_ratio_player_setup_callback(
					ratio_spec=ratio_spec,
					jitter=jitter,
					personality_seed=int(seed),
				)
			),
		)
		for row in condition_rows:
			metric_rows.append(
				{
					"condition": str(spec["condition"]),
					"ratio_spec": str(spec["ratio_spec"]),
					"is_zero_control": _yes_no(spec["is_zero_control"]),
					"is_reference_control": _yes_no(spec["is_reference_control"]),
					"seed": row["seed"],
					"cycle_level": row["cycle_level"],
					"stage3_score": row["stage3_score"],
					"turn_strength": row["turn_strength"],
					"env_gamma": row["env_gamma"],
					"env_gamma_r2": row["env_gamma_r2"],
					"env_gamma_n_peaks": row["env_gamma_n_peaks"],
					"success_rate": row["success_rate"],
					"out_csv": row["out_csv"],
					"provenance_json": row["provenance_json"],
				}
			)
		seed_csvs_by_condition[str(spec["condition"])] = seed_csvs
		results.append(
			{
				"spec": spec,
				"summary": condition_summary,
				"rows": condition_rows,
				"seed_csvs": seed_csvs,
			}
		)

	reference_result = next(result for result in results if bool(result["spec"]["is_reference_control"]))
	zero_result = next(result for result in results if bool(result["spec"]["is_zero_control"]))
	reference_mean_gamma = float(reference_result["summary"]["mean_env_gamma"])
	reference_mean_stage3 = float(reference_result["summary"]["mean_stage3_score"])
	combined_rows: list[dict[str, Any]] = []
	for result in results:
		spec = result["spec"]
		summary = result["summary"]
		rows = result["rows"]
		mean_gamma = float(summary["mean_env_gamma"])
		mean_stage3 = float(summary["mean_stage3_score"])
		level3_seed_count = sum(1 for row in rows if int(row["cycle_level"]) >= 3)
		gamma_ratio = _gamma_uplift_ratio(baseline_gamma=reference_mean_gamma, candidate_gamma=mean_gamma)
		stage3_uplift = mean_stage3 - reference_mean_stage3
		stage3_gate_pass = (not spec["is_zero_control"] and not spec["is_reference_control"] and stage3_uplift >= float(args.gate1_stage3_min_uplift))
		level3_seed_pass = (not spec["is_zero_control"] and not spec["is_reference_control"] and level3_seed_count >= int(args.gate1_min_level3_seeds))
		gate1_pass = bool(stage3_gate_pass and level3_seed_pass)
		if spec["is_zero_control"] or spec["is_reference_control"]:
			verdict = "control"
		elif gate1_pass:
			verdict = "pass"
		elif stage3_uplift > 0.0 or level3_seed_count > 0 or mean_gamma > reference_mean_gamma:
			verdict = "weak_positive"
		else:
			verdict = "fail"
		combined_rows.append(
			{
				"condition": str(spec["condition"]),
				"ratio_spec": str(spec["ratio_spec"]),
				"is_zero_control": _yes_no(spec["is_zero_control"]),
				"is_reference_control": _yes_no(spec["is_reference_control"]),
				"n_seeds": summary["n_seeds"],
				"mean_cycle_level": summary["mean_cycle_level"],
				"mean_stage3_score": summary["mean_stage3_score"],
				"mean_turn_strength": summary["mean_turn_strength"],
				"mean_env_gamma": summary["mean_env_gamma"],
				"mean_success_rate": summary["mean_success_rate"],
				"level_counts_json": summary["level_counts_json"],
				"p_level_ge_2": summary["p_level_ge_2"],
				"p_level_3": summary["p_level_3"],
				"level3_seed_count": level3_seed_count,
				"reference_mean_env_gamma": f"{reference_mean_gamma:.6f}",
				"reference_mean_stage3_score": f"{reference_mean_stage3:.6f}",
				"gamma_uplift_ratio_vs_reference": _format_ratio(gamma_ratio),
				"stage3_uplift_vs_reference": f"{stage3_uplift:.6f}",
				"stage3_gate_pass": _yes_no(stage3_gate_pass),
				"level3_seed_pass": _yes_no(level3_seed_pass),
				"gate1_pass": _yes_no(gate1_pass),
				"verdict": verdict,
				"events_json": summary["events_json"],
				"personality_jitter": summary["personality_jitter"],
				"selection_strength": f"{float(args.selection_strength):.6f}",
				"memory_kernel": int(args.memory_kernel),
				"out_dir": summary["out_dir"],
				"_gamma_ratio_numeric": gamma_ratio,
			}
		)

	pass_rows = [row for row in combined_rows if row["gate1_pass"] == "yes"]
	weak_rows = [row for row in combined_rows if row["verdict"] == "weak_positive"]
	best_row = max(pass_rows, key=_gate1_rank_key) if pass_rows else None
	if best_row is not None:
		decision = "pass"
	elif weak_rows:
		decision = "weak_positive"
	else:
		decision = "fail"

	_write_tsv(Path(args.gate1_summary_tsv), fieldnames=GATE1_METRIC_FIELDNAMES, rows=metric_rows)
	_write_tsv(
		Path(args.gate1_combined_tsv),
		fieldnames=GATE1_COMBINED_FIELDNAMES,
		rows=[{key: value for key, value in row.items() if not str(key).startswith("_")} for row in combined_rows],
	)
	_write_gate1_decision(
		Path(args.gate1_decision_md),
		decision=decision,
		reference_row=next(row for row in combined_rows if row["is_reference_control"] == "yes"),
		zero_row=next(row for row in combined_rows if row["is_zero_control"] == "yes"),
		combined_rows=combined_rows,
		best_row=best_row,
	)
	return {
		"decision": decision,
		"best_row": best_row,
		"combined_rows": combined_rows,
		"seed_csvs_by_condition": seed_csvs_by_condition,
	}


def _run_gate2(
	args: argparse.Namespace,
	*,
	source_condition: str,
	seed_csvs: Mapping[int, Path],
) -> dict[str, Any]:
	little_dragon = _load_little_dragon_module()
	step_rows: list[dict[str, Any]] = []
	summary_rows: list[dict[str, Any]] = []
	total_changed_steps = 0
	total_evaluable_changed_steps = 0
	total_response_pass = 0
	for seed, csv_path in sorted(seed_csvs.items()):
		with Path(csv_path).open(newline="", encoding="utf-8") as handle:
			rows = list(csv.DictReader(handle))
		snapshots = [row for row in rows if int(row["round"]) % int(args.gate2_sampling_interval) == 0]
		previous_dominant: str | None = None
		previous_event_type: str | None = None
		previous_a: float | None = None
		previous_b: float | None = None
		changed_steps = 0
		evaluable_changed_steps = 0
		response_pass_count = 0
		for snapshot in snapshots:
			global_p = {
				"aggressive": float(snapshot["p_aggressive"]),
				"defensive": float(snapshot["p_defensive"]),
				"balanced": float(snapshot["p_balanced"]),
			}
			result = little_dragon.generate_adaptive_dungeon(
				global_p,
				selection_strength=float(args.selection_strength),
				players=int(args.players),
				base_a=float(args.gate2_base_a),
				base_b=float(args.gate2_base_b),
			)
			dominant = str(result["dominant_strategy"])
			event_type = str(result["event_type"])
			dominant_changed = previous_dominant is not None and dominant != previous_dominant
			if dominant_changed:
				changed_steps += 1
			anti_pressure_pass = _anti_pressure_pass(
				result=result,
				base_a=float(args.gate2_base_a),
				base_b=float(args.gate2_base_b),
			)
			response_changed = (
				previous_event_type is not None
				and (
					event_type != previous_event_type
					or float(result["a"]) != float(previous_a)
					or float(result["b"]) != float(previous_b)
				)
			)
			response_pass = False
			if dominant_changed and float(result["dominance_bias"]) >= float(args.gate2_min_dominance_bias):
				evaluable_changed_steps += 1
				response_pass = bool(anti_pressure_pass and response_changed)
				if response_pass:
					response_pass_count += 1
			step_rows.append(
				{
					"source_condition": source_condition,
					"seed": int(seed),
					"round": int(snapshot["round"]),
					"dominant_strategy": dominant,
					"previous_dominant_strategy": "" if previous_dominant is None else previous_dominant,
					"dominant_changed": _yes_no(dominant_changed),
					"dominance_bias": f"{float(result['dominance_bias']):.6f}",
					"event_type": event_type,
					"previous_event_type": "" if previous_event_type is None else previous_event_type,
					"a": f"{float(result['a']):.6f}",
					"b": f"{float(result['b']):.6f}",
					"pressure": f"{float(result['pressure']):.6f}",
					"risk_drift": f"{float(result['risk_drift']):.6f}",
					"threshold_multiplier": f"{float(result['threshold_multiplier']):.6f}",
					"anti_pressure_pass": _yes_no(anti_pressure_pass),
					"response_pass": _yes_no(response_pass),
				}
			)
			previous_dominant = dominant
			previous_event_type = event_type
			previous_a = float(result["a"])
			previous_b = float(result["b"])
		anti_pressure_share = 0.0 if evaluable_changed_steps == 0 else float(response_pass_count) / float(evaluable_changed_steps)
		summary_rows.append(
			{
				"source_condition": source_condition,
				"seed": int(seed),
				"n_snapshots": len(snapshots),
				"n_changed_steps": changed_steps,
				"n_evaluable_changed_steps": evaluable_changed_steps,
				"n_response_pass": response_pass_count,
				"anti_pressure_share": f"{anti_pressure_share:.6f}",
				"decision_basis": "dominant_change" if evaluable_changed_steps > 0 else "no_evaluable_change",
			}
		)
		total_changed_steps += changed_steps
		total_evaluable_changed_steps += evaluable_changed_steps
		total_response_pass += response_pass_count

	anti_pressure_share = 0.0 if total_evaluable_changed_steps == 0 else float(total_response_pass) / float(total_evaluable_changed_steps)
	decision = "pass" if total_evaluable_changed_steps > 0 and anti_pressure_share >= float(args.gate2_min_response_share) else "fail"
	_write_tsv(Path(args.gate2_steps_tsv), fieldnames=GATE2_STEP_FIELDNAMES, rows=step_rows)
	_write_tsv(Path(args.gate2_summary_tsv), fieldnames=GATE2_SUMMARY_FIELDNAMES, rows=summary_rows)
	_write_gate2_decision(
		Path(args.gate2_decision_md),
		decision=decision,
		source_condition=source_condition,
		sampling_interval=int(args.gate2_sampling_interval),
		total_changed_steps=total_changed_steps,
		total_evaluable_changed_steps=total_evaluable_changed_steps,
		total_response_pass=total_response_pass,
		anti_pressure_share=anti_pressure_share,
	)
	return {
		"decision": decision,
		"anti_pressure_share": anti_pressure_share,
	}


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="H6 personality + event harness")
	parser.add_argument("--protocol", choices=["gate1", "gate2", "h6"], default="h6")
	parser.add_argument("--players", type=int, default=300)
	parser.add_argument("--rounds", type=int, default=4000)
	parser.add_argument("--seeds", type=str, default="45,47,49,51,53,55")
	parser.add_argument("--burn-in", type=int, default=800)
	parser.add_argument("--tail", type=int, default=1200)
	parser.add_argument("--selection-strength", type=float, default=0.06)
	parser.add_argument("--init-bias", type=float, default=0.12)
	parser.add_argument("--memory-kernel", type=int, default=3)
	parser.add_argument("--jitter", type=float, default=0.08)
	parser.add_argument("--events-json", type=Path, default=DEFAULT_GATE0_EVENTS_JSON)
	parser.add_argument("--a", type=float, default=1.0)
	parser.add_argument("--b", type=float, default=0.9)
	parser.add_argument("--cross", type=float, default=0.20)
	parser.add_argument("--gate1-out-root", type=Path, default=DEFAULT_GATE1_OUT_ROOT)
	parser.add_argument("--gate1-summary-tsv", type=Path, default=DEFAULT_GATE1_SUMMARY_TSV)
	parser.add_argument("--gate1-combined-tsv", type=Path, default=DEFAULT_GATE1_COMBINED_TSV)
	parser.add_argument("--gate1-decision-md", type=Path, default=DEFAULT_GATE1_DECISION_MD)
	parser.add_argument("--gate1-stage3-min-uplift", type=float, default=0.025)
	parser.add_argument("--gate1-min-level3-seeds", type=int, default=2)
	parser.add_argument("--gate2-input-dir", type=Path, default=None)
	parser.add_argument("--gate2-source-condition", type=str, default="")
	parser.add_argument("--gate2-steps-tsv", type=Path, default=DEFAULT_GATE2_STEPS_TSV)
	parser.add_argument("--gate2-summary-tsv", type=Path, default=DEFAULT_GATE2_SUMMARY_TSV)
	parser.add_argument("--gate2-decision-md", type=Path, default=DEFAULT_GATE2_DECISION_MD)
	parser.add_argument("--gate2-sampling-interval", type=int, default=300)
	parser.add_argument("--gate2-min-dominance-bias", type=float, default=0.02)
	parser.add_argument("--gate2-min-response-share", type=float, default=0.70)
	parser.add_argument("--gate2-base-a", type=float, default=DEFAULT_GATE2_BASE_A)
	parser.add_argument("--gate2-base-b", type=float, default=DEFAULT_GATE2_BASE_B)
	return parser


def main() -> int:
	parser = _build_parser()
	args = parser.parse_args()
	args.seeds = _parse_seeds(args.seeds)
	if args.protocol == "gate2" and args.gate2_input_dir is None:
		parser.error("--gate2-input-dir is required when --protocol gate2")

	if args.protocol == "gate1":
		result = _run_gate1(args)
		print(f"gate1_decision={result['decision']}")
		print(f"gate1_summary={Path(args.gate1_summary_tsv)}")
		print(f"gate1_combined={Path(args.gate1_combined_tsv)}")
		print(f"gate1_decision_md={Path(args.gate1_decision_md)}")
		return 0

	if args.protocol == "gate2":
		seed_csvs = _discover_seed_csvs(Path(args.gate2_input_dir))
		result = _run_gate2(
			args,
			source_condition=str(args.gate2_source_condition or Path(args.gate2_input_dir).name),
			seed_csvs=seed_csvs,
		)
		print(f"gate2_decision={result['decision']}")
		print(f"gate2_steps={Path(args.gate2_steps_tsv)}")
		print(f"gate2_summary={Path(args.gate2_summary_tsv)}")
		print(f"gate2_decision_md={Path(args.gate2_decision_md)}")
		return 0

	gate1_result = _run_gate1(args)
	print(f"gate1_decision={gate1_result['decision']}")
	print(f"gate1_summary={Path(args.gate1_summary_tsv)}")
	print(f"gate1_combined={Path(args.gate1_combined_tsv)}")
	print(f"gate1_decision_md={Path(args.gate1_decision_md)}")
	best_row = gate1_result["best_row"]
	if gate1_result["decision"] != "pass" or best_row is None:
		print("gate2_skipped=yes")
		return 0
	gate2_result = _run_gate2(
		args,
		source_condition=str(best_row["condition"]),
		seed_csvs=gate1_result["seed_csvs_by_condition"][str(best_row["condition"])],
	)
	print(f"gate2_decision={gate2_result['decision']}")
	print(f"gate2_steps={Path(args.gate2_steps_tsv)}")
	print(f"gate2_summary={Path(args.gate2_summary_tsv)}")
	print(f"gate2_decision_md={Path(args.gate2_decision_md)}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())