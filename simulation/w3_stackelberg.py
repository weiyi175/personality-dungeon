from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from analysis.event_provenance_summary import summarize_event_provenance
from core.stackelberg import StackelbergCommitment, leader_prefers, select_best_commitment
from simulation.personality_gate0 import DEFAULT_FULL_EVENTS_JSON
from simulation.personality_gate1 import seed_metrics
from simulation.run_simulation import SimConfig, _write_timeseries_csv, simulate


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "w3_stackelberg"
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "w3_stackelberg"
DEFAULT_SUMMARY_TSV = REPO_ROOT / "outputs" / "w3_stackelberg_summary.tsv"
DEFAULT_COMBINED_TSV = REPO_ROOT / "outputs" / "w3_stackelberg_combined.tsv"
DEFAULT_DECISION_MD = REPO_ROOT / "outputs" / "w3_stackelberg_decision.md"

SUMMARY_FIELDNAMES = [
	"protocol",
	"condition",
	"leader_action",
	"a",
	"b",
	"matrix_cross_coupling",
	"seed",
	"cycle_level",
	"mean_stage3_score",
	"mean_turn_strength",
	"mean_env_gamma",
	"env_gamma_r2",
	"env_gamma_n_peaks",
	"mean_success_rate",
	"level3_seed_count",
	"has_level3_seed",
	"out_csv",
	"provenance_json",
]

COMBINED_FIELDNAMES = [
	"protocol",
	"condition",
	"is_control",
	"leader_action",
	"a",
	"b",
	"matrix_cross_coupling",
	"n_seeds",
	"mean_cycle_level",
	"mean_stage3_score",
	"mean_turn_strength",
	"mean_env_gamma",
	"mean_success_rate",
	"level_counts_json",
	"p_level_3",
	"level3_seed_count",
	"control_mean_env_gamma",
	"control_mean_stage3_score",
	"gamma_uplift_ratio_vs_control",
	"stage3_uplift_vs_control",
	"leader_prefers_over_control",
	"is_best_commitment",
	"promotion_gate_pass",
	"verdict",
	"events_json",
	"selection_strength",
	"memory_kernel",
	"out_dir",
]

CellRunner = Callable[["W3CellConfig", int], Mapping[str, Any]]


@dataclass(frozen=True)
class W3CellConfig:
	commitment: StackelbergCommitment
	players: int
	rounds: int
	events_json: Path
	selection_strength: float
	init_bias: float
	memory_kernel: int
	burn_in: int
	tail: int
	out_dir: Path

	@property
	def condition(self) -> str:
		return str(self.commitment.condition)

	@property
	def is_control(self) -> bool:
		return self.condition == "control"

	def out_csv(self, seed: int) -> Path:
		return self.out_dir / f"seed{int(seed)}.csv"

	def provenance_path(self, seed: int) -> Path:
		return self.out_dir / f"seed{int(seed)}_provenance.json"


def _default_commitments() -> list[StackelbergCommitment]:
	return [
		StackelbergCommitment(
			condition="control",
			leader_action="baseline_commit",
			a=1.00,
			b=0.90,
			matrix_cross_coupling=0.20,
			description="Current sampled baseline",
		),
		StackelbergCommitment(
			condition="w3_cross_strong",
			leader_action="cross_guard",
			a=1.00,
			b=0.90,
			matrix_cross_coupling=0.35,
			description="Increase aggressive-defensive coexistence penalty",
		),
		StackelbergCommitment(
			condition="w3_edge_tilt",
			leader_action="edge_tilt",
			a=1.05,
			b=0.85,
			matrix_cross_coupling=0.20,
			description="Slightly sharpen cyclic edge while keeping baseline cross",
		),
		StackelbergCommitment(
			condition="w3_commit_push",
			leader_action="commit_push",
			a=1.05,
			b=0.85,
			matrix_cross_coupling=0.35,
			description="Combine edge tilt with stronger balanced push",
		),
	]


def _parse_seeds(spec: str) -> list[int]:
	parts = [part.strip() for part in str(spec).split(",") if part.strip()]
	if not parts:
		raise ValueError("--seeds cannot be empty")
	return [int(part) for part in parts]


def _safe_mean(values: list[float]) -> float:
	if not values:
		return 0.0
	return float(sum(values) / float(len(values)))


def _format_float(value: float) -> str:
	return f"{float(value):.6f}"


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


def _level_counts(metrics_rows: list[dict[str, Any]]) -> dict[int, int]:
	levels = [int(row["cycle_level"]) for row in metrics_rows]
	return {level: levels.count(level) for level in range(4)}


def _build_cell_config(
	*,
	commitment: StackelbergCommitment,
	players: int,
	rounds: int,
	events_json: Path,
	selection_strength: float,
	init_bias: float,
	memory_kernel: int,
	burn_in: int,
	tail: int,
	out_root: Path,
) -> W3CellConfig:
	return W3CellConfig(
		commitment=commitment,
		players=int(players),
		rounds=int(rounds),
		events_json=events_json,
		selection_strength=float(selection_strength),
		init_bias=float(init_bias),
		memory_kernel=int(memory_kernel),
		burn_in=int(burn_in),
		tail=int(tail),
		out_dir=out_root / commitment.condition,
	)


def run_w3_cell(config: W3CellConfig, seed: int) -> dict[str, Any]:
	config.out_dir.mkdir(parents=True, exist_ok=True)
	cfg = SimConfig(
		n_players=int(config.players),
		n_rounds=int(config.rounds),
		seed=int(seed),
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=float(config.commitment.a),
		b=float(config.commitment.b),
		matrix_cross_coupling=float(config.commitment.matrix_cross_coupling),
		init_bias=float(config.init_bias),
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=float(config.selection_strength),
		enable_events=True,
		events_json=config.events_json,
		out_csv=config.out_csv(int(seed)),
		memory_kernel=int(config.memory_kernel),
	)
	strategy_space, rows = simulate(cfg)
	_write_timeseries_csv(cfg.out_csv, strategy_space=strategy_space, rows=rows)
	seed_metric = seed_metrics(rows, burn_in=int(config.burn_in), tail=int(config.tail), eta=0.55, corr_threshold=0.09)
	provenance_path = config.provenance_path(int(seed))
	provenance_payload = {
		"protocol": PROTOCOL,
		"condition": config.condition,
		"leader_action": config.commitment.leader_action,
		"a": float(config.commitment.a),
		"b": float(config.commitment.b),
		"matrix_cross_coupling": float(config.commitment.matrix_cross_coupling),
		"event_provenance": summarize_event_provenance(cfg.out_csv, events_json=config.events_json),
	}
	provenance_path.write_text(json.dumps(provenance_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	cycle_level = int(seed_metric["cycle_level"])
	has_level3_seed = cycle_level >= 3
	return {
		"protocol": PROTOCOL,
		"condition": config.condition,
		"leader_action": config.commitment.leader_action,
		"a": _format_float(float(config.commitment.a)),
		"b": _format_float(float(config.commitment.b)),
		"matrix_cross_coupling": _format_float(float(config.commitment.matrix_cross_coupling)),
		"seed": int(seed),
		"cycle_level": cycle_level,
		"mean_stage3_score": _format_float(float(seed_metric["stage3_score"])),
		"mean_turn_strength": _format_float(float(seed_metric["turn_strength"])),
		"mean_env_gamma": _format_float(float(seed_metric["env_gamma"])),
		"env_gamma_r2": _format_float(float(seed_metric["env_gamma_r2"])),
		"env_gamma_n_peaks": int(seed_metric["env_gamma_n_peaks"]),
		"mean_success_rate": _format_float(float(seed_metric["success_rate"])),
		"level3_seed_count": 1 if has_level3_seed else 0,
		"has_level3_seed": _yes_no(has_level3_seed),
		"out_csv": str(cfg.out_csv),
		"provenance_json": str(provenance_path),
	}


def _cell_summary(
	rows: list[dict[str, Any]],
	*,
	control_row: Mapping[str, Any] | None,
	config: W3CellConfig,
) -> dict[str, Any]:
	mean_cycle_level = _safe_mean([float(row["cycle_level"]) for row in rows])
	mean_stage3_score = _safe_mean([float(row["mean_stage3_score"]) for row in rows])
	mean_turn_strength = _safe_mean([float(row["mean_turn_strength"]) for row in rows])
	mean_env_gamma = _safe_mean([float(row["mean_env_gamma"]) for row in rows])
	mean_success_rate = _safe_mean([float(row["mean_success_rate"]) for row in rows])
	level_counts = _level_counts(rows)
	level3_seed_count = sum(1 for row in rows if int(row["cycle_level"]) >= 3)
	den = float(len(rows)) or 1.0
	control_mean_env_gamma = mean_env_gamma if control_row is None else float(control_row["mean_env_gamma"])
	control_mean_stage3_score = mean_stage3_score if control_row is None else float(control_row["mean_stage3_score"])
	leader_prefers_over_control = (
		False
		if config.is_control or control_row is None
		else leader_prefers(
			{
				"condition": config.condition,
				"level3_seed_count": level3_seed_count,
				"mean_stage3_score": mean_stage3_score,
				"mean_env_gamma": mean_env_gamma,
				"mean_turn_strength": mean_turn_strength,
			},
			{
				"condition": str(control_row["condition"]),
				"level3_seed_count": int(control_row["level3_seed_count"]),
				"mean_stage3_score": float(control_row["mean_stage3_score"]),
				"mean_env_gamma": float(control_row["mean_env_gamma"]),
				"mean_turn_strength": float(control_row["mean_turn_strength"]),
			},
		)
	)
	promotion_gate_pass = (
		not config.is_control
		and level3_seed_count >= 1
		and mean_env_gamma >= 0.0
		and mean_stage3_score >= control_mean_stage3_score
	)
	if config.is_control:
		verdict = "control"
	elif promotion_gate_pass:
		verdict = "pass"
	elif leader_prefers_over_control:
		verdict = "weak_positive"
	else:
		verdict = "fail"
	return {
		"protocol": PROTOCOL,
		"condition": config.condition,
		"is_control": _yes_no(config.is_control),
		"leader_action": config.commitment.leader_action,
		"a": _format_float(float(config.commitment.a)),
		"b": _format_float(float(config.commitment.b)),
		"matrix_cross_coupling": _format_float(float(config.commitment.matrix_cross_coupling)),
		"n_seeds": len(rows),
		"mean_cycle_level": _format_float(mean_cycle_level),
		"mean_stage3_score": _format_float(mean_stage3_score),
		"mean_turn_strength": _format_float(mean_turn_strength),
		"mean_env_gamma": _format_float(mean_env_gamma),
		"mean_success_rate": _format_float(mean_success_rate),
		"level_counts_json": json.dumps(level_counts, sort_keys=True),
		"p_level_3": _format_float(level3_seed_count / den),
		"level3_seed_count": int(level3_seed_count),
		"control_mean_env_gamma": _format_float(control_mean_env_gamma),
		"control_mean_stage3_score": _format_float(control_mean_stage3_score),
		"gamma_uplift_ratio_vs_control": _format_ratio(
			_gamma_uplift_ratio(baseline_gamma=control_mean_env_gamma, candidate_gamma=mean_env_gamma)
		),
		"stage3_uplift_vs_control": _format_float(mean_stage3_score - control_mean_stage3_score),
		"leader_prefers_over_control": _yes_no(leader_prefers_over_control),
		"is_best_commitment": "no",
		"promotion_gate_pass": _yes_no(promotion_gate_pass),
		"verdict": verdict,
		"events_json": str(config.events_json),
		"selection_strength": _format_float(float(config.selection_strength)),
		"memory_kernel": int(config.memory_kernel),
		"out_dir": str(config.out_dir),
	}


def _write_decision(
	path: Path,
	*,
	decision: str,
	control_row: Mapping[str, Any],
	combined_rows: list[dict[str, Any]],
	best_condition: str | None,
) -> None:
	best_row = next((row for row in combined_rows if row["condition"] == best_condition), None)
	lines = [
		"# W3.1 Stackelberg Decision",
		"",
		f"- protocol: {PROTOCOL}",
		f"- decision: {decision}",
		f"- control_mean_stage3_score: {control_row['mean_stage3_score']}",
		f"- control_mean_env_gamma: {control_row['mean_env_gamma']}",
		f"- control_level3_seed_count: {control_row['level3_seed_count']}",
		f"- best_commitment: {'' if best_row is None else best_row['condition']}",
		f"- best_leader_action: {'' if best_row is None else best_row['leader_action']}",
		"",
		"## Commitment Verdicts",
		"",
		"| Cell | leader_action | Level 3 seeds | mean_stage3_score | mean_env_gamma | gamma_uplift_ratio_vs_control | leader_prefers_over_control | verdict |",
		"|---|---|---:|---:|---:|---:|---|---|",
	]
	for row in combined_rows:
		lines.append(
			f"| {row['condition']} | {row['leader_action']} | {row['level3_seed_count']} | {row['mean_stage3_score']} | {row['mean_env_gamma']} | {row['gamma_uplift_ratio_vs_control']} | {row['leader_prefers_over_control']} | {row['verdict']} |"
		)
	lines.extend(
		[
			"",
			"## Gate",
			"",
			"- pass: 至少 1 個 non-control cell 同時滿足 level3_seed_count >= 1、mean_env_gamma >= 0、且 mean_stage3_score 不低於 control。",
			"- close_w3_1: 若所有 non-control leader commitments 都沒有任何 Level 3 seed，則 W3.1 直接結案，不再做 a/b/cross 細網格微調。",
			"",
			"## Interpretation",
			"",
			"- leader_prefers_over_control 使用固定 lexicographic 排序：先比 Level 3、再比 stage3、再比 env_gamma、最後比 turn strength。",
			"- best commitment 只用來指定唯一下一個更高優先的 leader policy 候選，不等於自動通過 promotion gate。",
		]
	)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decide_overall(combined_rows: list[dict[str, Any]]) -> str:
	candidate_rows = [row for row in combined_rows if row["is_control"] == "no"]
	if any(row["promotion_gate_pass"] == "yes" for row in candidate_rows):
		return "pass"
	if all(int(row["level3_seed_count"]) == 0 for row in candidate_rows):
		return "close_w3_1"
	if any(row["leader_prefers_over_control"] == "yes" for row in candidate_rows):
		return "weak_positive"
	return "fail"


def run_w3_scout(
	*,
	seeds: list[int],
	out_root: Path,
	summary_tsv: Path,
	combined_tsv: Path,
	decision_md: Path,
	conditions: list[str] | None = None,
	events_json: Path = DEFAULT_FULL_EVENTS_JSON,
	players: int = 300,
	rounds: int = 3000,
	selection_strength: float = 0.06,
	init_bias: float = 0.12,
	memory_kernel: int = 3,
	burn_in: int = 1000,
	tail: int = 1000,
	cell_runner: CellRunner | None = None,
) -> dict[str, Any]:
	runner = cell_runner or run_w3_cell
	selected_conditions = set(str(value) for value in conditions) if conditions is not None else None
	out_root.mkdir(parents=True, exist_ok=True)
	cell_configs = [
		_build_cell_config(
			commitment=commitment,
			players=players,
			rounds=rounds,
			events_json=events_json,
			selection_strength=selection_strength,
			init_bias=init_bias,
			memory_kernel=memory_kernel,
			burn_in=burn_in,
			tail=tail,
			out_root=out_root,
		)
		for commitment in _default_commitments()
	]
	if selected_conditions is not None:
		cell_configs = [config for config in cell_configs if config.condition in selected_conditions]
		if not cell_configs:
			raise ValueError("no W3 cells selected")
	summary_rows: list[dict[str, Any]] = []
	by_condition: dict[str, list[dict[str, Any]]] = {config.condition: [] for config in cell_configs}
	for config in cell_configs:
		for seed in seeds:
			row = dict(runner(config, int(seed)))
			summary_rows.append(row)
			by_condition[config.condition].append(row)
	_write_tsv(summary_tsv, fieldnames=SUMMARY_FIELDNAMES, rows=summary_rows)
	control_config = next((config for config in cell_configs if config.condition == "control"), None)
	control_rows = by_condition.get("control", [])
	control_base = None
	combined_rows: list[dict[str, Any]] = []
	if control_config is not None:
		control_base = _cell_summary(control_rows, control_row=None, config=control_config)
		combined_rows.append(control_base)
	for config in cell_configs:
		if config.condition == "control":
			continue
		combined_rows.append(_cell_summary(by_condition[config.condition], control_row=control_base, config=config))
	if control_config is not None:
		combined_rows[0] = _cell_summary(control_rows, control_row=combined_rows[0], config=control_config)
	best_condition = select_best_commitment(combined_rows)
	for row in combined_rows:
		row["is_best_commitment"] = _yes_no((best_condition is not None) and row["condition"] == best_condition)
	_write_tsv(combined_tsv, fieldnames=COMBINED_FIELDNAMES, rows=combined_rows)
	if len(combined_rows) == 1 and combined_rows[0]["is_control"] == "yes":
		decision = "control_only"
		control_row = combined_rows[0]
	else:
		decision = _decide_overall(combined_rows)
		control_row = next(row for row in combined_rows if row["is_control"] == "yes")
	_write_decision(
		decision_md,
		decision=decision,
		control_row=control_row,
		combined_rows=combined_rows,
		best_condition=best_condition,
	)
	return {
		"decision": decision,
		"summary_rows": summary_rows,
		"combined_rows": combined_rows,
	}


def main(*, cell_runner: CellRunner | None = None) -> int:
	parser = argparse.ArgumentParser(description="W3.1 minimal Stackelberg formal scout")
	parser.add_argument("--conditions", type=str, default="control,w3_cross_strong,w3_edge_tilt,w3_commit_push")
	parser.add_argument("--seeds", type=str, default="45,47,49")
	parser.add_argument("--players", type=int, default=300)
	parser.add_argument("--rounds", type=int, default=3000)
	parser.add_argument("--selection-strength", type=float, default=0.06)
	parser.add_argument("--init-bias", type=float, default=0.12)
	parser.add_argument("--memory-kernel", type=int, default=3)
	parser.add_argument("--burn-in", type=int, default=1000)
	parser.add_argument("--tail", type=int, default=1000)
	parser.add_argument("--events-json", type=Path, default=DEFAULT_FULL_EVENTS_JSON)
	parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
	parser.add_argument("--summary-tsv", type=Path, default=DEFAULT_SUMMARY_TSV)
	parser.add_argument("--combined-tsv", type=Path, default=DEFAULT_COMBINED_TSV)
	parser.add_argument("--decision-md", type=Path, default=DEFAULT_DECISION_MD)
	args = parser.parse_args()
	run_w3_scout(
		conditions=[part.strip() for part in str(args.conditions).split(",") if part.strip()],
		seeds=_parse_seeds(args.seeds),
		out_root=args.out_root,
		summary_tsv=args.summary_tsv,
		combined_tsv=args.combined_tsv,
		decision_md=args.decision_md,
		events_json=args.events_json,
		players=int(args.players),
		rounds=int(args.rounds),
		selection_strength=float(args.selection_strength),
		init_bias=float(args.init_bias),
		memory_kernel=int(args.memory_kernel),
		burn_in=int(args.burn_in),
		tail=int(args.tail),
		cell_runner=cell_runner,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())