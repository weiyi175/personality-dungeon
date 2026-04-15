from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Mapping

from simulation.personality_coupling import summarize_personality_coupling
from simulation.personality_gate0 import DEFAULT_GATE0_EVENTS_JSON, PROTOTYPES, projected_initial_weights, sample_personality
from simulation.personality_gate1 import run_condition


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "h71_personality_coupled"
DEFAULT_SUMMARY_TSV = REPO_ROOT / "outputs" / "h71_personality_coupled_summary.tsv"
DEFAULT_COMBINED_TSV = REPO_ROOT / "outputs" / "h71_personality_coupled_combined.tsv"
DEFAULT_DECISION_MD = REPO_ROOT / "outputs" / "h71_personality_coupled_decision.md"

STRATEGY_ORDER = ("aggressive", "defensive", "balanced")
CELL_SPECS = [
	{"condition": "control", "lambda_mu": 0.00, "lambda_k": 0.00},
	{"condition": "inertia_only", "lambda_mu": 0.25, "lambda_k": 0.00},
	{"condition": "k_only", "lambda_mu": 0.00, "lambda_k": 0.25},
	{"condition": "combined_low", "lambda_mu": 0.15, "lambda_k": 0.15},
]

METRIC_FIELDNAMES = [
	"condition",
	"lambda_mu",
	"lambda_k",
	"mu_base",
	"k_base",
	"apply_event_trait_deltas",
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
	"personality_signal_mu_min",
	"personality_signal_mu_max",
	"personality_signal_mu_mean",
	"personality_signal_k_min",
	"personality_signal_k_max",
	"personality_signal_k_mean",
	"mu_player_min",
	"mu_player_max",
	"mu_player_mean",
	"k_player_min",
	"k_player_max",
	"k_player_mean",
	"out_csv",
	"provenance_json",
]

COMBINED_FIELDNAMES = [
	"condition",
	"lambda_mu",
	"lambda_k",
	"mu_base",
	"k_base",
	"apply_event_trait_deltas",
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
	"stage3_uplift_vs_control",
	"primary_pass",
	"secondary_pass",
	"longer_confirm_candidate",
	"verdict",
	"events_json",
	"personality_mode",
	"personality_jitter",
	"selection_strength",
	"memory_kernel",
	"out_dir",
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


def _relative_gain_pass(*, baseline: float, candidate: float, fraction: float) -> bool:
	base = float(baseline)
	alt = float(candidate)
	frac = float(fraction)
	if base > 0.0:
		return alt >= (1.0 + frac) * base
	if base < 0.0:
		return alt >= (1.0 - frac) * base
	return alt > 0.0


def _closer_to_zero(*, baseline: float, candidate: float) -> bool:
	return abs(float(candidate)) < abs(float(baseline))


def _gamma_primary_pass(*, baseline: float, candidate: float) -> bool:
	return _relative_gain_pass(baseline=baseline, candidate=candidate, fraction=0.25) or _closer_to_zero(
		baseline=baseline,
		candidate=candidate,
	)


def _gamma_secondary_pass(*, baseline: float, candidate: float) -> bool:
	return _relative_gain_pass(baseline=baseline, candidate=candidate, fraction=0.15)


def _sample_h71_personalities(*, n_players: int, jitter: float, seed: int) -> list[dict[str, float]]:
	if int(n_players) % 3 != 0:
		raise ValueError("H7.1 requires n_players divisible by 3 for the fixed 1:1:1 world")
	per_group = int(n_players) // 3
	rng = random.Random(int(seed))
	personalities: list[dict[str, float]] = []
	for strategy in STRATEGY_ORDER:
		prototype = PROTOTYPES[strategy]
		for _ in range(per_group):
			personalities.append(sample_personality(prototype, jitter=float(jitter), rng=rng))
	return personalities


def _build_player_setup_callback(*, personalities: list[Mapping[str, float]]) -> Any:
	def _setup(players: list[object], _strategy_space: list[str], cfg: Any) -> None:
		if len(players) != len(personalities):
			raise RuntimeError("personality count mismatch in H7.1 setup")
		for player, personality in zip(players, personalities, strict=True):
			player.personality = dict(personality)
			player.update_weights(projected_initial_weights(personality, init_bias=float(cfg.init_bias)))

	return _setup


def _level_counts(metrics_rows: list[dict[str, Any]]) -> dict[int, int]:
	levels = [int(row["cycle_level"]) for row in metrics_rows]
	return {level: levels.count(level) for level in range(4)}


def _summary_rank_key(row: Mapping[str, Any]) -> tuple[int, float, int, float]:
	return (
		1 if str(row["primary_pass"]) == "yes" else 0,
		float(row["stage3_uplift_vs_control"]),
		int(row["level3_seed_count"]),
		-float(abs(float(row["mean_env_gamma"]))),
	)


def _write_decision(
	path: Path,
	*,
	decision: str,
	control_row: Mapping[str, Any],
	combined_rows: list[dict[str, Any]],
	confirm_candidate: Mapping[str, Any] | None,
) -> None:
	lines = [
		"# H7.1 Decision",
		"",
		f"- decision: {decision}",
		f"- control_mean_env_gamma: {control_row['mean_env_gamma']}",
		f"- control_mean_stage3_score: {control_row['mean_stage3_score']}",
		f"- control_level3_seed_count: {control_row['level3_seed_count']}",
	]
	if confirm_candidate is None:
		lines.append("- longer_confirm_candidate: none")
	else:
		lines.extend(
			[
				f"- longer_confirm_candidate: {confirm_candidate['condition']}",
				f"- candidate_primary_pass: {confirm_candidate['primary_pass']}",
				f"- candidate_secondary_pass: {confirm_candidate['secondary_pass']}",
				f"- candidate_stage3_uplift_vs_control: {confirm_candidate['stage3_uplift_vs_control']}",
				f"- candidate_gamma_uplift_ratio_vs_control: {confirm_candidate['gamma_uplift_ratio_vs_control']}",
			]
		)
	lines.extend(["", "## Cell Verdicts", ""])
	for row in combined_rows:
		lines.append(
			f"- {row['condition']}: lambda_mu={row['lambda_mu']} lambda_k={row['lambda_k']} level3_seed_count={row['level3_seed_count']} stage3_uplift={row['stage3_uplift_vs_control']} gamma_ratio={row['gamma_uplift_ratio_vs_control']} primary={row['primary_pass']} secondary={row['secondary_pass']} verdict={row['verdict']}"
		)
	lines.extend(
		[
			"",
			"## Stop Rule",
			"",
			"- 若 4 個固定 cells 全部沒有 Level 3 seed，且整體只剩 weak_positive，H7.1 直接結案為 close_h71。",
			"- single longer confirm 最多只允許 1 個 candidate。",
		]
	)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_h71_scout(
	*,
	seeds: list[int],
	out_root: Path,
	summary_tsv: Path,
	combined_tsv: Path,
	decision_md: Path,
	events_json: Path,
	players: int = 300,
	rounds: int = 3000,
	selection_strength: float = 0.06,
	init_bias: float = 0.12,
	memory_kernel: int = 3,
	burn_in: int = 1000,
	tail: int = 1000,
	jitter: float = 0.08,
	a: float = 1.0,
	b: float = 0.9,
	cross: float = 0.20,
	mu_base: float = 0.0,
) -> dict[str, Any]:
	out_root.mkdir(parents=True, exist_ok=True)
	cell_metrics: dict[str, list[dict[str, Any]]] = {}
	cell_summaries: dict[str, dict[str, Any]] = {}
	control_by_seed: dict[int, dict[str, Any]] | None = None

	for spec in CELL_SPECS:
		condition = str(spec["condition"])
		lambda_mu = float(spec["lambda_mu"])
		lambda_k = float(spec["lambda_k"])
		seed_stats: dict[int, dict[str, float]] = {}

		def _factory(seed: int, *, _lambda_mu: float = lambda_mu, _lambda_k: float = lambda_k) -> Any:
			personalities = _sample_h71_personalities(n_players=int(players), jitter=float(jitter), seed=int(seed))
			seed_stats[int(seed)] = summarize_personality_coupling(
				personalities,
				mu_base=float(mu_base),
				lambda_mu=float(_lambda_mu),
				k_base=float(selection_strength),
				lambda_k=float(_lambda_k),
			)
			return _build_player_setup_callback(personalities=personalities)

		metrics_rows, summary_row, _seed_csvs, _action_counts, _seed_diagnostics = run_condition(
			condition=condition,
			world_mode="h71_equal_111",
			seeds=list(seeds),
			out_dir=out_root / condition,
			events_json=events_json,
			players=int(players),
			rounds=int(rounds),
			selection_strength=float(selection_strength),
			init_bias=float(init_bias),
			memory_kernel=int(memory_kernel),
			burn_in=int(burn_in),
			tail=int(tail),
			jitter=float(jitter),
			a=float(a),
			b=float(b),
			cross=float(cross),
			evolution_mode="personality_coupled",
			player_setup_callback_factory=_factory,
			sim_config_overrides={
				"apply_event_trait_deltas": False,
				"personality_coupling_mu_base": float(mu_base),
				"personality_coupling_lambda_mu": float(lambda_mu),
				"personality_coupling_lambda_k": float(lambda_k),
			},
		)

		annotated_metrics: list[dict[str, Any]] = []
		for row in metrics_rows:
			seed = int(row["seed"])
			stats = seed_stats[seed]
			annotated_metrics.append(
				{
					**row,
					"condition": condition,
					"lambda_mu": _format_float(lambda_mu),
					"lambda_k": _format_float(lambda_k),
					"mu_base": _format_float(mu_base),
					"k_base": _format_float(selection_strength),
					"apply_event_trait_deltas": "no",
					"control_env_gamma": "",
					"control_stage3_score": "",
					"gamma_uplift_ratio_vs_control_seed": "",
					"stage3_uplift_vs_control_seed": "",
					"has_level3_seed": _yes_no(int(row["cycle_level"]) >= 3),
					"personality_signal_mu_min": _format_float(stats["signal_mu_min"]),
					"personality_signal_mu_max": _format_float(stats["signal_mu_max"]),
					"personality_signal_mu_mean": _format_float(stats["signal_mu_mean"]),
					"personality_signal_k_min": _format_float(stats["signal_k_min"]),
					"personality_signal_k_max": _format_float(stats["signal_k_max"]),
					"personality_signal_k_mean": _format_float(stats["signal_k_mean"]),
					"mu_player_min": _format_float(stats["mu_min"]),
					"mu_player_max": _format_float(stats["mu_max"]),
					"mu_player_mean": _format_float(stats["mu_mean"]),
					"k_player_min": _format_float(stats["k_min"]),
					"k_player_max": _format_float(stats["k_max"]),
					"k_player_mean": _format_float(stats["k_mean"]),
				}
			)
		cell_metrics[condition] = annotated_metrics
		cell_summaries[condition] = {
			**summary_row,
			"condition": condition,
			"lambda_mu": _format_float(lambda_mu),
			"lambda_k": _format_float(lambda_k),
			"mu_base": _format_float(mu_base),
			"k_base": _format_float(selection_strength),
			"apply_event_trait_deltas": "no",
		}
		if condition == "control":
			control_by_seed = {int(row["seed"]): row for row in annotated_metrics}

	if control_by_seed is None:
		raise RuntimeError("missing H7.1 control metrics")

	control_summary = cell_summaries["control"]
	control_mean_env_gamma = float(control_summary["mean_env_gamma"])
	control_mean_stage3_score = float(control_summary["mean_stage3_score"])

	summary_rows: list[dict[str, Any]] = []
	combined_rows: list[dict[str, Any]] = []
	for spec in CELL_SPECS:
		condition = str(spec["condition"])
		metrics_rows = cell_metrics[condition]
		for row in metrics_rows:
			if condition == "control":
				row["control_env_gamma"] = _format_float(float(row["env_gamma"]))
				row["control_stage3_score"] = _format_float(float(row["stage3_score"]))
				row["gamma_uplift_ratio_vs_control_seed"] = _format_ratio(1.0)
				row["stage3_uplift_vs_control_seed"] = _format_float(0.0)
			else:
				control_row = control_by_seed[int(row["seed"])]
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

		level_counts = _level_counts(metrics_rows)
		level3_seed_count = sum(1 for row in metrics_rows if int(row["cycle_level"]) >= 3)
		mean_stage3_score = float(cell_summaries[condition]["mean_stage3_score"])
		mean_env_gamma = float(cell_summaries[condition]["mean_env_gamma"])
		stage3_uplift = mean_stage3_score - control_mean_stage3_score
		gamma_ratio = _gamma_uplift_ratio(
			baseline_gamma=control_mean_env_gamma,
			candidate_gamma=mean_env_gamma,
		)
		primary_pass = False
		secondary_pass = False
		verdict = "control" if condition == "control" else "fail"
		if condition != "control":
			primary_pass = (
				level3_seed_count >= 1
				and stage3_uplift >= 0.020
				and _gamma_primary_pass(baseline=control_mean_env_gamma, candidate=mean_env_gamma)
			)
			secondary_pass = (
				_relative_gain_pass(
					baseline=control_mean_stage3_score,
					candidate=mean_stage3_score,
					fraction=0.15,
				)
				and _gamma_secondary_pass(baseline=control_mean_env_gamma, candidate=mean_env_gamma)
			)
			if primary_pass:
				verdict = "primary_pass"
			elif secondary_pass:
				verdict = "secondary_pass"
			elif stage3_uplift > 0.0 or _closer_to_zero(baseline=control_mean_env_gamma, candidate=mean_env_gamma):
				verdict = "weak_positive"

		combined_rows.append(
			{
				**cell_summaries[condition],
				"n_seeds": len(metrics_rows),
				"level_counts_json": json.dumps(level_counts, sort_keys=True),
				"level3_seed_count": int(level3_seed_count),
				"control_mean_env_gamma": _format_float(control_mean_env_gamma),
				"control_mean_stage3_score": _format_float(control_mean_stage3_score),
				"gamma_uplift_ratio_vs_control": _format_ratio(gamma_ratio),
				"stage3_uplift_vs_control": _format_float(stage3_uplift),
				"primary_pass": _yes_no(primary_pass),
				"secondary_pass": _yes_no(secondary_pass),
				"longer_confirm_candidate": "no",
				"verdict": verdict,
				"selection_strength": _format_float(selection_strength),
				"memory_kernel": int(memory_kernel),
			}
		)

	candidate_rows = [row for row in combined_rows if str(row["condition"]) != "control"]
	primary_candidates = [row for row in candidate_rows if str(row["primary_pass"]) == "yes"]
	secondary_candidates = [row for row in candidate_rows if str(row["secondary_pass"]) == "yes"]
	any_level3_seed = any(int(row["level3_seed_count"]) > 0 for row in candidate_rows)
	confirm_candidate: dict[str, Any] | None = None
	decision = "fail"
	if primary_candidates:
		confirm_candidate = max(primary_candidates, key=_summary_rank_key)
		decision = "pass"
	elif len(secondary_candidates) == 1:
		confirm_candidate = secondary_candidates[0]
		decision = "close_h71" if not any_level3_seed else "weak_positive"
	elif len(secondary_candidates) > 1:
		decision = "close_h71" if not any_level3_seed else "weak_positive"
	elif any(str(row["verdict"]) == "weak_positive" for row in candidate_rows):
		decision = "close_h71" if not any_level3_seed else "weak_positive"

	if confirm_candidate is not None:
		for row in combined_rows:
			if str(row["condition"]) == str(confirm_candidate["condition"]):
				row["longer_confirm_candidate"] = "yes"
				break

	_write_tsv(summary_tsv, fieldnames=METRIC_FIELDNAMES, rows=summary_rows)
	_write_tsv(combined_tsv, fieldnames=COMBINED_FIELDNAMES, rows=combined_rows)
	_write_decision(
		decision_md,
		decision=decision,
		control_row=control_summary | {"level3_seed_count": int(combined_rows[0]["level3_seed_count"])},
		combined_rows=combined_rows,
		confirm_candidate=confirm_candidate,
	)
	return {
		"decision": decision,
		"summary_tsv": str(summary_tsv),
		"combined_tsv": str(combined_tsv),
		"decision_md": str(decision_md),
		"confirm_candidate": None if confirm_candidate is None else str(confirm_candidate["condition"]),
		"combined_rows": combined_rows,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Run the H7.1 personality-coupled 4-cell scout")
	parser.add_argument("--seeds", type=str, default="45,47,49")
	parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
	parser.add_argument("--summary-tsv", type=Path, default=DEFAULT_SUMMARY_TSV)
	parser.add_argument("--combined-tsv", type=Path, default=DEFAULT_COMBINED_TSV)
	parser.add_argument("--decision-md", type=Path, default=DEFAULT_DECISION_MD)
	parser.add_argument("--events-json", type=Path, default=DEFAULT_GATE0_EVENTS_JSON)
	parser.add_argument("--players", type=int, default=300)
	parser.add_argument("--rounds", type=int, default=3000)
	parser.add_argument("--selection-strength", type=float, default=0.06)
	parser.add_argument("--init-bias", type=float, default=0.12)
	parser.add_argument("--memory-kernel", type=int, default=3)
	parser.add_argument("--burn-in", type=int, default=1000)
	parser.add_argument("--tail", type=int, default=1000)
	parser.add_argument("--jitter", type=float, default=0.08)
	parser.add_argument("--a", type=float, default=1.0)
	parser.add_argument("--b", type=float, default=0.9)
	parser.add_argument("--cross", type=float, default=0.20)
	parser.add_argument("--mu-base", type=float, default=0.0)
	args = parser.parse_args()

	result = run_h71_scout(
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
		jitter=float(args.jitter),
		a=float(args.a),
		b=float(args.b),
		cross=float(args.cross),
		mu_base=float(args.mu_base),
	)
	print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()