from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping

from analysis.cycle_metrics import classify_cycle_level
from analysis.decay_rate import estimate_decay_gamma
from analysis.event_provenance_summary import summarize_event_provenance
from simulation.personality_gate0 import (
	DEFAULT_FULL_EVENTS_JSON,
	DEFAULT_GATE0_EVENTS_JSON,
	PROTOTYPES,
	build_cohorts,
	projected_initial_weights,
	write_reduced_events_json,
	zero_personality,
)
from simulation.run_simulation import SimConfig, _write_timeseries_csv, simulate


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_DIR = REPO_ROOT / "outputs" / "personality_gate1_baseline"
DEFAULT_HETERO_DIR = REPO_ROOT / "outputs" / "personality_gate1_heterogeneous"
DEFAULT_SUMMARY_TSV = REPO_ROOT / "outputs" / "personality_gate1_summary.tsv"
DEFAULT_COMBINED_TSV = REPO_ROOT / "outputs" / "personality_gate1_combined.tsv"
DEFAULT_DECISION_MD = REPO_ROOT / "outputs" / "personality_gate1_decision.md"

DEFAULT_H54_OUT_ROOT = REPO_ROOT / "outputs" / "h54_personality_inertia"
DEFAULT_H54_SUMMARY_TSV = REPO_ROOT / "outputs" / "h54_personality_inertia_summary.tsv"
DEFAULT_H54_COMBINED_TSV = REPO_ROOT / "outputs" / "h54_personality_inertia_combined.tsv"
DEFAULT_H54_DECISION_MD = REPO_ROOT / "outputs" / "h54_personality_inertia_decision.md"

DEFAULT_H55_OUT_ROOT = REPO_ROOT / "outputs" / "h55_personality_nonlinear"
DEFAULT_H55_SUMMARY_TSV = REPO_ROOT / "outputs" / "h55_personality_nonlinear_summary.tsv"
DEFAULT_H55_COMBINED_TSV = REPO_ROOT / "outputs" / "h55_personality_nonlinear_combined.tsv"
DEFAULT_H55_DECISION_MD = REPO_ROOT / "outputs" / "h55_personality_nonlinear_decision.md"

DEFAULT_H55R_OUT_ROOT = REPO_ROOT / "outputs" / "h55r_personality_nonlinear_refine"
DEFAULT_H55R_SUMMARY_TSV = REPO_ROOT / "outputs" / "h55r_personality_nonlinear_refine_summary.tsv"
DEFAULT_H55R_COMBINED_TSV = REPO_ROOT / "outputs" / "h55r_personality_nonlinear_refine_combined.tsv"
DEFAULT_H55R_DECISION_MD = REPO_ROOT / "outputs" / "h55r_personality_nonlinear_refine_decision.md"

METRIC_FIELDNAMES = [
	"condition",
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

CONDITION_SUMMARY_FIELDNAMES = [
	"condition",
	"n_seeds",
	"mean_cycle_level",
	"mean_stage3_score",
	"mean_turn_strength",
	"mean_env_gamma",
	"mean_success_rate",
	"level_counts_json",
	"p_level_ge_2",
	"p_level_3",
	"events_json",
	"personality_mode",
	"personality_jitter",
	"out_dir",
]

H54_METRIC_FIELDNAMES = [
	"condition",
	"sampled_inertia",
	"is_control",
	"is_tiebreak",
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
	"verdict",
	"out_csv",
	"provenance_json",
]

H54_COMBINED_FIELDNAMES = [
	"condition",
	"sampled_inertia",
	"is_control",
	"is_tiebreak",
	"n_seeds",
	"mean_cycle_level",
	"mean_stage3_score",
	"mean_turn_strength",
	"mean_env_gamma",
	"mean_success_rate",
	"level_counts_json",
	"p_level_ge_2",
	"p_level_3",
	"control_mean_env_gamma",
	"control_mean_stage3_score",
	"gamma_uplift_ratio_vs_control",
	"stage3_uplift_vs_control",
	"gamma_pass",
	"stage3_pass",
	"level3_pass",
	"full_pass",
	"verdict",
	"events_json",
	"personality_mode",
	"personality_jitter",
	"evolution_mode",
	"selection_strength",
	"memory_kernel",
	"out_dir",
]

H55_METRIC_FIELDNAMES = [
	"condition",
	"is_control",
	"seed",
	"cycle_level",
	"stage3_score",
	"turn_strength",
	"env_gamma",
	"env_gamma_r2",
	"env_gamma_n_peaks",
	"success_rate",
	"payoff_mode",
	"threshold_trigger",
	"threshold_theta",
	"threshold_theta_low",
	"threshold_theta_high",
	"threshold_state_alpha",
	"threshold_a_hi",
	"threshold_b_hi",
	"regime_high_share",
	"regime_switches",
	"mean_threshold_state",
	"switch_active",
	"control_env_gamma",
	"control_stage3_score",
	"gamma_uplift_ratio_vs_control_seed",
	"stage3_uplift_vs_control_seed",
	"has_level3_seed",
	"verdict",
	"out_csv",
	"provenance_json",
]

H55_COMBINED_FIELDNAMES = [
	"condition",
	"is_control",
	"n_seeds",
	"mean_cycle_level",
	"mean_stage3_score",
	"mean_turn_strength",
	"mean_env_gamma",
	"mean_success_rate",
	"level_counts_json",
	"p_level_ge_2",
	"p_level_3",
	"payoff_mode",
	"threshold_trigger",
	"threshold_theta",
	"threshold_theta_low",
	"threshold_theta_high",
	"threshold_state_alpha",
	"threshold_a_hi",
	"threshold_b_hi",
	"mean_regime_high_share",
	"mean_regime_switches",
	"p_switched_seed",
	"mean_threshold_state",
	"switch_pass",
	"control_mean_env_gamma",
	"control_mean_stage3_score",
	"gamma_uplift_ratio_vs_control",
	"stage3_uplift_vs_control",
	"gamma_pass",
	"stage3_pass",
	"level3_pass",
	"full_pass",
	"verdict",
	"events_json",
	"personality_mode",
	"personality_jitter",
	"selection_strength",
	"memory_kernel",
	"out_dir",
]

H55R_METRIC_FIELDNAMES = list(H55_METRIC_FIELDNAMES)

H55R_COMBINED_FIELDNAMES = [
	"condition",
	"is_control",
	"n_seeds",
	"mean_cycle_level",
	"mean_stage3_score",
	"mean_turn_strength",
	"mean_env_gamma",
	"mean_success_rate",
	"level_counts_json",
	"p_level_ge_2",
	"p_level_3",
	"payoff_mode",
	"threshold_trigger",
	"threshold_theta",
	"threshold_theta_low",
	"threshold_theta_high",
	"threshold_state_alpha",
	"threshold_a_hi",
	"threshold_b_hi",
	"mean_regime_high_share",
	"mean_regime_switches",
	"p_switched_seed",
	"mean_threshold_state",
	"switch_pass",
	"control_mean_env_gamma",
	"control_mean_stage3_score",
	"gamma_uplift_ratio_vs_control",
	"stage3_uplift_vs_control",
	"gamma_refine_pass",
	"stage3_refine_pass",
	"confirm_gate_pass",
	"confirm_rank",
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


def _parse_float_list(spec: str) -> list[float]:
	parts = [part.strip() for part in str(spec).split(",") if part.strip()]
	if not parts:
		raise ValueError("float list cannot be empty")
	return [float(part) for part in parts]


def _format_mu(value: float) -> str:
	return f"{float(value):.2f}"


def _format_ratio(value: float) -> str:
	if value == float("inf"):
		return "inf"
	if value == float("-inf"):
		return "-inf"
	return f"{float(value):.6f}"


def _yes_no(value: bool) -> str:
	return "yes" if bool(value) else "no"


def _format_optional_float(value: float | None) -> str:
	if value is None:
		return ""
	return f"{float(value):.6f}"


def _parse_ratio(value: Any) -> float:
	text = str(value).strip().lower()
	if text == "inf":
		return float("inf")
	if text == "-inf":
		return float("-inf")
	return float(text)


def _series_from_rows(rows: list[dict[str, Any]], *, prefix: str) -> dict[str, list[float]]:
	return {
		"aggressive": [float(row[f"{prefix}aggressive"]) for row in rows],
		"defensive": [float(row[f"{prefix}defensive"]) for row in rows],
		"balanced": [float(row[f"{prefix}balanced"]) for row in rows],
	}


def _safe_mean(values: list[float]) -> float | None:
	if not values:
		return None
	return float(sum(values) / float(len(values)))


def _aggregate_action_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
	counts: dict[str, dict[str, int]] = {}
	for row in rows:
		event_ids = json.loads(str(row.get("event_ids_json") or "[]"))
		action_names = json.loads(str(row.get("action_names_json") or "[]"))
		for event_id, action_name in zip(event_ids, action_names):
			event_bucket = counts.setdefault(str(event_id), {})
			event_bucket[str(action_name)] = int(event_bucket.get(str(action_name), 0)) + 1
	return counts


def _normalize_action_counts(action_counts: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, float]]:
	shares: dict[str, dict[str, float]] = {}
	for event_id, counts in action_counts.items():
		total = float(sum(int(value) for value in counts.values()))
		if total <= 0.0:
			shares[str(event_id)] = {}
			continue
		shares[str(event_id)] = {
			str(action_name): float(count) / total
			for action_name, count in counts.items()
		}
	return shares


def _parse_optional_bool(value: Any) -> bool | None:
	if value is None:
		return None
	text = str(value).strip().lower()
	if text == "":
		return None
	if text in {"1", "true", "yes"}:
		return True
	if text in {"0", "false", "no"}:
		return False
	raise ValueError(f"unsupported boolean-like value: {value!r}")


def _threshold_seed_diagnostics(rows: list[dict[str, Any]]) -> dict[str, float | int | bool | None]:
	regime_values: list[float] = []
	state_values: list[float] = []
	switches = 0
	previous_regime: bool | None = None
	for row in rows:
		regime = _parse_optional_bool(row.get("threshold_regime_hi"))
		if regime is not None:
			regime_values.append(1.0 if regime else 0.0)
			if previous_regime is not None and regime != previous_regime:
				switches += 1
			previous_regime = regime
		state_raw = str(row.get("threshold_state_value") or "").strip()
		if state_raw != "":
			state_values.append(float(state_raw))
	return {
		"regime_high_share": _safe_mean(regime_values),
		"regime_switches": switches if regime_values else None,
		"mean_threshold_state": _safe_mean(state_values),
		"switch_active": bool(switches > 0),
	}


def _aggregate_threshold_diagnostics(seed_diagnostics: Mapping[int, Mapping[str, Any]]) -> dict[str, float | None]:
	high_shares = [
		float(diag["regime_high_share"])
		for diag in seed_diagnostics.values()
		if diag.get("regime_high_share") is not None
	]
	switch_counts = [
		float(diag["regime_switches"])
		for diag in seed_diagnostics.values()
		if diag.get("regime_switches") is not None
	]
	state_values = [
		float(diag["mean_threshold_state"])
		for diag in seed_diagnostics.values()
		if diag.get("mean_threshold_state") is not None
	]
	den = float(len(seed_diagnostics)) or 1.0
	switched = sum(1 for diag in seed_diagnostics.values() if bool(diag.get("switch_active")))
	return {
		"mean_regime_high_share": _safe_mean(high_shares),
		"mean_regime_switches": _safe_mean(switch_counts),
		"p_switched_seed": float(switched) / den,
		"mean_threshold_state": _safe_mean(state_values),
	}


def _world_personalities(
	*,
	world_mode: str,
	n_players: int,
	jitter: float,
	seed: int,
) -> list[dict[str, float]]:
	if str(world_mode) == "zero":
		return [zero_personality() for _ in range(int(n_players))]
	if str(world_mode) != "heterogeneous":
		raise ValueError(f"unsupported world_mode: {world_mode}")
	if int(n_players) % 3 != 0:
		raise ValueError("heterogeneous world currently requires n_players divisible by 3")
	cohorts = build_cohorts(
		cohort_size=int(n_players) // 3,
		jitter=float(jitter),
		seed=int(seed),
		prototypes=PROTOTYPES,
	)
	personalities: list[dict[str, float]] = []
	for cohort in ("aggressive", "defensive", "balanced"):
		personalities.extend(cohorts[cohort])
	return personalities


def build_player_setup_callback(
	*,
	world_mode: str,
	jitter: float,
	personality_seed: int,
) -> Any:
	def _setup(players: list[object], strategy_space: list[str], cfg: SimConfig) -> None:
		personalities = _world_personalities(
			world_mode=str(world_mode),
			n_players=len(players),
			jitter=float(jitter),
			seed=int(personality_seed),
		)
		for player, personality in zip(players, personalities, strict=True):
			player.personality = dict(personality)
			player.update_weights(projected_initial_weights(personality, init_bias=float(cfg.init_bias)))

	return _setup


def seed_metrics(
	rows: list[dict[str, Any]],
	*,
	burn_in: int,
	tail: int,
	eta: float,
	corr_threshold: float,
) -> dict[str, float | int]:
	series_map = _series_from_rows(rows, prefix="p_")
	cycle = classify_cycle_level(
		series_map,
		burn_in=int(burn_in),
		tail=int(tail),
		amplitude_threshold=0.02,
		corr_threshold=float(corr_threshold),
		eta=float(eta),
		stage3_method="turning",
		phase_smoothing=1,
	)
	fit = estimate_decay_gamma(series_map, series_kind="p")
	success_rates = []
	for row in rows:
		event_count = float(row.get("event_count") or 0.0)
		if event_count <= 0.0:
			continue
		ss = float(row.get("success_count") or 0.0) / event_count
		success_rates.append(ss)
	return {
		"cycle_level": int(cycle.level),
		"stage3_score": float(cycle.stage3.score) if cycle.stage3 is not None else 0.0,
		"turn_strength": float(cycle.stage3.turn_strength) if cycle.stage3 is not None else 0.0,
		"env_gamma": float(fit.gamma) if fit is not None else 0.0,
		"env_gamma_r2": float(fit.r2) if fit is not None else 0.0,
		"env_gamma_n_peaks": int(fit.n_peaks) if fit is not None else 0,
		"success_rate": float(_safe_mean(success_rates) or 0.0),
	}


def gamma_improvement_flags(*, baseline_gamma: float, hetero_gamma: float) -> dict[str, bool]:
	base = float(baseline_gamma)
	alt = float(hetero_gamma)
	if base == 0.0:
		return {
			"gamma_relative_20pct": alt >= 0.0,
			"gamma_closer_to_zero": abs(alt) <= abs(base),
			"gamma_pass": alt >= 0.0,
		}
	relative = alt >= (base + 0.2 * abs(base))
	closer = abs(alt) <= 0.8 * abs(base)
	return {
		"gamma_relative_20pct": bool(relative),
		"gamma_closer_to_zero": bool(closer),
		"gamma_pass": bool(relative or closer),
	}


def _gamma_improvement_25pct(*, baseline_gamma: float, candidate_gamma: float) -> bool:
	return float(candidate_gamma) >= float(baseline_gamma) + 0.25 * abs(float(baseline_gamma))


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


def decide_gate1(
	*,
	baseline_summary: Mapping[str, Any],
	hetero_summary: Mapping[str, Any],
) -> dict[str, Any]:
	baseline_gamma = float(baseline_summary.get("mean_env_gamma") or 0.0)
	hetero_gamma = float(hetero_summary.get("mean_env_gamma") or 0.0)
	baseline_stage3 = float(baseline_summary.get("mean_stage3_score") or 0.0)
	hetero_stage3 = float(hetero_summary.get("mean_stage3_score") or 0.0)
	hetero_level_ge_2 = float(hetero_summary.get("p_level_ge_2") or 0.0)
	hetero_level_3 = float(hetero_summary.get("p_level_3") or 0.0)
	flags = gamma_improvement_flags(baseline_gamma=baseline_gamma, hetero_gamma=hetero_gamma)
	stage3_pass = hetero_stage3 >= baseline_stage3 + 0.020
	level_pass = hetero_level_ge_2 >= (2.0 / 3.0) and hetero_level_3 >= (1.0 / 3.0)
	passed_checks = int(flags["gamma_pass"]) + int(stage3_pass) + int(level_pass)
	if not flags["gamma_pass"]:
		decision = "fail"
	elif passed_checks == 3:
		decision = "pass"
	elif passed_checks >= 1:
		decision = "weak_positive"
	else:
		decision = "fail"
	return {
		"decision": decision,
		"gamma_pass": bool(flags["gamma_pass"]),
		"gamma_relative_20pct": bool(flags["gamma_relative_20pct"]),
		"gamma_closer_to_zero": bool(flags["gamma_closer_to_zero"]),
		"stage3_pass": bool(stage3_pass),
		"level_pass": bool(level_pass),
		"passed_checks": passed_checks,
		"baseline_mean_env_gamma": baseline_gamma,
		"heterogeneous_mean_env_gamma": hetero_gamma,
		"baseline_mean_stage3_score": baseline_stage3,
		"heterogeneous_mean_stage3_score": hetero_stage3,
	}


def _write_tsv(path: Path, *, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
		writer.writeheader()
		writer.writerows(rows)


def _condition_summary(
	*,
	condition: str,
	metrics_rows: list[dict[str, Any]],
	events_json: Path,
	out_dir: Path,
	personality_mode: str,
	personality_jitter: float,
) -> dict[str, Any]:
	levels = [int(row["cycle_level"]) for row in metrics_rows]
	level_counts = {level: levels.count(level) for level in range(4)}
	den = float(len(metrics_rows)) or 1.0
	return {
		"condition": condition,
		"n_seeds": len(metrics_rows),
		"mean_cycle_level": f"{(_safe_mean([float(level) for level in levels]) or 0.0):.6f}",
		"mean_stage3_score": f"{(_safe_mean([float(row['stage3_score']) for row in metrics_rows]) or 0.0):.6f}",
		"mean_turn_strength": f"{(_safe_mean([float(row['turn_strength']) for row in metrics_rows]) or 0.0):.6f}",
		"mean_env_gamma": f"{(_safe_mean([float(row['env_gamma']) for row in metrics_rows]) or 0.0):.6f}",
		"mean_success_rate": f"{(_safe_mean([float(row['success_rate']) for row in metrics_rows]) or 0.0):.6f}",
		"level_counts_json": json.dumps(level_counts, sort_keys=True),
		"p_level_ge_2": f"{sum(1 for level in levels if level >= 2) / den:.6f}",
		"p_level_3": f"{sum(1 for level in levels if level >= 3) / den:.6f}",
		"events_json": str(events_json),
		"personality_mode": personality_mode,
		"personality_jitter": f"{personality_jitter:.2f}",
		"out_dir": str(out_dir),
	}


def run_condition(
	*,
	condition: str,
	world_mode: str,
	seeds: list[int],
	out_dir: Path,
	events_json: Path,
	players: int,
	rounds: int,
	selection_strength: float,
	init_bias: float,
	memory_kernel: int,
	burn_in: int,
	tail: int,
	jitter: float,
	a: float,
	b: float,
	cross: float,
	payoff_mode: str = "matrix_ab",
	threshold_theta: float = 0.40,
	threshold_theta_low: float | None = None,
	threshold_theta_high: float | None = None,
	threshold_trigger: str = "ad_share",
	threshold_state_alpha: float = 1.0,
	threshold_a_hi: float | None = None,
	threshold_b_hi: float | None = None,
	evolution_mode: str = "sampled",
	sampled_inertia: float = 0.0,
	player_setup_callback: Any | None = None,
	player_setup_callback_factory: Any | None = None,
	sim_config_overrides: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[int, Path], dict[str, dict[str, float]], dict[int, dict[str, Any]]]:
	out_dir.mkdir(parents=True, exist_ok=True)
	metrics_rows: list[dict[str, Any]] = []
	seed_csvs: dict[int, Path] = {}
	action_counts_total: dict[str, dict[str, int]] = {}
	seed_diagnostics: dict[int, dict[str, Any]] = {}
	for seed in seeds:
		cfg = SimConfig(
			n_players=int(players),
			n_rounds=int(rounds),
			seed=int(seed),
			payoff_mode=str(payoff_mode),
			popularity_mode="sampled",
			gamma=0.1,
			epsilon=0.0,
			a=float(a),
			b=float(b),
			matrix_cross_coupling=float(cross),
			init_bias=float(init_bias),
			evolution_mode=str(evolution_mode),
			payoff_lag=1,
			selection_strength=float(selection_strength),
			enable_events=True,
			events_json=events_json,
			out_csv=out_dir / f"seed{seed}.csv",
			memory_kernel=int(memory_kernel),
			threshold_theta=float(threshold_theta),
			threshold_theta_low=threshold_theta_low,
			threshold_theta_high=threshold_theta_high,
			threshold_trigger=str(threshold_trigger),
			threshold_state_alpha=float(threshold_state_alpha),
			threshold_a_hi=threshold_a_hi,
			threshold_b_hi=threshold_b_hi,
			sampled_inertia=float(sampled_inertia),
		)
		if sim_config_overrides:
			cfg = replace(cfg, **dict(sim_config_overrides))
		strategy_space, rows = simulate(
			cfg,
			player_setup_callback=(
				player_setup_callback_factory(int(seed))
				if player_setup_callback_factory is not None
				else player_setup_callback
				if player_setup_callback is not None
				else build_player_setup_callback(
					world_mode=str(world_mode),
					jitter=float(jitter),
					personality_seed=int(seed),
				)
			),
		)
		_write_timeseries_csv(cfg.out_csv, strategy_space=strategy_space, rows=rows)
		seed_csvs[int(seed)] = cfg.out_csv
		seed_diagnostics[int(seed)] = _threshold_seed_diagnostics(rows)
		seed_metric = seed_metrics(rows, burn_in=int(burn_in), tail=int(tail), eta=0.55, corr_threshold=0.09)
		prov = summarize_event_provenance(cfg.out_csv, events_json=events_json)
		provenance_path = out_dir / f"seed{seed}_provenance.json"
		provenance_path.write_text(json.dumps(prov, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
		metrics_rows.append(
			{
				"condition": condition,
				"seed": int(seed),
				"cycle_level": int(seed_metric["cycle_level"]),
				"stage3_score": f"{float(seed_metric['stage3_score']):.6f}",
				"turn_strength": f"{float(seed_metric['turn_strength']):.6f}",
				"env_gamma": f"{float(seed_metric['env_gamma']):.6f}",
				"env_gamma_r2": f"{float(seed_metric['env_gamma_r2']):.6f}",
				"env_gamma_n_peaks": int(seed_metric["env_gamma_n_peaks"]),
				"success_rate": f"{float(seed_metric['success_rate']):.6f}",
				"out_csv": str(cfg.out_csv),
				"provenance_json": str(provenance_path),
			}
		)
		seed_counts = _aggregate_action_counts(rows)
		for event_id, counts in seed_counts.items():
			target = action_counts_total.setdefault(event_id, {})
			for action_name, count in counts.items():
				target[action_name] = int(target.get(action_name, 0)) + int(count)
	condition_summary = _condition_summary(
		condition=condition,
		metrics_rows=metrics_rows,
		events_json=events_json,
		out_dir=out_dir,
		personality_mode=str(world_mode),
		personality_jitter=float(jitter),
	)
	return metrics_rows, condition_summary, seed_csvs, _normalize_action_counts(action_counts_total), seed_diagnostics


def _write_gate1_decision(
	path: Path,
	*,
	decision: Mapping[str, Any],
	baseline_summary: Mapping[str, Any],
	hetero_summary: Mapping[str, Any],
	baseline_actions: Mapping[str, Mapping[str, float]],
	hetero_actions: Mapping[str, Mapping[str, float]],
) -> None:
	lines = [
		"# Personality Gate 1 Decision",
		"",
		f"- decision: {decision['decision']}",
		f"- gamma_pass: {_yes_no(decision['gamma_pass'])}",
		f"- stage3_pass: {_yes_no(decision['stage3_pass'])}",
		f"- level_pass: {_yes_no(decision['level_pass'])}",
		f"- baseline_mean_env_gamma: {decision['baseline_mean_env_gamma']:.6f}",
		f"- heterogeneous_mean_env_gamma: {decision['heterogeneous_mean_env_gamma']:.6f}",
		f"- baseline_mean_stage3_score: {decision['baseline_mean_stage3_score']:.6f}",
		f"- heterogeneous_mean_stage3_score: {decision['heterogeneous_mean_stage3_score']:.6f}",
		f"- baseline_level_counts: {baseline_summary['level_counts_json']}",
		f"- heterogeneous_level_counts: {hetero_summary['level_counts_json']}",
		"",
		"## Action Shares",
		"",
	]
	for event_id in sorted(set(baseline_actions) | set(hetero_actions)):
		lines.append(
			f"- {event_id}: baseline={json.dumps(baseline_actions.get(event_id, {}), sort_keys=True)} hetero={json.dumps(hetero_actions.get(event_id, {}), sort_keys=True)}"
		)
	lines.append("")
	lines.append("## Stop Rule")
	lines.append("")
	lines.append("- 如果 heterogeneous 未改善 env_gamma，Gate 1 直接視為 fail。")
	lines.append("- 如果只有部分條件成立，記為 weak_positive，可作為 Gate 2 offline Little Dragon 的輸入。")
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _index_metrics_by_seed(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
	return {int(row["seed"]): row for row in rows}


def _h54_point_verdict(
	*,
	control_mean_env_gamma: float,
	control_mean_stage3_score: float,
	candidate_mean_env_gamma: float,
	candidate_mean_stage3_score: float,
	p_level_3: float,
) -> dict[str, Any]:
	gamma_pass = _gamma_improvement_25pct(
		baseline_gamma=float(control_mean_env_gamma),
		candidate_gamma=float(candidate_mean_env_gamma),
	)
	stage3_pass = float(candidate_mean_stage3_score) >= float(control_mean_stage3_score) + 0.018
	level3_pass = float(p_level_3) > 0.0
	full_pass = gamma_pass and stage3_pass and level3_pass
	if full_pass:
		verdict = "pass"
	elif gamma_pass or stage3_pass or level3_pass:
		verdict = "weak_positive"
	else:
		verdict = "fail"
	return {
		"gamma_pass": gamma_pass,
		"stage3_pass": stage3_pass,
		"level3_pass": level3_pass,
		"full_pass": full_pass,
		"verdict": verdict,
	}


def _build_h54_seed_rows(
	*,
	metrics_rows: list[dict[str, Any]],
	control_metrics_rows: list[dict[str, Any]],
	sampled_inertia: float,
	is_control: bool,
	is_tiebreak: bool,
) -> list[dict[str, Any]]:
	control_by_seed = _index_metrics_by_seed(control_metrics_rows)
	rows: list[dict[str, Any]] = []
	for row in metrics_rows:
		seed = int(row["seed"])
		control_row = control_by_seed[seed]
		control_env_gamma = float(control_row["env_gamma"])
		control_stage3_score = float(control_row["stage3_score"])
		candidate_env_gamma = float(row["env_gamma"])
		candidate_stage3_score = float(row["stage3_score"])
		seed_decision = _h54_point_verdict(
			control_mean_env_gamma=control_env_gamma,
			control_mean_stage3_score=control_stage3_score,
			candidate_mean_env_gamma=candidate_env_gamma,
			candidate_mean_stage3_score=candidate_stage3_score,
			p_level_3=1.0 if int(row["cycle_level"]) >= 3 else 0.0,
		)
		if is_control:
			seed_verdict = "control"
		else:
			seed_verdict = str(seed_decision["verdict"])
		rows.append(
			{
				"condition": f"mu{_format_mu(sampled_inertia)}",
				"sampled_inertia": _format_mu(sampled_inertia),
				"is_control": _yes_no(is_control),
				"is_tiebreak": _yes_no(is_tiebreak),
				"seed": seed,
				"cycle_level": int(row["cycle_level"]),
				"stage3_score": row["stage3_score"],
				"turn_strength": row["turn_strength"],
				"env_gamma": row["env_gamma"],
				"env_gamma_r2": row["env_gamma_r2"],
				"env_gamma_n_peaks": row["env_gamma_n_peaks"],
				"success_rate": row["success_rate"],
				"control_env_gamma": f"{control_env_gamma:.6f}",
				"control_stage3_score": f"{control_stage3_score:.6f}",
				"gamma_uplift_ratio_vs_control_seed": _format_ratio(
					_gamma_uplift_ratio(baseline_gamma=control_env_gamma, candidate_gamma=candidate_env_gamma)
				),
				"stage3_uplift_vs_control_seed": f"{candidate_stage3_score - control_stage3_score:.6f}",
				"has_level3_seed": _yes_no(int(row["cycle_level"]) >= 3),
				"verdict": seed_verdict,
				"out_csv": row["out_csv"],
				"provenance_json": row["provenance_json"],
			}
		)
	return rows


def _build_h54_combined_row(
	*,
	summary: Mapping[str, Any],
	control_summary: Mapping[str, Any],
	sampled_inertia: float,
	is_control: bool,
	is_tiebreak: bool,
	evolution_mode: str,
	selection_strength: float,
	memory_kernel: int,
) -> dict[str, Any]:
	control_mean_env_gamma = float(control_summary["mean_env_gamma"])
	control_mean_stage3_score = float(control_summary["mean_stage3_score"])
	mean_env_gamma = float(summary["mean_env_gamma"])
	mean_stage3_score = float(summary["mean_stage3_score"])
	if is_control:
		gamma_pass = ""
		stage3_pass = ""
		level3_pass = ""
		full_pass = ""
		verdict = "control"
	else:
		point_decision = _h54_point_verdict(
			control_mean_env_gamma=control_mean_env_gamma,
			control_mean_stage3_score=control_mean_stage3_score,
			candidate_mean_env_gamma=mean_env_gamma,
			candidate_mean_stage3_score=mean_stage3_score,
			p_level_3=float(summary["p_level_3"]),
		)
		gamma_pass = _yes_no(point_decision["gamma_pass"])
		stage3_pass = _yes_no(point_decision["stage3_pass"])
		level3_pass = _yes_no(point_decision["level3_pass"])
		full_pass = _yes_no(point_decision["full_pass"])
		verdict = str(point_decision["verdict"])
	return {
		"condition": f"mu{_format_mu(sampled_inertia)}",
		"sampled_inertia": _format_mu(sampled_inertia),
		"is_control": _yes_no(is_control),
		"is_tiebreak": _yes_no(is_tiebreak),
		"n_seeds": summary["n_seeds"],
		"mean_cycle_level": summary["mean_cycle_level"],
		"mean_stage3_score": summary["mean_stage3_score"],
		"mean_turn_strength": summary["mean_turn_strength"],
		"mean_env_gamma": summary["mean_env_gamma"],
		"mean_success_rate": summary["mean_success_rate"],
		"level_counts_json": summary["level_counts_json"],
		"p_level_ge_2": summary["p_level_ge_2"],
		"p_level_3": summary["p_level_3"],
		"control_mean_env_gamma": f"{control_mean_env_gamma:.6f}",
		"control_mean_stage3_score": f"{control_mean_stage3_score:.6f}",
		"gamma_uplift_ratio_vs_control": _format_ratio(
			_gamma_uplift_ratio(baseline_gamma=control_mean_env_gamma, candidate_gamma=mean_env_gamma)
		),
		"stage3_uplift_vs_control": f"{mean_stage3_score - control_mean_stage3_score:.6f}",
		"gamma_pass": gamma_pass,
		"stage3_pass": stage3_pass,
		"level3_pass": level3_pass,
		"full_pass": full_pass,
		"verdict": verdict,
		"events_json": summary["events_json"],
		"personality_mode": summary["personality_mode"],
		"personality_jitter": summary["personality_jitter"],
		"evolution_mode": evolution_mode,
		"selection_strength": f"{float(selection_strength):.6f}",
		"memory_kernel": int(memory_kernel),
		"out_dir": summary["out_dir"],
	}


def decide_h54_overall(*, combined_rows: list[dict[str, Any]], auto_tiebreak: bool) -> dict[str, Any]:
	candidate_rows = [row for row in combined_rows if row["is_control"] == "no"]
	pass_rows = [row for row in candidate_rows if row["verdict"] == "pass"]
	weak_rows = [row for row in candidate_rows if row["verdict"] == "weak_positive"]
	fail_rows = [row for row in candidate_rows if row["verdict"] == "fail"]
	if len(pass_rows) >= 2:
		decision = "pass"
	elif pass_rows or weak_rows:
		decision = "weak_positive"
	else:
		decision = "fail"
	return {
		"decision": decision,
		"pass_points": len(pass_rows),
		"weak_positive_points": len(weak_rows),
		"fail_points": len(fail_rows),
		"auto_tiebreak": bool(auto_tiebreak),
	}


def _rewrite_h54_provenance_with_control(
	*,
	seed_csvs: Mapping[int, Path],
	control_seed_csvs: Mapping[int, Path],
	events_json: Path,
) -> None:
	for seed, csv_path in seed_csvs.items():
		provenance_path = csv_path.with_name(f"seed{seed}_provenance.json")
		prov = summarize_event_provenance(
			csv_path,
			events_json=events_json,
			baseline_csv=control_seed_csvs[int(seed)],
			compare_envelope_gamma=True,
			envelope_series="p",
		)
		provenance_path.write_text(json.dumps(prov, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_h54_decision(
	path: Path,
	*,
	overall_decision: Mapping[str, Any],
	control_row: Mapping[str, Any],
	combined_rows: list[dict[str, Any]],
	tiebreak_inertia: float,
	tiebreak_executed: bool,
) -> None:
	lines = [
		"# H5.4 Personality + Sampled Inertia Decision",
		"",
		f"- decision: {overall_decision['decision']}",
		f"- pass_points: {overall_decision['pass_points']}",
		f"- weak_positive_points: {overall_decision['weak_positive_points']}",
		f"- fail_points: {overall_decision['fail_points']}",
		f"- auto_tiebreak: {_yes_no(overall_decision['auto_tiebreak'])}",
		f"- tiebreak_inertia: {_format_mu(tiebreak_inertia)}",
		f"- tiebreak_executed: {_yes_no(tiebreak_executed)}",
		f"- matched_control_mean_env_gamma: {control_row['mean_env_gamma']}",
		f"- matched_control_mean_stage3_score: {control_row['mean_stage3_score']}",
		f"- matched_control_level_counts: {control_row['level_counts_json']}",
		"",
		"## Point Summary",
		"",
	]
	for row in combined_rows:
		if row["is_control"] == "yes":
			continue
		lines.append(
			"- "
			f"{row['condition']}: verdict={row['verdict']} "
			f"gamma={row['mean_env_gamma']} ratio={row['gamma_uplift_ratio_vs_control']} "
			f"stage3={row['mean_stage3_score']} uplift={row['stage3_uplift_vs_control']} "
			f"level_counts={row['level_counts_json']} p_level3={row['p_level_3']} "
			f"is_tiebreak={row['is_tiebreak']}"
		)
	lines.extend(
		[
			"",
			"## Stop Rule",
			"",
			"- 至少 2 個 sampled_inertia 點 full pass 才算 H5.4 pass。",
			"- 若只出現 weak_positive，且有啟用 auto tiebreak，才補 mu_s=0.30。",
			"- 若所有點都沒有 Level 3 且 stage3 uplift 不足，H5.4 視為 fail。",
		]
	)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Gate 1 / H5.4 / H5.5 personality smoke harness")
	parser.add_argument("--protocol", choices=["gate1", "h54", "h55", "h55r"], default="gate1")
	parser.add_argument("--players", type=int, default=300)
	parser.add_argument("--rounds", type=int, default=3000)
	parser.add_argument("--seeds", type=str, default="45,47,49")
	parser.add_argument("--memory-kernel", type=int, default=3)
	parser.add_argument("--selection-strength", type=float, default=0.06)
	parser.add_argument("--init-bias", type=float, default=0.12)
	parser.add_argument("--burn-in", type=int, default=900)
	parser.add_argument("--tail", type=int, default=1200)
	parser.add_argument("--jitter", type=float, default=0.08)
	parser.add_argument("--a", type=float, default=1.0)
	parser.add_argument("--b", type=float, default=0.9)
	parser.add_argument("--cross", type=float, default=0.20)
	parser.add_argument("--source-events-json", type=Path, default=DEFAULT_FULL_EVENTS_JSON)
	parser.add_argument("--events-json", type=Path, default=DEFAULT_GATE0_EVENTS_JSON)
	parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
	parser.add_argument("--hetero-dir", type=Path, default=DEFAULT_HETERO_DIR)
	parser.add_argument("--summary-tsv", type=Path, default=DEFAULT_SUMMARY_TSV)
	parser.add_argument("--combined-tsv", type=Path, default=DEFAULT_COMBINED_TSV)
	parser.add_argument("--decision-md", type=Path, default=DEFAULT_DECISION_MD)
	parser.add_argument("--world-mode", choices=["zero", "heterogeneous"], default="heterogeneous")
	parser.add_argument("--evolution-mode", choices=["sampled", "sampled_inertial"], default="sampled_inertial")
	parser.add_argument("--sampled-inertia", type=float, default=0.0)
	parser.add_argument("--out-root", type=Path, default=DEFAULT_H54_OUT_ROOT)
	parser.add_argument("--h54-inertias", type=str, default="0.15,0.25,0.35")
	parser.add_argument("--h54-tiebreak-inertia", type=float, default=0.30)
	parser.add_argument("--h54-auto-tiebreak", action="store_true")
	parser.add_argument("--h54-summary-tsv", type=Path, default=DEFAULT_H54_SUMMARY_TSV)
	parser.add_argument("--h54-combined-tsv", type=Path, default=DEFAULT_H54_COMBINED_TSV)
	parser.add_argument("--h54-decision-md", type=Path, default=DEFAULT_H54_DECISION_MD)
	parser.add_argument("--h55-out-root", type=Path, default=DEFAULT_H55_OUT_ROOT)
	parser.add_argument("--h55-summary-tsv", type=Path, default=DEFAULT_H55_SUMMARY_TSV)
	parser.add_argument("--h55-combined-tsv", type=Path, default=DEFAULT_H55_COMBINED_TSV)
	parser.add_argument("--h55-decision-md", type=Path, default=DEFAULT_H55_DECISION_MD)
	parser.add_argument("--h55-trigger", choices=["ad_share", "ad_product"], default="ad_share")
	parser.add_argument("--h55-base-theta", type=float, default=0.55)
	parser.add_argument("--h55-band-low", type=float, default=0.62)
	parser.add_argument("--h55-band-high", type=float, default=0.70)
	parser.add_argument("--h55-slow-alpha", type=float, default=0.35)
	parser.add_argument("--h55-a-hi", type=float, default=1.10)
	parser.add_argument("--h55-b-hi", type=float, default=1.00)
	parser.add_argument("--h55r-out-root", type=Path, default=DEFAULT_H55R_OUT_ROOT)
	parser.add_argument("--h55r-summary-tsv", type=Path, default=DEFAULT_H55R_SUMMARY_TSV)
	parser.add_argument("--h55r-combined-tsv", type=Path, default=DEFAULT_H55R_COMBINED_TSV)
	parser.add_argument("--h55r-decision-md", type=Path, default=DEFAULT_H55R_DECISION_MD)
	parser.add_argument("--h55r-legacy-a-hi-values", type=str, default="1.05,1.10,1.15")
	parser.add_argument("--h55r-legacy-b-hi-values", type=str, default="0.95,1.00,1.05")
	parser.add_argument("--h55r-band-low", type=float, default=0.62)
	parser.add_argument("--h55r-band-high", type=float, default=0.70)
	parser.add_argument("--h55r-alpha-values", type=str, default="0.20,0.35,0.50")
	parser.add_argument("--h55r-stage3-min-uplift", type=float, default=0.015)
	parser.add_argument("--h55r-gamma-min-ratio", type=float, default=1.10)
	parser.add_argument("--h55r-confirm-rounds", type=int, default=5000)
	parser.add_argument("--h55r-confirm-seeds", type=str, default="45,47,49,51,53,55")
	return parser.parse_args()


def _run_gate1_protocol(args: argparse.Namespace) -> int:
	write_reduced_events_json(args.events_json, source_json=args.source_events_json)
	seeds = _parse_seeds(args.seeds)
	baseline_rows, baseline_summary, baseline_csvs, baseline_actions, _baseline_diagnostics = run_condition(
		condition="zero_personality",
		world_mode="zero",
		seeds=seeds,
		out_dir=args.baseline_dir,
		events_json=args.events_json,
		players=args.players,
		rounds=args.rounds,
		selection_strength=args.selection_strength,
		init_bias=args.init_bias,
		memory_kernel=args.memory_kernel,
		burn_in=args.burn_in,
		tail=args.tail,
		jitter=args.jitter,
		a=args.a,
		b=args.b,
		cross=args.cross,
		evolution_mode="sampled",
		sampled_inertia=0.0,
	)
	hetero_rows, hetero_summary, hetero_csvs, hetero_actions, _hetero_diagnostics = run_condition(
		condition="heterogeneous_world",
		world_mode="heterogeneous",
		seeds=seeds,
		out_dir=args.hetero_dir,
		events_json=args.events_json,
		players=args.players,
		rounds=args.rounds,
		selection_strength=args.selection_strength,
		init_bias=args.init_bias,
		memory_kernel=args.memory_kernel,
		burn_in=args.burn_in,
		tail=args.tail,
		jitter=args.jitter,
		a=args.a,
		b=args.b,
		cross=args.cross,
		evolution_mode="sampled",
		sampled_inertia=0.0,
	)
	for seed in seeds:
		prov = summarize_event_provenance(
			hetero_csvs[int(seed)],
			events_json=args.events_json,
			baseline_csv=baseline_csvs[int(seed)],
			compare_envelope_gamma=True,
			envelope_series="p",
		)
		(Path(args.hetero_dir) / f"seed{seed}_provenance.json").write_text(
			json.dumps(prov, ensure_ascii=False, indent=2) + "\n",
			encoding="utf-8",
		)
	all_seed_rows = baseline_rows + hetero_rows
	_write_tsv(args.summary_tsv, fieldnames=METRIC_FIELDNAMES, rows=all_seed_rows)
	_write_tsv(args.combined_tsv, fieldnames=CONDITION_SUMMARY_FIELDNAMES, rows=[baseline_summary, hetero_summary])
	decision = decide_gate1(baseline_summary=baseline_summary, hetero_summary=hetero_summary)
	_write_gate1_decision(
		args.decision_md,
		decision=decision,
		baseline_summary=baseline_summary,
		hetero_summary=hetero_summary,
		baseline_actions=baseline_actions,
		hetero_actions=hetero_actions,
	)
	print(f"decision={decision['decision']}")
	print(f"summary_tsv={args.summary_tsv}")
	print(f"combined_tsv={args.combined_tsv}")
	print(f"decision_md={args.decision_md}")
	return 0


def _run_h54_protocol(args: argparse.Namespace) -> int:
	write_reduced_events_json(args.events_json, source_json=args.source_events_json)
	seeds = _parse_seeds(args.seeds)
	main_inertias = _parse_float_list(args.h54_inertias)
	control_mu = 0.0
	out_root = Path(args.out_root)
	control_dir = out_root / f"mu{_format_mu(control_mu)}"
	control_rows, control_summary, control_seed_csvs, _control_actions, _control_diagnostics = run_condition(
		condition=f"mu{_format_mu(control_mu)}",
		world_mode=str(args.world_mode),
		seeds=seeds,
		out_dir=control_dir,
		events_json=args.events_json,
		players=args.players,
		rounds=args.rounds,
		selection_strength=args.selection_strength,
		init_bias=args.init_bias,
		memory_kernel=args.memory_kernel,
		burn_in=args.burn_in,
		tail=args.tail,
		jitter=args.jitter,
		a=args.a,
		b=args.b,
		cross=args.cross,
		payoff_mode="matrix_ab",
		evolution_mode=str(args.evolution_mode),
		sampled_inertia=control_mu,
	)
	seed_summary_rows = _build_h54_seed_rows(
		metrics_rows=control_rows,
		control_metrics_rows=control_rows,
		sampled_inertia=control_mu,
		is_control=True,
		is_tiebreak=False,
	)
	combined_rows = [
		_build_h54_combined_row(
			summary=control_summary,
			control_summary=control_summary,
			sampled_inertia=control_mu,
			is_control=True,
			is_tiebreak=False,
			evolution_mode=str(args.evolution_mode),
			selection_strength=float(args.selection_strength),
			memory_kernel=int(args.memory_kernel),
		)
	]

	def _run_candidate(mu_value: float, *, is_tiebreak: bool) -> None:
		candidate_dir = out_root / f"mu{_format_mu(mu_value)}"
		candidate_rows, candidate_summary, candidate_seed_csvs, _candidate_actions, _candidate_diagnostics = run_condition(
			condition=f"mu{_format_mu(mu_value)}",
			world_mode=str(args.world_mode),
			seeds=seeds,
			out_dir=candidate_dir,
			events_json=args.events_json,
			players=args.players,
			rounds=args.rounds,
			selection_strength=args.selection_strength,
			init_bias=args.init_bias,
			memory_kernel=args.memory_kernel,
			burn_in=args.burn_in,
			tail=args.tail,
			jitter=args.jitter,
			a=args.a,
			b=args.b,
			cross=args.cross,
			payoff_mode="matrix_ab",
			evolution_mode=str(args.evolution_mode),
			sampled_inertia=float(mu_value),
		)
		_rewrite_h54_provenance_with_control(
			seed_csvs=candidate_seed_csvs,
			control_seed_csvs=control_seed_csvs,
			events_json=args.events_json,
		)
		seed_summary_rows.extend(
			_build_h54_seed_rows(
				metrics_rows=candidate_rows,
				control_metrics_rows=control_rows,
				sampled_inertia=float(mu_value),
				is_control=False,
				is_tiebreak=is_tiebreak,
			)
		)
		combined_rows.append(
			_build_h54_combined_row(
				summary=candidate_summary,
				control_summary=control_summary,
				sampled_inertia=float(mu_value),
				is_control=False,
				is_tiebreak=is_tiebreak,
				evolution_mode=str(args.evolution_mode),
				selection_strength=float(args.selection_strength),
				memory_kernel=int(args.memory_kernel),
			)
		)

	for mu_value in main_inertias:
		_run_candidate(float(mu_value), is_tiebreak=False)

	tiebreak_executed = False
	overall_decision = decide_h54_overall(combined_rows=combined_rows, auto_tiebreak=bool(args.h54_auto_tiebreak))
	if (
		bool(args.h54_auto_tiebreak)
		and overall_decision["decision"] == "weak_positive"
		and all(abs(float(mu) - float(args.h54_tiebreak_inertia)) > 1e-9 for mu in main_inertias)
	):
		_run_candidate(float(args.h54_tiebreak_inertia), is_tiebreak=True)
		tiebreak_executed = True
		overall_decision = decide_h54_overall(combined_rows=combined_rows, auto_tiebreak=bool(args.h54_auto_tiebreak))

	_write_tsv(args.h54_summary_tsv, fieldnames=H54_METRIC_FIELDNAMES, rows=seed_summary_rows)
	_write_tsv(args.h54_combined_tsv, fieldnames=H54_COMBINED_FIELDNAMES, rows=combined_rows)
	_write_h54_decision(
		args.h54_decision_md,
		overall_decision=overall_decision,
		control_row=combined_rows[0],
		combined_rows=combined_rows,
		tiebreak_inertia=float(args.h54_tiebreak_inertia),
		tiebreak_executed=tiebreak_executed,
	)
	print(f"decision={overall_decision['decision']}")
	print(f"summary_tsv={args.h54_summary_tsv}")
	print(f"combined_tsv={args.h54_combined_tsv}")
	print(f"decision_md={args.h54_decision_md}")
	return 0


def _h55_candidate_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
	band_center = (float(args.h55_band_low) + float(args.h55_band_high)) / 2.0
	return [
		{
			"condition": "threshold_legacy",
			"payoff_mode": "threshold_ab",
			"threshold_trigger": str(args.h55_trigger),
			"threshold_theta": float(args.h55_base_theta),
			"threshold_theta_low": None,
			"threshold_theta_high": None,
			"threshold_state_alpha": 1.0,
			"threshold_a_hi": float(args.h55_a_hi),
			"threshold_b_hi": float(args.h55_b_hi),
		},
		{
			"condition": "threshold_hysteresis",
			"payoff_mode": "threshold_ab",
			"threshold_trigger": str(args.h55_trigger),
			"threshold_theta": band_center,
			"threshold_theta_low": float(args.h55_band_low),
			"threshold_theta_high": float(args.h55_band_high),
			"threshold_state_alpha": 1.0,
			"threshold_a_hi": float(args.h55_a_hi),
			"threshold_b_hi": float(args.h55_b_hi),
		},
		{
			"condition": "threshold_slow_state",
			"payoff_mode": "threshold_ab",
			"threshold_trigger": str(args.h55_trigger),
			"threshold_theta": band_center,
			"threshold_theta_low": float(args.h55_band_low),
			"threshold_theta_high": float(args.h55_band_high),
			"threshold_state_alpha": float(args.h55_slow_alpha),
			"threshold_a_hi": float(args.h55_a_hi),
			"threshold_b_hi": float(args.h55_b_hi),
		},
	]


def _h55_point_verdict(
	*,
	control_mean_env_gamma: float,
	control_mean_stage3_score: float,
	candidate_mean_env_gamma: float,
	candidate_mean_stage3_score: float,
	p_level_3: float,
	p_switched_seed: float,
) -> dict[str, Any]:
	gamma_pass = _gamma_improvement_25pct(
		baseline_gamma=float(control_mean_env_gamma),
		candidate_gamma=float(candidate_mean_env_gamma),
	)
	stage3_pass = float(candidate_mean_stage3_score) >= float(control_mean_stage3_score) + 0.018
	level3_pass = float(p_level_3) > 0.0
	switch_pass = float(p_switched_seed) > 0.0
	full_pass = gamma_pass and stage3_pass and level3_pass and switch_pass
	if full_pass:
		verdict = "pass"
	elif gamma_pass or stage3_pass or level3_pass or switch_pass:
		verdict = "weak_positive"
	else:
		verdict = "fail"
	return {
		"gamma_pass": gamma_pass,
		"stage3_pass": stage3_pass,
		"level3_pass": level3_pass,
		"switch_pass": switch_pass,
		"full_pass": full_pass,
		"verdict": verdict,
	}


def _gamma_ratio_pass(*, baseline_gamma: float, candidate_gamma: float, min_ratio: float) -> bool:
	ratio = _gamma_uplift_ratio(baseline_gamma=float(baseline_gamma), candidate_gamma=float(candidate_gamma))
	if ratio == float("inf"):
		return True
	if ratio == float("-inf"):
		return False
	return float(ratio) >= float(min_ratio)


def _h55r_candidate_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
	legacy_a_hi_values = _parse_float_list(args.h55r_legacy_a_hi_values)
	legacy_b_hi_values = _parse_float_list(args.h55r_legacy_b_hi_values)
	alpha_values = _parse_float_list(args.h55r_alpha_values)
	if len(legacy_a_hi_values) != len(legacy_b_hi_values):
		raise ValueError("--h55r-legacy-a-hi-values and --h55r-legacy-b-hi-values must have the same length")
	if len(alpha_values) != 3:
		raise ValueError("--h55r-alpha-values must contain exactly 3 values for the locked H5.5R protocol")
	legacy_specs = [
		{
			"condition": f"legacy_geo_{label}",
			"payoff_mode": "threshold_ab",
			"threshold_trigger": "ad_share",
			"threshold_theta": float(args.h55_base_theta),
			"threshold_theta_low": None,
			"threshold_theta_high": None,
			"threshold_state_alpha": 1.0,
			"threshold_a_hi": float(a_hi),
			"threshold_b_hi": float(b_hi),
		}
		for label, a_hi, b_hi in zip(("lo", "mid", "hi"), legacy_a_hi_values, legacy_b_hi_values, strict=True)
	]
	slow_specs = [
		{
			"condition": f"slow_alpha_{int(round(float(alpha) * 100.0)):03d}",
			"payoff_mode": "threshold_ab",
			"threshold_trigger": "ad_share",
			"threshold_theta": (float(args.h55r_band_low) + float(args.h55r_band_high)) / 2.0,
			"threshold_theta_low": float(args.h55r_band_low),
			"threshold_theta_high": float(args.h55r_band_high),
			"threshold_state_alpha": float(alpha),
			"threshold_a_hi": float(args.h55_a_hi),
			"threshold_b_hi": float(args.h55_b_hi),
		}
		for alpha in alpha_values
	]
	return legacy_specs + slow_specs


def _h55r_point_verdict(
	*,
	control_mean_env_gamma: float,
	control_mean_stage3_score: float,
	candidate_mean_env_gamma: float,
	candidate_mean_stage3_score: float,
	p_switched_seed: float,
	stage3_min_uplift: float,
	gamma_min_ratio: float,
) -> dict[str, Any]:
	stage3_refine_pass = float(candidate_mean_stage3_score) >= float(control_mean_stage3_score) + float(stage3_min_uplift)
	gamma_refine_pass = _gamma_ratio_pass(
		baseline_gamma=float(control_mean_env_gamma),
		candidate_gamma=float(candidate_mean_env_gamma),
		min_ratio=float(gamma_min_ratio),
	)
	switch_pass = float(p_switched_seed) > 0.0
	confirm_gate_pass = stage3_refine_pass and gamma_refine_pass and switch_pass
	if confirm_gate_pass:
		verdict = "confirm_candidate"
	elif switch_pass:
		verdict = "switch_only"
	elif stage3_refine_pass or gamma_refine_pass:
		verdict = "uplift_only"
	else:
		verdict = "fail"
	return {
		"stage3_refine_pass": stage3_refine_pass,
		"gamma_refine_pass": gamma_refine_pass,
		"switch_pass": switch_pass,
		"confirm_gate_pass": confirm_gate_pass,
		"verdict": verdict,
	}


def _build_h55r_combined_row(
	*,
	summary: Mapping[str, Any],
	control_summary: Mapping[str, Any],
	seed_diagnostics: Mapping[int, Mapping[str, Any]],
	spec: Mapping[str, Any],
	selection_strength: float,
	memory_kernel: int,
	is_control: bool,
	stage3_min_uplift: float,
	gamma_min_ratio: float,
) -> dict[str, Any]:
	control_mean_env_gamma = float(control_summary["mean_env_gamma"])
	control_mean_stage3_score = float(control_summary["mean_stage3_score"])
	mean_env_gamma = float(summary["mean_env_gamma"])
	mean_stage3_score = float(summary["mean_stage3_score"])
	diagnostics = _aggregate_threshold_diagnostics(seed_diagnostics)
	if is_control:
		switch_pass = ""
		gamma_refine_pass = ""
		stage3_refine_pass = ""
		confirm_gate_pass = ""
		verdict = "control"
	else:
		point_decision = _h55r_point_verdict(
			control_mean_env_gamma=control_mean_env_gamma,
			control_mean_stage3_score=control_mean_stage3_score,
			candidate_mean_env_gamma=mean_env_gamma,
			candidate_mean_stage3_score=mean_stage3_score,
			p_switched_seed=float(diagnostics["p_switched_seed"] or 0.0),
			stage3_min_uplift=float(stage3_min_uplift),
			gamma_min_ratio=float(gamma_min_ratio),
		)
		switch_pass = _yes_no(point_decision["switch_pass"])
		gamma_refine_pass = _yes_no(point_decision["gamma_refine_pass"])
		stage3_refine_pass = _yes_no(point_decision["stage3_refine_pass"])
		confirm_gate_pass = _yes_no(point_decision["confirm_gate_pass"])
		verdict = str(point_decision["verdict"])
	return {
		"condition": str(spec["condition"]),
		"is_control": _yes_no(is_control),
		"n_seeds": summary["n_seeds"],
		"mean_cycle_level": summary["mean_cycle_level"],
		"mean_stage3_score": summary["mean_stage3_score"],
		"mean_turn_strength": summary["mean_turn_strength"],
		"mean_env_gamma": summary["mean_env_gamma"],
		"mean_success_rate": summary["mean_success_rate"],
		"level_counts_json": summary["level_counts_json"],
		"p_level_ge_2": summary["p_level_ge_2"],
		"p_level_3": summary["p_level_3"],
		"payoff_mode": str(spec["payoff_mode"]),
		"threshold_trigger": str(spec.get("threshold_trigger") or ""),
		"threshold_theta": _format_optional_float(spec.get("threshold_theta")),
		"threshold_theta_low": _format_optional_float(spec.get("threshold_theta_low")),
		"threshold_theta_high": _format_optional_float(spec.get("threshold_theta_high")),
		"threshold_state_alpha": _format_optional_float(spec.get("threshold_state_alpha")),
		"threshold_a_hi": _format_optional_float(spec.get("threshold_a_hi")),
		"threshold_b_hi": _format_optional_float(spec.get("threshold_b_hi")),
		"mean_regime_high_share": _format_optional_float(diagnostics.get("mean_regime_high_share")),
		"mean_regime_switches": _format_optional_float(diagnostics.get("mean_regime_switches")),
		"p_switched_seed": _format_optional_float(diagnostics.get("p_switched_seed")),
		"mean_threshold_state": _format_optional_float(diagnostics.get("mean_threshold_state")),
		"switch_pass": switch_pass,
		"control_mean_env_gamma": f"{control_mean_env_gamma:.6f}",
		"control_mean_stage3_score": f"{control_mean_stage3_score:.6f}",
		"gamma_uplift_ratio_vs_control": _format_ratio(
			_gamma_uplift_ratio(baseline_gamma=control_mean_env_gamma, candidate_gamma=mean_env_gamma)
		),
		"stage3_uplift_vs_control": f"{mean_stage3_score - control_mean_stage3_score:.6f}",
		"gamma_refine_pass": gamma_refine_pass,
		"stage3_refine_pass": stage3_refine_pass,
		"confirm_gate_pass": confirm_gate_pass,
		"confirm_rank": "",
		"verdict": verdict,
		"events_json": summary["events_json"],
		"personality_mode": summary["personality_mode"],
		"personality_jitter": summary["personality_jitter"],
		"selection_strength": f"{float(selection_strength):.6f}",
		"memory_kernel": int(memory_kernel),
		"out_dir": summary["out_dir"],
	}


def _rank_h55r_candidates(combined_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
	eligible = [row for row in combined_rows if row.get("confirm_gate_pass") == "yes"]
	eligible.sort(
		key=lambda row: (
			float(row["stage3_uplift_vs_control"]),
			_parse_ratio(row["gamma_uplift_ratio_vs_control"]),
			float(row["mean_regime_switches"] or 0.0),
		),
		reverse=True,
	)
	for rank, row in enumerate(eligible, start=1):
		row["confirm_rank"] = str(rank)
	return eligible[0] if eligible else None


def _write_h55r_decision(
	path: Path,
	*,
	overall_decision: str,
	control_row: Mapping[str, Any],
	combined_rows: list[dict[str, Any]],
	best_row: Mapping[str, Any] | None,
	confirm_rounds: int,
	confirm_seeds: str,
	stage3_min_uplift: float,
	gamma_min_ratio: float,
) -> None:
	lines = [
		"# H5.5R Personality Nonlinear Refinement Decision",
		"",
		f"- decision: {overall_decision}",
		f"- hard_stop_triggered: {_yes_no(best_row is None)}",
		f"- matched_control_mean_env_gamma: {control_row['mean_env_gamma']}",
		f"- matched_control_mean_stage3_score: {control_row['mean_stage3_score']}",
		f"- stage3_min_uplift: {float(stage3_min_uplift):.6f}",
		f"- gamma_min_ratio: {float(gamma_min_ratio):.6f}",
		f"- confirm_rounds: {int(confirm_rounds)}",
		f"- confirm_seeds: {confirm_seeds}",
		"",
		"## Point Summary",
		"",
	]
	for row in combined_rows:
		if row["is_control"] == "yes":
			continue
		lines.append(
			"- "
			f"{row['condition']}: verdict={row['verdict']} rank={row['confirm_rank'] or '-'} "
			f"gamma_ratio={row['gamma_uplift_ratio_vs_control']} stage3_uplift={row['stage3_uplift_vs_control']} "
			f"p_switched_seed={row['p_switched_seed']} mean_regime_switches={row['mean_regime_switches']}"
		)
	lines.extend(["", "## Confirm", ""])
	if best_row is None:
		lines.append("- no_confirm_candidate: yes")
		lines.append("- H5.5 line should be closed after this refinement.")
	else:
		lines.append(f"- best_condition: {best_row['condition']}")
		lines.append(f"- best_stage3_uplift: {best_row['stage3_uplift_vs_control']}")
		lines.append(f"- best_gamma_ratio: {best_row['gamma_uplift_ratio_vs_control']}")
		lines.append(f"- best_mean_regime_switches: {best_row['mean_regime_switches']}")
		lines.append("- recommended_confirm_protocol: run only this one cell with the locked 5000-round, 6-seed confirm window")
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_h55r_protocol(args: argparse.Namespace) -> int:
	write_reduced_events_json(args.events_json, source_json=args.source_events_json)
	seeds = _parse_seeds(args.seeds)
	out_root = Path(args.h55r_out_root)
	control_spec = {
		"condition": "matrix_control",
		"payoff_mode": "matrix_ab",
		"threshold_trigger": "",
		"threshold_theta": None,
		"threshold_theta_low": None,
		"threshold_theta_high": None,
		"threshold_state_alpha": None,
		"threshold_a_hi": None,
		"threshold_b_hi": None,
	}
	control_dir = out_root / str(control_spec["condition"])
	control_rows, control_summary, control_seed_csvs, _control_actions, control_diagnostics = run_condition(
		condition=str(control_spec["condition"]),
		world_mode=str(args.world_mode),
		seeds=seeds,
		out_dir=control_dir,
		events_json=args.events_json,
		players=args.players,
		rounds=args.rounds,
		selection_strength=args.selection_strength,
		init_bias=args.init_bias,
		memory_kernel=args.memory_kernel,
		burn_in=args.burn_in,
		tail=args.tail,
		jitter=args.jitter,
		a=args.a,
		b=args.b,
		cross=args.cross,
		payoff_mode="matrix_ab",
		evolution_mode="sampled",
		sampled_inertia=0.0,
	)
	seed_summary_rows = _build_h55_seed_rows(
		metrics_rows=control_rows,
		control_metrics_rows=control_rows,
		seed_diagnostics=control_diagnostics,
		spec=control_spec,
		is_control=True,
	)
	combined_rows = [
		_build_h55r_combined_row(
			summary=control_summary,
			control_summary=control_summary,
			seed_diagnostics=control_diagnostics,
			spec=control_spec,
			selection_strength=float(args.selection_strength),
			memory_kernel=int(args.memory_kernel),
			is_control=True,
			stage3_min_uplift=float(args.h55r_stage3_min_uplift),
			gamma_min_ratio=float(args.h55r_gamma_min_ratio),
		)
	]
	for spec in _h55r_candidate_specs(args):
		candidate_dir = out_root / str(spec["condition"])
		candidate_rows, candidate_summary, candidate_seed_csvs, _candidate_actions, candidate_diagnostics = run_condition(
			condition=str(spec["condition"]),
			world_mode=str(args.world_mode),
			seeds=seeds,
			out_dir=candidate_dir,
			events_json=args.events_json,
			players=args.players,
			rounds=args.rounds,
			selection_strength=args.selection_strength,
			init_bias=args.init_bias,
			memory_kernel=args.memory_kernel,
			burn_in=args.burn_in,
			tail=args.tail,
			jitter=args.jitter,
			a=args.a,
			b=args.b,
			cross=args.cross,
			payoff_mode=str(spec["payoff_mode"]),
			threshold_theta=float(spec["threshold_theta"]),
			threshold_theta_low=spec["threshold_theta_low"],
			threshold_theta_high=spec["threshold_theta_high"],
			threshold_trigger=str(spec["threshold_trigger"]),
			threshold_state_alpha=float(spec["threshold_state_alpha"]),
			threshold_a_hi=float(spec["threshold_a_hi"]),
			threshold_b_hi=float(spec["threshold_b_hi"]),
			evolution_mode="sampled",
			sampled_inertia=0.0,
		)
		_rewrite_h54_provenance_with_control(
			seed_csvs=candidate_seed_csvs,
			control_seed_csvs=control_seed_csvs,
			events_json=args.events_json,
		)
		seed_summary_rows.extend(
			_build_h55_seed_rows(
				metrics_rows=candidate_rows,
				control_metrics_rows=control_rows,
				seed_diagnostics=candidate_diagnostics,
				spec=spec,
				is_control=False,
			)
		)
		combined_rows.append(
			_build_h55r_combined_row(
				summary=candidate_summary,
				control_summary=control_summary,
				seed_diagnostics=candidate_diagnostics,
				spec=spec,
				selection_strength=float(args.selection_strength),
				memory_kernel=int(args.memory_kernel),
				is_control=False,
				stage3_min_uplift=float(args.h55r_stage3_min_uplift),
				gamma_min_ratio=float(args.h55r_gamma_min_ratio),
			)
		)
	best_row = _rank_h55r_candidates(combined_rows)
	overall_decision = "advance_to_confirm" if best_row is not None else "close_h55"
	_write_tsv(args.h55r_summary_tsv, fieldnames=H55R_METRIC_FIELDNAMES, rows=seed_summary_rows)
	_write_tsv(args.h55r_combined_tsv, fieldnames=H55R_COMBINED_FIELDNAMES, rows=combined_rows)
	_write_h55r_decision(
		args.h55r_decision_md,
		overall_decision=overall_decision,
		control_row=combined_rows[0],
		combined_rows=combined_rows,
		best_row=best_row,
		confirm_rounds=int(args.h55r_confirm_rounds),
		confirm_seeds=str(args.h55r_confirm_seeds),
		stage3_min_uplift=float(args.h55r_stage3_min_uplift),
		gamma_min_ratio=float(args.h55r_gamma_min_ratio),
	)
	print(f"decision={overall_decision}")
	print(f"summary_tsv={args.h55r_summary_tsv}")
	print(f"combined_tsv={args.h55r_combined_tsv}")
	print(f"decision_md={args.h55r_decision_md}")
	return 0


def _build_h55_seed_rows(
	*,
	metrics_rows: list[dict[str, Any]],
	control_metrics_rows: list[dict[str, Any]],
	seed_diagnostics: Mapping[int, Mapping[str, Any]],
	spec: Mapping[str, Any],
	is_control: bool,
) -> list[dict[str, Any]]:
	control_by_seed = _index_metrics_by_seed(control_metrics_rows)
	rows: list[dict[str, Any]] = []
	for row in metrics_rows:
		seed = int(row["seed"])
		control_row = control_by_seed[seed]
		control_env_gamma = float(control_row["env_gamma"])
		control_stage3_score = float(control_row["stage3_score"])
		candidate_env_gamma = float(row["env_gamma"])
		candidate_stage3_score = float(row["stage3_score"])
		diagnostics = seed_diagnostics.get(seed, {})
		switch_active = bool(diagnostics.get("switch_active"))
		seed_decision = _h55_point_verdict(
			control_mean_env_gamma=control_env_gamma,
			control_mean_stage3_score=control_stage3_score,
			candidate_mean_env_gamma=candidate_env_gamma,
			candidate_mean_stage3_score=candidate_stage3_score,
			p_level_3=1.0 if int(row["cycle_level"]) >= 3 else 0.0,
			p_switched_seed=1.0 if switch_active else 0.0,
		)
		rows.append(
			{
				"condition": str(spec["condition"]),
				"is_control": _yes_no(is_control),
				"seed": seed,
				"cycle_level": int(row["cycle_level"]),
				"stage3_score": row["stage3_score"],
				"turn_strength": row["turn_strength"],
				"env_gamma": row["env_gamma"],
				"env_gamma_r2": row["env_gamma_r2"],
				"env_gamma_n_peaks": row["env_gamma_n_peaks"],
				"success_rate": row["success_rate"],
				"payoff_mode": str(spec["payoff_mode"]),
				"threshold_trigger": str(spec.get("threshold_trigger") or ""),
				"threshold_theta": _format_optional_float(spec.get("threshold_theta")),
				"threshold_theta_low": _format_optional_float(spec.get("threshold_theta_low")),
				"threshold_theta_high": _format_optional_float(spec.get("threshold_theta_high")),
				"threshold_state_alpha": _format_optional_float(spec.get("threshold_state_alpha")),
				"threshold_a_hi": _format_optional_float(spec.get("threshold_a_hi")),
				"threshold_b_hi": _format_optional_float(spec.get("threshold_b_hi")),
				"regime_high_share": _format_optional_float(diagnostics.get("regime_high_share")),
				"regime_switches": "" if diagnostics.get("regime_switches") is None else int(diagnostics["regime_switches"]),
				"mean_threshold_state": _format_optional_float(diagnostics.get("mean_threshold_state")),
				"switch_active": _yes_no(switch_active),
				"control_env_gamma": f"{control_env_gamma:.6f}",
				"control_stage3_score": f"{control_stage3_score:.6f}",
				"gamma_uplift_ratio_vs_control_seed": _format_ratio(
					_gamma_uplift_ratio(baseline_gamma=control_env_gamma, candidate_gamma=candidate_env_gamma)
				),
				"stage3_uplift_vs_control_seed": f"{candidate_stage3_score - control_stage3_score:.6f}",
				"has_level3_seed": _yes_no(int(row["cycle_level"]) >= 3),
				"verdict": "control" if is_control else str(seed_decision["verdict"]),
				"out_csv": row["out_csv"],
				"provenance_json": row["provenance_json"],
			}
		)
	return rows


def _build_h55_combined_row(
	*,
	summary: Mapping[str, Any],
	control_summary: Mapping[str, Any],
	seed_diagnostics: Mapping[int, Mapping[str, Any]],
	spec: Mapping[str, Any],
	selection_strength: float,
	memory_kernel: int,
	is_control: bool,
) -> dict[str, Any]:
	control_mean_env_gamma = float(control_summary["mean_env_gamma"])
	control_mean_stage3_score = float(control_summary["mean_stage3_score"])
	mean_env_gamma = float(summary["mean_env_gamma"])
	mean_stage3_score = float(summary["mean_stage3_score"])
	diagnostics = _aggregate_threshold_diagnostics(seed_diagnostics)
	if is_control:
		switch_pass = ""
		gamma_pass = ""
		stage3_pass = ""
		level3_pass = ""
		full_pass = ""
		verdict = "control"
	else:
		point_decision = _h55_point_verdict(
			control_mean_env_gamma=control_mean_env_gamma,
			control_mean_stage3_score=control_mean_stage3_score,
			candidate_mean_env_gamma=mean_env_gamma,
			candidate_mean_stage3_score=mean_stage3_score,
			p_level_3=float(summary["p_level_3"]),
			p_switched_seed=float(diagnostics["p_switched_seed"] or 0.0),
		)
		switch_pass = _yes_no(point_decision["switch_pass"])
		gamma_pass = _yes_no(point_decision["gamma_pass"])
		stage3_pass = _yes_no(point_decision["stage3_pass"])
		level3_pass = _yes_no(point_decision["level3_pass"])
		full_pass = _yes_no(point_decision["full_pass"])
		verdict = str(point_decision["verdict"])
	return {
		"condition": str(spec["condition"]),
		"is_control": _yes_no(is_control),
		"n_seeds": summary["n_seeds"],
		"mean_cycle_level": summary["mean_cycle_level"],
		"mean_stage3_score": summary["mean_stage3_score"],
		"mean_turn_strength": summary["mean_turn_strength"],
		"mean_env_gamma": summary["mean_env_gamma"],
		"mean_success_rate": summary["mean_success_rate"],
		"level_counts_json": summary["level_counts_json"],
		"p_level_ge_2": summary["p_level_ge_2"],
		"p_level_3": summary["p_level_3"],
		"payoff_mode": str(spec["payoff_mode"]),
		"threshold_trigger": str(spec.get("threshold_trigger") or ""),
		"threshold_theta": _format_optional_float(spec.get("threshold_theta")),
		"threshold_theta_low": _format_optional_float(spec.get("threshold_theta_low")),
		"threshold_theta_high": _format_optional_float(spec.get("threshold_theta_high")),
		"threshold_state_alpha": _format_optional_float(spec.get("threshold_state_alpha")),
		"threshold_a_hi": _format_optional_float(spec.get("threshold_a_hi")),
		"threshold_b_hi": _format_optional_float(spec.get("threshold_b_hi")),
		"mean_regime_high_share": _format_optional_float(diagnostics.get("mean_regime_high_share")),
		"mean_regime_switches": _format_optional_float(diagnostics.get("mean_regime_switches")),
		"p_switched_seed": _format_optional_float(diagnostics.get("p_switched_seed")),
		"mean_threshold_state": _format_optional_float(diagnostics.get("mean_threshold_state")),
		"switch_pass": switch_pass,
		"control_mean_env_gamma": f"{control_mean_env_gamma:.6f}",
		"control_mean_stage3_score": f"{control_mean_stage3_score:.6f}",
		"gamma_uplift_ratio_vs_control": _format_ratio(
			_gamma_uplift_ratio(baseline_gamma=control_mean_env_gamma, candidate_gamma=mean_env_gamma)
		),
		"stage3_uplift_vs_control": f"{mean_stage3_score - control_mean_stage3_score:.6f}",
		"gamma_pass": gamma_pass,
		"stage3_pass": stage3_pass,
		"level3_pass": level3_pass,
		"full_pass": full_pass,
		"verdict": verdict,
		"events_json": summary["events_json"],
		"personality_mode": summary["personality_mode"],
		"personality_jitter": summary["personality_jitter"],
		"selection_strength": f"{float(selection_strength):.6f}",
		"memory_kernel": int(memory_kernel),
		"out_dir": summary["out_dir"],
	}


def decide_h55_overall(*, combined_rows: list[dict[str, Any]]) -> dict[str, Any]:
	candidate_rows = [row for row in combined_rows if row["is_control"] == "no"]
	pass_rows = [row for row in candidate_rows if row["verdict"] == "pass"]
	weak_rows = [row for row in candidate_rows if row["verdict"] == "weak_positive"]
	fail_rows = [row for row in candidate_rows if row["verdict"] == "fail"]
	if pass_rows:
		decision = "pass"
	elif weak_rows:
		decision = "weak_positive"
	else:
		decision = "fail"
	return {
		"decision": decision,
		"pass_points": len(pass_rows),
		"weak_positive_points": len(weak_rows),
		"fail_points": len(fail_rows),
	}


def _write_h55_decision(
	path: Path,
	*,
	overall_decision: Mapping[str, Any],
	control_row: Mapping[str, Any],
	combined_rows: list[dict[str, Any]],
) -> None:
	lines = [
		"# H5.5 Personality + Event-Driven Nonlinear Payoff Decision",
		"",
		f"- decision: {overall_decision['decision']}",
		f"- pass_points: {overall_decision['pass_points']}",
		f"- weak_positive_points: {overall_decision['weak_positive_points']}",
		f"- fail_points: {overall_decision['fail_points']}",
		f"- matched_control_mean_env_gamma: {control_row['mean_env_gamma']}",
		f"- matched_control_mean_stage3_score: {control_row['mean_stage3_score']}",
		f"- matched_control_level_counts: {control_row['level_counts_json']}",
		"",
		"## Point Summary",
		"",
	]
	for row in combined_rows:
		if row["is_control"] == "yes":
			continue
		lines.append(
			"- "
			f"{row['condition']}: verdict={row['verdict']} "
			f"gamma={row['mean_env_gamma']} ratio={row['gamma_uplift_ratio_vs_control']} "
			f"stage3={row['mean_stage3_score']} uplift={row['stage3_uplift_vs_control']} "
			f"p_level3={row['p_level_3']} p_switched_seed={row['p_switched_seed']} "
			f"mean_regime_switches={row['mean_regime_switches']}"
		)
	lines.extend(
		[
			"",
			"## Stop Rule",
			"",
			"- 任何一個 nonlinear cell 同時達到 gamma uplift、stage3 uplift、Level 3 與 switch evidence，H5.5 就算 pass。",
			"- 若只有部分 uplift 或只有 switch evidence，記為 weak_positive。",
			"- 若所有 cell 都沒有 uplift 且沒有 switch evidence，H5.5 視為 fail。",
		]
	)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_h55_protocol(args: argparse.Namespace) -> int:
	write_reduced_events_json(args.events_json, source_json=args.source_events_json)
	seeds = _parse_seeds(args.seeds)
	out_root = Path(args.h55_out_root)
	control_spec = {
		"condition": "matrix_control",
		"payoff_mode": "matrix_ab",
		"threshold_trigger": "",
		"threshold_theta": None,
		"threshold_theta_low": None,
		"threshold_theta_high": None,
		"threshold_state_alpha": None,
		"threshold_a_hi": None,
		"threshold_b_hi": None,
	}
	control_dir = out_root / str(control_spec["condition"])
	control_rows, control_summary, control_seed_csvs, _control_actions, control_diagnostics = run_condition(
		condition=str(control_spec["condition"]),
		world_mode=str(args.world_mode),
		seeds=seeds,
		out_dir=control_dir,
		events_json=args.events_json,
		players=args.players,
		rounds=args.rounds,
		selection_strength=args.selection_strength,
		init_bias=args.init_bias,
		memory_kernel=args.memory_kernel,
		burn_in=args.burn_in,
		tail=args.tail,
		jitter=args.jitter,
		a=args.a,
		b=args.b,
		cross=args.cross,
		payoff_mode="matrix_ab",
		evolution_mode="sampled",
		sampled_inertia=0.0,
	)
	seed_summary_rows = _build_h55_seed_rows(
		metrics_rows=control_rows,
		control_metrics_rows=control_rows,
		seed_diagnostics=control_diagnostics,
		spec=control_spec,
		is_control=True,
	)
	combined_rows = [
		_build_h55_combined_row(
			summary=control_summary,
			control_summary=control_summary,
			seed_diagnostics=control_diagnostics,
			spec=control_spec,
			selection_strength=float(args.selection_strength),
			memory_kernel=int(args.memory_kernel),
			is_control=True,
		)
	]
	for spec in _h55_candidate_specs(args):
		candidate_dir = out_root / str(spec["condition"])
		candidate_rows, candidate_summary, candidate_seed_csvs, _candidate_actions, candidate_diagnostics = run_condition(
			condition=str(spec["condition"]),
			world_mode=str(args.world_mode),
			seeds=seeds,
			out_dir=candidate_dir,
			events_json=args.events_json,
			players=args.players,
			rounds=args.rounds,
			selection_strength=args.selection_strength,
			init_bias=args.init_bias,
			memory_kernel=args.memory_kernel,
			burn_in=args.burn_in,
			tail=args.tail,
			jitter=args.jitter,
			a=args.a,
			b=args.b,
			cross=args.cross,
			payoff_mode=str(spec["payoff_mode"]),
			threshold_theta=float(spec["threshold_theta"]),
			threshold_theta_low=spec["threshold_theta_low"],
			threshold_theta_high=spec["threshold_theta_high"],
			threshold_trigger=str(spec["threshold_trigger"]),
			threshold_state_alpha=float(spec["threshold_state_alpha"]),
			threshold_a_hi=float(spec["threshold_a_hi"]),
			threshold_b_hi=float(spec["threshold_b_hi"]),
			evolution_mode="sampled",
			sampled_inertia=0.0,
		)
		_rewrite_h54_provenance_with_control(
			seed_csvs=candidate_seed_csvs,
			control_seed_csvs=control_seed_csvs,
			events_json=args.events_json,
		)
		seed_summary_rows.extend(
			_build_h55_seed_rows(
				metrics_rows=candidate_rows,
				control_metrics_rows=control_rows,
				seed_diagnostics=candidate_diagnostics,
				spec=spec,
				is_control=False,
			)
		)
		combined_rows.append(
			_build_h55_combined_row(
				summary=candidate_summary,
				control_summary=control_summary,
				seed_diagnostics=candidate_diagnostics,
				spec=spec,
				selection_strength=float(args.selection_strength),
				memory_kernel=int(args.memory_kernel),
				is_control=False,
			)
		)
	overall_decision = decide_h55_overall(combined_rows=combined_rows)
	_write_tsv(args.h55_summary_tsv, fieldnames=H55_METRIC_FIELDNAMES, rows=seed_summary_rows)
	_write_tsv(args.h55_combined_tsv, fieldnames=H55_COMBINED_FIELDNAMES, rows=combined_rows)
	_write_h55_decision(
		args.h55_decision_md,
		overall_decision=overall_decision,
		control_row=combined_rows[0],
		combined_rows=combined_rows,
	)
	print(f"decision={overall_decision['decision']}")
	print(f"summary_tsv={args.h55_summary_tsv}")
	print(f"combined_tsv={args.h55_combined_tsv}")
	print(f"decision_md={args.h55_decision_md}")
	return 0


def main() -> int:
	args = _parse_args()
	if str(args.protocol) == "h54":
		return _run_h54_protocol(args)
	if str(args.protocol) == "h55":
		return _run_h55_protocol(args)
	if str(args.protocol) == "h55r":
		return _run_h55r_protocol(args)
	return _run_gate1_protocol(args)


if __name__ == "__main__":
	raise SystemExit(main())