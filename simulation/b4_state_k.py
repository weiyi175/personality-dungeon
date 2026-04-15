from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from analysis.event_provenance_summary import summarize_event_provenance
from analysis.cycle_metrics import classify_cycle_level
from analysis.decay_rate import estimate_decay_gamma
from simulation.run_simulation import DEFAULT_EVENTS_JSON, SimConfig, _write_timeseries_csv, simulate


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "b4_state_k"
DEFAULT_SUMMARY_TSV = REPO_ROOT / "outputs" / "b4_state_k_summary.tsv"
DEFAULT_COMBINED_TSV = REPO_ROOT / "outputs" / "b4_state_k_combined.tsv"
DEFAULT_DECISION_MD = REPO_ROOT / "outputs" / "b4_state_k_decision.md"

LOWER_K_CLAMP = 0.03
UPPER_K_CLAMP = 0.09
K_CLAMP_TOL = 1e-12

SUMMARY_FIELDNAMES = [
	"condition",
	"beta_state_k",
	"k_base",
	"seed",
	"cycle_level",
	"stage3_score",
	"turn_strength",
	"env_gamma",
	"env_gamma_r2",
	"env_gamma_n_peaks",
	"control_env_gamma",
	"control_stage3_score",
	"gamma_uplift_ratio_vs_control_seed",
	"stage3_uplift_vs_control_seed",
	"has_level3_seed",
	"k_clamped_ratio",
	"mean_effective_k",
	"min_effective_k",
	"max_effective_k",
	"mean_state_dominance",
	"mean_state_k_factor",
	"out_csv",
	"provenance_json",
]

COMBINED_FIELDNAMES = [
	"condition",
	"beta_state_k",
	"k_base",
	"is_control",
	"n_seeds",
	"mean_cycle_level",
	"mean_stage3_score",
	"mean_turn_strength",
	"mean_env_gamma",
	"level_counts_json",
	"p_level_3",
	"level3_seed_count",
	"control_mean_env_gamma",
	"control_mean_stage3_score",
	"gamma_uplift_ratio_vs_control",
	"stage3_uplift_vs_control",
	"mean_k_clamped_ratio",
	"mean_effective_k",
	"mean_state_dominance",
	"mean_state_k_factor",
	"gamma_gate_pass",
	"short_scout_pass",
	"hard_stop_fail",
	"longer_confirm_candidate",
	"verdict",
	"representative_seed",
	"representative_simplex_png",
	"enable_events",
	"events_json",
	"selection_strength",
	"memory_kernel",
	"players",
	"rounds",
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


def _safe_mean(values: list[float]) -> float:
	if not values:
		return 0.0
	return float(sum(values) / float(len(values)))


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


def _write_tsv(path: Path, *, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
		writer.writeheader()
		writer.writerows(rows)


def _seed_metrics(rows: list[dict[str, Any]], *, burn_in: int, tail: int, eta: float, corr_threshold: float) -> dict[str, float | int]:
	series_map = {
		"aggressive": [float(row["p_aggressive"]) for row in rows],
		"defensive": [float(row["p_defensive"]) for row in rows],
		"balanced": [float(row["p_balanced"]) for row in rows],
	}
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
	return {
		"cycle_level": int(cycle.level),
		"stage3_score": float(cycle.stage3.score) if cycle.stage3 is not None else 0.0,
		"turn_strength": float(cycle.stage3.turn_strength) if cycle.stage3 is not None else 0.0,
		"env_gamma": float(fit.gamma) if fit is not None else 0.0,
		"env_gamma_r2": float(fit.r2) if fit is not None else 0.0,
		"env_gamma_n_peaks": int(fit.n_peaks) if fit is not None else 0,
	}


def _condition_name(*, beta_state_k: float, k_base: float) -> str:
	def _token(value: float) -> str:
		return f"{value:.2f}".replace("-", "m").replace(".", "p")

	return f"beta{_token(beta_state_k)}_k{_token(k_base)}"


def _control_condition_name(*, k_base: float) -> str:
	return _condition_name(beta_state_k=0.0, k_base=k_base)


def _slice_tail(rows: list[dict[str, Any]], *, burn_in: int, tail: int) -> list[dict[str, Any]]:
	if not rows:
		return []
	begin = max(int(burn_in), len(rows) - int(tail))
	return rows[begin:]


def _simplex_xy(row: dict[str, Any]) -> tuple[float, float]:
	p_a = float(row["p_aggressive"])
	p_d = float(row["p_defensive"])
	p_b = float(row["p_balanced"])
	return (p_d - p_b, p_a - 0.5 * (p_d + p_b))


def _write_simplex_plot(rows: list[dict[str, Any]], *, out_png: Path, title: str, burn_in: int, tail: int) -> None:
	try:
		import matplotlib.pyplot as plt
		from matplotlib.collections import LineCollection
		import numpy as np
	except ImportError as exc:
		raise RuntimeError("matplotlib is required to write B4 simplex plots") from exc

	tail_rows = _slice_tail(rows, burn_in=int(burn_in), tail=int(tail))
	points = [_simplex_xy(row) for row in tail_rows]
	if len(points) < 2:
		return
	xs = np.array([pt[0] for pt in points], dtype=float)
	ys = np.array([pt[1] for pt in points], dtype=float)
	segments = np.stack(
		[
			np.column_stack([xs[:-1], ys[:-1]]),
			np.column_stack([xs[1:], ys[1:]]),
		],
		axis=1,
	)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
	line = LineCollection(segments, cmap="viridis", linewidths=2.0)
	line.set_array(np.linspace(0.0, 1.0, len(segments)))
	ax.add_collection(line)
	ax.scatter(xs[0], ys[0], color="#1f77b4", s=45, marker="o", label="start")
	ax.scatter(xs[-1], ys[-1], color="#d62728", s=55, marker="X", label="end")
	ax.axhline(0.0, color="#cccccc", linewidth=0.8)
	ax.axvline(0.0, color="#cccccc", linewidth=0.8)
	ax.set_title(title)
	ax.set_xlabel("p_defensive - p_balanced")
	ax.set_ylabel("p_aggressive - (p_defensive + p_balanced) / 2")
	ax.legend(loc="best")
	ax.set_aspect("equal", adjustable="box")
	fig.colorbar(line, ax=ax, fraction=0.046, pad=0.04, label="tail time")
	fig.savefig(out_png, dpi=160)
	plt.close(fig)


class _B4RoundDiagnostics:
	def __init__(self) -> None:
		self.total_player_rounds = 0
		self.clamped_player_rounds = 0
		self.k_values: list[float] = []
		self.state_dominances: list[float] = []
		self.state_factors: list[float] = []

	def callback(
		self,
		_round_index: int,
		cfg: SimConfig,
		players: list[object],
		_dungeon: object,
		_step_records: list[dict[str, object]],
		row: dict[str, Any],
	) -> None:
		fallback_state_dominance = max(
			float(row["p_aggressive"]),
			float(row["p_defensive"]),
			float(row["p_balanced"]),
		)
		for player in players:
			k_value = float(getattr(player, "personality_selection_strength", cfg.selection_strength))
			state_dominance = float(getattr(player, "personality_state_dominance", fallback_state_dominance))
			state_factor = float(getattr(player, "personality_state_k_factor", 1.0))
			self.total_player_rounds += 1
			if abs(k_value - LOWER_K_CLAMP) <= K_CLAMP_TOL or abs(k_value - UPPER_K_CLAMP) <= K_CLAMP_TOL:
				self.clamped_player_rounds += 1
			self.k_values.append(k_value)
			self.state_dominances.append(state_dominance)
			self.state_factors.append(state_factor)

	def finalize(self) -> dict[str, float]:
		total = float(self.total_player_rounds) if self.total_player_rounds > 0 else 1.0
		return {
			"k_clamped_ratio": float(self.clamped_player_rounds) / total,
			"mean_effective_k": _safe_mean(self.k_values),
			"min_effective_k": min(self.k_values) if self.k_values else 0.0,
			"max_effective_k": max(self.k_values) if self.k_values else 0.0,
			"mean_state_dominance": _safe_mean(self.state_dominances),
			"mean_state_k_factor": _safe_mean(self.state_factors),
		}


def _build_condition_summary(
	*,
	condition: str,
	beta_state_k: float,
	k_base: float,
	metrics_rows: list[dict[str, Any]],
	per_seed_rows: list[dict[str, Any]],
	out_dir: Path,
	players: int,
	rounds: int,
	selection_strength: float,
	memory_kernel: int,
	enable_events: bool,
	events_json: Path | None,
) -> dict[str, Any]:
	levels = [int(row["cycle_level"]) for row in metrics_rows]
	level_counts = {level: levels.count(level) for level in range(4)}
	den = float(len(metrics_rows)) or 1.0
	return {
		"condition": condition,
		"beta_state_k": _format_float(beta_state_k),
		"k_base": _format_float(k_base),
		"is_control": _yes_no(float(beta_state_k) == 0.0),
		"n_seeds": len(metrics_rows),
		"mean_cycle_level": _format_float(_safe_mean([float(level) for level in levels])),
		"mean_stage3_score": _format_float(_safe_mean([float(row["stage3_score"]) for row in metrics_rows])),
		"mean_turn_strength": _format_float(_safe_mean([float(row["turn_strength"]) for row in metrics_rows])),
		"mean_env_gamma": _format_float(_safe_mean([float(row["env_gamma"]) for row in metrics_rows])),
		"level_counts_json": json.dumps(level_counts, sort_keys=True),
		"p_level_3": _format_float(sum(1 for level in levels if level >= 3) / den),
		"level3_seed_count": sum(1 for level in levels if level >= 3),
		"mean_k_clamped_ratio": _format_float(_safe_mean([float(row["k_clamped_ratio"]) for row in per_seed_rows])),
		"mean_effective_k": _format_float(_safe_mean([float(row["mean_effective_k"]) for row in per_seed_rows])),
		"mean_state_dominance": _format_float(_safe_mean([float(row["mean_state_dominance"]) for row in per_seed_rows])),
		"mean_state_k_factor": _format_float(_safe_mean([float(row["mean_state_k_factor"]) for row in per_seed_rows])),
		"control_mean_env_gamma": "",
		"control_mean_stage3_score": "",
		"gamma_uplift_ratio_vs_control": "",
		"stage3_uplift_vs_control": "",
		"gamma_gate_pass": "",
		"short_scout_pass": "",
		"hard_stop_fail": "",
		"longer_confirm_candidate": "no",
		"verdict": "control" if float(beta_state_k) == 0.0 else "pending",
		"representative_seed": "",
		"representative_simplex_png": "",
		"enable_events": _yes_no(enable_events),
		"events_json": str(events_json) if events_json is not None else "",
		"selection_strength": _format_float(selection_strength),
		"memory_kernel": int(memory_kernel),
		"players": int(players),
		"rounds": int(rounds),
		"out_dir": str(out_dir),
	}


def _representative_seed_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
	return max(
		rows,
		key=lambda row: (
			int(row["cycle_level"]),
			float(row["stage3_score"]),
			-float(abs(float(row["env_gamma"]))),
			-int(row["seed"]),
		),
	)


def _write_decision(
	path: Path,
	*,
	combined_rows: list[dict[str, Any]],
	recommended_candidates: list[dict[str, Any]],
) -> None:
	lines = [
		"# B4 Short Scout Decision",
		"",
		"## Conditions",
		"",
	]
	for row in combined_rows:
		lines.append(
			f"- {row['condition']}: beta_state_k={row['beta_state_k']} k_base={row['k_base']} level3_seed_count={row['level3_seed_count']} stage3_uplift={row['stage3_uplift_vs_control']} gamma_ratio={row['gamma_uplift_ratio_vs_control']} k_clamped_ratio={row['mean_k_clamped_ratio']} short_scout_pass={row['short_scout_pass']} hard_stop_fail={row['hard_stop_fail']} verdict={row['verdict']} simplex_png={row['representative_simplex_png']}"
		)
	lines.extend(["", "## Recommendation", ""])
	if not recommended_candidates:
		lines.append("- longer_confirm_candidate: none")
	else:
		for row in recommended_candidates:
			lines.append(
				f"- longer_confirm_candidate: {row['condition']} stage3_uplift={row['stage3_uplift_vs_control']} gamma_ratio={row['gamma_uplift_ratio_vs_control']} k_clamped_ratio={row['mean_k_clamped_ratio']} representative_seed={row['representative_seed']}"
			)
	lines.extend(
		[
			"",
			"## Stop Rule",
			"",
			"- 若 0/3 seeds 達 Level 3 且 mean_stage3_score uplift < 0.02，該 cell 直接記為 fail，不做 Longer Confirm。",
			"- 若 k_clamped_ratio > 0.3，視為高 clamp 飽和；若 > 0.5，視為嚴重飽和，可在同一 B4 family 內升級到更寬 clamp 或 exponential。",
		]
	)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_b4_scout(
	*,
	seeds: list[int],
	beta_state_ks: list[float],
	k_bases: list[float],
	out_root: Path,
	summary_tsv: Path,
	combined_tsv: Path,
	decision_md: Path,
	players: int = 300,
	rounds: int = 3000,
	burn_in: int = 1000,
	tail: int = 1000,
	selection_strength: float | None = None,
	memory_kernel: int = 3,
	init_bias: float = 0.12,
	a: float = 1.0,
	b: float = 0.9,
	cross: float = 0.20,
	eta: float = 0.55,
	corr_threshold: float = 0.09,
	enable_events: bool = True,
	events_json: Path | None = DEFAULT_EVENTS_JSON,
) -> dict[str, Any]:
	if 0.0 not in {float(value) for value in beta_state_ks}:
		raise ValueError("B4 short scout requires beta_state_ks to include 0.0 for matched controls")
	out_root.mkdir(parents=True, exist_ok=True)
	all_summary_rows: list[dict[str, Any]] = []
	all_combined_rows: list[dict[str, Any]] = []
	condition_seed_rows: dict[str, list[dict[str, Any]]] = {}
	condition_timeseries: dict[str, dict[int, list[dict[str, Any]]]] = {}

	for k_base in k_bases:
		for beta_state_k in beta_state_ks:
			condition = _condition_name(beta_state_k=float(beta_state_k), k_base=float(k_base))
			out_dir = out_root / condition
			out_dir.mkdir(parents=True, exist_ok=True)
			per_seed_rows: list[dict[str, Any]] = []
			metric_rows: list[dict[str, Any]] = []
			seed_rows_map: dict[int, list[dict[str, Any]]] = {}

			for seed in seeds:
				diagnostics = _B4RoundDiagnostics()
				cfg = SimConfig(
					n_players=int(players),
					n_rounds=int(rounds),
					seed=int(seed),
					payoff_mode="matrix_ab",
					popularity_mode="sampled",
					gamma=0.1,
					epsilon=0.0,
					a=float(a),
					b=float(b),
					matrix_cross_coupling=float(cross),
					init_bias=float(init_bias),
					evolution_mode="personality_coupled",
					payoff_lag=1,
					selection_strength=float(k_base if selection_strength is None else selection_strength),
					enable_events=bool(enable_events),
					events_json=events_json if enable_events else None,
					out_csv=out_dir / f"seed{seed}.csv",
					memory_kernel=int(memory_kernel),
					personality_coupling_mu_base=0.0,
					personality_coupling_lambda_mu=0.0,
					personality_coupling_lambda_k=0.0,
					personality_coupling_beta_state_k=float(beta_state_k),
				)
				strategy_space, rows = simulate(cfg, round_callback=diagnostics.callback)
				_write_timeseries_csv(cfg.out_csv, strategy_space=strategy_space, rows=rows)
				seed_rows_map[int(seed)] = rows
				seed_metric = _seed_metrics(rows, burn_in=int(burn_in), tail=int(tail), eta=float(eta), corr_threshold=float(corr_threshold))
				diag_stats = diagnostics.finalize()
				provenance = {
					"condition": condition,
					"beta_state_k": float(beta_state_k),
					"k_base": float(k_base),
					"config": {
						"players": int(players),
						"rounds": int(rounds),
						"burn_in": int(burn_in),
						"tail": int(tail),
						"memory_kernel": int(memory_kernel),
						"init_bias": float(init_bias),
						"a": float(a),
						"b": float(b),
						"matrix_cross_coupling": float(cross),
						"enable_events": bool(enable_events),
						"events_json": str(events_json) if enable_events and events_json is not None else None,
					},
					"round_diagnostics": diag_stats,
				}
				if enable_events and events_json is not None:
					provenance["event_provenance"] = summarize_event_provenance(cfg.out_csv, events_json=events_json)
				provenance_path = out_dir / f"seed{seed}_provenance.json"
				provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

				per_seed_rows.append(
					{
						"condition": condition,
						"beta_state_k": _format_float(beta_state_k),
						"k_base": _format_float(k_base),
						"seed": int(seed),
						"cycle_level": int(seed_metric["cycle_level"]),
						"stage3_score": _format_float(seed_metric["stage3_score"]),
						"turn_strength": _format_float(seed_metric["turn_strength"]),
						"env_gamma": _format_float(seed_metric["env_gamma"]),
						"env_gamma_r2": _format_float(seed_metric["env_gamma_r2"]),
						"env_gamma_n_peaks": int(seed_metric["env_gamma_n_peaks"]),
						"control_env_gamma": "",
						"control_stage3_score": "",
						"gamma_uplift_ratio_vs_control_seed": "",
						"stage3_uplift_vs_control_seed": "",
						"has_level3_seed": _yes_no(int(seed_metric["cycle_level"]) >= 3),
						"k_clamped_ratio": _format_float(diag_stats["k_clamped_ratio"]),
						"mean_effective_k": _format_float(diag_stats["mean_effective_k"]),
						"min_effective_k": _format_float(diag_stats["min_effective_k"]),
						"max_effective_k": _format_float(diag_stats["max_effective_k"]),
						"mean_state_dominance": _format_float(diag_stats["mean_state_dominance"]),
						"mean_state_k_factor": _format_float(diag_stats["mean_state_k_factor"]),
						"out_csv": str(cfg.out_csv),
						"provenance_json": str(provenance_path),
					}
				)
				metric_rows.append(per_seed_rows[-1])

			condition_seed_rows[condition] = per_seed_rows
			condition_timeseries[condition] = seed_rows_map
			all_combined_rows.append(
				_build_condition_summary(
					condition=condition,
					beta_state_k=float(beta_state_k),
					k_base=float(k_base),
					metrics_rows=metric_rows,
					per_seed_rows=per_seed_rows,
					out_dir=out_dir,
					players=int(players),
					rounds=int(rounds),
					selection_strength=float(k_base if selection_strength is None else selection_strength),
					memory_kernel=int(memory_kernel),
					enable_events=bool(enable_events),
					events_json=events_json if enable_events else None,
				)
			)

	for combined_row in all_combined_rows:
		condition = str(combined_row["condition"])
		k_base = float(combined_row["k_base"])
		is_control = str(combined_row["is_control"]) == "yes"
		control_condition = _control_condition_name(k_base=k_base)
		control_row = next(row for row in all_combined_rows if str(row["condition"]) == control_condition)
		control_mean_env_gamma = float(control_row["mean_env_gamma"])
		control_mean_stage3_score = float(control_row["mean_stage3_score"])
		mean_env_gamma = float(combined_row["mean_env_gamma"])
		mean_stage3_score = float(combined_row["mean_stage3_score"])
		level3_seed_count = int(combined_row["level3_seed_count"])
		stage3_uplift = mean_stage3_score - control_mean_stage3_score
		gamma_ratio = _gamma_uplift_ratio(baseline_gamma=control_mean_env_gamma, candidate_gamma=mean_env_gamma)
		gamma_gate_pass = abs(mean_env_gamma) <= 5.0e-4
		short_scout_pass = (level3_seed_count >= 1) and gamma_gate_pass and (not is_control)
		hard_stop_fail = (level3_seed_count == 0) and (stage3_uplift < 0.02) and (not is_control)
		if is_control:
			verdict = "control"
		elif short_scout_pass:
			verdict = "pass"
		elif hard_stop_fail:
			verdict = "fail"
		else:
			verdict = "weak_positive"
		combined_row["control_mean_env_gamma"] = _format_float(control_mean_env_gamma)
		combined_row["control_mean_stage3_score"] = _format_float(control_mean_stage3_score)
		combined_row["gamma_uplift_ratio_vs_control"] = _format_ratio(1.0 if is_control else gamma_ratio)
		combined_row["stage3_uplift_vs_control"] = _format_float(0.0 if is_control else stage3_uplift)
		combined_row["gamma_gate_pass"] = _yes_no(True if is_control else gamma_gate_pass)
		combined_row["short_scout_pass"] = _yes_no(False if is_control else short_scout_pass)
		combined_row["hard_stop_fail"] = _yes_no(False if is_control else hard_stop_fail)
		combined_row["longer_confirm_candidate"] = _yes_no(verdict == "pass")
		combined_row["verdict"] = verdict

		representative_row = _representative_seed_row(condition_seed_rows[condition])
		representative_seed = int(representative_row["seed"])
		representative_png = Path(combined_row["out_dir"]) / f"seed{representative_seed}_simplex.png"
		_write_simplex_plot(
			condition_timeseries[condition][representative_seed],
			out_png=representative_png,
			title=f"{condition} seed={representative_seed}",
			burn_in=int(burn_in),
			tail=int(tail),
		)
		combined_row["representative_seed"] = representative_seed
		combined_row["representative_simplex_png"] = str(representative_png)

	for condition, per_seed_rows in condition_seed_rows.items():
		k_base = float(next(row for row in all_combined_rows if str(row["condition"]) == condition)["k_base"])
		control_rows = condition_seed_rows[_control_condition_name(k_base=k_base)]
		control_by_seed = {int(row["seed"]): row for row in control_rows}
		for row in per_seed_rows:
			seed = int(row["seed"])
			control_seed_row = control_by_seed[seed]
			row["control_env_gamma"] = control_seed_row["env_gamma"]
			row["control_stage3_score"] = control_seed_row["stage3_score"]
			if str(row["condition"]) == _control_condition_name(k_base=k_base):
				row["gamma_uplift_ratio_vs_control_seed"] = _format_ratio(1.0)
				row["stage3_uplift_vs_control_seed"] = _format_float(0.0)
			else:
				row["gamma_uplift_ratio_vs_control_seed"] = _format_ratio(
					_gamma_uplift_ratio(
						baseline_gamma=float(control_seed_row["env_gamma"]),
						candidate_gamma=float(row["env_gamma"]),
					)
				)
				row["stage3_uplift_vs_control_seed"] = _format_float(
					float(row["stage3_score"]) - float(control_seed_row["stage3_score"])
				)
		all_summary_rows.extend(per_seed_rows)

	recommended_candidates = [
		row
		for row in all_combined_rows
		if str(row["verdict"]) == "pass"
	]
	recommended_candidates.sort(
		key=lambda row: (
			float(row["stage3_uplift_vs_control"]),
			int(row["level3_seed_count"]),
			-float(abs(float(row["mean_env_gamma"]))),
		),
		reverse=True,
	)

	_write_tsv(summary_tsv, fieldnames=SUMMARY_FIELDNAMES, rows=all_summary_rows)
	_write_tsv(combined_tsv, fieldnames=COMBINED_FIELDNAMES, rows=all_combined_rows)
	_write_decision(decision_md, combined_rows=all_combined_rows, recommended_candidates=recommended_candidates)

	return {
		"summary_tsv": str(summary_tsv),
		"combined_tsv": str(combined_tsv),
		"decision_md": str(decision_md),
		"recommended_candidates": [dict(row) for row in recommended_candidates],
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="B4 state-dependent k short-scout harness")
	parser.add_argument("--seeds", type=str, default="45,47,49")
	parser.add_argument("--beta-state-ks", type=str, default="0.0,0.3,0.6,1.0")
	parser.add_argument("--k-bases", type=str, default="0.06,0.08")
	parser.add_argument("--players", type=int, default=300)
	parser.add_argument("--rounds", type=int, default=3000)
	parser.add_argument("--burn-in", type=int, default=1000)
	parser.add_argument("--tail", type=int, default=1000)
	parser.add_argument("--memory-kernel", type=int, default=3)
	parser.add_argument("--init-bias", type=float, default=0.12)
	parser.add_argument("--a", type=float, default=1.0)
	parser.add_argument("--b", type=float, default=0.9)
	parser.add_argument("--matrix-cross-coupling", type=float, default=0.20)
	parser.add_argument("--eta", type=float, default=0.55)
	parser.add_argument("--corr-threshold", type=float, default=0.09)
	parser.add_argument("--disable-events", action="store_true")
	parser.add_argument("--events-json", type=Path, default=DEFAULT_EVENTS_JSON)
	parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
	parser.add_argument("--summary-tsv", type=Path, default=DEFAULT_SUMMARY_TSV)
	parser.add_argument("--combined-tsv", type=Path, default=DEFAULT_COMBINED_TSV)
	parser.add_argument("--decision-md", type=Path, default=DEFAULT_DECISION_MD)
	args = parser.parse_args()

	result = run_b4_scout(
		seeds=_parse_seeds(args.seeds),
		beta_state_ks=_parse_float_list(args.beta_state_ks),
		k_bases=_parse_float_list(args.k_bases),
		out_root=args.out_root,
		summary_tsv=args.summary_tsv,
		combined_tsv=args.combined_tsv,
		decision_md=args.decision_md,
		players=int(args.players),
		rounds=int(args.rounds),
		burn_in=int(args.burn_in),
		tail=int(args.tail),
		memory_kernel=int(args.memory_kernel),
		init_bias=float(args.init_bias),
		a=float(args.a),
		b=float(args.b),
		cross=float(args.matrix_cross_coupling),
		eta=float(args.eta),
		corr_threshold=float(args.corr_threshold),
		enable_events=not bool(args.disable_events),
		events_json=args.events_json,
	)
	print(f"summary_tsv={result['summary_tsv']}")
	print(f"combined_tsv={result['combined_tsv']}")
	print(f"decision_md={result['decision_md']}")


if __name__ == "__main__":
	main()