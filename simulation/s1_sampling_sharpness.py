"""S1 sampling sharpness (power-law β) – G2 scout harness.

Follows SDD §(S1) protocol lock.  CLI-compatible with the B/A-series harness pattern.
No G1 deterministic gate (mean-field has no players; sampling β is N/A).
"""
from __future__ import annotations

import argparse
import csv
import json
from math import atan2, exp, isfinite, log, sqrt
from pathlib import Path
from typing import Any

from analysis.cycle_metrics import classify_cycle_level
from analysis.decay_rate import estimate_decay_gamma
from analysis.event_provenance_summary import summarize_event_provenance
from simulation.run_simulation import (
	DEFAULT_EVENTS_JSON,
	SimConfig,
	_initial_weights,
	_write_timeseries_csv,
	simulate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "s1_sampling"
DEFAULT_SUMMARY_TSV = REPO_ROOT / "outputs" / "s1_sampling_summary.tsv"
DEFAULT_COMBINED_TSV = REPO_ROOT / "outputs" / "s1_sampling_combined.tsv"
DEFAULT_DECISION_MD = REPO_ROOT / "outputs" / "s1_sampling_decision.md"

GATE = "g2_sampled"

SUMMARY_FIELDNAMES = [
	"condition",
	"sampling_beta",
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
	"mean_strategy_entropy",
	"phase_amplitude_stability",
	"out_csv",
	"provenance_json",
]

COMBINED_FIELDNAMES = [
	"condition",
	"sampling_beta",
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
	"mean_strategy_entropy",
	"phase_amplitude_stability",
	"short_scout_pass",
	"hard_stop_fail",
	"longer_confirm_candidate",
	"verdict",
	"representative_seed",
	"representative_simplex_png",
	"representative_phase_amplitude_png",
	"enable_events",
	"events_json",
	"memory_kernel",
	"players",
	"rounds",
	"out_dir",
]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


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


def _seed_metrics(
	rows: list[dict[str, Any]],
	*,
	burn_in: int,
	tail: int,
	eta: float,
	corr_threshold: float,
) -> dict[str, float | int]:
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


def _tail_begin(n_rows: int, *, burn_in: int, tail: int) -> int:
	if n_rows <= 0:
		return 0
	return max(int(burn_in), n_rows - int(tail))


def _slice_tail(rows: list[dict[str, Any]], *, burn_in: int, tail: int) -> list[dict[str, Any]]:
	begin = _tail_begin(len(rows), burn_in=int(burn_in), tail=int(tail))
	return rows[begin:]


def _condition_name(*, beta: float) -> str:
	token = f"{float(beta):.2f}".replace(".", "p")
	return f"{GATE}_beta{token}"


def _control_condition_name() -> str:
	return _condition_name(beta=1.0)


def _simplex_xy(row: dict[str, Any]) -> tuple[float, float]:
	p_a = float(row["p_aggressive"])
	p_d = float(row["p_defensive"])
	p_b = float(row["p_balanced"])
	return (p_d - p_b, p_a - 0.5 * (p_d + p_b))


def _phase_angle_from_row(row: dict[str, Any]) -> float:
	p_a = float(row["p_aggressive"])
	p_d = float(row["p_defensive"])
	p_b = float(row["p_balanced"])
	return float(atan2(sqrt(3.0) * (p_d - p_b), 2.0 * p_a - p_d - p_b))


def _amplitude_from_row(row: dict[str, Any]) -> float:
	p_a = float(row["p_aggressive"])
	p_d = float(row["p_defensive"])
	p_b = float(row["p_balanced"])
	return float(((p_a - (1.0 / 3.0)) ** 2 + (p_d - (1.0 / 3.0)) ** 2 + (p_b - (1.0 / 3.0)) ** 2) ** 0.5)


def _phase_amplitude_stability(rows: list[dict[str, Any]], *, burn_in: int, tail: int) -> float:
	tail_rows = _slice_tail(rows, burn_in=int(burn_in), tail=int(tail))
	if not tail_rows:
		return 0.0
	amplitudes = [_amplitude_from_row(row) for row in tail_rows]
	mean_amp = _safe_mean(amplitudes)
	if mean_amp <= 1e-12:
		return 0.0
	variance = _safe_mean([(float(value) - mean_amp) ** 2 for value in amplitudes])
	stability = 1.0 - min(1.0, (variance ** 0.5) / mean_amp)
	return float(max(0.0, stability))


def _strategy_entropy_from_row(row: dict[str, Any]) -> float:
	"""Shannon entropy of the popularity distribution for this round."""
	keys = ["p_aggressive", "p_defensive", "p_balanced"]
	probs = [max(1e-30, float(row[k])) for k in keys]
	total = sum(probs)
	entropy = 0.0
	for p in probs:
		q = p / total
		if q > 0.0:
			entropy -= q * log(q)
	return float(entropy)


# ---------------------------------------------------------------------------
# Round-level diagnostics callback
# ---------------------------------------------------------------------------


class _S1RoundDiagnostics:
	"""Collect per-round sampling diagnostics during simulate() via round_callback."""

	def __init__(self, *, beta: float, seed: int) -> None:
		self.beta = float(beta)
		self.seed = int(seed)
		self.records: list[dict[str, Any]] = []

	def callback(
		self,
		round_index: int,
		cfg: SimConfig,
		players: list[object],
		_dungeon: object,
		step_records: list[dict[str, object]],
		row: dict[str, Any],
	) -> None:
		entropy = _strategy_entropy_from_row(row)
		self.records.append({
			"strategy_entropy": entropy,
		})

	def finalize(self, *, rows: list[dict[str, Any]], burn_in: int, tail: int) -> dict[str, Any]:
		begin = _tail_begin(len(rows), burn_in=int(burn_in), tail=int(tail))
		window = self.records[begin:]
		return {
			"mean_strategy_entropy": _safe_mean([float(r["strategy_entropy"]) for r in window]),
			"phase_amplitude_stability": _phase_amplitude_stability(rows, burn_in=int(burn_in), tail=int(tail)),
		}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _write_simplex_plot(
	rows: list[dict[str, Any]],
	*,
	out_png: Path,
	title: str,
	burn_in: int,
	tail: int,
) -> None:
	try:
		import matplotlib.pyplot as plt
		from matplotlib.collections import LineCollection
		import numpy as np
	except ImportError as exc:
		raise RuntimeError("matplotlib is required to write S1 simplex plots") from exc

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


def _write_phase_amplitude_plot(
	rows: list[dict[str, Any]],
	*,
	out_png: Path,
	title: str,
	burn_in: int,
	tail: int,
) -> None:
	try:
		import matplotlib.pyplot as plt
		import numpy as np
	except ImportError as exc:
		raise RuntimeError("matplotlib is required to write S1 phase-amplitude plots") from exc

	tail_rows = _slice_tail(rows, burn_in=int(burn_in), tail=int(tail))
	if not tail_rows:
		return
	rounds = np.array([int(row["round"]) for row in tail_rows], dtype=float)
	phases = np.unwrap(np.array([_phase_angle_from_row(row) for row in tail_rows], dtype=float))
	amplitudes = np.array([_amplitude_from_row(row) for row in tail_rows], dtype=float)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, constrained_layout=True)
	axes[0].plot(rounds, phases, color="#1f77b4", linewidth=1.8)
	axes[0].set_ylabel("phase angle")
	axes[0].set_title(title)
	axes[0].grid(alpha=0.25)
	axes[1].plot(rounds, amplitudes, color="#d62728", linewidth=1.8)
	axes[1].set_ylabel("amplitude")
	axes[1].set_xlabel("round")
	axes[1].grid(alpha=0.25)
	fig.savefig(out_png, dpi=160)
	plt.close(fig)


# ---------------------------------------------------------------------------
# Condition summary builder
# ---------------------------------------------------------------------------


def _build_condition_summary(
	*,
	condition: str,
	beta: float,
	metrics_rows: list[dict[str, Any]],
	per_seed_rows: list[dict[str, Any]],
	out_dir: Path,
	players: int,
	rounds: int,
	memory_kernel: int,
	enable_events: bool,
	events_json: Path | None,
) -> dict[str, Any]:
	levels = [int(row["cycle_level"]) for row in metrics_rows]
	level_counts = {level: levels.count(level) for level in range(4)}
	den = float(len(metrics_rows)) or 1.0
	is_ctrl = abs(float(beta) - 1.0) < 1e-12
	return {
		"condition": condition,
		"sampling_beta": _format_float(beta),
		"is_control": _yes_no(is_ctrl),
		"n_seeds": len(metrics_rows),
		"mean_cycle_level": _format_float(_safe_mean([float(l) for l in levels])),
		"mean_stage3_score": _format_float(_safe_mean([float(r["stage3_score"]) for r in metrics_rows])),
		"mean_turn_strength": _format_float(_safe_mean([float(r["turn_strength"]) for r in metrics_rows])),
		"mean_env_gamma": _format_float(_safe_mean([float(r["env_gamma"]) for r in metrics_rows])),
		"level_counts_json": json.dumps(level_counts, sort_keys=True),
		"p_level_3": _format_float(sum(1 for l in levels if l >= 3) / den),
		"level3_seed_count": sum(1 for l in levels if l >= 3),
		"control_mean_env_gamma": "",
		"control_mean_stage3_score": "",
		"gamma_uplift_ratio_vs_control": "",
		"stage3_uplift_vs_control": "",
		"mean_strategy_entropy": _format_float(_safe_mean([float(r["mean_strategy_entropy"]) for r in per_seed_rows])),
		"phase_amplitude_stability": _format_float(_safe_mean([float(r["phase_amplitude_stability"]) for r in per_seed_rows])),
		"short_scout_pass": "",
		"hard_stop_fail": "",
		"longer_confirm_candidate": "no",
		"verdict": "control" if is_ctrl else "pending",
		"representative_seed": "",
		"representative_simplex_png": "",
		"representative_phase_amplitude_png": "",
		"enable_events": _yes_no(enable_events),
		"events_json": str(events_json) if events_json is not None else "",
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
			-int(row["seed"]),
		),
	)


# ---------------------------------------------------------------------------
# Decision writer
# ---------------------------------------------------------------------------


def _write_decision(
	path: Path,
	*,
	combined_rows: list[dict[str, Any]],
	recommended_candidates: list[dict[str, Any]],
	close_s1: bool,
) -> None:
	lines = [
		"# S1 Sampling Sharpness Decision",
		"",
		"## G2 Sampled Short Scout",
		"",
	]
	for row in combined_rows:
		lines.append(
			f"- {row['condition']}: beta={row['sampling_beta']}"
			f" level3_seed_count={row['level3_seed_count']}"
			f" stage3_uplift={row['stage3_uplift_vs_control']}"
			f" mean_strategy_entropy={row['mean_strategy_entropy']}"
			f" phase_amplitude_stability={row['phase_amplitude_stability']}"
			f" short_scout_pass={row['short_scout_pass']}"
			f" hard_stop_fail={row['hard_stop_fail']}"
			f" verdict={row['verdict']}"
		)
	lines.extend(["", "## Recommendation", ""])
	if not recommended_candidates:
		lines.append("- longer_confirm_candidate: none")
	else:
		for row in recommended_candidates:
			lines.append(
				f"- longer_confirm_candidate: {row['condition']}"
				f" beta={row['sampling_beta']}"
				f" stage3_uplift={row['stage3_uplift_vs_control']}"
				f" mean_strategy_entropy={row['mean_strategy_entropy']}"
				f" representative_seed={row['representative_seed']}"
			)
	lines.extend(["", "## Stop Rule", ""])
	lines.append("- G2: 若所有 sampling_beta != 1.0 的 active cells 全部 0/3 Level 3 且 mean_stage3_score uplift < 0.02，S1 直接 closure。")
	if close_s1:
		lines.append("- overall_verdict: close_s1")
	else:
		lines.append("- overall_verdict: keep_s1_open")
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main scout
# ---------------------------------------------------------------------------


def run_s1_scout(
	*,
	seeds: list[int],
	betas: list[float],
	out_root: Path,
	summary_tsv: Path,
	combined_tsv: Path,
	decision_md: Path,
	players: int = 300,
	rounds: int = 3000,
	burn_in: int = 1000,
	tail: int = 1000,
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
	if not any(abs(float(beta) - 1.0) < 1e-12 for beta in betas):
		raise ValueError("S1 short scout requires betas to include 1.0 for matched controls")
	out_root.mkdir(parents=True, exist_ok=True)

	all_summary_rows: list[dict[str, Any]] = []
	all_combined_rows: list[dict[str, Any]] = []
	condition_seed_rows: dict[str, list[dict[str, Any]]] = {}
	condition_timeseries: dict[str, dict[int, list[dict[str, Any]]]] = {}

	for beta in betas:
		beta_f = float(beta)
		condition = _condition_name(beta=beta_f)
		out_dir = out_root / condition
		out_dir.mkdir(parents=True, exist_ok=True)
		per_seed_rows: list[dict[str, Any]] = []
		metric_rows: list[dict[str, Any]] = []
		seed_rows_map: dict[int, list[dict[str, Any]]] = {}

		for seed in seeds:
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
				evolution_mode="sampled",
				payoff_lag=1,
				selection_strength=0.06,
				enable_events=bool(enable_events),
				events_json=events_json if enable_events else None,
				out_csv=out_dir / f"seed{seed}.csv",
				memory_kernel=int(memory_kernel),
				sampling_beta=beta_f,
			)
			round_diagnostics = _S1RoundDiagnostics(beta=beta_f, seed=int(seed))
			strategy_space, rows = simulate(
				cfg,
				round_callback=round_diagnostics.callback,
			)
			_write_timeseries_csv(cfg.out_csv, strategy_space=strategy_space, rows=rows)
			seed_rows_map[int(seed)] = rows

			seed_metric = _seed_metrics(
				rows,
				burn_in=int(burn_in),
				tail=int(tail),
				eta=float(eta),
				corr_threshold=float(corr_threshold),
			)
			diag_stats = round_diagnostics.finalize(rows=rows, burn_in=int(burn_in), tail=int(tail))

			provenance = {
				"condition": condition,
				"sampling_beta": beta_f,
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
					"events_json": str(events_json) if events_json is not None else None,
				},
				"round_diagnostics": {
					"mean_strategy_entropy": float(diag_stats["mean_strategy_entropy"]),
					"phase_amplitude_stability": float(diag_stats["phase_amplitude_stability"]),
				},
			}
			if bool(enable_events) and events_json is not None:
				provenance["event_provenance"] = summarize_event_provenance(
					cfg.out_csv, events_json=events_json,
				)
			provenance_path = out_dir / f"seed{seed}_provenance.json"
			provenance_path.write_text(
				json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
				encoding="utf-8",
			)

			per_seed_rows.append({
				"condition": condition,
				"sampling_beta": _format_float(beta_f),
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
				"mean_strategy_entropy": _format_float(diag_stats["mean_strategy_entropy"]),
				"phase_amplitude_stability": _format_float(diag_stats["phase_amplitude_stability"]),
				"out_csv": str(cfg.out_csv),
				"provenance_json": str(provenance_path),
			})
			metric_rows.append(per_seed_rows[-1])

		condition_seed_rows[condition] = per_seed_rows
		condition_timeseries[condition] = seed_rows_map
		all_combined_rows.append(
			_build_condition_summary(
				condition=condition,
				beta=beta_f,
				metrics_rows=metric_rows,
				per_seed_rows=per_seed_rows,
				out_dir=out_dir,
				players=int(players),
				rounds=int(rounds),
				memory_kernel=int(memory_kernel),
				enable_events=bool(enable_events),
				events_json=events_json if enable_events else None,
			)
		)

	# Compute uplift vs control
	control_cond = _control_condition_name()
	control_combined = next(
		(r for r in all_combined_rows if str(r["condition"]) == control_cond), None,
	)
	control_mean_env_gamma = float(control_combined["mean_env_gamma"]) if control_combined else 0.0
	control_mean_stage3_score = float(control_combined["mean_stage3_score"]) if control_combined else 0.0

	for combined_row in all_combined_rows:
		beta_val = float(combined_row["sampling_beta"])
		mean_env_gamma = float(combined_row["mean_env_gamma"])
		mean_stage3_score = float(combined_row["mean_stage3_score"])
		level3_seed_count = int(combined_row["level3_seed_count"])
		is_control = abs(beta_val - 1.0) < 1e-12
		stage3_uplift = mean_stage3_score - control_mean_stage3_score
		gamma_ratio = _gamma_uplift_ratio(
			baseline_gamma=control_mean_env_gamma, candidate_gamma=mean_env_gamma,
		)
		combined_row["control_mean_env_gamma"] = _format_float(control_mean_env_gamma)
		combined_row["control_mean_stage3_score"] = _format_float(control_mean_stage3_score)
		combined_row["gamma_uplift_ratio_vs_control"] = _format_ratio(1.0 if is_control else gamma_ratio)
		combined_row["stage3_uplift_vs_control"] = _format_float(0.0 if is_control else stage3_uplift)

		short_scout_pass = (not is_control) and level3_seed_count >= 1 and stage3_uplift >= 0.02
		hard_stop_fail = (
			not is_control
			and level3_seed_count == 0
			and stage3_uplift < 0.02
		)
		if is_control:
			verdict = "control"
		elif short_scout_pass:
			verdict = "pass"
		elif hard_stop_fail:
			verdict = "fail"
		else:
			verdict = "weak_positive"
		combined_row["short_scout_pass"] = _yes_no(short_scout_pass)
		combined_row["hard_stop_fail"] = _yes_no(hard_stop_fail)
		combined_row["longer_confirm_candidate"] = _yes_no(verdict == "pass")
		combined_row["verdict"] = verdict

	# Representative seeds + plots
	for combined_row in all_combined_rows:
		condition = str(combined_row["condition"])
		representative_row = _representative_seed_row(condition_seed_rows[condition])
		representative_seed = int(representative_row["seed"])
		representative_png = Path(combined_row["out_dir"]) / f"seed{representative_seed}_simplex.png"
		representative_phase_png = Path(combined_row["out_dir"]) / f"seed{representative_seed}_phase_amplitude.png"
		_write_simplex_plot(
			condition_timeseries[condition][representative_seed],
			out_png=representative_png,
			title=f"{condition} seed={representative_seed}",
			burn_in=int(burn_in),
			tail=int(tail),
		)
		_write_phase_amplitude_plot(
			condition_timeseries[condition][representative_seed],
			out_png=representative_phase_png,
			title=f"{condition} seed={representative_seed}",
			burn_in=int(burn_in),
			tail=int(tail),
		)
		combined_row["representative_seed"] = representative_seed
		combined_row["representative_simplex_png"] = str(representative_png)
		combined_row["representative_phase_amplitude_png"] = str(representative_phase_png)

	# Per-seed control uplift
	control_per_seed = condition_seed_rows.get(control_cond, [])
	control_by_seed = {int(r["seed"]): r for r in control_per_seed}
	for condition, per_seed_rows in condition_seed_rows.items():
		for row in per_seed_rows:
			seed = int(row["seed"])
			ctrl = control_by_seed.get(seed)
			if ctrl is None:
				continue
			row["control_env_gamma"] = ctrl["env_gamma"]
			row["control_stage3_score"] = ctrl["stage3_score"]
			is_ctrl_cond = str(row["condition"]) == control_cond
			if is_ctrl_cond:
				row["gamma_uplift_ratio_vs_control_seed"] = _format_ratio(1.0)
				row["stage3_uplift_vs_control_seed"] = _format_float(0.0)
			else:
				row["gamma_uplift_ratio_vs_control_seed"] = _format_ratio(
					_gamma_uplift_ratio(
						baseline_gamma=float(ctrl["env_gamma"]),
						candidate_gamma=float(row["env_gamma"]),
					)
				)
				row["stage3_uplift_vs_control_seed"] = _format_float(
					float(row["stage3_score"]) - float(ctrl["stage3_score"])
				)
		all_summary_rows.extend(per_seed_rows)

	# Closure decision
	recommended_candidates = [
		row for row in all_combined_rows if str(row["verdict"]) == "pass"
	]
	recommended_candidates.sort(
		key=lambda row: (
			float(row["stage3_uplift_vs_control"]),
			int(row["level3_seed_count"]),
		),
		reverse=True,
	)
	active_rows = [
		row
		for row in all_combined_rows
		if str(row["is_control"]) == "no"
	]
	close_s1 = bool(active_rows) and all(str(row["hard_stop_fail"]) == "yes" for row in active_rows)

	_write_tsv(summary_tsv, fieldnames=SUMMARY_FIELDNAMES, rows=all_summary_rows)
	_write_tsv(combined_tsv, fieldnames=COMBINED_FIELDNAMES, rows=all_combined_rows)
	_write_decision(
		decision_md,
		combined_rows=all_combined_rows,
		recommended_candidates=recommended_candidates,
		close_s1=close_s1,
	)

	return {
		"summary_tsv": str(summary_tsv),
		"combined_tsv": str(combined_tsv),
		"decision_md": str(decision_md),
		"recommended_candidates": [dict(row) for row in recommended_candidates],
		"close_s1": bool(close_s1),
	}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
	parser = argparse.ArgumentParser(description="S1 sampling sharpness (power-law β) G2 scout harness")
	parser.add_argument("--seeds", type=str, default="45,47,49")
	parser.add_argument("--betas", type=str, default="0.5,1.0,2.0,5.0,50.0")
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

	result = run_s1_scout(
		seeds=_parse_seeds(args.seeds),
		betas=_parse_float_list(args.betas),
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
		events_json=(args.events_json if not bool(args.disable_events) else None),
	)
	print(f"summary_tsv={result['summary_tsv']}")
	print(f"combined_tsv={result['combined_tsv']}")
	print(f"decision_md={result['decision_md']}")
	print(f"close_s1={result['close_s1']}")


if __name__ == "__main__":
	main()
