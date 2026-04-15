"""B1 tangential projection replicator – G0 / G1 / G2 scout harness.

Follows SDD §2.6 B1 protocol lock.  CLI-compatible with the B-series harness pattern.
"""
from __future__ import annotations

import argparse
import csv
import json
from math import atan2, isfinite, pi, sqrt
from pathlib import Path
from typing import Any

from analysis.cycle_metrics import classify_cycle_level
from analysis.decay_rate import estimate_decay_gamma
from analysis.event_provenance_summary import summarize_event_provenance
from evolution.replicator_dynamics import (
	sampled_growth_vector,
	tangential_projection_replicator_step,
	_tangential_projection,
	_sampled_simplex_vector,
)
from simulation.run_simulation import (
	DEFAULT_EVENTS_JSON,
	SimConfig,
	_average_simplex_history,
	_initial_weights,
	_matrix_ab_payoff_vec,
	_normalize_simplex,
	_write_timeseries_csv,
	simulate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "b1_tangential"
DEFAULT_SUMMARY_TSV = REPO_ROOT / "outputs" / "b1_tangential_summary.tsv"
DEFAULT_COMBINED_TSV = REPO_ROOT / "outputs" / "b1_tangential_combined.tsv"
DEFAULT_DECISION_MD = REPO_ROOT / "outputs" / "b1_tangential_decision.md"

G1_GATE = "g1_mean_field"
G2_GATE = "g2_sampled"

SUMMARY_FIELDNAMES = [
	"gate",
	"condition",
	"tangential_alpha",
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
	"g1_gate_pass",
	"mean_radial_norm",
	"mean_tangential_norm",
	"mean_tangential_ratio",
	"mean_growth_angle_rad",
	"phase_amplitude_stability",
	"out_csv",
	"provenance_json",
]

COMBINED_FIELDNAMES = [
	"gate",
	"condition",
	"tangential_alpha",
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
	"g1_turn_ratio_vs_control",
	"g1_level_gate_pass",
	"g1_turn_gate_pass",
	"g1_gate_pass",
	"mean_radial_norm",
	"mean_tangential_norm",
	"mean_tangential_ratio",
	"mean_growth_angle_rad",
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
# helpers (mirrors B5 harness)
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


def _mean_ratio(values: list[float]) -> float:
	if not values:
		return 0.0
	if any(value == float("inf") for value in values):
		return float("inf")
	if any(value == float("-inf") for value in values):
		return float("-inf")
	finite = [float(value) for value in values if isfinite(float(value))]
	if not finite:
		return 0.0
	return _safe_mean(finite)


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


def _turn_ratio(*, baseline_turn: float, candidate_turn: float) -> float:
	base = float(baseline_turn)
	alt = float(candidate_turn)
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


def _tail_begin(n_rows: int, *, burn_in: int, tail: int) -> int:
	if n_rows <= 0:
		return 0
	return max(int(burn_in), n_rows - int(tail))


def _slice_tail(rows: list[dict[str, Any]], *, burn_in: int, tail: int) -> list[dict[str, Any]]:
	begin = _tail_begin(len(rows), burn_in=int(burn_in), tail=int(tail))
	return rows[begin:]


def _condition_name(*, gate: str, alpha: float) -> str:
	token = f"{float(alpha):.3f}".replace("-", "m").replace(".", "p")
	return f"{gate}_alpha{token}"


def _control_condition_name(*, gate: str) -> str:
	return _condition_name(gate=gate, alpha=0.0)


def _simplex_from_row(row: dict[str, Any]) -> dict[str, float]:
	return {
		"aggressive": float(row["p_aggressive"]),
		"defensive": float(row["p_defensive"]),
		"balanced": float(row["p_balanced"]),
	}


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


# ---------------------------------------------------------------------------
# Sampled-path round-level diagnostics (B1 specific)
# ---------------------------------------------------------------------------

class _B1SampledRoundDiagnostics:
	"""Collect per-round tangential projection diagnostics during G2 simulate()."""

	def __init__(self, *, alpha: float) -> None:
		self.alpha = float(alpha)
		self.records: list[dict[str, Any]] = []

	def callback(
		self,
		_round_index: int,
		_cfg: SimConfig,
		players: list[object],
		_dungeon: object,
		_step_records: list[dict[str, object]],
		row: dict[str, Any],
	) -> None:
		strategy_space = ["aggressive", "defensive", "balanced"]
		simplex = _simplex_from_row(row)
		growth = sampled_growth_vector(players, strategy_space)
		_, diagnostics = _tangential_projection(growth, simplex, strategy_space, self.alpha)
		self.records.append(diagnostics)

	def finalize(self, *, rows: list[dict[str, Any]], burn_in: int, tail: int) -> dict[str, Any]:
		begin = _tail_begin(len(rows), burn_in=int(burn_in), tail=int(tail))
		window = self.records[begin:]
		return {
			"mean_radial_norm": _safe_mean([float(r["radial_norm"]) for r in window]),
			"mean_tangential_norm": _safe_mean([float(r["tangential_norm"]) for r in window]),
			"mean_tangential_ratio": _safe_mean([float(r["tangential_ratio"]) for r in window]),
			"mean_growth_angle_rad": _safe_mean([float(r["growth_angle_rad"]) for r in window]),
			"phase_amplitude_stability": _phase_amplitude_stability(rows, burn_in=int(burn_in), tail=int(tail)),
		}


# ---------------------------------------------------------------------------
# Mean-field diagnostics (G1 gate)
# ---------------------------------------------------------------------------

def _compute_mean_field_round_diagnostics(
	*,
	cfg: SimConfig,
	rows: list[dict[str, Any]],
	alpha: float,
	burn_in: int,
	tail: int,
) -> dict[str, Any]:
	strategy_space = ["aggressive", "defensive", "balanced"]
	weights = _initial_weights(strategy_space=strategy_space, init_bias=float(cfg.init_bias))
	mean_weight = sum(float(v) for v in weights.values()) / float(len(weights)) if weights else 1.0
	weights = {s: float(weights[s]) / float(mean_weight) for s in strategy_space}
	x_cur = _normalize_simplex(weights)
	x_history: list[dict[str, float]] = [dict(x_cur)]
	records: list[dict[str, Any]] = []
	for row in rows:
		x_pay = _average_simplex_history(
			x_history,
			strategy_space=strategy_space,
			kernel=int(cfg.memory_kernel),
			lag=int(cfg.payoff_lag),
		)
		payoff = _matrix_ab_payoff_vec(
			strategy_space=strategy_space,
			a=float(cfg.a),
			b=float(cfg.b),
			matrix_cross_coupling=float(cfg.matrix_cross_coupling),
			x=x_pay,
		)
		u_bar = sum(float(x_cur[s]) * float(payoff[s]) for s in strategy_space)
		growth = {s: float(payoff[s]) - float(u_bar) for s in strategy_space}
		_, diagnostics = _tangential_projection(growth, x_cur, strategy_space, float(alpha))
		records.append(diagnostics)
		x_cur = _simplex_from_row(row)
		x_history.append(dict(x_cur))
	begin = _tail_begin(len(rows), burn_in=int(burn_in), tail=int(tail))
	window = records[begin:]
	return {
		"mean_radial_norm": _safe_mean([float(r["radial_norm"]) for r in window]),
		"mean_tangential_norm": _safe_mean([float(r["tangential_norm"]) for r in window]),
		"mean_tangential_ratio": _safe_mean([float(r["tangential_ratio"]) for r in window]),
		"mean_growth_angle_rad": _safe_mean([float(r["growth_angle_rad"]) for r in window]),
		"phase_amplitude_stability": _phase_amplitude_stability(rows, burn_in=int(burn_in), tail=int(tail)),
	}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _write_simplex_plot(rows: list[dict[str, Any]], *, out_png: Path, title: str, burn_in: int, tail: int) -> None:
	try:
		import matplotlib.pyplot as plt
		from matplotlib.collections import LineCollection
		import numpy as np
	except ImportError as exc:
		raise RuntimeError("matplotlib is required to write B1 simplex plots") from exc

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


def _write_phase_amplitude_plot(rows: list[dict[str, Any]], *, out_png: Path, title: str, burn_in: int, tail: int) -> None:
	try:
		import matplotlib.pyplot as plt
		import numpy as np
	except ImportError as exc:
		raise RuntimeError("matplotlib is required to write B1 phase-amplitude plots") from exc

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
	gate: str,
	condition: str,
	alpha: float,
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
	return {
		"gate": str(gate),
		"condition": condition,
		"tangential_alpha": _format_float(alpha),
		"is_control": _yes_no(float(alpha) == 0.0),
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
		"g1_turn_ratio_vs_control": "",
		"g1_level_gate_pass": "",
		"g1_turn_gate_pass": "",
		"g1_gate_pass": "",
		"mean_radial_norm": _format_float(_safe_mean([float(r["mean_radial_norm"]) for r in per_seed_rows])),
		"mean_tangential_norm": _format_float(_safe_mean([float(r["mean_tangential_norm"]) for r in per_seed_rows])),
		"mean_tangential_ratio": _format_float(_safe_mean([float(r["mean_tangential_ratio"]) for r in per_seed_rows])),
		"mean_growth_angle_rad": _format_float(_safe_mean([float(r["mean_growth_angle_rad"]) for r in per_seed_rows])),
		"phase_amplitude_stability": _format_float(_safe_mean([float(r["phase_amplitude_stability"]) for r in per_seed_rows])),
		"short_scout_pass": "",
		"hard_stop_fail": "",
		"longer_confirm_candidate": "no",
		"verdict": "control" if float(alpha) == 0.0 else "pending",
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
			float(row["mean_tangential_ratio"]),
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
	close_b1: bool,
) -> None:
	g1_rows = [row for row in combined_rows if str(row["gate"]) == G1_GATE]
	g2_rows = [row for row in combined_rows if str(row["gate"]) == G2_GATE]
	lines = [
		"# B1 Tangential Projection Decision",
		"",
		"## G1 Deterministic Gate",
		"",
	]
	for row in g1_rows:
		lines.append(
			f"- {row['condition']}: alpha={row['tangential_alpha']}"
			f" turn_ratio_vs_control={row['g1_turn_ratio_vs_control']}"
			f" g1_level_gate_pass={row['g1_level_gate_pass']}"
			f" g1_turn_gate_pass={row['g1_turn_gate_pass']}"
			f" g1_gate_pass={row['g1_gate_pass']}"
			f" mean_tangential_ratio={row['mean_tangential_ratio']}"
			f" verdict={row['verdict']}"
		)
	lines.extend(["", "## G2 Short Scout", ""])
	for row in g2_rows:
		lines.append(
			f"- {row['condition']}: alpha={row['tangential_alpha']}"
			f" g1_gate_pass={row['g1_gate_pass']}"
			f" level3_seed_count={row['level3_seed_count']}"
			f" stage3_uplift={row['stage3_uplift_vs_control']}"
			f" mean_radial_norm={row['mean_radial_norm']}"
			f" mean_tangential_norm={row['mean_tangential_norm']}"
			f" mean_tangential_ratio={row['mean_tangential_ratio']}"
			f" mean_growth_angle_rad={row['mean_growth_angle_rad']}"
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
				f" alpha={row['tangential_alpha']}"
				f" stage3_uplift={row['stage3_uplift_vs_control']}"
				f" mean_tangential_ratio={row['mean_tangential_ratio']}"
				f" representative_seed={row['representative_seed']}"
			)
	lines.extend(["", "## Stop Rule", ""])
	lines.append("- G1: 若 turn_strength < 0.92 × baseline 或 cycle level 掉出 3，該 alpha 不得進 G2。")
	lines.append("- G2: 若 alpha <= 1.0 的 active cells 全部 0/3 Level 3 且 mean_stage3_score uplift < 0.02，B1 直接 closure。")
	if close_b1:
		lines.append("- overall_verdict: close_b1")
	else:
		lines.append("- overall_verdict: keep_b1_open")
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main scout
# ---------------------------------------------------------------------------

def run_b1_scout(
	*,
	seeds: list[int],
	tangential_alphas: list[float],
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
	if 0.0 not in {float(alpha) for alpha in tangential_alphas}:
		raise ValueError("B1 short scout requires tangential_alphas to include 0.0 for matched controls")
	out_root.mkdir(parents=True, exist_ok=True)
	all_summary_rows: list[dict[str, Any]] = []
	all_combined_rows: list[dict[str, Any]] = []
	condition_seed_rows: dict[str, list[dict[str, Any]]] = {}
	condition_timeseries: dict[str, dict[int, list[dict[str, Any]]]] = {}

	gate_specs = [
		{
			"gate": G1_GATE,
			"n_players": 0,
			"popularity_mode": "expected",
			"evolution_mode": "mean_field",
			"enable_events": False,
			"events_json": None,
		},
		{
			"gate": G2_GATE,
			"n_players": int(players),
			"popularity_mode": "sampled",
			"evolution_mode": "sampled",
			"enable_events": bool(enable_events),
			"events_json": events_json if enable_events else None,
		},
	]

	for gate_spec in gate_specs:
		gate = str(gate_spec["gate"])
		for alpha in tangential_alphas:
			alpha_f = float(alpha)
			condition = _condition_name(gate=gate, alpha=alpha_f)
			out_dir = out_root / condition
			out_dir.mkdir(parents=True, exist_ok=True)
			per_seed_rows: list[dict[str, Any]] = []
			metric_rows: list[dict[str, Any]] = []
			seed_rows_map: dict[int, list[dict[str, Any]]] = {}

			for seed in seeds:
				cfg = SimConfig(
					n_players=int(gate_spec["n_players"]),
					n_rounds=int(rounds),
					seed=int(seed),
					payoff_mode="matrix_ab",
					popularity_mode=str(gate_spec["popularity_mode"]),
					gamma=0.1,
					epsilon=0.0,
					a=float(a),
					b=float(b),
					matrix_cross_coupling=float(cross),
					init_bias=float(init_bias),
					evolution_mode=str(gate_spec["evolution_mode"]),
					payoff_lag=1,
					selection_strength=0.06,
					enable_events=bool(gate_spec["enable_events"]),
					events_json=gate_spec["events_json"],
					out_csv=out_dir / f"seed{seed}.csv",
					memory_kernel=int(memory_kernel),
					# G1 mean-field path doesn't use tangential_alpha in simulate();
					# tangential projection diagnostics are computed post-hoc.
					tangential_alpha=alpha_f if gate == G2_GATE else 0.0,
				)
				round_diagnostics = _B1SampledRoundDiagnostics(alpha=alpha_f) if gate == G2_GATE else None
				strategy_space, rows = simulate(
					cfg,
					round_callback=(round_diagnostics.callback if round_diagnostics is not None else None),
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
				if gate == G2_GATE:
					diag_stats = round_diagnostics.finalize(rows=rows, burn_in=int(burn_in), tail=int(tail))
				else:
					diag_stats = _compute_mean_field_round_diagnostics(
						cfg=cfg, rows=rows, alpha=alpha_f,
						burn_in=int(burn_in), tail=int(tail),
					)
				provenance = {
					"gate": gate,
					"condition": condition,
					"tangential_alpha": alpha_f,
					"config": {
						"players": int(gate_spec["n_players"]),
						"rounds": int(rounds),
						"burn_in": int(burn_in),
						"tail": int(tail),
						"memory_kernel": int(memory_kernel),
						"init_bias": float(init_bias),
						"a": float(a),
						"b": float(b),
						"matrix_cross_coupling": float(cross),
						"enable_events": bool(gate_spec["enable_events"]),
						"events_json": str(gate_spec["events_json"]) if gate_spec["events_json"] is not None else None,
					},
					"round_diagnostics": {
						"mean_radial_norm": float(diag_stats["mean_radial_norm"]),
						"mean_tangential_norm": float(diag_stats["mean_tangential_norm"]),
						"mean_tangential_ratio": float(diag_stats["mean_tangential_ratio"]),
						"mean_growth_angle_rad": float(diag_stats["mean_growth_angle_rad"]),
						"phase_amplitude_stability": float(diag_stats["phase_amplitude_stability"]),
					},
				}
				if gate == G2_GATE and bool(gate_spec["enable_events"]) and gate_spec["events_json"] is not None:
					provenance["event_provenance"] = summarize_event_provenance(cfg.out_csv, events_json=gate_spec["events_json"])
				provenance_path = out_dir / f"seed{seed}_provenance.json"
				provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

				per_seed_rows.append(
					{
						"gate": gate,
						"condition": condition,
						"tangential_alpha": _format_float(alpha_f),
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
						"g1_gate_pass": "",
						"mean_radial_norm": _format_float(diag_stats["mean_radial_norm"]),
						"mean_tangential_norm": _format_float(diag_stats["mean_tangential_norm"]),
						"mean_tangential_ratio": _format_float(diag_stats["mean_tangential_ratio"]),
						"mean_growth_angle_rad": _format_float(diag_stats["mean_growth_angle_rad"]),
						"phase_amplitude_stability": _format_float(diag_stats["phase_amplitude_stability"]),
						"out_csv": str(cfg.out_csv),
						"provenance_json": str(provenance_path),
					}
				)
				metric_rows.append(per_seed_rows[-1])

			condition_seed_rows[condition] = per_seed_rows
			condition_timeseries[condition] = seed_rows_map
			all_combined_rows.append(
				_build_condition_summary(
					gate=gate,
					condition=condition,
					alpha=alpha_f,
					metrics_rows=metric_rows,
					per_seed_rows=per_seed_rows,
					out_dir=out_dir,
					players=int(gate_spec["n_players"]),
					rounds=int(rounds),
					memory_kernel=int(memory_kernel),
					enable_events=bool(gate_spec["enable_events"]),
					events_json=gate_spec["events_json"],
				)
			)

	# G1 gate evaluation
	g1_gate_pass_by_alpha: dict[float, bool] = {}
	for combined_row in all_combined_rows:
		gate = str(combined_row["gate"])
		alpha_val = float(combined_row["tangential_alpha"])
		control_row = next(
			r for r in all_combined_rows
			if str(r["gate"]) == gate and str(r["condition"]) == _control_condition_name(gate=gate)
		)
		control_mean_env_gamma = float(control_row["mean_env_gamma"])
		control_mean_stage3_score = float(control_row["mean_stage3_score"])
		control_mean_turn_strength = float(control_row["mean_turn_strength"])
		mean_env_gamma = float(combined_row["mean_env_gamma"])
		mean_stage3_score = float(combined_row["mean_stage3_score"])
		mean_turn_strength = float(combined_row["mean_turn_strength"])
		level3_seed_count = int(combined_row["level3_seed_count"])
		stage3_uplift = mean_stage3_score - control_mean_stage3_score
		gamma_ratio = _gamma_uplift_ratio(baseline_gamma=control_mean_env_gamma, candidate_gamma=mean_env_gamma)
		combined_row["control_mean_env_gamma"] = _format_float(control_mean_env_gamma)
		combined_row["control_mean_stage3_score"] = _format_float(control_mean_stage3_score)
		combined_row["gamma_uplift_ratio_vs_control"] = _format_ratio(1.0 if alpha_val == 0.0 else gamma_ratio)
		combined_row["stage3_uplift_vs_control"] = _format_float(0.0 if alpha_val == 0.0 else stage3_uplift)

		if gate == G1_GATE:
			turn_ratio_val = _turn_ratio(baseline_turn=control_mean_turn_strength, candidate_turn=mean_turn_strength)
			level_gate_pass = int(combined_row["mean_cycle_level"].split(".")[0]) >= 3 and level3_seed_count == int(combined_row["n_seeds"])
			turn_gate_pass = mean_turn_strength >= 0.92 * control_mean_turn_strength if control_mean_turn_strength > 0.0 else True
			g1_gate_pass = (alpha_val == 0.0) or (level_gate_pass and turn_gate_pass)
			g1_gate_pass_by_alpha[alpha_val] = bool(g1_gate_pass)
			combined_row["g1_turn_ratio_vs_control"] = _format_ratio(1.0 if alpha_val == 0.0 else turn_ratio_val)
			combined_row["g1_level_gate_pass"] = _yes_no(True if alpha_val == 0.0 else level_gate_pass)
			combined_row["g1_turn_gate_pass"] = _yes_no(True if alpha_val == 0.0 else turn_gate_pass)
			combined_row["g1_gate_pass"] = _yes_no(g1_gate_pass)
			combined_row["short_scout_pass"] = ""
			combined_row["hard_stop_fail"] = ""
			combined_row["longer_confirm_candidate"] = "no"
			combined_row["verdict"] = "control" if alpha_val == 0.0 else ("pass" if g1_gate_pass else "fail")
		else:
			g1_gate_pass = bool(g1_gate_pass_by_alpha.get(alpha_val, alpha_val == 0.0))
			short_scout_pass = (alpha_val != 0.0) and g1_gate_pass and level3_seed_count >= 1 and stage3_uplift >= 0.02
			hard_stop_fail = (
				alpha_val != 0.0
				and g1_gate_pass
				and alpha_val <= 1.0 + 1e-12
				and level3_seed_count == 0
				and stage3_uplift < 0.02
			)
			if alpha_val == 0.0:
				verdict = "control"
			elif not g1_gate_pass:
				verdict = "blocked_by_g1"
			elif short_scout_pass:
				verdict = "pass"
			elif hard_stop_fail:
				verdict = "fail"
			else:
				verdict = "weak_positive"
			combined_row["g1_turn_ratio_vs_control"] = ""
			combined_row["g1_level_gate_pass"] = ""
			combined_row["g1_turn_gate_pass"] = ""
			combined_row["g1_gate_pass"] = _yes_no(g1_gate_pass)
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
	for condition, per_seed_rows in condition_seed_rows.items():
		gate = str(next(r["gate"] for r in all_combined_rows if str(r["condition"]) == condition))
		control_rows = condition_seed_rows[_control_condition_name(gate=gate)]
		control_by_seed = {int(r["seed"]): r for r in control_rows}
		g1_pass_lookup = {
			float(r["tangential_alpha"]): str(r["g1_gate_pass"])
			for r in all_combined_rows
			if str(r["gate"]) == G1_GATE
		}
		for row in per_seed_rows:
			seed = int(row["seed"])
			ctrl = control_by_seed[seed]
			row["control_env_gamma"] = ctrl["env_gamma"]
			row["control_stage3_score"] = ctrl["stage3_score"]
			row["g1_gate_pass"] = g1_pass_lookup.get(float(row["tangential_alpha"]), "yes" if float(row["tangential_alpha"]) == 0.0 else "no")
			if str(row["condition"]) == _control_condition_name(gate=gate):
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
		row for row in all_combined_rows if str(row["gate"]) == G2_GATE and str(row["verdict"]) == "pass"
	]
	recommended_candidates.sort(
		key=lambda row: (
			float(row["stage3_uplift_vs_control"]),
			int(row["level3_seed_count"]),
			float(row["mean_tangential_ratio"]),
		),
		reverse=True,
	)
	active_g2_rows = [
		row
		for row in all_combined_rows
		if str(row["gate"]) == G2_GATE and str(row["is_control"]) == "no" and float(row["tangential_alpha"]) <= 1.0 + 1e-12
	]
	close_b1 = bool(active_g2_rows) and all(str(row["hard_stop_fail"]) == "yes" for row in active_g2_rows)

	_write_tsv(summary_tsv, fieldnames=SUMMARY_FIELDNAMES, rows=all_summary_rows)
	_write_tsv(combined_tsv, fieldnames=COMBINED_FIELDNAMES, rows=all_combined_rows)
	_write_decision(
		decision_md,
		combined_rows=all_combined_rows,
		recommended_candidates=recommended_candidates,
		close_b1=close_b1,
	)

	return {
		"summary_tsv": str(summary_tsv),
		"combined_tsv": str(combined_tsv),
		"decision_md": str(decision_md),
		"recommended_candidates": [dict(row) for row in recommended_candidates],
		"close_b1": bool(close_b1),
	}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
	parser = argparse.ArgumentParser(description="B1 tangential projection G1/G2 scout harness")
	parser.add_argument("--seeds", type=str, default="45,47,49")
	parser.add_argument("--tangential-alphas", type=str, default="0.0,0.3,0.5,1.0,2.0")
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

	result = run_b1_scout(
		seeds=_parse_seeds(args.seeds),
		tangential_alphas=_parse_float_list(args.tangential_alphas),
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
	print(f"close_b1={result['close_b1']}")


if __name__ == "__main__":
	main()
