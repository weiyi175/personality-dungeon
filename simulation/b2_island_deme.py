"""B2 Island/Deme-Structured Topology harness.

Core architectural difference from existing simulate():
- Deme-local payoff (per-deme strategy distribution -> payoff matrix)
- Deme-local sampled growth vector (only players in the same deme)
- Deme-local weight broadcast (each deme gets its own replicator weights)
- Physical migration: periodic pooled random redistribution between demes
- Global p_*(t) calculated by merging all demes for Stage3 analysis

This module does NOT use simulate() — it runs its own inner loop to enforce
the deme-local invariants required by SDD §19.7.3.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from math import atan2, exp, isfinite, pi, sqrt
from pathlib import Path
from typing import Any

from analysis.cycle_metrics import classify_cycle_level
from analysis.decay_rate import estimate_decay_gamma
from analysis.metrics import strategy_distribution
from evolution.replicator_dynamics import replicator_step, sampled_growth_vector
from players.base_player import BasePlayer
from simulation.run_simulation import (
	_initial_weights,
	_matrix_ab_payoff_vec,
	_normalize_simplex,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "b2_island_deme"
DEFAULT_SUMMARY_TSV = REPO_ROOT / "outputs" / "b2_island_deme_summary.tsv"
DEFAULT_COMBINED_TSV = REPO_ROOT / "outputs" / "b2_island_deme_combined.tsv"
DEFAULT_DECISION_MD = REPO_ROOT / "outputs" / "b2_island_deme_decision.md"

G2_GATE = "g2_sampled"

STRATEGY_SPACE = ["aggressive", "defensive", "balanced"]

SUMMARY_FIELDNAMES = [
	"gate",
	"condition",
	"num_demes",
	"migration_fraction",
	"migration_interval",
	"seed",
	"cycle_level",
	"stage3_score",
	"turn_strength",
	"env_gamma",
	"env_gamma_r2",
	"env_gamma_n_peaks",
	"control_env_gamma",
	"control_stage3_score",
	"stage3_uplift_vs_control_seed",
	"has_level3_seed",
	"mean_inter_deme_phase_spread",
	"max_inter_deme_phase_spread",
	"mean_inter_deme_growth_cosine",
	"phase_amplitude_stability",
	"out_csv",
	"provenance_json",
]

COMBINED_FIELDNAMES = [
	"gate",
	"condition",
	"num_demes",
	"migration_fraction",
	"migration_interval",
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
	"stage3_uplift_vs_control",
	"mean_inter_deme_phase_spread",
	"max_inter_deme_phase_spread",
	"mean_inter_deme_growth_cosine",
	"phase_amplitude_stability",
	"short_scout_pass",
	"hard_stop_fail",
	"longer_confirm_candidate",
	"verdict",
	"representative_seed",
	"representative_deme_simplex_png",
	"representative_phase_amplitude_png",
	"players",
	"rounds",
	"out_dir",
]


# ---------------------------------------------------------------------------
# Utilities
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


def _parse_int_list(spec: str) -> list[int]:
	parts = [part.strip() for part in str(spec).split(",") if part.strip()]
	if not parts:
		raise ValueError("int list cannot be empty")
	return [int(part) for part in parts]


def _format_float(value: float) -> str:
	return f"{float(value):.6f}"


def _yes_no(value: bool) -> str:
	return "yes" if bool(value) else "no"


def _safe_mean(values: list[float]) -> float:
	if not values:
		return 0.0
	return float(sum(values) / float(len(values)))


def _write_tsv(path: Path, *, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
		writer.writeheader()
		writer.writerows(rows)


def _phase_angle(p_a: float, p_d: float, p_b: float) -> float:
	return float(atan2(sqrt(3.0) * (p_d - p_b), 2.0 * p_a - p_d - p_b))


def _phase_angle_from_dist(dist: dict[str, float]) -> float:
	return _phase_angle(
		float(dist.get("aggressive", 1.0 / 3.0)),
		float(dist.get("defensive", 1.0 / 3.0)),
		float(dist.get("balanced", 1.0 / 3.0)),
	)


def _amplitude_from_dist(dist: dict[str, float]) -> float:
	p_a = float(dist.get("aggressive", 1.0 / 3.0))
	p_d = float(dist.get("defensive", 1.0 / 3.0))
	p_b = float(dist.get("balanced", 1.0 / 3.0))
	return float(((p_a - 1.0 / 3.0) ** 2 + (p_d - 1.0 / 3.0) ** 2 + (p_b - 1.0 / 3.0) ** 2) ** 0.5)


def _vec_dot(a: dict[str, float], b: dict[str, float]) -> float:
	return sum(float(a.get(s, 0.0)) * float(b.get(s, 0.0)) for s in STRATEGY_SPACE)


def _vec_norm(v: dict[str, float]) -> float:
	return float(_vec_dot(v, v) ** 0.5)


def _vec_cosine(a: dict[str, float], b: dict[str, float]) -> float:
	na = _vec_norm(a)
	nb = _vec_norm(b)
	if na <= 1e-12 or nb <= 1e-12:
		return 0.0
	return float(_vec_dot(a, b) / (na * nb))


def _simplex_xy(dist: dict[str, float]) -> tuple[float, float]:
	p_a = float(dist.get("aggressive", 1.0 / 3.0))
	p_d = float(dist.get("defensive", 1.0 / 3.0))
	p_b = float(dist.get("balanced", 1.0 / 3.0))
	return (p_d - p_b, p_a - 0.5 * (p_d + p_b))


def _condition_name(*, num_demes: int, migration_fraction: float, migration_interval: int) -> str:
	f_token = f"{float(migration_fraction):.2f}".replace(".", "p")
	return f"g2_M{num_demes}_f{f_token}_T{migration_interval}"


def _control_condition_name() -> str:
	return "g2_M1_control"


# ---------------------------------------------------------------------------
# Deme-local inner loop
# ---------------------------------------------------------------------------

def _deme_local_distribution(players: list[BasePlayer]) -> dict[str, float]:
	"""Compute strategy distribution within a deme from player last_strategy."""
	counts: dict[str, int] = {s: 0 for s in STRATEGY_SPACE}
	total = 0
	for p in players:
		s = getattr(p, "last_strategy", None)
		if s is not None and s in counts:
			counts[s] += 1
			total += 1
	if total <= 0:
		return {s: 1.0 / len(STRATEGY_SPACE) for s in STRATEGY_SPACE}
	return {s: float(counts[s]) / float(total) for s in STRATEGY_SPACE}


def _deme_weight_distribution(players: list[BasePlayer]) -> dict[str, float]:
	"""Compute strategy distribution from player weights (before first step)."""
	totals: dict[str, float] = {s: 0.0 for s in STRATEGY_SPACE}
	n = 0
	for p in players:
		w = getattr(p, "strategy_weights", {})
		for s in STRATEGY_SPACE:
			totals[s] += float(w.get(s, 1.0))
		n += 1
	if n <= 0:
		return {s: 1.0 / len(STRATEGY_SPACE) for s in STRATEGY_SPACE}
	total = sum(totals.values())
	if total <= 0:
		return {s: 1.0 / len(STRATEGY_SPACE) for s in STRATEGY_SPACE}
	return {s: totals[s] / total for s in STRATEGY_SPACE}


def _do_migration(
	demes: list[list[BasePlayer]],
	*,
	migration_fraction: float,
	rng: random.Random,
) -> None:
	"""Pooled random redistribution: extract f% from each deme, pool, redistribute."""
	pool: list[tuple[int, int]] = []  # (deme_index, player_index_in_deme)
	for d_idx, deme in enumerate(demes):
		n_migrate = max(1, int(round(len(deme) * float(migration_fraction))))
		if n_migrate >= len(deme):
			n_migrate = max(1, len(deme) - 1)
		chosen = rng.sample(range(len(deme)), k=n_migrate)
		for p_idx in chosen:
			pool.append((d_idx, p_idx))
	if len(pool) < 2:
		return
	# Collect migrant players
	migrants: list[BasePlayer] = []
	# Sort pool in reverse order within each deme to safely pop
	pool_by_deme: dict[int, list[int]] = {}
	for d_idx, p_idx in pool:
		pool_by_deme.setdefault(d_idx, []).append(p_idx)
	for d_idx in pool_by_deme:
		pool_by_deme[d_idx].sort(reverse=True)
	for d_idx, indices in pool_by_deme.items():
		for p_idx in indices:
			migrants.append(demes[d_idx].pop(p_idx))
	# Shuffle and redistribute evenly
	rng.shuffle(migrants)
	m = len(demes)
	for i, player in enumerate(migrants):
		target_deme = i % m
		demes[target_deme].append(player)


def _run_island_simulation(
	*,
	num_demes: int,
	migration_fraction: float,
	migration_interval: int,
	n_players: int,
	n_rounds: int,
	seed: int,
	selection_strength: float,
	init_bias: float,
	a: float,
	b: float,
	matrix_cross_coupling: float,
	memory_kernel: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
	"""Run island-deme simulation and return (global_rows, per_round_diagnostics).

	Returns
	-------
	global_rows : list[dict]
		One row per round with global p_*(t) and per-deme columns.
	round_diagnostics : list[dict]
		Per-round deme-level diagnostic data (growth vectors, phase angles).
	"""
	rng = random.Random(int(seed))

	players = [
		BasePlayer(STRATEGY_SPACE, rng=random.Random(int(seed) + i))
		for i in range(n_players)
	]
	init_w = _initial_weights(strategy_space=STRATEGY_SPACE, init_bias=float(init_bias))
	for pl in players:
		pl.update_weights(init_w)

	# Assign players to demes
	m = max(1, int(num_demes))
	demes: list[list[BasePlayer]] = [[] for _ in range(m)]
	for idx, pl in enumerate(players):
		demes[idx % m].append(pl)

	global_rows: list[dict[str, Any]] = []
	round_diagnostics: list[dict[str, Any]] = []

	for t in range(int(n_rounds)):
		# --- Step 1: Each deme plays independently ---
		for deme in demes:
			# Each player in this deme chooses a strategy
			for pl in deme:
				pl.choose_strategy()
			# Compute deme-local payoff from deme-local distribution
			deme_dist = _deme_local_distribution(deme)
			# Evaluate each player's reward using deme-local payoff
			payoff = _matrix_ab_payoff_vec(
				strategy_space=STRATEGY_SPACE,
				a=float(a),
				b=float(b),
				matrix_cross_coupling=float(matrix_cross_coupling),
				x=deme_dist,
			)
			for pl in deme:
				strategy = str(pl.last_strategy)
				reward = float(payoff.get(strategy, 0.0))
				pl.update_utility(reward)
				pl.last_reward = reward

		# --- Step 2: Deme-local evolution ---
		per_deme_growth: list[dict[str, float]] = []
		per_deme_dist: list[dict[str, float]] = []

		for deme in demes:
			deme_dist = _deme_local_distribution(deme)
			per_deme_dist.append(deme_dist)
			# Deme-local growth vector
			growth = sampled_growth_vector(deme, STRATEGY_SPACE)
			per_deme_growth.append(growth)
			# Deme-local replicator step -> deme-local weights
			new_weights = replicator_step(
				deme,
				STRATEGY_SPACE,
				selection_strength=float(selection_strength),
			)
			# Broadcast weights only to this deme's players
			for pl in deme:
				pl.update_weights(new_weights)

		# --- Step 3: Migration ---
		if m > 1 and int(migration_interval) > 0 and (t + 1) % int(migration_interval) == 0:
			_do_migration(demes, migration_fraction=float(migration_fraction), rng=rng)

		# --- Step 4: Global observation ---
		all_players: list[BasePlayer] = []
		for deme in demes:
			all_players.extend(deme)
		global_dist = strategy_distribution(all_players, STRATEGY_SPACE)

		row: dict[str, Any] = {
			"round": t,
		}
		for s in STRATEGY_SPACE:
			row[f"p_{s}"] = float(global_dist.get(s, 1.0 / 3.0))
		# Per-deme columns
		for d_idx in range(m):
			d_dist = _deme_local_distribution(demes[d_idx])
			for s in STRATEGY_SPACE:
				row[f"deme{d_idx}_p_{s}"] = float(d_dist.get(s, 1.0 / 3.0))
		global_rows.append(row)

		# Diagnostics
		deme_phases = [_phase_angle_from_dist(d) for d in per_deme_dist]
		if len(deme_phases) >= 2:
			phase_spread = float(max(deme_phases) - min(deme_phases))
			# Wrap to [0, 2*pi] for more robust spread
			if phase_spread > pi:
				phase_spread = 2.0 * pi - phase_spread
		else:
			phase_spread = 0.0

		# Pairwise cosine between deme growth vectors
		cosines: list[float] = []
		for i in range(len(per_deme_growth)):
			for j in range(i + 1, len(per_deme_growth)):
				cosines.append(_vec_cosine(per_deme_growth[i], per_deme_growth[j]))

		round_diagnostics.append({
			"deme_phases": list(deme_phases),
			"inter_deme_phase_spread": float(phase_spread),
			"inter_deme_growth_cosines": list(cosines),
			"per_deme_growth": [dict(g) for g in per_deme_growth],
		})

	return global_rows, round_diagnostics


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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


def _phase_amplitude_stability(rows: list[dict[str, Any]], *, burn_in: int, tail: int) -> float:
	begin = _tail_begin(len(rows), burn_in=int(burn_in), tail=int(tail))
	tail_rows = rows[begin:]
	if not tail_rows:
		return 0.0
	amplitudes = [_amplitude_from_dist(row) for row in tail_rows]
	mean_amp = _safe_mean(amplitudes)
	if mean_amp <= 1e-12:
		return 0.0
	variance = _safe_mean([(float(v) - mean_amp) ** 2 for v in amplitudes])
	stability = 1.0 - min(1.0, (variance ** 0.5) / mean_amp)
	return float(max(0.0, stability))


def _tail_diagnostics(
	round_diagnostics: list[dict[str, Any]],
	*,
	n_rows: int,
	burn_in: int,
	tail: int,
) -> dict[str, float]:
	begin = _tail_begin(n_rows, burn_in=int(burn_in), tail=int(tail))
	window = round_diagnostics[begin:]
	if not window:
		return {
			"mean_inter_deme_phase_spread": 0.0,
			"max_inter_deme_phase_spread": 0.0,
			"mean_inter_deme_growth_cosine": 0.0,
		}
	spreads = [float(d["inter_deme_phase_spread"]) for d in window]
	all_cosines: list[float] = []
	for d in window:
		all_cosines.extend(float(c) for c in d["inter_deme_growth_cosines"])
	return {
		"mean_inter_deme_phase_spread": _safe_mean(spreads),
		"max_inter_deme_phase_spread": float(max(spreads)) if spreads else 0.0,
		"mean_inter_deme_growth_cosine": _safe_mean(all_cosines),
	}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _write_deme_simplex_plot(
	rows: list[dict[str, Any]],
	*,
	num_demes: int,
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
		raise RuntimeError("matplotlib is required to write B2 deme simplex plots") from exc

	begin = _tail_begin(len(rows), burn_in=int(burn_in), tail=int(tail))
	tail_rows = rows[begin:]
	if len(tail_rows) < 2:
		return
	m = max(1, int(num_demes))
	n_cols = min(m + 1, 4)
	n_rows_grid = ((m + 1) + n_cols - 1) // n_cols
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(n_rows_grid, n_cols, figsize=(5 * n_cols, 5 * n_rows_grid), constrained_layout=True)
	if n_rows_grid == 1 and n_cols == 1:
		axes = [[axes]]
	elif n_rows_grid == 1:
		axes = [axes]
	elif n_cols == 1:
		axes = [[ax] for ax in axes]

	def _plot_trajectory(ax: Any, xs: Any, ys: Any, label: str) -> None:
		segments = np.stack(
			[
				np.column_stack([xs[:-1], ys[:-1]]),
				np.column_stack([xs[1:], ys[1:]]),
			],
			axis=1,
		)
		line = LineCollection(segments, cmap="viridis", linewidths=1.5)
		line.set_array(np.linspace(0.0, 1.0, len(segments)))
		ax.add_collection(line)
		ax.scatter(xs[0], ys[0], color="#1f77b4", s=30, marker="o")
		ax.scatter(xs[-1], ys[-1], color="#d62728", s=40, marker="X")
		ax.axhline(0.0, color="#cccccc", linewidth=0.5)
		ax.axvline(0.0, color="#cccccc", linewidth=0.5)
		ax.set_title(label, fontsize=9)
		ax.set_aspect("equal", adjustable="box")

	# Global trajectory
	ax0 = axes[0][0]
	global_points = [_simplex_xy(r) for r in tail_rows]
	gx = np.array([p[0] for p in global_points], dtype=float)
	gy = np.array([p[1] for p in global_points], dtype=float)
	_plot_trajectory(ax0, gx, gy, "global")

	# Per-deme trajectories
	for d_idx in range(m):
		r_idx = (d_idx + 1) // n_cols
		c_idx = (d_idx + 1) % n_cols
		if r_idx >= len(axes) or c_idx >= len(axes[r_idx]):
			break
		ax = axes[r_idx][c_idx]
		deme_points = []
		for row in tail_rows:
			p_a = float(row.get(f"deme{d_idx}_p_aggressive", 1.0 / 3.0))
			p_d = float(row.get(f"deme{d_idx}_p_defensive", 1.0 / 3.0))
			p_b = float(row.get(f"deme{d_idx}_p_balanced", 1.0 / 3.0))
			deme_points.append(_simplex_xy({"aggressive": p_a, "defensive": p_d, "balanced": p_b}))
		dx = np.array([p[0] for p in deme_points], dtype=float)
		dy = np.array([p[1] for p in deme_points], dtype=float)
		_plot_trajectory(ax, dx, dy, f"deme {d_idx}")

	# Turn off unused axes
	for r_idx in range(n_rows_grid):
		for c_idx in range(n_cols):
			flat_idx = r_idx * n_cols + c_idx
			if flat_idx > m:
				axes[r_idx][c_idx].set_visible(False)

	fig.suptitle(title, fontsize=11)
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
		raise RuntimeError("matplotlib is required to write B2 phase-amplitude plots") from exc

	begin = _tail_begin(len(rows), burn_in=int(burn_in), tail=int(tail))
	tail_rows = rows[begin:]
	if not tail_rows:
		return
	rounds = np.array([int(row["round"]) for row in tail_rows], dtype=float)
	phases = np.unwrap(np.array([_phase_angle_from_dist(row) for row in tail_rows], dtype=float))
	amplitudes = np.array([_amplitude_from_dist(row) for row in tail_rows], dtype=float)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, constrained_layout=True)
	axes[0].plot(rounds, phases, color="#1f77b4", linewidth=1.5)
	axes[0].set_ylabel("phase angle")
	axes[0].set_title(title)
	axes[0].grid(alpha=0.25)
	axes[1].plot(rounds, amplitudes, color="#d62728", linewidth=1.5)
	axes[1].set_ylabel("amplitude")
	axes[1].set_xlabel("round")
	axes[1].grid(alpha=0.25)
	fig.savefig(out_png, dpi=160)
	plt.close(fig)


# ---------------------------------------------------------------------------
# Write CSV per condition/seed
# ---------------------------------------------------------------------------

def _write_timeseries_csv(
	path: Path,
	*,
	rows: list[dict[str, Any]],
	num_demes: int,
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = ["round"] + [f"p_{s}" for s in STRATEGY_SPACE]
	for d_idx in range(int(num_demes)):
		for s in STRATEGY_SPACE:
			fieldnames.append(f"deme{d_idx}_p_{s}")
	with path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
		writer.writeheader()
		writer.writerows(rows)


# ---------------------------------------------------------------------------
# Combined summary builder
# ---------------------------------------------------------------------------

def _build_condition_summary(
	*,
	condition: str,
	num_demes: int,
	migration_fraction: float,
	migration_interval: int,
	metric_rows: list[dict[str, Any]],
	per_seed_rows: list[dict[str, Any]],
	out_dir: Path,
	players: int,
	rounds: int,
) -> dict[str, Any]:
	levels = [int(row["cycle_level"]) for row in metric_rows]
	level_counts = {level: levels.count(level) for level in range(4)}
	den = float(len(metric_rows)) or 1.0
	is_control = (int(num_demes) <= 1)
	return {
		"gate": G2_GATE,
		"condition": condition,
		"num_demes": int(num_demes),
		"migration_fraction": _format_float(migration_fraction),
		"migration_interval": int(migration_interval),
		"is_control": _yes_no(is_control),
		"n_seeds": len(metric_rows),
		"mean_cycle_level": _format_float(_safe_mean([float(l) for l in levels])),
		"mean_stage3_score": _format_float(_safe_mean([float(r["stage3_score"]) for r in metric_rows])),
		"mean_turn_strength": _format_float(_safe_mean([float(r["turn_strength"]) for r in metric_rows])),
		"mean_env_gamma": _format_float(_safe_mean([float(r["env_gamma"]) for r in metric_rows])),
		"level_counts_json": json.dumps(level_counts, sort_keys=True),
		"p_level_3": _format_float(sum(1 for l in levels if l >= 3) / den),
		"level3_seed_count": sum(1 for l in levels if l >= 3),
		"control_mean_env_gamma": "",
		"control_mean_stage3_score": "",
		"stage3_uplift_vs_control": "",
		"mean_inter_deme_phase_spread": _format_float(_safe_mean([float(r["mean_inter_deme_phase_spread"]) for r in per_seed_rows])),
		"max_inter_deme_phase_spread": _format_float(max(float(r["max_inter_deme_phase_spread"]) for r in per_seed_rows) if per_seed_rows else 0.0),
		"mean_inter_deme_growth_cosine": _format_float(_safe_mean([float(r["mean_inter_deme_growth_cosine"]) for r in per_seed_rows])),
		"phase_amplitude_stability": _format_float(_safe_mean([float(r["phase_amplitude_stability"]) for r in per_seed_rows])),
		"short_scout_pass": "",
		"hard_stop_fail": "",
		"longer_confirm_candidate": "no",
		"verdict": "control" if is_control else "pending",
		"representative_seed": "",
		"representative_deme_simplex_png": "",
		"representative_phase_amplitude_png": "",
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
			float(row["mean_inter_deme_phase_spread"]),
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
	close_b2: bool,
) -> None:
	lines = [
		"# B2 Island Deme Decision",
		"",
		"## G2 Short Scout",
		"",
	]
	for row in combined_rows:
		lines.append(
			f"- {row['condition']}: M={row['num_demes']} f={row['migration_fraction']} T={row['migration_interval']}"
			f" level3_seed_count={row['level3_seed_count']}"
			f" stage3_uplift={row['stage3_uplift_vs_control']}"
			f" mean_inter_deme_phase_spread={row['mean_inter_deme_phase_spread']}"
			f" max_inter_deme_phase_spread={row['max_inter_deme_phase_spread']}"
			f" mean_inter_deme_growth_cosine={row['mean_inter_deme_growth_cosine']}"
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
				f" stage3_uplift={row['stage3_uplift_vs_control']}"
				f" mean_inter_deme_phase_spread={row['mean_inter_deme_phase_spread']}"
			)
	lines.extend(["", "## Stop Rule", ""])
	lines.append("- G0: M=1 control 必須與 well-mixed sampled 行為一致。")
	lines.append("- G2: 若所有 M=3 active cells 都 0/3 Level 3 且 max(stage3_uplift) < 0.02，B2 直接 closure。")
	lines.append("- Phase spread early-stop: 若所有 conditions 的 mean_inter_deme_phase_spread < 0.10 rad，topology 效應未形成。")
	lines.append("- B2 是 topology 對照，若 fail 則證明局部結構本身仍不足以對抗全局均化。")
	if close_b2:
		lines.append("- overall_verdict: close_b2")
	else:
		lines.append("- overall_verdict: keep_b2_open")
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main harness entry point
# ---------------------------------------------------------------------------

def run_b2_scout(
	*,
	seeds: list[int],
	num_demes_list: list[int],
	migration_fractions: list[float],
	migration_intervals: list[int],
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
	selection_strength: float = 0.06,
	eta: float = 0.55,
	corr_threshold: float = 0.09,
) -> dict[str, Any]:
	out_root.mkdir(parents=True, exist_ok=True)

	all_summary_rows: list[dict[str, Any]] = []
	all_combined_rows: list[dict[str, Any]] = []
	condition_seed_rows: dict[str, list[dict[str, Any]]] = {}
	condition_timeseries: dict[str, dict[int, list[dict[str, Any]]]] = {}

	# Build condition list: M=1 control + active conditions
	conditions: list[dict[str, Any]] = []
	# M=1 control
	conditions.append({
		"condition": _control_condition_name(),
		"num_demes": 1,
		"migration_fraction": 0.0,
		"migration_interval": 0,
		"is_control": True,
	})
	for m_val in num_demes_list:
		for f_val in migration_fractions:
			for t_val in migration_intervals:
				conditions.append({
					"condition": _condition_name(num_demes=int(m_val), migration_fraction=float(f_val), migration_interval=int(t_val)),
					"num_demes": int(m_val),
					"migration_fraction": float(f_val),
					"migration_interval": int(t_val),
					"is_control": False,
				})

	for cond in conditions:
		condition = str(cond["condition"])
		num_d = int(cond["num_demes"])
		mig_f = float(cond["migration_fraction"])
		mig_t = int(cond["migration_interval"])
		out_dir = out_root / condition
		out_dir.mkdir(parents=True, exist_ok=True)

		per_seed_rows: list[dict[str, Any]] = []
		metric_rows: list[dict[str, Any]] = []
		seed_rows_map: dict[int, list[dict[str, Any]]] = {}

		for seed in seeds:
			global_rows, round_diag = _run_island_simulation(
				num_demes=num_d,
				migration_fraction=mig_f,
				migration_interval=mig_t,
				n_players=int(players),
				n_rounds=int(rounds),
				seed=int(seed),
				selection_strength=float(selection_strength),
				init_bias=float(init_bias),
				a=float(a),
				b=float(b),
				matrix_cross_coupling=float(cross),
				memory_kernel=int(memory_kernel),
			)
			csv_path = out_dir / f"seed{seed}.csv"
			_write_timeseries_csv(csv_path, rows=global_rows, num_demes=num_d)
			seed_rows_map[int(seed)] = global_rows

			seed_metric = _seed_metrics(
				global_rows,
				burn_in=int(burn_in),
				tail=int(tail),
				eta=float(eta),
				corr_threshold=float(corr_threshold),
			)
			tail_diag = _tail_diagnostics(
				round_diag,
				n_rows=len(global_rows),
				burn_in=int(burn_in),
				tail=int(tail),
			)
			pa_stability = _phase_amplitude_stability(
				global_rows,
				burn_in=int(burn_in),
				tail=int(tail),
			)

			provenance = {
				"condition": condition,
				"num_demes": num_d,
				"migration_fraction": mig_f,
				"migration_interval": mig_t,
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
					"selection_strength": float(selection_strength),
				},
				"tail_diagnostics": {
					"mean_inter_deme_phase_spread": float(tail_diag["mean_inter_deme_phase_spread"]),
					"max_inter_deme_phase_spread": float(tail_diag["max_inter_deme_phase_spread"]),
					"mean_inter_deme_growth_cosine": float(tail_diag["mean_inter_deme_growth_cosine"]),
					"phase_amplitude_stability": float(pa_stability),
				},
			}
			prov_path = out_dir / f"seed{seed}_provenance.json"
			prov_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

			per_seed_rows.append({
				"gate": G2_GATE,
				"condition": condition,
				"num_demes": int(num_d),
				"migration_fraction": _format_float(mig_f),
				"migration_interval": int(mig_t),
				"seed": int(seed),
				"cycle_level": int(seed_metric["cycle_level"]),
				"stage3_score": _format_float(seed_metric["stage3_score"]),
				"turn_strength": _format_float(seed_metric["turn_strength"]),
				"env_gamma": _format_float(seed_metric["env_gamma"]),
				"env_gamma_r2": _format_float(seed_metric["env_gamma_r2"]),
				"env_gamma_n_peaks": int(seed_metric["env_gamma_n_peaks"]),
				"control_env_gamma": "",
				"control_stage3_score": "",
				"stage3_uplift_vs_control_seed": "",
				"has_level3_seed": _yes_no(int(seed_metric["cycle_level"]) >= 3),
				"mean_inter_deme_phase_spread": _format_float(tail_diag["mean_inter_deme_phase_spread"]),
				"max_inter_deme_phase_spread": _format_float(tail_diag["max_inter_deme_phase_spread"]),
				"mean_inter_deme_growth_cosine": _format_float(tail_diag["mean_inter_deme_growth_cosine"]),
				"phase_amplitude_stability": _format_float(pa_stability),
				"out_csv": str(csv_path),
				"provenance_json": str(prov_path),
			})
			metric_rows.append(per_seed_rows[-1])

		condition_seed_rows[condition] = per_seed_rows
		condition_timeseries[condition] = seed_rows_map
		all_combined_rows.append(
			_build_condition_summary(
				condition=condition,
				num_demes=num_d,
				migration_fraction=mig_f,
				migration_interval=mig_t,
				metric_rows=metric_rows,
				per_seed_rows=per_seed_rows,
				out_dir=out_dir,
				players=int(players),
				rounds=int(rounds),
			)
		)

	# --- Populate control references and verdicts ---
	control_row = next(
		(row for row in all_combined_rows if str(row["condition"]) == _control_condition_name()),
		None,
	)
	control_mean_env_gamma = float(control_row["mean_env_gamma"]) if control_row else 0.0
	control_mean_stage3_score = float(control_row["mean_stage3_score"]) if control_row else 0.0

	for combined_row in all_combined_rows:
		is_control = str(combined_row["is_control"]) == "yes"
		mean_stage3_score = float(combined_row["mean_stage3_score"])
		mean_env_gamma = float(combined_row["mean_env_gamma"])
		level3_seed_count = int(combined_row["level3_seed_count"])
		stage3_uplift = mean_stage3_score - control_mean_stage3_score

		combined_row["control_mean_env_gamma"] = _format_float(control_mean_env_gamma)
		combined_row["control_mean_stage3_score"] = _format_float(control_mean_stage3_score)
		combined_row["stage3_uplift_vs_control"] = _format_float(0.0 if is_control else stage3_uplift)

		if is_control:
			combined_row["short_scout_pass"] = ""
			combined_row["hard_stop_fail"] = ""
			combined_row["verdict"] = "control"
		else:
			short_scout_pass = (
				level3_seed_count >= 1
				and abs(mean_env_gamma) <= 5e-4
			)
			hard_stop_fail = (
				level3_seed_count == 0
				and stage3_uplift < 0.02
			)
			if short_scout_pass:
				verdict = "pass"
			elif hard_stop_fail:
				verdict = "fail"
			else:
				verdict = "weak_positive"
			combined_row["short_scout_pass"] = _yes_no(short_scout_pass)
			combined_row["hard_stop_fail"] = _yes_no(hard_stop_fail)
			combined_row["verdict"] = verdict
			combined_row["longer_confirm_candidate"] = _yes_no(short_scout_pass)

	# --- Populate representative seeds and plots ---
	for combined_row in all_combined_rows:
		condition = str(combined_row["condition"])
		representative = _representative_seed_row(condition_seed_rows[condition])
		rep_seed = int(representative["seed"])
		num_d = int(combined_row["num_demes"])
		out_dir_path = Path(combined_row["out_dir"])

		rep_simplex_png = out_dir_path / f"seed{rep_seed}_deme_simplex.png"
		rep_phase_png = out_dir_path / f"seed{rep_seed}_phase_amplitude.png"

		_write_deme_simplex_plot(
			condition_timeseries[condition][rep_seed],
			num_demes=num_d,
			out_png=rep_simplex_png,
			title=f"{condition} seed={rep_seed}",
			burn_in=int(burn_in),
			tail=int(tail),
		)
		_write_phase_amplitude_plot(
			condition_timeseries[condition][rep_seed],
			out_png=rep_phase_png,
			title=f"{condition} seed={rep_seed}",
			burn_in=int(burn_in),
			tail=int(tail),
		)
		combined_row["representative_seed"] = rep_seed
		combined_row["representative_deme_simplex_png"] = str(rep_simplex_png)
		combined_row["representative_phase_amplitude_png"] = str(rep_phase_png)

	# --- Populate per-seed control references ---
	control_seed_rows = condition_seed_rows.get(_control_condition_name(), [])
	control_by_seed = {int(r["seed"]): r for r in control_seed_rows}
	for condition, per_seed_rows in condition_seed_rows.items():
		for row in per_seed_rows:
			seed = int(row["seed"])
			ctrl = control_by_seed.get(seed)
			if ctrl is not None:
				row["control_env_gamma"] = ctrl["env_gamma"]
				row["control_stage3_score"] = ctrl["stage3_score"]
				if str(row["condition"]) == _control_condition_name():
					row["stage3_uplift_vs_control_seed"] = _format_float(0.0)
				else:
					row["stage3_uplift_vs_control_seed"] = _format_float(
						float(row["stage3_score"]) - float(ctrl["stage3_score"])
					)
		all_summary_rows.extend(per_seed_rows)

	# --- Stop rule ---
	active_rows = [
		row for row in all_combined_rows
		if str(row["is_control"]) == "no"
	]
	close_b2 = bool(active_rows) and all(str(row["hard_stop_fail"]) == "yes" for row in active_rows)

	recommended_candidates = [
		row for row in all_combined_rows
		if str(row["verdict"]) == "pass"
	]
	recommended_candidates.sort(
		key=lambda row: (
			float(row["stage3_uplift_vs_control"]),
			int(row["level3_seed_count"]),
			float(row["mean_inter_deme_phase_spread"]),
		),
		reverse=True,
	)

	_write_tsv(summary_tsv, fieldnames=SUMMARY_FIELDNAMES, rows=all_summary_rows)
	_write_tsv(combined_tsv, fieldnames=COMBINED_FIELDNAMES, rows=all_combined_rows)
	_write_decision(
		decision_md,
		combined_rows=all_combined_rows,
		recommended_candidates=recommended_candidates,
		close_b2=close_b2,
	)

	return {
		"summary_tsv": str(summary_tsv),
		"combined_tsv": str(combined_tsv),
		"decision_md": str(decision_md),
		"recommended_candidates": [dict(row) for row in recommended_candidates],
		"close_b2": bool(close_b2),
	}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
	parser = argparse.ArgumentParser(description="B2 island deme G2 scout harness")
	parser.add_argument("--seeds", type=str, default="45,47,49")
	parser.add_argument("--num-demes", type=str, default="3")
	parser.add_argument("--migration-fractions", type=str, default="0.02,0.05,0.10")
	parser.add_argument("--migration-intervals", type=str, default="100,200")
	parser.add_argument("--players", type=int, default=300)
	parser.add_argument("--rounds", type=int, default=3000)
	parser.add_argument("--burn-in", type=int, default=1000)
	parser.add_argument("--tail", type=int, default=1000)
	parser.add_argument("--memory-kernel", type=int, default=3)
	parser.add_argument("--init-bias", type=float, default=0.12)
	parser.add_argument("--a", type=float, default=1.0)
	parser.add_argument("--b", type=float, default=0.9)
	parser.add_argument("--matrix-cross-coupling", type=float, default=0.20)
	parser.add_argument("--selection-strength", type=float, default=0.06)
	parser.add_argument("--eta", type=float, default=0.55)
	parser.add_argument("--corr-threshold", type=float, default=0.09)
	parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
	parser.add_argument("--summary-tsv", type=Path, default=DEFAULT_SUMMARY_TSV)
	parser.add_argument("--combined-tsv", type=Path, default=DEFAULT_COMBINED_TSV)
	parser.add_argument("--decision-md", type=Path, default=DEFAULT_DECISION_MD)
	args = parser.parse_args()

	result = run_b2_scout(
		seeds=_parse_seeds(args.seeds),
		num_demes_list=_parse_int_list(args.num_demes),
		migration_fractions=_parse_float_list(args.migration_fractions),
		migration_intervals=_parse_int_list(args.migration_intervals),
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
		selection_strength=float(args.selection_strength),
		eta=float(args.eta),
		corr_threshold=float(args.corr_threshold),
	)
	print(f"summary_tsv={result['summary_tsv']}")
	print(f"combined_tsv={result['combined_tsv']}")
	print(f"decision_md={result['decision_md']}")
	print(f"close_b2={result['close_b2']}")


if __name__ == "__main__":
	main()
