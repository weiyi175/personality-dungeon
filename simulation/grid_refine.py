"""Grid + Refine automation CLI (research).

Why
- Stage3/Level3 can be structurally rare and noisy; discrete pass/fail is a bad optimization target.
- This tool provides a stable workflow:
  1) Coarse grid (few seeds, often single seed) to map a continuous score (rotation consistency).
  2) Candidate selection by score threshold.
  3) Local refinement grid around candidates with multi-seed aggregation.
  4) Ranking table: mean/variance/pass-rate.

Layering
- Simulation orchestration lives in simulation/.
- Pure metrics come from analysis/cycle_metrics.py.

Example
  python3 -m simulation.grid_refine \
    --payoff-mode matrix_ab --gamma 0.01 --epsilon 0.18 \
    --players 300 --rounds 6000 --selection-strength 0.02 \
    --series p --burn-in-frac 0.3 --tail 4000 \
    --a-grid 0.30:0.40:0.01 --b-grid 0.05:0.12:0.01 \
    --coarse-seeds 0 \
    --coarse-rot-score-threshold 0.62 \
    --refine-seeds 0:9 \
    --refine-span-a 0.02 --refine-span-b 0.02 --refine-step 0.005 \
		--stage3-quantile 0.75 \
    --out-prefix outputs/refine_demo
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from analysis.cycle_metrics import (
	assess_stage1_amplitude,
	assess_stage2_frequency,
	classify_cycle_level,
	normalize_simplex_timeseries,
	phase_direction_consistency_turning,
)
from simulation.run_simulation import SimConfig, simulate_series_window


def _auto_jobs() -> int:
	cpus = os.cpu_count() or 1
	return max(1, int(cpus) - 1)


def _parse_seeds(spec: str) -> list[int]:
	s = spec.strip()
	if not s:
		raise ValueError("seed spec cannot be empty")
	if "," in s:
		out: list[int] = []
		for part in s.split(","):
			part = part.strip()
			if not part:
				continue
			out.append(int(part))
		if not out:
			raise ValueError("seed list is empty")
		return out
	if ":" in s:
		parts = [p.strip() for p in s.split(":")]
		if len(parts) not in (2, 3):
			raise ValueError("Range format must be start:end or start:end:step")
		start = int(parts[0])
		end = int(parts[1])
		step = int(parts[2]) if len(parts) == 3 else 1
		if step == 0:
			raise ValueError("step cannot be 0")
		if step > 0:
			return list(range(start, end + 1, step))
		return list(range(start, end - 1, step))
	return [int(s)]


def _parse_float_grid(spec: str) -> list[float]:
	s = spec.strip()
	if not s:
		raise ValueError("grid spec cannot be empty")
	if "," in s:
		out: list[float] = []
		for part in s.split(","):
			part = part.strip()
			if not part:
				continue
			out.append(float(part))
		if not out:
			raise ValueError("grid list is empty")
		return out
	if ":" in s:
		parts = [p.strip() for p in s.split(":")]
		if len(parts) not in (2, 3):
			raise ValueError("Range format must be start:end or start:end:step")
		start = float(parts[0])
		end = float(parts[1])
		step = float(parts[2]) if len(parts) == 3 else 0.1
		if step == 0.0:
			raise ValueError("step cannot be 0")
		if step < 0:
			step = -step
			start, end = end, start
		out: list[float] = []
		x = start
		tol = 1e-12
		while x <= end + tol:
			out.append(float(x))
			x += step
		return out
	return [float(s)]


def _frange_inclusive(center: float, *, span: float, step: float) -> list[float]:
	if step <= 0:
		raise ValueError("step must be > 0")
	lo = float(center) - float(span)
	hi = float(center) + float(span)
	out: list[float] = []
	x = lo
	tol = 1e-12
	while x <= hi + tol:
		out.append(float(x))
		x += float(step)
	return out


def build_refine_points(
	candidates: Iterable[tuple[float, float]],
	*,
	span_a: float,
	span_b: float,
	step: float,
	dedupe_decimals: int = 6,
) -> list[tuple[float, float]]:
	"""Generate local refine grid points around candidates.

	Deduping is done by rounding to avoid huge overlap when candidates are near.
	"""
	seen: set[tuple[float, float]] = set()
	points: list[tuple[float, float]] = []
	for a0, b0 in candidates:
		for a in _frange_inclusive(float(a0), span=float(span_a), step=float(step)):
			for b in _frange_inclusive(float(b0), span=float(span_b), step=float(step)):
				key = (round(float(a), dedupe_decimals), round(float(b), dedupe_decimals))
				if key in seen:
					continue
				seen.add(key)
				points.append((float(a), float(b)))
	return points


def _quantile(values: list[float], *, q: float) -> float:
	if not values:
		return 0.0
	qq = float(q)
	if qq < 0.0 or qq > 1.0:
		raise ValueError("q must be in [0,1]")
	sv = sorted(float(v) for v in values)
	n = len(sv)
	idx = int(qq * float(n - 1))
	idx = max(0, min(n - 1, idx))
	return float(sv[idx])


def _stage3_windowed_scores(
	series3: dict[str, list[float]],
	*,
	window: int,
	step: int,
	eta: float,
	min_turn_strength: float,
	phase_smoothing: int,
) -> tuple[list[float], list[float]]:
	"""Compute turning-based Stage3 scores over sliding windows.

	Returns
	- scores: list of ph.score per window
	- strengths: list of ph.turn_strength per window

	Notes
	- If series length < window, falls back to a single window (the full series).
	"""
	keys = ("aggressive", "defensive", "balanced")
	n = min(len(series3.get(k, [])) for k in keys)
	if n <= 0:
		return [], []
	w = int(window)
	if w <= 0:
		raise ValueError("window must be > 0")
	if w > n:
		w = n
	s = int(step)
	if s <= 0:
		raise ValueError("step must be > 0")

	scores: list[float] = []
	strengths: list[float] = []
	start = 0
	while start + w <= n:
		seg = {k: list(series3[k][start : start + w]) for k in keys}
		ph = phase_direction_consistency_turning(
			seg,
			burn_in=0,
			tail=None,
			eta=float(eta),
			min_turn_strength=float(min_turn_strength),
			phase_smoothing=int(phase_smoothing),
		)
		scores.append(float(ph.score))
		strengths.append(float(ph.turn_strength))
		start += s

	if not scores:
		ph = phase_direction_consistency_turning(
			{k: list(series3.get(k, [])) for k in keys},
			burn_in=0,
			tail=None,
			eta=float(eta),
			min_turn_strength=float(min_turn_strength),
			phase_smoothing=int(phase_smoothing),
		)
		return [float(ph.score)], [float(ph.turn_strength)]

	return scores, strengths


@dataclass(frozen=True)
class OneRunMetrics:
	a: float
	b: float
	seed: int
	rot_score_max: float
	rot_turn_strength_max: float
	rot_score_q: float
	rot_turn_strength_q: float
	max_amp: float
	max_corr: float
	cycle_level: int
	stage3_passed: bool


def _eval_one(
	*,
	players: int,
	rounds: int,
	seed: int,
	payoff_mode: str,
	popularity_mode: str,
	gamma: float,
	epsilon: float,
	a: float,
	b: float,
	selection_strength: float,
	memory_kernel: int,
	series: str,
	burn_in: int,
	tail: int | None,
	min_lag: int,
	max_lag: int,
	corr_threshold: float,
	amplitude_threshold: float,
	eta: float,
	min_turn_strength: float,
	phase_smoothing: int,
	stage3_window: int,
	stage3_step: int,
	stage3_quantile: float,
) -> OneRunMetrics:
	cfg = SimConfig(
		n_players=int(players),
		n_rounds=int(rounds),
		seed=int(seed),
		payoff_mode=str(payoff_mode),
		gamma=float(gamma),
		popularity_mode=str(popularity_mode),
		epsilon=float(epsilon),
		a=float(a),
		b=float(b),
		matrix_cross_coupling=0.0,
		init_bias=0.0,
		evolution_mode="sampled",
		payoff_lag=1,
		memory_kernel=int(memory_kernel),
		selection_strength=float(selection_strength),
		out_csv=Path("outputs") / "_ignored.csv",
	)
	series_map = simulate_series_window(
		cfg,
		series=str(series),
		burn_in=int(burn_in),
		tail=int(tail) if tail is not None else None,
	)

	# Continuous diagnostics
	stage1 = assess_stage1_amplitude(series_map, burn_in=0, tail=None, threshold=float(amplitude_threshold), aggregation="any")
	max_amp = float(max(stage1.amplitudes.values())) if stage1.amplitudes else 0.0
	stage2 = assess_stage2_frequency(
		series_map,
		burn_in=0,
		tail=None,
		min_lag=int(min_lag),
		max_lag=int(max_lag),
		corr_threshold=float(corr_threshold),
		aggregation="any",
	)
	max_corr = float(max(df.corr for df in stage2.frequencies.values())) if stage2.frequencies else 0.0

	phase_input_map: dict[str, list[float]] = {k: list(v) for k, v in series_map.items()}
	if str(series) == "w":
		phase_input_map = {
			k: list(v)
			for k, v in normalize_simplex_timeseries(series_map).items()
		}

	scores, strengths = _stage3_windowed_scores(
		phase_input_map,
		window=int(stage3_window),
		step=int(stage3_step),
		eta=float(eta),
		min_turn_strength=float(min_turn_strength),
		phase_smoothing=int(phase_smoothing),
	)
	rot_score_max = float(max(scores)) if scores else 0.0
	rot_turn_strength_max = float(max(strengths)) if strengths else 0.0
	rot_score_q = float(_quantile(scores, q=float(stage3_quantile))) if scores else 0.0
	rot_turn_strength_q = float(_quantile(strengths, q=float(stage3_quantile))) if strengths else 0.0

	# Discrete level (for pass-rate reporting)
	res = classify_cycle_level(
		series_map,
		burn_in=0,
		tail=None,
		amplitude_threshold=float(amplitude_threshold),
		min_lag=int(min_lag),
		max_lag=int(max_lag),
		corr_threshold=float(corr_threshold),
		eta=float(eta),
		min_turn_strength=float(min_turn_strength),
		normalize_for_phase=(str(series) == "w"),
		stage3_method="turning",
		phase_smoothing=int(phase_smoothing),
		stage3_window=int(stage3_window),
		stage3_step=int(stage3_step),
		stage3_quantile=float(stage3_quantile),
	)
	stage3_passed = bool(res.stage3.passed) if res.stage3 is not None else False
	return OneRunMetrics(
		a=float(a),
		b=float(b),
		seed=int(seed),
		rot_score_max=float(rot_score_max),
		rot_turn_strength_max=float(rot_turn_strength_max),
		rot_score_q=float(rot_score_q),
		rot_turn_strength_q=float(rot_turn_strength_q),
		max_amp=float(max_amp),
		max_corr=float(max_corr),
		cycle_level=int(res.level),
		stage3_passed=bool(stage3_passed),
	)


def _mean(xs: list[float]) -> float:
	return float(sum(xs) / float(len(xs))) if xs else 0.0


def _var(xs: list[float]) -> float:
	if not xs:
		return 0.0
	m = _mean(xs)
	return float(sum((x - m) * (x - m) for x in xs) / float(len(xs)))


def select_candidates(
	agg_rows: list[dict],
	*,
	rot_score_threshold: float,
	min_turn_strength: float,
	min_max_amp: float | None = None,
	min_max_corr: float | None = None,
) -> list[tuple[float, float]]:
	out: list[tuple[float, float]] = []
	for r in agg_rows:
		if float(r.get("mean_rot_score_max", 0.0)) < float(rot_score_threshold):
			continue
		if float(r.get("mean_rot_turn_strength_max", 0.0)) < float(min_turn_strength):
			continue
		if min_max_amp is not None and float(r.get("mean_max_amp", 0.0)) < float(min_max_amp):
			continue
		if min_max_corr is not None and float(r.get("mean_max_corr", 0.0)) < float(min_max_corr):
			continue
		out.append((float(r["a"]), float(r["b"])))
	return out


def _threshold_from_top_frac(values: list[float], *, top_frac: float) -> float:
	"""Return a cutoff threshold such that roughly top_frac of values are >= threshold."""
	if not values:
		return 0.0
	f = float(top_frac)
	if f <= 0.0 or f > 1.0:
		raise ValueError("top_frac must be in (0, 1]")
	sv = sorted(float(v) for v in values)
	# Percentile at (1 - top_frac). Use (n-1) scale for stability.
	n = len(sv)
	pos = (1.0 - f) * float(n - 1)
	idx = int(math.floor(pos))
	idx = max(0, min(n - 1, idx))
	return float(sv[idx])


def main() -> None:
	p = argparse.ArgumentParser(description="Grid + Refine automation (score-threshold based)")

	# simulation params
	p.add_argument("--players", type=int, default=300)
	p.add_argument("--rounds", type=int, default=6000)
	p.add_argument("--payoff-mode", type=str, default="matrix_ab", choices=["count_cycle", "matrix_ab"])
	p.add_argument(
		"--popularity-mode",
		type=str,
		default="sampled",
		choices=["sampled", "expected"],
		help="Dungeon popularity update: sampled (default) or expected (denoised)",
	)
	p.add_argument("--gamma", type=float, default=0.01)
	p.add_argument("--epsilon", type=float, default=0.18)
	p.add_argument("--selection-strength", type=float, default=0.02)
	p.add_argument("--memory-kernel", type=int, default=1, choices=[1, 3, 5])

	# series / window
	p.add_argument("--series", type=str, default="p", choices=["w", "p"], help="Use weights (w_*) or proportions (p_*)")
	p.add_argument("--burn-in", type=int, default=None)
	p.add_argument("--burn-in-frac", type=float, default=0.3)
	p.add_argument("--tail", type=int, default=4000)

	# coarse grid
	p.add_argument("--a-grid", type=str, required=True, help='Coarse grid for a: "0.3:0.4:0.01" or "0.3,0.31"')
	p.add_argument("--b-grid", type=str, required=True, help='Coarse grid for b: "0.05:0.12:0.01" or "0.05,0.06"')
	p.add_argument("--coarse-seeds", type=str, default="0", help='Seeds for coarse scan (default single seed): "0" or "0:4"')
	p.add_argument(
		"--coarse-top-frac",
		type=float,
		default=0.15,
		help="Auto-select candidates from coarse grid by top fraction of mean_rot_score. Set 0 to disable.",
	)
	p.add_argument(
		"--coarse-rot-score-threshold",
		type=float,
		default=None,
		help="Candidate threshold on mean_rot_score. If omitted, uses --coarse-top-frac.",
	)
	p.add_argument("--coarse-min-turn-strength", type=float, default=0.0, help="Candidate threshold on mean_rot_turn_strength")
	p.add_argument(
		"--coarse-min-max-amp",
		type=float,
		default=None,
		help="Optional coarse filter: require mean_max_amp >= this value.",
	)
	p.add_argument(
		"--coarse-min-max-corr",
		type=float,
		default=None,
		help="Optional coarse filter: require mean_max_corr >= this value.",
	)

	# refine grid
	p.add_argument("--refine-seeds", type=str, default="0:9", help='Seeds for refine aggregation: "0:9"')
	p.add_argument("--refine-span-a", type=float, default=0.02)
	p.add_argument("--refine-span-b", type=float, default=0.02)
	p.add_argument("--refine-step", type=float, default=0.005)

	# metric thresholds for discrete level/pass-rate
	p.add_argument("--amplitude-threshold", type=float, default=0.02)
	p.add_argument("--min-lag", type=int, default=2)
	p.add_argument("--max-lag", type=int, default=500)
	p.add_argument("--corr-threshold", type=float, default=0.09)
	p.add_argument("--eta", type=float, default=0.6)
	p.add_argument("--min-turn-strength", type=float, default=0.0)
	p.add_argument("--phase-smoothing", type=int, default=1)
	# stage3 windowing (two-stage workflow)
	p.add_argument(
		"--stage3-window",
		type=int,
		default=80,
		help="Sliding window length (steps) for Stage3 turning-based score aggregation.",
	)
	p.add_argument(
		"--stage3-step",
		type=int,
		default=20,
		help="Sliding window step (stride) for Stage3 aggregation.",
	)
	p.add_argument(
		"--stage3-quantile",
		type=float,
		default=0.75,
		help="Quantile in [0,1] used for windowed Stage3 score/strength aggregation and refine ranking (default 0.75).",
	)

	# execution/output
	p.add_argument("--jobs", type=int, default=0, help="Parallel workers: 0=auto, 1=sequential")
	p.add_argument("--out-prefix", type=Path, default=Path("outputs") / "grid_refine", help="Output prefix (no extension)")
	args = p.parse_args()

	jobs = int(args.jobs)
	if jobs == 0:
		jobs = _auto_jobs()
	if jobs < 1:
		raise ValueError("--jobs must be >= 0")

	burn_in = args.burn_in
	if burn_in is None:
		burn_in = int(round(float(args.rounds) * float(args.burn_in_frac)))
	burn_in = max(0, int(burn_in))

	a_vals = _parse_float_grid(str(args.a_grid))
	b_vals = _parse_float_grid(str(args.b_grid))
	coarse_seeds = _parse_seeds(str(args.coarse_seeds))
	refine_seeds = _parse_seeds(str(args.refine_seeds))

	out_prefix: Path = args.out_prefix
	out_prefix.parent.mkdir(parents=True, exist_ok=True)
	coarse_points = [(float(a), float(b)) for a in a_vals for b in b_vals]

	def run_points(points: list[tuple[float, float]], seeds: list[int]) -> list[OneRunMetrics]:
		tasks = [(a, b, sd) for (a, b) in points for sd in seeds]
		results: list[OneRunMetrics] = []
		if jobs == 1:
			for a, b, sd in tasks:
				results.append(
					_eval_one(
						players=int(args.players),
						rounds=int(args.rounds),
						seed=int(sd),
						payoff_mode=str(args.payoff_mode),
						popularity_mode=str(args.popularity_mode),
						gamma=float(args.gamma),
						epsilon=float(args.epsilon),
						a=float(a),
						b=float(b),
						selection_strength=float(args.selection_strength),
						memory_kernel=int(args.memory_kernel),
						series=str(args.series),
						burn_in=int(burn_in),
						tail=int(args.tail) if args.tail is not None else None,
						min_lag=int(args.min_lag),
						max_lag=int(args.max_lag),
						corr_threshold=float(args.corr_threshold),
						amplitude_threshold=float(args.amplitude_threshold),
						eta=float(args.eta),
						min_turn_strength=float(args.min_turn_strength),
						phase_smoothing=int(args.phase_smoothing),
						stage3_window=int(args.stage3_window),
						stage3_step=int(args.stage3_step),
						stage3_quantile=float(args.stage3_quantile),
					)
				)
			return results

		with ProcessPoolExecutor(max_workers=int(jobs)) as ex:
			futs = [
				ex.submit(
					_eval_one,
					players=int(args.players),
					rounds=int(args.rounds),
					seed=int(sd),
					payoff_mode=str(args.payoff_mode),
					popularity_mode=str(args.popularity_mode),
					gamma=float(args.gamma),
					epsilon=float(args.epsilon),
					a=float(a),
					b=float(b),
					selection_strength=float(args.selection_strength),
					memory_kernel=int(args.memory_kernel),
					series=str(args.series),
					burn_in=int(burn_in),
					tail=int(args.tail) if args.tail is not None else None,
					min_lag=int(args.min_lag),
					max_lag=int(args.max_lag),
					corr_threshold=float(args.corr_threshold),
					amplitude_threshold=float(args.amplitude_threshold),
					eta=float(args.eta),
					min_turn_strength=float(args.min_turn_strength),
					phase_smoothing=int(args.phase_smoothing),
					stage3_window=int(args.stage3_window),
					stage3_step=int(args.stage3_step),
					stage3_quantile=float(args.stage3_quantile),
				)
				for (a, b, sd) in tasks
			]
			for fut in as_completed(futs):
				results.append(fut.result())
		return results

	# 1) Coarse scan
	coarse_runs = run_points(coarse_points, coarse_seeds)
	by_point: dict[tuple[float, float], list[OneRunMetrics]] = defaultdict(list)
	for r in coarse_runs:
		by_point[(float(r.a), float(r.b))].append(r)

	coarse_rows: list[dict] = []
	for (a, b) in coarse_points:
		rs = by_point.get((float(a), float(b)), [])
		rot_scores_max = [x.rot_score_max for x in rs]
		rot_turns_max = [x.rot_turn_strength_max for x in rs]
		rot_scores_q = [x.rot_score_q for x in rs]
		rot_turns_q = [x.rot_turn_strength_q for x in rs]
		max_amps = [x.max_amp for x in rs]
		max_corrs = [x.max_corr for x in rs]
		stage3_passes = [1.0 if x.stage3_passed else 0.0 for x in rs]
		lvl3s = [1.0 if x.cycle_level == 3 else 0.0 for x in rs]
		coarse_rows.append(
			{
				"a": float(a),
				"b": float(b),
				"n_seeds": int(len(rs)),
				"stage3_quantile": float(args.stage3_quantile),
				"mean_rot_score_max": _mean(rot_scores_max),
				"var_rot_score_max": _var(rot_scores_max),
				"mean_rot_turn_strength_max": _mean(rot_turns_max),
				"var_rot_turn_strength_max": _var(rot_turns_max),
				"mean_rot_score_q": _mean(rot_scores_q),
				"var_rot_score_q": _var(rot_scores_q),
				"mean_rot_turn_strength_q": _mean(rot_turns_q),
				"var_rot_turn_strength_q": _var(rot_turns_q),
				"mean_max_amp": _mean(max_amps),
				"mean_max_corr": _mean(max_corrs),
				"p_stage3_pass": _mean(stage3_passes),
				"p_level_3": _mean(lvl3s),
			}
		)

	coarse_csv = Path(str(out_prefix) + "_coarse.csv")
	with coarse_csv.open("w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=list(coarse_rows[0].keys()) if coarse_rows else ["a", "b"])
		w.writeheader()
		w.writerows(coarse_rows)

	# 2) Candidate selection by score threshold (explicit) or top-fraction (auto)
	thr = args.coarse_rot_score_threshold
	if thr is None:
		top_frac = float(args.coarse_top_frac)
		if top_frac <= 0.0:
			raise ValueError("Provide --coarse-rot-score-threshold or set --coarse-top-frac > 0")
		thr = _threshold_from_top_frac([float(r["mean_rot_score_max"]) for r in coarse_rows], top_frac=top_frac)
		print(f"Auto coarse threshold from top_frac={top_frac:g}: rot_score_threshold={thr:.6g}")

	candidates = select_candidates(
		coarse_rows,
		rot_score_threshold=float(thr),
		min_turn_strength=float(args.coarse_min_turn_strength),
		min_max_amp=(float(args.coarse_min_max_amp) if args.coarse_min_max_amp is not None else None),
		min_max_corr=(float(args.coarse_min_max_corr) if args.coarse_min_max_corr is not None else None),
	)
	print(f"Coarse grid wrote: {coarse_csv} (points={len(coarse_points)}, candidates={len(candidates)})")
	if not candidates:
		print("No candidates selected. Try lowering --coarse-rot-score-threshold or increasing players/rounds.")
		return

	# 3) Refine around candidates
	refine_points = build_refine_points(
		candidates,
		span_a=float(args.refine_span_a),
		span_b=float(args.refine_span_b),
		step=float(args.refine_step),
	)
	refine_runs = run_points(refine_points, refine_seeds)
	by_ref: dict[tuple[float, float], list[OneRunMetrics]] = defaultdict(list)
	for r in refine_runs:
		by_ref[(float(r.a), float(r.b))].append(r)

	refine_rows: list[dict] = []
	for (a, b) in refine_points:
		rs = by_ref.get((float(a), float(b)), [])
		rot_scores_max = [x.rot_score_max for x in rs]
		rot_turns_max = [x.rot_turn_strength_max for x in rs]
		rot_scores_q = [x.rot_score_q for x in rs]
		rot_turns_q = [x.rot_turn_strength_q for x in rs]
		max_amps = [x.max_amp for x in rs]
		max_corrs = [x.max_corr for x in rs]
		stage3_passes = [1.0 if x.stage3_passed else 0.0 for x in rs]
		lvl3s = [1.0 if x.cycle_level == 3 else 0.0 for x in rs]
		refine_rows.append(
			{
				"a": float(a),
				"b": float(b),
				"n_seeds": int(len(rs)),
				"stage3_quantile": float(args.stage3_quantile),
				"mean_rot_score_max": _mean(rot_scores_max),
				"var_rot_score_max": _var(rot_scores_max),
				"mean_rot_turn_strength_max": _mean(rot_turns_max),
				"var_rot_turn_strength_max": _var(rot_turns_max),
				"mean_rot_score_q": _mean(rot_scores_q),
				"var_rot_score_q": _var(rot_scores_q),
				"mean_rot_turn_strength_q": _mean(rot_turns_q),
				"var_rot_turn_strength_q": _var(rot_turns_q),
				"mean_max_amp": _mean(max_amps),
				"mean_max_corr": _mean(max_corrs),
				"p_stage3_pass": _mean(stage3_passes),
				"p_level_3": _mean(lvl3s),
			}
		)

	# 4) Ranking output (two-stage workflow)
	# - Coarse selection uses max score to avoid missing intermittent rotation.
	# - Refine/report ranks by window-quantile score to avoid being fooled by rare bursts.
	refine_rows.sort(key=lambda r: (float(r["mean_rot_score_q"]), float(r["p_stage3_pass"])), reverse=True)
	refine_csv = Path(str(out_prefix) + "_refine_rank.csv")
	with refine_csv.open("w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=list(refine_rows[0].keys()) if refine_rows else ["a", "b"])
		w.writeheader()
		w.writerows(refine_rows)

	print(f"Refine grid wrote: {refine_csv} (points={len(refine_points)}, seeds={len(refine_seeds)})")
	q = float(args.stage3_quantile)
	print(f"Top 10 (by mean_rot_score_q at q={q:g}, then p_stage3_pass):")
	for r in refine_rows[:10]:
		print(
			f"a={r['a']:.6g} b={r['b']:.6g} mean_rot_score_q={r['mean_rot_score_q']:.3f} "
			f"p_stage3_pass={r['p_stage3_pass']:.2f} p_L3={r['p_level_3']:.2f} "
			f"mean_max_corr={r['mean_max_corr']:.3f}"
		)


if __name__ == "__main__":
	main()
