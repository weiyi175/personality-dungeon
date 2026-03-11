"""Multi-seed stability report CLI (research / SDD).

Goal
- For a fixed parameterization, run multiple seeds and summarize cycle_level distribution.
- Keeps layering clean: simulation orchestrates, analysis provides pure metrics.

Example
	python3 -m simulation.seed_stability \
		--payoff-mode matrix_ab --a 1.0 --b 1.2 \
		--players 300 --rounds 3000 --selection-strength 0.02 \
		--seeds 0:9 \
		--series w \
		--burn-in-frac 0.3 --tail 2000
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from analysis.cycle_metrics import classify_cycle_level
from analysis.decay_rate import estimate_decay_gamma
from analysis.jacobian_rotation import (
	lagged_jacobian_at_uniform,
	ode_jacobian_at_uniform,
	summarize_lagged,
	summarize_ode,
)
from simulation.run_simulation import SimConfig, simulate_series_window


def _auto_jobs() -> int:
	# os.cpu_count() returns logical CPUs; for heavy CPU-bound workloads, using all
	# logical CPUs can still be okay, but leaving 1 core helps responsiveness.
	cpus = os.cpu_count() or 1
	return max(1, int(cpus) - 1)


def _run_one(
	*,
	players: int,
	rounds: int,
	seed: int,
	payload: dict,
	series: str,
	burn_in: int,
	tail: int | None,
	selection_strength: float,
	metric_kwargs: dict,
) -> tuple[tuple[float, float] | None, int, bool, bool, bool, float, float, float, float, float, float, int]:
	"""Worker: run one simulation and return key + continuous diagnostics.

	Returns
	- key: (a,b) tuple in grid mode, else None
	- level: cycle_level (0..3)
	- s1/s2/s3: stage pass booleans
	- stage3_score/turn_strength: Stage3 diagnostics (0 when Stage3 not applicable)
	- max_amp: max Stage1 amplitude across series keys
	- max_corr: max Stage2 dominant autocorr score across series keys (0 when Stage2 not applicable)
	- gamma_fit: empirical envelope slope from peak amplitudes (None->0)
	- gamma_r2: goodness of fit for gamma (None->0)
	- gamma_n_peaks: number of peaks used (None->0)
	"""
	cfg = SimConfig(
		n_players=int(players),
		n_rounds=int(rounds),
		seed=int(seed),
		payoff_mode=str(payload["payoff_mode"]),
		gamma=float(payload["gamma"]),
		popularity_mode=str(payload.get("popularity_mode", "sampled")),
		epsilon=float(payload["epsilon"]),
		a=float(payload.get("a", 0.0)),
		b=float(payload.get("b", 0.0)),
		init_bias=float(payload.get("init_bias", 0.0)),
		evolution_mode=str(payload.get("evolution_mode", "sampled")),
		payoff_lag=int(payload.get("payoff_lag", 1)),
		selection_strength=float(selection_strength),
		out_csv=Path("outputs") / "_ignored.csv",
	)
	series_map = simulate_series_window(
		cfg,
		series=str(series),
		burn_in=int(burn_in),
		tail=int(tail) if tail is not None else None,
	)

	fit = estimate_decay_gamma(series_map, series_kind=str(series))
	env_gamma = float(fit.gamma) if fit is not None else 0.0
	env_gamma_r2 = float(fit.r2) if fit is not None else 0.0
	env_gamma_n_peaks = int(fit.n_peaks) if fit is not None else 0

	# We already windowed the series; classify over the full returned segment.
	res = classify_cycle_level(
		series_map,
		burn_in=0,
		tail=None,
		**metric_kwargs,
	)
	stage1_passed = bool(res.stage1.passed)
	stage2_passed = bool(res.stage2.passed) if res.stage2 is not None else False
	stage3_passed = bool(res.stage3.passed) if res.stage3 is not None else False
	stage3_score = float(res.stage3.score) if res.stage3 is not None else 0.0
	stage3_turn_strength = float(res.stage3.turn_strength) if res.stage3 is not None else 0.0
	max_amp = float(max(res.stage1.amplitudes.values())) if res.stage1.amplitudes else 0.0
	max_corr = (
		float(max(df.corr for df in res.stage2.frequencies.values()))
		if (res.stage2 is not None and res.stage2.frequencies)
		else 0.0
	)
	key = payload.get("key")
	return (
		key,
		int(res.level),
		stage1_passed,
		stage2_passed,
		stage3_passed,
		stage3_score,
		stage3_turn_strength,
		max_amp,
		max_corr,
		env_gamma,
		env_gamma_r2,
		env_gamma_n_peaks,
	)


def _parse_seeds(spec: str) -> list[int]:
	"""Parse seed spec.

	Supported formats:
	- "1,2,3"
	- "0:9" (inclusive range)
	- "0:9:2" (inclusive range with step)
	"""

	s = spec.strip()
	if not s:
		raise ValueError("--seeds cannot be empty")

	if "," in s:
		out: list[int] = []
		for part in s.split(","):
			part = part.strip()
			if not part:
				continue
			out.append(int(part))
		if not out:
			raise ValueError("--seeds list is empty")
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
		# inclusive end
		if step > 0:
			return list(range(start, end + 1, step))
		return list(range(start, end - 1, step))

	return [int(s)]


def _parse_float_grid(spec: str) -> list[float]:
	"""Parse float grid spec.

	Supported formats:
	- "0.5,1.0,1.5"
	- "0.5:1.5" (inclusive range, step=0.1 by default)
	- "0.5:1.5:0.25" (inclusive range with step)

	Notes
	- Range default step=0.1 is intentionally small but not too dense.
	- Inclusive end uses a tolerance to avoid float rounding issues.
	"""

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
			# normalize to positive step and swap bounds
			step = -step
			start, end = end, start
		# inclusive end, robust to floating point
		out: list[float] = []
		x = start
		tol = 1e-12
		while x <= end + tol:
			out.append(float(x))
			x += step
		return out

	return [float(s)]


def _parse_int_grid(spec: str) -> list[int]:
	"""Parse int grid spec.

	Supported formats:
	- "100,300,1000"
	- "100:1000" (inclusive range, step=100 by default)
	- "100:1000:200" (inclusive range with step)
	"""

	s = spec.strip()
	if not s:
		raise ValueError("grid spec cannot be empty")

	if "," in s:
		out: list[int] = []
		for part in s.split(","):
			part = part.strip()
			if not part:
				continue
			out.append(int(part))
		if not out:
			raise ValueError("grid list is empty")
		return out

	if ":" in s:
		parts = [p.strip() for p in s.split(":")]
		if len(parts) not in (2, 3):
			raise ValueError("Range format must be start:end or start:end:step")
		start = int(parts[0])
		end = int(parts[1])
		step = int(parts[2]) if len(parts) == 3 else 100
		if step == 0:
			raise ValueError("step cannot be 0")
		# inclusive end
		if step > 0:
			return list(range(start, end + 1, step))
		return list(range(start, end - 1, step))

	return [int(s)]


def _hopf_select_candidate(
	*,
	mode: str,
	scan: str,
	a: float,
	b: float,
	lo: float,
	hi: float,
	n: int,
	selection_strength: float,
	fd_h: float,
	summary_tol: float,
) -> tuple[float, float, str]:
	"""Pick the closest near-neutral point from a coarse Jacobian scan.

	Returns (a_sel, b_sel, note) where note is a short human-readable summary.
	"""
	mm = str(mode)
	ax = str(scan)
	if ax not in ("a", "b"):
		raise ValueError("scan must be 'a' or 'b'")
	if int(n) < 2:
		raise ValueError("n must be >= 2")
	aa0 = float(a)
	bb0 = float(b)
	lo2 = float(lo)
	hi2 = float(hi)
	if lo2 == hi2:
		raise ValueError("scan range is degenerate")
	if lo2 > hi2:
		lo2, hi2 = hi2, lo2

	grid = [lo2 + (hi2 - lo2) * i / float(int(n) - 1) for i in range(int(n))]

	best_param: float | None = None
	best_metric: float | None = None
	best_label: str = ""
	best_omega: float | None = None

	for param in grid:
		if ax == "a":
			aa, bb = float(param), float(bb0)
		else:
			aa, bb = float(aa0), float(param)

		if mm == "ode":
			rep = ode_jacobian_at_uniform(a=aa, b=bb, h=float(fd_h))
			s = summarize_ode(rep, tol=float(summary_tol))
			metric = float(s.alpha)
			omega = float(s.omega)
			label = str(s.stability)
		else:
			rep2 = lagged_jacobian_at_uniform(a=aa, b=bb, selection_strength=float(selection_strength), h=float(fd_h))
			s2 = summarize_lagged(rep2, tol=float(summary_tol))
			metric = float(s2.rho - 1.0)
			omega = None
			label = str(s2.stability)

		if best_metric is None or abs(metric) < abs(best_metric):
			best_metric = float(metric)
			best_param = float(param)
			best_label = str(label)
			best_omega = omega

	assert best_param is not None and best_metric is not None
	if ax == "a":
		aa_sel, bb_sel = float(best_param), float(bb0)
	else:
		aa_sel, bb_sel = float(aa0), float(best_param)

	if mm == "ode":
		note = f"hopf_select(mode=ode): closest |alpha| at {ax}={best_param:.8g} => alpha={best_metric:+.6g}, omega={float(best_omega or 0.0):.6g}, label={best_label}"
	else:
		note = f"hopf_select(mode=lagged): closest |rho-1| at {ax}={best_param:.8g} => rho-1={best_metric:+.6g}, label={best_label}"
	return (aa_sel, bb_sel, note)


def _aggregate_seed_rows(rows_out: list[dict]) -> dict:
	"""Aggregate per-seed rows into a compact summary dict."""
	counts = {0: 0, 1: 0, 2: 0, 3: 0}
	for r in rows_out:
		lv = int(r["cycle_level"])
		counts[lv] = counts.get(lv, 0) + 1
	den = len(rows_out) if rows_out else 1
	p_ge_2 = (counts.get(2, 0) + counts.get(3, 0)) / float(den)
	p_l3 = counts.get(3, 0) / float(den)
	mean_stage3_score = sum(float(r["stage3_score"]) for r in rows_out) / float(den)
	mean_stage3_turn_strength = sum(float(r["stage3_turn_strength"]) for r in rows_out) / float(den)
	mean_max_amp = sum(float(r["max_amp"]) for r in rows_out) / float(den)
	mean_max_corr = sum(float(r["max_corr"]) for r in rows_out) / float(den)
	return {
		"n_seeds": int(len(rows_out)),
		"p_level_ge_2": float(p_ge_2),
		"p_level_3": float(p_l3),
		"mean_max_amp": float(mean_max_amp),
		"mean_max_corr": float(mean_max_corr),
		"mean_stage3_score": float(mean_stage3_score),
		"mean_stage3_turn_strength": float(mean_stage3_turn_strength),
		"n_level_0": int(counts.get(0, 0)),
		"n_level_1": int(counts.get(1, 0)),
		"n_level_2": int(counts.get(2, 0)),
		"n_level_3": int(counts.get(3, 0)),
	}


def _rows_to_series(rows: list[dict], *, prefix: str) -> dict[str, list[float]]:
	out: dict[str, list[float]] = {}
	for row in rows:
		for k, v in row.items():
			if not k.startswith(prefix):
				continue
			name = k[len(prefix) :]
			out.setdefault(name, []).append(float(v))
	return out


def main() -> None:
	p = argparse.ArgumentParser(description="Multi-seed stability report (cycle_level distribution)")
	# simulation params
	p.add_argument("--players", type=int, default=300)
	p.add_argument(
		"--players-grid",
		type=str,
		default=None,
		help='Optional int grid for players: "100,300,1000" or "100:1000:200". If set, runs a players sweep (multi-seed per players).',
	)
	p.add_argument("--rounds", type=int, default=3000)
	p.add_argument("--seeds", type=str, default="0:9", help='Seed spec: "1,2,3" or "0:9" or "0:9:2"')
	p.add_argument(
		"--payoff-mode",
		type=str,
		default="matrix_ab",
		choices=["count_cycle", "matrix_ab"],
	)
	p.add_argument(
		"--popularity-mode",
		type=str,
		default="sampled",
		choices=["sampled", "expected"],
		help="How dungeon popularity is updated each round: sampled choices (default) or expected distribution from weights (denoised)",
	)
	p.add_argument(
		"--evolution-mode",
		type=str,
		default="sampled",
		choices=["sampled", "mean_field"],
		help="Weight update rule. 'sampled' uses per-strategy averages from sampled players (default). 'mean_field' uses deterministic expected rewards u=Ax and replicator mapping (matrix_ab only).",
	)
	p.add_argument(
		"--payoff-lag",
		type=int,
		default=1,
		choices=[0, 1],
		help="Only used in evolution_mode=mean_field. 0: use current x_t for payoff u(x_t). 1: use lagged x_{t-1} for payoff u(x_{t-1}).",
	)
	p.add_argument("--gamma", type=float, default=0.1)
	p.add_argument("--epsilon", type=float, default=0.0)
	p.add_argument("--a", type=float, default=1.0)
	p.add_argument("--b", type=float, default=1.2)
	p.add_argument(
		"--init-bias",
		type=float,
		default=0.0,
		help="Initial symmetry-breaking bias applied to weights: w0=(1+bias,1-bias,1) for (aggressive,defensive,balanced). Useful with popularity_mode=expected to induce phase rotation; requires |bias|<1.",
	)
	p.add_argument(
		"--a-grid",
		type=str,
		default=None,
		help='Optional float grid for a: "0.5:1.5:0.25" or "0.5,1.0,1.5". If set, runs a/b grid mode.',
	)
	p.add_argument(
		"--b-grid",
		type=str,
		default=None,
		help='Optional float grid for b: "0.5:1.5:0.25" or "0.5,1.0,1.5". If set, runs a/b grid mode.',
	)
	p.add_argument("--selection-strength", type=float, default=0.02)
	p.add_argument(
		"--jobs",
		type=int,
		default=1,
		help="Parallel worker processes. Use 1 for sequential (default). Use 0 for auto (cpu_count-1).",
	)
	# metric options
	p.add_argument("--series", type=str, default="w", choices=["w", "p"], help="Use weights (w_*) or proportions (p_*)")
	p.add_argument("--burn-in", type=int, default=None)
	p.add_argument("--burn-in-frac", type=float, default=0.3)
	p.add_argument("--tail", type=int, default=2000)
	p.add_argument("--amplitude-threshold", type=float, default=0.02)
	p.add_argument("--min-lag", type=int, default=2)
	p.add_argument("--max-lag", type=int, default=500)
	p.add_argument("--corr-threshold", type=float, default=0.3)
	p.add_argument("--eta", type=float, default=0.6)
	p.add_argument("--min-turn-strength", type=float, default=0.0)
	p.add_argument(
		"--stage3-method",
		type=str,
		default="turning",
		choices=["turning", "centroid"],
		help="Stage3 direction metric: 'turning' (3-point local turning, recommended) or 'centroid' (2-point around centroid)",
	)
	p.add_argument(
		"--phase-smoothing",
		type=int,
		default=1,
		help="Optional moving-average window (odd int) applied before Stage3 when using stage3-method=turning. Use 1 to disable.",
	)
	p.add_argument(
		"--stage3-window",
		type=int,
		default=None,
		help="Optional sliding-window length for Stage3 turning-based aggregation. If set, Stage3 uses windowed quantile instead of full-tail score.",
	)
	p.add_argument(
		"--stage3-step",
		type=int,
		default=20,
		help="Sliding-window step for Stage3 aggregation when --stage3-window is set.",
	)
	p.add_argument(
		"--stage3-quantile",
		type=float,
		default=0.75,
		help="Quantile (e.g., 0.75/0.85/0.90) used for windowed Stage3 score/strength when --stage3-window is set.",
	)
	p.add_argument("--out", type=Path, default=None, help="Optional CSV report output")

	# Optional: auto-select a near-neutral candidate via Jacobian scan, then validate via simulation.
	p.add_argument(
		"--hopf-select",
		action="store_true",
		help="If set, first scan along one parameter axis using Jacobian summary (alpha or rho-1) and replace (a,b) with the closest near-neutral point before running simulation.",
	)
	p.add_argument("--hopf-mode", choices=["ode", "lagged"], default="lagged")
	p.add_argument("--hopf-scan", choices=["a", "b"], default="b")
	p.add_argument("--hopf-min", type=float, default=None, help="Scan range min for the chosen axis")
	p.add_argument("--hopf-max", type=float, default=None, help="Scan range max for the chosen axis")
	p.add_argument("--hopf-n", type=int, default=41, help="Number of scan points for hopf-select")
	p.add_argument("--hopf-fd-h", type=float, default=1e-6)
	p.add_argument("--hopf-summary-tol", type=float, default=1e-6)
	args = p.parse_args()

	seed_list = _parse_seeds(args.seeds)
	burn_in = args.burn_in
	if burn_in is None:
		burn_in = int(round(float(args.rounds) * float(args.burn_in_frac)))
	burn_in = max(0, burn_in)

	normalize_for_phase = (args.series == "w")
	jobs = int(args.jobs)
	if jobs == 0:
		jobs = _auto_jobs()
	if jobs < 1:
		raise ValueError("--jobs must be >= 0")

	if args.players_grid is not None and (args.a_grid is not None or args.b_grid is not None):
		raise ValueError("Do not combine --players-grid with --a-grid/--b-grid (would explode the sweep).")

	# Optional Hopf-based candidate selection (coarse scan).
	if bool(args.hopf_select):
		ax = str(args.hopf_scan)
		lo = args.hopf_min
		hi = args.hopf_max
		if lo is None or hi is None:
			raise ValueError("--hopf-min/--hopf-max are required when using --hopf-select")
		aa_sel, bb_sel, note = _hopf_select_candidate(
			mode=str(args.hopf_mode),
			scan=ax,
			a=float(args.a),
			b=float(args.b),
			lo=float(lo),
			hi=float(hi),
			n=int(args.hopf_n),
			selection_strength=float(args.selection_strength),
			fd_h=float(args.hopf_fd_h),
			summary_tol=float(args.hopf_summary_tol),
		)
		print("=== hopf-select (Jacobian) ===")
		print(note)
		args.a = float(aa_sel)
		args.b = float(bb_sel)

	# Grid mode: sweep a/b and write a table of P(level>=2).
	if args.a_grid is not None or args.b_grid is not None:
		a_vals = _parse_float_grid(args.a_grid) if args.a_grid is not None else [float(args.a)]
		b_vals = _parse_float_grid(args.b_grid) if args.b_grid is not None else [float(args.b)]

		out_path = args.out
		if out_path is None:
			out_path = Path("outputs") / "ab_grid_report.csv"

		metric_kwargs = {
			"amplitude_threshold": float(args.amplitude_threshold),
			"min_lag": int(args.min_lag),
			"max_lag": int(args.max_lag),
			"corr_threshold": float(args.corr_threshold),
			"eta": float(args.eta),
			"min_turn_strength": float(args.min_turn_strength),
			"normalize_for_phase": bool(normalize_for_phase),
			"stage3_method": str(args.stage3_method),
			"phase_smoothing": int(args.phase_smoothing),
			"stage3_window": (int(args.stage3_window) if args.stage3_window is not None else None),
			"stage3_step": int(args.stage3_step),
			"stage3_quantile": float(args.stage3_quantile),
		}

		# Launch per-(a,b,seed) tasks.
		tasks: list[dict] = []
		for a in a_vals:
			for b in b_vals:
				key = (float(a), float(b))
				for sd in seed_list:
					tasks.append(
						{
							"seed": int(sd),
							"payload": {
								"payoff_mode": str(args.payoff_mode),
								"popularity_mode": str(args.popularity_mode),
								"gamma": float(args.gamma),
								"epsilon": float(args.epsilon),
								"a": float(a),
								"b": float(b),
								"key": key,
							},
						}
					)

		# Collect results grouped by (a,b).
		agg_levels: dict[tuple[float, float], list[int]] = { (float(a), float(b)): [] for a in a_vals for b in b_vals }
		agg_s3_scores: dict[tuple[float, float], list[float]] = { (float(a), float(b)): [] for a in a_vals for b in b_vals }
		agg_s3_turns: dict[tuple[float, float], list[float]] = { (float(a), float(b)): [] for a in a_vals for b in b_vals }
		agg_max_amp: dict[tuple[float, float], list[float]] = { (float(a), float(b)): [] for a in a_vals for b in b_vals }
		agg_max_corr: dict[tuple[float, float], list[float]] = { (float(a), float(b)): [] for a in a_vals for b in b_vals }
		agg_env_gamma: dict[tuple[float, float], list[float]] = { (float(a), float(b)): [] for a in a_vals for b in b_vals }
		agg_env_gamma_r2: dict[tuple[float, float], list[float]] = { (float(a), float(b)): [] for a in a_vals for b in b_vals }
		agg_env_gamma_n_peaks: dict[tuple[float, float], list[int]] = { (float(a), float(b)): [] for a in a_vals for b in b_vals }

		if jobs == 1:
			for task in tasks:
				key, lv, _s1, _s2, _s3, s3, ts, max_amp, max_corr, gfit, gr2, gn = _run_one(
					players=int(args.players),
					rounds=int(args.rounds),
					seed=int(task["seed"]),
					payload=task["payload"],
					series=str(args.series),
					burn_in=burn_in,
					tail=int(args.tail) if args.tail is not None else None,
					selection_strength=float(args.selection_strength),
					metric_kwargs=metric_kwargs,
				)
				if key is None:
					continue
				agg_levels[key].append(lv)
				agg_s3_scores[key].append(s3)
				agg_s3_turns[key].append(ts)
				agg_max_amp[key].append(max_amp)
				agg_max_corr[key].append(max_corr)
				agg_env_gamma[key].append(float(gfit))
				agg_env_gamma_r2[key].append(float(gr2))
				agg_env_gamma_n_peaks[key].append(int(gn))
		else:
			with ProcessPoolExecutor(max_workers=int(jobs)) as ex:
				futs = [
					ex.submit(
						_run_one,
						players=int(args.players),
						rounds=int(args.rounds),
						seed=int(task["seed"]),
						payload=task["payload"],
						series=str(args.series),
						burn_in=burn_in,
						tail=int(args.tail) if args.tail is not None else None,
						selection_strength=float(args.selection_strength),
						metric_kwargs=metric_kwargs,
					)
					for task in tasks
				]
				for fut in as_completed(futs):
					key, lv, _s1, _s2, _s3, s3, ts, max_amp, max_corr, gfit, gr2, gn = fut.result()
					if key is None:
						continue
					agg_levels[key].append(lv)
					agg_s3_scores[key].append(s3)
					agg_s3_turns[key].append(ts)
					agg_max_amp[key].append(max_amp)
					agg_max_corr[key].append(max_corr)
					agg_env_gamma[key].append(float(gfit))
					agg_env_gamma_r2[key].append(float(gr2))
					agg_env_gamma_n_peaks[key].append(int(gn))

		out_rows: list[dict] = []
		for a in a_vals:
			for b in b_vals:
				key = (float(a), float(b))
				levels = agg_levels.get(key, [])
				stage3_scores = agg_s3_scores.get(key, [])
				stage3_turn_strengths = agg_s3_turns.get(key, [])
				max_amps = agg_max_amp.get(key, [])
				max_corrs = agg_max_corr.get(key, [])
				gfits = agg_env_gamma.get(key, [])
				gr2s = agg_env_gamma_r2.get(key, [])
				gns = agg_env_gamma_n_peaks.get(key, [])
				counts = {0: 0, 1: 0, 2: 0, 3: 0}
				for lv in levels:
					counts[lv] = counts.get(lv, 0) + 1
				den = len(levels) if levels else 1
				p_ge_2 = (counts.get(2, 0) + counts.get(3, 0)) / float(den)
				p_l3 = counts.get(3, 0) / float(den)
				mean_stage3_score = (sum(stage3_scores) / float(len(stage3_scores))) if stage3_scores else 0.0
				mean_stage3_turn_strength = (
					sum(stage3_turn_strengths) / float(len(stage3_turn_strengths))
				) if stage3_turn_strengths else 0.0
				mean_max_amp = (sum(max_amps) / float(len(max_amps))) if max_amps else 0.0
				mean_max_corr = (sum(max_corrs) / float(len(max_corrs))) if max_corrs else 0.0
				mean_env_gamma = (sum(gfits) / float(len(gfits))) if gfits else 0.0
				mean_env_gamma_r2 = (sum(gr2s) / float(len(gr2s))) if gr2s else 0.0
				# Note: env_gamma_n_peaks is per-seed; 0 means "fit unavailable".
				mean_env_gamma_n_peaks = (sum(int(x) for x in gns) / float(len(gns))) if gns else 0.0
				out_rows.append(
					{
						"a": float(a),
						"b": float(b),
						"n_seeds": int(len(seed_list)),
						"p_level_ge_2": float(p_ge_2),
						"p_level_3": float(p_l3),
						"mean_max_amp": float(mean_max_amp),
						"mean_max_corr": float(mean_max_corr),
						"mean_stage3_score": float(mean_stage3_score),
						"mean_stage3_turn_strength": float(mean_stage3_turn_strength),
						"mean_env_gamma": float(mean_env_gamma),
						"mean_env_gamma_r2": float(mean_env_gamma_r2),
						"mean_env_gamma_n_peaks": float(mean_env_gamma_n_peaks),
						"n_level_0": int(counts.get(0, 0)),
						"n_level_1": int(counts.get(1, 0)),
						"n_level_2": int(counts.get(2, 0)),
						"n_level_3": int(counts.get(3, 0)),
					}
				)

		out_path.parent.mkdir(parents=True, exist_ok=True)
		with out_path.open("w", newline="") as f:
			w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ["a", "b", "p_level_ge_2"])
			w.writeheader()
			w.writerows(out_rows)

		print("=== a/b grid stability report ===")
		print(
			f"payoff_mode={args.payoff_mode} a={args.a} b={args.b} gamma={args.gamma} epsilon={args.epsilon} "
			f"popularity_mode={args.popularity_mode} evolution_mode={args.evolution_mode} payoff_lag={args.payoff_lag} init_bias={args.init_bias}"
		)
		print(f"players={args.players} rounds={args.rounds} selection_strength={args.selection_strength}")
		print(f"series={args.series} burn_in={burn_in} tail={args.tail}")
		print(f"seeds_n={len(seed_list)} a_points={len(a_vals)} b_points={len(b_vals)}")
		print(f"Wrote grid CSV: {out_path}")
		return

	levels: list[int] = []
	rows_out: list[dict] = []
	metric_kwargs = {
		"amplitude_threshold": float(args.amplitude_threshold),
		"min_lag": int(args.min_lag),
		"max_lag": int(args.max_lag),
		"corr_threshold": float(args.corr_threshold),
		"eta": float(args.eta),
		"min_turn_strength": float(args.min_turn_strength),
		"normalize_for_phase": bool(normalize_for_phase),
		"stage3_method": str(args.stage3_method),
		"phase_smoothing": int(args.phase_smoothing),
		"stage3_window": (int(args.stage3_window) if args.stage3_window is not None else None),
		"stage3_step": int(args.stage3_step),
		"stage3_quantile": float(args.stage3_quantile),
	}

	payload_base = {
		"payoff_mode": str(args.payoff_mode),
		"popularity_mode": str(args.popularity_mode),
		"evolution_mode": str(args.evolution_mode),
		"payoff_lag": int(args.payoff_lag),
		"gamma": float(args.gamma),
		"epsilon": float(args.epsilon),
		"a": float(args.a),
		"b": float(args.b),
		"init_bias": float(args.init_bias),
		"key": None,
	}

	# Players sweep mode: run the same multi-seed report across multiple n_players.
	if args.players_grid is not None:
		players_vals = _parse_int_grid(str(args.players_grid))
		if not players_vals:
			raise ValueError("--players-grid resolved to empty")

		out_rows: list[dict] = []
		print("=== Players sweep (multi-seed) ===")
		print(f"payoff_mode={args.payoff_mode} a={args.a} b={args.b} gamma={args.gamma} epsilon={args.epsilon}")
		print(f"rounds={args.rounds} selection_strength={args.selection_strength} popularity_mode={args.popularity_mode}")
		print(f"series={args.series} burn_in={burn_in} tail={args.tail}")
		print(f"seeds: {seed_list[0]}..{seed_list[-1]} (n={len(seed_list)})" if seed_list else "seeds: (none)")
		print("\nplayers\tP(level>=2)\tP(level3)\tmean_s3_score\tmean_s3_strength\tmean_env_gamma")

		for npl in players_vals:
			rows_local: list[dict] = []
			if jobs == 1:
				for sd in seed_list:
					_key, lv, s1, s2, s3p, s3_score, ts, max_amp, max_corr, gfit, gr2, gn = _run_one(
						players=int(npl),
						rounds=int(args.rounds),
						seed=int(sd),
						payload=payload_base,
						series=str(args.series),
						burn_in=burn_in,
						tail=int(args.tail) if args.tail is not None else None,
						selection_strength=float(args.selection_strength),
						metric_kwargs=metric_kwargs,
					)
					rows_local.append(
						{
							"seed": int(sd),
							"cycle_level": int(lv),
							"stage1_passed": bool(s1),
							"stage2_passed": bool(s2),
							"stage3_passed": bool(s3p),
							"max_amp": float(max_amp),
							"max_corr": float(max_corr),
							"stage3_score": float(s3_score),
							"stage3_turn_strength": float(ts),
							"env_gamma": float(gfit),
							"env_gamma_r2": float(gr2),
							"env_gamma_n_peaks": int(gn),
						}
					)
			else:
				with ProcessPoolExecutor(max_workers=int(jobs)) as ex:
					futs = {
						ex.submit(
							_run_one,
							players=int(npl),
							rounds=int(args.rounds),
							seed=int(sd),
							payload=payload_base,
							series=str(args.series),
							burn_in=burn_in,
							tail=int(args.tail) if args.tail is not None else None,
							selection_strength=float(args.selection_strength),
							metric_kwargs=metric_kwargs,
						): int(sd)
						for sd in seed_list
					}
					for fut in as_completed(futs):
						sd = futs[fut]
						_key, lv, s1, s2, s3p, s3_score, ts, max_amp, max_corr, gfit, gr2, gn = fut.result()
						rows_local.append(
							{
								"seed": int(sd),
								"cycle_level": int(lv),
								"stage1_passed": bool(s1),
								"stage2_passed": bool(s2),
								"stage3_passed": bool(s3p),
								"max_amp": float(max_amp),
								"max_corr": float(max_corr),
								"stage3_score": float(s3_score),
								"stage3_turn_strength": float(ts),
								"env_gamma": float(gfit),
								"env_gamma_r2": float(gr2),
								"env_gamma_n_peaks": int(gn),
							}
						)

			agg = _aggregate_seed_rows(rows_local)
			# gamma summary (empirical envelope slope)
			mean_env_gamma = sum(float(r["env_gamma"]) for r in rows_local) / float(len(rows_local) if rows_local else 1)
			mean_env_gamma_r2 = sum(float(r["env_gamma_r2"]) for r in rows_local) / float(len(rows_local) if rows_local else 1)
			mean_env_gamma_n_peaks = sum(int(r["env_gamma_n_peaks"]) for r in rows_local) / float(len(rows_local) if rows_local else 1)
			print(
				f"{int(npl)}\t{agg['p_level_ge_2']:.3f}\t\t{agg['p_level_3']:.3f}\t\t{agg['mean_stage3_score']:.3f}\t\t{agg['mean_stage3_turn_strength']:.6g}\t\t{mean_env_gamma:+.4g} (r2~{mean_env_gamma_r2:.2f}, peaks~{mean_env_gamma_n_peaks:.1f})"
			)
			out_rows.append(
				{
					"players": int(npl),
					"a": float(args.a),
					"b": float(args.b),
					"rounds": int(args.rounds),
					"selection_strength": float(args.selection_strength),
					"series": str(args.series),
					"burn_in": int(burn_in),
					"tail": (int(args.tail) if args.tail is not None else None),
					"mean_env_gamma": float(mean_env_gamma),
					"mean_env_gamma_r2": float(mean_env_gamma_r2),
					"mean_env_gamma_n_peaks": float(mean_env_gamma_n_peaks),
					**agg,
				}
			)

		if args.out is not None:
			args.out.parent.mkdir(parents=True, exist_ok=True)
			with args.out.open("w", newline="") as f:
				w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ["players", "p_level_ge_2"])
				w.writeheader()
				w.writerows(out_rows)
			print(f"\nWrote players sweep CSV: {args.out}")
		return

	if jobs == 1:
		for sd in seed_list:
			_key, lv, s1, s2, s3p, s3_score, ts, max_amp, max_corr, gfit, gr2, gn = _run_one(
				players=int(args.players),
				rounds=int(args.rounds),
				seed=int(sd),
				payload=payload_base,
				series=str(args.series),
				burn_in=burn_in,
				tail=int(args.tail) if args.tail is not None else None,
				selection_strength=float(args.selection_strength),
				metric_kwargs=metric_kwargs,
			)
			levels.append(int(lv))
			rows_out.append(
				{
					"seed": int(sd),
					"cycle_level": int(lv),
					"stage1_passed": bool(s1),
					"stage2_passed": bool(s2),
					"stage3_passed": bool(s3p),
					"max_amp": float(max_amp),
					"max_corr": float(max_corr),
					"stage3_score": float(s3_score),
					"stage3_turn_strength": float(ts),
					"env_gamma": float(gfit),
					"env_gamma_r2": float(gr2),
					"env_gamma_n_peaks": int(gn),
				}
			)
	else:
		with ProcessPoolExecutor(max_workers=int(jobs)) as ex:
			futs = {
				ex.submit(
					_run_one,
					players=int(args.players),
					rounds=int(args.rounds),
					seed=int(sd),
					payload=payload_base,
					series=str(args.series),
					burn_in=burn_in,
					tail=int(args.tail) if args.tail is not None else None,
					selection_strength=float(args.selection_strength),
					metric_kwargs=metric_kwargs,
				): int(sd)
				for sd in seed_list
			}
			for fut in as_completed(futs):
				sd = futs[fut]
				_key, lv, s1, s2, s3p, s3_score, ts, max_amp, max_corr, gfit, gr2, gn = fut.result()
				levels.append(int(lv))
				rows_out.append(
					{
						"seed": int(sd),
						"cycle_level": int(lv),
						"stage1_passed": bool(s1),
						"stage2_passed": bool(s2),
						"stage3_passed": bool(s3p),
						"max_amp": float(max_amp),
						"max_corr": float(max_corr),
						"stage3_score": float(s3_score),
						"stage3_turn_strength": float(ts),
						"env_gamma": float(gfit),
						"env_gamma_r2": float(gr2),
						"env_gamma_n_peaks": int(gn),
					}
				)

	# Summary
	counts = {0: 0, 1: 0, 2: 0, 3: 0}
	for lv in levels:
		counts[lv] = counts.get(lv, 0) + 1
	den = len(levels) if levels else 1
	p_ge_2 = (counts.get(2, 0) + counts.get(3, 0)) / float(den)

	print("=== Multi-seed stability report ===")
	print(
		f"payoff_mode={args.payoff_mode} a={args.a} b={args.b} gamma={args.gamma} epsilon={args.epsilon} "
		f"popularity_mode={args.popularity_mode} evolution_mode={args.evolution_mode} payoff_lag={args.payoff_lag} init_bias={args.init_bias}"
	)
	print(f"players={args.players} rounds={args.rounds} selection_strength={args.selection_strength}")
	print(f"series={args.series} burn_in={burn_in} tail={args.tail}")
	print(f"thresholds: amp={args.amplitude_threshold} corr={args.corr_threshold} eta={args.eta} min_turn_strength={args.min_turn_strength}")
	print(f"seeds: {seed_list[0]}..{seed_list[-1]} (n={len(seed_list)})" if seed_list else "seeds: (none)")
	print("level_counts:", counts)
	print(f"P(level>=2) = {p_ge_2:.3f}")
	mean_env_gamma = sum(float(r["env_gamma"]) for r in rows_out) / float(len(rows_out) if rows_out else 1)
	mean_env_gamma_r2 = sum(float(r["env_gamma_r2"]) for r in rows_out) / float(len(rows_out) if rows_out else 1)
	mean_env_gamma_n_peaks = sum(int(r["env_gamma_n_peaks"]) for r in rows_out) / float(len(rows_out) if rows_out else 1)
	print(f"mean_env_gamma = {mean_env_gamma:+.6g} (r2~{mean_env_gamma_r2:.2f}, peaks~{mean_env_gamma_n_peaks:.1f})")

	if args.out is not None:
		args.out.parent.mkdir(parents=True, exist_ok=True)
		with args.out.open("w", newline="") as f:
			w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()) if rows_out else ["seed", "cycle_level"])
			w.writeheader()
			w.writerows(rows_out)
		print(f"Wrote report CSV: {args.out}")


if __name__ == "__main__":
	main()
