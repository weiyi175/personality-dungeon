"""Rho-curve sweep CLI.

Sweeps selection strength k and estimates P(Level3) as a function of the
linearized criticality measure rho (spectral radius of the lagged Jacobian).

Design goals
- Streaming CSV writes (flush every point) so partial runs are useful.
- Optional multiprocessing across seeds.
- Minimal dependencies (stdlib + existing project modules).

Example
	python3 -m simulation.rho_curve \
		--payoff-mode matrix_ab --a 0.4 --b 0.2425406997 \
		--gamma 0.1 --epsilon 0.0 \
		--popularity-mode sampled --evolution-mode sampled --payoff-lag 1 \
		--players-grid 50,200,1000 --rounds 4000 --seeds 0:29 \
		--k-grid 0.1:1.0:0.02 --series w --burn-in-frac 0.25 --tail 600 \
		--eta 0.55 --jobs 10 \
		--out outputs/sweeps/rho_curve/main.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from analysis.cycle_metrics import classify_cycle_level
from analysis.decay_rate import estimate_decay_gamma
from analysis.jacobian_rotation import lagged_jacobian_at_uniform, summarize_lagged
from simulation.run_simulation import SimConfig, simulate_series_window


SWEEP_CSV_FIELDNAMES = [
	"players",
	"a",
	"b",
	"selection_strength",
	"rho",
	"rho_minus_1",
	"p_level_ge_2",
	"p_level_3",
	"mean_stage3_score",
	"mean_max_amp",
	"mean_max_corr",
	"mean_stage2_statistic",
	"stage2_statistic_name",
	"mean_stage2_effective_window_n",
	"mean_env_gamma",
	"mean_env_gamma_r2",
	"mean_env_gamma_n_peaks",
	"n_seeds",
	"series",
	"rounds",
	"burn_in",
	"tail",
	# protocol provenance (so CSV can reconstruct the sweep config)
	"payoff_mode",
	"gamma",
	"epsilon",
	"seeds",
	"k_grid",
	"players_grid",
	# Stage1
	"amplitude_threshold",
	"amplitude_normalization",
	"amplitude_threshold_factor",
	"amplitude_control_reference",
	"amplitude_threshold_effective",
	"mean_max_amp_norm",
	# Stage2
	"corr_threshold",
	"stage2_method",
	"stage2_prefilter",
	"power_ratio_kappa",
	"permutation_alpha",
	"permutation_resamples",
	"permutation_seed",
	# Level3 / Stage3
	"eta",
	"min_turn_strength",
	"stage3_method",
	"phase_smoothing",
	"stage3_window",
	"stage3_step",
	"stage3_quantile",
	# other model knobs
	"popularity_mode",
	"evolution_mode",
	"payoff_lag",
	"init_bias",
	# provenance for Stage1 normalization input
	"amplitude_control_cache",
]


def _load_amplitude_control_cache(
	path: Path,
	*,
	expected_series: str,
	expected_burn_in: int,
	expected_tail: int | None,
) -> dict[int, dict[str, float]]:
	"""Load control cache for amplitude normalization.

	Expected JSON shape (minimal):
	{
	  "series": "p" | "w",
	  "burn_in": <int>,
	  "tail": <int|null>,
	  "by_players": {
		  "100": {"mean_max_amp": 0.01, "std_max_amp": 0.003},
		  ...
	  }
	}
	"""
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	series = str(data.get("series", ""))
	if series and series != str(expected_series):
		raise ValueError(
			f"Control cache series mismatch: cache={series!r} vs expected={expected_series!r}"
		)

	burn_in = data.get("burn_in")
	if burn_in is not None and int(burn_in) != int(expected_burn_in):
		raise ValueError(
			f"Control cache burn_in mismatch: cache={int(burn_in)} vs expected={int(expected_burn_in)}"
		)

	tail = data.get("tail")
	if tail is not None:
		tail = int(tail)
	if tail != expected_tail:
		raise ValueError(
			f"Control cache tail mismatch: cache={tail!r} vs expected={expected_tail!r}"
		)

	by_players = data.get("by_players")
	if not isinstance(by_players, dict):
		raise ValueError("Control cache must contain dict field 'by_players'")

	out: dict[int, dict[str, float]] = {}
	for k, v in by_players.items():
		try:
			players = int(k)
		except Exception:
			continue
		if not isinstance(v, dict):
			continue
		row: dict[str, float] = {}
		for name in ("mean_max_amp", "std_max_amp"):
			if name in v:
				try:
					row[name] = float(v[name])
				except Exception:
					pass
		if row:
			out[int(players)] = row

	if not out:
		raise ValueError(f"Control cache has no usable entries: {path}")
	return out


def _resolve_amplitude_control_reference(
	*,
	players: int,
	normalization: str,
	override_reference: float | None,
	cache: dict[int, dict[str, float]] | None,
) -> float | None:
	if str(normalization) == "none":
		return None
	if override_reference is not None:
		return float(override_reference)
	if cache is None:
		return None
	row = cache.get(int(players))
	if not row:
		return None
	if str(normalization) == "control_mean":
		return float(row.get("mean_max_amp", 0.0))
	if str(normalization) == "control_std":
		return float(row.get("std_max_amp", 0.0))
	raise ValueError(f"Unknown amplitude normalization: {normalization!r}")


def _auto_jobs() -> int:
	cpus = os.cpu_count() or 1
	return max(1, int(cpus) - 1)


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
		if step > 0:
			return list(range(start, end + 1, step))
		return list(range(start, end - 1, step))

	return [int(s)]


@dataclass(frozen=True, slots=True)
class OneSeedResult:
	level: int
	stage3_passed: bool
	stage3_score: float
	env_gamma: float
	env_gamma_r2: float
	env_gamma_n_peaks: int
	max_amp: float
	max_corr: float
	stage2_statistic_name: str
	stage2_statistic: float
	stage2_effective_window_n: int


def _run_one_seed(
	*,
	seed: int,
	players: int,
	rounds: int,
	payoff_mode: str,
	popularity_mode: str,
	evolution_mode: str,
	payoff_lag: int,
	gamma: float,
	epsilon: float,
	a: float,
	b: float,
	init_bias: float,
	selection_strength: float,
	series: str,
	burn_in: int,
	tail: int | None,
	metric_kwargs: dict,
) -> OneSeedResult:
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
		init_bias=float(init_bias),
		evolution_mode=str(evolution_mode),
		payoff_lag=int(payoff_lag),
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

	res = classify_cycle_level(series_map, burn_in=0, tail=None, **metric_kwargs)
	max_amp = float(max(res.stage1.amplitudes.values())) if res.stage1.amplitudes else 0.0
	max_corr = (
		float(max(df.corr for df in res.stage2.frequencies.values()))
		if (res.stage2 is not None and res.stage2.frequencies)
		else 0.0
	)
	stage2_statistic_name = ""
	stage2_statistic = 0.0
	stage2_effective_window_n = 0
	if res.stage2 is not None:
		stage2_statistic_name = str(res.stage2.statistic_name or "")
		stage2_statistic = float(res.stage2.statistic) if res.stage2.statistic is not None else 0.0
		stage2_effective_window_n = int(res.stage2.effective_window_n)
	stage3_passed = bool(res.stage3.passed) if res.stage3 is not None else False
	stage3_score = float(res.stage3.score) if res.stage3 is not None else 0.0
	return OneSeedResult(
		level=int(res.level),
		stage3_passed=bool(stage3_passed),
		stage3_score=float(stage3_score),
		env_gamma=float(env_gamma),
		env_gamma_r2=float(env_gamma_r2),
		env_gamma_n_peaks=int(env_gamma_n_peaks),
		max_amp=float(max_amp),
		max_corr=float(max_corr),
		stage2_statistic_name=str(stage2_statistic_name),
		stage2_statistic=float(stage2_statistic),
		stage2_effective_window_n=int(stage2_effective_window_n),
	)


def _load_completed_points(path: Path) -> set[tuple[int, float]]:
	completed: set[tuple[int, float]] = set()
	with path.open(newline="") as rf:
		reader = csv.DictReader(rf)
		for row in reader:
			try:
				p = int(row["players"])
				k = float(row["selection_strength"])
			except Exception:
				continue
			completed.add((p, round(k, 10)))
	return completed


def main() -> None:
	p = argparse.ArgumentParser(description="Criticality curve: P(Level3) vs rho")
	# Core model params
	p.add_argument("--payoff-mode", choices=["count_cycle", "matrix_ab"], default="matrix_ab")
	p.add_argument("--a", type=float, default=0.4)
	p.add_argument("--b", type=float, default=0.27)
	p.add_argument("--gamma", type=float, default=0.1)
	p.add_argument("--epsilon", type=float, default=0.0)
	p.add_argument("--popularity-mode", choices=["sampled", "expected"], default="sampled")
	p.add_argument("--evolution-mode", choices=["sampled", "mean_field"], default="sampled")
	p.add_argument("--payoff-lag", type=int, choices=[0, 1], default=1)
	p.add_argument("--init-bias", type=float, default=0.0)

	# Sweep axes
	p.add_argument(
		"--k-grid",
		type=str,
		required=True,
		help='Selection strength grid: "0.05:0.6:0.05" or "0.1,0.2,0.3"',
	)

	# Simulation params
	p.add_argument("--players", type=int, default=200)
	p.add_argument(
		"--players-grid",
		type=str,
		default=None,
		help='Optional players grid: "50,200,1000" or "50:1000:50". If set, runs N-scaling sweeps.',
	)
	p.add_argument("--rounds", type=int, default=6000)
	p.add_argument("--seeds", type=str, default="0:19")
	p.add_argument("--series", choices=["w", "p"], default="w")
	p.add_argument("--burn-in", type=int, default=None)
	p.add_argument("--burn-in-frac", type=float, default=0.3)
	p.add_argument("--tail", type=int, default=3000)
	p.add_argument("--jobs", type=int, default=0, help="Worker processes. 0 => auto (cpu_count-1)")

	# Metrics (keep aligned with seed_stability defaults)
	p.add_argument("--amplitude-threshold", type=float, default=0.02)
	p.add_argument(
		"--amplitude-normalization",
		choices=["none", "control_mean", "control_std"],
		default="none",
		help="Stage1 amplitude gate normalization. 'none' keeps absolute threshold; others require control reference.",
	)
	p.add_argument(
		"--amplitude-control-reference",
		type=float,
		default=None,
		help="Override control reference for Stage1 amplitude normalization (bypasses cache).",
	)
	p.add_argument(
		"--amplitude-threshold-factor",
		type=float,
		default=3.0,
		help="Effective threshold = factor * control_reference when normalization != none.",
	)
	p.add_argument(
		"--amplitude-control-cache",
		type=Path,
		default=None,
		help="Path to JSON cache with control mean/std max amplitude by players.",
	)
	p.add_argument("--min-lag", type=int, default=2)
	p.add_argument("--max-lag", type=int, default=500)
	p.add_argument("--corr-threshold", type=float, default=0.3)
	p.add_argument(
		"--stage2-method",
		choices=["autocorr_threshold", "fft_power_ratio", "permutation_p"],
		default="autocorr_threshold",
		help="Stage2 frequency significance method. Default keeps legacy autocorr threshold.",
	)
	p.add_argument(
		"--stage2-prefilter",
		type=int,
		choices=[0, 1],
		default=1,
		help="If 1, require autocorr threshold before running heavier Stage2 methods.",
	)
	p.add_argument(
		"--power-ratio-kappa",
		type=float,
		default=8.0,
		help="Stage2 fft_power_ratio threshold kappa: pass if power_ratio >= kappa.",
	)
	p.add_argument(
		"--permutation-alpha",
		type=float,
		default=0.05,
		help="Stage2 permutation_p alpha: pass if p_value <= alpha.",
	)
	p.add_argument(
		"--permutation-resamples",
		type=int,
		default=200,
		help="Stage2 permutation_p resamples.",
	)
	p.add_argument(
		"--permutation-seed",
		type=int,
		default=None,
		help="Stage2 permutation_p RNG seed (for reproducibility).",
	)
	p.add_argument("--eta", type=float, default=0.6)
	p.add_argument("--min-turn-strength", type=float, default=0.0)
	p.add_argument("--stage3-method", choices=["turning", "centroid"], default="turning")
	p.add_argument("--phase-smoothing", type=int, default=1)
	p.add_argument("--stage3-window", type=int, default=None)
	p.add_argument("--stage3-step", type=int, default=20)
	p.add_argument("--stage3-quantile", type=float, default=0.75)

	# rho computation
	p.add_argument("--fd-h", type=float, default=1e-6)
	p.add_argument("--summary-tol", type=float, default=1e-6)

	p.add_argument("--out", type=Path, required=True)
	p.add_argument(
		"--resume",
		action="store_true",
		help="If set and --out exists, append and skip points already present in the CSV.",
	)
	args = p.parse_args()

	seed_list = _parse_seeds(str(args.seeds))
	k_vals = _parse_float_grid(str(args.k_grid))
	players_vals = [int(args.players)] if args.players_grid is None else _parse_int_grid(str(args.players_grid))
	if not players_vals:
		raise ValueError("players grid resolved to empty")

	jobs = int(args.jobs)
	if jobs == 0:
		jobs = _auto_jobs()
	if jobs < 1:
		raise ValueError("--jobs must be >= 0")

	rounds = int(args.rounds)
	burn_in = args.burn_in
	if burn_in is None:
		burn_in = int(round(float(rounds) * float(args.burn_in_frac)))
	burn_in = max(0, int(burn_in))

	metric_kwargs = {
		"amplitude_threshold": float(args.amplitude_threshold),
		"amplitude_normalization": str(args.amplitude_normalization),
		"amplitude_control_reference": None,
		"amplitude_threshold_factor": float(args.amplitude_threshold_factor),
		"min_lag": int(args.min_lag),
		"max_lag": int(args.max_lag),
		"corr_threshold": float(args.corr_threshold),
		"stage2_method": str(args.stage2_method),
		"stage2_prefilter": bool(int(args.stage2_prefilter) == 1),
		"power_ratio_kappa": float(args.power_ratio_kappa),
		"permutation_alpha": float(args.permutation_alpha),
		"permutation_resamples": int(args.permutation_resamples),
		"permutation_seed": (int(args.permutation_seed) if args.permutation_seed is not None else None),
		"eta": float(args.eta),
		"min_turn_strength": float(args.min_turn_strength),
		"normalize_for_phase": bool(args.series == "w"),
		"stage3_method": str(args.stage3_method),
		"phase_smoothing": int(args.phase_smoothing),
		"stage3_window": (int(args.stage3_window) if args.stage3_window is not None else None),
		"stage3_step": int(args.stage3_step),
		"stage3_quantile": float(args.stage3_quantile),
	}

	args.out.parent.mkdir(parents=True, exist_ok=True)

	amp_norm = str(args.amplitude_normalization)
	amp_cache: dict[int, dict[str, float]] | None = None
	if amp_norm != "none" and args.amplitude_control_reference is None:
		if args.amplitude_control_cache is None:
			raise ValueError(
				"--amplitude-normalization requires --amplitude-control-reference or --amplitude-control-cache"
			)
		amp_cache = _load_amplitude_control_cache(
			Path(args.amplitude_control_cache),
			expected_series=str(args.series),
			expected_burn_in=int(burn_in),
			expected_tail=(int(args.tail) if args.tail is not None else None),
		)

	fieldnames = list(SWEEP_CSV_FIELDNAMES)

	completed: set[tuple[int, float]] = set()
	if bool(args.resume) and args.out.exists():
		try:
			completed = _load_completed_points(args.out)
			print(f"resume: found {len(completed)} completed points in {args.out}", flush=True)
		except Exception as e:
			print(f"resume: failed to read existing CSV ({e}); continuing without resume", flush=True)
			completed = set()

	print("=== rho curve sweep ===", flush=True)
	print(
		f"payoff_mode={args.payoff_mode} a={args.a} b={args.b} gamma={args.gamma} epsilon={args.epsilon} "
		f"popularity_mode={args.popularity_mode} evolution_mode={args.evolution_mode} payoff_lag={args.payoff_lag} init_bias={args.init_bias}",
		flush=True,
	)
	print(f"players_grid={players_vals} rounds={rounds} seeds_n={len(seed_list)}", flush=True)
	print(f"k_points={len(k_vals)} series={args.series} burn_in={burn_in} tail={args.tail}", flush=True)
	print(
		f"thresholds: amp={args.amplitude_threshold} amp_norm={amp_norm} amp_factor={args.amplitude_threshold_factor} "
		f"corr={args.corr_threshold} stage2_method={args.stage2_method} eta={args.eta} min_turn_strength={args.min_turn_strength}",
		flush=True,
	)
	print(f"jobs={jobs} (max_workers per point is min(jobs, n_seeds))", flush=True)

	write_header = True
	file_mode = "w"
	if bool(args.resume) and args.out.exists():
		write_header = False
		file_mode = "a"

	with args.out.open(file_mode, newline="") as f:
		w = csv.DictWriter(f, fieldnames=fieldnames)
		if write_header:
			w.writeheader()
			f.flush()

		for npl in players_vals:
			amp_ref = _resolve_amplitude_control_reference(
				players=int(npl),
				normalization=amp_norm,
				override_reference=(float(args.amplitude_control_reference) if args.amplitude_control_reference is not None else None),
				cache=amp_cache,
			)
			if amp_norm != "none" and (amp_ref is None):
				raise ValueError(
					f"No control reference found for N={int(npl)} (normalization={amp_norm}); "
					"use --amplitude-control-reference or provide a cache with matching by_players entry"
				)
			metric_kwargs_n = dict(metric_kwargs)
			metric_kwargs_n["amplitude_control_reference"] = amp_ref
			for k in k_vals:
				key = (int(npl), round(float(k), 10))
				if completed and key in completed:
					continue

				print(f"BEGIN N={int(npl)} k={float(k):.4g}", flush=True)

				# Local linearization (lagged rho around uniform equilibrium)
				rep = lagged_jacobian_at_uniform(
					a=float(args.a),
					b=float(args.b),
					selection_strength=float(k),
					h=float(args.fd_h),
				)
				sum_l = summarize_lagged(rep, tol=float(args.summary_tol))
				rho = float(sum_l.rho)

				# Multi-seed simulation estimate
				per_seed: list[OneSeedResult] = []
				if jobs == 1:
					for sd in seed_list:
						per_seed.append(
							_run_one_seed(
								seed=int(sd),
								players=int(npl),
								rounds=int(rounds),
								payoff_mode=str(args.payoff_mode),
								popularity_mode=str(args.popularity_mode),
								evolution_mode=str(args.evolution_mode),
								payoff_lag=int(args.payoff_lag),
								gamma=float(args.gamma),
								epsilon=float(args.epsilon),
								a=float(args.a),
								b=float(args.b),
								init_bias=float(args.init_bias),
								selection_strength=float(k),
								series=str(args.series),
								burn_in=int(burn_in),
								tail=int(args.tail) if args.tail is not None else None,
								metric_kwargs=metric_kwargs_n,
							)
						)
						print(f"  seed {int(sd)} done ({len(per_seed)}/{len(seed_list)})", flush=True)
				else:
					max_workers = min(int(jobs), len(seed_list))
					with ProcessPoolExecutor(max_workers=int(max_workers)) as ex:
						futs = [
							ex.submit(
								_run_one_seed,
								seed=int(sd),
								players=int(npl),
								rounds=int(rounds),
								payoff_mode=str(args.payoff_mode),
								popularity_mode=str(args.popularity_mode),
								evolution_mode=str(args.evolution_mode),
								payoff_lag=int(args.payoff_lag),
								gamma=float(args.gamma),
								epsilon=float(args.epsilon),
								a=float(args.a),
								b=float(args.b),
								init_bias=float(args.init_bias),
								selection_strength=float(k),
								series=str(args.series),
								burn_in=int(burn_in),
								tail=int(args.tail) if args.tail is not None else None,
								metric_kwargs=metric_kwargs_n,
							)
							for sd in seed_list
						]
						for fut in as_completed(futs):
							per_seed.append(fut.result())
							print(f"  seed done ({len(per_seed)}/{len(seed_list)})", flush=True)

				den = float(len(per_seed) if per_seed else 1)
				p_level3 = sum(1 for r in per_seed if r.level == 3) / den
				p_ge_2 = sum(1 for r in per_seed if r.level >= 2) / den
				mean_s3_score = sum(float(r.stage3_score) for r in per_seed) / den
				mean_env_gamma = sum(float(r.env_gamma) for r in per_seed) / den
				mean_env_gamma_r2 = sum(float(r.env_gamma_r2) for r in per_seed) / den
				mean_env_gamma_n_peaks = sum(int(r.env_gamma_n_peaks) for r in per_seed) / den
				mean_max_amp = sum(float(r.max_amp) for r in per_seed) / den
				mean_max_corr = sum(float(r.max_corr) for r in per_seed) / den
				mean_stage2_stat = sum(float(r.stage2_statistic) for r in per_seed) / den
				mean_stage2_win = sum(int(r.stage2_effective_window_n) for r in per_seed) / den
				stage2_stat_name = ""
				for r in per_seed:
					if str(r.stage2_statistic_name):
						stage2_stat_name = str(r.stage2_statistic_name)
						break

				amp_thr_eff = float(args.amplitude_threshold)
				mean_max_amp_norm: float | None = None
				if amp_norm != "none" and amp_ref is not None:
					amp_thr_eff = float(args.amplitude_threshold_factor) * float(amp_ref)
					if float(amp_ref) > 0.0:
						mean_max_amp_norm = float(mean_max_amp) / float(amp_ref)

				row = {
					"players": int(npl),
					"a": float(args.a),
					"b": float(args.b),
					"selection_strength": float(k),
					"rho": float(rho),
					"rho_minus_1": float(rho - 1.0),
					"p_level_ge_2": float(p_ge_2),
					"p_level_3": float(p_level3),
					"mean_stage3_score": float(mean_s3_score),
					"mean_max_amp": float(mean_max_amp),
					"mean_max_corr": float(mean_max_corr),
					"mean_stage2_statistic": float(mean_stage2_stat),
					"stage2_statistic_name": str(stage2_stat_name),
					"mean_stage2_effective_window_n": float(mean_stage2_win),
					"mean_env_gamma": float(mean_env_gamma),
					"mean_env_gamma_r2": float(mean_env_gamma_r2),
					"mean_env_gamma_n_peaks": float(mean_env_gamma_n_peaks),
					"n_seeds": int(len(per_seed)),
					"series": str(args.series),
					"rounds": int(rounds),
					"burn_in": int(burn_in),
					"tail": int(args.tail) if args.tail is not None else None,
					"payoff_mode": str(args.payoff_mode),
					"gamma": float(args.gamma),
					"epsilon": float(args.epsilon),
					"seeds": str(args.seeds),
					"k_grid": str(args.k_grid),
					"players_grid": (str(args.players_grid) if args.players_grid is not None else ""),
					"amplitude_threshold": float(args.amplitude_threshold),
					"amplitude_normalization": str(amp_norm),
					"amplitude_threshold_factor": float(args.amplitude_threshold_factor),
					"amplitude_control_reference": (float(amp_ref) if amp_ref is not None else ""),
					"amplitude_threshold_effective": float(amp_thr_eff),
					"mean_max_amp_norm": (float(mean_max_amp_norm) if mean_max_amp_norm is not None else ""),
					"corr_threshold": float(args.corr_threshold),
					"stage2_method": str(args.stage2_method),
					"stage2_prefilter": int(args.stage2_prefilter),
					"power_ratio_kappa": float(args.power_ratio_kappa),
					"permutation_alpha": float(args.permutation_alpha),
					"permutation_resamples": int(args.permutation_resamples),
					"permutation_seed": (int(args.permutation_seed) if args.permutation_seed is not None else ""),
					"eta": float(args.eta),
					"min_turn_strength": float(args.min_turn_strength),
					"stage3_method": str(args.stage3_method),
					"phase_smoothing": int(args.phase_smoothing),
					"stage3_window": (int(args.stage3_window) if args.stage3_window is not None else ""),
					"stage3_step": int(args.stage3_step),
					"stage3_quantile": float(args.stage3_quantile),
					"popularity_mode": str(args.popularity_mode),
					"evolution_mode": str(args.evolution_mode),
					"payoff_lag": int(args.payoff_lag),
					"init_bias": float(args.init_bias),
					"amplitude_control_cache": (str(args.amplitude_control_cache) if args.amplitude_control_cache is not None else ""),
				}
				w.writerow(row)
				f.flush()

				print(
					f"N={int(npl)} k={float(k):.4g} rho-1={rho-1:+.4g}  P(L3)={p_level3:.3f}  P(L>=2)={p_ge_2:.3f}  mean_s3={mean_s3_score:.3f}",
					flush=True,
				)

	print(f"Wrote CSV: {args.out}", flush=True)


if __name__ == "__main__":
	main()
