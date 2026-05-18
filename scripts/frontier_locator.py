#!/usr/bin/env python3
from __future__ import annotations

"""frontier_locator.py — candidate orchestration for frontier positioning.

This script wires the research pipeline together:
1. Load Pareto candidates from the NSGA summary.
2. Select a diverse seed set with non-dominated sorting + crowding distance.
3. Run SYM vs ASY scans over the randomness axis.
4. Extract frontier metrics via analysis.metrics.
5. Export a JSON report with labels such as [The Survivor] and [The Pulsar].

The scan axis is treated as an ordered history for the drift/resonance metrics.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from analysis.metrics import (
	calculate_resonance_score,
	calculate_slr,
	calculate_peak_relative_width,
	calculate_z_drift,
)
from evolution.personality_mapping import (
	WEIGHTS_EXPLORING_ASYMMETRIC,
	WEIGHTS_SYMMETRIC,
	compute_z_signals,
)
from evolution.replicator_dynamics import personality_guided_replicator_step


DEFAULT_SUMMARY_PATHS = [
	REPO_ROOT / "outputs" / "nsga2_9d_v2_summary.json",
	Path("/home/user/Sync/find_lv3/scripts/sweep_v2_200t/nsga2_9d_v2_summary.json"),
	Path("/home/user/Sync/find_lv3/scripts/nsga2_9d_v2_summary.json"),
]
DEFAULT_OUT_PATH = REPO_ROOT / "outputs" / "personality_frontier_v1.json"
DEFAULT_TRAIT_PRECISION = 6
DEFAULT_T_STEPS = 500
DEFAULT_TAIL_START = 250
DEFAULT_SCAN_MIN = -1.0
DEFAULT_SCAN_MAX = 0.0
DEFAULT_SCAN_STEP = 0.01
DEFAULT_PEAK_FRACTION = 0.95
DEFAULT_GAP_TOLERANCE = 1
DEFAULT_RESONANCE_MAX_LAG = 20
DEFAULT_FITNESS_K = math.e
EPS = 1e-12


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Frontier positioning orchestration CLI")
	parser.add_argument("--summary-json", type=Path, default=None, help="Path to nsga2_9d_v2_summary.json")
	parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_PATH, help="Output JSON path")
	parser.add_argument("--select-limit", type=int, default=50, help="Number of diverse seeds to keep")
	parser.add_argument("--scan-min", type=float, default=DEFAULT_SCAN_MIN, help="Scan axis lower bound")
	parser.add_argument("--scan-max", type=float, default=DEFAULT_SCAN_MAX, help="Scan axis upper bound")
	parser.add_argument("--scan-step", type=float, default=DEFAULT_SCAN_STEP, help="Scan axis step")
	parser.add_argument("--t-steps", type=int, default=DEFAULT_T_STEPS, help="Replicator steps per scan point")
	parser.add_argument("--tail-start", type=int, default=DEFAULT_TAIL_START, help="Tail start for RE computation")
	parser.add_argument("--peak-fraction", type=float, default=DEFAULT_PEAK_FRACTION, help="Relative width threshold as a fraction of re_peak")
	parser.add_argument("--gap-tolerance", type=int, default=DEFAULT_GAP_TOLERANCE, help="Allowed hole length inside sweet spot")
	parser.add_argument("--resonance-max-lag", type=int, default=DEFAULT_RESONANCE_MAX_LAG, help="Max lag for resonance correlation")
	parser.add_argument("--fitness-k", type=float, default=DEFAULT_FITNESS_K, help="Fitness correction factor k; fitness uses np.log(k)")
	parser.add_argument("--trait-precision", type=int, default=DEFAULT_TRAIT_PRECISION, help="Rounding precision for deduping trait vectors")
	parser.add_argument("--verbose", action="store_true", help="Print a compact progress table")
	return parser.parse_args()


def _resolve_summary_path(candidate: Path | None) -> Path:
	paths: list[Path] = []
	if candidate is not None:
		paths.append(candidate)
	paths.extend(DEFAULT_SUMMARY_PATHS)
	seen: set[str] = set()
	for path in paths:
		key = str(path)
		if key in seen:
			continue
		seen.add(key)
		if path.exists():
			return path
	raise FileNotFoundError("Could not find nsga2_9d_v2_summary.json in any known location")


def _load_summary(summary_path: Path) -> dict:
	data = json.loads(summary_path.read_text(encoding="utf-8"))
	candidates = data.get("pareto_front") or data.get("pareto_frontier") or data.get("frontier")
	if not isinstance(candidates, list) or not candidates:
		raise ValueError(f"summary file has no pareto_front candidates: {summary_path}")
	return data


def _trait_keys(summary: dict) -> list[str]:
	meta_keys = summary.get("meta", {}).get("trait_keys")
	if isinstance(meta_keys, list) and meta_keys:
		return [str(k) for k in meta_keys]
	return [
		"impulsiveness",
		"assertiveness",
		"optimism",
		"risk_aversion",
		"suspicion",
		"endurance",
		"randomness",
		"stability_seeking",
		"curiosity",
	]


def _summary_objectives(candidate: dict) -> tuple[float, float, float]:
	score = candidate.get("score", {})
	return (
		float(score.get("center_drift", 0.0)),
		-float(score.get("rotational_energy", 0.0)),
		-float(score.get("min_x", 0.0)),
	)


def _candidate_signature(candidate: dict, trait_keys: Sequence[str], precision: int) -> tuple[float, ...]:
	traits = candidate.get("traits_vector", {})
	return tuple(round(float(traits.get(key, 0.0)), precision) for key in trait_keys)


def dedupe_candidates(candidates: Sequence[dict], *, trait_keys: Sequence[str], precision: int = DEFAULT_TRAIT_PRECISION) -> list[dict]:
	"""Collapse identical trait vectors and keep the strongest objective tuple."""
	best_by_sig: dict[tuple[float, ...], dict] = {}
	for candidate in candidates:
		sig = _candidate_signature(candidate, trait_keys, precision)
		existing = best_by_sig.get(sig)
		if existing is None or _summary_objectives(candidate) < _summary_objectives(existing):
			best_by_sig[sig] = candidate
	return list(best_by_sig.values())


def dominates(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
	return all(a <= b for a, b in zip(left, right)) and any(a < b for a, b in zip(left, right))


def non_dominated_sort(items: Sequence[dict]) -> list[list[int]]:
	objectives = [tuple(item["objectives"]) for item in items]
	s_sets: list[list[int]] = [[] for _ in items]
	dominated_count = [0 for _ in items]
	fronts: list[list[int]] = [[]]

	for p_idx in range(len(items)):
		for q_idx in range(len(items)):
			if p_idx == q_idx:
				continue
			if dominates(objectives[p_idx], objectives[q_idx]):
				s_sets[p_idx].append(q_idx)
			elif dominates(objectives[q_idx], objectives[p_idx]):
				dominated_count[p_idx] += 1
		if dominated_count[p_idx] == 0:
			fronts[0].append(p_idx)

	i = 0
	while i < len(fronts) and fronts[i]:
		next_front: list[int] = []
		for p_idx in fronts[i]:
			for q_idx in s_sets[p_idx]:
				dominated_count[q_idx] -= 1
				if dominated_count[q_idx] == 0:
					next_front.append(q_idx)
		if next_front:
			fronts.append(next_front)
		i += 1

	return fronts


def crowding_distance(front_indices: Sequence[int], items: Sequence[dict]) -> dict[int, float]:
	if not front_indices:
		return {}
	distances = {idx: 0.0 for idx in front_indices}
	objective_count = len(items[front_indices[0]]["objectives"])

	for obj_idx in range(objective_count):
		sorted_idx = sorted(front_indices, key=lambda idx: items[idx]["objectives"][obj_idx])
		lo = items[sorted_idx[0]]["objectives"][obj_idx]
		hi = items[sorted_idx[-1]]["objectives"][obj_idx]
		distances[sorted_idx[0]] = math.inf
		distances[sorted_idx[-1]] = math.inf
		if hi == lo:
			continue
		for pos in range(1, len(sorted_idx) - 1):
			prev_v = items[sorted_idx[pos - 1]]["objectives"][obj_idx]
			next_v = items[sorted_idx[pos + 1]]["objectives"][obj_idx]
			distances[sorted_idx[pos]] += (next_v - prev_v) / (hi - lo)

	return distances


def select_diverse_candidates(candidates: Sequence[dict], *, limit: int) -> list[dict]:
	items = [{**candidate, "objectives": _summary_objectives(candidate)} for candidate in candidates]
	fronts = non_dominated_sort(items)
	selected: list[dict] = []

	for rank, front in enumerate(fronts):
		if not front:
			continue
		distance_map = crowding_distance(front, items)
		if len(selected) + len(front) <= limit:
			for idx in front:
				selected.append({**items[idx], "pareto_rank": rank, "crowding_distance": float(distance_map.get(idx, 0.0))})
			continue

		ranked = sorted(
			front,
			key=lambda idx: (
				distance_map.get(idx, 0.0),
				-float(items[idx].get("score", {}).get("rotational_energy", 0.0)),
			),
			reverse=True,
		)
		remain = max(0, limit - len(selected))
		for idx in ranked[:remain]:
			selected.append({**items[idx], "pareto_rank": rank, "crowding_distance": float(distance_map.get(idx, 0.0))})
		break

	return selected


def _axis_values(scan_min: float, scan_max: float, scan_step: float) -> np.ndarray:
	if scan_step <= 0:
		raise ValueError("scan_step must be > 0")
	if scan_max < scan_min:
		scan_min, scan_max = scan_max, scan_min
	span = float(scan_max - scan_min)
	if span <= 0.0:
		return np.array([float(scan_min)], dtype=float)
	count = int(round(span / float(scan_step))) + 1
	count = max(2, count)
	return np.linspace(float(scan_min), float(scan_max), count, dtype=float)


def _simulate_re_for_point(
	z_expanding: float,
	z_contracting: float,
	z_exploring: float,
	*,
	t_steps: int,
	tail_start: int,
) -> float:
	simplex: dict[str, float] = {
		"expanding": 1.0 / 3.0,
		"contracting": 1.0 / 3.0,
		"exploring": 1.0 / 3.0,
	}
	expanding_traj = np.empty(int(t_steps), dtype=float)
	for step in range(int(t_steps)):
		simplex, _ = personality_guided_replicator_step(
			simplex,
			z_expanding=float(z_expanding),
			z_contracting=float(z_contracting),
			z_exploring=float(z_exploring),
			win=1.0,
			loss=1.2,
			mu=0.05,
			dt=1.0,
		)
		expanding_traj[step] = float(simplex["expanding"])
	tail = expanding_traj[int(tail_start):]
	rfft_vals = np.fft.rfft(tail)
	return float(np.sum(np.square(np.abs(rfft_vals[1:]))))


def _max_abs_slope(axis_values: np.ndarray, values: np.ndarray) -> float:
	if axis_values.size < 2 or values.size < 2:
		return 0.0
	grad = np.gradient(values.astype(float), axis_values.astype(float))
	return float(np.max(np.abs(grad))) if grad.size else 0.0


def scan_candidate(
	candidate: dict,
	*,
	weights: dict,
	axis_values: np.ndarray,
	t_steps: int,
	tail_start: int,
	peak_fraction: float,
	gap_tolerance: int,
	resonance_max_lag: int,
) -> dict:
	traits_base = candidate["traits_vector"]
	re_values: list[float] = []
	z_history: list[list[float]] = []
	for axis_value in axis_values:
		traits = {**traits_base, "randomness": float(axis_value)}
		z_exp, z_con, z_expl = compute_z_signals(traits, weights)
		re_values.append(
			_simulate_re_for_point(
				z_exp,
				z_con,
				z_expl,
				t_steps=t_steps,
				tail_start=tail_start,
			)
		)
		z_history.append([float(z_exp), float(z_con), float(z_expl)])

	re_array = np.asarray(re_values, dtype=float)
	z_array = np.asarray(z_history, dtype=float)
	width_points = int(calculate_peak_relative_width(re_array, peak_fraction=peak_fraction, gap_tolerance=gap_tolerance))
	width_axis = float(width_points) * float(abs(axis_values[1] - axis_values[0])) if axis_values.size > 1 else 0.0
	drift_median, drift_p95 = calculate_z_drift(z_array)
	resonance_scores = calculate_resonance_score(re_array, z_array, max_lag=resonance_max_lag)
	resonance_mean = float(np.mean(resonance_scores))
	resonance_max = float(np.max(resonance_scores))
	peak_idx = int(np.argmax(re_array))
	peak_re = float(re_array[peak_idx])
	peak_axis = float(axis_values[peak_idx])
	max_slope = _max_abs_slope(axis_values, re_array)

	return {
		"axis_values": axis_values.tolist(),
		"re_tail": re_array.tolist(),
		"z_history": z_array.tolist(),
		"metrics": {
			"sweetspot_width_points": width_points,
			"sweetspot_width_axis": width_axis,
			"drift_median": float(drift_median),
			"drift_p95": float(drift_p95),
			"resonance_scores": [float(v) for v in resonance_scores],
			"resonance_mean": resonance_mean,
			"resonance_max": resonance_max,
			"peak_re": peak_re,
			"peak_axis": peak_axis,
			"max_abs_slope": max_slope,
		},
	}


def calculate_fitness(metrics: dict, *, fitness_k: float, peak_reference: float, scan_span: float) -> float:
	if fitness_k <= 1.0:
		raise ValueError("fitness_k must be > 1.0 so np.log(k) is positive")
	log_factor = float(np.log(float(fitness_k)))
	width_score = float(metrics["sweetspot_width_axis"]) / float(scan_span) if scan_span > 0 else 0.0
	drift_score = 1.0 / (1.0 + float(metrics["drift_p95"]))
	resonance_score = float(metrics["resonance_mean"])
	slr_score = 1.0 / (1.0 + float(metrics["slr"]))
	peak_score = float(metrics["peak_re"]) / float(peak_reference) if peak_reference > 0 else 0.0
	raw_score = float(np.mean([width_score, drift_score, resonance_score, slr_score, peak_score]))
	return float(log_factor * raw_score)


def assign_labels(metrics: dict, thresholds: dict) -> list[str]:
	labels: list[str] = []
	survivor = (
		float(metrics["sweetspot_width_axis"]) >= float(thresholds["width_p90"])
		and float(metrics["drift_p95"]) <= float(thresholds["drift_p50"])
	)
	pulsar = (
		float(metrics["peak_re"]) >= float(thresholds["peak_p90"])
		and float(metrics["resonance_mean"]) >= float(thresholds["resonance_p50"])
	)
	if survivor:
		labels.append("[The Survivor]")
	if pulsar:
		labels.append("[The Pulsar]")
	if survivor and pulsar:
		labels.append("[Alpha Seed]")
	return labels


def _annotate_final_candidates(candidates: list[dict], *, fitness_k: float, scan_span: float) -> dict:
	asy_widths = np.asarray([cand["asy"]["metrics"]["sweetspot_width_axis"] for cand in candidates], dtype=float)
	asy_peaks = np.asarray([cand["asy"]["metrics"]["peak_re"] for cand in candidates], dtype=float)
	asy_drifts = np.asarray([cand["asy"]["metrics"]["drift_p95"] for cand in candidates], dtype=float)
	asy_slrs = np.asarray([cand["slr"] for cand in candidates], dtype=float)

	thresholds = {
		"width_p90": float(np.quantile(asy_widths, 0.90)),
		"width_p50": float(np.quantile(asy_widths, 0.50)),
		"peak_p90": float(np.quantile(asy_peaks, 0.90)),
		"resonance_p50": float(np.quantile(np.asarray([cand["asy"]["metrics"]["resonance_mean"] for cand in candidates], dtype=float), 0.50)),
		"drift_p50": float(np.quantile(asy_drifts, 0.50)),
		"slr_p50": float(np.quantile(asy_slrs, 0.50)),
	}
	peak_reference = float(np.max(asy_peaks)) if asy_peaks.size else 0.0

	for candidate in candidates:
		asy_metrics = dict(candidate["asy"]["metrics"])
		asy_metrics["slr"] = float(candidate["slr"])
		candidate["asy"]["metrics"] = asy_metrics
		sym_metrics = dict(candidate["sym"]["metrics"])
		sym_metrics["slr"] = float(candidate["slr"])
		candidate["sym"]["metrics"] = sym_metrics
		candidate["labels"] = assign_labels(candidate["asy"]["metrics"], thresholds)
		candidate["asy"]["metrics"]["narrow_band"] = float(candidate["asy"]["metrics"]["sweetspot_width_axis"]) <= float(thresholds["width_p50"])
		candidate["asy"]["metrics"]["fitness"] = calculate_fitness(
			candidate["asy"]["metrics"],
			fitness_k=fitness_k,
			peak_reference=peak_reference,
			scan_span=scan_span,
		)
		candidate["sym"]["metrics"]["fitness"] = calculate_fitness(
			candidate["sym"]["metrics"],
			fitness_k=fitness_k,
			peak_reference=float(np.max([cand["sym"]["metrics"]["peak_re"] for cand in candidates])) if candidates else 0.0,
			scan_span=scan_span,
		)

	for candidate in candidates:
		candidate["objectives"] = (
			float(candidate["asy"]["metrics"]["drift_p95"]),
			-float(candidate["asy"]["metrics"]["sweetspot_width_axis"]),
			-float(candidate["asy"]["metrics"]["resonance_mean"]),
			-float(candidate["asy"]["metrics"]["peak_re"]),
			float(candidate["slr"]),
		)

	fronts = non_dominated_sort(candidates)
	for rank, front in enumerate(fronts):
		distance_map = crowding_distance(front, candidates)
		for idx in front:
			candidates[idx]["frontier_rank"] = rank
			candidates[idx]["crowding_distance"] = float(distance_map.get(idx, 0.0))

	candidates.sort(
		key=lambda cand: (
			int(cand.get("frontier_rank", 999)),
			-float(cand["asy"]["metrics"].get("fitness", 0.0)),
			-float(cand.get("crowding_distance", 0.0)),
		),
	)

	return {
		"thresholds": thresholds,
		"peak_reference": peak_reference,
		"candidates": candidates,
	}


def build_report(args: argparse.Namespace) -> dict:
	summary_path = _resolve_summary_path(args.summary_json)
	summary = _load_summary(summary_path)
	trait_keys = _trait_keys(summary)
	raw_candidates = summary.get("pareto_front") or summary.get("pareto_frontier") or summary.get("frontier") or []
	unique_candidates = dedupe_candidates(raw_candidates, trait_keys=trait_keys, precision=args.trait_precision)
	selected_candidates = select_diverse_candidates(unique_candidates, limit=args.select_limit)
	axis_values = _axis_values(args.scan_min, args.scan_max, args.scan_step)
	scan_span = float(abs(axis_values[-1] - axis_values[0])) if axis_values.size > 1 else 0.0

	final_candidates: list[dict] = []
	for seed_item in selected_candidates:
		sym_scan = scan_candidate(
			seed_item,
			weights=WEIGHTS_SYMMETRIC,
			axis_values=axis_values,
			t_steps=args.t_steps,
			tail_start=args.tail_start,
			peak_fraction=args.peak_fraction,
			gap_tolerance=args.gap_tolerance,
			resonance_max_lag=args.resonance_max_lag,
		)
		asy_scan = scan_candidate(
			seed_item,
			weights=WEIGHTS_EXPLORING_ASYMMETRIC,
			axis_values=axis_values,
			t_steps=args.t_steps,
			tail_start=args.tail_start,
			peak_fraction=args.peak_fraction,
			gap_tolerance=args.gap_tolerance,
			resonance_max_lag=args.resonance_max_lag,
		)
		sl = calculate_slr(sym_scan["metrics"]["max_abs_slope"], asy_scan["metrics"]["max_abs_slope"])
		final_candidates.append(
			{
				"trial_id": seed_item.get("trial_id"),
				"seed": seed_item.get("seed"),
				"pareto_rank": seed_item.get("pareto_rank"),
				"crowding_distance": seed_item.get("crowding_distance"),
				"source_score": seed_item.get("score", {}),
				"traits_vector": seed_item.get("traits_vector", {}),
				"signal_snapshot": seed_item.get("signal_snapshot", {}),
				"base_randomness": float(seed_item.get("traits_vector", {}).get("randomness", 0.0)),
				"base_z_exploring": float(seed_item.get("signal_snapshot", {}).get("z_exploring", 0.0)),
				"sym": sym_scan,
				"asy": asy_scan,
				"slr": float(sl),
			}
		)

	annotated = _annotate_final_candidates(final_candidates, fitness_k=args.fitness_k, scan_span=scan_span)
	return {
		"schema_version": "v1.0",
		"source_summary_path": str(summary_path),
		"selection": {
			"method": "non_dominated_sort + crowding_distance",
			"trait_precision": args.trait_precision,
			"raw_candidates": len(raw_candidates),
			"unique_candidates": len(unique_candidates),
			"selected_candidates": len(selected_candidates),
		},
		"scan": {
			"axis": "randomness",
			"scan_min": float(args.scan_min),
			"scan_max": float(args.scan_max),
			"scan_step": float(args.scan_step),
			"axis_values": axis_values.tolist(),
			"t_steps": int(args.t_steps),
			"tail_start": int(args.tail_start),
			"peak_fraction": float(args.peak_fraction),
			"width_rule": "peak_relative_ma3",
			"gap_tolerance": int(args.gap_tolerance),
			"resonance_max_lag": int(args.resonance_max_lag),
			"fitness_k": float(args.fitness_k),
		},
		"thresholds": annotated["thresholds"],
		"frontier": annotated["candidates"],
	}


def _print_progress(report: dict) -> None:
	frontier = report["frontier"]
	print("=" * 100)
	print("  Frontier Positioning Summary")
	print("=" * 100)
	print(f"  summary: {report['source_summary_path']}")
	print(f"  selected: {report['selection']['selected_candidates']} / unique={report['selection']['unique_candidates']} / raw={report['selection']['raw_candidates']}")
	print(f"  axis: randomness in [{report['scan']['scan_min']:.2f}, {report['scan']['scan_max']:.2f}] step={report['scan']['scan_step']:.2f}")
	print(f"  thresholds: width_p90={report['thresholds']['width_p90']:.4f} peak_p90={report['thresholds']['peak_p90']:.1f} drift_p50={report['thresholds']['drift_p50']:.6f}")
	print(f"  width rule: {report['scan']['width_rule']} @ peak_fraction={report['scan']['peak_fraction']:.2f}")
	print("")
	print(f"  {'rank':>4}  {'trial':>5}  {'fitness':>8}  {'width':>7}  {'drift':>9}  {'resonance':>9}  {'slr':>7}  labels")
	print("  " + "-" * 92)
	for cand in frontier[:20]:
		metrics = cand["asy"]["metrics"]
		print(
			f"  {int(cand.get('frontier_rank', 999)):>4}  {int(cand.get('trial_id', -1)):>5}  "
			f"{float(metrics.get('fitness', 0.0)):>8.4f}  {float(metrics['sweetspot_width_axis']):>7.3f}  "
			f"{float(metrics['drift_p95']):>9.6f}  {float(metrics['resonance_mean']):>9.4f}  "
			f"{float(cand['slr']):>7.4f}  {' '.join(cand.get('labels', []))}"
		)


def main() -> None:
	args = _parse_args()
	if args.fitness_k <= 1.0:
		raise ValueError("--fitness-k must be > 1.0 so np.log(k) stays positive")
	report = build_report(args)
	args.out_json.parent.mkdir(parents=True, exist_ok=True)
	args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
	if args.verbose:
		_print_progress(report)
	print(f"Wrote frontier report to {args.out_json}")


if __name__ == "__main__":
	main()