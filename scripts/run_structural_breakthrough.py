#!/usr/bin/env python3
"""Structural breakthrough scan for personality_coupled evolution.

This H7.2 diagnostic keeps the sampled + events baseline fixed while relaxing
the personality_coupling k clamp and lowering mu_base to probe an ignition point.
It compares control_none vs random_9persona across 12 seeds and a small
selection_strength grid.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.cycle_metrics import classify_cycle_level, phase_rotation_r2
from players.base_player import DEFAULT_PERSONALITY_KEYS
from simulation.run_simulation import SimConfig, _write_timeseries_csv, simulate

DEFAULT_SEEDS = [45, 47, 49, 51, 53, 55, 91, 93, 95, 97, 99, 101]


@dataclass
class RunMetric:
	selection_strength: float
	condition: str
	seed: int
	csv_path: str
	cycle_level: int
	phase_velocity: float
	l3_pass: bool
	velocity_pass: bool
	both_pass: bool


def _parse_int_list(text: str) -> list[int]:
	return [int(part.strip()) for part in text.split(",") if part.strip()]


def _extract_series(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
	return {
		"aggressive": [float(row.get("p_aggressive", 0.0)) for row in rows],
		"defensive": [float(row.get("p_defensive", 0.0)) for row in rows],
		"balanced": [float(row.get("p_balanced", 0.0)) for row in rows],
	}


def _random_9persona_setup(seed: int):
	def cb(players: list[object], _strategy_space: list[str], _cfg: SimConfig) -> None:
		for idx, player in enumerate(players):
			rng = random.Random(int(seed) * 10000 + idx)
			for key in DEFAULT_PERSONALITY_KEYS:
				player.personality[key] = rng.uniform(-0.4, 0.4)

	return cb


def _run_one(
	*,
	selection_strength: float,
	condition: str,
	seed: int,
	rounds: int,
	players: int,
	mu_base: float,
	lambda_mu: float,
	lambda_k: float,
	mu_lower: float,
	mu_upper: float,
	k_lower: float,
	k_upper: float,
	out_dir: Path,
	velocity_threshold: float,
) -> RunMetric:
	tag = f"ss{selection_strength:.3f}"
	out_csv = out_dir / tag / condition / f"seed{seed}.csv"
	out_csv.parent.mkdir(parents=True, exist_ok=True)

	cfg = SimConfig(
		n_players=int(players),
		n_rounds=int(rounds),
		seed=int(seed),
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.16,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.20,
		init_bias=0.5,
		evolution_mode="personality_coupled",
		payoff_lag=1,
		selection_strength=float(selection_strength),
		enable_events=False,
		events_json=None,
		out_csv=out_csv,
		memory_kernel=1,
		synergy_type="nonlinear",
		synergy_gamma=0.16,
		synergy_nonlinear_type="power",
		synergy_nonlinear_power=3.2,
		personality_coupling_mu_base=float(mu_base),
		personality_coupling_lambda_mu=float(lambda_mu),
		personality_coupling_lambda_k=float(lambda_k),
		personality_coupling_mu_lower=float(mu_lower),
		personality_coupling_mu_upper=float(mu_upper),
		personality_coupling_k_lower=float(k_lower),
		personality_coupling_k_upper=float(k_upper),
	)

	player_setup = _random_9persona_setup(seed) if condition == "random_9persona" else None
	strategy_space, rows = simulate(cfg, player_setup_callback=player_setup)
	_write_timeseries_csv(out_csv, strategy_space=strategy_space, rows=rows)

	series = _extract_series(rows)
	cyc = classify_cycle_level(series, burn_in=2000, tail=2000, amplitude_threshold=0.02, eta=0.6, min_turn_strength=0.0)
	rot = phase_rotation_r2(series, burn_in=2000, tail=2000)
	window_length = rot.window_length if rot.window_length > 0 else 1
	phase_velocity = float(rot.cumulative_rotation) / float(window_length)

	l3_pass = int(cyc.level) >= 3
	velocity_pass = phase_velocity > float(velocity_threshold)
	return RunMetric(
		selection_strength=float(selection_strength),
		condition=condition,
		seed=int(seed),
		csv_path=str(out_csv.relative_to(ROOT)),
		cycle_level=int(cyc.level),
		phase_velocity=float(phase_velocity),
		l3_pass=bool(l3_pass),
		velocity_pass=bool(velocity_pass),
		both_pass=bool(l3_pass and velocity_pass),
	)


def _agg(metrics: list[RunMetric]) -> dict[str, float | int]:
	n = len(metrics)
	if n == 0:
		return {"n": 0, "l3_rate": 0.0, "velocity_pass_rate": 0.0, "both_pass_rate": 0.0, "mean_phase_velocity": 0.0}
	return {
		"n": n,
		"l3_rate": sum(1 for metric in metrics if metric.l3_pass) / n,
		"velocity_pass_rate": sum(1 for metric in metrics if metric.velocity_pass) / n,
		"both_pass_rate": sum(1 for metric in metrics if metric.both_pass) / n,
		"mean_phase_velocity": sum(metric.phase_velocity for metric in metrics) / n,
	}


def main() -> int:
	parser = argparse.ArgumentParser(description="Structural breakthrough scan (personality_coupled)")
	parser.add_argument("--rounds", type=int, default=10000)
	parser.add_argument("--players", type=int, default=300)
	parser.add_argument("--velocity-threshold", type=float, default=0.001)
	parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
	parser.add_argument("--selection-strengths", type=str, default="0.10,0.15,0.20")
	parser.add_argument("--mu-base", type=float, default=0.05)
	parser.add_argument("--lambda-mu", type=float, default=0.10)
	parser.add_argument("--lambda-k", type=float, default=0.20)
	parser.add_argument("--mu-lower", type=float, default=0.0)
	parser.add_argument("--mu-upper", type=float, default=0.60)
	parser.add_argument("--k-lower", type=float, default=0.05)
	parser.add_argument("--k-upper", type=float, default=0.25)
	args = parser.parse_args()

	seeds = _parse_int_list(args.seeds)
	selection_strengths = [float(value) for value in args.selection_strengths.split(",") if value.strip()]
	out_dir = ROOT / "outputs" / "structural_breakthrough_personality_coupled"
	out_dir.mkdir(parents=True, exist_ok=True)

	conditions = ["control_none", "random_9persona"]
	all_results: list[RunMetric] = []

	total = len(selection_strengths) * len(conditions) * len(seeds)
	index = 0
	print("Structural breakthrough scan (personality_coupled)")
	print(f"rounds={args.rounds}, players={args.players}, mu_base={args.mu_base}, lambda_mu={args.lambda_mu}, lambda_k={args.lambda_k}")
	print(f"k_clamp=[{args.k_lower}, {args.k_upper}], seeds={seeds}")
	print(f"selection_strengths={selection_strengths}")
	print(f"runs={total}")

	for selection_strength in selection_strengths:
		combo_metrics: list[RunMetric] = []
		for condition in conditions:
			for seed in seeds:
				index += 1
				print(f"[{index}/{total}] ss={selection_strength} {condition} seed={seed}", end=" ", flush=True)
				metric = _run_one(
					selection_strength=selection_strength,
					condition=condition,
					seed=seed,
					rounds=int(args.rounds),
					players=int(args.players),
					mu_base=float(args.mu_base),
					lambda_mu=float(args.lambda_mu),
					lambda_k=float(args.lambda_k),
					mu_lower=float(args.mu_lower),
					mu_upper=float(args.mu_upper),
					k_lower=float(args.k_lower),
					k_upper=float(args.k_upper),
					out_dir=out_dir,
					velocity_threshold=float(args.velocity_threshold),
				)
				combo_metrics.append(metric)
				all_results.append(metric)
				print(f"L{metric.cycle_level}, v={metric.phase_velocity:.6f}, both={'Y' if metric.both_pass else 'N'}")

		by_condition = {condition: [metric for metric in combo_metrics if metric.condition == condition] for condition in conditions}
		aggregate = {condition: _agg(values) for condition, values in by_condition.items()}
		combo_json = out_dir / f"ss{selection_strength:.3f}_summary.json"
		combo_json.write_text(
			json.dumps(
				{
					"selection_strength": float(selection_strength),
					"mu_base": float(args.mu_base),
					"lambda_mu": float(args.lambda_mu),
					"lambda_k": float(args.lambda_k),
					"mu_clamp": [float(args.mu_lower), float(args.mu_upper)],
					"k_clamp": [float(args.k_lower), float(args.k_upper)],
					"aggregate": aggregate,
				},
				indent=2,
			)
		)

	full = {
		"experiment": "Structural breakthrough scan (personality_coupled)",
		"seeds": seeds,
		"selection_strengths": selection_strengths,
		"rounds": int(args.rounds),
		"players": int(args.players),
		"mu_base": float(args.mu_base),
		"lambda_mu": float(args.lambda_mu),
		"lambda_k": float(args.lambda_k),
		"mu_clamp": [float(args.mu_lower), float(args.mu_upper)],
		"k_clamp": [float(args.k_lower), float(args.k_upper)],
		"results": [asdict(metric) for metric in all_results],
	}
	(out_dir / "summary.json").write_text(json.dumps(full, indent=2))
	print(f"Saved summary to {out_dir / 'summary.json'}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())