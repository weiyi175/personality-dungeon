#!/usr/bin/env python3
"""H7 High-Inertia Diagnostic Test

Quick validation of whether increasing mu_base (inertia) can unlock L3.

Default settings:
- mu_base = 0.30 (vs H7.2's 0.05)
- lambda_mu = 0.05 (reduced, to focus on high baseline)
- selection_strength = 0.15 (mid-range)
- k_clamp = [0.05, 0.25] (from H7.2)
- 6000 rounds (shorter than H7.2's 10000 for faster iteration)
- 12 seeds (same as H7.2)
- Both conditions: control_none, random_9persona
"""

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


def run_single_seed(
    seed: int,
    condition: str,
    output_dir: Path,
    selection_strength: float = 0.15,
    rounds: int = 6000,
    mu_base: float = 0.30,
) -> RunMetric:
    """Run one seed under one condition."""
    tag_dir = output_dir / f"ss{selection_strength:.3f}" / condition
    tag_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tag_dir / f"seed{seed}.csv"

    cfg = SimConfig(
        n_players=300,
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
        out_csv=str(out_csv),
        memory_kernel=1,
        synergy_type="nonlinear",
        synergy_gamma=0.16,
        synergy_nonlinear_type="power",
        synergy_nonlinear_power=3.2,
        personality_coupling_mu_base=float(mu_base),  # HIGH INERTIA HERE
        personality_coupling_lambda_mu=0.05,
        personality_coupling_lambda_k=0.20,
        personality_coupling_mu_lower=0.0,
        personality_coupling_mu_upper=0.60,
        personality_coupling_k_lower=0.05,
        personality_coupling_k_upper=0.25,
    )

    player_setup = _random_9persona_setup(seed) if condition == "random_9persona" else None
    try:
        strategy_space, rows = simulate(cfg, player_setup_callback=player_setup)
        _write_timeseries_csv(out_csv, strategy_space=strategy_space, rows=rows)

        series = _extract_series(rows)
        cyc = classify_cycle_level(series, burn_in=2000, tail=2000, amplitude_threshold=0.02, eta=0.6, min_turn_strength=0.0)
        rot = phase_rotation_r2(series, burn_in=2000, tail=2000)
        window_length = rot.window_length if rot.window_length > 0 else 1
        phase_velocity = float(rot.cumulative_rotation) / float(window_length)

        l3_pass = int(cyc.level) >= 3
        velocity_pass = phase_velocity > 0.001
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
    except Exception as e:
        print(f"ERROR: seed={seed}, condition={condition}: {e}")
        return RunMetric(
            selection_strength=float(selection_strength),
            condition=condition,
            seed=int(seed),
            csv_path=str(out_csv),
            cycle_level=0,
            phase_velocity=0.0,
            l3_pass=False,
            velocity_pass=False,
            both_pass=False,
        )


def main():
    """Run H7 high-inertia diagnostic."""
    import argparse
    parser = argparse.ArgumentParser(description="H7 high-inertia diagnostic test")
    parser.add_argument("--quick", action="store_true", help="Quick test with 3 seeds and 3000 rounds")
    parser.add_argument("--rounds", type=int, default=6000, help="Number of rounds per run")
    parser.add_argument("--mu-base", type=float, default=0.30, help="High inertia baseline")
    parser.add_argument("--seeds", type=str, default="45,47,49,51,53,55,91,93,95,97,99,123",
                        help="Comma-separated list of seeds")
    args = parser.parse_args()
    
    out_root = ROOT / "outputs" / "h7_high_inertia_test"
    if args.quick:
        out_root = ROOT / "outputs" / "h7_high_inertia_quick"
    out_root.mkdir(parents=True, exist_ok=True)
    
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.quick:
        seeds = seeds[:3]
    conditions = ["control_none", "random_9persona"]
    
    rounds_per_run = 3000 if args.quick else args.rounds
    
    print(f"H7 High-Inertia Diagnostic Test")
    print(f"  mu_base = {args.mu_base} (vs H7.2: 0.05)")
    print(f"  lambda_mu = 0.05")
    print(f"  selection_strength = 0.15")
    print(f"  k_clamp = [0.05, 0.25]")
    print(f"  rounds = {rounds_per_run}")
    print(f"  seeds = {len(seeds)}")
    print(f"  conditions = {len(conditions)}")
    print(f"  total runs = {len(seeds) * len(conditions)}")
    print(f"  output_dir = {out_root}")
    print()
    
    results = []
    total = len(seeds) * len(conditions)
    count = 0
    
    for seed in seeds:
        for condition in conditions:
            count += 1
            print(f"[{count:2d}/{total}] Running seed={seed:3d}, condition={condition}...", end=" ", flush=True)
            metric = run_single_seed(
                seed=seed,
                condition=condition,
                output_dir=out_root,
                selection_strength=0.15,
                rounds=rounds_per_run,
                mu_base=float(args.mu_base),
            )
            results.append(asdict(metric))
            print(f"L{metric.cycle_level}, v={metric.phase_velocity:.6f}, L3={'Y' if metric.l3_pass else 'N'}")
    
    # Summary by condition
    by_condition = {}
    for condition in conditions:
        metrics = [r for r in results if r["condition"] == condition]
        n = len(metrics)
        l3_count = sum(1 for r in metrics if r["l3_pass"])
        by_condition[condition] = {
            "n": n,
            "l3_pass_count": l3_count,
            "l3_rate": float(l3_count) / float(n) if n > 0 else 0.0,
            "mean_phase_velocity": sum(float(r["phase_velocity"]) for r in metrics) / float(n) if n > 0 else 0.0,
        }
    
    print()
    print(f"Summary by condition:")
    for condition, stats in by_condition.items():
        print(f"  {condition}: {stats['l3_pass_count']}/{stats['n']} L3 ({100.0*stats['l3_rate']:.1f}%), mean_v={stats['mean_phase_velocity']:.6f}")
    
    # Write summary
    summary_path = out_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "experiment": "H7 High-Inertia Diagnostic",
            "config": {
                "mu_base": float(args.mu_base),
                "lambda_mu": 0.05,
                "lambda_k": 0.20,
                "selection_strength": 0.15,
                "k_clamp": [0.05, 0.25],
                "rounds": rounds_per_run,
                "seeds": seeds,
                "conditions": conditions,
            },
            "results": results,
            "summary": by_condition,
        }, f, indent=2)
    
    print(f"Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
