#!/usr/bin/env python3
"""Run refined C-direction power scan.

Default grid:
- gamma: 0.11..0.16 (step 0.01)
- power: 2.3, 2.6, 2.9, 3.2
- seeds: 45, 47, 49, 51, 53, 55
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Add repo root to import path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.run_simulation import SimConfig, simulate


DEFAULT_GAMMAS = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
DEFAULT_POWERS = [2.3, 2.6, 2.9, 3.2]
DEFAULT_SEEDS = [45, 47, 49, 51, 53, 55]


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def run_one(gamma: float, power: float, seed: int, out_csv: Path, rounds: int) -> dict:
    start = time.time()
    try:
        cfg = SimConfig(
            n_players=300,
            n_rounds=rounds,
            seed=seed,
            payoff_mode="matrix_ab",
            popularity_mode="sampled",
            gamma=float(gamma),
            epsilon=0.0,
            a=1.0,
            b=0.9,
            matrix_cross_coupling=0.20,
            init_bias=0.5,
            evolution_mode="mean_field",
            payoff_lag=0,
            selection_strength=0.02,
            memory_kernel=1,
            synergy_type="nonlinear",
            synergy_gamma=float(gamma),
            synergy_nonlinear_type="power",
            synergy_nonlinear_power=float(power),
        )

        _, rows = simulate(cfg)
        if not rows:
            return {
                "gamma": gamma,
                "power": power,
                "seed": seed,
                "success": False,
                "error": "simulate() returned no rows",
                "elapsed_sec": time.time() - start,
            }

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        return {
            "gamma": gamma,
            "power": power,
            "seed": seed,
            "success": True,
            "csv_path": str(out_csv),
            "n_rows": len(rows),
            "elapsed_sec": time.time() - start,
        }
    except Exception as e:
        return {
            "gamma": gamma,
            "power": power,
            "seed": seed,
            "success": False,
            "error": str(e),
            "elapsed_sec": time.time() - start,
        }


def main() -> int:
    p = argparse.ArgumentParser(description="Run refined C-power scan")
    p.add_argument("--gammas", type=str, default=",".join(str(g) for g in DEFAULT_GAMMAS))
    p.add_argument("--powers", type=str, default=",".join(str(v) for v in DEFAULT_POWERS))
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--rounds", type=int, default=6000)
    args = p.parse_args()

    gammas = _parse_float_list(args.gammas)
    powers = _parse_float_list(args.powers)
    seeds = _parse_int_list(args.seeds)

    out_dir = Path(__file__).parent.parent / "outputs" / "c_power_refine_scan"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[float, float, int]] = []
    for gamma in gammas:
        for power in powers:
            for seed in seeds:
                runs.append((gamma, power, seed))

    print("C power refine scan")
    print(f"grid: {len(gammas)} gammas x {len(powers)} powers x {len(seeds)} seeds = {len(runs)} runs")
    print(f"output: {out_dir}")

    results = []
    ok = 0
    for idx, (gamma, power, seed) in enumerate(runs, 1):
        out_csv = out_dir / f"run_g{gamma:.2f}_p{power:.1f}_s{seed}.csv"
        print(f"[{idx:3d}/{len(runs)}] g={gamma:.2f} p={power:.1f} s={seed} ", end="", flush=True)
        r = run_one(gamma, power, seed, out_csv, rounds=int(args.rounds))
        results.append(r)
        if r.get("success"):
            ok += 1
            print(f"OK ({r.get('elapsed_sec', 0.0):.2f}s)")
        else:
            print(f"FAIL: {r.get('error', 'unknown')}")

    summary = {
        "experiment": "C power refine scan",
        "gammas": gammas,
        "powers": powers,
        "seeds": seeds,
        "rounds": int(args.rounds),
        "total_runs": len(runs),
        "successful_runs": ok,
        "failed_runs": len(runs) - ok,
        "results": results,
    }

    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"saved: {summary_path}")

    return 0 if ok == len(runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
