#!/usr/bin/env python3
"""
W4-Action A: Five-point Synergy Scan (with γ=0.15 encryption point)

Executes 36 runs total:
- 1 control run (γ=0.0)
- 5 scan runs (γ ∈ {0.05, 0.1, 0.15, 0.2, 0.4})
- 6 seeds per gamma (seeds ∈ {45, 47, 49, 51, 53, 55})

All runs use:
- players: 300
- rounds: 6000
- payoff_mode: matrix_ab
- a: 1.0, b: 0.9
- matrix_cross_coupling: 0.20
- selection_strength: 0.02
- init_bias: 0.0
- memory_kernel: 1
- payoff_lag: 1
- evolution_mode: mean_field
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
import time

# Constants from SDD Section 二 (Protocol lock)
PLAYERS = 300
ROUNDS = 6000
PAYOFF_MODE = "matrix_ab"
A_PARAM = 1.0
B_PARAM = 0.9
CROSS_COUPLING = 0.20
SELECTION_STRENGTH = 0.02
INIT_BIAS = 0.0
MEMORY_KERNEL = 1
PAYOFF_LAG = 1
EVOLUTION_MODE = "mean_field"

# Five-point gamma grid: control + scan
GAMMA_CONTROL = 0.0
GAMMA_SCAN = [0.05, 0.1, 0.15, 0.2, 0.4]
SEEDS = [45, 47, 49, 51, 53, 55]

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "w4_synergy_smoke"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_single_simulation(
    gamma: float,
    seed: int,
    output_csv: Path,
) -> dict[str, Any]:
    """Execute a single simulation run with given gamma and seed."""
    cmd = [
        "./venv/bin/python",
        "-m",
        "simulation.run_simulation",
        "--payoff-mode", PAYOFF_MODE,
        "--a", str(A_PARAM),
        "--b", str(B_PARAM),
        "--matrix-cross-coupling", str(CROSS_COUPLING),
        "--selection-strength", str(SELECTION_STRENGTH),
        "--init-bias", str(INIT_BIAS),
        "--players", str(PLAYERS),
        "--rounds", str(ROUNDS),
        "--seed", str(seed),
        "--memory-kernel", str(MEMORY_KERNEL),
        "--payoff-lag", str(PAYOFF_LAG),
        "--evolution-mode", EVOLUTION_MODE,
        "--synergy-gamma", str(gamma),
        "--out", str(output_csv),
    ]

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent.parent),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes max per run
        )
        elapsed = time.time() - start_time

        if result.returncode != 0:
            return {
                "gamma": gamma,
                "seed": seed,
                "success": False,
                "error": result.stderr or result.stdout,
                "elapsed_sec": elapsed,
            }

        # Verify output CSV exists
        if not output_csv.exists():
            return {
                "gamma": gamma,
                "seed": seed,
                "success": False,
                "error": f"Output CSV not created: {output_csv}",
                "elapsed_sec": elapsed,
            }

        return {
            "gamma": gamma,
            "seed": seed,
            "success": True,
            "output_csv": str(output_csv),
            "elapsed_sec": elapsed,
        }

    except subprocess.TimeoutExpired:
        return {
            "gamma": gamma,
            "seed": seed,
            "success": False,
            "error": "Timeout (>300s)",
            "elapsed_sec": 300,
        }
    except Exception as e:
        return {
            "gamma": gamma,
            "seed": seed,
            "success": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="W4-Action A Five-point Synergy Scan (γ=0.15 加密點)"
    )
    parser.add_argument(
        "--skip-control",
        action="store_true",
        help="Skip control run (γ=0.0) and run only scan gammas",
    )
    parser.add_argument(
        "--gamma-only",
        type=float,
        default=None,
        help="Run only a specific gamma value (for testing)",
    )
    parser.add_argument(
        "--seed-only",
        type=int,
        default=None,
        help="Run only a specific seed (for testing)",
    )
    args = parser.parse_args()

    # Build run list
    runs = []

    # Control run (unless skipped)
    if not args.skip_control:
        for seed in SEEDS:
            if args.seed_only is not None and seed != args.seed_only:
                continue
            runs.append((GAMMA_CONTROL, seed))

    # Scan runs
    for gamma in GAMMA_SCAN:
        if args.gamma_only is not None and gamma != args.gamma_only:
            continue
        for seed in SEEDS:
            if args.seed_only is not None and seed != args.seed_only:
                continue
            runs.append((gamma, seed))

    total_runs = len(runs)
    print(f"📋 W4-Action A Five-Point Synergy Scan")
    print(f"   Total runs: {total_runs}")
    print(f"   γ grid: {[GAMMA_CONTROL] + GAMMA_SCAN if not args.skip_control else GAMMA_SCAN}")
    print(f"   seeds: {SEEDS}")
    print(f"   Output: {OUTPUT_DIR}")
    print()

    results = []
    success_count = 0
    fail_count = 0

    for idx, (gamma, seed) in enumerate(runs, 1):
        output_csv = OUTPUT_DIR / f"run_g{gamma:.2f}_s{seed}.csv"

        print(f"[{idx:2d}/{total_runs}] γ={gamma:.2f} seed={seed:2d}  ", end="", flush=True)

        result = run_single_simulation(
            gamma=gamma,
            seed=seed,
            output_csv=output_csv,
        )
        results.append(result)

        if result["success"]:
            elapsed = result.get("elapsed_sec", 0)
            print(f"✅ ({elapsed:.1f}s)")
            success_count += 1
        else:
            error_msg = result.get("error", "Unknown error")[:50]
            print(f"❌ {error_msg}")
            fail_count += 1

    # Summary
    print()
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"✅ Successful: {success_count}/{total_runs}")
    print(f"❌ Failed: {fail_count}/{total_runs}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Save results summary
    summary_path = OUTPUT_DIR / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "experiment": "W4-Action A Five-Point Synergy Scan",
                "protocol_lock": {
                    "players": PLAYERS,
                    "rounds": ROUNDS,
                    "payoff_mode": PAYOFF_MODE,
                    "a": A_PARAM,
                    "b": B_PARAM,
                    "matrix_cross_coupling": CROSS_COUPLING,
                    "selection_strength": SELECTION_STRENGTH,
                    "init_bias": INIT_BIAS,
                    "memory_kernel": MEMORY_KERNEL,
                    "payoff_lag": PAYOFF_LAG,
                    "evolution_mode": EVOLUTION_MODE,
                },
                "gamma_grid": [GAMMA_CONTROL] + GAMMA_SCAN if not args.skip_control else GAMMA_SCAN,
                "seeds": SEEDS,
                "total_runs": total_runs,
                "successful_runs": success_count,
                "failed_runs": fail_count,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"📊 Summary saved: {summary_path}")

    # Exit with appropriate code
    if fail_count > 0:
        print(f"\n⚠️  {fail_count} runs failed. Check output above.")
        sys.exit(1)
    else:
        print(f"\n✨ All {total_runs} runs completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
