#!/usr/bin/env python3
"""
W4-Action A: 診斷指標分析 & 結果統計

計算所有 36 個 run 的：
1. phase_velocity (phase rotation 速率)
2. turn_strength (轉向強度)
3. STAGNANT flag (活力崩潰診斷)
4. Δmetrics vs control (γ=0)

輸出：JSON + 表格摘要
"""

import csv
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any
from math import sqrt, log

# Add personality-dungeon to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.cycle_metrics import (
    phase_direction_consistency_turning,
    phase_rotation_r2,
    classify_cycle_level,
)


@dataclass
class DiagnosticResult:
    """Per-run diagnostic metrics."""
    gamma: float
    seed: int
    short_s3_mean: float  # tail-window mean of p_balanced
    phase_velocity: float  # cumulative_rotation / window_length
    turn_strength: float  # direction consistency turning strength
    cycle_level: int  # 0, 1, 2, 3
    stagnant_flag: bool  # velocity_ratio ≤ 0.5 AND turn_ratio ≤ 0.5


def load_timeseries_csv(csv_path: Path) -> dict[str, list[float]]:
    """Load proportions from timeseries CSV."""
    proportions = {
        "aggressive": [],
        "defensive": [],
        "balanced": [],
    }
    
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                proportions["aggressive"].append(float(row.get("p_aggressive", 0.0)))
                proportions["defensive"].append(float(row.get("p_defensive", 0.0)))
                proportions["balanced"].append(float(row.get("p_balanced", 0.0)))
    except Exception as e:
        print(f"❌ Error loading {csv_path}: {e}", file=sys.stderr)
        return proportions
    
    return proportions


def compute_diagnostics(
    proportions: dict[str, list[float]],
    gamma: float,
    seed: int,
) -> DiagnosticResult:
    """Compute diagnostic metrics for a single run."""
    
    if not proportions["balanced"]:
        return DiagnosticResult(
            gamma=gamma,
            seed=seed,
            short_s3_mean=0.0,
            phase_velocity=0.0,
            turn_strength=0.0,
            cycle_level=0,
            stagnant_flag=True,
        )
    
    # 1. short_s3_mean: tail-window mean (last 1000 rounds)
    tail = 1000
    s3_tail = proportions["balanced"][-tail:] if len(proportions["balanced"]) > tail else proportions["balanced"]
    short_s3_mean = sum(s3_tail) / len(s3_tail) if s3_tail else 0.0
    
    # 2. phase_velocity: R² slope + cumulative rotation / window length
    try:
        rot = phase_rotation_r2(proportions, burn_in=0, tail=tail)
        cumul_rot = rot.cumulative_rotation
        window_len = rot.window_length if rot.window_length > 0 else 1
        phase_velocity = cumul_rot / window_len  # rotation per step
    except Exception:
        cumul_rot = 0.0
        phase_velocity = 0.0
    
    # 3. turn_strength: direction consistency via 3-point metric
    try:
        phase = phase_direction_consistency_turning(
            proportions,
            burn_in=0,
            tail=tail,
            eta=0.6,
            min_turn_strength=0.0,
            phase_smoothing=1,
        )
        turn_strength = phase.turn_strength
    except Exception:
        turn_strength = 0.0
    
    # 4. cycle_level: full classification
    try:
        cycle_result = classify_cycle_level(
            proportions,
            burn_in=0,
            tail=tail,
            amplitude_threshold=0.02,
            eta=0.6,
            min_turn_strength=0.0,
        )
        cycle_level = cycle_result.level
    except Exception:
        cycle_level = 0
    
    # 5. STAGNANT flag: (velocity_ratio ≤ 0.5) AND (turn_ratio ≤ 0.5)
    # Placeholder: we'll compute this after control baseline
    stagnant_flag = False
    
    return DiagnosticResult(
        gamma=gamma,
        seed=seed,
        short_s3_mean=short_s3_mean,
        phase_velocity=phase_velocity,
        turn_strength=turn_strength,
        cycle_level=cycle_level,
        stagnant_flag=stagnant_flag,
    )


def main():
    """Main analysis pipeline."""
    
    output_dir = Path(__file__).parent.parent / "outputs" / "w4_synergy_smoke"
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Gamma grid and seeds
    gamma_grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.4]
    seeds = [45, 47, 49, 51, 53, 55]
    
    print(f"📊 W4-Action A Diagnostic Analysis")
    print(f"   Output dir: {output_dir}")
    print()
    
    # Load all runs
    all_results: list[DiagnosticResult] = []
    control_results: dict[int, DiagnosticResult] = {}  # seed -> control result
    
    for gamma in gamma_grid:
        for seed in seeds:
            csv_path = output_dir / f"run_g{gamma:.2f}_s{seed}.csv"
            
            print(f"📂 Loading γ={gamma:.2f} seed={seed:2d}  ", end="", flush=True)
            
            if not csv_path.exists():
                print(f"❌ Not found")
                continue
            
            proportions = load_timeseries_csv(csv_path)
            result = compute_diagnostics(proportions, gamma, seed)
            all_results.append(result)
            
            if gamma == 0.0:
                control_results[seed] = result
            
            print(f"✅ (s3_mean={result.short_s3_mean:.4f}, L{result.cycle_level})")
    
    print()
    print(f"✅ Loaded {len(all_results)} runs (control: {len(control_results)} seeds)")
    
    # Compute STAGNANT flags using control baseline
    if control_results:
        # Calculate control baseline metrics
        control_velocities = [r.phase_velocity for r in control_results.values()]
        control_turns = [r.turn_strength for r in control_results.values()]
        
        median_ctrl_vel = sorted(control_velocities)[len(control_velocities) // 2] if control_velocities else 1.0
        median_ctrl_turn = sorted(control_turns)[len(control_turns) // 2] if control_turns else 1.0
        
        # Mark STAGNANT for non-control runs
        for result in all_results:
            if result.gamma > 0.0:
                vel_ratio = result.phase_velocity / median_ctrl_vel if median_ctrl_vel > 0 else 0.0
                turn_ratio = result.turn_strength / median_ctrl_turn if median_ctrl_turn > 0 else 0.0
                result.stagnant_flag = (vel_ratio <= 0.5) and (turn_ratio <= 0.5)
    
    # Save detailed results
    results_json = output_dir / "diagnostics.json"
    with open(results_json, "w") as f:
        json.dump(
            {
                "experiment": "W4-Action A Diagnostics",
                "gamma_grid": gamma_grid,
                "seeds": seeds,
                "results": [asdict(r) for r in all_results],
            },
            f,
            indent=2,
        )
    print(f"💾 Diagnostics saved: {results_json}")
    
    # Generate summary tables
    print()
    print("=" * 90)
    print("Per-Seed Summary (last 1000 rounds, tail window)")
    print("=" * 90)
    print()
    
    # Per-seed table
    print("| Seed | γ=0.0  | Δ(0.05) | Δ(0.1)  | Δ(0.15) | Δ(0.2)  | Δ(0.4)  |")
    print("|------|--------|---------|---------|---------|---------|---------|")
    
    for seed in seeds:
        control = control_results.get(seed)
        if not control:
            continue
        
        row_data = [seed]
        row_data.append(f"{control.short_s3_mean:.4f}")
        
        for gamma in [0.05, 0.1, 0.15, 0.2, 0.4]:
            scan_run = next((r for r in all_results if r.gamma == gamma and r.seed == seed), None)
            if scan_run:
                delta = scan_run.short_s3_mean - control.short_s3_mean
                row_data.append(f"{delta:+.4f}")
            else:
                row_data.append("N/A")
        
        print(f"| {row_data[0]:4d} | {row_data[1]:6s} | {row_data[2]:7s} | {row_data[3]:7s} | {row_data[4]:7s} | {row_data[5]:7s} | {row_data[6]:7s} |")
    
    print()
    print("=" * 90)
    print("Per-Gamma Aggregation")
    print("=" * 90)
    print()
    
    # Per-gamma table
    print("| γ     | n_seeds | median(Δs3) | IQR(Δs3) | min_cycle_lvl | max_cycle_lvl | n_stagnant |")
    print("|-------|---------|-------------|----------|---------------|---------------|------------|")
    
    for gamma in gamma_grid:
        if gamma == 0.0:
            print(f"| {gamma:.2f}  | control | baseline    | —        | —             | —             | —          |")
            continue
        
        gamma_runs = [r for r in all_results if r.gamma == gamma]
        if not gamma_runs:
            continue
        
        # Δs3 vs control
        deltas = []
        for run in gamma_runs:
            ctrl = control_results.get(run.seed)
            if ctrl:
                deltas.append(run.short_s3_mean - ctrl.short_s3_mean)
        
        if deltas:
            deltas_sorted = sorted(deltas)
            median_delta = deltas_sorted[len(deltas_sorted) // 2]
            q1 = deltas_sorted[len(deltas_sorted) // 4]
            q3 = deltas_sorted[3 * len(deltas_sorted) // 4]
            iqr_delta = q3 - q1
        else:
            median_delta = 0.0
            iqr_delta = 0.0
        
        min_cycle = min((r.cycle_level for r in gamma_runs), default=0)
        max_cycle = max((r.cycle_level for r in gamma_runs), default=0)
        n_stagnant = sum(1 for r in gamma_runs if r.stagnant_flag)
        
        print(f"| {gamma:.2f}  | {len(gamma_runs):7d} | {median_delta:+11.4f} | {iqr_delta:8.4f} | {min_cycle:13d} | {max_cycle:13d} | {n_stagnant:10d} |")
    
    print()
    print("✨ Analysis complete!")


if __name__ == "__main__":
    main()
