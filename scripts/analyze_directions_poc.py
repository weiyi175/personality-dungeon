#!/usr/bin/env python3
"""
Analyze PoC directions results and append summary to 研發日誌.md

Usage: PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python scripts/analyze_directions_poc.py
"""

import csv
import json
import sys
from pathlib import Path
from dataclasses import asdict, dataclass

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.cycle_metrics import (
    phase_direction_consistency_turning,
    phase_rotation_r2,
    classify_cycle_level,
)


@dataclass
class RunDiag:
    name: str
    csv_path: str
    cycle_level: int
    phase_velocity: float
    turn_strength: float
    stagnant: bool
    l3_pass: bool
    velocity_pass: bool


def load_timeseries(csv_path: Path) -> dict:
    props = {"aggressive": [], "defensive": [], "balanced": []}
    try:
        with open(csv_path, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                props["aggressive"].append(float(row.get("p_aggressive", 0.0)))
                props["defensive"].append(float(row.get("p_defensive", 0.0)))
                props["balanced"].append(float(row.get("p_balanced", 0.0)))
    except Exception:
        pass
    return props


def compute_diag(proportions: dict, tail: int = 1000):
    if not proportions["balanced"]:
        return 0, 0.0, 0.0

    # phase_velocity via phase_rotation_r2 cumulative rotation/window
    try:
        rot = phase_rotation_r2(proportions, burn_in=0, tail=tail)
        window_len = rot.window_length if rot.window_length > 0 else 1
        phase_velocity = float(rot.cumulative_rotation) / float(window_len)
    except Exception:
        phase_velocity = 0.0

    try:
        turning = phase_direction_consistency_turning(proportions, burn_in=0, tail=tail)
        turn_strength = float(turning.turn_strength)
    except Exception:
        turn_strength = 0.0

    try:
        cycle = classify_cycle_level(proportions, burn_in=0, tail=tail)
        cycle_level = int(cycle.level)
    except Exception:
        cycle_level = 0

    stagnant = (phase_velocity <= 0.001)
    return cycle_level, phase_velocity, turn_strength, stagnant


def main():
    root = Path(__file__).parent.parent
    results_file = root / "outputs" / "poc_directions_results.json"
    if not results_file.exists():
        print("poc_directions_results.json not found", file=sys.stderr)
        sys.exit(1)

    results = json.loads(results_file.read_text())
    velocity_threshold = 0.001
    summary = {
        "velocity_threshold": velocity_threshold,
        "directions": {},
    }

    for dir_key, dir_data in results.get("directions", {}).items():
        runs = dir_data.get("runs", [])
        run_diags = []
        for r in runs:
            csvp = Path(r.get("csv_path", ""))
            proportions = load_timeseries(csvp)
            cycle_level, phase_velocity, turn_strength, stagnant = compute_diag(proportions)
            l3_pass = cycle_level >= 3
            velocity_pass = phase_velocity > velocity_threshold
            rd = RunDiag(
                name=r.get("name", ""),
                csv_path=str(csvp),
                cycle_level=cycle_level,
                phase_velocity=phase_velocity,
                turn_strength=turn_strength,
                stagnant=stagnant,
                l3_pass=l3_pass,
                velocity_pass=velocity_pass,
            )
            run_diags.append(rd)

        total = len(run_diags)
        succ_count = sum(1 for rd in run_diags if rd.cycle_level >= 2 and rd.phase_velocity > velocity_threshold)
        l3_count = sum(1 for rd in run_diags if rd.l3_pass)
        velocity_count = sum(1 for rd in run_diags if rd.velocity_pass)
        stagnant_pct = 100.0 * sum(1 for rd in run_diags if rd.stagnant) / total if total else 100.0
        l3_rate = (l3_count / total) if total else 0.0
        velocity_pass_rate = (velocity_count / total) if total else 0.0

        direction_pass = (succ_count / total >= 0.5) and (stagnant_pct < 50.0) if total else False

        summary["directions"][dir_key] = {
            "name": dir_data.get("name"),
            "total_runs": total,
            "successful_runs": succ_count,
            "l3_successful_runs": l3_count,
            "velocity_pass_runs": velocity_count,
            "l3_achievement_rate": l3_rate,
            "velocity_pass_rate": velocity_pass_rate,
            "stagnant_percent": stagnant_pct,
            "pass": direction_pass,
            "runs": [asdict(r) for r in run_diags],
        }

    outp = root / "outputs" / "poc_directions_analysis.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, indent=2))
    print(f"Saved analysis: {outp}")

    # Append short summary to 研發日誌.md
    logf = root / "研發日誌.md"
    lines = ["\n## PoC Directions Analysis\n", f"Timestamp: {__import__('datetime').datetime.now().isoformat()}\n"]
    for k, v in summary["directions"].items():
        status = "PASS" if v["pass"] else "FAIL"
        lines.append(f"- {k} ({v['name']}): {status} — {v['successful_runs']}/{v['total_runs']} runs successful, stagnant={v['stagnant_percent']:.1f}%\n")
        lines.append(
            f"  L3_rate={v['l3_achievement_rate']:.1%}, velocity_pass_rate={v['velocity_pass_rate']:.1%} (v>{velocity_threshold})\n"
        )

    try:
        with open(logf, "a") as f:
            f.writelines(lines)
        print(f"Appended summary to {logf}")
    except Exception as e:
        print(f"Failed to append to 研發日誌.md: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
