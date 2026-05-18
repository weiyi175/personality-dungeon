#!/usr/bin/env python3
"""Analyze C power refine scan and output stability window table."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.cycle_metrics import classify_cycle_level, phase_direction_consistency_turning, phase_rotation_r2


@dataclass
class RunDiag:
    gamma: float
    power: float
    seed: int
    csv_path: str
    cycle_level: int
    phase_velocity: float
    turn_strength: float
    l3_pass: bool
    velocity_pass: bool
    both_pass: bool


def load_timeseries(csv_path: Path) -> dict[str, list[float]]:
    p = {"aggressive": [], "defensive": [], "balanced": []}
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            p["aggressive"].append(float(row.get("p_aggressive", 0.0)))
            p["defensive"].append(float(row.get("p_defensive", 0.0)))
            p["balanced"].append(float(row.get("p_balanced", 0.0)))
    return p


def compute_diag(gamma: float, power: float, seed: int, csv_path: Path, velocity_threshold: float) -> RunDiag:
    props = load_timeseries(csv_path)

    rot = phase_rotation_r2(props, burn_in=0, tail=1000)
    window_len = rot.window_length if rot.window_length > 0 else 1
    phase_velocity = float(rot.cumulative_rotation) / float(window_len)

    turning = phase_direction_consistency_turning(props, burn_in=0, tail=1000)
    turn_strength = float(turning.turn_strength)

    c = classify_cycle_level(props, burn_in=0, tail=1000)
    cycle_level = int(c.level)

    l3 = cycle_level >= 3
    vel = phase_velocity > velocity_threshold
    both = l3 and vel

    return RunDiag(
        gamma=float(gamma),
        power=float(power),
        seed=int(seed),
        csv_path=str(csv_path),
        cycle_level=cycle_level,
        phase_velocity=phase_velocity,
        turn_strength=turn_strength,
        l3_pass=l3,
        velocity_pass=vel,
        both_pass=both,
    )


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def main() -> int:
    root = Path(__file__).parent.parent
    out_dir = root / "outputs" / "c_power_refine_scan"
    summary_path = out_dir / "run_summary.json"
    if not summary_path.exists():
        print(f"missing: {summary_path}", file=sys.stderr)
        return 1

    velocity_threshold = 0.001
    summary = json.loads(summary_path.read_text())
    results = summary.get("results", [])

    run_diags: list[RunDiag] = []
    for r in results:
        if not r.get("success"):
            continue
        csv_path = Path(r["csv_path"])
        run_diags.append(
            compute_diag(
                gamma=float(r["gamma"]),
                power=float(r["power"]),
                seed=int(r["seed"]),
                csv_path=csv_path,
                velocity_threshold=velocity_threshold,
            )
        )

    by_combo: dict[tuple[float, float], list[RunDiag]] = {}
    for d in run_diags:
        key = (d.gamma, d.power)
        by_combo.setdefault(key, []).append(d)

    combo_rows = []
    for (gamma, power), rows in sorted(by_combo.items()):
        n = len(rows)
        l3_rate = sum(1 for x in rows if x.l3_pass) / n if n else 0.0
        vel_rate = sum(1 for x in rows if x.velocity_pass) / n if n else 0.0
        both_rate = sum(1 for x in rows if x.both_pass) / n if n else 0.0
        phase_vals = [x.phase_velocity for x in rows]
        median_v = _median(phase_vals)
        min_v = min(phase_vals) if phase_vals else 0.0
        stable = (both_rate >= 0.5)
        combo_rows.append(
            {
                "gamma": gamma,
                "power": power,
                "n_seeds": n,
                "l3_achievement_rate": l3_rate,
                "velocity_pass_rate": vel_rate,
                "both_pass_rate": both_rate,
                "median_phase_velocity": median_v,
                "min_phase_velocity": min_v,
                "stable_window": stable,
            }
        )

    stable_rows = [r for r in combo_rows if r["stable_window"]]
    stable_rows.sort(key=lambda x: (x["both_pass_rate"], x["median_phase_velocity"]), reverse=True)

    analysis = {
        "experiment": "C power refine scan analysis",
        "velocity_threshold": velocity_threshold,
        "run_count": len(run_diags),
        "combo_count": len(combo_rows),
        "stable_combo_count": len(stable_rows),
        "combo_summary": combo_rows,
        "stable_windows": stable_rows,
        "run_diagnostics": [asdict(x) for x in run_diags],
    }

    analysis_path = out_dir / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2))

    md_path = out_dir / "stability_window_table.md"
    lines = []
    lines.append("# C Power Refine Stability Window\n")
    lines.append(f"velocity_threshold: {velocity_threshold}\n")
    lines.append("")
    lines.append("| gamma | power | n_seeds | L3_rate | velocity_pass_rate | both_pass_rate | median_velocity | min_velocity | stable |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in combo_rows:
        lines.append(
            "| "
            f"{row['gamma']:.2f} | {row['power']:.1f} | {row['n_seeds']} | "
            f"{row['l3_achievement_rate']:.1%} | {row['velocity_pass_rate']:.1%} | {row['both_pass_rate']:.1%} | "
            f"{row['median_phase_velocity']:.6f} | {row['min_phase_velocity']:.6f} | "
            f"{'yes' if row['stable_window'] else 'no'} |"
        )

    lines.append("")
    lines.append("## Stable Windows")
    if stable_rows:
        for row in stable_rows:
            lines.append(
                f"- gamma={row['gamma']:.2f}, power={row['power']:.1f}, both_pass_rate={row['both_pass_rate']:.1%}, median_velocity={row['median_phase_velocity']:.6f}"
            )
    else:
        lines.append("- none")

    md_path.write_text("\n".join(lines) + "\n")

    print(f"saved: {analysis_path}")
    print(f"saved: {md_path}")
    print(f"stable windows: {len(stable_rows)}/{len(combo_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
