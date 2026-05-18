"""Option 3 - Signal projection 10k confirm runner.

This runner reuses the Option 3 projection scan helpers but restricts the
experiment to the top-3 projection cells from the 6,000-round confirm:
- proj_triad_rotate_g0.75_th0.02
- proj_triad_soft_g0.50_th0.02
- proj_triad_focus_g0.75_th0.02

Outputs:
- outputs/actionC_signal_projection_confirm/signal_projection_confirm_summary.json
- outputs/actionC_signal_projection_confirm/signal_projection_confirm_onepager.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_option3_signal_projection_scan import (  # noqa: E402
    DEFAULT_SEEDS,
    _agg,
    _run_task,
)

OUT_DIR = ROOT / "outputs" / "actionC_signal_projection_confirm"
TOP_CONDITIONS = [
    ("proj_triad_rotate_g0.75_th0.02", "triad_rotate", 0.75, 0.02),
    ("proj_triad_soft_g0.50_th0.02", "triad_soft", 0.50, 0.02),
    ("proj_triad_focus_g0.75_th0.02", "triad_focus", 0.75, 0.02),
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Option 3 signal projection 10k confirm")
    p.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    p.add_argument("--rounds", type=int, default=10000)
    p.add_argument("--players", type=int, default=300)
    p.add_argument("--selection-strength", type=float, default=0.10)
    p.add_argument("--init-bias", type=float, default=0.12)
    p.add_argument("--memory-kernel", type=int, default=3)
    p.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) // 2)))
    return p


def main() -> int:
    args = _build_parser().parse_args()
    seeds = [int(s) for s in args.seeds]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[Any, ...]] = []
    for seed in seeds:
        tasks.append(
            (
                "control",
                seed,
                int(args.rounds),
                int(args.players),
                float(args.selection_strength),
                float(args.init_bias),
                int(args.memory_kernel),
                None,
                None,
                None,
            )
        )

    for condition, matrix_label, gain, theta in TOP_CONDITIONS:
        for seed in seeds:
            tasks.append(
                (
                    condition,
                    seed,
                    int(args.rounds),
                    int(args.players),
                    float(args.selection_strength),
                    float(args.init_bias),
                    int(args.memory_kernel),
                    matrix_label,
                    float(gain),
                    float(theta),
                )
            )

    total = len(tasks)
    workers = max(1, int(args.workers))
    print(f"[confirm] total_tasks={total} workers={workers}", flush=True)

    rows: list[dict[str, Any]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_run_task, *t) for t in tasks]
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            done += 1
            if done % 12 == 0 or done == total:
                print(f"[confirm {done}/{total}] latest: cond={row['condition']} seed={row['seed']}", flush=True)

    control_rows = [r for r in rows if r["condition"] == "control"]
    control_agg = _agg(control_rows)
    control_dropout = int(control_agg["dropout_median"]) if control_agg["dropout_median"] is not None else 0

    grid: list[dict[str, Any]] = []
    for condition, matrix_label, gain, theta in TOP_CONDITIONS:
        sub = [r for r in rows if r["condition"] == condition]
        a = _agg(sub)
        dropout_shift = None
        if a["dropout_median"] is not None and control_dropout > 0:
            dropout_shift = int(a["dropout_median"]) - control_dropout
        s3_shift = round(float(a["short_s3_mean"]) - float(control_agg["short_s3_mean"]), 5)
        grid.append(
            {
                "condition": condition,
                "matrix_label": matrix_label,
                "projection_gain": gain,
                "activation_theta": theta,
                **a,
                "dropout_shift_vs_control": dropout_shift,
                "short_s3_shift_vs_control": s3_shift,
            }
        )

    grid_sorted = sorted(
        grid,
        key=lambda x: (
            -(x["dropout_shift_vs_control"] if x["dropout_shift_vs_control"] is not None else -9999),
            -x["short_s3_shift_vs_control"],
        ),
    )

    payload = {
        "generated_at": datetime.now().isoformat(),
        "scope": {
            "seeds": seeds,
            "rounds": int(args.rounds),
            "players": int(args.players),
            "selection_strength": float(args.selection_strength),
            "init_bias": float(args.init_bias),
            "memory_kernel": int(args.memory_kernel),
            "top_conditions": [
                {
                    "condition": condition,
                    "matrix_label": matrix_label,
                    "projection_gain": gain,
                    "activation_theta": theta,
                }
                for condition, matrix_label, gain, theta in TOP_CONDITIONS
            ],
        },
        "control": control_agg,
        "grid": grid_sorted,
        "rows": rows,
    }

    out_json = OUT_DIR / "signal_projection_confirm_summary.json"
    out_md = OUT_DIR / "signal_projection_confirm_onepager.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Option 3 - Signal Projection Matrix 10k Confirm")
    lines.append("")
    lines.append(
        f"Control baseline: dropout_median={control_agg['dropout_median']} short_s3_mean={control_agg['short_s3_mean']:.5f}"
    )
    lines.append("")
    lines.append("| matrix | gain | theta | short L3 | long L3 | short_s3 | long_s3 | dropout_med | dropout_shift | s3_shift | act_count_mean | switch_mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for g in grid_sorted:
        dm = "None" if g["dropout_median"] is None else str(g["dropout_median"])
        ds = "None" if g["dropout_shift_vs_control"] is None else f"{int(g['dropout_shift_vs_control']):+d}"
        ss = f"{float(g['short_s3_shift_vs_control']):+.5f}"
        lines.append(
            "| {matrix} | {gain:.2f} | {theta:.2f} | {short_l3} | {long_l3} | {short_s3:.5f} | {long_s3:.5f} | {dm} | {ds} | {ss} | {act:.2f} | {sw:.2f} |".format(
                matrix=g["matrix_label"],
                gain=float(g["projection_gain"]),
                theta=float(g["activation_theta"]),
                short_l3=int(g["short_l3_count"]),
                long_l3=int(g["long_l3_count"]),
                short_s3=float(g["short_s3_mean"]),
                long_s3=float(g["long_s3_mean"]),
                dm=dm,
                ds=ds,
                ss=ss,
                act=float(g["projection_activation_count_mean"]),
                sw=float(g["projected_dominant_switch_count_mean"]),
            )
        )

    lines.append("")
    lines.append("## Heuristic gates")
    lines.append("- lifespan signal: dropout_shift_vs_control >= +200")
    lines.append("- relight signal: short_l3_count >= 2 and short_s3_mean >= 0.55")
    lines.append("- structural move: lifespan signal with activation_count_mean > 0")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"written: {out_json.relative_to(ROOT)}")
    print(f"written: {out_md.relative_to(ROOT)}")
    print(
        f"Control baseline: dropout_median={control_agg['dropout_median']} short_s3_mean={control_agg['short_s3_mean']:.5f}"
    )
    print("Top-3 confirm cells by dropout_shift:")
    for g in grid_sorted:
        ds = "None" if g["dropout_shift_vs_control"] is None else f"{int(g['dropout_shift_vs_control']):+d}"
        print(
            "  matrix={matrix:11s} gain={gain:.2f} theta={theta:.2f} dropout_shift={ds:>6s} s3_shift={s3:+.5f} short_l3={short_l3}".format(
                matrix=str(g["matrix_label"]),
                gain=float(g["projection_gain"]),
                theta=float(g["activation_theta"]),
                ds=ds,
                s3=float(g["short_s3_shift_vs_control"]),
                short_l3=int(g["short_l3_count"]),
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())