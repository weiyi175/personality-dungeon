"""Reduced-persona scan: collapse 9-persona -> 3-persona test runner.

This runner reuses the simulation harness and injects per-player personalities
that only have non-zero values on a reduced set of trait keys (default:
assertiveness, stability_seeking, curiosity).

Outputs:
- outputs/actionD_reduced_persona_scan/reduced_persona_scan_summary.json
- outputs/actionD_reduced_persona_scan/*/seed{seed}.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.cycle_metrics import classify_cycle_level
from simulation.run_simulation import SimConfig, _write_timeseries_csv, simulate
from players.base_player import DEFAULT_PERSONALITY_KEYS

OUT_DIR = ROOT / "outputs" / "actionD_reduced_persona_scan"
DEFAULT_SEEDS = [45, 47, 49, 51, 53, 55]


def _extract_series(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {
        "aggressive": [float(r["p_aggressive"]) for r in rows],
        "defensive": [float(r["p_defensive"]) for r in rows],
        "balanced": [float(r["p_balanced"]) for r in rows],
    }


def _window_eval(series3: Mapping[str, list[float]]) -> dict[str, dict[str, float | int]]:
    def _one(burn: int, tail: int) -> dict[str, float | int]:
        cyc = classify_cycle_level(
            series3,
            burn_in=burn,
            tail=tail,
            amplitude_threshold=0.02,
            corr_threshold=0.09,
            eta=0.55,
            stage3_method="turning",
            phase_smoothing=1,
            min_lag=2,
            max_lag=500,
        )
        return {"level": int(cyc.level), "stage3_score": float(cyc.stage3.score if cyc.stage3 else 0.0)}

    return {"burn1000_tail1000": _one(1000, 1000), "burn2000_tail2000": _one(2000, 2000)}


def _rolling_dropout_round(series3: Mapping[str, list[float]], sustain_windows: int = 5) -> int | None:
    window = 1000
    step = 20
    n = min(len(v) for v in series3.values())
    points: list[int] = []
    round_ends: list[int] = []
    for end in range(window, n + 1, step):
        seg = {k: series3[k][end - window : end] for k in series3}
        cyc = classify_cycle_level(
            seg,
            burn_in=0,
            tail=None,
            amplitude_threshold=0.02,
            corr_threshold=0.09,
            eta=0.55,
            stage3_method="turning",
            phase_smoothing=1,
            min_lag=2,
            max_lag=500,
        )
        points.append(int(cyc.level) >= 3)
        round_ends.append(end)
    if not points:
        return None
    if 1 not in points:
        return None
    first_pass = points.index(1)
    k = max(1, int(sustain_windows))
    for i in range(first_pass + 1, max(first_pass + 1, len(points) - k + 1)):
        if points[i] == 0 and all(points[j] == 0 for j in range(i, i + k)):
            return int(round_ends[i])
    return None


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    short_s3 = [float(r["window_eval"]["burn1000_tail1000"]["stage3_score"]) for r in rows]
    long_s3 = [float(r["window_eval"]["burn2000_tail2000"]["stage3_score"]) for r in rows]
    short_l3 = sum(1 for r in rows if int(r["window_eval"]["burn1000_tail1000"]["level"]) >= 3)
    long_l3 = sum(1 for r in rows if int(r["window_eval"]["burn2000_tail2000"]["level"]) >= 3)
    dropouts = sorted(int(r["dropout_round"]) for r in rows if r["dropout_round"] is not None)
    return {
        "n": n,
        "short_l3_count": short_l3,
        "long_l3_count": long_l3,
        "short_s3_mean": round(sum(short_s3) / max(n, 1), 5),
        "long_s3_mean": round(sum(long_s3) / max(n, 1), 5),
        "dropout_median": dropouts[len(dropouts) // 2] if dropouts else None,
        "dropout_count": len(dropouts),
    }


def _player_setup_reduced(keys: list[str], seed: int):
    """Return a callback that assigns per-player personalities with only `keys` non-zero.

    Values are sampled deterministically from seed+index in [-0.4, 0.4].
    """

    def cb(players, strategy_space, cfg):
        import random

        for i, pl in enumerate(players):
            rng = random.Random(int(seed) + i + 1000)
            # zero out all keys first
            for k in DEFAULT_PERSONALITY_KEYS:
                pl.personality[k] = 0.0
            # assign reduced keys
            for k in keys:
                if k not in pl.personality:
                    continue
                pl.personality[k] = rng.uniform(-0.4, 0.4)

    return cb


def _run_task(condition: str, seed: int, rounds: int, players: int, memory_kernel: int, reduced_keys: list[str] | None) -> dict[str, Any]:
    is_control = condition == "control"
    if is_control:
        out_dir = OUT_DIR / "control"
    else:
        label = "_".join(reduced_keys) if reduced_keys is not None else "reduced"
        out_dir = OUT_DIR / f"reduced3_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"seed{seed}.csv"

    player_setup = None
    if not is_control and reduced_keys is not None:
        player_setup = _player_setup_reduced(reduced_keys, seed)

    cfg = SimConfig(
        n_players=int(players),
        n_rounds=int(rounds),
        seed=int(seed),
        payoff_mode="matrix_ab",
        popularity_mode="sampled",
        gamma=0.1,
        epsilon=0.0,
        a=1.00,
        b=0.90,
        matrix_cross_coupling=0.20,
        init_bias=0.12,
        evolution_mode="sampled",
        payoff_lag=1,
        selection_strength=0.1,
        enable_events=False,
        events_json=None,
        out_csv=out_csv,
        memory_kernel=int(memory_kernel),
    )

    strategy_space, rows = simulate(cfg, player_setup_callback=player_setup)
    _write_timeseries_csv(out_csv, strategy_space=strategy_space, rows=rows)

    series3 = _extract_series(rows)
    win = _window_eval(series3)
    dropout = _rolling_dropout_round(series3)

    return {
        "condition": condition,
        "seed": int(seed),
        "reduced_keys": reduced_keys,
        "window_eval": win,
        "dropout_round": dropout,
        "csv": str(out_csv.relative_to(ROOT)),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reduced-persona scan (3-core axes) ")
    p.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    p.add_argument("--rounds", type=int, default=6000)
    p.add_argument("--players", type=int, default=300)
    p.add_argument("--memory-kernel", type=int, default=3)
    p.add_argument("--workers", type=int, default=min(6, max(1, (os.cpu_count() or 2) // 2)))
    p.add_argument(
        "--reduced-keys",
        type=str,
        nargs="*",
        default=["assertiveness", "stability_seeking", "curiosity"],
        help="Trait keys to keep non-zero (defaults to assertiveness, stability_seeking, curiosity)",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    seeds = [int(s) for s in args.seeds]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[Any, ...]] = []
    for seed in seeds:
        tasks.append(("control", seed, int(args.rounds), int(args.players), int(args.memory_kernel), None))
        tasks.append(("reduced3", seed, int(args.rounds), int(args.players), int(args.memory_kernel), list(args.reduced_keys)))

    total = len(tasks)
    workers = max(1, int(args.workers))
    print(f"[reduced-scan] total_tasks={total} workers={workers}", flush=True)

    rows: list[dict[str, Any]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_run_task, *t) for t in tasks]
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            done += 1
            if done % 10 == 0 or done == total:
                print(f"[reduced-scan {done}/{total}] latest: cond={row['condition']} seed={row['seed']}", flush=True)

    control_rows = [r for r in rows if r["condition"] == "control"]
    control_agg = _agg(control_rows)
    control_dropout = int(control_agg["dropout_median"]) if control_agg["dropout_median"] is not None else 0

    grid: list[dict[str, Any]] = []
    for cond in {r["condition"] for r in rows if r["condition"] != "control"}:
        sub = [r for r in rows if r["condition"] == cond]
        a = _agg(sub)
        dropout_shift = None
        if a["dropout_median"] is not None and control_dropout > 0:
            dropout_shift = int(a["dropout_median"]) - control_dropout
        s3_shift = round(float(a["short_s3_mean"]) - float(control_agg["short_s3_mean"]), 5)
        grid.append({"condition": cond, **a, "dropout_shift_vs_control": dropout_shift, "short_s3_shift_vs_control": s3_shift})

    grid_sorted = sorted(grid, key=lambda x: (-(x["dropout_shift_vs_control"] if x["dropout_shift_vs_control"] is not None else -9999), -x["short_s3_shift_vs_control"]))

    payload = {
        "generated_at": datetime.now().isoformat(),
        "scope": {"seeds": seeds, "rounds": int(args.rounds), "players": int(args.players), "memory_kernel": int(args.memory_kernel), "reduced_keys": list(args.reduced_keys)},
        "control": control_agg,
        "grid": grid_sorted,
        "rows": rows,
    }

    out_json = OUT_DIR / "reduced_persona_scan_summary.json"
    out_md = OUT_DIR / "reduced_persona_scan_onepager.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Reduced-persona Scan")
    lines.append("")
    lines.append(f"Control baseline: dropout_median={control_agg['dropout_median']} short_s3_mean={control_agg['short_s3_mean']:.4f}")
    lines.append("")
    lines.append("| condition | short L3 | long L3 | short_s3 | long_s3 | dropout_med | dropout_shift | s3_shift |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for g in grid_sorted:
        dm = "None" if g["dropout_median"] is None else str(g["dropout_median"])
        ds = "None" if g["dropout_shift_vs_control"] is None else f"{int(g['dropout_shift_vs_control']):+d}"
        ss = f"{float(g['short_s3_shift_vs_control']):+.4f}"
        lines.append(f"| {g['condition']} | {int(g['short_l3_count'])} | {int(g['long_l3_count'])} | {float(g['short_s3_mean']):.4f} | {float(g['long_s3_mean']):.4f} | {dm} | {ds} | {ss} |")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"written: {out_json.relative_to(ROOT)}")
    print(f"written: {out_md.relative_to(ROOT)}")
    print(f"Control baseline: dropout_median={control_agg['dropout_median']} short_s3_mean={control_agg['short_s3_mean']:.4f}")
    print("Top-5 by dropout_shift:")
    for g in grid_sorted[:5]:
        ds = "None" if g["dropout_shift_vs_control"] is None else f"{int(g['dropout_shift_vs_control']):+d}"
        print(f"  cond={g['condition']} dropout_shift={ds} s3_shift={g['short_s3_shift_vs_control']:+.4f} short_l3={g['short_l3_count']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
