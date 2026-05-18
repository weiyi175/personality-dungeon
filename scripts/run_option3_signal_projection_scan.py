"""Option 3 - Minimal signal projection matrix scan.

This scan tests whether a projected signal layer can delay dropout in the 9-persona
setting without using periodic scalar pulses.

Grid:
- control (baseline geometry only)
- projection_matrix in {triad_soft, triad_focus, triad_rotate}
- projection_gain in {0.25, 0.50, 0.75}
- activation_theta in {0.02, 0.05}

Output:
- outputs/actionC_signal_projection_scan/signal_projection_scan_summary.json
- outputs/actionC_signal_projection_scan/signal_projection_scan_onepager.md
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
from core.stackelberg import StackelbergCommitment
from simulation.personality_gate0 import DEFAULT_FULL_EVENTS_JSON
from simulation.run_simulation import SimConfig, _write_timeseries_csv, simulate

OUT_DIR = ROOT / "outputs" / "actionC_signal_projection_scan"
DEFAULT_SEEDS = [45, 47, 49, 51, 53, 55, 91, 93, 95, 97, 99, 101]

BASELINE = StackelbergCommitment(
    condition="control_signal_projection",
    leader_action="baseline_projection_policy",
    a=1.00,
    b=0.90,
    matrix_cross_coupling=0.20,
    description="Baseline payoff geometry",
)

PROJECTED_GEOMETRIES: dict[str, StackelbergCommitment] = {
    "aggressive": StackelbergCommitment(
        condition="proj_aggressive",
        leader_action="proj_aggressive",
        a=1.05,
        b=0.85,
        matrix_cross_coupling=0.35,
        description="Projection chose aggressive sector",
    ),
    "defensive": StackelbergCommitment(
        condition="proj_defensive",
        leader_action="proj_defensive",
        a=1.00,
        b=0.90,
        matrix_cross_coupling=0.35,
        description="Projection chose defensive sector",
    ),
    "balanced": StackelbergCommitment(
        condition="proj_balanced",
        leader_action="proj_balanced",
        a=1.05,
        b=0.85,
        matrix_cross_coupling=0.20,
        description="Projection chose balanced sector",
    ),
}

PROJECTION_MATRICES: dict[str, tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = {
    # Mild cross-talk projection.
    "triad_soft": (
        (1.00, 0.12, 0.12),
        (0.12, 1.00, 0.12),
        (0.12, 0.12, 1.00),
    ),
    # Focuses each sector while preserving non-zero off-diagonals.
    "triad_focus": (
        (1.15, 0.08, 0.04),
        (0.08, 1.15, 0.04),
        (0.04, 0.08, 1.15),
    ),
    # Rotating coupling to test phase-shift style remapping.
    "triad_rotate": (
        (1.00, 0.18, 0.02),
        (0.02, 1.00, 0.18),
        (0.18, 0.02, 1.00),
    ),
}

PROJECTION_GAINS = [0.25, 0.50, 0.75]
ACTIVATION_THETAS = [0.02, 0.05]


@dataclass
class _RollPoint:
    round_end: int
    passed: bool


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
        return {
            "level": int(cyc.level),
            "stage3_score": float(cyc.stage3.score if cyc.stage3 else 0.0),
        }

    return {
        "burn1000_tail1000": _one(1000, 1000),
        "burn2000_tail2000": _one(2000, 2000),
    }


def _rolling_dropout_round(series3: Mapping[str, list[float]], sustain_windows: int = 5) -> int | None:
    window = 1000
    step = 20
    n = min(len(v) for v in series3.values())
    points: list[_RollPoint] = []
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
        points.append(_RollPoint(round_end=end, passed=int(cyc.level) >= 3))
        round_ends.append(end)
    if not points:
        return None
    flags = [1 if p.passed else 0 for p in points]
    if 1 not in flags:
        return None
    first_pass = flags.index(1)
    k = max(1, int(sustain_windows))
    for i in range(first_pass + 1, max(first_pass + 1, len(flags) - k + 1)):
        if flags[i] == 0 and all(flags[j] == 0 for j in range(i, i + k)):
            return int(round_ends[i])
    return None


def _blend_projection(
    raw_vec: tuple[float, float, float],
    matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    gain: float,
) -> tuple[float, float, float]:
    rx, ry, rz = raw_vec
    px = matrix[0][0] * rx + matrix[0][1] * ry + matrix[0][2] * rz
    py = matrix[1][0] * rx + matrix[1][1] * ry + matrix[1][2] * rz
    pz = matrix[2][0] * rx + matrix[2][1] * ry + matrix[2][2] * rz

    g = float(gain)
    bx = (1.0 - g) * rx + g * px
    by = (1.0 - g) * ry + g * py
    bz = (1.0 - g) * rz + g * pz

    # Keep simplex-compatible, then renormalize.
    bx = max(0.0, bx)
    by = max(0.0, by)
    bz = max(0.0, bz)
    s = bx + by + bz
    if s <= 0.0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (bx / s, by / s, bz / s)


def _dominant(vec: tuple[float, float, float]) -> tuple[str, float]:
    labels = ["aggressive", "defensive", "balanced"]
    vals = [(labels[0], vec[0]), (labels[1], vec[1]), (labels[2], vec[2])]
    vals.sort(key=lambda x: (-x[1], labels.index(x[0])))
    top = vals[0]
    margin = float(vals[0][1] - vals[1][1])
    return top[0], margin


class _SignalProjectionAdapter:
    def __init__(self, matrix_label: str, gain: float, theta: float) -> None:
        self.matrix_label = str(matrix_label)
        self.matrix = PROJECTION_MATRICES[self.matrix_label]
        self.gain = float(gain)
        self.theta = float(theta)

        self.activation_count = 0
        self.first_activation_round: int | None = None
        self.dominant_switch_count = 0
        self._last_projected_dominant: str | None = None

    def __call__(
        self,
        round_index: int,
        _cfg: SimConfig,
        _players: list[object],
        dungeon: Any,
        _step_records: list[dict[str, object]],
        row: dict[str, Any],
    ) -> None:
        raw = (
            float(row["p_aggressive"]),
            float(row["p_defensive"]),
            float(row["p_balanced"]),
        )
        projected = _blend_projection(raw, self.matrix, self.gain)
        dominant, margin = _dominant(projected)

        if self._last_projected_dominant is not None and dominant != self._last_projected_dominant:
            self.dominant_switch_count += 1
        self._last_projected_dominant = dominant

        if margin >= self.theta:
            c = PROJECTED_GEOMETRIES[dominant]
            self.activation_count += 1
            if self.first_activation_round is None:
                self.first_activation_round = int(round_index) + 1
        else:
            c = BASELINE

        dungeon.a = float(c.a)
        dungeon.b = float(c.b)
        dungeon.matrix_cross_coupling = float(c.matrix_cross_coupling)


def _run_task(
    condition: str,
    seed: int,
    rounds: int,
    players: int,
    selection_strength: float,
    init_bias: float,
    memory_kernel: int,
    matrix_label: str | None,
    projection_gain: float | None,
    activation_theta: float | None,
) -> dict[str, Any]:
    is_control = condition == "control"
    if is_control:
        out_dir = OUT_DIR / "control"
    else:
        out_dir = OUT_DIR / f"{matrix_label}_g{projection_gain:.2f}_th{activation_theta:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"seed{seed}.csv"

    adapter = None
    if not is_control:
        adapter = _SignalProjectionAdapter(
            matrix_label=str(matrix_label),
            gain=float(projection_gain),
            theta=float(activation_theta),
        )

    cfg = SimConfig(
        n_players=int(players),
        n_rounds=int(rounds),
        seed=int(seed),
        payoff_mode="matrix_ab",
        popularity_mode="sampled",
        gamma=0.1,
        epsilon=0.0,
        a=float(BASELINE.a),
        b=float(BASELINE.b),
        matrix_cross_coupling=float(BASELINE.matrix_cross_coupling),
        init_bias=float(init_bias),
        evolution_mode="sampled",
        payoff_lag=1,
        selection_strength=float(selection_strength),
        enable_events=True,
        events_json=DEFAULT_FULL_EVENTS_JSON,
        out_csv=out_csv,
        memory_kernel=int(memory_kernel),
    )

    strategy_space, rows = simulate(cfg, round_callback=adapter)
    _write_timeseries_csv(out_csv, strategy_space=strategy_space, rows=rows)

    series3 = _extract_series(rows)
    win = _window_eval(series3)
    dropout = _rolling_dropout_round(series3)

    return {
        "condition": condition,
        "seed": int(seed),
        "matrix_label": matrix_label,
        "projection_gain": projection_gain,
        "activation_theta": activation_theta,
        "window_eval": win,
        "dropout_round": dropout,
        "projection_activation_count": int(adapter.activation_count) if adapter is not None else 0,
        "first_projection_activation_round": int(adapter.first_activation_round) if adapter is not None and adapter.first_activation_round is not None else None,
        "projected_dominant_switch_count": int(adapter.dominant_switch_count) if adapter is not None else 0,
        "csv": str(out_csv.relative_to(ROOT)),
    }


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
        "projection_activation_count_mean": round(
            sum(float(r["projection_activation_count"]) for r in rows) / max(n, 1), 2
        ),
        "projected_dominant_switch_count_mean": round(
            sum(float(r["projected_dominant_switch_count"]) for r in rows) / max(n, 1), 2
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Option 3 minimal signal projection matrix scan")
    p.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    p.add_argument("--rounds", type=int, default=3000)
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

    for matrix_label in PROJECTION_MATRICES:
        for gain in PROJECTION_GAINS:
            for theta in ACTIVATION_THETAS:
                condition = f"proj_{matrix_label}_g{gain:.2f}_th{theta:.2f}"
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
    print(f"[scan] total_tasks={total} workers={workers}", flush=True)

    rows: list[dict[str, Any]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_run_task, *t) for t in tasks]
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            done += 1
            if done % 20 == 0 or done == total:
                print(f"[scan {done}/{total}] latest: cond={row['condition']} seed={row['seed']}", flush=True)

    control_rows = [r for r in rows if r["condition"] == "control"]
    control_agg = _agg(control_rows)
    control_dropout = int(control_agg["dropout_median"]) if control_agg["dropout_median"] is not None else 0

    grid: list[dict[str, Any]] = []
    for matrix_label in PROJECTION_MATRICES:
        for gain in PROJECTION_GAINS:
            for theta in ACTIVATION_THETAS:
                condition = f"proj_{matrix_label}_g{gain:.2f}_th{theta:.2f}"
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
        key=lambda x: (-(x["dropout_shift_vs_control"] if x["dropout_shift_vs_control"] is not None else -9999), -x["short_s3_shift_vs_control"]),
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
            "projection_matrices": list(PROJECTION_MATRICES.keys()),
            "projection_gains": PROJECTION_GAINS,
            "activation_thetas": ACTIVATION_THETAS,
        },
        "control": control_agg,
        "grid": grid_sorted,
        "rows": rows,
    }

    out_json = OUT_DIR / "signal_projection_scan_summary.json"
    out_md = OUT_DIR / "signal_projection_scan_onepager.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Option 3 - Signal Projection Matrix Minimal Scan")
    lines.append("")
    lines.append(
        f"Control baseline: dropout_median={control_agg['dropout_median']} short_s3_mean={control_agg['short_s3_mean']:.4f}"
    )
    lines.append("")
    lines.append("| matrix | gain | theta | short L3 | long L3 | short_s3 | long_s3 | dropout_med | dropout_shift | s3_shift | act_count_mean | switch_mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for g in grid_sorted:
        dm = "None" if g["dropout_median"] is None else str(g["dropout_median"])
        ds = "None" if g["dropout_shift_vs_control"] is None else f"{int(g['dropout_shift_vs_control']):+d}"
        ss = f"{float(g['short_s3_shift_vs_control']):+.4f}"
        lines.append(
            "| {matrix} | {gain:.2f} | {theta:.2f} | {short_l3} | {long_l3} | {short_s3:.4f} | {long_s3:.4f} | {dm} | {ds} | {ss} | {act:.2f} | {sw:.2f} |".format(
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
        f"Control baseline: dropout_median={control_agg['dropout_median']} short_s3_mean={control_agg['short_s3_mean']:.4f}"
    )
    print("Top-5 by dropout_shift:")
    for g in grid_sorted[:5]:
        ds = "None" if g["dropout_shift_vs_control"] is None else f"{int(g['dropout_shift_vs_control']):+d}"
        print(
            "  matrix={matrix:11s} gain={gain:.2f} theta={theta:.2f} dropout_shift={ds:>6s} s3_shift={s3:+.4f} short_l3={short_l3}".format(
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
