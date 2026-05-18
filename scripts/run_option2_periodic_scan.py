"""Option 2 — Periodic pulse 2D scan: interval × intensity.

目的：找出是否存在 (interval, intensity) 組合能顯著延後 dropout_round，
      即使 mean_s3 仍在 0.51 左右，也算確認「外力可移動崩塌牆」。

掃描空間：
  - seeds: 12 seeds (同 Step 2)
  - conditions: control + periodic
  - periodic_interval: [180, 240, 300, 360, 480]
  - intensity_label: low / medium / high
      low    : a=1.02, b=0.88, cross=0.25  (小擾動)
      medium : a=1.05, b=0.85, cross=0.35  (與 smoke 相同)
      high   : a=1.10, b=0.80, cross=0.40  (強擾動)

輸出：
  outputs/actionB_periodic_scan/
    periodic_scan_summary.json
    periodic_scan_onepager.md
"""
from __future__ import annotations

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


OUT_DIR = ROOT / "outputs" / "actionB_periodic_scan"

SEEDS = [45, 47, 49, 51, 53, 55, 91, 93, 95, 97, 99, 101]
ROUNDS = 3000
PLAYERS = 300
SELECTION_STRENGTH = 0.10
INIT_BIAS = 0.12
MEMORY_KERNEL = 3

BASELINE = StackelbergCommitment(
    condition="control_pulse_policy",
    leader_action="baseline_pulse_policy",
    a=1.00,
    b=0.90,
    matrix_cross_coupling=0.20,
    description="Baseline payoff geometry",
)

# Three intensity levels — parameterised as (a, b, cross)
INTENSITIES: dict[str, dict[str, float]] = {
    "low":    {"a": 1.02, "b": 0.88, "cross": 0.25},
    "medium": {"a": 1.05, "b": 0.85, "cross": 0.35},
    "high":   {"a": 1.10, "b": 0.80, "cross": 0.40},
}

INTERVALS = [180, 240, 300, 360, 480]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _RollPoint:
    round_end: int
    level: int
    passed: bool


def _extract_series(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {
        "aggressive": [float(r["p_aggressive"]) for r in rows],
        "defensive":  [float(r["p_defensive"])  for r in rows],
        "balanced":   [float(r["p_balanced"])   for r in rows],
    }


def _window_eval(series3: Mapping[str, list[float]]) -> dict[str, dict[str, float | int]]:
    def _one(burn: int, tail: int) -> dict[str, float | int]:
        cyc = classify_cycle_level(
            series3, burn_in=burn, tail=tail,
            amplitude_threshold=0.02, corr_threshold=0.09, eta=0.55,
            stage3_method="turning", phase_smoothing=1, min_lag=2, max_lag=500,
        )
        return {"level": int(cyc.level), "stage3_score": float(cyc.stage3.score if cyc.stage3 else 0.0)}

    return {
        "burn1000_tail1000": _one(1000, 1000),
        "burn2000_tail2000": _one(2000, 2000),
    }


def _rolling_dropout_round(
    series3: Mapping[str, list[float]], *, sustain_windows: int = 5
) -> int | None:
    window, step = 1000, 20
    roll: list[_RollPoint] = []
    n = min(len(v) for v in series3.values())
    for end in range(window, n + 1, step):
        seg = {k: series3[k][end - window : end] for k in series3}
        cyc = classify_cycle_level(
            seg, burn_in=0, tail=None,
            amplitude_threshold=0.02, corr_threshold=0.09, eta=0.55,
            stage3_method="turning", phase_smoothing=1, min_lag=2, max_lag=500,
        )
        roll.append(_RollPoint(round_end=end, level=int(cyc.level), passed=cyc.level >= 3))
    if not roll:
        return None
    flags = [1 if p.passed else 0 for p in roll]
    try:
        first_pass_idx = flags.index(1)
    except ValueError:
        return None
    k = max(1, sustain_windows)
    for i in range(first_pass_idx + 1, max(first_pass_idx + 1, len(flags) - k + 1)):
        if flags[i] == 0 and all(flags[j] == 0 for j in range(i, i + k)):
            return roll[i].round_end
    return None


# ---------------------------------------------------------------------------
# Periodic pulse adapter (minimal, no extra deps)
# ---------------------------------------------------------------------------

class _PeriodicAdapter:
    def __init__(
        self,
        *,
        interval: int,
        pulse_horizon: int,
        refractory_rounds: int,
        pulse_commitment: StackelbergCommitment,
    ) -> None:
        self.interval = int(interval)
        self.pulse_horizon = int(pulse_horizon)
        self.refractory_rounds = int(refractory_rounds)
        self.pulse_commitment = pulse_commitment
        self.pulse_rounds_left = 0
        self.refractory_rounds_left = 0
        self.pulse_count = 0
        self.first_pulse_round: int | None = None

    def _advance(self) -> None:
        if self.pulse_rounds_left > 0:
            self.pulse_rounds_left -= 1
            if self.pulse_rounds_left == 0 and self.refractory_rounds > 0:
                self.refractory_rounds_left = self.refractory_rounds
            return
        if self.refractory_rounds_left > 0:
            self.refractory_rounds_left -= 1

    def __call__(
        self,
        round_index: int,
        _cfg: SimConfig,
        _players: list[object],
        dungeon: Any,
        _step_records: list[dict[str, object]],
        _row: dict[str, Any],
    ) -> None:
        self._advance()
        round_no = int(round_index) + 1
        if (
            self.interval > 0
            and round_no % self.interval == 0
            and self.pulse_rounds_left == 0
            and self.refractory_rounds_left == 0
        ):
            self.pulse_rounds_left = self.pulse_horizon
            self.pulse_count += 1
            if self.first_pulse_round is None:
                self.first_pulse_round = round_no
        if self.pulse_rounds_left > 0:
            c = self.pulse_commitment
        else:
            c = BASELINE
        dungeon.a = float(c.a)
        dungeon.b = float(c.b)
        dungeon.matrix_cross_coupling = float(c.matrix_cross_coupling)


# ---------------------------------------------------------------------------
# Per-task runner (must be top-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _run_task(
    condition: str,
    seed: int,
    interval: int | None,
    intensity_label: str | None,
    pulse_horizon: int,
    refractory_rounds: int,
    rounds: int,
    players: int,
    selection_strength: float,
    init_bias: float,
    memory_kernel: int,
) -> dict[str, Any]:
    is_control = condition == "control"
    out_dir = OUT_DIR / (condition if is_control else f"interval{interval}_int{intensity_label}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"seed{seed}.csv"

    if is_control:
        pulse_c = BASELINE
        adapter = None
    else:
        ip = INTENSITIES[intensity_label]
        pulse_c = StackelbergCommitment(
            condition=condition,
            leader_action=f"pulse_{intensity_label}",
            a=float(ip["a"]),
            b=float(ip["b"]),
            matrix_cross_coupling=float(ip["cross"]),
            description=f"Periodic pulse intensity={intensity_label}",
        )
        adapter = _PeriodicAdapter(
            interval=int(interval),
            pulse_horizon=int(pulse_horizon),
            refractory_rounds=int(refractory_rounds),
            pulse_commitment=pulse_c,
        )

    cfg = SimConfig(
        n_players=int(players),
        n_rounds=int(rounds),
        seed=int(seed),
        payoff_mode="matrix_ab",
        popularity_mode="sampled",
        gamma=0.1, epsilon=0.0,
        a=float(BASELINE.a), b=float(BASELINE.b),
        matrix_cross_coupling=float(BASELINE.matrix_cross_coupling),
        init_bias=float(init_bias),
        evolution_mode="sampled", payoff_lag=1,
        selection_strength=float(selection_strength),
        enable_events=True, events_json=DEFAULT_FULL_EVENTS_JSON,
        out_csv=out_csv,
        memory_kernel=int(memory_kernel),
    )

    strategy_space, rows = simulate(cfg, round_callback=adapter)
    _write_timeseries_csv(out_csv, strategy_space=strategy_space, rows=rows)
    series3 = _extract_series(rows)
    window = _window_eval(series3)
    dropout = _rolling_dropout_round(series3)

    return {
        "condition": condition,
        "interval": interval,
        "intensity_label": intensity_label,
        "seed": int(seed),
        "csv": str(out_csv.relative_to(ROOT)),
        "window_eval": window,
        "dropout_round": dropout,
        "pulse_count": int(adapter.pulse_count) if adapter is not None else 0,
        "first_pulse_round": int(adapter.first_pulse_round) if adapter is not None and adapter.first_pulse_round is not None else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks: list[tuple[Any, ...]] = []

    # Control (single condition across all seeds)
    for seed in SEEDS:
        tasks.append(("control", seed, None, None, 120, 240, ROUNDS, PLAYERS, SELECTION_STRENGTH, INIT_BIAS, MEMORY_KERNEL))

    # Periodic grid: 5 intervals × 3 intensities × 12 seeds = 180 tasks
    for interval in INTERVALS:
        for intensity_label in ("low", "medium", "high"):
            condition = f"periodic_i{interval}_{intensity_label}"
            for seed in SEEDS:
                tasks.append((condition, seed, interval, intensity_label, 120, 240, ROUNDS, PLAYERS, SELECTION_STRENGTH, INIT_BIAS, MEMORY_KERNEL))

    total = len(tasks)
    max_workers = min(8, max(1, (os.cpu_count() or 2) // 2))
    print(f"[scan] total_tasks={total}  workers={max_workers}", flush=True)

    all_rows: list[dict[str, Any]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_run_task, *t): t for t in tasks}
        for future in as_completed(future_map):
            row = future.result()
            all_rows.append(row)
            done += 1
            if done % 20 == 0 or done == total:
                print(f"[scan {done}/{total}] latest: cond={row['condition']} seed={row['seed']}", flush=True)

    # ------------------------------------------------------------------
    # Aggregate: per (interval, intensity) vs control
    # ------------------------------------------------------------------
    def _agg(sub: list[dict[str, Any]]) -> dict[str, Any]:
        short_s3 = [float(r["window_eval"]["burn1000_tail1000"]["stage3_score"]) for r in sub]
        long_s3  = [float(r["window_eval"]["burn2000_tail2000"]["stage3_score"])  for r in sub]
        short_l3 = sum(1 for r in sub if int(r["window_eval"]["burn1000_tail1000"]["level"]) >= 3)
        long_l3  = sum(1 for r in sub if int(r["window_eval"]["burn2000_tail2000"]["level"])  >= 3)
        dropouts = sorted(int(r["dropout_round"]) for r in sub if r["dropout_round"] is not None)
        n = len(sub)
        return {
            "n": n,
            "short_l3_count": short_l3,
            "long_l3_count": long_l3,
            "short_s3_mean": round(sum(short_s3) / max(n, 1), 5),
            "long_s3_mean":  round(sum(long_s3)  / max(n, 1), 5),
            "dropout_median": dropouts[len(dropouts) // 2] if dropouts else None,
            "dropout_count": len(dropouts),
            "pulse_count_mean": round(sum(float(r["pulse_count"]) for r in sub) / max(n, 1), 2),
        }

    ctrl_rows = [r for r in all_rows if r["condition"] == "control"]
    ctrl_agg = _agg(ctrl_rows)
    ctrl_dropout_median = ctrl_agg["dropout_median"] or 0

    grid: list[dict[str, Any]] = []
    for interval in INTERVALS:
        for intensity_label in ("low", "medium", "high"):
            condition = f"periodic_i{interval}_{intensity_label}"
            sub = [r for r in all_rows if r["condition"] == condition]
            a = _agg(sub)
            dropout_shift = (
                (a["dropout_median"] - ctrl_dropout_median)
                if a["dropout_median"] is not None and ctrl_dropout_median > 0
                else None
            )
            s3_shift = round(a["short_s3_mean"] - ctrl_agg["short_s3_mean"], 5)
            grid.append({
                "interval": interval,
                "intensity": intensity_label,
                "condition": condition,
                **a,
                "dropout_shift_vs_ctrl": dropout_shift,
                "short_s3_shift_vs_ctrl": s3_shift,
            })

    # Sort by dropout_shift desc (primary), s3_shift desc (secondary)
    grid_sorted = sorted(
        grid,
        key=lambda x: (-(x["dropout_shift_vs_ctrl"] or -9999), -x["short_s3_shift_vs_ctrl"]),
    )

    payload = {
        "generated_at": datetime.now().isoformat(),
        "scope": {
            "seeds": SEEDS,
            "rounds": ROUNDS,
            "players": PLAYERS,
            "selection_strength": SELECTION_STRENGTH,
            "memory_kernel": MEMORY_KERNEL,
            "intervals": INTERVALS,
            "intensities": INTENSITIES,
            "pulse_horizon": 120,
            "refractory_rounds": 240,
        },
        "control": ctrl_agg,
        "grid": grid_sorted,
        "rows": all_rows,
    }

    out_json = OUT_DIR / "periodic_scan_summary.json"
    out_md   = OUT_DIR / "periodic_scan_onepager.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Option 2 — Periodic Pulse 2D Scan (interval × intensity)")
    lines.append("")
    lines.append(f"Control baseline: dropout_median={ctrl_agg['dropout_median']}  short_s3_mean={ctrl_agg['short_s3_mean']:.4f}")
    lines.append("")
    lines.append("## 完整掃描結果（按 dropout_shift 排序）")
    lines.append("")
    lines.append("| interval | intensity | short L3 | long L3 | short_s3 | long_s3 | dropout_med | dropout_shift | s3_shift | pulse_mean |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for g in grid_sorted:
        d = "None" if g["dropout_median"] is None else str(g["dropout_median"])
        ds = "None" if g["dropout_shift_vs_ctrl"] is None else f"{g['dropout_shift_vs_ctrl']:+d}"
        ss = f"{g['short_s3_shift_vs_ctrl']:+.4f}"
        lines.append(
            f"| {g['interval']} | {g['intensity']} | {g['short_l3_count']} | {g['long_l3_count']}"
            f" | {g['short_s3_mean']:.4f} | {g['long_s3_mean']:.4f} | {d} | {ds} | {ss} | {g['pulse_count_mean']:.1f} |"
        )
    lines.append("")
    lines.append("## 診斷標準")
    lines.append("- **延壽訊號**：dropout_shift >= +200 → 確認外力可移動崩塌牆")
    lines.append("- **重燃訊號**：short_l3_count >= 2 且 short_s3 >= 0.55 → 確認 L3 再點火")
    lines.append("- **無效區**：dropout_shift <= 0 且 short_l3_count = 0 → 脈衝強度/頻率不足")
    lines.append("")
    lines.append("## 排行榜 Top-5（dropout_shift 優先）")
    lines.append("")
    lines.append("| rank | interval | intensity | dropout_shift | s3_shift | short_l3 |")
    lines.append("| ---: | ---: | --- | ---: | ---: | ---: |")
    for rank, g in enumerate(grid_sorted[:5], 1):
        ds = "None" if g["dropout_shift_vs_ctrl"] is None else f"{g['dropout_shift_vs_ctrl']:+d}"
        ss = f"{g['short_s3_shift_vs_ctrl']:+.4f}"
        lines.append(f"| {rank} | {g['interval']} | {g['intensity']} | {ds} | {ss} | {g['short_l3_count']} |")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nwritten: {out_json.relative_to(ROOT)}")
    print(f"written: {out_md.relative_to(ROOT)}")
    print()
    print(f"Control baseline: dropout_median={ctrl_agg['dropout_median']}  short_s3_mean={ctrl_agg['short_s3_mean']:.4f}")
    print()
    print("Top-5 by dropout_shift:")
    for g in grid_sorted[:5]:
        ds = "None" if g["dropout_shift_vs_ctrl"] is None else f"{g['dropout_shift_vs_ctrl']:+d}"
        print(f"  interval={g['interval']:4d} intensity={g['intensity']:7s}  dropout_shift={ds:>6s}  s3_shift={g['short_s3_shift_vs_ctrl']:+.4f}  short_l3={g['short_l3_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
