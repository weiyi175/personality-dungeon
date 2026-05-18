#!/usr/bin/env python3
"""H7.6 Noise Amplitude Sweep：相轉移階數偵測

目的：以細粒度噪聲強度掃描定位各盆地類型的臨界噪聲振幅 A_c，
並判斷相轉移為突然崖邊（1st-order）還是漸進衰退（2nd-order）。

────────────────────────────────────────────────────────────────────────────
實驗設計
────────────────────────────────────────────────────────────────────────────
  Seeds        : {47, 51, 53, 55, 95, 97, 99}
                   47/51/53/55  ← late-trans 抗噪組（H7.5 C3/C4 全維持 L3）
                   95/97        ← deep-stable 組（H7.5 C3/C4 跌至 L2）
                   99           ← 崩潰候選（H7.5 C3/C4 post=L3 → late=L0）
  Amplitudes   : {0.0=ctrl, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40}
  Conditions   : ctrl (no noise) + noise at each amplitude
  Pulse        : 無（聚焦純噪聲效應；脈衝效益在 H7.5 已確認為 0pp）
  規模         : 7 seeds × 8 levels = 56 runs
  n_rounds     : 6000（固定）

────────────────────────────────────────────────────────────────────────────
CSD（Critical Slowing Down）指標
────────────────────────────────────────────────────────────────────────────
  在非線性系統即將發生相變前，系統返回平衡的速度變慢、方差暴增。
  本實驗記錄各時間窗的 simplex velocity std dev：

  vs_std_early : std dev of |Δs[t]| in [500,  1500] — baseline (pre-transition)
  vs_std_post  : std dev of |Δs[t]| in [2500, 3500] — PRIMARY CSD window
  vs_std_late  : std dev of |Δs[t]| in [4000, 6000] — late-phase CSD

  其中 |Δs[t]| = sqrt(Δp_agg² + Δp_def² + Δp_bal²)（simplex Euclidean velocity）
  近臨界點時應觀察到 vs_std_post 指數級上升。

────────────────────────────────────────────────────────────────────────────
驗收條件（H7.6 3-gate）
────────────────────────────────────────────────────────────────────────────
  H7.6-G1  CRITICAL_AMPLITUDE_FOUND  seeds {95,97} 找到明確 A_c
  H7.6-G2  BASIN_DIFFERENTIAL        deep-stable A_c ≠ late-trans A_c
  H7.6-G3  ORDER_CLASSIFIED          至少一個 seed 分類為 1st/2nd-order

────────────────────────────────────────────────────────────────────────────
輸出目錄結構
────────────────────────────────────────────────────────────────────────────
  outputs/h76_noise_sweep/
    ctrl/            seed47.csv  seed51.csv  ...  (amplitude=0.0)
    amp_0.05/        seed47.csv  ...
    amp_0.10/        ...
    ...
    summary.json     2D matrix: seed → amp_label → metrics

Usage:
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u \\
    scripts/run_w76_noise_sweep.py
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u \\
    scripts/run_w76_noise_sweep.py --resume
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u \\
    scripts/run_w76_noise_sweep.py --seeds 95 97 99 --amplitudes 0.10 0.20 0.30
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.cycle_metrics import classify_cycle_level
from players.base_player import DEFAULT_PERSONALITY_KEYS
from simulation.run_simulation import SimConfig, _write_timeseries_csv, simulate

# ──────────────────────────────────────────────────────────────────────────────
# Fixed golden-point boundaries (H7.3 / H7.4 locked values)
# ──────────────────────────────────────────────────────────────────────────────
MU_BASE            = 0.30
LAMBDA_MU          = 0.05
LAMBDA_K           = 0.20
SELECTION_STRENGTH = 0.15
K_LOWER            = 0.05
K_UPPER            = 0.25
GAMMA_BASE         = 0.16
SYNERGY_POWER      = 3.2
FULL_ROUNDS        = 6000

# Analysis windows (consistent with H7.5 for cross-experiment comparability)
EARLY_WIN_START = 500
EARLY_WIN_LEN   = 1000   # [500,  1500] — baseline CSD reference (pre-transition)
POST_WIN_START  = 2500
POST_WIN_LEN    = 1000   # [2500, 3500] — primary CSD window
LATE_WIN_START  = 4000
LATE_WIN_LEN    = 2000   # [4000, 6000] — primary level evaluation
STD_BURN_IN     = 2000
STD_TAIL        = 2000   # [2000, 4000] — H7.5-compatible std window

# Default experiment parameters
DEFAULT_SEEDS = [47, 51, 53, 55, 95, 97, 99]
DEFAULT_AMPLITUDES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

# Basin reference (H7.4 classification)
DEEP_STABLE_SEEDS = {95, 97}
LATE_TRANS_SEEDS  = {47, 51, 53, 55, 99}


# ──────────────────────────────────────────────────────────────────────────────
# Persona setup
# ──────────────────────────────────────────────────────────────────────────────
def _noise_persona_setup(seed: int, amplitude: float) -> Callable:
    """Return a player_setup_callback that sets all 9 personality dims to
    uniform(-amplitude, +amplitude).

    RNG derivation is identical to H7.5 (seed * 10000 + player_idx), ensuring
    that at amplitude=0.4 the results are directly comparable to H7.5 C3/C4.
    """
    def cb(players: list, _strategy_space: list, _cfg: SimConfig) -> None:
        for idx, player in enumerate(players):
            rng = random.Random(int(seed) * 10000 + idx)
            for key in DEFAULT_PERSONALITY_KEYS:
                if hasattr(player, "personality") and key in player.personality:
                    player.personality[key] = rng.uniform(-amplitude, amplitude)
    return cb


# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SweepMetric:
    seed: int
    amplitude: float         # 0.0 = ctrl (no noise)
    amp_label: str           # "ctrl" | "0.05" | "0.10" | ...
    csv_path: str
    # Level classifications
    std_level: int           # [2000, 4000]
    post_level: int          # [2500, 3500]
    late_level: int          # [4000, 6000]  ← primary
    l3_late: bool
    l3_post: bool
    # CSD: simplex velocity std dev per window
    vs_std_early: float      # [500,  1500] baseline
    vs_std_post: float       # [2500, 3500] PRIMARY CSD indicator
    vs_std_late: float       # [4000, 6000] late-phase CSD


# ──────────────────────────────────────────────────────────────────────────────
# Computation helpers
# ──────────────────────────────────────────────────────────────────────────────
def _extract_proportions(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {
        "aggressive": [float(r.get("p_aggressive", 0.0)) for r in rows],
        "defensive":  [float(r.get("p_defensive",  0.0)) for r in rows],
        "balanced":   [float(r.get("p_balanced",   0.0)) for r in rows],
    }


def _classify_window(rows: list[dict[str, Any]], start: int, length: int) -> int:
    end = min(start + length, len(rows))
    sub = rows[start:end]
    if len(sub) < 50:
        return 0
    props = _extract_proportions(sub)
    result = classify_cycle_level(
        props, burn_in=0, tail=len(sub),
        amplitude_threshold=0.02, eta=0.6, min_turn_strength=0.0,
    )
    return int(result.level)


def _vs_std(rows: list[dict[str, Any]], start: int, length: int) -> float:
    """Std dev of per-step Euclidean simplex velocity in window [start, start+length).

    v_s[t] = sqrt(Δp_agg[t]² + Δp_def[t]² + Δp_bal[t]²)

    Near a critical transition, variance of v_s inflates (CSD signature).
    Returns float('nan') when window is too short (<10 rows).
    """
    end = min(start + length, len(rows))
    sub = rows[start:end]
    if len(sub) < 10:
        return float("nan")
    p_agg = [float(r.get("p_aggressive", 0.0)) for r in sub]
    p_def = [float(r.get("p_defensive",  0.0)) for r in sub]
    p_bal = [float(r.get("p_balanced",   0.0)) for r in sub]
    vs: list[float] = []
    for i in range(len(sub) - 1):
        da = p_agg[i + 1] - p_agg[i]
        dd = p_def[i + 1] - p_def[i]
        db = p_bal[i + 1] - p_bal[i]
        vs.append(math.sqrt(da * da + dd * dd + db * db))
    if not vs:
        return float("nan")
    n = len(vs)
    mean = sum(vs) / n
    var = sum((v - mean) ** 2 for v in vs) / n
    return math.sqrt(var)


# ──────────────────────────────────────────────────────────────────────────────
# Core run
# ──────────────────────────────────────────────────────────────────────────────
def run_one(
    *,
    seed: int,
    amplitude: float,
    out_dir: Path,
    resume: bool = False,
) -> SweepMetric:
    """Simulate one (seed, amplitude) combination and return metrics.

    With resume=True, an existing CSV is read back instead of re-simulating.
    Output path: out_dir/ctrl/seed{seed}.csv  (amplitude == 0.0)
                 out_dir/amp_{amp:.2f}/seed{seed}.csv  (amplitude > 0.0)
    """
    is_ctrl = amplitude == 0.0
    amp_label = "ctrl" if is_ctrl else f"{amplitude:.2f}"
    subdir = out_dir / ("ctrl" if is_ctrl else f"amp_{amplitude:.2f}")
    subdir.mkdir(parents=True, exist_ok=True)
    out_csv = subdir / f"seed{seed}.csv"

    cfg = SimConfig(
        n_players=300,
        n_rounds=FULL_ROUNDS,
        seed=int(seed),
        payoff_mode="matrix_ab",
        popularity_mode="sampled",
        gamma=GAMMA_BASE,
        epsilon=0.0,
        a=1.0,
        b=0.9,
        matrix_cross_coupling=0.20,
        init_bias=0.5,
        evolution_mode="personality_coupled",
        payoff_lag=1,
        selection_strength=float(SELECTION_STRENGTH),
        enable_events=False,
        events_json=None,
        out_csv=out_csv,
        memory_kernel=1,
        synergy_type="nonlinear",
        synergy_gamma=float(GAMMA_BASE),
        synergy_nonlinear_type="power",
        synergy_nonlinear_power=float(SYNERGY_POWER),
        personality_coupling_mu_base=float(MU_BASE),
        personality_coupling_lambda_mu=float(LAMBDA_MU),
        personality_coupling_lambda_k=float(LAMBDA_K),
        personality_coupling_mu_lower=0.0,
        personality_coupling_mu_upper=0.60,
        personality_coupling_k_lower=float(K_LOWER),
        personality_coupling_k_upper=float(K_UPPER),
        # No pulse in H7.6 — pure noise effect
        synergy_pulse_t_start=None,
        synergy_pulse_duration=None,
        synergy_pulse_delta_gamma=0.0,
    )

    # Resume: re-read existing CSV instead of re-simulating
    if resume and out_csv.exists():
        with out_csv.open(newline="") as f:
            rows = list(_csv.DictReader(f))
    else:
        persona_cb = _noise_persona_setup(seed, amplitude) if not is_ctrl else None
        _strategy_space, rows = simulate(cfg, player_setup_callback=persona_cb)
        _write_timeseries_csv(out_csv, strategy_space=_strategy_space, rows=rows)

    std_level  = _classify_window(rows, STD_BURN_IN,    STD_TAIL)
    post_level = _classify_window(rows, POST_WIN_START,  POST_WIN_LEN)
    late_level = _classify_window(rows, LATE_WIN_START,  LATE_WIN_LEN)

    vs_std_early = _vs_std(rows, EARLY_WIN_START, EARLY_WIN_LEN)
    vs_std_post  = _vs_std(rows, POST_WIN_START,  POST_WIN_LEN)
    vs_std_late  = _vs_std(rows, LATE_WIN_START,  LATE_WIN_LEN)

    rel_csv = str(out_csv.relative_to(ROOT)) if out_csv.is_relative_to(ROOT) else str(out_csv)
    return SweepMetric(
        seed=int(seed),
        amplitude=float(amplitude),
        amp_label=amp_label,
        csv_path=rel_csv,
        std_level=std_level,
        post_level=post_level,
        late_level=late_level,
        l3_late=late_level >= 3,
        l3_post=post_level >= 3,
        vs_std_early=vs_std_early,
        vs_std_post=vs_std_post,
        vs_std_late=vs_std_late,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────────────
def _find_critical_amplitude(
    by_seed: dict[str, list[SweepMetric]],
    seed: int,
    amplitudes: list[float],
) -> float | None:
    """Return the first amplitude ≥ 0.01 where the seed loses late L3.

    Returns None if:
    - ctrl (amp=0.0) is not L3 (no meaningful A_c to find)
    - L3 never lost within the scanned range
    """
    mets = {m.amplitude: m for m in by_seed.get(str(seed), [])}
    ctrl = mets.get(0.0)
    if ctrl is None or not ctrl.l3_late:
        return None
    for amp in sorted(a for a in amplitudes if a > 0.0):
        m = mets.get(amp)
        if m is not None and not m.l3_late:
            return float(amp)
    return None  # still L3 at all scanned amplitudes


def _classify_transition_order(
    by_seed: dict[str, list[SweepMetric]],
    seed: int,
    amplitudes: list[float],
) -> str:
    """Classify the phase transition mode.

    Returns one of:
    - "1st_order"        : L3 → L0 cliff (no L2 intermediate)
    - "2nd_order"        : L3 → L2 → L0 gradual decay (all three levels seen)
    - "2nd_order_partial": L3 → L2 seen but L0 not yet reached in range
    - "no_transition"    : stays L3 throughout the scanned range
    - "undetermined"     : ctrl is not L3, or insufficient data
    """
    mets = {m.amplitude: m for m in by_seed.get(str(seed), [])}
    sorted_amps = sorted(a for a in amplitudes if a >= 0.0)
    levels = [mets[a].late_level for a in sorted_amps if a in mets]

    if not levels or levels[0] < 3:
        return "undetermined"

    seen = set(levels)
    if seen == {3}:
        return "no_transition"
    if 3 in seen and 0 in seen and 2 not in seen:
        return "1st_order"
    if 3 in seen and 2 in seen and 0 in seen:
        return "2nd_order"
    if 3 in seen and 2 in seen and 0 not in seen:
        return "2nd_order_partial"
    return "undetermined"


def _csd_peak_amplitude(
    by_seed: dict[str, list[SweepMetric]],
    seed: int,
) -> float | None:
    """Return the amplitude at which vs_std_post peaks (CSD maximum).

    A peak in vs_std_post at amplitude A ≈ A_c is the canonical CSD signature.
    Returns None if data are insufficient.
    """
    mets = sorted(by_seed.get(str(seed), []), key=lambda m: m.amplitude)
    valid = [(m.amplitude, m.vs_std_post) for m in mets
             if m.amplitude > 0.0 and not math.isnan(m.vs_std_post)]
    if len(valid) < 2:
        return None
    return float(max(valid, key=lambda x: x[1])[0])


def _build_summary(
    all_results: list[SweepMetric],
    seeds: list[int],
    amplitudes: list[float],
) -> dict:
    # Group by seed
    by_seed: dict[str, list[SweepMetric]] = {}
    for m in all_results:
        by_seed.setdefault(str(m.seed), []).append(m)

    # 2D matrix: seed → amp_label → metric dict
    def _clean(v: float) -> Any:
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, float):
            return round(v, 6)
        return v

    matrix: dict[str, dict[str, dict]] = {}
    for seed in seeds:
        key = str(seed)
        matrix[key] = {}
        for m in sorted(by_seed.get(key, []), key=lambda x: x.amplitude):
            matrix[key][m.amp_label] = {
                "std_level":    m.std_level,
                "post_level":   m.post_level,
                "late_level":   m.late_level,
                "l3_late":      m.l3_late,
                "l3_post":      m.l3_post,
                "vs_std_early": _clean(m.vs_std_early),
                "vs_std_post":  _clean(m.vs_std_post),
                "vs_std_late":  _clean(m.vs_std_late),
                "csv_path":     m.csv_path,
            }

    # Per-seed analysis
    crit_amps: dict[str, Any]    = {}
    trans_orders: dict[str, str] = {}
    csd_peaks: dict[str, Any]    = {}
    for seed in seeds:
        crit_amps[str(seed)]   = _find_critical_amplitude(by_seed, seed, amplitudes)
        trans_orders[str(seed)] = _classify_transition_order(by_seed, seed, amplitudes)
        csd_peaks[str(seed)]   = _csd_peak_amplitude(by_seed, seed)

    # Gate evaluation
    deep_crit = [crit_amps[str(s)] for s in DEEP_STABLE_SEEDS if str(s) in crit_amps]
    late_crit  = [crit_amps[str(s)] for s in LATE_TRANS_SEEDS  if str(s) in crit_amps]

    g1 = any(v is not None for v in deep_crit)
    deep_found  = set(v for v in deep_crit if v is not None)
    late_found  = set(v for v in late_crit  if v is not None)
    g2 = bool(deep_found) and bool(late_found) and deep_found != late_found
    g3 = any(v in ("1st_order", "2nd_order") for v in trans_orders.values())

    verdict = ("PASS"    if g1 and g2 and g3 else
               "PARTIAL" if g1 or  g2 or  g3 else
               "FAIL")

    return {
        "meta": {
            "experiment": "H7.6",
            "seeds": seeds,
            "amplitudes": amplitudes,
            "n_runs": len(all_results),
            "golden_point": {
                "mu_base": MU_BASE, "lambda_mu": LAMBDA_MU,
                "SS": SELECTION_STRENGTH, "gamma_base": GAMMA_BASE,
                "synergy_power": SYNERGY_POWER,
            },
            "windows": {
                "early": [EARLY_WIN_START, EARLY_WIN_START + EARLY_WIN_LEN],
                "post":  [POST_WIN_START,  POST_WIN_START  + POST_WIN_LEN],
                "late":  [LATE_WIN_START,  LATE_WIN_START  + LATE_WIN_LEN],
                "std":   [STD_BURN_IN,     STD_BURN_IN     + STD_TAIL],
            },
        },
        "matrix": matrix,
        "critical_amplitudes": crit_amps,
        "transition_orders": trans_orders,
        "csd_peak_amplitudes": csd_peaks,
        "gates": {
            "g1_critical_amplitude_found": g1,
            "g2_basin_differential": g2,
            "g3_order_classified": g3,
        },
        "verdict": verdict,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Output printing
# ──────────────────────────────────────────────────────────────────────────────
def _basin_label(seed: int) -> str:
    if seed in DEEP_STABLE_SEEDS:
        return "deep-stable"
    if seed in LATE_TRANS_SEEDS:
        return "late-trans"
    return "?"


def _print_sweep_table(all_results: list[SweepMetric], summary: dict) -> None:
    W = 130
    print()
    print("=" * W)
    print("H7.6 Noise Amplitude Sweep — Results")
    print(f"  Windows: std[{STD_BURN_IN},{STD_BURN_IN+STD_TAIL}] | "
          f"post[{POST_WIN_START},{POST_WIN_START+POST_WIN_LEN}] | "
          f"late[{LATE_WIN_START},{LATE_WIN_START+LATE_WIN_LEN}]")
    print(f"  CSD: vs_std = std dev of per-step Euclidean simplex velocity")
    print("=" * W)

    by_sa: dict[tuple[int, float], SweepMetric] = {
        (m.seed, m.amplitude): m for m in all_results
    }
    seeds_present = sorted({m.seed for m in all_results})
    amps_present  = sorted({m.amplitude for m in all_results})

    amp_labels = ["ctrl" if a == 0.0 else f"{a:.2f}" for a in amps_present]
    hdr = "  ".join(f"{lb:^6}" for lb in amp_labels)

    # ── Table 1: late_level ──────────────────────────────────────────────────
    print()
    print(f"  Late Level (primary):   {'seed':>5}  {'basin':>12}  {hdr}  {'A_c':>5}  order")
    print("  " + "-" * (W - 2))
    for seed in seeds_present:
        cells = []
        for amp in amps_present:
            m = by_sa.get((seed, amp))
            if m is None:
                cells.append(" ?  ")
            elif m.late_level == 3:
                cells.append("L3✓ ")
            elif m.late_level == 2:
                cells.append("L2  ")
            elif m.late_level == 1:
                cells.append("L1  ")
            else:
                cells.append("L0  ")
        crit = summary["critical_amplitudes"].get(str(seed))
        crit_s = f"{crit:.2f}" if crit is not None else " — "
        order_s = summary["transition_orders"].get(str(seed), "?")
        row = "  ".join(f"{c:^6}" for c in cells)
        print(f"  {' ':>5}  {seed:>5}  {_basin_label(seed):>12}  {row}  {crit_s:>5}  {order_s}")

    # ── Table 2: vs_std_post (CSD) ───────────────────────────────────────────
    print()
    print(f"  CSD vs_std_post:        {'seed':>5}  {'basin':>12}  {hdr}  CSD_peak")
    print("  " + "-" * (W - 2))
    for seed in seeds_present:
        cells = []
        for amp in amps_present:
            m = by_sa.get((seed, amp))
            if m is None or math.isnan(m.vs_std_post):
                cells.append("  N/A  ")
            else:
                cells.append(f"{m.vs_std_post:.4f}")
        csd_peak = summary["csd_peak_amplitudes"].get(str(seed))
        csd_s = f"{csd_peak:.2f}" if csd_peak is not None else " — "
        row = "  ".join(f"{c:^6}" for c in cells)
        print(f"  {' ':>5}  {seed:>5}  {_basin_label(seed):>12}  {row}  {csd_s:>5}")

    print()
    print("=" * W)

    # ── Gate summary ─────────────────────────────────────────────────────────
    gates = summary["gates"]
    crit_amps = summary["critical_amplitudes"]
    trans_orders = summary["transition_orders"]
    csd_peaks = summary["csd_peak_amplitudes"]
    print()
    print("  Critical amplitudes (A_c):")
    for seed in seeds_present:
        s = str(seed)
        ac = crit_amps.get(s)
        to = trans_orders.get(s, "?")
        cp = csd_peaks.get(s)
        ac_s  = f"{ac:.2f}" if ac is not None else "  —  "
        cp_s  = f"{cp:.2f}" if cp is not None else "  —  "
        print(f"    seed={seed:>3}  [{_basin_label(seed):>12}]"
              f"  A_c={ac_s}  order={to:<20}  CSD_peak={cp_s}")
    print()
    g1s = "✓" if gates["g1_critical_amplitude_found"] else "✗"
    g2s = "✓" if gates["g2_basin_differential"] else "✗"
    g3s = "✓" if gates["g3_order_classified"] else "✗"
    print(f"  H7.6-G1 CRITICAL_AMPLITUDE_FOUND  {g1s}  (seeds {{95,97}} have A_c in range)")
    print(f"  H7.6-G2 BASIN_DIFFERENTIAL         {g2s}  (deep-stable A_c ≠ late-trans A_c)")
    print(f"  H7.6-G3 ORDER_CLASSIFIED            {g3s}  (≥1 seed classified as 1st/2nd-order)")
    print()
    print(f"  VERDICT: {summary['verdict']}")
    print("=" * W)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="H7.6 Noise Amplitude Sweep — Critical Slowing Down Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip simulation if CSV already exists; re-read and recompute metrics.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        metavar="SEED",
        help=f"Seeds to run (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--amplitudes", type=float, nargs="+", default=DEFAULT_AMPLITUDES,
        metavar="AMP",
        help="Noise amplitudes (0.0 = ctrl always included; "
             f"default: {DEFAULT_AMPLITUDES})",
    )
    args = parser.parse_args()

    seeds = sorted(set(args.seeds))
    amplitudes = sorted(set(args.amplitudes))
    if 0.0 not in amplitudes:
        amplitudes = [0.0] + amplitudes

    out_dir = ROOT / "outputs" / "h76_noise_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / "summary.json"

    n_runs = len(seeds) * len(amplitudes)
    print("H7.6 Noise Amplitude Sweep")
    print(f"  Golden point : μ={MU_BASE}, λ_μ={LAMBDA_MU}, SS={SELECTION_STRENGTH}, "
          f"γ_base={GAMMA_BASE}, power={SYNERGY_POWER}")
    print(f"  Seeds        : {seeds}")
    print(f"  Amplitudes   : {amplitudes}")
    print(f"  Total runs   : {n_runs}")
    print(f"  Output dir   : {out_dir}")
    if args.resume:
        print(f"  Resume       : ON  (existing CSVs will be reused)")
    print()

    all_results: list[SweepMetric] = []
    for run_idx, seed in enumerate(seeds):
        for amp_idx, amp in enumerate(amplitudes):
            global_idx = run_idx * len(amplitudes) + amp_idx + 1
            amp_label = "ctrl" if amp == 0.0 else f"{amp:.2f}"

            m = run_one(seed=seed, amplitude=amp, out_dir=out_dir, resume=args.resume)
            all_results.append(m)

            csd_s = (f"  vs_std_post={m.vs_std_post:.5f}"
                     if not math.isnan(m.vs_std_post) else "")
            print(
                f"[{global_idx:>3}/{n_runs}]  seed={seed:>4}  amp={amp_label:<5}  "
                f"late=L{m.late_level}{'✓' if m.l3_late else ' '}  "
                f"post=L{m.post_level}{'✓' if m.l3_post else ' '}  "
                f"std=L{m.std_level}{csd_s}"
            )

    summary = _build_summary(all_results, seeds, amplitudes)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _print_sweep_table(all_results, summary)
    print(f"\nSummary saved → {summary_json}")


if __name__ == "__main__":
    main()
