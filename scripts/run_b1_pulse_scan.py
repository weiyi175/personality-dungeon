#!/usr/bin/env python3
"""B1 Phase-Defibrillation Pulse Scan

目的：在 μ=0.30 黃金土壤上測試「暫時增強 synergy_gamma」是否能將
L2 的種子推入 L3 吸子，並觀察脈衝消失後的「餘震持續率」。

────────────────────────────────────────────────────────────────────────────
脈衝機制
────────────────────────────────────────────────────────────────────────────
  在固定時間窗口 [T_START, T_START + duration) 內，將有效 synergy_gamma
  臨時提升 Δγ：
    γ_eff(t) = γ_base + Δγ   (if T_START ≤ t < T_START + duration)
    γ_eff(t) = γ_base         (otherwise)

  注意：Δγ 作用在 nonlinear power 項上，形成非均一推力：
    Δu_s = γ_eff · H(x) · sign(1/3 − x_s) · |1/3 − x_s|^p
  這與「均一成長加法（g_s + γ_pulse）」完全不同——後者在正規化後消失。

────────────────────────────────────────────────────────────────────────────
固定邊界（Golden Point，來自 H7.3）
────────────────────────────────────────────────────────────────────────────
  μ_base=0.30, λ_μ=0.05, λ_k=0.20, SS=0.15, k_clamp=[0.05,0.25]
  synergy_type=nonlinear, γ_base=0.16, power=3.2
  T_START=1500  →  脈衝在 burn-in 尾段介入，分析窗 [2000,4000] 純量餘震
  n_rounds=6000, burn_in=2000, tail=2000

────────────────────────────────────────────────────────────────────────────
掃描格局
────────────────────────────────────────────────────────────────────────────
  掃描變數：
    Δγ       ∈ {0.05, 0.10, 0.20}
    duration ∈ {500, 1000}
  條件：no_pulse (Δγ=0), pulse_dg{Δγ}_d{duration}
  Full : 12 seeds × (1 no_pulse + 3×2 pulse) = 84 runs
  Quick: 3  seeds × (1 no_pulse + 2×2 pulse) = 15 runs (Δγ ∈ {0.10, 0.20})

────────────────────────────────────────────────────────────────────────────
驗收條件（B1 3-gate）
────────────────────────────────────────────────────────────────────────────
  B1-G1  UPLIFT     L3_pulse > L3_no_pulse  (at least 1pp gain)
  B1-G2  AFTERSHOCK L3_aftershock ≥ 50%    (脈衝消失後仍維持 L3)
  B1-G3  RESCUE     rescued_seeds ≥ 2      (H7.3 卡在 L2 的種子被救出)

Usage:
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python scripts/run_b1_pulse_scan.py
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python scripts/run_b1_pulse_scan.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.cycle_metrics import classify_cycle_level, phase_rotation_r2
from simulation.run_simulation import SimConfig, _write_timeseries_csv, simulate

# ──────────────────────────────────────────────────────────────────────────────
# Fixed golden-point boundaries (from H7.3)
# ──────────────────────────────────────────────────────────────────────────────
MU_BASE = 0.30
LAMBDA_MU = 0.05
LAMBDA_K = 0.20
SELECTION_STRENGTH = 0.15
K_LOWER = 0.05
K_UPPER = 0.25
GAMMA_BASE = 0.16
SYNERGY_POWER = 3.2
PULSE_T_START = 1500   # fires in burn-in tail; aftershock = analysis window [2000,4000]
FULL_ROUNDS = 6000
QUICK_ROUNDS = 6000    # same rounds; quick just uses fewer seeds
BURN_IN = 2000
TAIL = 2000

LATE_WINDOW_START = 4000  # fixed late window start for all conditions
LATE_WINDOW_TAIL = 2000   # [4000, 6000] — well after any pulse ends

# Scan variables
FULL_DELTA_GAMMAS: list[float] = [0.05, 0.10, 0.20]
QUICK_DELTA_GAMMAS: list[float] = [0.10, 0.20]
FULL_DURATIONS: list[int] = [500, 1000]
QUICK_DURATIONS: list[int] = [500, 1000]

FULL_SEEDS = [45, 47, 49, 51, 53, 55, 91, 93, 95, 97, 99, 123]
QUICK_SEEDS = [45, 47, 49]

# H7.3 seeds known to be stuck at L2 (control_none, mu=0.30, 6000 rounds)
# Used for B1-G3 (rescue metric)
H73_L2_SEEDS = {45, 49, 91, 123}   # L0: 93 (excluded; L0 != L2)

# ──────────────────────────────────────────────────────────────────────────────
# Acceptance gate thresholds
# ──────────────────────────────────────────────────────────────────────────────
GATE_UPLIFT_MIN: float = 0.01        # B1-G1: at least 1pp gain vs no_pulse
GATE_AFTERSHOCK_MIN: float = 0.50   # B1-G2: ≥50% aftershock L3 rate
GATE_RESCUE_MIN: int = 2             # B1-G3: rescue ≥2 H7.3-L2 seeds


# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RunMetric:
    seed: int
    delta_gamma: float   # 0.0 = no_pulse control
    duration: int        # 0 = no_pulse control
    condition_label: str
    csv_path: str
    # Standard analysis window [burn_in, burn_in+tail]
    cycle_level: int
    phase_velocity: float
    l3_pass: bool
    # Aftershock window [T_START+duration, T_START+duration+TAIL]
    aftershock_level: int
    l3_aftershock: bool
    # Late window [4000, 6000] — tests permanence
    late_level: int
    l3_late: bool


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _extract_series(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {
        "aggressive": [float(row.get("p_aggressive", 0.0)) for row in rows],
        "defensive": [float(row.get("p_defensive", 0.0)) for row in rows],
        "balanced": [float(row.get("p_balanced", 0.0)) for row in rows],
    }


def _classify_window(rows: list[dict[str, Any]], start: int, tail: int) -> tuple[int, float]:
    """Classify cycle level in a sub-window [start, start+tail]."""
    end = min(start + tail, len(rows))
    sub = rows[start:end]
    if len(sub) < 50:
        return 0, 0.0
    series = _extract_series(sub)
    cyc = classify_cycle_level(
        series, burn_in=0, tail=len(sub),
        amplitude_threshold=0.02, eta=0.6, min_turn_strength=0.0,
    )
    rot = phase_rotation_r2(series, burn_in=0, tail=len(sub))
    wlen = rot.window_length if rot.window_length > 0 else 1
    vel = float(rot.cumulative_rotation) / float(wlen)
    return int(cyc.level), vel


# ──────────────────────────────────────────────────────────────────────────────
# Core run
# ──────────────────────────────────────────────────────────────────────────────
def run_one(
    *,
    seed: int,
    delta_gamma: float,
    duration: int,
    out_dir: Path,
    rounds: int,
) -> RunMetric:
    """Run a single simulation and return metrics."""
    if delta_gamma == 0.0 or duration == 0:
        label = "no_pulse"
        pulse_t_start = None
        pulse_duration = None
        pulse_dg = 0.0
    else:
        label = f"pulse_dg{delta_gamma:.2f}_d{duration}"
        pulse_t_start = PULSE_T_START
        pulse_duration = duration
        pulse_dg = float(delta_gamma)

    tag_dir = out_dir / label
    tag_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tag_dir / f"seed{seed}.csv"

    cfg = SimConfig(
        n_players=300,
        n_rounds=int(rounds),
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
        # B1 pulse fields
        synergy_pulse_t_start=pulse_t_start,
        synergy_pulse_duration=pulse_duration,
        synergy_pulse_delta_gamma=pulse_dg,
    )

    _strategy_space, rows = simulate(cfg)
    _write_timeseries_csv(out_csv, strategy_space=_strategy_space, rows=rows)

    # Standard analysis window
    cyc_level, vel = _classify_window(rows, BURN_IN, TAIL)

    # Aftershock window: begins immediately after pulse ends
    if delta_gamma > 0.0 and duration > 0:
        aftershock_start = PULSE_T_START + duration
    else:
        aftershock_start = BURN_IN   # no_pulse: same as standard window
    aftershock_level, _ = _classify_window(rows, aftershock_start, TAIL)

    # Late window: fixed [4000, 6000] — tests permanence of rescue
    late_level, _ = _classify_window(rows, LATE_WINDOW_START, LATE_WINDOW_TAIL)

    return RunMetric(
        seed=int(seed),
        delta_gamma=float(delta_gamma),
        duration=int(duration),
        condition_label=label,
        csv_path=str(out_csv.relative_to(ROOT)) if out_csv.is_relative_to(ROOT) else str(out_csv),
        cycle_level=cyc_level,
        phase_velocity=vel,
        l3_pass=cyc_level >= 3,
        aftershock_level=aftershock_level,
        l3_aftershock=aftershock_level >= 3,
        late_level=late_level,
        l3_late=late_level >= 3,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def _evaluate_b1(all_results: list[RunMetric]) -> dict:
    """Evaluate B1 gates for each (delta_gamma, duration) combination.

    All three gates use ``l3_aftershock`` as the primary metric so that pulse
    conditions are assessed in the clean post-pulse window
    [T_START+duration, T_START+duration+TAIL].  For no_pulse runs the
    aftershock window equals the standard analysis window, so the comparison
    is always apples-to-apples (same-length, same-offset-from-pulse-end).

    ``l3_pass`` (standard window [BURN_IN, BURN_IN+TAIL]) is recorded for
    reference but is not used in gate decisions to avoid the window-overlap
    contamination that occurs when duration=1000 overlaps the analysis window.

    B1 no_pulse L2 seeds (not H7.3 L2 seeds) are used as the rescue reference
    because the B1 baseline uses a different synergy config than H7.3.
    """
    no_pulse = [r for r in all_results if r.delta_gamma == 0.0]
    # Uniform metric: l3_aftershock
    no_pulse_shock_rate = (
        sum(1 for r in no_pulse if r.l3_aftershock) / len(no_pulse) if no_pulse else 0.0
    )
    no_pulse_l2_seeds = {r.seed for r in no_pulse if not r.l3_aftershock}
    no_pulse_l3_seeds = {r.seed for r in no_pulse if r.l3_aftershock}

    combos: dict[tuple[float, int], dict] = {}
    for dg in sorted({r.delta_gamma for r in all_results if r.delta_gamma > 0.0}):
        for dur in sorted({r.duration for r in all_results if r.delta_gamma > 0.0}):
            pulse_runs = [r for r in all_results if r.delta_gamma == dg and r.duration == dur]
            if not pulse_runs:
                continue
            shock_rate = sum(1 for r in pulse_runs if r.l3_aftershock) / len(pulse_runs)
            late_rate = sum(1 for r in pulse_runs if r.l3_late) / len(pulse_runs)
            # Rescue: seeds that were L2 in no_pulse baseline but L3 in post-pulse window
            rescued = {r.seed for r in pulse_runs if r.l3_aftershock and r.seed in no_pulse_l2_seeds}

            g1 = (shock_rate - no_pulse_shock_rate) >= GATE_UPLIFT_MIN
            g2 = shock_rate >= GATE_AFTERSHOCK_MIN
            g3 = len(rescued) >= GATE_RESCUE_MIN

            combos[(dg, dur)] = {
                "delta_gamma": dg,
                "duration": dur,
                "n_runs": len(pulse_runs),
                "no_pulse_shock_rate": no_pulse_shock_rate,
                "pulse_shock_rate": shock_rate,
                "pulse_late_rate": late_rate,
                "uplift_pp": (shock_rate - no_pulse_shock_rate) * 100,
                "aftershock_rate": shock_rate,
                "rescued_seeds": sorted(rescued),
                "n_rescued": len(rescued),
                "g1_uplift": g1,
                "g2_aftershock": g2,
                "g3_rescue": g3,
                "passes": g1 and g2 and g3,
            }
    return {
        "no_pulse_shock_rate": no_pulse_shock_rate,
        "no_pulse_l2_seeds": sorted(no_pulse_l2_seeds),
        "no_pulse_l3_seeds": sorted(no_pulse_l3_seeds),
        "combos": combos,
    }


def _print_b1_table(eval_result: dict) -> None:
    W = 130
    np_l3 = eval_result["no_pulse_shock_rate"]
    np_l2 = eval_result["no_pulse_l2_seeds"]
    print()
    print("=" * W)
    print("B1 Phase-Defibrillation Acceptance Table")
    print(f"  Baseline (no_pulse): aftershock-L3={np_l3*100:.1f}%  |  B1-baseline L2 seeds: {np_l2}")
    print(f"  All gates use l3_aftershock (post-pulse window [{PULSE_T_START}+dur, {PULSE_T_START}+dur+{TAIL}])")
    print(f"  B1-G1 UPLIFT:    aftershock uplift > {GATE_UPLIFT_MIN*100:.0f}pp vs no_pulse")
    print(f"  B1-G2 AFTERSHOCK: aftershock L3 ≥ {GATE_AFTERSHOCK_MIN*100:.0f}%")
    print(f"  B1-G3 RESCUE:    rescued B1-baseline-L2 seeds ≥ {GATE_RESCUE_MIN}")
    print(f"  (late% = L3 in [{LATE_WINDOW_START},{LATE_WINDOW_START+LATE_WINDOW_TAIL}] — permanence indicator, not a gate)")
    print("=" * W)
    print(f"  {'Δγ':>5}  {'dur':>5}  {'shock%':>7}  {'uplift':>8}  G1   {'shock%':>7}  G2  "
          f"{'rescued':>9}  G3  {'late%':>6}  {'PASS':>6}")
    print("-" * W)

    golden = []
    for (dg, dur), v in sorted(eval_result["combos"].items()):
        g1s = "✓" if v["g1_uplift"] else "✗"
        g2s = "✓" if v["g2_aftershock"] else "✗"
        g3s = "✓" if v["g3_rescue"] else "✗"
        mark = "★ YES" if v["passes"] else "   no"
        if v["passes"]:
            golden.append((dg, dur))
        print(f"  {dg:>5.2f}  {dur:>5d}  {v['pulse_shock_rate']*100:>6.1f}%  "
              f"{v['uplift_pp']:>+7.1f}pp  {g1s}   {v['aftershock_rate']*100:>6.1f}%  "
              f"{g2s}  {str(v['rescued_seeds']):>9}  {g3s}  {v.get('pulse_late_rate',0)*100:>5.0f}%  {mark}")
    print("=" * W)

    if golden:
        print(f"\n★ Golden Pulse(s): {golden}")
        best_dg, best_dur = min(golden, key=lambda x: x[0])
        print(f"  Recommendation: Δγ={best_dg:.2f}, duration={best_dur}")
    else:
        best = max(eval_result["combos"].items(),
                   key=lambda kv: (kv[1]["g1_uplift"], kv[1]["uplift_pp"]))
        print(f"\n✗ No combination passed all 3 B1 gates.")
        v = best[1]
        print(f"  Best: Δγ={v['delta_gamma']:.2f}, dur={v['duration']}, "
              f"L3={v['pulse_l3_rate']*100:.1f}% ({v['uplift_pp']:+.1f}pp), "
              f"aftershock={v['aftershock_rate']*100:.1f}%")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="B1 Phase-Defibrillation Pulse Scan")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 seeds, Δγ ∈ {0.10, 0.20}")
    args = parser.parse_args()

    seeds = QUICK_SEEDS if args.quick else FULL_SEEDS
    delta_gammas = QUICK_DELTA_GAMMAS if args.quick else FULL_DELTA_GAMMAS
    durations = QUICK_DURATIONS if args.quick else FULL_DURATIONS
    rounds = QUICK_ROUNDS if args.quick else FULL_ROUNDS
    out_dir = ROOT / "outputs" / ("b1_pulse_quick" if args.quick else "b1_pulse_scan")
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions: list[tuple[float, int]] = [(0.0, 0)]   # no_pulse always first
    for dg in delta_gammas:
        for dur in durations:
            conditions.append((dg, dur))

    n_total = len(seeds) * len(conditions)
    mode_str = "QUICK" if args.quick else "FULL"
    print(f"B1 Phase-Defibrillation Pulse Scan  [{mode_str}]")
    print(f"  Golden point   : μ={MU_BASE}, λ_μ={LAMBDA_MU}, SS={SELECTION_STRENGTH}, "
          f"γ_base={GAMMA_BASE}, power={SYNERGY_POWER}")
    print(f"  Pulse T_start  : {PULSE_T_START}  (burn-in tail; aftershock = analysis window)")
    print(f"  Δγ values      : {delta_gammas}")
    print(f"  durations      : {durations}")
    print(f"  seeds          : {len(seeds)}")
    print(f"  total runs     : {n_total}")
    print(f"  output dir     : {out_dir}")
    print()

    all_results: list[RunMetric] = []
    idx = 0
    for seed in seeds:
        for dg, dur in conditions:
            idx += 1
            r = run_one(seed=seed, delta_gamma=dg, duration=dur,
                        out_dir=out_dir, rounds=rounds)
            l3_mark = " L3" if r.l3_pass else ""
            shock_mark = " shock✓" if r.l3_aftershock else ""
            late_mark = " late✓" if r.l3_late else ""
            print(f"[{idx:>4}/{n_total}] Δγ={dg:.2f} dur={dur:4d} seed={seed:4d}  "
                  f"L{r.cycle_level}  v={r.phase_velocity:.5f}"
                  f"{l3_mark}{shock_mark}{late_mark}")
            all_results.append(r)

    eval_result = _evaluate_b1(all_results)
    _print_b1_table(eval_result)

    # Serialize summary
    summary = {
        "experiment": "B1 phase-defibrillation pulse scan",
        "mode": mode_str.lower(),
        "config": {
            "mu_base": MU_BASE,
            "lambda_mu": LAMBDA_MU,
            "lambda_k": LAMBDA_K,
            "selection_strength": SELECTION_STRENGTH,
            "gamma_base": GAMMA_BASE,
            "synergy_power": SYNERGY_POWER,
            "pulse_t_start": PULSE_T_START,
            "seeds": seeds,
            "delta_gammas": delta_gammas,
            "durations": durations,
            "rounds": rounds,
            "burn_in": BURN_IN,
            "tail": TAIL,
        },
        "acceptance_gates": {
            "B1G1_uplift_min_pp": GATE_UPLIFT_MIN * 100,
            "B1G2_aftershock_min": GATE_AFTERSHOCK_MIN,
            "B1G3_rescue_min": GATE_RESCUE_MIN,
        },
        "h73_l2_seeds": sorted(H73_L2_SEEDS),
        "no_pulse_shock_rate": eval_result["no_pulse_shock_rate"],
        "no_pulse_l2_seeds": eval_result["no_pulse_l2_seeds"],
        "no_pulse_l3_seeds": eval_result["no_pulse_l3_seeds"],
        "combos": [
            {**v, "rescued_seeds": v["rescued_seeds"]}
            for v in eval_result["combos"].values()
        ],
        "runs": [
            {
                "seed": r.seed,
                "delta_gamma": r.delta_gamma,
                "duration": r.duration,
                "condition": r.condition_label,
                "cycle_level": r.cycle_level,
                "phase_velocity": r.phase_velocity,
                "l3_pass": r.l3_pass,
                "aftershock_level": r.aftershock_level,
                "l3_aftershock": r.l3_aftershock,
                "late_level": r.late_level,
                "l3_late": r.l3_late,
                "csv_path": r.csv_path,
            }
            for r in all_results
        ],
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
