#!/usr/bin/env python3
"""H7.5 Path Dependency Test

目的：驗證脈衝加速進入 L3 的種子（晚期轉型型）是否比自然到達 L3 的種子
（深穩型）更脆弱——以 random_9persona 全員噪聲環境作為壓力探針。

────────────────────────────────────────────────────────────────────────────
4-Condition Design（固定 Golden Pulse：Δγ=0.05, dur=1000, t_start=1500）
────────────────────────────────────────────────────────────────────────────
  C1  control_none   + no_pulse   → baseline（與 H7.4 no_pulse 相同）
  C2  control_none   + pulse      → H7.4 golden result（複現確認）
  C3  random_9persona + no_pulse  → 噪聲基準
  C4  random_9persona + pulse     → 核心測試：脆弱路徑假說

  random_9persona：所有 300 名玩家的 9 個人格維度於初始化時設為
  uniform(-0.4, 0.4)，以 rng = random.Random(seed*10000 + player_idx)
  確保可重現性。

────────────────────────────────────────────────────────────────────────────
Basin Category Reference（H7.4 no_pulse 分類）
────────────────────────────────────────────────────────────────────────────
  DEEP_STABLE    = {95, 97}           ← 所有窗口均 L3
  EARLY_TRANS    = {49, 91, 93}       ← std+shock L3，late 衰退
  LATE_TRANS     = {47, 51, 53, 55, 99}  ← late-only；47/51/53 可被脈衝加速
  PERMA_L2       = {45, 123}          ← 所有窗口 L2

────────────────────────────────────────────────────────────────────────────
3-Window Metrics
────────────────────────────────────────────────────────────────────────────
  Standard    [2000, 4000]  — 含脈衝（僅供參考）
  Post-pulse  [2500, 3500]  — 脈衝結束後立即的 1000 輪回穩期（相位重置偵測）
  Late        [4000, 6000]  — 主要評判窗口（permanence）

────────────────────────────────────────────────────────────────────────────
驗收條件（H7.5 3-gate）
────────────────────────────────────────────────────────────────────────────
  H7.5-G1  PULSE_UPLIFT       C4 late% > C3 late%           (脈衝有長效效益)
  H7.5-G2  DEEP_STABLE_ROBUST seeds {95,97} 在 C4 均 l3_late (深穩盆地抗噪)
  H7.5-G3  PATH_SIGNAL        Δ_pulse_late(late-trans) < Δ_pulse_late(deep-stable)

  PASS    = G2 ∧ G3  (路徑依賴確認)
  PARTIAL = G1 ∧ G2  (脈衝有效 + 深穩抗噪，但路徑獨立)
  FAIL    = ¬G2       (噪聲過強，需重新校準探針)

Usage:
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u scripts/run_w41_path_dependency.py
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u scripts/run_w41_path_dependency.py --quick
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.cycle_metrics import classify_cycle_level, phase_rotation_r2
from players.base_player import DEFAULT_PERSONALITY_KEYS
from simulation.run_simulation import SimConfig, _write_timeseries_csv, simulate

# ──────────────────────────────────────────────────────────────────────────────
# Fixed golden-point boundaries (from H7.3 / H7.4)
# ──────────────────────────────────────────────────────────────────────────────
MU_BASE = 0.30
LAMBDA_MU = 0.05
LAMBDA_K = 0.20
SELECTION_STRENGTH = 0.15
K_LOWER = 0.05
K_UPPER = 0.25
GAMMA_BASE = 0.16
SYNERGY_POWER = 3.2

# Golden pulse (minimum effective dose from H7.4)
PULSE_T_START = 1500
PULSE_DELTA_GAMMA = 0.05
PULSE_DURATION = 1000

# Analysis windows
FULL_ROUNDS = 6000
BURN_IN = 2000
TAIL = 2000
POST_PULSE_START = PULSE_T_START + PULSE_DURATION   # 2500
POST_PULSE_TAIL = 1000                              # [2500, 3500]
LATE_WINDOW_START = 4000                            # [4000, 6000]
LATE_WINDOW_TAIL = 2000

# Seeds
FULL_SEEDS = [45, 47, 49, 51, 53, 55, 91, 93, 95, 97, 99, 123]
QUICK_SEEDS = [45, 47, 49, 51, 95, 97]  # covers all 4 basin categories

# Basin category reference (H7.4 no_pulse classification)
DEEP_STABLE_SEEDS = {95, 97}
LATE_TRANS_SEEDS_PULSE_BENEFIT = {47, 51, 53}   # gained shock-L3 from H7.4 pulse
PERMA_L2_SEEDS = {45, 123}

# ──────────────────────────────────────────────────────────────────────────────
# Gate thresholds
# ──────────────────────────────────────────────────────────────────────────────
# H7.5-G1: pulse provides net long-run uplift even under noise
GATE_G1_LATE_UPLIFT: float = 0.0   # C4 late% strictly > C3 late%
# H7.5-G2: deep-stable seeds maintain L3 under noise (binary check on {95,97})
GATE_G2_DEEP_STABLE_SEEDS = DEEP_STABLE_SEEDS  # all must pass l3_late in C4
# H7.5-G3: differential pulse effect (late-trans group gets less benefit than deep-stable)
# Satisfied when Δ_pulse_late(late-trans) < Δ_pulse_late(deep-stable)
# (could be negative vs positive, or both positive but smaller)


# ──────────────────────────────────────────────────────────────────────────────
# Persona setup helpers
# ──────────────────────────────────────────────────────────────────────────────
def _random_9persona_setup(seed: int) -> Callable:
    """Return a player_setup_callback that randomizes all player personalities.

    Uses rng = random.Random(seed * 10000 + player_idx) for reproducibility.
    Each personality key is set to uniform(-0.4, 0.4).
    """
    def cb(players: list, _strategy_space: list[str], _cfg: SimConfig) -> None:
        for idx, player in enumerate(players):
            rng = random.Random(int(seed) * 10000 + idx)
            for key in DEFAULT_PERSONALITY_KEYS:
                if hasattr(player, "personality") and key in player.personality:
                    player.personality[key] = rng.uniform(-0.4, 0.4)

    return cb


# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RunMetric:
    seed: int
    persona_condition: str   # "control_none" or "random_9persona"
    pulse: bool              # True = pulse(0.05, 1000)
    condition_label: str     # C1 / C2 / C3 / C4
    csv_path: str
    # Standard window [2000, 4000]
    std_level: int
    std_vel: float
    l3_std: bool
    # Post-pulse stabilization window [2500, 3500]
    post_level: int
    l3_post: bool
    # Late window [4000, 6000] — primary evaluation
    late_level: int
    l3_late: bool


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _extract_series(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {
        "aggressive": [float(row.get("p_aggressive", 0.0)) for row in rows],
        "defensive":  [float(row.get("p_defensive", 0.0)) for row in rows],
        "balanced":   [float(row.get("p_balanced", 0.0)) for row in rows],
    }


def _classify_window(rows: list[dict[str, Any]], start: int, tail: int) -> tuple[int, float]:
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
    persona_condition: str,   # "control_none" or "random_9persona"
    use_pulse: bool,
    out_dir: Path,
    resume: bool = False,
) -> RunMetric:
    pulse_str = "pulse" if use_pulse else "no_pulse"
    condition_label = {
        ("control_none",    False): "C1",
        ("control_none",    True):  "C2",
        ("random_9persona", False): "C3",
        ("random_9persona", True):  "C4",
    }[(persona_condition, use_pulse)]

    tag_dir = out_dir / f"{condition_label}_{persona_condition}_{pulse_str}"
    tag_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tag_dir / f"seed{seed}.csv"

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
        # Pulse fields (H7.4 golden dose)
        synergy_pulse_t_start=PULSE_T_START if use_pulse else None,
        synergy_pulse_duration=PULSE_DURATION if use_pulse else None,
        synergy_pulse_delta_gamma=PULSE_DELTA_GAMMA if use_pulse else 0.0,
    )

    # Resume: re-read existing CSV and skip simulation
    if resume and out_csv.exists():
        import csv as _csv
        with out_csv.open(newline="") as f:
            rows = list(_csv.DictReader(f))
    else:
        persona_cb = _random_9persona_setup(seed) if persona_condition == "random_9persona" else None
        _strategy_space, rows = simulate(cfg, player_setup_callback=persona_cb)
        _write_timeseries_csv(out_csv, strategy_space=_strategy_space, rows=rows)

    std_level, std_vel = _classify_window(rows, BURN_IN, TAIL)
    post_level, _ = _classify_window(rows, POST_PULSE_START, POST_PULSE_TAIL)
    late_level, _ = _classify_window(rows, LATE_WINDOW_START, LATE_WINDOW_TAIL)

    return RunMetric(
        seed=int(seed),
        persona_condition=persona_condition,
        pulse=use_pulse,
        condition_label=condition_label,
        csv_path=str(out_csv.relative_to(ROOT)) if out_csv.is_relative_to(ROOT) else str(out_csv),
        std_level=std_level,
        std_vel=std_vel,
        l3_std=std_level >= 3,
        post_level=post_level,
        l3_post=post_level >= 3,
        late_level=late_level,
        l3_late=late_level >= 3,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def _group_late_rate(results: list[RunMetric], persona: str, pulse: bool) -> float:
    subset = [r for r in results if r.persona_condition == persona and r.pulse == pulse]
    return sum(1 for r in subset if r.l3_late) / len(subset) if subset else 0.0


def _group_late_rate_for_seeds(
    results: list[RunMetric], persona: str, pulse: bool, seeds: set[int]
) -> float:
    subset = [r for r in results if r.persona_condition == persona
              and r.pulse == pulse and r.seed in seeds]
    return sum(1 for r in subset if r.l3_late) / len(subset) if subset else float("nan")


def _evaluate_h75(all_results: list[RunMetric]) -> dict:
    # Overall late rates for each condition
    c1_late = _group_late_rate(all_results, "control_none",    False)
    c2_late = _group_late_rate(all_results, "control_none",    True)
    c3_late = _group_late_rate(all_results, "random_9persona", False)
    c4_late = _group_late_rate(all_results, "random_9persona", True)

    delta_pulse_control = c2_late - c1_late    # pulse benefit without noise
    delta_pulse_noise   = c4_late - c3_late    # pulse benefit under noise

    # Per-group late rates
    for_seeds = _group_late_rate_for_seeds
    deep_c2 = for_seeds(all_results, "control_none",    True,  DEEP_STABLE_SEEDS)
    deep_c4 = for_seeds(all_results, "random_9persona", True,  DEEP_STABLE_SEEDS)
    late_c2 = for_seeds(all_results, "control_none",    True,  LATE_TRANS_SEEDS_PULSE_BENEFIT)
    late_c4 = for_seeds(all_results, "random_9persona", True,  LATE_TRANS_SEEDS_PULSE_BENEFIT)
    late_c1 = for_seeds(all_results, "control_none",    False, LATE_TRANS_SEEDS_PULSE_BENEFIT)
    late_c3 = for_seeds(all_results, "random_9persona", False, LATE_TRANS_SEEDS_PULSE_BENEFIT)
    deep_c1 = for_seeds(all_results, "control_none",    False, DEEP_STABLE_SEEDS)
    deep_c3 = for_seeds(all_results, "random_9persona", False, DEEP_STABLE_SEEDS)

    delta_pulse_late_trans   = late_c4 - late_c3
    delta_pulse_deep_stable  = deep_c4 - deep_c3

    # Gate evaluation
    g1 = delta_pulse_noise > GATE_G1_LATE_UPLIFT   # pulse helps in noise environment
    # G2: all deep-stable seeds pass l3_late in C4
    c4_results = {r.seed: r for r in all_results if r.condition_label == "C4"}
    g2 = all(
        c4_results[s].l3_late
        for s in GATE_G2_DEEP_STABLE_SEEDS
        if s in c4_results
    )
    # G3: differential effect (late-trans loses more pulse benefit than deep-stable)
    g3 = (not (delta_pulse_late_trans != delta_pulse_late_trans or  # nan guard
               delta_pulse_deep_stable != delta_pulse_deep_stable) and
          delta_pulse_late_trans < delta_pulse_deep_stable)

    if g2 and g3:
        verdict = "PASS (路徑依賴確認)"
    elif g1 and g2:
        verdict = "PARTIAL (脈衝有效 + 深穩抗噪，但路徑獨立)"
    elif not g2:
        verdict = "FAIL (噪聲過強，深穩盆地被摧毀)"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "late_rates": {
            "C1_control_no_pulse": c1_late,
            "C2_control_pulse":    c2_late,
            "C3_noise_no_pulse":   c3_late,
            "C4_noise_pulse":      c4_late,
        },
        "delta_pulse_control": delta_pulse_control,
        "delta_pulse_noise":   delta_pulse_noise,
        "per_group": {
            "deep_stable": {
                "c1": deep_c1, "c2": deep_c2, "c3": deep_c3, "c4": deep_c4,
                "delta_control": deep_c2 - deep_c1,
                "delta_noise":   delta_pulse_deep_stable,
            },
            "late_transition_pulse_benefit": {
                "c1": late_c1, "c2": late_c2, "c3": late_c3, "c4": late_c4,
                "delta_control": late_c2 - late_c1,
                "delta_noise":   delta_pulse_late_trans,
            },
        },
        "gates": {
            "g1_pulse_uplift_in_noise": g1,
            "g2_deep_stable_robust_c4": g2,
            "g3_path_signal": g3,
        },
        "verdict": verdict,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Output printing
# ──────────────────────────────────────────────────────────────────────────────
def _print_h75_table(all_results: list[RunMetric], eval_result: dict) -> None:
    W = 120
    print()
    print("=" * W)
    print("H7.5 Path Dependency Test — Results")
    print(f"  Golden Pulse: Δγ={PULSE_DELTA_GAMMA}, dur={PULSE_DURATION}, t_start={PULSE_T_START}")
    print(f"  3-Window: std[{BURN_IN},{BURN_IN+TAIL}] | post[{POST_PULSE_START},"
          f"{POST_PULSE_START+POST_PULSE_TAIL}] | late[{LATE_WINDOW_START},"
          f"{LATE_WINDOW_START+LATE_WINDOW_TAIL}]")
    print("=" * W)

    # Per-seed table (C1/C2/C3/C4 side by side, using late metric)
    by_seed_cond: dict[tuple[int, str], RunMetric] = {}
    for r in all_results:
        by_seed_cond[(r.seed, r.condition_label)] = r

    def lvl(seed: int, cond: str) -> str:
        r = by_seed_cond.get((seed, cond))
        if r is None:
            return "  -  "
        late = "L3" if r.l3_late else "L2"
        post = "✓" if r.l3_post else "·"
        return f"{late}/{post}"

    print()
    print(f"  {'seed':>6}  {'basin_cat':>16}  "
          f"C1(ctrl/no) C2(ctrl/pls) C3(rand/no) C4(rand/pls)")
    print(f"  {'':>6}  {'':>16}  "
          f"[late/post]  [late/post]  [late/post]  [late/post]")
    print("-" * 100)

    cat_map = {}
    for s in FULL_SEEDS:
        if s in DEEP_STABLE_SEEDS:
            cat_map[s] = "deep-stable"
        elif s in LATE_TRANS_SEEDS_PULSE_BENEFIT:
            cat_map[s] = "late-trans(+)"
        elif s in PERMA_L2_SEEDS:
            cat_map[s] = "perma-L2"
        else:
            cat_map[s] = "early-trans"

    seeds_in_results = sorted({r.seed for r in all_results})
    for seed in seeds_in_results:
        cat = cat_map.get(seed, "?")
        print(f"  {seed:>6}  {cat:>16}  "
              f"{lvl(seed,'C1'):>11}  {lvl(seed,'C2'):>11}  "
              f"{lvl(seed,'C3'):>10}  {lvl(seed,'C4'):>10}")

    print("=" * W)
    rates = eval_result["late_rates"]
    print()
    print("  Late% by condition (primary metric):")
    print(f"    C1 control/no_pulse:  {rates['C1_control_no_pulse']*100:.1f}%")
    print(f"    C2 control/pulse:     {rates['C2_control_pulse']*100:.1f}%"
          f"  (Δ_pulse_control = {eval_result['delta_pulse_control']*100:+.1f}pp)")
    print(f"    C3 noise/no_pulse:    {rates['C3_noise_no_pulse']*100:.1f}%")
    print(f"    C4 noise/pulse:       {rates['C4_noise_pulse']*100:.1f}%"
          f"  (Δ_pulse_noise   = {eval_result['delta_pulse_noise']*100:+.1f}pp)")

    print()
    print("  Per-group Δ_pulse_late (C4-C3 vs C2-C1):")
    pg = eval_result["per_group"]
    for grp_name, grp_label in [
        ("deep_stable", "Deep-stable {95,97}"),
        ("late_transition_pulse_benefit", "Late-trans {47,51,53}"),
    ]:
        g = pg[grp_name]
        print(f"    {grp_label:35s}: Δ_control={g['delta_control']*100:+.1f}pp"
              f"  Δ_noise={g['delta_noise']*100:+.1f}pp")

    print()
    gates = eval_result["gates"]
    g1s = "✓" if gates["g1_pulse_uplift_in_noise"] else "✗"
    g2s = "✓" if gates["g2_deep_stable_robust_c4"] else "✗"
    g3s = "✓" if gates["g3_path_signal"] else "✗"
    print(f"  H7.5-G1 PULSE_UPLIFT       {g1s}  "
          f"(C4 late > C3 late: {rates['C4_noise_pulse']*100:.1f}% > {rates['C3_noise_no_pulse']*100:.1f}%)")
    print(f"  H7.5-G2 DEEP_STABLE_ROBUST {g2s}  "
          f"(seeds {{95,97}} l3_late in C4)")
    deep_dn = pg["deep_stable"]["delta_noise"]
    late_dn = pg["late_transition_pulse_benefit"]["delta_noise"]
    print(f"  H7.5-G3 PATH_SIGNAL        {g3s}  "
          f"(Δ_late-trans={late_dn*100:+.1f}pp < Δ_deep-stable={deep_dn*100:+.1f}pp)")
    print()
    print(f"  VERDICT: {eval_result['verdict']}")
    print("=" * W)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="H7.5 Path Dependency Test")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 6 seeds (covers all 4 basin categories)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs whose output CSV already exists")
    args = parser.parse_args()

    seeds = QUICK_SEEDS if args.quick else FULL_SEEDS
    mode_str = "QUICK" if args.quick else "FULL"
    out_dir = ROOT / "outputs" / "h75_path_dependency"
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions: list[tuple[str, bool]] = [
        ("control_none",    False),   # C1
        ("control_none",    True),    # C2
        ("random_9persona", False),   # C3
        ("random_9persona", True),    # C4
    ]
    n_total = len(seeds) * len(conditions)

    print(f"H7.5 Path Dependency Test  [{mode_str}]")
    print(f"  Golden point : μ={MU_BASE}, λ_μ={LAMBDA_MU}, SS={SELECTION_STRENGTH}, "
          f"γ_base={GAMMA_BASE}, power={SYNERGY_POWER}")
    print(f"  Golden pulse : Δγ={PULSE_DELTA_GAMMA}, dur={PULSE_DURATION}, "
          f"t_start={PULSE_T_START}")
    print(f"  Conditions   : C1(ctrl/no)  C2(ctrl/pulse)  "
          f"C3(rand9/no)  C4(rand9/pulse)")
    print(f"  Seeds        : {len(seeds)}")
    print(f"  Total runs   : {n_total}")
    print(f"  Output dir   : {out_dir}")
    print()

    all_results: list[RunMetric] = []
    idx = 0
    for seed in seeds:
        for persona, pulse in conditions:
            idx += 1
            r = run_one(
                seed=seed,
                persona_condition=persona,
                use_pulse=pulse,
                out_dir=out_dir,
                resume=args.resume,
            )
            late_mark = " late✓" if r.l3_late else ""
            post_mark = " post✓" if r.l3_post else ""
            std_mark  = " std✓"  if r.l3_std  else ""
            print(
                f"[{idx:>4}/{n_total}] {r.condition_label}  "
                f"persona={persona:14s}  pulse={str(pulse):5s}  seed={seed:4d}  "
                f"std=L{r.std_level}  post=L{r.post_level}  late=L{r.late_level}"
                f"{std_mark}{post_mark}{late_mark}"
            )
            all_results.append(r)

    eval_result = _evaluate_h75(all_results)
    _print_h75_table(all_results, eval_result)

    # Serialize summary
    summary = {
        "experiment": "H7.5 Path Dependency Test",
        "mode": mode_str.lower(),
        "config": {
            "mu_base": MU_BASE,
            "lambda_mu": LAMBDA_MU,
            "lambda_k": LAMBDA_K,
            "selection_strength": SELECTION_STRENGTH,
            "gamma_base": GAMMA_BASE,
            "synergy_power": SYNERGY_POWER,
            "pulse_t_start": PULSE_T_START,
            "pulse_delta_gamma": PULSE_DELTA_GAMMA,
            "pulse_duration": PULSE_DURATION,
            "seeds": seeds,
            "n_rounds": FULL_ROUNDS,
            "burn_in": BURN_IN,
            "tail": TAIL,
            "post_pulse_window": [POST_PULSE_START, POST_PULSE_START + POST_PULSE_TAIL],
            "late_window": [LATE_WINDOW_START, LATE_WINDOW_START + LATE_WINDOW_TAIL],
        },
        "basin_reference": {
            "deep_stable": sorted(DEEP_STABLE_SEEDS),
            "late_trans_pulse_benefit": sorted(LATE_TRANS_SEEDS_PULSE_BENEFIT),
            "perma_l2": sorted(PERMA_L2_SEEDS),
        },
        "acceptance_gates": {
            "H75G1_pulse_uplift_in_noise": GATE_G1_LATE_UPLIFT,
            "H75G2_deep_stable_seeds": sorted(GATE_G2_DEEP_STABLE_SEEDS),
            "H75G3_delta_late_trans_lt_delta_deep_stable": True,
        },
        "late_rates": eval_result["late_rates"],
        "delta_pulse_control": eval_result["delta_pulse_control"],
        "delta_pulse_noise": eval_result["delta_pulse_noise"],
        "per_group": eval_result["per_group"],
        "gates": eval_result["gates"],
        "verdict": eval_result["verdict"],
        "runs": [
            {
                "seed": r.seed,
                "persona_condition": r.persona_condition,
                "pulse": r.pulse,
                "condition_label": r.condition_label,
                "csv_path": r.csv_path,
                "std_level": r.std_level,
                "l3_std": r.l3_std,
                "post_level": r.post_level,
                "l3_post": r.l3_post,
                "late_level": r.late_level,
                "l3_late": r.l3_late,
            }
            for r in all_results
        ],
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
