"""Step 2: Group test — scale=0.85 vs scale=1.0 on all 12 seeds.

Seeds: [45,47,49,51,53,55,91,93,95,97,99,101]
Modules: w3_stackelberg (w3_commit_push) + w3_policy (w3_policy_crossguard)
Scale tested: 0.85 and 1.0 baseline.

Output: outputs/actionA_group_scale085/
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
from simulation.w3_policy import W3PolicyAdapter, _write_tsv as write_tsv, POLICY_STEP_FIELDNAMES, W3PolicyCellConfig


OUT_DIR = ROOT / "outputs" / "actionA_group_scale085"
SEEDS = [45, 47, 49, 51, 53, 55, 91, 93, 95, 97, 99, 101]
SCALES = [1.0, 0.85]
ROUNDS = 6000
PLAYERS = 300
SELECTION_STRENGTH = 0.10
INIT_BIAS = 0.12
MEMORY_KERNEL = 3


@dataclass
class RollingPoint:
    round_end: int
    level: int
    passed: bool


def _scale_exploring_traits(players: list[object], scale: float) -> None:
    for pl in players:
        personality = getattr(pl, "personality", None)
        if not isinstance(personality, dict):
            continue
        for key in ("randomness", "stability_seeking", "curiosity"):
            base = float(personality.get(key, 0.0))
            personality[key] = max(-1.0, min(1.0, base * float(scale)))


def _rolling_dropout_round(series3: Mapping[str, list[float]], *, sustain_windows: int = 5) -> int | None:
    window = 1000
    step = 20
    roll: list[RollingPoint] = []
    n = min(len(series3["aggressive"]), len(series3["defensive"]), len(series3["balanced"]))
    for end in range(window, n + 1, step):
        seg = {
            "aggressive": series3["aggressive"][end - window : end],
            "defensive": series3["defensive"][end - window : end],
            "balanced": series3["balanced"][end - window : end],
        }
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
        roll.append(RollingPoint(round_end=end, level=int(cyc.level), passed=bool(cyc.level >= 3)))

    if not roll:
        return None
    flags = [1 if p.passed else 0 for p in roll]
    try:
        first_pass_idx = flags.index(1)
    except ValueError:
        return None

    k = max(1, int(sustain_windows))
    for i in range(first_pass_idx + 1, max(first_pass_idx + 1, len(flags) - k + 1)):
        if flags[i] == 0 and all(flags[j] == 0 for j in range(i, i + k)):
            return int(roll[i].round_end)
    return None


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


def _extract_series(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {
        "aggressive": [float(r["p_aggressive"]) for r in rows],
        "defensive": [float(r["p_defensive"]) for r in rows],
        "balanced": [float(r["p_balanced"]) for r in rows],
    }


def _run_b1_stackelberg(scale: float, seed: int) -> dict[str, Any]:
    cond_dir = OUT_DIR / "w3_stackelberg" / f"scale_{scale:.2f}"
    cond_dir.mkdir(parents=True, exist_ok=True)
    out_csv = cond_dir / f"seed{seed}.csv"

    cfg = SimConfig(
        n_players=PLAYERS,
        n_rounds=ROUNDS,
        seed=seed,
        payoff_mode="matrix_ab",
        popularity_mode="sampled",
        gamma=0.1,
        epsilon=0.0,
        a=1.05,
        b=0.85,
        matrix_cross_coupling=0.35,
        init_bias=INIT_BIAS,
        evolution_mode="sampled",
        payoff_lag=1,
        selection_strength=SELECTION_STRENGTH,
        enable_events=True,
        events_json=DEFAULT_FULL_EVENTS_JSON,
        out_csv=out_csv,
        memory_kernel=MEMORY_KERNEL,
    )

    def _setup(players: list[object], _strategy_space: list[str], _cfg: SimConfig) -> None:
        _scale_exploring_traits(players, scale)

    def _round_cb(
        _round_index: int,
        _cfg: SimConfig,
        players: list[object],
        _dungeon: object,
        _step_records: list[dict[str, object]],
        _ctx: dict[str, Any],
    ) -> None:
        _scale_exploring_traits(players, scale)

    strategy_space, rows = simulate(cfg, player_setup_callback=_setup, round_callback=_round_cb)
    _write_timeseries_csv(out_csv, strategy_space=strategy_space, rows=rows)
    series3 = _extract_series(rows)
    return {
        "module": "w3_stackelberg",
        "condition": "w3_commit_push",
        "scale": scale,
        "seed": seed,
        "csv": str(out_csv.relative_to(ROOT)),
        "window_eval": _window_eval(series3),
        "dropout_round": _rolling_dropout_round(series3),
    }


def _run_b2_policy(scale: float, seed: int) -> dict[str, Any]:
    cond_dir = OUT_DIR / "w3_policy" / f"scale_{scale:.2f}"
    cond_dir.mkdir(parents=True, exist_ok=True)
    out_csv = cond_dir / f"seed{seed}.csv"
    policy_steps = cond_dir / f"w3_policy_steps_seed{seed}.tsv"

    baseline = StackelbergCommitment(
        condition="control_policy",
        leader_action="baseline_policy",
        a=1.00,
        b=0.90,
        matrix_cross_coupling=0.20,
        description="baseline",
    )
    active = StackelbergCommitment(
        condition="w3_policy_crossguard",
        leader_action="cross_guard_policy",
        a=1.00,
        b=0.90,
        matrix_cross_coupling=0.35,
        description="crossguard",
    )

    cell = W3PolicyCellConfig(
        condition="w3_policy_crossguard",
        baseline_commitment=baseline,
        active_commitment=active,
        players=PLAYERS,
        rounds=ROUNDS,
        events_json=DEFAULT_FULL_EVENTS_JSON,
        selection_strength=SELECTION_STRENGTH,
        init_bias=INIT_BIAS,
        memory_kernel=MEMORY_KERNEL,
        burn_in=1000,
        tail=1000,
        policy_update_interval=150,
        theta_low=0.08,
        theta_high=0.12,
        out_dir=cond_dir,
    )
    adapter = W3PolicyAdapter(config=cell, seed=seed)

    cfg = SimConfig(
        n_players=PLAYERS,
        n_rounds=ROUNDS,
        seed=seed,
        payoff_mode="matrix_ab",
        popularity_mode="sampled",
        gamma=0.1,
        epsilon=0.0,
        a=float(baseline.a),
        b=float(baseline.b),
        matrix_cross_coupling=float(baseline.matrix_cross_coupling),
        init_bias=INIT_BIAS,
        evolution_mode="sampled",
        payoff_lag=1,
        selection_strength=SELECTION_STRENGTH,
        enable_events=True,
        events_json=DEFAULT_FULL_EVENTS_JSON,
        out_csv=out_csv,
        memory_kernel=MEMORY_KERNEL,
    )

    def _setup(players: list[object], _strategy_space: list[str], _cfg: SimConfig) -> None:
        _scale_exploring_traits(players, scale)

    def _round_cb(
        round_index: int,
        cfg_now: SimConfig,
        players: list[object],
        dungeon: object,
        step_records: list[dict[str, object]],
        ctx: dict[str, Any],
    ) -> None:
        adapter(round_index, cfg_now, players, dungeon, step_records, ctx)
        _scale_exploring_traits(players, scale)

    strategy_space, rows = simulate(cfg, player_setup_callback=_setup, round_callback=_round_cb)
    adapter.flush_final()
    _write_timeseries_csv(out_csv, strategy_space=strategy_space, rows=rows)
    write_tsv(policy_steps, fieldnames=POLICY_STEP_FIELDNAMES, rows=adapter.policy_rows)
    series3 = _extract_series(rows)
    return {
        "module": "w3_policy",
        "condition": "w3_policy_crossguard",
        "scale": scale,
        "seed": seed,
        "csv": str(out_csv.relative_to(ROOT)),
        "policy_steps_tsv": str(policy_steps.relative_to(ROOT)),
        "window_eval": _window_eval(series3),
        "dropout_round": _rolling_dropout_round(series3),
    }


def _classify_effect(base: dict[str, Any], cand: dict[str, Any]) -> str:
    base_drop = base.get("dropout_round")
    cand_drop = cand.get("dropout_round")
    base_short = int(base["window_eval"]["burn1000_tail1000"]["level"])
    cand_short = int(cand["window_eval"]["burn1000_tail1000"]["level"])

    if float(cand["scale"]) == 1.0:
        return "BASELINE"
    if base_short >= 3 and cand_short < 3 and cand_drop is None:
        return "L3_SUPPRESSED_EARLY"
    if base_drop is None and cand_drop is None:
        return "NO_DROPOUT_BOTH"
    if base_drop is None and cand_drop is not None:
        return "DROPOUT_INTRODUCED"
    if base_drop is not None and cand_drop is None:
        if cand_short >= 3:
            return "DROPOUT_REMOVED"
        return "L3_SUPPRESSED_EARLY"
    assert isinstance(base_drop, int) and isinstance(cand_drop, int)
    if cand_drop >= base_drop + 1000:
        return "SHIFTED_TO_3000_PLUS"
    if cand_drop > base_drop:
        return "SHIFTED_LATER"
    if cand_drop == base_drop:
        return "NO_SHIFT"
    return "SHIFTED_EARLIER"


def _run_task(module: str, scale: float, seed: int) -> dict[str, Any]:
    if module == "w3_stackelberg":
        return _run_b1_stackelberg(scale, seed)
    if module == "w3_policy":
        return _run_b2_policy(scale, seed)
    raise ValueError(f"Unknown module: {module}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []

    tasks: list[tuple[str, float, int]] = []
    for scale in SCALES:
        for seed in SEEDS:
            tasks.append(("w3_stackelberg", scale, seed))
            tasks.append(("w3_policy", scale, seed))

    total = len(tasks)
    max_workers = min(8, max(1, (os.cpu_count() or 2) // 2))
    print(f"[step2] parallel workers={max_workers}, total_tasks={total}", flush=True)

    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_run_task, module, scale, seed): (module, scale, seed)
            for module, scale, seed in tasks
        }
        for future in as_completed(future_to_task):
            module, scale, seed = future_to_task[future]
            row = future.result()
            all_rows.append(row)
            done += 1
            print(f"[step2 {done}/{total}] done {module} scale={scale:.2f} seed={seed}", flush=True)

    # Per-module, per-seed comparison: scale=0.85 vs 1.0
    comparisons: list[dict[str, Any]] = []
    for module in ("w3_stackelberg", "w3_policy"):
        for seed in SEEDS:
            base = next(r for r in all_rows if r["module"] == module and r["seed"] == seed and float(r["scale"]) == 1.0)
            cand = next(r for r in all_rows if r["module"] == module and r["seed"] == seed and float(r["scale"]) == 0.85)
            effect = _classify_effect(base, cand)
            comparisons.append(
                {
                    "module": module,
                    "seed": seed,
                    "dropout_round_1.0": base["dropout_round"],
                    "dropout_round_0.85": cand["dropout_round"],
                    "short_level_1.0": int(base["window_eval"]["burn1000_tail1000"]["level"]),
                    "short_level_0.85": int(cand["window_eval"]["burn1000_tail1000"]["level"]),
                    "long_level_1.0": int(base["window_eval"]["burn2000_tail2000"]["level"]),
                    "long_level_0.85": int(cand["window_eval"]["burn2000_tail2000"]["level"]),
                    "short_s3_1.0": float(base["window_eval"]["burn1000_tail1000"]["stage3_score"]),
                    "short_s3_0.85": float(cand["window_eval"]["burn1000_tail1000"]["stage3_score"]),
                    "effect": effect,
                }
            )

    # Aggregate counts per module
    agg: dict[str, dict[str, int]] = {}
    for c in comparisons:
        m = c["module"]
        if m not in agg:
            agg[m] = {}
        e = c["effect"]
        agg[m][e] = agg[m].get(e, 0) + 1

    payload = {
        "generated_at": datetime.now().isoformat(),
        "scope": {
            "seeds": SEEDS,
            "scales_tested": SCALES,
            "target_scale": 0.85,
            "rounds": ROUNDS,
            "players": PLAYERS,
            "selection_strength": SELECTION_STRENGTH,
            "memory_kernel": MEMORY_KERNEL,
            "trait_ablation": "scale randomness/stability_seeking/curiosity each round",
        },
        "rows": all_rows,
        "comparisons": comparisons,
        "aggregate": agg,
    }

    out_json = OUT_DIR / "step2_group_scale085_summary.json"
    out_md = OUT_DIR / "step2_group_scale085_onepager.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Step 2 Group Scale=0.85 一頁摘要")
    lines.append("")
    lines.append(f"- 設定: seeds={SEEDS}, rounds={ROUNDS}, SS={SELECTION_STRENGTH}, MK={MEMORY_KERNEL}")
    lines.append("- 測試 scale=0.85 vs scale=1.0 基準")
    lines.append("")

    for module in ("w3_stackelberg", "w3_policy"):
        lines.append(f"## {module}")
        lines.append("")
        lines.append("| seed | dropout@1.0 | dropout@0.85 | short_L(1.0→0.85) | long_L(1.0→0.85) | effect |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | --- |")
        for c in [x for x in comparisons if x["module"] == module]:
            d1 = str(c["dropout_round_1.0"]) if c["dropout_round_1.0"] is not None else "None"
            d2 = str(c["dropout_round_0.85"]) if c["dropout_round_0.85"] is not None else "None"
            lines.append(
                f"| {c['seed']} | {d1} | {d2}"
                f" | {c['short_level_1.0']}->{c['short_level_0.85']}"
                f" | {c['long_level_1.0']}->{c['long_level_0.85']}"
                f" | {c['effect']} |"
            )
        # Aggregate
        agg_m = agg.get(module, {})
        lines.append("")
        lines.append("**效果統計：**")
        for effect, count in sorted(agg_m.items(), key=lambda x: -x[1]):
            lines.append(f"- {effect}: {count}/{len(SEEDS)} seeds")
        lines.append("")

    lines.append("## 核心問題")
    lines.append("- Seed 91 的掉隊是個體悲劇，還是所有 12 seeds 的共同物理地平線？")
    lines.append("- 若 >8/12 seeds 出現 SHIFTED_LATER/SHIFTED_TO_3000_PLUS → 物理地平線")
    lines.append("- 若 ≤3/12 seeds 有效 → 個體悲劇，seed 91 敏感性獨特")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nwritten: {out_json.relative_to(ROOT)}")
    print(f"written: {out_md.relative_to(ROOT)}")
    print()
    print("=== Step 2 Group Scale=0.85 結果 ===")
    for module in ("w3_stackelberg", "w3_policy"):
        print(f"\n  [{module}]")
        for c in [x for x in comparisons if x["module"] == module]:
            print(
                f"    seed={c['seed']:3d}  dropout={str(c['dropout_round_1.0']):>6s}->{str(c['dropout_round_0.85']):>6s}"
                f"  short_L={c['short_level_1.0']}->{c['short_level_0.85']}"
                f"  effect={c['effect']}"
            )
        agg_m = agg.get(module, {})
        print(f"  aggregate: {agg_m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
