from __future__ import annotations

import json
import sys
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


OUT_DIR = ROOT / "outputs" / "actionA_trait_ablation"
SEED = 91
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


def _run_b1_stackelberg(scale: float) -> dict[str, Any]:
    cond_dir = OUT_DIR / "w3_stackelberg" / f"scale_{scale:.2f}"
    cond_dir.mkdir(parents=True, exist_ok=True)
    out_csv = cond_dir / f"seed{SEED}.csv"

    cfg = SimConfig(
        n_players=PLAYERS,
        n_rounds=ROUNDS,
        seed=SEED,
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
        "seed": SEED,
        "csv": str(out_csv.relative_to(ROOT)),
        "window_eval": _window_eval(series3),
        "dropout_round": _rolling_dropout_round(series3),
    }


def _run_b2_policy(scale: float) -> dict[str, Any]:
    cond_dir = OUT_DIR / "w3_policy" / f"scale_{scale:.2f}"
    cond_dir.mkdir(parents=True, exist_ok=True)
    out_csv = cond_dir / f"seed{SEED}.csv"
    policy_steps = cond_dir / f"w3_policy_steps_seed{SEED}.tsv"

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
    adapter = W3PolicyAdapter(config=cell, seed=SEED)

    cfg = SimConfig(
        n_players=PLAYERS,
        n_rounds=ROUNDS,
        seed=SEED,
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
        "seed": SEED,
        "csv": str(out_csv.relative_to(ROOT)),
        "policy_steps_tsv": str(policy_steps.relative_to(ROOT)),
        "window_eval": _window_eval(series3),
        "dropout_round": _rolling_dropout_round(series3),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scales = [1.0, 0.5]
    rows: list[dict[str, Any]] = []

    for scale in scales:
        rows.append(_run_b1_stackelberg(scale))
        rows.append(_run_b2_policy(scale))

    by_key = {(r["module"], float(r["scale"])): r for r in rows}
    comparisons: list[dict[str, Any]] = []
    for module in ("w3_stackelberg", "w3_policy"):
        base = by_key[(module, 1.0)]
        half = by_key[(module, 0.5)]
        base_drop = base.get("dropout_round")
        half_drop = half.get("dropout_round")
        base_short = int(base["window_eval"]["burn1000_tail1000"]["level"])
        half_short = int(half["window_eval"]["burn1000_tail1000"]["level"])
        if base_short >= 3 and half_short < 3 and half_drop is None:
            effect = "L3_SUPPRESSED_EARLY"
        elif base_drop is None and half_drop is None:
            effect = "NO_DROPOUT_BOTH"
        elif base_drop is None and half_drop is not None:
            effect = "DROPOUT_INTRODUCED"
        elif base_drop is not None and half_drop is None:
            effect = "DROPOUT_REMOVED"
        else:
            assert isinstance(base_drop, int) and isinstance(half_drop, int)
            if half_drop >= base_drop + 1000:
                effect = "SHIFTED_TO_3000_PLUS"
            elif half_drop > base_drop:
                effect = "SHIFTED_LATER"
            elif half_drop == base_drop:
                effect = "NO_SHIFT"
            else:
                effect = "SHIFTED_EARLIER"
        comparisons.append(
            {
                "module": module,
                "dropout_round_scale1.0": base_drop,
                "dropout_round_scale0.5": half_drop,
                "effect": effect,
                "short_level_scale1.0": base_short,
                "short_level_scale0.5": half_short,
                "long_level_scale1.0": int(base["window_eval"]["burn2000_tail2000"]["level"]),
                "long_level_scale0.5": int(half["window_eval"]["burn2000_tail2000"]["level"]),
            }
        )

    payload = {
        "generated_at": datetime.now().isoformat(),
        "scope": {
            "seed": SEED,
            "rounds": ROUNDS,
            "players": PLAYERS,
            "selection_strength": SELECTION_STRENGTH,
            "memory_kernel": MEMORY_KERNEL,
            "trait_ablation": "scale randomness/stability_seeking/curiosity each round",
            "scales": scales,
        },
        "rows": rows,
        "comparisons": comparisons,
    }

    out_json = OUT_DIR / "actionA_trait_ablation_summary.json"
    out_md = OUT_DIR / "actionA_trait_ablation_onepager.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Action A Trait Ablation 一頁摘要")
    lines.append("")
    lines.append("- 設定: seed=91, rounds=6000, SS=0.10, MK=3")
    lines.append("- Ablation: 每輪將 randomness/stability_seeking/curiosity 乘上 scale")
    lines.append("")
    lines.append("| module | dropout@scale1.0 | dropout@scale0.5 | effect | short L(1.0→0.5) | long L(1.0→0.5) |")
    lines.append("| --- | ---: | ---: | --- | ---: | ---: |")
    for c in comparisons:
        lines.append(
            "| {m} | {d1} | {d2} | {e} | {s1}->{s2} | {l1}->{l2} |".format(
                m=c["module"],
                d1=c["dropout_round_scale1.0"],
                d2=c["dropout_round_scale0.5"],
                e=c["effect"],
                s1=c["short_level_scale1.0"],
                s2=c["short_level_scale0.5"],
                l1=c["long_level_scale1.0"],
                l2=c["long_level_scale0.5"],
            )
        )
    lines.append("")
    lines.append("## 判讀")
    lines.append("- 若 effect=SHIFTED_TO_3000_PLUS 或 DROPOUT_REMOVED，支持 z_exploring 隨機注入是掉隊牆主因。")
    lines.append("- 若 effect=L3_SUPPRESSED_EARLY，代表消去法把短期 L3 也一起壓掉，不能解讀成結構穩定化。")
    lines.append("- 若 effect=NO_SHIFT/SHIFTED_EARLIER，表示掉隊更偏向結構性不穩，需考慮 metrics/模型層修正。")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"written: {out_json.relative_to(ROOT)}")
    print(f"written: {out_md.relative_to(ROOT)}")
    for c in comparisons:
        print(
            "module={m} d1={d1} d2={d2} effect={e}".format(
                m=c["module"],
                d1=c["dropout_round_scale1.0"],
                d2=c["dropout_round_scale0.5"],
                e=c["effect"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
