"""Step 1: Narrow scale sweep on Seed 91.

Scales: [0.65, 0.75, 0.85, 0.95, 1.0]
For both w3_stackelberg (w3_commit_push) and w3_policy (w3_policy_crossguard).

Output: outputs/actionA_scale_sweep/
"""
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


OUT_DIR = ROOT / "outputs" / "actionA_scale_sweep"
SEED = 91
ROUNDS = 6000
PLAYERS = 300
SELECTION_STRENGTH = 0.10
INIT_BIAS = 0.12
MEMORY_KERNEL = 3
SCALES = [0.65, 0.75, 0.85, 0.95, 1.0]


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


def _classify_effect(base: dict[str, Any], cand: dict[str, Any]) -> str:
    """Classify effect of `cand` scale relative to `base` (scale=1.0)."""
    base_drop = base.get("dropout_round")
    cand_drop = cand.get("dropout_round")
    base_short = int(base["window_eval"]["burn1000_tail1000"]["level"])
    cand_short = int(cand["window_eval"]["burn1000_tail1000"]["level"])

    if float(cand["scale"]) == 1.0:
        return "BASELINE"
    # L3 ignition failure: scale caused L3 to never reach short window
    if base_short >= 3 and cand_short < 3 and cand_drop is None:
        return "L3_SUPPRESSED_EARLY"
    if base_drop is None and cand_drop is None:
        return "NO_DROPOUT_BOTH"
    if base_drop is None and cand_drop is not None:
        return "DROPOUT_INTRODUCED"
    if base_drop is not None and cand_drop is None:
        # Check if short L3 still reached
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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []

    for scale in SCALES:
        print(f"[step1] running w3_stackelberg scale={scale:.2f} seed={SEED} ...", flush=True)
        all_rows.append(_run_b1_stackelberg(scale))
        print(f"[step1] running w3_policy     scale={scale:.2f} seed={SEED} ...", flush=True)
        all_rows.append(_run_b2_policy(scale))

    # Build per-module comparison table (each scale vs 1.0 baseline)
    comparisons: list[dict[str, Any]] = []
    for module in ("w3_stackelberg", "w3_policy"):
        module_rows = [r for r in all_rows if r["module"] == module]
        base = next(r for r in module_rows if float(r["scale"]) == 1.0)
        for r in sorted(module_rows, key=lambda x: float(x["scale"])):
            effect = _classify_effect(base, r)
            comparisons.append(
                {
                    "module": module,
                    "scale": float(r["scale"]),
                    "dropout_round": r["dropout_round"],
                    "short_level": int(r["window_eval"]["burn1000_tail1000"]["level"]),
                    "short_s3": float(r["window_eval"]["burn1000_tail1000"]["stage3_score"]),
                    "long_level": int(r["window_eval"]["burn2000_tail2000"]["level"]),
                    "long_s3": float(r["window_eval"]["burn2000_tail2000"]["stage3_score"]),
                    "effect_vs_baseline": effect,
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
            "scales": SCALES,
        },
        "rows": all_rows,
        "comparisons": comparisons,
    }

    out_json = OUT_DIR / "step1_scale_sweep_summary.json"
    out_md = OUT_DIR / "step1_scale_sweep_onepager.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Step 1 Scale Sweep 一頁摘要")
    lines.append("")
    lines.append(f"- 設定: seed={SEED}, rounds={ROUNDS}, SS={SELECTION_STRENGTH}, MK={MEMORY_KERNEL}")
    lines.append(f"- Scales: {SCALES}")
    lines.append("- Ablation: 每輪將 randomness/stability_seeking/curiosity 乘上 scale")
    lines.append("")

    for module in ("w3_stackelberg", "w3_policy"):
        lines.append(f"## {module}")
        lines.append("")
        lines.append("| scale | dropout_round | short_L | short_s3 | long_L | long_s3 | effect_vs_1.0 |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for c in [x for x in comparisons if x["module"] == module]:
            dropout_str = str(c["dropout_round"]) if c["dropout_round"] is not None else "None"
            lines.append(
                f"| {c['scale']:.2f} | {dropout_str} | {c['short_level']}"
                f" | {c['short_s3']:.3f} | {c['long_level']} | {c['long_s3']:.3f}"
                f" | {c['effect_vs_baseline']} |"
            )
        lines.append("")

    lines.append("## 判讀指引")
    lines.append("- **SHIFTED_TO_3000_PLUS** / **DROPOUT_REMOVED** → 甜點區，z_exploring 是掉隊牆主因")
    lines.append("- **SHIFTED_LATER** → 部分延長，scale 接近甜點")
    lines.append("- **L3_SUPPRESSED_EARLY** → 低於啟動門檻，scale 太小")
    lines.append("- **NO_SHIFT / SHIFTED_EARLIER** → 結構性不穩，scale 無效")
    lines.append("")
    lines.append("## 預期壽命/強度反比曲線")
    lines.append("若 scale ↓ 導致 short_s3 ↓ 但 dropout_round ↑，則存在反比曲線，")
    lines.append("表示系統確實在「啟動門檻」與「相干壽命」之間存在 Energy Gap Trade-off。")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nwritten: {out_json.relative_to(ROOT)}")
    print(f"written: {out_md.relative_to(ROOT)}")
    print()
    print("=== Step 1 Scale Sweep 結果 ===")
    for c in comparisons:
        print(
            f"  {c['module']:20s} scale={c['scale']:.2f}  dropout={str(c['dropout_round']):>6s}"
            f"  short_L={c['short_level']}  long_L={c['long_level']}"
            f"  effect={c['effect_vs_baseline']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
