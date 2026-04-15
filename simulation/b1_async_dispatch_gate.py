from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.cycle_metrics import classify_cycle_level
from analysis.decay_rate import estimate_decay_gamma
from simulation.personality_rl_runtime import PersonalityRLConfig, run_personality_rl


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS_JSON = REPO_ROOT / "docs" / "personality_dungeon_v1" / "02_event_templates_smoke_v1.json"
DEFAULT_BASELINE_JSON = REPO_ROOT / "outputs" / "pers_cal_baseline_gate60_summary.json"
DEFAULT_SMOKE_JSON = REPO_ROOT / "outputs" / "b1_async_dispatch_smoke_summary.json"
DEFAULT_GATE_JSON = REPO_ROOT / "outputs" / "b1_async_dispatch_gate60_summary.json"


@dataclass(frozen=True)
class B1SeedOutcome:
    seed: int
    level: int
    s3: float
    turn: float
    gamma: float
    elapsed_sec: float
    dispatch_fairness_pass: bool
    event_sync_index_mean: float
    reward_perturb_corr: float
    trap_entry_round: int | None


def parse_seeds(spec: str) -> list[int]:
    token = str(spec).strip()
    if not token:
        raise ValueError("seed spec cannot be empty")
    if ".." in token and "," not in token:
        lo_s, hi_s = token.split("..", 1)
        lo = int(lo_s)
        hi = int(hi_s)
        if hi < lo:
            raise ValueError("seed range must be ascending")
        return list(range(lo, hi + 1))
    parts = [part.strip() for part in token.split(",") if part.strip()]
    if not parts:
        raise ValueError("seed spec cannot be empty")
    return [int(part) for part in parts]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    q = max(0.0, min(1.0, float(q)))
    idx = int(round((len(arr) - 1) * q))
    return float(arr[idx])


def build_b1_config(
    *,
    n_players: int,
    n_rounds: int,
    events_json: str,
    event_reward_scale: float,
    event_reward_mode: str,
    event_reward_multiplier_cap: float,
    event_impact_mode: str,
    event_impact_horizon: int,
    event_impact_decay: float,
    event_dispatch_mode: str,
    event_dispatch_target_rate: float,
    event_dispatch_batch_size: int,
    event_dispatch_seed_offset: int,
    event_dispatch_fairness_window: int,
    event_dispatch_fairness_tolerance: float,
    event_warmup_rounds: int,
    event_risk_enabled: bool,
) -> PersonalityRLConfig:
    return PersonalityRLConfig(
        n_players=int(n_players),
        n_rounds=int(n_rounds),
        personality_mode="random",
        lambda_alpha=0.15,
        lambda_beta=0.10,
        lambda_r=0.20,
        lambda_risk=0.20,
        lambda_beta_comp=0.0,
        events_json=str(events_json),
        event_rate=float(event_dispatch_target_rate),
        event_reward_scale=float(event_reward_scale),
        event_reward_mode=str(event_reward_mode),
        event_reward_multiplier_cap=float(event_reward_multiplier_cap),
        event_impact_mode=str(event_impact_mode),
        event_impact_horizon=int(event_impact_horizon),
        event_impact_decay=float(event_impact_decay),
        event_warmup_rounds=int(event_warmup_rounds),
        event_risk_enabled=bool(event_risk_enabled),
        event_dispatch_mode=str(event_dispatch_mode),
        event_dispatch_target_rate=float(event_dispatch_target_rate),
        event_dispatch_batch_size=int(event_dispatch_batch_size),
        event_dispatch_seed_offset=int(event_dispatch_seed_offset),
        event_dispatch_fairness_window=int(event_dispatch_fairness_window),
        event_dispatch_fairness_tolerance=float(event_dispatch_fairness_tolerance),
        world_feedback=False,
    )


def evaluate_seed(
    cfg: PersonalityRLConfig,
    *,
    seed: int,
    burn_in: int,
    tail: int,
) -> B1SeedOutcome:
    t0 = time.time()
    result = run_personality_rl(cfg, seed=seed)
    elapsed = time.time() - t0

    series_map = {
        "aggressive": [float(row["p_aggressive"]) for row in result.rows],
        "defensive": [float(row["p_defensive"]) for row in result.rows],
        "balanced": [float(row["p_balanced"]) for row in result.rows],
    }
    cyc = classify_cycle_level(
        series_map,
        burn_in=int(burn_in),
        tail=int(tail),
        amplitude_threshold=0.02,
        corr_threshold=0.09,
        eta=0.55,
        stage3_method="turning",
        phase_smoothing=1,
        stage2_fallback_r2_threshold=0.85,
        stage2_fallback_min_rotation=20.0,
    )
    fit = estimate_decay_gamma(series_map, series_kind="p")

    diag = result.diagnostics
    return B1SeedOutcome(
        seed=int(seed),
        level=int(cyc.level),
        s3=float(cyc.stage3.score) if cyc.stage3 is not None else 0.0,
        turn=float(cyc.stage3.turn_strength) if cyc.stage3 is not None else 0.0,
        gamma=float(fit.gamma) if fit is not None else 0.0,
        elapsed_sec=float(elapsed),
        dispatch_fairness_pass=bool(diag.get("dispatch_fairness_pass", True)),
        event_sync_index_mean=float(diag.get("event_sync_index_mean", 0.0)),
        reward_perturb_corr=float(diag.get("reward_perturb_corr", 0.0)),
        trap_entry_round=diag.get("trap_entry_round"),
    )


def _load_baseline_map(path: Path, *, healthy_threshold: float) -> dict[int, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    outcomes = data.get("outcomes", [])
    baseline: dict[int, dict[str, Any]] = {}
    for row in outcomes:
        seed = int(row["seed"])
        s3 = float(row["s3"])
        level = int(row["level"])
        baseline[seed] = {
            "level": level,
            "s3": s3,
            "healthy": s3 >= healthy_threshold,
        }
    return baseline


def summarize_stage(
    outcomes: list[B1SeedOutcome],
    *,
    healthy_threshold: float,
    gate_max_l1: int,
    gate_min_healthy: int,
    baseline_map: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    n = len(outcomes)
    l1 = sum(1 for o in outcomes if o.level == 1)
    l2 = sum(1 for o in outcomes if o.level == 2)
    l3 = sum(1 for o in outcomes if o.level == 3)
    healthy = sum(1 for o in outcomes if o.s3 >= healthy_threshold)
    marginal = n - healthy - l1
    fairness_fail_count = sum(1 for o in outcomes if not o.dispatch_fairness_pass)

    s3_values = [o.s3 for o in outcomes]
    gamma_values = [o.gamma for o in outcomes]

    new_l1 = None
    rescued = None
    broke = None
    if baseline_map is not None:
        new_l1 = 0
        rescued = 0
        broke = 0
        for o in outcomes:
            b = baseline_map.get(o.seed)
            if b is None:
                continue
            if b["healthy"] and o.level == 1:
                new_l1 += 1
            if b["healthy"] and o.s3 < healthy_threshold:
                broke += 1
            if (not b["healthy"]) and o.s3 >= healthy_threshold:
                rescued += 1

    l1_pass = l1 <= int(gate_max_l1)
    healthy_pass = healthy >= int(gate_min_healthy)
    fairness_pass = fairness_fail_count == 0
    new_l1_pass = (new_l1 == 0) if new_l1 is not None else False

    return {
        "total_seeds": n,
        "l1": l1,
        "l2": l2,
        "l3": l3,
        "healthy": healthy,
        "marginal": marginal,
        "fairness_fail_count": fairness_fail_count,
        "mean_s3": (sum(s3_values) / n) if n else 0.0,
        "median_s3": _percentile(s3_values, 0.5),
        "p10_s3": _percentile(s3_values, 0.1),
        "mean_gamma": (sum(gamma_values) / n) if n else 0.0,
        "new_l1": new_l1,
        "rescued": rescued,
        "broke": broke,
        "gate": {
            "max_l1": int(gate_max_l1),
            "min_healthy": int(gate_min_healthy),
            "healthy_threshold": float(healthy_threshold),
            "l1_pass": l1_pass,
            "healthy_pass": healthy_pass,
            "fairness_pass": fairness_pass,
            "new_l1_pass": new_l1_pass,
            "overall_pass": bool(l1_pass and healthy_pass and fairness_pass and new_l1_pass),
        },
    }


def run_stage(
    *,
    stage_name: str,
    seeds: list[int],
    cfg: PersonalityRLConfig,
    burn_in: int,
    tail: int,
    healthy_threshold: float,
    gate_max_l1: int,
    gate_min_healthy: int,
    baseline_map: dict[int, dict[str, Any]] | None = None,
) -> tuple[list[B1SeedOutcome], dict[str, Any]]:
    outcomes: list[B1SeedOutcome] = []
    total = len(seeds)
    print("=" * 80)
    print(f"{stage_name}")
    print(f"Seeds: {seeds[0]}..{seeds[-1]}  (N={len(seeds)})")
    print("=" * 80)

    for idx, seed in enumerate(seeds, start=1):
        item = evaluate_seed(cfg, seed=seed, burn_in=burn_in, tail=tail)
        outcomes.append(item)
        status = "OK" if item.s3 >= healthy_threshold else ("L1" if item.level == 1 else "MARGINAL")
        fair = "F" if item.dispatch_fairness_pass else "FAIR_FAIL"
        print(
            f"[{idx:2d}/{total}] seed={seed:3d}  L{item.level}  "
            f"s3={item.s3:.4f}  turn={item.turn:.4f}  gamma={item.gamma:.6f}  "
            f"{status}  {fair}  ({item.elapsed_sec:.1f}s)"
        )

    summary = summarize_stage(
        outcomes,
        healthy_threshold=healthy_threshold,
        gate_max_l1=gate_max_l1,
        gate_min_healthy=gate_min_healthy,
        baseline_map=baseline_map,
    )
    return outcomes, summary


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B1 async-dispatch protocol: smoke -> gate60")
    parser.add_argument("--smoke-seeds", default="42,44,45,67,73,90")
    parser.add_argument("--gate-seeds", default="42..101")
    parser.add_argument("--baseline-summary-json", default=str(DEFAULT_BASELINE_JSON))
    parser.add_argument("--events-json", default=str(DEFAULT_EVENTS_JSON))
    parser.add_argument("--n-players", type=int, default=300)
    parser.add_argument("--n-rounds", type=int, default=12000)
    parser.add_argument("--burn-in", type=int, default=4000)
    parser.add_argument("--tail", type=int, default=4000)
    parser.add_argument("--healthy-threshold", type=float, default=0.80)
    parser.add_argument("--gate-max-l1", type=int, default=3)
    parser.add_argument("--gate-min-healthy", type=int, default=42)
    parser.add_argument("--event-reward-scale", type=float, default=0.01)
    parser.add_argument(
        "--event-reward-mode",
        default="additive",
        choices=["additive", "multiplicative"],
    )
    parser.add_argument("--event-reward-multiplier-cap", type=float, default=0.25)
    parser.add_argument(
        "--event-impact-mode",
        default="instant",
        choices=["instant", "spread"],
    )
    parser.add_argument("--event-impact-horizon", type=int, default=1)
    parser.add_argument("--event-impact-decay", type=float, default=0.70)
    parser.add_argument("--event-warmup-rounds", type=int, default=0)
    parser.add_argument("--event-risk-enabled", action="store_true", default=False)
    parser.add_argument(
        "--event-dispatch-mode",
        default="async_poisson",
        choices=["sync", "async_round_robin", "async_poisson"],
    )
    parser.add_argument("--event-dispatch-target-rate", type=float, default=0.08)
    parser.add_argument("--event-dispatch-batch-size", type=int, default=0)
    parser.add_argument("--event-dispatch-seed-offset", type=int, default=0)
    parser.add_argument("--event-dispatch-fairness-window", type=int, default=200)
    parser.add_argument("--event-dispatch-fairness-tolerance", type=float, default=0.15)
    parser.add_argument("--smoke-out-json", default=str(DEFAULT_SMOKE_JSON))
    parser.add_argument("--gate-out-json", default=str(DEFAULT_GATE_JSON))
    args = parser.parse_args(argv)

    smoke_seeds = parse_seeds(args.smoke_seeds)
    gate_seeds = parse_seeds(args.gate_seeds)

    cfg = build_b1_config(
        n_players=args.n_players,
        n_rounds=args.n_rounds,
        events_json=args.events_json,
        event_reward_scale=args.event_reward_scale,
        event_reward_mode=args.event_reward_mode,
        event_reward_multiplier_cap=args.event_reward_multiplier_cap,
        event_impact_mode=args.event_impact_mode,
        event_impact_horizon=args.event_impact_horizon,
        event_impact_decay=args.event_impact_decay,
        event_dispatch_mode=args.event_dispatch_mode,
        event_dispatch_target_rate=args.event_dispatch_target_rate,
        event_dispatch_batch_size=args.event_dispatch_batch_size,
        event_dispatch_seed_offset=args.event_dispatch_seed_offset,
        event_dispatch_fairness_window=args.event_dispatch_fairness_window,
        event_dispatch_fairness_tolerance=args.event_dispatch_fairness_tolerance,
        event_warmup_rounds=args.event_warmup_rounds,
        event_risk_enabled=args.event_risk_enabled,
    )

    # Stage 1: smoke (flow lock)
    smoke_outcomes, smoke_summary = run_stage(
        stage_name="B1 Smoke (fixed flow stage 1/2)",
        seeds=smoke_seeds,
        cfg=cfg,
        burn_in=args.burn_in,
        tail=args.tail,
        healthy_threshold=args.healthy_threshold,
        gate_max_l1=args.gate_max_l1,
        gate_min_healthy=args.gate_min_healthy,
    )
    smoke_summary["stage"] = "smoke"
    smoke_summary["flow_lock"] = "smoke_then_gate60"
    smoke_summary["seeds"] = smoke_seeds
    smoke_summary["outcomes"] = [item.__dict__ for item in smoke_outcomes]
    _write_summary(Path(args.smoke_out_json), smoke_summary)
    print(f"Wrote smoke summary JSON: {args.smoke_out_json}")

    smoke_pass = smoke_summary["fairness_fail_count"] == 0
    if not smoke_pass:
        print("Smoke failed fairness checks; gate60 is blocked by protocol lock.")
        return 2

    # Stage 2: gate60 (flow lock)
    baseline_path = Path(args.baseline_summary_json)
    if not baseline_path.exists():
        print(f"Missing baseline summary JSON: {baseline_path}")
        return 3
    baseline_map = _load_baseline_map(
        baseline_path,
        healthy_threshold=args.healthy_threshold,
    )

    gate_outcomes, gate_summary = run_stage(
        stage_name="B1 Gate60 (fixed flow stage 2/2)",
        seeds=gate_seeds,
        cfg=cfg,
        burn_in=args.burn_in,
        tail=args.tail,
        healthy_threshold=args.healthy_threshold,
        gate_max_l1=args.gate_max_l1,
        gate_min_healthy=args.gate_min_healthy,
        baseline_map=baseline_map,
    )
    gate_summary["stage"] = "gate60"
    gate_summary["flow_lock"] = "smoke_then_gate60"
    gate_summary["seeds"] = gate_seeds
    gate_summary["baseline_summary_json"] = str(baseline_path)
    gate_summary["outcomes"] = [item.__dict__ for item in gate_outcomes]
    _write_summary(Path(args.gate_out_json), gate_summary)
    print(f"Wrote gate summary JSON: {args.gate_out_json}")

    overall = bool(gate_summary["gate"]["overall_pass"])
    print("B1 async-dispatch protocol:", "PASS" if overall else "FAIL")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
