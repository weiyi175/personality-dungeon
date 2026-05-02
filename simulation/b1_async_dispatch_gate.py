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
    event_neutrality_max_abs_mean: float
    event_neutrality_pass: bool
    event_trigger_guard_check_count: int
    event_trigger_guard_block_count: int
    event_trigger_guard_block_rate: float
    event_trigger_guard_pass: bool
    world_feedback_mode: str
    world_readonly_applied: bool
    readonly_leak_score: float
    readonly_leak_pass: bool
    difficulty_modulation_applied: bool
    difficulty_index_mean: float
    event_difficulty_multiplier_mean: float
    payoff_static_score: float
    payoff_static_pass: bool
    replicator_update_mode: str
    async_update_applied: bool
    async_update_ratio_mean: float
    update_skew_index: float
    event_queue_mode: str
    player_event_queue_depth_mean: float
    player_event_queue_depth_p95: float
    queue_overflow_count: int
    phase_lag_index_mean: float
    reward_multiplier_raw_mean: float
    reward_multiplier_clamped_mean: float
    log_reward_multiplier_mean: float
    modulation_gain_effective_mean: float
    modulation_zero_mean_residual_max: float
    multiplicative_static_pass: bool
    event_impact_mode: str
    impact_spread_applied: bool
    impact_spread_radius: float
    impact_spread_delay_mean: float
    impact_spread_alignment: float
    impact_kernel_mass_local: float
    impact_kernel_mass_neighbor: float
    impact_kernel_mass_error: float


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
    event_modulation_mode: str,
    event_modulation_gain: float,
    event_modulation_log_center: float,
    event_modulation_zero_mean: bool,
    event_modulation_floor: float,
    event_modulation_ceiling: float,
    event_neutralize_payoff: bool,
    event_neutralize_eps: float,
    event_impact_mode: str,
    event_impact_horizon: int,
    event_impact_decay: float,
    impact_spread_kernel_id: str,
    impact_spread_local_mass: float,
    impact_spread_neighbor_mass: float,
    impact_spread_neighbor_hop: int,
    impact_spread_memory_kernel: int,
    event_dispatch_mode: str,
    event_dispatch_target_rate: float,
    event_dispatch_batch_size: int,
    event_dispatch_seed_offset: int,
    event_dispatch_fairness_window: int,
    event_dispatch_fairness_tolerance: float,
    event_trigger_mode: str,
    event_trigger_entropy_threshold: float,
    event_warmup_rounds: int,
    event_risk_enabled: bool,
    world_feedback_mode: str,
    lambda_world: float,
    world_update_interval: int,
    replicator_update_mode: str,
    replicator_async_minibatch: int,
    replicator_async_jitter: float,
    event_queue_mode: str,
    event_queue_cap: int,
    event_queue_drain_rate: float,
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
        event_modulation_mode=str(event_modulation_mode),
        event_modulation_gain=float(event_modulation_gain),
        event_modulation_log_center=float(event_modulation_log_center),
        event_modulation_zero_mean=bool(event_modulation_zero_mean),
        event_modulation_floor=float(event_modulation_floor),
        event_modulation_ceiling=float(event_modulation_ceiling),
        event_neutralize_payoff=bool(event_neutralize_payoff),
        event_neutralize_eps=float(event_neutralize_eps),
        event_impact_mode=str(event_impact_mode),
        event_impact_horizon=int(event_impact_horizon),
        event_impact_decay=float(event_impact_decay),
        impact_spread_kernel_id=str(impact_spread_kernel_id),
        impact_spread_local_mass=float(impact_spread_local_mass),
        impact_spread_neighbor_mass=float(impact_spread_neighbor_mass),
        impact_spread_neighbor_hop=int(impact_spread_neighbor_hop),
        impact_spread_memory_kernel=int(impact_spread_memory_kernel),
        event_warmup_rounds=int(event_warmup_rounds),
        event_risk_enabled=bool(event_risk_enabled),
        event_dispatch_mode=str(event_dispatch_mode),
        event_dispatch_target_rate=float(event_dispatch_target_rate),
        event_dispatch_batch_size=int(event_dispatch_batch_size),
        event_dispatch_seed_offset=int(event_dispatch_seed_offset),
        event_dispatch_fairness_window=int(event_dispatch_fairness_window),
        event_dispatch_fairness_tolerance=float(event_dispatch_fairness_tolerance),
        event_trigger_mode=str(event_trigger_mode),
        event_trigger_entropy_threshold=float(event_trigger_entropy_threshold),
        world_feedback=(str(world_feedback_mode).strip().lower() == "adaptive_world"),
        world_feedback_mode=str(world_feedback_mode),
        lambda_world=float(lambda_world),
        world_update_interval=int(world_update_interval),
        replicator_update_mode=str(replicator_update_mode),
        replicator_async_minibatch=int(replicator_async_minibatch),
        replicator_async_jitter=float(replicator_async_jitter),
        event_queue_mode=str(event_queue_mode),
        event_queue_cap=int(event_queue_cap),
        event_queue_drain_rate=float(event_queue_drain_rate),
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
        event_neutrality_max_abs_mean=float(diag.get("event_neutrality_max_abs_mean", 0.0)),
        event_neutrality_pass=bool(diag.get("event_neutrality_pass", True)),
        event_trigger_guard_check_count=int(diag.get("event_trigger_guard_check_count", 0)),
        event_trigger_guard_block_count=int(diag.get("event_trigger_guard_block_count", 0)),
        event_trigger_guard_block_rate=float(diag.get("event_trigger_guard_block_rate", 0.0)),
        event_trigger_guard_pass=bool(diag.get("event_trigger_guard_pass", True)),
        world_feedback_mode=str(diag.get("world_feedback_mode", "off")),
        world_readonly_applied=bool(diag.get("world_readonly_applied", False)),
        readonly_leak_score=float(diag.get("readonly_leak_score", 0.0)),
        readonly_leak_pass=bool(diag.get("readonly_leak_pass", True)),
        difficulty_modulation_applied=bool(diag.get("difficulty_modulation_applied", False)),
        difficulty_index_mean=float(diag.get("difficulty_index_mean", 0.0)),
        event_difficulty_multiplier_mean=float(diag.get("event_difficulty_multiplier_mean", 1.0)),
        payoff_static_score=float(diag.get("payoff_static_score", 0.0)),
        payoff_static_pass=bool(diag.get("payoff_static_pass", True)),
        replicator_update_mode=str(diag.get("replicator_update_mode", "sync_global")),
        async_update_applied=bool(diag.get("async_update_applied", False)),
        async_update_ratio_mean=float(diag.get("async_update_ratio_mean", 0.0)),
        update_skew_index=float(diag.get("update_skew_index", 0.0)),
        event_queue_mode=str(diag.get("event_queue_mode", "off")),
        player_event_queue_depth_mean=float(diag.get("player_event_queue_depth_mean", 0.0)),
        player_event_queue_depth_p95=float(diag.get("player_event_queue_depth_p95", 0.0)),
        queue_overflow_count=int(diag.get("queue_overflow_count", 0)),
        phase_lag_index_mean=float(diag.get("phase_lag_index_mean", 0.0)),
        reward_multiplier_raw_mean=float(diag.get("reward_multiplier_raw_mean", 1.0)),
        reward_multiplier_clamped_mean=float(diag.get("reward_multiplier_clamped_mean", 1.0)),
        log_reward_multiplier_mean=float(diag.get("log_reward_multiplier_mean", 0.0)),
        modulation_gain_effective_mean=float(diag.get("modulation_gain_effective_mean", 0.0)),
        modulation_zero_mean_residual_max=float(diag.get("modulation_zero_mean_residual_max", 0.0)),
        multiplicative_static_pass=bool(diag.get("multiplicative_static_pass", True)),
        event_impact_mode=str(diag.get("event_impact_mode", cfg.event_impact_mode)),
        impact_spread_applied=bool(diag.get("impact_spread_applied", False)),
        impact_spread_radius=float(diag.get("impact_spread_radius", 0.0)),
        impact_spread_delay_mean=float(diag.get("impact_spread_delay_mean", 0.0)),
        impact_spread_alignment=float(diag.get("impact_spread_alignment_mean", 0.0)),
        impact_kernel_mass_local=float(diag.get("impact_kernel_mass_local", 1.0)),
        impact_kernel_mass_neighbor=float(diag.get("impact_kernel_mass_neighbor", 0.0)),
        impact_kernel_mass_error=float(diag.get("impact_kernel_mass_error", 0.0)),
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
    readonly_leak_threshold: float,
    payoff_static_threshold: float,
    require_async_update: bool,
    queue_overflow_max: int,
    phase_lag_min: float,
    baseline_map: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    n = len(outcomes)
    l1 = sum(1 for o in outcomes if o.level == 1)
    l2 = sum(1 for o in outcomes if o.level == 2)
    l3 = sum(1 for o in outcomes if o.level == 3)
    healthy = sum(1 for o in outcomes if o.s3 >= healthy_threshold)
    marginal = n - healthy - l1
    fairness_fail_count = sum(1 for o in outcomes if not o.dispatch_fairness_pass)
    neutrality_fail_count = sum(1 for o in outcomes if not o.event_neutrality_pass)
    trigger_guard_fail_count = sum(1 for o in outcomes if not o.event_trigger_guard_pass)
    readonly_leak_fail_count = sum(
        1
        for o in outcomes
        if float(o.readonly_leak_score) > float(readonly_leak_threshold)
    )
    payoff_static_fail_count = sum(
        1
        for o in outcomes
        if float(o.payoff_static_score) > float(payoff_static_threshold)
    )
    async_update_fail_count = sum(
        1
        for o in outcomes
        if bool(require_async_update) and (not bool(getattr(o, "async_update_applied", False)))
    )
    queue_overflow_fail_count = sum(
        1
        for o in outcomes
        if int(getattr(o, "queue_overflow_count", 0)) > int(queue_overflow_max)
    )
    phase_lag_fail_count = sum(
        1
        for o in outcomes
        if float(getattr(o, "phase_lag_index_mean", 0.0)) < float(phase_lag_min)
    )
    multiplicative_static_fail_count = sum(
        1
        for o in outcomes
        if not bool(getattr(o, "multiplicative_static_pass", True))
    )
    impact_kernel_mass_error_fail_count = sum(
        1
        for o in outcomes
        if float(getattr(o, "impact_kernel_mass_error", 0.0)) > 1e-9
    )
    impact_spread_not_applied_fail_count = sum(
        1
        for o in outcomes
        if str(getattr(o, "event_impact_mode", "instant")) == "spread"
        and (not bool(getattr(o, "impact_spread_applied", False)))
    )
    modulation_zero_mean_residual_max = max(
        (float(getattr(o, "modulation_zero_mean_residual_max", 0.0)) for o in outcomes),
        default=0.0,
    )
    impact_kernel_mass_error_max = max(
        (float(getattr(o, "impact_kernel_mass_error", 0.0)) for o in outcomes),
        default=0.0,
    )

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
    invariant_neutrality_pass = neutrality_fail_count == 0
    invariant_trigger_guard_pass = trigger_guard_fail_count == 0
    invariant_readonly_leak_pass = readonly_leak_fail_count == 0
    invariant_payoff_static_pass = payoff_static_fail_count == 0
    invariant_async_update_pass = async_update_fail_count == 0
    invariant_queue_overflow_pass = queue_overflow_fail_count == 0
    invariant_phase_lag_pass = phase_lag_fail_count == 0
    invariant_multiplicative_static_pass = multiplicative_static_fail_count == 0
    invariant_impact_kernel_mass_pass = impact_kernel_mass_error_fail_count == 0
    invariant_impact_spread_applied_pass = impact_spread_not_applied_fail_count == 0
    invariant_overall_pass = bool(
        invariant_neutrality_pass
        and invariant_trigger_guard_pass
        and invariant_readonly_leak_pass
        and invariant_payoff_static_pass
        and invariant_async_update_pass
        and invariant_queue_overflow_pass
        and invariant_phase_lag_pass
        and invariant_multiplicative_static_pass
        and invariant_impact_kernel_mass_pass
        and invariant_impact_spread_applied_pass
    )

    return {
        "total_seeds": n,
        "l1": l1,
        "l2": l2,
        "l3": l3,
        "healthy": healthy,
        "marginal": marginal,
        "fairness_fail_count": fairness_fail_count,
        "neutrality_fail_count": neutrality_fail_count,
        "trigger_guard_fail_count": trigger_guard_fail_count,
        "readonly_leak_fail_count": readonly_leak_fail_count,
        "payoff_static_fail_count": payoff_static_fail_count,
        "async_update_fail_count": async_update_fail_count,
        "queue_overflow_fail_count": queue_overflow_fail_count,
        "phase_lag_fail_count": phase_lag_fail_count,
        "multiplicative_static_fail_count": multiplicative_static_fail_count,
        "impact_kernel_mass_error_fail_count": impact_kernel_mass_error_fail_count,
        "impact_spread_not_applied_fail_count": impact_spread_not_applied_fail_count,
        "modulation_zero_mean_residual_max": modulation_zero_mean_residual_max,
        "impact_kernel_mass_error_max": impact_kernel_mass_error_max,
        "mean_s3": (sum(s3_values) / n) if n else 0.0,
        "median_s3": _percentile(s3_values, 0.5),
        "p10_s3": _percentile(s3_values, 0.1),
        "mean_gamma": (sum(gamma_values) / n) if n else 0.0,
        "mean_event_neutrality_max_abs_mean": (
            sum(o.event_neutrality_max_abs_mean for o in outcomes) / n
        ) if n else 0.0,
        "mean_event_trigger_guard_block_rate": (
            sum(o.event_trigger_guard_block_rate for o in outcomes) / n
        ) if n else 0.0,
        "mean_readonly_leak_score": (
            sum(o.readonly_leak_score for o in outcomes) / n
        ) if n else 0.0,
        "mean_difficulty_index": (
            sum(o.difficulty_index_mean for o in outcomes) / n
        ) if n else 0.0,
        "mean_event_difficulty_multiplier": (
            sum(o.event_difficulty_multiplier_mean for o in outcomes) / n
        ) if n else 1.0,
        "mean_payoff_static_score": (
            sum(o.payoff_static_score for o in outcomes) / n
        ) if n else 0.0,
        "mean_async_update_ratio": (
            sum(float(getattr(o, "async_update_ratio_mean", 0.0)) for o in outcomes) / n
        ) if n else 0.0,
        "mean_update_skew_index": (
            sum(float(getattr(o, "update_skew_index", 0.0)) for o in outcomes) / n
        ) if n else 0.0,
        "mean_player_event_queue_depth": (
            sum(float(getattr(o, "player_event_queue_depth_mean", 0.0)) for o in outcomes) / n
        ) if n else 0.0,
        "mean_player_event_queue_depth_p95": (
            sum(float(getattr(o, "player_event_queue_depth_p95", 0.0)) for o in outcomes) / n
        ) if n else 0.0,
        "mean_phase_lag_index": (
            sum(float(getattr(o, "phase_lag_index_mean", 0.0)) for o in outcomes) / n
        ) if n else 0.0,
        "mean_reward_multiplier_raw": (
            sum(float(getattr(o, "reward_multiplier_raw_mean", 1.0)) for o in outcomes) / n
        ) if n else 1.0,
        "mean_reward_multiplier_clamped": (
            sum(float(getattr(o, "reward_multiplier_clamped_mean", 1.0)) for o in outcomes) / n
        ) if n else 1.0,
        "mean_log_reward_multiplier": (
            sum(float(getattr(o, "log_reward_multiplier_mean", 0.0)) for o in outcomes) / n
        ) if n else 0.0,
        "mean_modulation_gain_effective": (
            sum(float(getattr(o, "modulation_gain_effective_mean", 0.0)) for o in outcomes) / n
        ) if n else 0.0,
        "mean_impact_spread_alignment": (
            sum(float(getattr(o, "impact_spread_alignment", 0.0)) for o in outcomes) / n
        ) if n else 0.0,
        "max_queue_overflow_count": max(
            (int(getattr(o, "queue_overflow_count", 0)) for o in outcomes),
            default=0,
        ),
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
            "readonly_leak_threshold": float(readonly_leak_threshold),
            "payoff_static_threshold": float(payoff_static_threshold),
            "require_async_update": bool(require_async_update),
            "queue_overflow_max": int(queue_overflow_max),
            "phase_lag_min": float(phase_lag_min),
            "invariant_neutrality_pass": invariant_neutrality_pass,
            "invariant_trigger_guard_pass": invariant_trigger_guard_pass,
            "invariant_readonly_leak_pass": invariant_readonly_leak_pass,
            "invariant_payoff_static_pass": invariant_payoff_static_pass,
            "invariant_async_update_pass": invariant_async_update_pass,
            "invariant_queue_overflow_pass": invariant_queue_overflow_pass,
            "invariant_phase_lag_pass": invariant_phase_lag_pass,
            "invariant_multiplicative_static_pass": invariant_multiplicative_static_pass,
            "invariant_impact_kernel_mass_pass": invariant_impact_kernel_mass_pass,
            "invariant_impact_spread_applied_pass": invariant_impact_spread_applied_pass,
            "invariant_overall_pass": invariant_overall_pass,
            "overall_pass": bool(
                l1_pass
                and healthy_pass
                and fairness_pass
                and new_l1_pass
                and invariant_overall_pass
            ),
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
    readonly_leak_threshold: float,
    payoff_static_threshold: float,
    require_async_update: bool,
    queue_overflow_max: int,
    phase_lag_min: float,
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
        readonly_leak_threshold=readonly_leak_threshold,
        payoff_static_threshold=payoff_static_threshold,
        require_async_update=require_async_update,
        queue_overflow_max=queue_overflow_max,
        phase_lag_min=phase_lag_min,
        baseline_map=baseline_map,
    )
    return outcomes, summary


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _phase_run_metadata_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "phase_id": str(args.phase_id),
        "bridge_id": str(args.bridge_id),
        "bridge_count": max(0, int(args.bridge_count)),
        "anchor_profile_id": str(args.anchor_profile_id),
    }


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
        "--event-modulation-mode",
        default="off",
        choices=["off", "multiplicative_v2"],
    )
    parser.add_argument("--event-modulation-gain", type=float, default=0.0)
    parser.add_argument("--event-modulation-log-center", type=float, default=0.0)
    parser.add_argument("--event-modulation-zero-mean", action="store_true", default=False)
    parser.add_argument("--event-modulation-floor", type=float, default=1.0)
    parser.add_argument("--event-modulation-ceiling", type=float, default=1.0)
    parser.add_argument("--event-neutralize-payoff", action="store_true", default=False)
    parser.add_argument("--event-neutralize-eps", type=float, default=1e-9)
    parser.add_argument(
        "--event-impact-mode",
        default="instant",
        choices=["instant", "spread"],
    )
    parser.add_argument("--event-impact-horizon", type=int, default=1)
    parser.add_argument("--event-impact-decay", type=float, default=0.70)
    parser.add_argument(
        "--impact-spread-kernel-id",
        default="legacy_v1",
        choices=["legacy_v1", "hierarchical_v2"],
    )
    parser.add_argument("--impact-spread-local-mass", type=float, default=1.0)
    parser.add_argument("--impact-spread-neighbor-mass", type=float, default=0.0)
    parser.add_argument("--impact-spread-neighbor-hop", type=int, default=0)
    parser.add_argument("--impact-spread-memory-kernel", type=int, default=1)
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
    parser.add_argument(
        "--event-trigger-mode",
        default="always",
        choices=["always", "entropy_guard"],
    )
    parser.add_argument("--event-trigger-entropy-threshold", type=float, default=0.85)
    parser.add_argument(
        "--replicator-update-mode",
        default="sync_global",
        choices=["sync_global", "async_per_player"],
    )
    parser.add_argument("--replicator-async-minibatch", type=int, default=0)
    parser.add_argument("--replicator-async-jitter", type=float, default=0.0)
    parser.add_argument(
        "--event-queue-mode",
        default="off",
        choices=["off", "per_player"],
    )
    parser.add_argument("--event-queue-cap", type=int, default=0)
    parser.add_argument("--event-queue-drain-rate", type=float, default=1.0)
    parser.add_argument(
        "--world-feedback-mode",
        default="off",
        choices=["off", "adaptive_world", "read_only", "difficulty_only"],
    )
    parser.add_argument("--lambda-world", type=float, default=0.08)
    parser.add_argument("--world-update-interval", type=int, default=200)
    parser.add_argument("--readonly-leak-threshold", type=float, default=1e-6)
    parser.add_argument("--payoff-static-threshold", type=float, default=1e-9)
    parser.add_argument("--require-async-update", action="store_true", default=False)
    parser.add_argument("--queue-overflow-max", type=int, default=0)
    parser.add_argument("--phase-lag-min", type=float, default=0.0)
    parser.add_argument("--phase-id", default="")
    parser.add_argument("--bridge-id", default="")
    parser.add_argument("--bridge-count", type=int, default=0)
    parser.add_argument("--anchor-profile-id", default="")
    parser.add_argument("--smoke-out-json", default=str(DEFAULT_SMOKE_JSON))
    parser.add_argument("--gate-out-json", default=str(DEFAULT_GATE_JSON))
    args = parser.parse_args(argv)

    phase_metadata = _phase_run_metadata_from_args(args)

    smoke_seeds = parse_seeds(args.smoke_seeds)
    gate_seeds = parse_seeds(args.gate_seeds)

    cfg = build_b1_config(
        n_players=args.n_players,
        n_rounds=args.n_rounds,
        events_json=args.events_json,
        event_reward_scale=args.event_reward_scale,
        event_reward_mode=args.event_reward_mode,
        event_reward_multiplier_cap=args.event_reward_multiplier_cap,
        event_modulation_mode=args.event_modulation_mode,
        event_modulation_gain=args.event_modulation_gain,
        event_modulation_log_center=args.event_modulation_log_center,
        event_modulation_zero_mean=args.event_modulation_zero_mean,
        event_modulation_floor=args.event_modulation_floor,
        event_modulation_ceiling=args.event_modulation_ceiling,
        event_neutralize_payoff=args.event_neutralize_payoff,
        event_neutralize_eps=args.event_neutralize_eps,
        event_impact_mode=args.event_impact_mode,
        event_impact_horizon=args.event_impact_horizon,
        event_impact_decay=args.event_impact_decay,
        impact_spread_kernel_id=args.impact_spread_kernel_id,
        impact_spread_local_mass=args.impact_spread_local_mass,
        impact_spread_neighbor_mass=args.impact_spread_neighbor_mass,
        impact_spread_neighbor_hop=args.impact_spread_neighbor_hop,
        impact_spread_memory_kernel=args.impact_spread_memory_kernel,
        event_dispatch_mode=args.event_dispatch_mode,
        event_dispatch_target_rate=args.event_dispatch_target_rate,
        event_dispatch_batch_size=args.event_dispatch_batch_size,
        event_dispatch_seed_offset=args.event_dispatch_seed_offset,
        event_dispatch_fairness_window=args.event_dispatch_fairness_window,
        event_dispatch_fairness_tolerance=args.event_dispatch_fairness_tolerance,
        event_trigger_mode=args.event_trigger_mode,
        event_trigger_entropy_threshold=args.event_trigger_entropy_threshold,
        event_warmup_rounds=args.event_warmup_rounds,
        event_risk_enabled=args.event_risk_enabled,
        world_feedback_mode=args.world_feedback_mode,
        lambda_world=args.lambda_world,
        world_update_interval=args.world_update_interval,
        replicator_update_mode=args.replicator_update_mode,
        replicator_async_minibatch=args.replicator_async_minibatch,
        replicator_async_jitter=args.replicator_async_jitter,
        event_queue_mode=args.event_queue_mode,
        event_queue_cap=args.event_queue_cap,
        event_queue_drain_rate=args.event_queue_drain_rate,
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
        readonly_leak_threshold=args.readonly_leak_threshold,
        payoff_static_threshold=args.payoff_static_threshold,
        require_async_update=args.require_async_update,
        queue_overflow_max=args.queue_overflow_max,
        phase_lag_min=args.phase_lag_min,
    )
    smoke_summary["stage"] = "smoke"
    smoke_summary["flow_lock"] = "smoke_then_gate60"
    smoke_summary["seeds"] = smoke_seeds
    smoke_summary["phase_run_metadata"] = dict(phase_metadata)
    smoke_summary.update(phase_metadata)
    smoke_summary["outcomes"] = [
        {**item.__dict__, **phase_metadata}
        for item in smoke_outcomes
    ]
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
        readonly_leak_threshold=args.readonly_leak_threshold,
        payoff_static_threshold=args.payoff_static_threshold,
        require_async_update=args.require_async_update,
        queue_overflow_max=args.queue_overflow_max,
        phase_lag_min=args.phase_lag_min,
        baseline_map=baseline_map,
    )
    gate_summary["stage"] = "gate60"
    gate_summary["flow_lock"] = "smoke_then_gate60"
    gate_summary["seeds"] = gate_seeds
    gate_summary["baseline_summary_json"] = str(baseline_path)
    gate_summary["phase_run_metadata"] = dict(phase_metadata)
    gate_summary.update(phase_metadata)
    gate_summary["outcomes"] = [
        {**item.__dict__, **phase_metadata}
        for item in gate_outcomes
    ]
    _write_summary(Path(args.gate_out_json), gate_summary)
    print(f"Wrote gate summary JSON: {args.gate_out_json}")

    overall = bool(gate_summary["gate"]["overall_pass"])
    print("B1 async-dispatch protocol:", "PASS" if overall else "FAIL")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
