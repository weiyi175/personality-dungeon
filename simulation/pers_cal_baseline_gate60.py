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
DEFAULT_OUT_JSON = REPO_ROOT / "outputs" / "pers_cal_baseline_gate60_summary.json"


@dataclass(frozen=True)
class SeedOutcome:
    seed: int
    level: int
    s3: float
    turn: float
    gamma: float
    elapsed_sec: float


def parse_seeds(spec: str) -> list[int]:
    token = str(spec).strip()
    if not token:
        raise ValueError("--seeds cannot be empty")
    if ".." in token and "," not in token:
        lo_s, hi_s = token.split("..", 1)
        lo = int(lo_s)
        hi = int(hi_s)
        if hi < lo:
            raise ValueError("seed range must be ascending")
        return list(range(lo, hi + 1))
    parts = [part.strip() for part in token.split(",") if part.strip()]
    if not parts:
        raise ValueError("--seeds cannot be empty")
    return [int(part) for part in parts]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    q = max(0.0, min(1.0, float(q)))
    idx = int(round((len(arr) - 1) * q))
    return float(arr[idx])


def build_pers_cal_config(*, n_players: int, n_rounds: int) -> PersonalityRLConfig:
    return PersonalityRLConfig(
        n_players=int(n_players),
        n_rounds=int(n_rounds),
        personality_mode="random",
        lambda_alpha=0.15,
        lambda_beta=0.10,
        lambda_r=0.20,
        lambda_risk=0.20,
        lambda_beta_comp=0.0,
        events_json="",
        event_rate=0.0,
        world_feedback=False,
    )


def evaluate_seed(
    cfg: PersonalityRLConfig,
    *,
    seed: int,
    burn_in: int,
    tail: int,
) -> SeedOutcome:
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

    s3 = float(cyc.stage3.score) if cyc.stage3 is not None else 0.0
    turn = float(cyc.stage3.turn_strength) if cyc.stage3 is not None else 0.0
    gamma = float(fit.gamma) if fit is not None else 0.0

    return SeedOutcome(
        seed=int(seed),
        level=int(cyc.level),
        s3=s3,
        turn=turn,
        gamma=gamma,
        elapsed_sec=float(elapsed),
    )


def summarize_outcomes(
    outcomes: list[SeedOutcome],
    *,
    healthy_threshold: float,
    gate_max_l1: int,
    gate_min_healthy: int,
) -> dict[str, Any]:
    n = len(outcomes)
    l1 = sum(1 for item in outcomes if item.level == 1)
    l2 = sum(1 for item in outcomes if item.level == 2)
    l3 = sum(1 for item in outcomes if item.level == 3)
    healthy = sum(1 for item in outcomes if item.s3 >= healthy_threshold)
    marginal = n - healthy - l1

    s3_values = [item.s3 for item in outcomes]
    gamma_values = [item.gamma for item in outcomes]

    l1_pass = l1 <= int(gate_max_l1)
    healthy_pass = healthy >= int(gate_min_healthy)

    non_healthy = [
        {
            "seed": item.seed,
            "level": item.level,
            "s3": round(item.s3, 6),
            "gamma": round(item.gamma, 6),
        }
        for item in outcomes
        if item.s3 < healthy_threshold
    ]

    return {
        "total_seeds": n,
        "l1": l1,
        "l2": l2,
        "l3": l3,
        "healthy": healthy,
        "marginal": marginal,
        "mean_s3": (sum(s3_values) / n) if n else 0.0,
        "median_s3": _percentile(s3_values, 0.5),
        "p10_s3": _percentile(s3_values, 0.1),
        "mean_gamma": (sum(gamma_values) / n) if n else 0.0,
        "gate": {
            "max_l1": int(gate_max_l1),
            "min_healthy": int(gate_min_healthy),
            "healthy_threshold": float(healthy_threshold),
            "l1_pass": l1_pass,
            "healthy_pass": healthy_pass,
            "overall_pass": bool(l1_pass and healthy_pass),
        },
        "non_healthy": non_healthy,
    }


def run_gate(
    *,
    seeds: list[int],
    n_players: int,
    n_rounds: int,
    burn_in: int,
    tail: int,
    healthy_threshold: float,
    gate_max_l1: int,
    gate_min_healthy: int,
) -> tuple[list[SeedOutcome], dict[str, Any]]:
    cfg = build_pers_cal_config(n_players=n_players, n_rounds=n_rounds)
    outcomes: list[SeedOutcome] = []

    total = len(seeds)
    for idx, seed in enumerate(seeds, start=1):
        item = evaluate_seed(cfg, seed=seed, burn_in=burn_in, tail=tail)
        outcomes.append(item)
        status = "OK" if item.s3 >= healthy_threshold else ("L1" if item.level == 1 else "MARGINAL")
        print(
            f"[{idx:2d}/{total}] seed={seed:3d}  L{item.level}  "
            f"s3={item.s3:.4f}  turn={item.turn:.4f}  gamma={item.gamma:.6f}  "
            f"{status}  ({item.elapsed_sec:.1f}s)"
        )

    summary = summarize_outcomes(
        outcomes,
        healthy_threshold=healthy_threshold,
        gate_max_l1=gate_max_l1,
        gate_min_healthy=gate_min_healthy,
    )
    return outcomes, summary


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PERS-CAL baseline 60-seed gate")
    parser.add_argument("--seeds", default="42..101")
    parser.add_argument("--n-players", type=int, default=300)
    parser.add_argument("--n-rounds", type=int, default=12000)
    parser.add_argument("--burn-in", type=int, default=4000)
    parser.add_argument("--tail", type=int, default=4000)
    parser.add_argument("--healthy-threshold", type=float, default=0.80)
    parser.add_argument("--gate-max-l1", type=int, default=3)
    parser.add_argument("--gate-min-healthy", type=int, default=42)
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    args = parser.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    t0 = time.time()

    print("=" * 80)
    print("PERS-CAL Baseline 60-seed Gate")
    print("  personality_mode=random, lambda=(0.15,0.10,0.20,0.20), events=off")
    print(f"Seeds: {seeds[0]}..{seeds[-1]}  (N={len(seeds)})")
    print("=" * 80)

    outcomes, summary = run_gate(
        seeds=seeds,
        n_players=args.n_players,
        n_rounds=args.n_rounds,
        burn_in=args.burn_in,
        tail=args.tail,
        healthy_threshold=args.healthy_threshold,
        gate_max_l1=args.gate_max_l1,
        gate_min_healthy=args.gate_min_healthy,
    )

    elapsed = time.time() - t0
    summary["elapsed_sec"] = elapsed
    summary["seeds"] = seeds
    summary["outcomes"] = [item.__dict__ for item in outcomes]

    print()
    print("SUMMARY")
    print("=" * 80)
    print(f"Total seeds:    {summary['total_seeds']}")
    print(f"L1:             {summary['l1']} ({summary['l1']/summary['total_seeds']*100:.1f}%)")
    print(f"L2:             {summary['l2']} ({summary['l2']/summary['total_seeds']*100:.1f}%)")
    print(f"L3:             {summary['l3']} ({summary['l3']/summary['total_seeds']*100:.1f}%)")
    print(
        "Healthy (s3>=%.2f): %d (%.1f%%)"
        % (
            summary["gate"]["healthy_threshold"],
            summary["healthy"],
            summary["healthy"] / summary["total_seeds"] * 100,
        )
    )
    print(f"Mean s3:        {summary['mean_s3']:.4f}")
    print(f"Median s3:      {summary['median_s3']:.4f}")
    print(f"P10 s3:         {summary['p10_s3']:.4f}")
    print(f"Mean gamma:     {summary['mean_gamma']:.6f}")
    print(f"Total time:     {elapsed:.1f}s")

    gate = summary["gate"]
    print()
    print(f"Gate L1 <= {gate['max_l1']}:     {summary['l1']} -> {'PASS' if gate['l1_pass'] else 'FAIL'}")
    print(
        f"Gate healthy >= {gate['min_healthy']}: {summary['healthy']} -> "
        f"{'PASS' if gate['healthy_pass'] else 'FAIL'}"
    )
    print()
    print(
        "PERS-CAL Baseline 60-seed Gate: "
        + ("PASS" if gate["overall_pass"] else "FAIL")
    )

    out_json = Path(args.out_json)
    _write_summary(out_json, summary)
    print(f"Wrote summary JSON: {out_json}")
    return 0 if gate["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
