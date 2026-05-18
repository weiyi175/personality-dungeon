#!/usr/bin/env python3
"""Confirm whether a 9-persona random wall exists for W4 C-power setting with personality_coupled evolution.

This script compares two conditions under identical dynamics:
1) control_none: no personality perturbation (all-zero traits)
2) random_9persona: full 9-trait random personalities per player

In personality_coupled mode, personalities actually affect per-player inertia and selection strength.

Protocol defaults:
- gamma=0.13, power=2.8
- rounds=10000, players=300
- evolution_mode=personality_coupled (vs. sampled in baseline)
- personality_coupling_mu_base=0.1, lambda_mu=0.1, lambda_k=0.2
- seeds=45,47,49,51,53,55,91,93,95,97,99,101
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.cycle_metrics import classify_cycle_level, phase_rotation_r2
from players.base_player import DEFAULT_PERSONALITY_KEYS
from simulation.run_simulation import SimConfig, _write_timeseries_csv, simulate


DEFAULT_SEEDS = [45, 47, 49, 51, 53, 55, 91, 93, 95, 97, 99, 101]


@dataclass
class RunMetric:
    condition: str
    seed: int
    csv_path: str
    cycle_level: int
    phase_velocity: float
    l3_pass: bool
    velocity_pass: bool
    both_pass: bool


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _extract_series(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {
        "aggressive": [float(r.get("p_aggressive", 0.0)) for r in rows],
        "defensive": [float(r.get("p_defensive", 0.0)) for r in rows],
        "balanced": [float(r.get("p_balanced", 0.0)) for r in rows],
    }


def _random_9persona_setup(seed: int):
    def cb(players: list[object], _strategy_space: list[str], _cfg: SimConfig) -> None:
        for idx, pl in enumerate(players):
            rng = random.Random(int(seed) * 10000 + idx)
            for k in DEFAULT_PERSONALITY_KEYS:
                pl.personality[k] = rng.uniform(-0.4, 0.4)

    return cb


def _run_one(
    *,
    condition: str,
    seed: int,
    gamma: float,
    power: float,
    rounds: int,
    players: int,
    out_dir: Path,
    velocity_threshold: float,
    personality_coupling_mu_base: float = 0.1,
    personality_coupling_lambda_mu: float = 0.1,
    personality_coupling_lambda_k: float = 0.2,
) -> RunMetric:
    out_csv = out_dir / condition / f"seed{seed}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cfg = SimConfig(
        n_players=int(players),
        n_rounds=int(rounds),
        seed=int(seed),
        payoff_mode="matrix_ab",
        popularity_mode="sampled",
        gamma=float(gamma),
        epsilon=0.0,
        a=1.0,
        b=0.9,
        matrix_cross_coupling=0.20,
        init_bias=0.5,
        evolution_mode="personality_coupled",  # Changed from "sampled"
        payoff_lag=1,
        selection_strength=0.02,
        enable_events=False,
        events_json=None,
        out_csv=out_csv,
        memory_kernel=1,
        synergy_type="nonlinear",
        synergy_gamma=float(gamma),
        synergy_nonlinear_type="power",
        synergy_nonlinear_power=float(power),
        personality_coupling_mu_base=float(personality_coupling_mu_base),
        personality_coupling_lambda_mu=float(personality_coupling_lambda_mu),
        personality_coupling_lambda_k=float(personality_coupling_lambda_k),
        personality_coupling_beta_state_k=0.0,
    )

    player_setup = _random_9persona_setup(seed) if condition == "random_9persona" else None
    strategy_space, rows = simulate(cfg, player_setup_callback=player_setup)
    _write_timeseries_csv(out_csv, strategy_space=strategy_space, rows=rows)

    series = _extract_series(rows)
    cyc = classify_cycle_level(
        series,
        burn_in=2000,
        tail=2000,
        amplitude_threshold=0.02,
        eta=0.6,
        min_turn_strength=0.0,
    )
    rot = phase_rotation_r2(series, burn_in=2000, tail=2000)
    win_len = rot.window_length if rot.window_length > 0 else 1
    phase_velocity = float(rot.cumulative_rotation) / float(win_len)

    l3_pass = int(cyc.level) >= 3
    velocity_pass = phase_velocity > float(velocity_threshold)
    return RunMetric(
        condition=condition,
        seed=int(seed),
        csv_path=str(out_csv.relative_to(ROOT)),
        cycle_level=int(cyc.level),
        phase_velocity=float(phase_velocity),
        l3_pass=bool(l3_pass),
        velocity_pass=bool(velocity_pass),
        both_pass=bool(l3_pass and velocity_pass),
    )


def _agg(metrics: list[RunMetric]) -> dict[str, float | int]:
    n = len(metrics)
    if n == 0:
        return {
            "n": 0,
            "l3_rate": 0.0,
            "velocity_pass_rate": 0.0,
            "both_pass_rate": 0.0,
            "mean_phase_velocity": 0.0,
        }
    return {
        "n": n,
        "l3_rate": sum(1 for m in metrics if m.l3_pass) / n,
        "velocity_pass_rate": sum(1 for m in metrics if m.velocity_pass) / n,
        "both_pass_rate": sum(1 for m in metrics if m.both_pass) / n,
        "mean_phase_velocity": sum(m.phase_velocity for m in metrics) / n,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Confirm 9-persona random wall for C-power with personality_coupled")
    p.add_argument("--gamma", type=float, default=0.13)
    p.add_argument("--power", type=float, default=2.8)
    p.add_argument("--rounds", type=int, default=10000)
    p.add_argument("--players", type=int, default=300)
    p.add_argument("--velocity-threshold", type=float, default=0.001)
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--personality-coupling-mu-base", type=float, default=0.1)
    p.add_argument("--personality-coupling-lambda-mu", type=float, default=0.1)
    p.add_argument("--personality-coupling-lambda-k", type=float, default=0.2)
    args = p.parse_args()

    seeds = _parse_int_list(args.seeds)
    out_dir = ROOT / "outputs" / "c_power_9persona_wall_check_personality_coupled"
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = ["control_none", "random_9persona"]
    all_metrics: list[RunMetric] = []

    total = len(conditions) * len(seeds)
    i = 0
    print("C-power 9-persona random wall confirmation [personality_coupled mode]")
    print(f"gamma={args.gamma}, power={args.power}, rounds={args.rounds}, players={args.players}")
    print(f"personality_coupling: mu_base={args.personality_coupling_mu_base}, lambda_mu={args.personality_coupling_lambda_mu}, lambda_k={args.personality_coupling_lambda_k}")
    print(f"seeds={seeds}")
    print(f"runs={total}")

    for cond in conditions:
        for seed in seeds:
            i += 1
            print(f"[{i:2d}/{total}] {cond} seed={seed}", end=" ", flush=True)
            m = _run_one(
                condition=cond,
                seed=seed,
                gamma=float(args.gamma),
                power=float(args.power),
                rounds=int(args.rounds),
                players=int(args.players),
                out_dir=out_dir,
                velocity_threshold=float(args.velocity_threshold),
                personality_coupling_mu_base=float(args.personality_coupling_mu_base),
                personality_coupling_lambda_mu=float(args.personality_coupling_lambda_mu),
                personality_coupling_lambda_k=float(args.personality_coupling_lambda_k),
            )
            all_metrics.append(m)
            print(f"L{m.cycle_level}, v={m.phase_velocity:.6f}, both={'Y' if m.both_pass else 'N'}")

    by_cond = {
        cond: [m for m in all_metrics if m.condition == cond]
        for cond in conditions
    }
    agg = {cond: _agg(vals) for cond, vals in by_cond.items()}

    control_rate = float(agg["control_none"]["both_pass_rate"])
    random_rate = float(agg["random_9persona"]["both_pass_rate"])
    random_wall_established = (control_rate >= 0.8) and (random_rate <= 0.3)

    summary = {
        "experiment": "C-power 9-persona random wall check [personality_coupled mode]",
        "gamma": float(args.gamma),
        "power": float(args.power),
        "rounds": int(args.rounds),
        "players": int(args.players),
        "velocity_threshold": float(args.velocity_threshold),
        "evolution_mode": "personality_coupled",
        "personality_coupling_mu_base": float(args.personality_coupling_mu_base),
        "personality_coupling_lambda_mu": float(args.personality_coupling_lambda_mu),
        "personality_coupling_lambda_k": float(args.personality_coupling_lambda_k),
        "seeds": seeds,
        "aggregate": agg,
        "random_wall_rule": {
            "control_both_pass_rate_min": 0.8,
            "random_both_pass_rate_max": 0.3,
            "established": random_wall_established,
        },
        "runs": [asdict(m) for m in all_metrics],
    }

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))

    md_lines = [
        "# C-power 9-persona Random Wall Check [personality_coupled mode]",
        "",
        f"- gamma: {args.gamma}",
        f"- power: {args.power}",
        f"- rounds: {args.rounds}",
        f"- seeds: {len(seeds)}",
        f"- evolution_mode: personality_coupled",
        f"- personality_coupling_mu_base: {args.personality_coupling_mu_base}",
        f"- personality_coupling_lambda_mu: {args.personality_coupling_lambda_mu}",
        f"- personality_coupling_lambda_k: {args.personality_coupling_lambda_k}",
        "",
        "| condition | n | L3_rate | velocity_pass_rate | both_pass_rate | mean_phase_velocity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cond in conditions:
        a = agg[cond]
        md_lines.append(
            f"| {cond} | {a['n']} | {a['l3_rate']:.1%} | {a['velocity_pass_rate']:.1%} | {a['both_pass_rate']:.1%} | {a['mean_phase_velocity']:.6f} |"
        )
    md_lines.append("")
    md_lines.append(f"- random_wall_established: {'YES' if random_wall_established else 'NO'}")
    md_lines.append(
        "- rule: control both_pass_rate >= 80% AND random_9persona both_pass_rate <= 30%"
    )

    summary_md = out_dir / "summary.md"
    summary_md.write_text("\n".join(md_lines) + "\n")

    print(f"saved: {summary_json}")
    print(f"saved: {summary_md}")
    print(f"random_wall_established: {'YES' if random_wall_established else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
