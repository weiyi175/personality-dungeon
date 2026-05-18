#!/usr/bin/env python3
"""Aggressive C-power parameter scan under personality_coupled evolution.

Scans gamma ∈ {0.14,0.15,0.16} and power ∈ {3.0,3.1,3.2} across standard seeds.
Saves per-run CSVs and a summary.json with aggregate metrics per combo.
"""

from __future__ import annotations

import argparse
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
    gamma: float
    power: float
    condition: str
    seed: int
    csv_path: str
    cycle_level: int
    phase_velocity: float
    l3_pass: bool
    velocity_pass: bool
    both_pass: bool


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
    gamma: float,
    power: float,
    condition: str,
    seed: int,
    rounds: int,
    players: int,
    out_dir: Path,
    velocity_threshold: float,
) -> RunMetric:
    out_csv = out_dir / f"g{gamma:.3f}_p{power:.3f}" / condition / f"seed{seed}.csv"
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
        evolution_mode="personality_coupled",
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
        personality_coupling_mu_base=0.1,
        personality_coupling_lambda_mu=0.1,
        personality_coupling_lambda_k=0.2,
        personality_coupling_beta_state_k=0.0,
    )

    player_setup = _random_9persona_setup(seed) if condition == "random_9persona" else None
    strategy_space, rows = simulate(cfg, player_setup_callback=player_setup)
    _write_timeseries_csv(out_csv, strategy_space=strategy_space, rows=rows)

    series = _extract_series(rows)
    cyc = classify_cycle_level(series, burn_in=2000, tail=2000, amplitude_threshold=0.02, eta=0.6, min_turn_strength=0.0)
    rot = phase_rotation_r2(series, burn_in=2000, tail=2000)
    win_len = rot.window_length if rot.window_length > 0 else 1
    phase_velocity = float(rot.cumulative_rotation) / float(win_len)

    l3_pass = int(cyc.level) >= 3
    velocity_pass = phase_velocity > float(velocity_threshold)
    return RunMetric(
        gamma=float(gamma),
        power=float(power),
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
        return {"n": 0, "l3_rate": 0.0, "velocity_pass_rate": 0.0, "both_pass_rate": 0.0, "mean_phase_velocity": 0.0}
    return {
        "n": n,
        "l3_rate": sum(1 for m in metrics if m.l3_pass) / n,
        "velocity_pass_rate": sum(1 for m in metrics if m.velocity_pass) / n,
        "both_pass_rate": sum(1 for m in metrics if m.both_pass) / n,
        "mean_phase_velocity": sum(m.phase_velocity for m in metrics) / n,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Aggressive C-power scan (personality_coupled)")
    p.add_argument("--rounds", type=int, default=10000)
    p.add_argument("--players", type=int, default=300)
    p.add_argument("--velocity-threshold", type=float, default=0.001)
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s]
    gammas = [0.14, 0.15, 0.16]
    powers = [3.0, 3.1, 3.2]
    out_dir = ROOT / "outputs" / "c_power_aggressive_personality_coupled"
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = ["control_none", "random_9persona"]
    all_results: list[RunMetric] = []

    total = len(gammas) * len(powers) * len(conditions) * len(seeds)
    i = 0
    print(f"Aggressive C-power scan (personality_coupled): runs={total}")

    for g in gammas:
        for pwr in powers:
            combo_metrics: list[RunMetric] = []
            for cond in conditions:
                for seed in seeds:
                    i += 1
                    print(f"[{i}/{total}] g={g} p={pwr} {cond} seed={seed}", end=" ", flush=True)
                    m = _run_one(
                        gamma=g,
                        power=pwr,
                        condition=cond,
                        seed=seed,
                        rounds=int(args.rounds),
                        players=int(args.players),
                        out_dir=out_dir,
                        velocity_threshold=float(args.velocity_threshold),
                    )
                    combo_metrics.append(m)
                    all_results.append(m)
                    print(f"L{m.cycle_level}, v={m.phase_velocity:.6f}, both={'Y' if m.both_pass else 'N'}")

            # aggregate per combo
            by_cond = {cond: [m for m in combo_metrics if m.condition == cond] for cond in conditions}
            agg = {cond: _agg(vals) for cond, vals in by_cond.items()}
            summary = {
                "gamma": float(g),
                "power": float(pwr),
                "aggregate": agg,
            }
            combo_json = out_dir / f"g{g:.3f}_p{pwr:.3f}_summary.json"
            combo_json.write_text(json.dumps(summary, indent=2))

    # write full summary
    full = {
        "experiment": "Aggressive C-power scan (personality_coupled)",
        "gammas": gammas,
        "powers": powers,
        "seeds": seeds,
        "rounds": int(args.rounds),
        "players": int(args.players),
        "results": [asdict(r) for r in all_results],
    }
    (out_dir / "summary.json").write_text(json.dumps(full, indent=2))
    print(f"Saved full summary to {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
