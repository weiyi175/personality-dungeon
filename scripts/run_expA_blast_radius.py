"""Exp A — comparative trait blast-radius ablation.

Question: if we *redefine* a single trait (replace its construct), how far does
that drag the population dynamics?  We bound it by the most violent perturbation
— ablating the trait to 0 every round — and measure how far the resulting
strategy-share trajectory drifts from the unperturbed baseline (same seed, so all
drift is causally attributable to that one trait).

Candidates:
  suspicion, optimism   -- formula-free (not in signal_mu/k, not in recklessness R)
  curiosity             -- contrast: in recklessness R, not in mu/k
  endurance             -- contrast: in signal_mu AND in R  (expected largest drift)

Primary DV  : trajectory_drift = mean_t [ |dp_agg| + |dp_def| + |dp_bal| ]   vs baseline
Secondary DV: cycle level (short/long window) + stage3 score (noisy; sanity only)

Output: outputs/expA_blast_radius/expA_summary.json  + console ranking table.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.cycle_metrics import classify_cycle_level
from simulation.personality_gate0 import DEFAULT_FULL_EVENTS_JSON
from simulation.run_simulation import SimConfig, simulate

OUT_DIR = ROOT / "outputs" / "expA_blast_radius"
ROUNDS = 4000
PLAYERS = 300
SEEDS = [91, 7, 23]
SELECTION_STRENGTH = 0.10
INIT_BIAS = 0.12
MEMORY_KERNEL = 3
CANDIDATES = ["suspicion", "optimism", "curiosity", "endurance"]


def _ablate_trait(players: list[object], trait: str, scale: float) -> None:
    for pl in players:
        personality = getattr(pl, "personality", None)
        if not isinstance(personality, dict):
            continue
        base = float(personality.get(trait, 0.0))
        personality[trait] = max(-1.0, min(1.0, base * float(scale)))


def _series(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {
        "aggressive": [float(r["p_aggressive"]) for r in rows],
        "defensive": [float(r["p_defensive"]) for r in rows],
        "balanced": [float(r["p_balanced"]) for r in rows],
    }


def _cfg(seed: int, csv: Path) -> SimConfig:
    return SimConfig(
        n_players=PLAYERS, n_rounds=ROUNDS, seed=seed,
        payoff_mode="matrix_ab", popularity_mode="sampled",
        gamma=0.1, epsilon=0.0, a=1.05, b=0.85,
        matrix_cross_coupling=0.35, init_bias=INIT_BIAS,
        evolution_mode="sampled", payoff_lag=1,
        selection_strength=SELECTION_STRENGTH,
        enable_events=True, events_json=DEFAULT_FULL_EVENTS_JSON,
        out_csv=csv, memory_kernel=MEMORY_KERNEL,
    )


def _run(seed: int, trait: str | None) -> dict[str, list[float]]:
    csv = Path("/tmp") / f"expA_{trait or 'baseline'}_{seed}.csv"
    cfg = _cfg(seed, csv)
    if trait is None:
        _, rows = simulate(cfg)
    else:
        def _setup(players, _s, _c):
            _ablate_trait(players, trait, 0.0)

        def _round_cb(_i, _c, players, _d, _sr, _ctx):
            _ablate_trait(players, trait, 0.0)

        _, rows = simulate(cfg, player_setup_callback=_setup, round_callback=_round_cb)
    return _series(rows)


def _drift(base: Mapping[str, list[float]], pert: Mapping[str, list[float]]) -> float:
    n = min(len(base["aggressive"]), len(pert["aggressive"]))
    if n == 0:
        return 0.0
    total = 0.0
    for t in range(n):
        total += (
            abs(base["aggressive"][t] - pert["aggressive"][t])
            + abs(base["defensive"][t] - pert["defensive"][t])
            + abs(base["balanced"][t] - pert["balanced"][t])
        )
    return total / n


def _cycle(series: Mapping[str, list[float]], burn: int, tail: int) -> dict[str, float]:
    cyc = classify_cycle_level(
        series, burn_in=burn, tail=tail,
        amplitude_threshold=0.02, corr_threshold=0.09, eta=0.55,
        stage3_method="turning", phase_smoothing=1, min_lag=2, max_lag=500,
    )
    return {"level": int(cyc.level), "stage3": float(cyc.stage3.score if cyc.stage3 else 0.0)}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    per_seed: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        base = _run(seed, None)
        rec: dict[str, Any] = {
            "baseline_cycle_short": _cycle(base, 1000, 1000),
            "baseline_cycle_long": _cycle(base, 2000, 2000),
            "candidates": {},
        }
        for trait in CANDIDATES:
            pert = _run(seed, trait)
            rec["candidates"][trait] = {
                "trajectory_drift": _drift(base, pert),
                "cycle_short": _cycle(pert, 1000, 1000),
                "cycle_long": _cycle(pert, 2000, 2000),
            }
            print(f"[seed {seed}] {trait:18s} drift={rec['candidates'][trait]['trajectory_drift']:.5f}", flush=True)
        per_seed[seed] = rec

    ranking = []
    for trait in CANDIDATES:
        drifts = [per_seed[s]["candidates"][trait]["trajectory_drift"] for s in SEEDS]
        ranking.append({
            "trait": trait,
            "mean_drift": sum(drifts) / len(drifts),
            "drifts_per_seed": dict(zip([str(s) for s in SEEDS], [round(d, 5) for d in drifts])),
        })
    ranking.sort(key=lambda r: r["mean_drift"])

    payload = {
        "scope": {"rounds": ROUNDS, "players": PLAYERS, "seeds": SEEDS,
                  "ablation": "scale single trait -> 0.0 every round (upper-bound perturbation)",
                  "primary_dv": "trajectory_drift = mean_t L1 distance of (agg,def,bal) shares vs same-seed baseline",
                  "elapsed_s": round(time.time() - t0, 1)},
        "ranking_safest_first": ranking,
        "per_seed": per_seed,
    }
    (OUT_DIR / "expA_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Exp A blast-radius ranking (safest=smallest drift first) ===")
    for r in ranking:
        print(f"  {r['trait']:18s} mean_drift={r['mean_drift']:.5f}  per_seed={r['drifts_per_seed']}")
    print(f"\nwritten: {(OUT_DIR / 'expA_summary.json').relative_to(ROOT)}  (elapsed {payload['scope']['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
