#!/usr/bin/env python
"""
P7-H primary-DV robustness sweep.

Confirms the chosen primary objective DV (`max_proximity`) separates
experiment from control **stably** — not just for a lucky seed or a single
tuning — across the conditions the live study will actually run under:

  - ΔP personality feedback ON (realistic engine dynamics) and OFF (isolation),
  - a range of event intensities,
  - many random seeds,

all in the capped, sub-saturation regime (≤4 events / session) adopted in the
pre-registration. For each (feedback, intensity) cell it reports the
distribution of Cohen's d on max_proximity over seeds, plus the fraction of
seeds reaching one-sided p<0.05. This gives a defensible effect-size range for
the pre-reg power section.

Usage
-----
  ./venv/bin/python scripts/experiments/sweep_p7h_robustness.py \
      --seeds 8 --sessions 40 --out reports/experiments/p7h_engine_sim
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from api.player_test_tracker import PlayerTestTracker
from scripts.experiments.analyze_p7h_real_study import cohens_d
from scripts.experiments.run_p7h_engine_sim import (
    load_initial_distribution,
    simulate_engine_player,
)

DV = "max_proximity"


def _one_run(seed, sessions, n_actions, event_every, intensity, headroom,
             agents, feedback, init_pool):
    rng = np.random.RandomState(seed)
    tracker = PlayerTestTracker()
    n_exp = sessions // 2
    n_ctrl = sessions - n_exp
    plan = (
        [(f"exp_{i:03d}", "experiment") for i in range(n_exp)]
        + [(f"ctrl_{i:03d}", "control") for i in range(n_ctrl)]
    )
    rng.shuffle(plan)
    for sid, group in plan:
        simulate_engine_player(
            session_id=sid, group=group, n_actions=n_actions, rng=rng,
            tracker=tracker, init_pool=init_pool, event_every=event_every,
            intensity_scale=intensity, headroom=headroom, n_players=agents,
            feedback=feedback,
        )
    exp = np.array([s.max_proximity for s in tracker._sessions.values()
                    if s.group == "experiment"])
    ctrl = np.array([s.max_proximity for s in tracker._sessions.values()
                     if s.group == "control"])
    t, p_two = stats.ttest_ind(exp, ctrl, equal_var=False)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    d, _ = cohens_d(exp, ctrl)
    return {"d": float(d), "p_one": float(p_one),
            "exp_mean": float(exp.mean()), "ctrl_mean": float(ctrl.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--sessions", type=int, default=40)
    ap.add_argument("--n-actions", type=int, default=12, help="capped regime (4 events)")
    ap.add_argument("--event-every", type=int, default=3)
    ap.add_argument("--headroom", type=float, default=0.5)
    ap.add_argument("--agents", type=int, default=50)
    ap.add_argument("--intensities", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--out", type=str, default="reports/experiments/p7h_engine_sim")
    args = ap.parse_args()

    init_pool = load_initial_distribution()
    cells = []
    for feedback in (True, False):
        for intensity in args.intensities:
            ds, ps = [], []
            for s in range(args.seeds):
                r = _one_run(
                    seed=1000 + s, sessions=args.sessions, n_actions=args.n_actions,
                    event_every=args.event_every, intensity=intensity,
                    headroom=args.headroom, agents=args.agents, feedback=feedback,
                    init_pool=init_pool,
                )
                ds.append(r["d"])
                ps.append(r["p_one"])
            ds = np.array(ds)
            ps = np.array(ps)
            cells.append({
                "feedback": feedback,
                "intensity_scale": intensity,
                "n_seeds": args.seeds,
                "d_mean": float(ds.mean()),
                "d_sd": float(ds.std(ddof=1)),
                "d_min": float(ds.min()),
                "d_max": float(ds.max()),
                "frac_sig_p05": float((ps < 0.05).mean()),
            })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "dv": DV,
        "regime": {
            "n_actions": args.n_actions, "event_every": args.event_every,
            "events_per_session": args.n_actions // args.event_every,
            "sessions_per_run": args.sessions, "agents": args.agents,
            "headroom": args.headroom,
        },
        "cells": cells,
    }
    json_path = out / "p7h_robustness_sweep.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"P7-H PRIMARY-DV ROBUSTNESS SWEEP  (DV={DV})")
    print("=" * 72)
    print(f"Regime: {args.n_actions} actions / {args.n_actions // args.event_every} "
          f"events, {args.sessions} sessions/run, {args.agents} agents, "
          f"{args.seeds} seeds/cell\n")
    print(f"{'feedback':9s} {'intensity':9s} {'d_mean':>8s} {'d_sd':>7s} "
          f"{'d_min':>7s} {'d_max':>7s} {'%sig':>6s}")
    for c in cells:
        print(f"{('on' if c['feedback'] else 'off'):9s} "
              f"{c['intensity_scale']:<9.2f} {c['d_mean']:>8.2f} {c['d_sd']:>7.2f} "
              f"{c['d_min']:>7.2f} {c['d_max']:>7.2f} {c['frac_sig_p05']*100:>5.0f}%")
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
