#!/usr/bin/env python
"""
P7-H power analysis under realistic human noise.

The simulated apparatus produces a very large objective effect (max_proximity
d≈2.4–5.3) because the agent populations respond deterministically. Real humans
will be noisier and will not all engage with the manipulation. This script asks:
**under plausible human-noise scenarios, what N per group do we actually need,
and is the planned 106/group safe?**

Two independent, defensible degradations of the objective DV (max_proximity):

  1. Engagement attenuation `a` ∈ [0,1] — the fraction of players who actually
     respond to the manipulation. With probability (1−a) an experiment session is
     treated as non-responsive and its DV is drawn from the CONTROL distribution
     instead (the aligned events had no special effect for that player). This
     shrinks the between-group separation.
  2. Individual noise `σ` — additive Gaussian on max_proximity, then clipped to
     [0,1] (the DV is bounded). Inflates within-group variance.

Method: take the simulated noiseless per-group max_proximity distributions
(realistic feedback-ON, capped 4-event regime) as the ground truth, then
Monte-Carlo: for each (a, σ) draw n/group with replacement, apply the
degradations, run a one-sided Welch t-test, and estimate (i) power at the planned
N and (ii) the minimum n/group for 80% power.

H2 (subjective survey) is fully parametric in the simulator, so its power is
reported analytically across a range of assumed standardized effect sizes.

Usage
-----
  ./venv/bin/python scripts/experiments/power_p7h_human_noise.py \
      --planned-n 106 --reps 2000 --out reports/experiments/p7h_power
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from api.player_test_tracker import PlayerTestTracker
from scripts.experiments.run_p7h_engine_sim import (
    load_initial_distribution,
    simulate_engine_player,
)

_ZA = norm.ppf(1 - 0.025)
_ZB = norm.ppf(0.80)


# ── Ground-truth distributions from the noiseless apparatus ────────────────────

def base_distributions(n_sessions: int, agents: int, seed: int):
    """Empirical per-session max_proximity for exp/ctrl (feedback ON, capped)."""
    rng = np.random.RandomState(seed)
    tracker = PlayerTestTracker()
    init_pool = load_initial_distribution()
    n_exp = n_sessions // 2
    plan = (
        [(f"exp_{i:04d}", "experiment") for i in range(n_exp)]
        + [(f"ctrl_{i:04d}", "control") for i in range(n_sessions - n_exp)]
    )
    rng.shuffle(plan)
    for sid, group in plan:
        simulate_engine_player(
            session_id=sid, group=group, n_actions=12, rng=rng, tracker=tracker,
            init_pool=init_pool, event_every=3, intensity_scale=1.0,
            headroom=0.5, n_players=agents, feedback=True,
        )
    exp = np.array([s.max_proximity for s in tracker._sessions.values()
                    if s.group == "experiment"])
    ctrl = np.array([s.max_proximity for s in tracker._sessions.values()
                     if s.group == "control"])
    return exp, ctrl


# ── Human-noise sampling ───────────────────────────────────────────────────────

def _draw_group(base_self, base_other, n, attenuation, sigma, rng, is_experiment):
    vals = rng.choice(base_self, size=n, replace=True)
    if is_experiment and attenuation < 1.0:
        # Non-responders behave like the other (control) arm.
        non_resp = rng.random(n) >= attenuation
        if non_resp.any():
            vals[non_resp] = rng.choice(base_other, size=int(non_resp.sum()), replace=True)
    if sigma > 0:
        vals = vals + rng.normal(0, sigma, size=n)
    return np.clip(vals, 0.0, 1.0)


def mc_power(base_exp, base_ctrl, n, attenuation, sigma, reps, rng):
    sig = 0
    for _ in range(reps):
        e = _draw_group(base_exp, base_ctrl, n, attenuation, sigma, rng, True)
        c = _draw_group(base_ctrl, base_exp, n, 1.0, sigma, rng, False)
        if e.std() == 0 and c.std() == 0:
            sig += int(e.mean() > c.mean())
            continue
        t, p_two = stats.ttest_ind(e, c, equal_var=False)
        p_one = p_two / 2 if t > 0 else 1 - p_two / 2
        sig += int(p_one < 0.05)
    return sig / reps


def min_n_for_power(base_exp, base_ctrl, attenuation, sigma, reps, rng,
                    target=0.80, n_grid=(10, 15, 20, 25, 30, 40, 50, 64, 80, 106, 150, 212)):
    for n in n_grid:
        if mc_power(base_exp, base_ctrl, n, attenuation, sigma, reps, rng) >= target:
            return n
    return None


# ── H2 analytic power (parametric survey) ──────────────────────────────────────

def analytic_power(d, n):
    ncp = d * np.sqrt(n / 2)
    return float(1 - norm.cdf(_ZA - ncp) + norm.cdf(-_ZA - ncp))


def n_for_power(d, target=0.80):
    if d <= 0:
        return float("inf")
    return 2 * (_ZA + _ZB) ** 2 / d ** 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planned-n", type=int, default=106)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--base-sessions", type=int, default=400)
    ap.add_argument("--agents", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260606)
    ap.add_argument("--attenuations", type=float, nargs="+", default=[1.0, 0.6, 0.4, 0.25])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.15, 0.30])
    ap.add_argument("--out", type=str, default="reports/experiments/p7h_power")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    base_exp, base_ctrl = base_distributions(args.base_sessions, args.agents, args.seed)
    raw_d = (base_exp.mean() - base_ctrl.mean()) / np.sqrt(
        (base_exp.var(ddof=1) + base_ctrl.var(ddof=1)) / 2
    )

    print("=" * 76)
    print("P7-H POWER UNDER HUMAN NOISE  (primary DV = max_proximity)")
    print("=" * 76)
    print(f"Base (noiseless, feedback ON, capped) : "
          f"exp={base_exp.mean():.3f}±{base_exp.std(ddof=1):.3f}  "
          f"ctrl={base_ctrl.mean():.3f}±{base_ctrl.std(ddof=1):.3f}  raw d={raw_d:.2f}")
    print(f"Planned N/group = {args.planned_n}   MC reps = {args.reps}\n")

    print("H1 — power @ planned N  /  min N for 80% power, per scenario:")
    header = "  attenuation  " + "".join(f"σ={s:<4.2f}        " for s in args.sigmas)
    print(header)
    cells = []
    for a in args.attenuations:
        row = f"  a={a:<4.2f}      "
        for s in args.sigmas:
            pwr = mc_power(base_exp, base_ctrl, args.planned_n, a, s, args.reps, rng)
            nmin = min_n_for_power(base_exp, base_ctrl, a, s, max(args.reps // 2, 500), rng)
            nmin_s = f"{nmin}" if nmin is not None else ">212"
            row += f"{pwr*100:4.0f}% /N≥{nmin_s:<5s} "
            cells.append({"attenuation": a, "sigma": s,
                          "power_at_planned": pwr, "min_n_80": nmin})
        print(row)

    print("\nH2 (subjective UX) — analytic power, by assumed standardized effect d:")
    h2 = []
    for d in (0.2, 0.3, 0.5, 0.8):
        pwr = analytic_power(d, args.planned_n)
        need = n_for_power(d)
        print(f"  d={d:.1f}  power@{args.planned_n}={pwr*100:4.0f}%  N/group for 80%={need:.0f}")
        h2.append({"d": d, "power_at_planned": pwr, "n_for_80": need})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "primary_dv": "max_proximity",
        "base": {
            "exp_mean": float(base_exp.mean()), "exp_sd": float(base_exp.std(ddof=1)),
            "ctrl_mean": float(base_ctrl.mean()), "ctrl_sd": float(base_ctrl.std(ddof=1)),
            "raw_cohens_d": float(raw_d), "base_sessions": args.base_sessions,
        },
        "planned_n": args.planned_n, "mc_reps": args.reps,
        "H1_scenarios": cells, "H2_analytic": h2,
        "model_notes": (
            "attenuation a: P(experiment session responds to manipulation); "
            "non-responders drawn from control distribution. sigma: additive "
            "Gaussian on max_proximity, clipped to [0,1]."
        ),
    }
    json_path = out / "p7h_power_human_noise.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
