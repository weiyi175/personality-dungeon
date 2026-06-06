#!/usr/bin/env python
"""
P7-H apparatus validation — RL-engine Space-A event path.

Unlike run_p7h_player_test.py (which uses a standalone per-player decay model),
this driver exercises the REAL redesigned apparatus the live Godot study will
use:

  - a 300/50-agent RLSessionEngine per session,
  - events applied via engine.apply_personality_event() in Space A,
  - personality feedback dynamics (ΔP) ENABLED between events, so we genuinely
    test that an event's perturbation persists across RL rounds instead of being
    washed out (the P7-H memory's failure mode #2),
  - the DV read straight from the engine's Space-A displacement
    (snapshot.personality_displacement / mean_personality).

Each step's population-mean personality is written to a PlayerTestTracker in the
SAME JSON schema the live study produces, so the output can be fed directly to
the pre-registered analysis (analyze_p7h_real_study.py).

Predictions (H1): experiment (v1-aligned events) accumulates displacement
coherently while control (random direction, equal magnitude) random-walks, so
mean total_displacement(exp) > mean total_displacement(control), and proximity
should NOT saturate at 1.0 for everyone (the Space A/B realignment fix).

Usage
-----
  ./venv/bin/python scripts/experiments/run_p7h_engine_sim.py \
      --n-players 60 --n-actions 30 --event-every 3 \
      --out reports/experiments/p7h_engine_sim
  ./venv/bin/python scripts/experiments/analyze_p7h_real_study.py \
      --sessions reports/experiments/p7h_engine_sim/p7h_player_test_sessions.json \
      --survey   reports/experiments/p7h_engine_sim/p7h_survey_responses.json \
      --out      reports/experiments/p7h_engine_sim
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from players.rl_player import _PERSONALITY_KEYS
from simulation.bifurcation_detector import compute_bifurcation_distance
from simulation.event_generator import design_bifurcation_event
from simulation.rl_session_engine import RLSessionConfig, RLSessionEngine
from api.player_test_tracker import PlayerTestTracker
from api.survey_manager import SurveyManager
from api.ab_test_manager import ABTestManager

# Reuse the realistic initial-condition pool + survey synthesis from the
# standalone simulation so the two drivers are comparable.
from scripts.experiments.run_p7h_player_test import (
    _ACTION_TEMPLATES,
    load_initial_distribution,
    sample_initial,
    synthesise_survey,
)

_PROX_MODE = "projection"


def _proximity(pv: np.ndarray) -> float:
    return compute_bifurcation_distance(pv, mode=_PROX_MODE)["bifurcation_proximity"]


def _build_engine(
    session_id: str,
    p0: np.ndarray,
    n_players: int,
    feedback: bool,
    seed: int,
) -> RLSessionEngine:
    """One RL session whose population all start at p0 (Space A), events on."""
    config = RLSessionConfig(
        seed=seed,
        n_players=n_players,
        n_rounds=10_000_000,  # large: we never want "session complete" mid-run
        burn_in=0,
        personality_mode="static",
        fixed_personality_vector={k: float(p0[i]) for i, k in enumerate(_PERSONALITY_KEYS)},
        personality_update_enabled=feedback,
        personality_feedback_strength=0.5 if feedback else 0.0,
        space_a_events_enabled=True,
    )
    engine = RLSessionEngine(config, session_id=session_id)
    engine.reset()
    return engine


def simulate_engine_player(
    session_id: str,
    group: str,
    n_actions: int,
    rng: np.random.RandomState,
    tracker: PlayerTestTracker,
    init_pool: np.ndarray,
    event_every: int,
    intensity_scale: float,
    headroom: float,
    n_players: int,
    feedback: bool,
) -> None:
    """Drive one session through the RL-engine Space-A event path."""
    p0 = sample_initial(rng, init_pool, headroom=headroom)
    engine = _build_engine(
        session_id, p0, n_players=n_players, feedback=feedback,
        seed=int(rng.randint(1, 2**31 - 1)),
    )
    tracker.start_session(session_id, group, player_alias=f"eng_{session_id[-4:]}")

    for action_idx in range(n_actions):
        mean_before = np.asarray(engine.snapshot().mean_personality, dtype=float)
        prox_before = _proximity(mean_before)
        response_ms = float(rng.lognormal(mean=7.0, sigma=0.4))
        action_text = _ACTION_TEMPLATES[action_idx % len(_ACTION_TEMPLATES)]
        event_type = "none"

        if (action_idx + 1) % event_every == 0:
            # Design the event from the CURRENT population mean (Space A). Both
            # arms get the same magnitude; only the direction differs.
            if group == "experiment":
                ev = design_bifurcation_event(
                    mean_before, intensity_scale=intensity_scale,
                    app_calibrated=True, direction_mode="aligned",
                )
                event_type = "personality_shift"
            else:
                ev = design_bifurcation_event(
                    mean_before, intensity_scale=intensity_scale,
                    app_calibrated=True, direction_mode="random",
                    rng=np.random.RandomState(int(rng.randint(1, 2**31 - 1))),
                )
                event_type = "random_event"
            engine.apply_personality_event(ev["displacement"])
        else:
            # Free RL dynamics (ΔP runs if feedback enabled) — tests persistence.
            engine.step()

        mean_after = np.asarray(engine.snapshot().mean_personality, dtype=float)
        prox_after = _proximity(mean_after)

        tracker.record_step(
            session_id=session_id,
            action_text=action_text,
            personality_before=mean_before.tolist(),
            personality_after=mean_after.tolist(),
            proximity_before=prox_before,
            proximity_after=prox_after,
            event_type=event_type,
            response_time_ms=response_ms,
        )

    tracker.end_session(session_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-players", type=int, default=60,
                        help="Total simulated sessions (half control, half experiment)")
    parser.add_argument("--agents", type=int, default=50,
                        help="RL agents per session (population size)")
    parser.add_argument("--n-actions", type=int, default=30)
    parser.add_argument("--event-every", type=int, default=3)
    parser.add_argument("--intensity-scale", type=float, default=1.0)
    parser.add_argument("--headroom", type=float, default=0.5)
    parser.add_argument("--no-feedback", action="store_true",
                        help="Disable ΔP personality dynamics (isolate event effect)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str,
                        default="reports/experiments/p7h_engine_sim")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    tracker = PlayerTestTracker()
    survey = SurveyManager()
    ab = ABTestManager(seed=args.seed)
    init_pool = load_initial_distribution()
    feedback = not args.no_feedback

    n_experiment = args.n_players // 2
    n_control = args.n_players - n_experiment
    sessions = (
        [(f"exp_{i:03d}", "experiment") for i in range(n_experiment)]
        + [(f"ctrl_{i:03d}", "control") for i in range(n_control)]
    )
    rng.shuffle(sessions)

    t0 = time.time()
    for sid, group in sessions:
        ab.assign_session(sid)
        ab._sessions[sid].group = group  # type: ignore[attr-defined]
        simulate_engine_player(
            session_id=sid, group=group, n_actions=args.n_actions, rng=rng,
            tracker=tracker, init_pool=init_pool, event_every=args.event_every,
            intensity_scale=args.intensity_scale, headroom=args.headroom,
            n_players=args.agents, feedback=feedback,
        )

    for sid, group in sessions:
        sess = tracker._sessions[sid]  # type: ignore[attr-defined]
        synthesise_survey(sid, group, sess.max_proximity, rng, survey)

    elapsed = time.time() - t0
    out = Path(args.out)
    t_path = tracker.save(out)
    s_path = survey.save(out)
    gsummary = tracker.group_summary()
    ssummary = survey.summary()

    print("=" * 70)
    print("P7-H APPARATUS VALIDATION — RL-ENGINE SPACE-A EVENT PATH")
    print("=" * 70)
    print(f"\nConfig: sessions={args.n_players}, agents/session={args.agents}, "
          f"n_actions={args.n_actions}, event_every={args.event_every}, "
          f"feedback={'on' if feedback else 'off'}")
    print(f"Elapsed: {elapsed:.2f}s\n")

    print("── Trajectory Metrics (Space-A displacement DV) ───────")
    for grp in ("control", "experiment"):
        g = gsummary.get(grp, {})
        if not g.get("n_total"):
            continue
        print(f"  {grp:12s}  n={g['n_total']:<3}  "
              f"mean_disp={g['mean_displacement']:.5f}  "
              f"mean_max_prox={g['mean_max_proximity']:.3f}  "
              f"crossings={g['mean_critical_crossings']:.2f}")
    print(f"  Cohen's d = {gsummary['effect_size_cohens_d']:+.3f}")

    exp = gsummary.get("experiment", {})
    ctrl = gsummary.get("control", {})
    direction_ok = exp.get("mean_displacement", 0) > ctrl.get("mean_displacement", 0)
    not_saturated = max(exp.get("mean_max_proximity", 1.0),
                        ctrl.get("mean_max_proximity", 1.0)) < 0.999
    print("\n── Apparatus checks ───────────────────────────────────")
    print(f"  H1 direction exp>ctrl : {'PASS' if direction_ok else 'FAIL'}")
    print(f"  proximity not saturated: {'PASS' if not_saturated else 'WARN (all ~1.0)'}")

    print(f"\nReports saved:\n  {t_path}\n  {s_path}")
    print("\nNext: run analyze_p7h_real_study.py on the sessions/survey JSON "
          "for the pre-registered H1/H2/H3 stats.")


if __name__ == "__main__":
    main()
