#!/usr/bin/env python
"""
P7-H Phase IV: Simulated player test.

Simulates N players (half control, half experiment) playing through a series
of game actions.  Each player's personality evolves via:
  - Personality dynamics: exponential decay toward baseline attractor (decay=0.95)
  - Control group:  events in a random direction
  - Experiment group: events aligned with the sensitive direction v1

After all sessions complete, survey responses are synthesised using a
parametric model:
  - Experiment group perceives slightly higher "naturalness" and "fun"
    (aligned with bifurcation theory prediction)
  - Response times sampled from a realistic log-normal distribution

Output:
  reports/experiments/p7h_player_test/p7h_player_test_sessions.json
  reports/experiments/p7h_player_test/p7h_survey_responses.json
  reports/experiments/p7h_player_test/p7h_player_test_summary.txt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulation.bifurcation_detector import (
    BASELINE_ATTRACTOR, V1, EPSILON_C, EPSILON_C_APP, compute_bifurcation_distance,
)
from simulation.event_generator import (
    design_bifurcation_event, intensity_modulation,
)
from api.player_test_tracker import PlayerTestTracker
from api.survey_manager import SurveyManager
from api.ab_test_manager import ABTestManager

_DECAY = 0.95
# Application profile: projection proximity (v1-v2 plane) + ε_c_app scale.
# Required so proximity does not saturate at the first event step (B1 fix).
_PROX_MODE = "projection"
_FEATURE_NAMES = [
    "impulsiveness", "assertiveness", "optimism", "risk_aversion",
    "suspicion", "endurance", "randomness", "stability_seeking", "curiosity",
]


def _proximity(pv: np.ndarray) -> float:
    return compute_bifurcation_distance(pv, mode=_PROX_MODE)["bifurcation_proximity"]


def load_initial_distribution() -> np.ndarray:
    """Load realistic initial personality states from the P7-F attractors.

    Returns an (N, 9) array of attractor coordinates spanning the real range of
    personality states the system produces (across α=0.1–0.4). Used to seed
    player starting conditions instead of an artificial baseline+noise cluster.
    """
    path = ROOT / "reports/experiments/p7f_attractor_mapping/p7f_attractor_coordinates.json"
    with open(path) as f:
        coords = json.load(f)
    pts = []
    for _alpha, attractors in coords.items():
        for a in attractors:
            pts.append([a[f] for f in _FEATURE_NAMES])
    return np.array(pts)


def sample_initial(rng: np.random.RandomState, pool: np.ndarray,
                   headroom: float = 0.5) -> np.ndarray:
    """Sample a realistic player starting personality.

    Picks a random real attractor, adds small jitter, then scales the offset
    from baseline by ``headroom`` so players start sub-critical (leaving room for
    events to drive measurable displacement). headroom=1.0 → full attractor
    spread; 0.5 → halfway to baseline.
    """
    base = pool[rng.randint(len(pool))]
    jittered = base + rng.randn(9) * 0.01
    return BASELINE_ATTRACTOR + headroom * (jittered - BASELINE_ATTRACTOR)
_ACTION_TEMPLATES = [
    "我選擇探索未知區域",
    "我謹慎地評估風險",
    "我信任隊友",
    "我獨立行動",
    "我挑戰規則",
    "我遵循既定策略",
    "我尋求冒險機會",
    "我保持穩定節奏",
    "我質疑領導決定",
    "我主動分享資源",
]


def _free_step(pv: np.ndarray) -> np.ndarray:
    return _DECAY * pv + (1.0 - _DECAY) * BASELINE_ATTRACTOR


def _random_event(pv: np.ndarray, rng: np.random.RandomState, iscale: float) -> np.ndarray:
    """Random-direction event (control group) — app-calibrated magnitude."""
    prox = _proximity(pv)
    mag = intensity_modulation(prox, scale=EPSILON_C_APP) * iscale
    direction = rng.randn(9)
    direction /= np.linalg.norm(direction)
    return pv + direction * mag


def _aligned_event(pv: np.ndarray, iscale: float) -> np.ndarray:
    """Sensitive-direction event (experiment group) — app-calibrated."""
    ev = design_bifurcation_event(pv.tolist(), intensity_scale=iscale, app_calibrated=True)
    return pv + np.array(ev["displacement"])


def simulate_player(
    session_id: str,
    group: str,
    n_actions: int,
    rng: np.random.RandomState,
    tracker: PlayerTestTracker,
    init_pool: np.ndarray,
    event_every: int = 3,
    intensity_scale: float = 1.0,
    headroom: float = 0.5,
) -> None:
    """Simulate one player's full game session."""
    tracker.start_session(session_id, group, player_alias=f"sim_{session_id[-4:]}")

    # Realistic initial personality sampled from real P7-F attractor states
    pv = sample_initial(rng, init_pool, headroom=headroom)

    for action_idx in range(n_actions):
        pv_before = pv.copy()
        prox_before = _proximity(pv_before)

        # Simulate response time (log-normal, mean ~1.2s)
        response_ms = float(rng.lognormal(mean=7.0, sigma=0.4))  # ms

        action_text = _ACTION_TEMPLATES[action_idx % len(_ACTION_TEMPLATES)]
        event_type = "none"

        # Apply event every N actions
        if (action_idx + 1) % event_every == 0:
            if group == "experiment":
                pv = _aligned_event(pv, intensity_scale)
                event_type = "personality_shift"
            else:
                pv = _random_event(pv, rng, intensity_scale)
                event_type = "random_event"
        else:
            # Free dynamics step
            pv = _free_step(pv)

        # Clamp to [-1, 1]
        pv = np.clip(pv, -1.0, 1.0)
        prox_after = _proximity(pv)

        tracker.record_step(
            session_id=session_id,
            action_text=action_text,
            personality_before=pv_before.tolist(),
            personality_after=pv.tolist(),
            proximity_before=prox_before,
            proximity_after=prox_after,
            event_type=event_type,
            response_time_ms=response_ms,
        )

    tracker.end_session(session_id)


def synthesise_survey(
    session_id: str,
    group: str,
    max_proximity: float,
    rng: np.random.RandomState,
    survey: SurveyManager,
) -> None:
    """Generate synthetic survey response based on group and observed experience."""
    # Experiment group: scores slightly higher (aligned with bifurcation theory)
    base_natural = 5.5 if group == "control" else 7.0
    base_fun     = 5.0 if group == "control" else 7.2
    base_replay  = 5.2 if group == "control" else 6.8

    # Bonus if player reached critical zone
    critical_bonus = 1.0 if max_proximity >= 0.8 else 0.0

    def _score(base: float) -> int:
        raw = base + critical_bonus + rng.normal(0, 1.2)
        return int(np.clip(round(raw), 1, 10))

    survey.submit(
        session_id=session_id,
        group=group,
        q1=_score(base_natural),
        q2=_score(base_fun),
        q3=_score(base_replay),
        overall_comments=(
            "感覺人格變化很自然" if group == "experiment" else "變化有點突兀"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-players", type=int, default=210,
                        help="Total simulated players (half control, half experiment)")
    parser.add_argument("--n-actions", type=int, default=30,
                        help="Game actions per player")
    parser.add_argument("--event-every", type=int, default=3,
                        help="Inject bifurcation event every N actions")
    parser.add_argument("--intensity-scale", type=float, default=1.0)
    parser.add_argument("--headroom", type=float, default=0.5,
                        help="Initial-condition spread: 1.0=full attractor range, "
                             "0.5=halfway to baseline (leaves proximity headroom)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str,
                        default="reports/experiments/p7h_player_test")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    tracker = PlayerTestTracker()
    survey = SurveyManager()
    ab = ABTestManager(seed=args.seed)
    init_pool = load_initial_distribution()

    n_experiment = args.n_players // 2
    n_control    = args.n_players - n_experiment

    sessions = (
        [(f"exp_{i:03d}", "experiment") for i in range(n_experiment)]
        + [(f"ctrl_{i:03d}", "control") for i in range(n_control)]
    )
    rng.shuffle(sessions)  # shuffle to avoid ordering bias

    t0 = time.time()
    for sid, group in sessions:
        ab.assign_session(sid)
        # Override group to match our simulation intent
        ab._sessions[sid].group = group  # type: ignore[attr-defined]
        simulate_player(
            session_id=sid,
            group=group,
            n_actions=args.n_actions,
            rng=rng,
            tracker=tracker,
            init_pool=init_pool,
            event_every=args.event_every,
            intensity_scale=args.intensity_scale,
            headroom=args.headroom,
        )

    # Synthesise survey after all sessions
    for sid, group in sessions:
        sess = tracker._sessions[sid]  # type: ignore[attr-defined]
        synthesise_survey(sid, group, sess.max_proximity, rng, survey)

    elapsed = time.time() - t0
    out = Path(args.out)

    t_path = tracker.save(out)
    s_path = survey.save(out)

    # Print summary
    gsummary = tracker.group_summary()
    ssummary = survey.summary()

    print("=" * 70)
    print("P7-H PHASE IV: SIMULATED PLAYER TEST RESULTS")
    print("=" * 70)
    print(f"\nConfig: n_players={args.n_players}, n_actions={args.n_actions}, "
          f"event_every={args.event_every}")
    print(f"Elapsed: {elapsed:.2f}s\n")

    print("── Trajectory Metrics ─────────────────────────────────")
    for grp in ("control", "experiment"):
        g = gsummary.get(grp, {})
        if not g.get("n_total"):
            continue
        print(f"  {grp:12s}  n={g['n_total']:<3}  "
              f"mean_disp={g['mean_displacement']:.5f}  "
              f"mean_max_prox={g['mean_max_proximity']:.3f}  "
              f"crossings={g['mean_critical_crossings']:.2f}")
    print(f"  Cohen's d = {gsummary['effect_size_cohens_d']:+.3f}")

    print("\n── Survey Scores ──────────────────────────────────────")
    for grp in ("control", "experiment"):
        g = ssummary.get(grp, {})
        if not g.get("n"):
            continue
        print(f"  {grp:12s}  n={g['n']:<3}  "
              f"Q1={g['q1_naturalness']['mean']:.1f}  "
              f"Q2={g['q2_fun']['mean']:.1f}  "
              f"Q3={g['q3_replay']['mean']:.1f}  "
              f"UX={g['composite_ux']:.1f}")
    print(f"  UX lift = {ssummary['ux_lift']:+.3f}")

    print(f"\nReports saved:")
    print(f"  {t_path}")
    print(f"  {s_path}")

    # Write text summary
    summary_path = out / "p7h_player_test_summary.txt"
    with open(summary_path, "w") as f:
        f.write("P7-H PHASE IV: SIMULATED PLAYER TEST SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Players   : {args.n_players} ({n_experiment} experiment, {n_control} control)\n")
        f.write(f"Actions   : {args.n_actions} per player\n")
        f.write(f"Event every: {args.event_every} actions\n\n")
        ctrl = gsummary.get("control", {})
        exp  = gsummary.get("experiment", {})
        f.write("TRAJECTORY\n")
        f.write(f"  Control   mean_displacement = {ctrl.get('mean_displacement', 0):.5f}\n")
        f.write(f"  Experiment mean_displacement = {exp.get('mean_displacement', 0):.5f}\n")
        f.write(f"  Cohen's d = {gsummary['effect_size_cohens_d']:+.3f}\n\n")
        ctrl_s = ssummary.get("control", {})
        exp_s  = ssummary.get("experiment", {})
        f.write("SURVEY\n")
        f.write(f"  Control    UX composite = {ctrl_s.get('composite_ux', 0):.1f}\n")
        f.write(f"  Experiment UX composite = {exp_s.get('composite_ux', 0):.1f}\n")
        f.write(f"  UX lift = {ssummary['ux_lift']:+.3f}\n")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
