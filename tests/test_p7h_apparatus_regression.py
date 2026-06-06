"""Regression guard for the validated P7-H apparatus.

Locks in the redesigned apparatus behaviour (Space A/B realignment + Space-A
engine event path) so a future change cannot silently break the experiment:
in the capped sub-saturation regime, the experiment arm (v1-aligned events)
must reach higher bifurcation proximity than the control arm (random direction,
equal magnitude), and control must NOT be pinned at saturation (the original
Space A/B coordinate-mismatch failure mode).

This is a fast, small-N smoke version of scripts/experiments/run_p7h_engine_sim.py
+ analyze_p7h_real_study.py (the full validation is in
reports/experiments/p7h_engine_sim/REGIME_FINDING.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from api.player_test_tracker import PlayerTestTracker
from scripts.experiments.analyze_p7h_real_study import cohens_d
from scripts.experiments.run_p7h_engine_sim import (
    load_initial_distribution,
    simulate_engine_player,
)


def _run(feedback: bool, seed: int = 7, sessions: int = 16, agents: int = 20):
    rng = np.random.RandomState(seed)
    tracker = PlayerTestTracker()
    init_pool = load_initial_distribution()
    n_exp = sessions // 2
    plan = (
        [(f"exp_{i:02d}", "experiment") for i in range(n_exp)]
        + [(f"ctrl_{i:02d}", "control") for i in range(sessions - n_exp)]
    )
    rng.shuffle(plan)
    for sid, group in plan:
        simulate_engine_player(
            session_id=sid, group=group, n_actions=12, rng=rng, tracker=tracker,
            init_pool=init_pool, event_every=3, intensity_scale=1.0,
            headroom=0.5, n_players=agents, feedback=feedback,
        )
    exp = np.array([s.max_proximity for s in tracker._sessions.values()
                    if s.group == "experiment"])
    ctrl = np.array([s.max_proximity for s in tracker._sessions.values()
                     if s.group == "control"])
    return exp, ctrl


@pytest.mark.parametrize("feedback", [True, False])
def test_experiment_reaches_higher_proximity_than_control(feedback):
    exp, ctrl = _run(feedback=feedback)
    assert exp.mean() > ctrl.mean(), (
        f"apparatus regressed: exp max_proximity {exp.mean():.3f} "
        f"!> ctrl {ctrl.mean():.3f} (feedback={feedback})"
    )
    d, _ = cohens_d(exp, ctrl)
    # Validated effect is huge (d>2 even in the hardest cell); guard well below
    # that so the test is stable but still catches a real regression.
    assert d > 1.0, f"effect collapsed: Cohen's d={d:.2f} (feedback={feedback})"


def test_control_is_not_saturated():
    """The Space A/B fix means control proximity must vary, not pin at ~1.0.

    (The original broken apparatus returned proximity≈1.0 for every vector.)
    """
    _exp, ctrl = _run(feedback=False)
    assert ctrl.mean() < 0.9, (
        f"control proximity saturated ({ctrl.mean():.3f}) — Space A/B mismatch back?"
    )


def test_apply_event_displacement_persists_in_engine_path():
    """Sanity: events actually move the population (DV > 0) via the engine path."""
    exp, _ctrl = _run(feedback=False)
    # Experiment arm drives proximity up materially in the capped regime.
    assert exp.mean() > 0.6
