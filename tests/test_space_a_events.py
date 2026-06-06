"""Tests for the Space-A bifurcation event path (P7-H).

Covers RLSessionEngine.apply_personality_event and the mean_personality /
personality_displacement snapshot fields — the engine half of the A↔B
realignment (events are designed, applied AND measured in Space A so the DV
is coordinate-consistent and persists across rounds).
"""

from __future__ import annotations

import numpy as np
import pytest

from players.rl_player import _PERSONALITY_KEYS
from simulation.rl_session_engine import RLSessionConfig, RLSessionEngine

DIM = len(_PERSONALITY_KEYS)


def _engine(events=True, personality_mode="random_9persona", **kw):
    config = RLSessionConfig(
        seed=42,
        n_players=20,
        burn_in=10,
        personality_mode=personality_mode,
        space_a_events_enabled=events,
        **kw,
    )
    eng = RLSessionEngine(config, session_id="evt")
    eng.reset()
    return eng


def test_snapshot_exposes_space_a_personality():
    eng = _engine()
    snap = eng.snapshot()
    assert len(snap.mean_personality) == DIM
    # No event applied yet → displacement DV is zero.
    assert snap.personality_displacement == pytest.approx(0.0)


def test_apply_event_requires_flag():
    eng = _engine(events=False)
    with pytest.raises(RuntimeError):
        eng.apply_personality_event([0.1] * DIM)


def test_apply_event_rejects_wrong_length():
    eng = _engine()
    with pytest.raises(ValueError):
        eng.apply_personality_event([0.1] * (DIM - 1))


def test_apply_event_shifts_mean_and_dv():
    # Static zero personality so a small push stays interior (no clamping).
    eng = _engine(
        personality_mode="static",
        fixed_personality_vector={k: 0.0 for k in _PERSONALITY_KEYS},
    )
    before = np.array(eng.snapshot().mean_personality)
    disp = np.array([0.05] * DIM)
    eng.apply_personality_event(disp)
    snap = eng.snapshot()
    after = np.array(snap.mean_personality)
    # Mean shifts by the displacement (modulo clamping; values are interior here).
    assert np.allclose(after - before, disp, atol=1e-6)
    assert snap.personality_displacement == pytest.approx(
        float(np.linalg.norm(disp)), abs=1e-6
    )


def test_event_persists_across_steps():
    """With personality feedback off, an applied event must not wash out."""
    eng = _engine()  # personality_update_enabled defaults False
    eng.apply_personality_event([0.03] * DIM)
    dv_after_event = eng.snapshot().personality_displacement
    for _ in range(5):
        eng.step()
    dv_after_steps = eng.snapshot().personality_displacement
    assert dv_after_steps == pytest.approx(dv_after_event, abs=1e-9)
    assert dv_after_steps > 0.0


def test_apply_event_clamps_to_unit_box():
    eng = _engine()
    eng.apply_personality_event([10.0] * DIM)  # huge push
    mean = np.array(eng.snapshot().mean_personality)
    assert np.all(mean <= 1.0 + 1e-9)
    assert np.all(mean >= -1.0 - 1e-9)


def test_reset_rebaselines_dv():
    eng = _engine()
    eng.apply_personality_event([0.1] * DIM)
    assert eng.snapshot().personality_displacement > 0.0
    eng.reset()
    assert eng.snapshot().personality_displacement == pytest.approx(0.0)
