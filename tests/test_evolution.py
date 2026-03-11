"""演化層測試（SDD invariants）。"""

import math

import pytest

from evolution.replicator_dynamics import replicator_step


class _P:
    def __init__(self, s, r):
        self.last_strategy = s
        self.last_reward = r


def test_replicator_step_negative_selection_strength_raises():
    with pytest.raises(ValueError):
        replicator_step([], ["a", "b"], selection_strength=-0.01)


def test_replicator_step_returns_mean_one_positive_finite():
    players = [
        _P("aggressive", 1.0),
        _P("aggressive", 1.0),
        _P("defensive", 2.0),
        _P("balanced", 0.0),
    ]
    ws = replicator_step(players, ["aggressive", "defensive", "balanced"], selection_strength=0.5)
    assert set(ws.keys()) == {"aggressive", "defensive", "balanced"}
    vals = list(ws.values())
    assert all((v > 0 and math.isfinite(v)) for v in vals)
    mean_w = sum(vals) / len(vals)
    assert mean_w == pytest.approx(1.0, abs=1e-9)


def test_replicator_step_no_rewards_returns_all_ones():
    players = [_P("aggressive", None), _P(None, 1.0)]
    ws = replicator_step(players, ["aggressive", "defensive", "balanced"], selection_strength=0.2)
    assert ws == {"aggressive": 1.0, "defensive": 1.0, "balanced": 1.0}
