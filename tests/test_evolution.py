"""演化層測試（SDD invariants）。"""

import math

import pytest

from evolution.replicator_dynamics import anchored_subgroup_payoff_shift, anchored_subgroup_weight_pull, apply_tangential_drift, bidirectional_subgroup_payoff_shifts, deterministic_replicator_step, inertial_deterministic_replicator_step, inertial_sampled_replicator_step, replicator_step, state_dependent_anchored_subgroup_payoff_shift, tangential_drift_vector


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


def test_replicator_step_equal_hetero_strengths_match_homogeneous_case():
    players = [
        _P("aggressive", 1.0),
        _P("aggressive", 1.0),
        _P("defensive", 2.0),
        _P("balanced", 0.0),
    ]
    base = replicator_step(players, ["aggressive", "defensive", "balanced"], selection_strength=0.5)
    hetero = replicator_step(
        players,
        ["aggressive", "defensive", "balanced"],
        selection_strength=0.5,
        strategy_selection_strengths=(0.5, 0.5, 0.5),
    )
    assert hetero == pytest.approx(base, abs=0.0)


def test_replicator_step_invalid_hetero_strengths_raise():
    with pytest.raises(ValueError):
        replicator_step(
            [],
            ["aggressive", "defensive", "balanced"],
            selection_strength=0.2,
            strategy_selection_strengths=(0.2, 0.1),
        )
    with pytest.raises(ValueError):
        replicator_step(
            [],
            ["aggressive", "defensive", "balanced"],
            selection_strength=0.2,
            strategy_selection_strengths=(0.2, -0.1, 0.2),
        )


def test_deterministic_replicator_step_returns_mean_one_positive_finite():
    ws = deterministic_replicator_step(
        {"aggressive": 1.0, "defensive": 1.0, "balanced": 1.0},
        ["aggressive", "defensive", "balanced"],
        payoff_vector={"aggressive": 0.2, "defensive": -0.1, "balanced": 0.0},
        selection_strength=0.5,
    )
    assert set(ws.keys()) == {"aggressive", "defensive", "balanced"}
    vals = list(ws.values())
    assert all((v > 0 and math.isfinite(v)) for v in vals)
    assert sum(vals) / len(vals) == pytest.approx(1.0, abs=1e-12)


def test_tangential_drift_vector_returns_zero_at_simplex_center():
    drift = tangential_drift_vector([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], ["aggressive", "defensive", "balanced"], 0.01)
    assert drift == pytest.approx({"aggressive": 0.0, "defensive": 0.0, "balanced": 0.0}, abs=0.0)


def test_tangential_drift_vector_is_tangent_and_delta_normalized():
    strategy_space = ["aggressive", "defensive", "balanced"]
    simplex = [0.5, 0.3, 0.2]
    delta = 0.01
    drift = tangential_drift_vector(simplex, strategy_space, delta)
    drift_vec = [float(drift[s]) for s in strategy_space]
    radial = [float(value) - (1.0 / 3.0) for value in simplex]
    ones = [1.0, 1.0, 1.0]

    assert sum(drift_vec[i] * radial[i] for i in range(3)) == pytest.approx(0.0, abs=1e-12)
    assert sum(drift_vec[i] * ones[i] for i in range(3)) == pytest.approx(0.0, abs=1e-12)
    assert math.sqrt(sum(value * value for value in drift_vec)) == pytest.approx(delta, abs=1e-12)


def test_apply_tangential_drift_zero_delta_matches_original_growth():
    growth = {"aggressive": 0.2, "defensive": -0.1, "balanced": -0.1}
    drifted, diagnostics = apply_tangential_drift(
        growth,
        ["aggressive", "defensive", "balanced"],
        simplex=[0.5, 0.3, 0.2],
        delta=0.0,
    )
    assert drifted == pytest.approx(growth, abs=0.0)
    assert diagnostics["drift_norm"] == pytest.approx(0.0, abs=0.0)


def test_replicator_step_zero_tangential_drift_matches_baseline():
    players = [
        _P("aggressive", 1.0),
        _P("aggressive", 1.0),
        _P("defensive", 2.0),
        _P("balanced", 0.0),
    ]
    base = replicator_step(players, ["aggressive", "defensive", "balanced"], selection_strength=0.5)
    drifted = replicator_step(
        players,
        ["aggressive", "defensive", "balanced"],
        selection_strength=0.5,
        tangential_drift_simplex=[0.5, 0.25, 0.25],
        tangential_drift_delta=0.0,
    )
    assert drifted == pytest.approx(base, abs=0.0)


def test_deterministic_replicator_step_zero_tangential_drift_matches_baseline():
    current = {"aggressive": 1.0, "defensive": 1.0, "balanced": 1.0}
    payoff = {"aggressive": 0.2, "defensive": -0.1, "balanced": 0.0}
    base = deterministic_replicator_step(
        current,
        ["aggressive", "defensive", "balanced"],
        payoff_vector=payoff,
        selection_strength=0.5,
    )
    drifted = deterministic_replicator_step(
        current,
        ["aggressive", "defensive", "balanced"],
        payoff_vector=payoff,
        selection_strength=0.5,
        tangential_drift_simplex=[0.4, 0.35, 0.25],
        tangential_drift_delta=0.0,
    )
    assert drifted == pytest.approx(base, abs=0.0)


def test_inertial_deterministic_replicator_step_zero_inertia_matches_h4_update():
    current = {"aggressive": 1.0, "defensive": 1.0, "balanced": 1.0}
    payoff = {"aggressive": 0.2, "defensive": -0.1, "balanced": 0.0}
    base = deterministic_replicator_step(
        current,
        ["aggressive", "defensive", "balanced"],
        payoff_vector=payoff,
        selection_strength=0.5,
    )
    inertial, velocity = inertial_deterministic_replicator_step(
        current,
        ["aggressive", "defensive", "balanced"],
        payoff_vector=payoff,
        previous_velocity={"aggressive": 3.0, "defensive": -2.0, "balanced": 1.0},
        inertia=0.0,
        selection_strength=0.5,
    )
    assert inertial == pytest.approx(base, abs=1e-12)
    assert set(velocity.keys()) == {"aggressive", "defensive", "balanced"}
    assert all(math.isfinite(v) for v in velocity.values())


def test_inertial_deterministic_replicator_step_returns_mean_one_positive_finite():
    ws, velocity = inertial_deterministic_replicator_step(
        {"aggressive": 1.0, "defensive": 1.0, "balanced": 1.0},
        ["aggressive", "defensive", "balanced"],
        payoff_vector={"aggressive": 0.2, "defensive": -0.1, "balanced": 0.0},
        previous_velocity={"aggressive": 0.05, "defensive": -0.02, "balanced": -0.03},
        inertia=0.4,
        selection_strength=0.5,
    )
    vals = list(ws.values())
    assert all((v > 0 and math.isfinite(v)) for v in vals)
    assert sum(vals) / len(vals) == pytest.approx(1.0, abs=1e-12)
    assert all(math.isfinite(v) for v in velocity.values())


def test_inertial_sampled_replicator_step_zero_inertia_matches_sampled_update():
    players = [
        _P("aggressive", 1.0),
        _P("aggressive", 1.0),
        _P("defensive", 2.0),
        _P("balanced", 0.0),
    ]
    base = replicator_step(players, ["aggressive", "defensive", "balanced"], selection_strength=0.5)
    inertial, velocity = inertial_sampled_replicator_step(
        players,
        ["aggressive", "defensive", "balanced"],
        previous_velocity={"aggressive": 3.0, "defensive": -2.0, "balanced": 1.0},
        inertia=0.0,
        selection_strength=0.5,
    )
    assert inertial == pytest.approx(base, abs=1e-12)
    assert set(velocity.keys()) == {"aggressive", "defensive", "balanced"}
    assert all(math.isfinite(v) for v in velocity.values())


def test_inertial_sampled_replicator_step_returns_mean_one_positive_finite():
    players = [
        _P("aggressive", 1.0),
        _P("aggressive", 1.0),
        _P("defensive", 2.0),
        _P("balanced", 0.0),
    ]
    ws, velocity = inertial_sampled_replicator_step(
        players,
        ["aggressive", "defensive", "balanced"],
        previous_velocity={"aggressive": 0.05, "defensive": -0.02, "balanced": -0.03},
        inertia=0.4,
        selection_strength=0.5,
    )
    vals = list(ws.values())
    assert all((v > 0 and math.isfinite(v)) for v in vals)
    assert sum(vals) / len(vals) == pytest.approx(1.0, abs=1e-12)
    assert all(math.isfinite(v) for v in velocity.values())


def test_anchored_subgroup_payoff_shift_zero_when_anchor_matches_adaptive():
    shift = anchored_subgroup_payoff_shift(
        ["aggressive", "defensive", "balanced"],
        a=1.0,
        b=0.9,
        anchor_simplex={"aggressive": 0.2, "defensive": 0.3, "balanced": 0.5},
        adaptive_simplex={"aggressive": 0.2, "defensive": 0.3, "balanced": 0.5},
    )
    assert shift == pytest.approx({"aggressive": 0.0, "defensive": 0.0, "balanced": 0.0}, abs=0.0)


def test_anchored_subgroup_payoff_shift_requires_three_strategies():
    with pytest.raises(ValueError):
        anchored_subgroup_payoff_shift(
            ["a", "b"],
            a=1.0,
            b=0.9,
            anchor_simplex={"a": 0.5, "b": 0.5},
            adaptive_simplex={"a": 0.5, "b": 0.5},
        )


def test_anchored_subgroup_weight_pull_matches_anchor_and_adaptive_endpoints():
    strategy_space = ["aggressive", "defensive", "balanced"]
    adaptive = {"aggressive": 1.2, "defensive": 0.9, "balanced": 0.9}
    anchor = {"aggressive": 0.8, "defensive": 0.8, "balanced": 1.4}
    assert anchored_subgroup_weight_pull(
        strategy_space,
        adaptive_weights=adaptive,
        anchor_weights=anchor,
        anchor_pull_strength=0.0,
    ) == pytest.approx(adaptive, abs=1e-12)
    assert anchored_subgroup_weight_pull(
        strategy_space,
        adaptive_weights=adaptive,
        anchor_weights=anchor,
        anchor_pull_strength=1.0,
    ) == pytest.approx(anchor, abs=1e-12)


def test_anchored_subgroup_weight_pull_rejects_invalid_strength():
    with pytest.raises(ValueError):
        anchored_subgroup_weight_pull(
            ["aggressive", "defensive", "balanced"],
            adaptive_weights={"aggressive": 1.0, "defensive": 1.0, "balanced": 1.0},
            anchor_weights={"aggressive": 0.8, "defensive": 0.8, "balanced": 1.4},
            anchor_pull_strength=1.2,
        )


def test_state_dependent_anchored_subgroup_payoff_shift_scales_base_shift_by_sigmoid_gate():
    anchor = {"aggressive": 1.0, "defensive": 0.0, "balanced": 0.0}
    adaptive = {"aggressive": 0.0, "defensive": 1.0, "balanced": 0.0}
    base_shift = anchored_subgroup_payoff_shift(
        ["aggressive", "defensive", "balanced"],
        a=1.0,
        b=0.9,
        anchor_simplex=anchor,
        adaptive_simplex=adaptive,
    )
    shift, gap_norm, gate = state_dependent_anchored_subgroup_payoff_shift(
        ["aggressive", "defensive", "balanced"],
        a=1.0,
        b=0.9,
        anchor_simplex=anchor,
        adaptive_simplex=adaptive,
        base_coupling_strength=0.4,
        beta=6.0,
        theta=math.sqrt(2.0),
        signal="gap_norm",
    )
    assert gap_norm == pytest.approx(math.sqrt(2.0), abs=1e-12)
    assert gate == pytest.approx(0.5, abs=1e-12)
    assert shift == pytest.approx({k: 0.2 * float(v) for k, v in base_shift.items()}, abs=1e-12)


def test_state_dependent_anchored_subgroup_payoff_shift_rejects_unsupported_signal():
    with pytest.raises(ValueError):
        state_dependent_anchored_subgroup_payoff_shift(
            ["aggressive", "defensive", "balanced"],
            a=1.0,
            b=0.9,
            anchor_simplex={"aggressive": 0.2, "defensive": 0.3, "balanced": 0.5},
            adaptive_simplex={"aggressive": 0.1, "defensive": 0.4, "balanced": 0.5},
            base_coupling_strength=0.2,
            beta=6.0,
            theta=0.1,
            signal="corr",
        )


def test_bidirectional_subgroup_payoff_shifts_are_equal_and_opposite():
    fixed_shift, adaptive_shift = bidirectional_subgroup_payoff_shifts(
        ["aggressive", "defensive", "balanced"],
        a=1.0,
        b=0.9,
        fixed_simplex={"aggressive": 0.2, "defensive": 0.3, "balanced": 0.5},
        adaptive_simplex={"aggressive": 0.4, "defensive": 0.2, "balanced": 0.4},
        coupling_strength=0.3,
    )
    for key in ["aggressive", "defensive", "balanced"]:
        assert fixed_shift[key] == pytest.approx(-adaptive_shift[key], abs=1e-12)


def test_bidirectional_subgroup_payoff_shifts_reject_negative_strength():
    with pytest.raises(ValueError):
        bidirectional_subgroup_payoff_shifts(
            ["aggressive", "defensive", "balanced"],
            a=1.0,
            b=0.9,
            fixed_simplex={"aggressive": 0.2, "defensive": 0.3, "balanced": 0.5},
            adaptive_simplex={"aggressive": 0.4, "defensive": 0.2, "balanced": 0.4},
            coupling_strength=-0.1,
        )
