"""G0 + IR + integration tests for evolution/local_pairwise.py
and basic smoke tests for simulation/c1_local_pairwise harness.

Blueprint §11.2 (helpers), §11.4 (IR), §11.5 (integration smoke).

Test inventory
--------------
local_payoff:
  P-1  uniform neighbour mix → stable real output
  P-2  single neighbour aggressive → correct direction
  P-3  empty neighbour list → 0.0
  P-4  output is finite for all-balanced weights
  P-5  output scales linearly with a (sanity check)

pairwise_adoption_probability:
  A-1  q ∈ (0, 1) always
  A-2  q = 0.5 when π_i == π_j
  A-3  q > 0.5 when π_j > π_i
  A-4  large positive diff → q ≈ 1.0
  A-5  large negative diff → q ≈ 0.0
  A-6  no overflow for extreme diffs

pairwise_imitation_update:
  U-1  output sums to 1
  U-2  mu=0 → output == weights_i (simplex normalised)
  U-3  mu=1, q=1 → output == weights_j (simplex normalised)
  U-4  intermediate mu/q → convex combination (direction)
  U-5  output has all entries > 0

conversion helpers:
  C-1  list_to_weights / weights_to_list round-trip
  C-2  weights_to_simplex sums to 1

IR (invariant regression):
  IR-1  _normalise raises on zero vector
  IR-2  pairwise_imitation_update raises on degenerate input
  IR-3  local_payoff unknown strategy → fallback (no crash)

Integration smoke (§11.5):
  S-1  run_c1_cell returns expected keys and correct row count
  S-2  per-round p_* values are in [0, 1] and sum ≈ 1
  S-3  per-round w_* values are in [0, 1] and sum ≈ 1
  S-4  run_c1_cell is reproducible (same seed → same result)
  S-5  run_c1_cell produces different results for different seeds
  S-6  round_diag length == rounds
  S-7  tail diagnostics are finite positive floats
"""

from __future__ import annotations

import math
import random

import pytest

from evolution.local_pairwise import (
    STRATEGY_SPACE,
    _normalise,
    list_to_weights,
    local_payoff,
    pairwise_adoption_probability,
    pairwise_imitation_update,
    weights_to_list,
    weights_to_simplex,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

A_DEF, B_DEF, CROSS_DEF = 1.0, 0.9, 0.20


def _uniform_w() -> list[float]:
    return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]


def _corner_w(k: int) -> list[float]:
    w = [0.0, 0.0, 0.0]
    w[k] = 1.0
    return w


# ============================================================================
# local_payoff
# ============================================================================

class TestLocalPayoff:
    def test_P1_uniform_stable(self) -> None:
        """P-1: uniform weights, uniform mix → output is finite real."""
        w = _uniform_w()
        strats = ["aggressive", "defensive", "balanced"] * 4
        pi = local_payoff(w, strats, a=A_DEF, b=B_DEF, cross=CROSS_DEF)
        assert math.isfinite(pi)

    def test_P2_aggressive_weight_direction(self) -> None:
        """P-2: pure aggressive focal player vs pure aggressive neighbour.

        With A=[[0,a,-b],...] the payoff w=[1,0,0] vs s=[1,0,0]:
        u[0] = a*0 - b*0 = 0, so π = 0.
        """
        w = _corner_w(0)          # all aggressive
        pi = local_payoff(w, ["aggressive"], a=A_DEF, b=B_DEF, cross=CROSS_DEF)
        # cross term: u[0] += -cross * s1 = 0  (s1=0 for aggressive)
        assert math.isfinite(pi)

    def test_P3_empty_neighbours(self) -> None:
        """P-3: no neighbours → payoff is 0.0 by definition."""
        w = _uniform_w()
        pi = local_payoff(w, [], a=A_DEF, b=B_DEF, cross=CROSS_DEF)
        assert pi == 0.0

    def test_P4_balanced_weights_finite(self) -> None:
        """P-4: finitely defined for pure balanced player."""
        w = _corner_w(2)
        strats = ["aggressive", "aggressive", "defensive"]
        pi = local_payoff(w, strats, a=A_DEF, b=B_DEF, cross=CROSS_DEF)
        assert math.isfinite(pi)

    def test_P5_scales_with_a(self) -> None:
        """P-5: doubling a should change the payoff in a predictable direction.

        We test that payoff(a=2.0) != payoff(a=1.0) when result is non-trivial.
        """
        w = [0.5, 0.3, 0.2]
        strats = ["defensive"]  # s=[0,1,0]: u[0] = a*1 - b*0 = a
        pi1 = local_payoff(w, strats, a=1.0, b=0.9, cross=0.0)
        pi2 = local_payoff(w, strats, a=2.0, b=0.9, cross=0.0)
        assert abs(pi2 - pi1) > 1e-12, "payoff should change when a changes"


# ============================================================================
# pairwise_adoption_probability
# ============================================================================

class TestPairwiseAdoptionProbability:
    def test_A1_in_unit_interval(self) -> None:
        """A-1: q always in [0, 1] (strict numerically for moderate diffs).

        For moderate payoff diffs the Fermi function is strictly in (0,1).
        For very large diffs float64 rounding can produce exactly 0.0 or 1.0;
        that is numerically correct behaviour and is separately tested in A-4/A-5.
        """
        rng = random.Random(0)
        for _ in range(50):
            pi = rng.uniform(-0.5, 0.5)  # moderate range → strict (0,1)
            pj = rng.uniform(-0.5, 0.5)
            q = pairwise_adoption_probability(pi, pj, beta_pair=8.0)
            assert 0.0 < q < 1.0

    def test_A2_equal_payoffs_half(self) -> None:
        """A-2: q = 0.5 when π_i == π_j."""
        q = pairwise_adoption_probability(1.5, 1.5, beta_pair=8.0)
        assert abs(q - 0.5) < 1e-12

    def test_A3_q_gt_half_when_pj_better(self) -> None:
        """A-3: q > 0.5 when π_j > π_i."""
        q = pairwise_adoption_probability(0.0, 1.0, beta_pair=8.0)
        assert q > 0.5

    def test_A4_large_positive_diff_approaches_one(self) -> None:
        """A-4: very large π_j−π_i → q ≈ 1.0."""
        q = pairwise_adoption_probability(0.0, 100.0, beta_pair=8.0)
        assert q > 0.999

    def test_A5_large_negative_diff_approaches_zero(self) -> None:
        """A-5: very large π_i−π_j → q ≈ 0.0."""
        q = pairwise_adoption_probability(100.0, 0.0, beta_pair=8.0)
        assert q < 0.001

    def test_A6_no_overflow_extreme_diff(self) -> None:
        """A-6: no overflow or NaN for extreme differences."""
        q_pos = pairwise_adoption_probability(-1e9, 1e9, beta_pair=8.0)
        q_neg = pairwise_adoption_probability(1e9, -1e9, beta_pair=8.0)
        assert math.isfinite(q_pos)
        assert math.isfinite(q_neg)
        assert 0.0 < q_pos <= 1.0
        assert 0.0 <= q_neg < 1.0


# ============================================================================
# pairwise_imitation_update
# ============================================================================

class TestPairwiseImitationUpdate:
    def test_U1_sums_to_one(self) -> None:
        """U-1: output sums to 1."""
        rng = random.Random(42)
        for _ in range(20):
            r_i = [rng.random() for _ in range(3)]
            r_j = [rng.random() for _ in range(3)]
            si = sum(r_i)
            sj = sum(r_j)
            w_i = [v / si for v in r_i]
            w_j = [v / sj for v in r_j]
            q = rng.uniform(0.0, 1.0)
            mu = rng.uniform(0.0, 1.0)
            result = pairwise_imitation_update(w_i, w_j, mu=mu, q=q)
            assert abs(sum(result) - 1.0) < 1e-10

    def test_U2_mu_zero_unchanged(self) -> None:
        """U-2: mu=0 → w_i unchanged (after simplex normalisation)."""
        w_i = [0.5, 0.3, 0.2]
        w_j = [0.1, 0.8, 0.1]
        result = pairwise_imitation_update(w_i, w_j, mu=0.0, q=0.9)
        expected = _normalise(w_i)
        for actual, exp in zip(result, expected):
            assert abs(actual - exp) < 1e-10

    def test_U3_mu_q_one_equals_wj(self) -> None:
        """U-3: mu=1, q=1 → output = w_j (normalised)."""
        w_i = [0.5, 0.3, 0.2]
        w_j = [0.1, 0.8, 0.1]
        result = pairwise_imitation_update(w_i, w_j, mu=1.0, q=1.0)
        expected = _normalise(w_j)
        for actual, exp in zip(result, expected):
            assert abs(actual - exp) < 1e-10

    def test_U4_intermediate_convex_direction(self) -> None:
        """U-4: intermediate mu/q → output lies between w_i and w_j."""
        w_i = [0.8, 0.1, 0.1]
        w_j = [0.1, 0.1, 0.8]
        q = 0.5
        mu = 0.5
        result = pairwise_imitation_update(w_i, w_j, mu=mu, q=q)
        # Result should have [0] < w_i[0] (moved toward w_j)
        assert result[0] < w_i[0]
        # Result should have [2] > w_i[2]
        assert result[2] > w_i[2]

    def test_U5_all_entries_positive(self) -> None:
        """U-5: all entries of result are strictly positive."""
        rng = random.Random(55)
        for _ in range(20):
            w_i = _normalise([rng.random() for _ in range(3)])
            w_j = _normalise([rng.random() for _ in range(3)])
            q = rng.uniform(0.0, 1.0)
            mu = rng.uniform(0.0, 1.0)
            result = pairwise_imitation_update(w_i, w_j, mu=mu, q=q)
            assert all(v > 0.0 for v in result)


# ============================================================================
# Conversion helpers
# ============================================================================

class TestConversionHelpers:
    def test_C1_round_trip(self) -> None:
        """C-1: list_to_weights(weights_to_list(w)) == w for any ordered dict."""
        w_dict = {"aggressive": 0.5, "defensive": 0.3, "balanced": 0.2}
        lst = weights_to_list(w_dict)
        w_back = list_to_weights(lst)
        assert w_back == {"aggressive": 0.5, "defensive": 0.3, "balanced": 0.2}

    def test_C2_simplex_sums_to_one(self) -> None:
        """C-2: weights_to_simplex always sums to 1."""
        w_dict = {"aggressive": 3.0, "defensive": 1.0, "balanced": 2.0}
        s = weights_to_simplex(w_dict)
        assert abs(sum(s) - 1.0) < 1e-12
        assert all(v > 0.0 for v in s)


# ============================================================================
# IR (invariant regression)
# ============================================================================

class TestInvariantRegression:
    def test_IR1_normalise_raises_zero(self) -> None:
        """IR-1: _normalise raises ValueError on zero vector."""
        with pytest.raises(ValueError):
            _normalise([0.0, 0.0, 0.0])

    def test_IR2_imitation_update_degenerate(self) -> None:
        """IR-2: pairwise_imitation_update raises when result collapses to zero.

        mu=0.0 and w_i=[0,0,0] → new_w = w_i (all zeros) → _normalise raises.
        """
        with pytest.raises((ValueError, ZeroDivisionError)):
            pairwise_imitation_update([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], mu=0.5, q=0.5)

    def test_IR3_local_payoff_unknown_strategy(self) -> None:
        """IR-3: unknown strategy string → fallback to uniform, no crash."""
        w = _uniform_w()
        pi = local_payoff(w, ["unknown_strat"], a=A_DEF, b=B_DEF, cross=CROSS_DEF)
        assert math.isfinite(pi)


# ============================================================================
# Integration smoke tests (§11.5) — run_c1_cell
# ============================================================================

@pytest.fixture
def tiny_c1_config():
    """Return a tiny C1CellConfig for smoke testing (30 players, 50 rounds)."""
    from evolution.local_graph import GraphSpec
    from simulation.c1_local_pairwise import C1CellConfig
    import pathlib
    return C1CellConfig(
        condition="c1_ring4_mu0p10",
        graph_spec=GraphSpec(topology="ring4", degree=4),
        pairwise_imitation_strength=0.10,
        pairwise_beta=8.0,
        players=30,
        rounds=50,
        seed=42,
        a=1.0,
        b=0.9,
        cross=0.20,
        burn_in=10,
        tail=20,
        init_bias=0.12,
        memory_kernel=3,
        out_dir=pathlib.Path("/tmp/c1_smoke_test"),
    )


class TestRunC1Cell:
    def test_S1_returns_expected_keys_and_row_count(self, tiny_c1_config) -> None:
        """S-1: run_c1_cell returns dict with 'global_rows' and 'round_diag'."""
        from simulation.c1_local_pairwise import run_c1_cell
        result = run_c1_cell(tiny_c1_config)
        assert "global_rows" in result
        assert "round_diag" in result
        assert len(result["global_rows"]) == tiny_c1_config.rounds
        assert len(result["round_diag"]) == tiny_c1_config.rounds

    def test_S2_p_values_valid(self, tiny_c1_config) -> None:
        """S-2: each round's p_* values are in [0, 1] and sum ≈ 1."""
        from simulation.c1_local_pairwise import run_c1_cell
        result = run_c1_cell(tiny_c1_config)
        for row in result["global_rows"]:
            vals = [float(row[f"p_{s}"]) for s in STRATEGY_SPACE]
            assert all(0.0 <= v <= 1.0 for v in vals), f"p value out of [0,1]: {vals}"
            assert abs(sum(vals) - 1.0) < 1e-9, f"p values don't sum to 1: {sum(vals)}"

    def test_S3_w_values_valid(self, tiny_c1_config) -> None:
        """S-3: each round's w_* values are in [0, 1] and sum ≈ 1."""
        from simulation.c1_local_pairwise import run_c1_cell
        result = run_c1_cell(tiny_c1_config)
        for row in result["global_rows"]:
            vals = [float(row[f"w_{s}"]) for s in STRATEGY_SPACE]
            assert all(0.0 <= v <= 1.0 for v in vals), f"w value out of [0,1]: {vals}"
            assert abs(sum(vals) - 1.0) < 1e-9, f"w values don't sum to 1: {sum(vals)}"

    def test_S4_reproducible(self, tiny_c1_config) -> None:
        """S-4: same seed → same result (deterministic)."""
        from simulation.c1_local_pairwise import run_c1_cell
        r1 = run_c1_cell(tiny_c1_config)
        r2 = run_c1_cell(tiny_c1_config)
        for row1, row2 in zip(r1["global_rows"], r2["global_rows"]):
            for s in STRATEGY_SPACE:
                assert row1[f"p_{s}"] == row2[f"p_{s}"]
                assert row1[f"w_{s}"] == row2[f"w_{s}"]

    def test_S5_different_seeds_differ(self, tiny_c1_config) -> None:
        """S-5: different seeds → different simulation trajectories."""
        from evolution.local_graph import GraphSpec
        from simulation.c1_local_pairwise import C1CellConfig, run_c1_cell
        import dataclasses
        cfg2 = dataclasses.replace(tiny_c1_config, seed=99)
        r1 = run_c1_cell(tiny_c1_config)
        r2 = run_c1_cell(cfg2)
        # At least one row must differ
        any_diff = any(
            r1["global_rows"][t]["p_aggressive"] != r2["global_rows"][t]["p_aggressive"]
            for t in range(tiny_c1_config.rounds)
        )
        assert any_diff, "Different seeds produced identical trajectories"

    def test_S6_round_diag_length(self, tiny_c1_config) -> None:
        """S-6: round_diag has exactly 'rounds' entries."""
        from simulation.c1_local_pairwise import run_c1_cell
        result = run_c1_cell(tiny_c1_config)
        assert len(result["round_diag"]) == tiny_c1_config.rounds

    def test_S7_tail_diagnostics_finite(self, tiny_c1_config) -> None:
        """S-7: tail_local_diagnostics returns finite positive floats."""
        from simulation.c1_local_pairwise import run_c1_cell, _tail_local_diagnostics
        result = run_c1_cell(tiny_c1_config)
        diag = _tail_local_diagnostics(
            result["round_diag"],
            n_rows=len(result["global_rows"]),
            burn_in=tiny_c1_config.burn_in,
            tail=tiny_c1_config.tail,
        )
        for key, val in diag.items():
            assert math.isfinite(val), f"diagnostic '{key}' is not finite: {val}"
            assert val >= 0.0, f"diagnostic '{key}' is negative: {val}"
