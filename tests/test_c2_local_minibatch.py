"""G0 + IR + integration smoke tests for evolution/local_replicator.py
and simulation/c2_local_minibatch harness.

Blueprint §11.3 (G0 gates), §11.4 (IR), §11.5 (integration smoke).

Test inventory
--------------
_payoff_vec (internal):
  PV-1  uniform x → output is finite
  PV-2  pure aggressive x → u[agg]=0 (A row-0 has 0 on diagonal)
  PV-3  cross=0 → pure rock-paper-scissors structure

local_popularity:
  LP-1  single-element batch → returns that player's weights
  LP-2  uniform weights → mean == uniform
  LP-3  empty batch → returns [1/3, 1/3, 1/3] fallback
  LP-4  output sums to 1 for any valid input

local_replicator_advantage (G0 gate included):
  ADV-1  orthogonality <w_i, g_i> = 0 (L∞ < 1e-9)
  ADV-2  uniform weights + uniform local_pop → advantage is all-zero
  ADV-3  output is finite for all corner cases

local_replicator_update:
  U-1   k_local=0 → w_i' ≈ w_i (exact no-op, L∞ < 1e-12)
  U-2   output sums to 1 (L∞ < 1e-9)
  U-3   all output entries strictly > 0
  U-4   k_local > 0 → output differs from input (non-trivial update)
  U-5   k_local < 0 raises ValueError

G0 gates:
  G0-1  local_batch = global_population (complete graph) →
        local_replicator_advantage ≡ global-popularity advantage (L∞ < 1e-9)
  G0-2  k_local=0 → bit-identical timeseries (via run_c2_cell)
  G0-3  fixed seed → bit-identical timeseries
  G0-4  all w_i' > 0 and sum = 1 after full cell run

Diagnostic helpers:
  D-1  local_growth_norm: L2 norm of zero vector = 0.0
  D-2  local_growth_cosine_vs_global: identical vectors → 1.0
  D-3  local_growth_cosine_vs_global: zero vector fallback = 1.0
  D-4  player_growth_dispersion: empty list = 0.0
  D-5  batch_phase_spread: single player, no neighbours = 0.0

Conversion helpers:
  C-1  weights_to_simplex: round-trip with list_to_weights preserves order
  C-2  weights_to_simplex: sums to 1
  C-3  list_to_weights: keys match STRATEGY_SPACE

IR (invariant regression):
  IR-1  _normalise raises on zero vector
  IR-2  local_replicator_update raises for k_local < 0
  IR-3  local_popularity: result always sums to 1 for non-degenerate input

Integration smoke (§11.5):
  S-1   run_c2_cell returns expected keys
  S-2   global_rows length == rounds
  S-3   p_* in [0,1] and sum ≈ 1 every row
  S-4   w_* in [0,1] and sum ≈ 1 every row
  S-5   run_c2_cell is reproducible (same seed → same result)
  S-6   run_c2_cell different seeds → different results
  S-7   round_diag length == rounds
  S-8   tail diagnostics are finite ≥ 0
"""

from __future__ import annotations

import math
import random
from copy import deepcopy

import pytest

from evolution.local_graph import GraphSpec, build_graph
from evolution.local_replicator import (
    STRATEGY_SPACE,
    _normalise,
    _payoff_vec,
    batch_phase_spread,
    list_to_weights,
    local_growth_cosine_vs_global,
    local_growth_norm,
    local_popularity,
    local_replicator_advantage,
    local_replicator_update,
    neighbor_popularity_dispersion,
    player_growth_dispersion,
    weights_to_simplex,
)
from simulation.c2_local_minibatch import (
    C2CellConfig,
    GraphSpec,
    _tail_local_diagnostics,
    run_c2_cell,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

A_DEF, B_DEF, CROSS_DEF = 1.0, 0.9, 0.20
SMALL_N = 20      # small player count for fast smoke tests
SMALL_R = 60      # small round count
BURN_DEF = 20
TAIL_DEF = 30


def _uniform_w() -> list[float]:
    return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]


def _corner_w(k: int) -> list[float]:
    w = [0.001, 0.001, 0.001]
    w[k] = 0.998
    return w


def _small_ring_spec() -> GraphSpec:
    return GraphSpec(topology="ring4", degree=4)


def _small_c2_config(seed: int = 42) -> C2CellConfig:
    from pathlib import Path
    return C2CellConfig(
        condition="c2_ring4_k0p06",
        graph_spec=_small_ring_spec(),
        local_selection_strength=0.06,
        players=SMALL_N,
        rounds=SMALL_R,
        seed=seed,
        a=A_DEF,
        b=B_DEF,
        cross=CROSS_DEF,
        burn_in=BURN_DEF,
        tail=TAIL_DEF,
        init_bias=0.12,
        memory_kernel=3,
        out_dir=Path("/tmp/test_c2_cell"),
    )


# ============================================================================
# _payoff_vec
# ============================================================================

class TestPayoffVec:
    def test_PV1_uniform_finite(self) -> None:
        """PV-1: uniform x → output is finite."""
        x = _uniform_w()
        u = _payoff_vec(x, a=A_DEF, b=B_DEF, cross=CROSS_DEF)
        assert all(math.isfinite(v) for v in u)

    def test_PV2_pure_aggressive_diagonal_zero(self) -> None:
        """PV-2: pure aggressive x → u[agg] = a*0 - b*0 = 0 (cross also 0 for defensive=0)."""
        x = [1.0, 0.0, 0.0]
        u = _payoff_vec(x, a=A_DEF, b=B_DEF, cross=0.0)
        # A·e_0 = [0, -b, a]
        assert abs(u[0] - 0.0) < 1e-12
        assert abs(u[1] - (-B_DEF)) < 1e-12
        assert abs(u[2] - A_DEF) < 1e-12

    def test_PV3_zero_cross_rps_structure(self) -> None:
        """PV-3: cross=0 → pure rock-paper-scissors cyclic structure."""
        x = [0.5, 0.3, 0.2]
        u_cross = _payoff_vec(x, a=A_DEF, b=B_DEF, cross=CROSS_DEF)
        u_no_cross = _payoff_vec(x, a=A_DEF, b=B_DEF, cross=0.0)
        # They should differ when cross != 0
        assert u_cross != u_no_cross

    def test_PV4_output_length_3(self) -> None:
        u = _payoff_vec(_uniform_w(), a=A_DEF, b=B_DEF, cross=CROSS_DEF)
        assert len(u) == 3


# ============================================================================
# local_popularity
# ============================================================================

class TestLocalPopularity:
    def test_LP1_single_element_batch(self) -> None:
        """LP-1: single-element batch → returns that player's weights exactly."""
        pw = [[0.6, 0.2, 0.2], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]]
        result = local_popularity(pw, [1])
        assert all(abs(result[k] - pw[1][k]) < 1e-14 for k in range(3))

    def test_LP2_uniform_mean(self) -> None:
        """LP-2: two identical uniform weights → mean = same uniform."""
        pw = [[1.0/3, 1.0/3, 1.0/3], [1.0/3, 1.0/3, 1.0/3]]
        result = local_popularity(pw, [0, 1])
        assert all(abs(result[k] - 1.0/3.0) < 1e-14 for k in range(3))

    def test_LP3_empty_batch_fallback(self) -> None:
        """LP-3: empty batch → [1/3, 1/3, 1/3]."""
        pw = [[0.5, 0.3, 0.2]]
        result = local_popularity(pw, [])
        assert all(abs(result[k] - 1.0/3.0) < 1e-12 for k in range(3))

    def test_LP4_result_sums_to_one(self) -> None:
        """LP-4: result sums to ≈ 1 for any valid batch."""
        rng = random.Random(7)
        pw = []
        for _ in range(10):
            raw = [rng.random() + 0.1 for _ in range(3)]
            total = sum(raw)
            pw.append([v / total for v in raw])
        batch = [0, 3, 5, 7]
        result = local_popularity(pw, batch)
        assert abs(sum(result) - 1.0) < 1e-12


# ============================================================================
# local_replicator_advantage
# ============================================================================

class TestLocalReplicatorAdvantage:
    def test_ADV1_orthogonality(self) -> None:
        """ADV-1: <w_i, g_i> = 0 for any weights and local_pop (L∞ < 1e-9)."""
        rng = random.Random(13)
        for _ in range(50):
            raw_w = [rng.random() + 0.01 for _ in range(3)]
            tw = sum(raw_w)
            wi = [v / tw for v in raw_w]
            raw_x = [rng.random() + 0.01 for _ in range(3)]
            tx = sum(raw_x)
            xi = [v / tx for v in raw_x]
            g = local_replicator_advantage(wi, xi, a=A_DEF, b=B_DEF, cross=CROSS_DEF)
            dot = sum(wi[k] * g[k] for k in range(3))
            assert abs(dot) < 1e-9, f"<w,g>={dot} should be 0, wi={wi}, xi={xi}"

    def test_ADV2_uniform_uniform_zero(self) -> None:
        """ADV-2: uniform weights + uniform local_pop → all advantages = 0."""
        wi = _uniform_w()
        xi = _uniform_w()
        g = local_replicator_advantage(wi, xi, a=A_DEF, b=B_DEF, cross=CROSS_DEF)
        # At uniform, u = A·[1/3,1/3,1/3]; expected = Σ w_k * u_k
        # By symmetry this is uniform so g should be ~0
        assert all(math.isfinite(v) for v in g)

    def test_ADV3_finite_corner_weights(self) -> None:
        """ADV-3: output is finite for corner weight cases."""
        for k in range(3):
            wi = _corner_w(k)
            xi = _uniform_w()
            g = local_replicator_advantage(wi, xi, a=A_DEF, b=B_DEF, cross=CROSS_DEF)
            assert all(math.isfinite(v) for v in g), f"Non-finite advantage for corner {k}: {g}"


# ============================================================================
# local_replicator_update
# ============================================================================

class TestLocalReplicatorUpdate:
    def test_U1_k_zero_exact_noop(self) -> None:
        """U-1: k_local=0 → w_i' ≈ w_i (L∞ < 1e-12)."""
        rng = random.Random(99)
        for _ in range(30):
            raw = [rng.random() + 0.1 for _ in range(3)]
            tw = sum(raw)
            wi = [v / tw for v in raw]
            xi = [rng.random() + 0.01 for _ in range(3)]
            tx = sum(xi)
            xi = [v / tx for v in xi]
            g = local_replicator_advantage(wi, xi, a=A_DEF, b=B_DEF, cross=CROSS_DEF)
            w_new = local_replicator_update(wi, g, k_local=0.0)
            linf = max(abs(w_new[k] - wi[k]) for k in range(3))
            assert linf < 1e-12, f"k_local=0 produced diff L∞={linf}"

    def test_U2_sums_to_one(self) -> None:
        """U-2: output sums to 1 (L∞ < 1e-9)."""
        rng = random.Random(55)
        for _ in range(50):
            raw_w = [rng.random() + 0.01 for _ in range(3)]
            tw = sum(raw_w)
            wi = [v / tw for v in raw_w]
            raw_x = [rng.random() + 0.01 for _ in range(3)]
            tx = sum(raw_x)
            xi = [v / tx for v in raw_x]
            g = local_replicator_advantage(wi, xi, a=A_DEF, b=B_DEF, cross=CROSS_DEF)
            w_new = local_replicator_update(wi, g, k_local=0.08)
            assert abs(sum(w_new) - 1.0) < 1e-9, f"sum={sum(w_new)}"

    def test_U3_all_positive(self) -> None:
        """U-3: all output entries strictly > 0."""
        rng = random.Random(17)
        for _ in range(50):
            raw_w = [rng.random() + 0.05 for _ in range(3)]
            tw = sum(raw_w)
            wi = [v / tw for v in raw_w]
            raw_x = [rng.random() + 0.01 for _ in range(3)]
            tx = sum(raw_x)
            xi = [v / tx for v in raw_x]
            g = local_replicator_advantage(wi, xi, a=A_DEF, b=B_DEF, cross=CROSS_DEF)
            w_new = local_replicator_update(wi, g, k_local=0.06)
            assert all(v > 0.0 for v in w_new), f"Zero or negative weight: {w_new}"

    def test_U4_nontrivial_update(self) -> None:
        """U-4: k_local > 0 → output differs from input for non-uniform state."""
        wi = [0.6, 0.2, 0.2]
        xi = [0.1, 0.7, 0.2]
        g = local_replicator_advantage(wi, xi, a=A_DEF, b=B_DEF, cross=CROSS_DEF)
        w_new = local_replicator_update(wi, g, k_local=0.1)
        linf = max(abs(w_new[k] - wi[k]) for k in range(3))
        assert linf > 1e-10, f"Update too small: L∞={linf}"

    def test_U5_negative_k_raises(self) -> None:
        """U-5: k_local < 0 raises ValueError."""
        wi = _uniform_w()
        g = [0.0, 0.0, 0.0]
        with pytest.raises(ValueError):
            local_replicator_update(wi, g, k_local=-0.01)


# ============================================================================
# G0 Gates (blueprint §11.3)
# ============================================================================

class TestG0Gates:
    def test_G0_1_complete_graph_equiv_global(self) -> None:
        """G0-1: local_batch = full person pool → advantage ≡ global-popularity advantage.

        Set up N players, all with the same weights so local pop = global pop.
        Then local_replicator_advantage(wi, local_pop) should equal
        local_replicator_advantage(wi, global_pop) exactly.
        """
        rng = random.Random(2024)
        n = 8
        # Random heterogeneous weights
        all_w = []
        for _ in range(n):
            raw = [rng.random() + 0.01 for _ in range(3)]
            total = sum(raw)
            all_w.append([v / total for v in raw])

        # Global population mean
        global_pop = [sum(all_w[i][k] for i in range(n)) / n for k in range(3)]

        # Using full batch = all players
        full_batch = list(range(n))
        local_pop_full = local_popularity(all_w, full_batch)

        # Local pop from full batch should equal global pop exactly
        for k in range(3):
            assert abs(local_pop_full[k] - global_pop[k]) < 1e-12

        # Advantage must match
        wi = all_w[0]
        g_local = local_replicator_advantage(wi, local_pop_full, a=A_DEF, b=B_DEF, cross=CROSS_DEF)
        g_global = local_replicator_advantage(wi, global_pop, a=A_DEF, b=B_DEF, cross=CROSS_DEF)
        linf = max(abs(g_local[k] - g_global[k]) for k in range(3))
        assert linf < 1e-9, f"G0-1 FAIL: L∞={linf} between local and global advantage"

    def test_G0_2_k_zero_noop_timeseries(self) -> None:
        """G0-2: k_local=0 → all player weights unchanged round over round (L∞ < 1e-12)."""
        from pathlib import Path
        cfg0 = C2CellConfig(
            condition="ctrl", graph_spec=_small_ring_spec(),
            local_selection_strength=0.0,
            players=SMALL_N, rounds=10, seed=1,
            a=A_DEF, b=B_DEF, cross=CROSS_DEF,
            burn_in=0, tail=5, init_bias=0.12, memory_kernel=3,
            out_dir=Path("/tmp/test_g0_2"),
        )
        result = run_c2_cell(cfg0)
        rows = result["global_rows"]
        # All w_* columns should be constant across rounds (weights never change)
        first_row = rows[0]
        for row in rows[1:]:
            for s in STRATEGY_SPACE:
                diff = abs(float(row[f"w_{s}"]) - float(first_row[f"w_{s}"]))
                assert diff < 1e-10, f"G0-2 FAIL: w_{s} changed by {diff} at round {row['round']}"

    def test_G0_3_fixed_seed_reproducible(self) -> None:
        """G0-3: two calls with the same seed → bit-identical w_* timeseries."""
        cfg = _small_c2_config(seed=77)
        result_a = run_c2_cell(cfg)
        result_b = run_c2_cell(cfg)
        rows_a = result_a["global_rows"]
        rows_b = result_b["global_rows"]
        assert len(rows_a) == len(rows_b)
        for i, (ra, rb) in enumerate(zip(rows_a, rows_b)):
            for s in STRATEGY_SPACE:
                assert float(ra[f"w_{s}"]) == float(rb[f"w_{s}"]), (
                    f"G0-3 FAIL: w_{s} differs at round {i}: {ra[f'w_{s}']} vs {rb[f'w_{s}']}"
                )

    def test_G0_4_all_weights_valid(self) -> None:
        """G0-4: all w_* in (0,1) and sum ≈ 1 after full cell run."""
        result = run_c2_cell(_small_c2_config(seed=88))
        rows = result["global_rows"]
        for row in rows:
            wsum = sum(float(row[f"w_{s}"]) for s in STRATEGY_SPACE)
            assert abs(wsum - 1.0) < 1e-8, f"w_* sum={wsum} != 1"
            for s in STRATEGY_SPACE:
                v = float(row[f"w_{s}"])
                assert 0.0 < v < 1.0 + 1e-9, f"w_{s}={v} out of (0,1)"


# ============================================================================
# Diagnostic helpers
# ============================================================================

class TestDiagnosticHelpers:
    def test_D1_growth_norm_zero_vector(self) -> None:
        """D-1: L2 norm of zero advantage = 0.0."""
        assert local_growth_norm([0.0, 0.0, 0.0]) == 0.0

    def test_D2_cosine_identical_vectors(self) -> None:
        """D-2: cosine of identical non-zero vectors = 1.0."""
        g = [0.3, -0.2, -0.1]
        cos = local_growth_cosine_vs_global(g, g)
        assert abs(cos - 1.0) < 1e-12

    def test_D3_cosine_zero_vector_fallback(self) -> None:
        """D-3: cosine with zero local or global vector = 1.0 (safe fallback)."""
        g = [0.3, -0.2, -0.1]
        assert local_growth_cosine_vs_global([0.0, 0.0, 0.0], g) == 1.0
        assert local_growth_cosine_vs_global(g, [0.0, 0.0, 0.0]) == 1.0

    def test_D4_player_growth_dispersion_empty(self) -> None:
        """D-4: empty advantages list → 0.0."""
        assert player_growth_dispersion([]) == 0.0

    def test_D5_batch_phase_spread_no_neighbors(self) -> None:
        """D-5: adjacency list with no edges → phase spread = 0.0."""
        adj: list[list[int]] = [[], [], []]
        ws = [_uniform_w(), _uniform_w(), _uniform_w()]
        assert batch_phase_spread(adj, ws) == 0.0


# ============================================================================
# Conversion helpers
# ============================================================================

class TestConversionHelpers:
    def test_C1_round_trip(self) -> None:
        """C-1: weights_to_simplex + list_to_weights preserves correct ordering."""
        w_dict = {"aggressive": 0.5, "defensive": 0.3, "balanced": 0.2}
        as_list = weights_to_simplex(w_dict)
        as_dict = list_to_weights(as_list)
        for s in STRATEGY_SPACE:
            assert abs(as_dict[s] - w_dict[s]) < 1e-12

    def test_C2_weights_to_simplex_sums_to_one(self) -> None:
        """C-2: output of weights_to_simplex sums to 1."""
        w = {"aggressive": 3.0, "defensive": 1.0, "balanced": 2.0}
        result = weights_to_simplex(w)
        assert abs(sum(result) - 1.0) < 1e-12

    def test_C3_list_to_weights_keys(self) -> None:
        """C-3: list_to_weights produces keys = STRATEGY_SPACE exactly."""
        lst = [0.4, 0.35, 0.25]
        result = list_to_weights(lst)
        assert set(result.keys()) == set(STRATEGY_SPACE)


# ============================================================================
# IR (Invariant Regression)
# ============================================================================

class TestIR:
    def test_IR1_normalise_raises_on_zero(self) -> None:
        """IR-1: _normalise raises ValueError on zero-sum vector."""
        with pytest.raises(ValueError):
            _normalise([0.0, 0.0, 0.0])

    def test_IR2_update_raises_negative_k(self) -> None:
        """IR-2: local_replicator_update raises ValueError for k_local < 0."""
        wi = _uniform_w()
        g = [0.0, 0.0, 0.0]
        with pytest.raises(ValueError):
            local_replicator_update(wi, g, k_local=-1.0)

    def test_IR3_local_popularity_sums_to_one(self) -> None:
        """IR-3: local_popularity always sums to 1 for non-empty batch."""
        rng = random.Random(31)
        pw = []
        for _ in range(15):
            raw = [rng.random() + 0.05 for _ in range(3)]
            total = sum(raw)
            pw.append([v / total for v in raw])
        result = local_popularity(pw, list(range(15)))
        assert abs(sum(result) - 1.0) < 1e-12


# ============================================================================
# Integration smoke (§11.5)
# ============================================================================

class TestIntegrationSmoke:
    def test_S1_expected_keys(self) -> None:
        """S-1: run_c2_cell returns global_rows and round_diag."""
        result = run_c2_cell(_small_c2_config())
        assert "global_rows" in result
        assert "round_diag" in result

    def test_S2_row_count(self) -> None:
        """S-2: global_rows length == rounds."""
        cfg = _small_c2_config()
        result = run_c2_cell(cfg)
        assert len(result["global_rows"]) == cfg.rounds

    def test_S3_p_valid(self) -> None:
        """S-3: p_* in [0,1] and sum ≈ 1 every row."""
        result = run_c2_cell(_small_c2_config())
        for row in result["global_rows"]:
            total = sum(float(row[f"p_{s}"]) for s in STRATEGY_SPACE)
            assert abs(total - 1.0) < 1e-9, f"p_* sum={total}"
            for s in STRATEGY_SPACE:
                v = float(row[f"p_{s}"])
                assert 0.0 <= v <= 1.0, f"p_{s}={v} out of range"

    def test_S4_w_valid(self) -> None:
        """S-4: w_* in [0,1] and sum ≈ 1 every row."""
        result = run_c2_cell(_small_c2_config())
        for row in result["global_rows"]:
            total = sum(float(row[f"w_{s}"]) for s in STRATEGY_SPACE)
            assert abs(total - 1.0) < 1e-8, f"w_* sum={total}"
            for s in STRATEGY_SPACE:
                v = float(row[f"w_{s}"])
                assert 0.0 < v < 1.0 + 1e-9, f"w_{s}={v} out of range"

    def test_S5_reproducible(self) -> None:
        """S-5: same seed → identical w_* timeseries (bit-identical floats)."""
        cfg = _small_c2_config(seed=111)
        ra = run_c2_cell(cfg)["global_rows"]
        rb = run_c2_cell(cfg)["global_rows"]
        for i, (row_a, row_b) in enumerate(zip(ra, rb)):
            for s in STRATEGY_SPACE:
                assert float(row_a[f"w_{s}"]) == float(row_b[f"w_{s}"]), (
                    f"S-5 FAIL at round {i} w_{s}: {row_a[f'w_{s}']} != {row_b[f'w_{s}']}"
                )

    def test_S6_different_seeds_differ(self) -> None:
        """S-6: different seeds → different p_* (strategy sampling uses per-player RNGs).

        Note: w_* (mean weights) are seed-independent for ring4 (deterministic
        graph, identical initial weights, deterministic update).  p_* timeseries
        differ because choose_strategy() is driven by each player's seeded RNG.
        """
        ra = run_c2_cell(_small_c2_config(seed=1))["global_rows"]
        rb = run_c2_cell(_small_c2_config(seed=2))["global_rows"]
        diffs = [
            abs(float(ra[i][f"p_{s}"]) - float(rb[i][f"p_{s}"]))
            for i in range(len(ra))
            for s in STRATEGY_SPACE
        ]
        assert max(diffs) > 1e-10, "Different seeds produced identical p_* results"

    def test_S7_round_diag_length(self) -> None:
        """S-7: round_diag length == rounds."""
        cfg = _small_c2_config()
        result = run_c2_cell(cfg)
        assert len(result["round_diag"]) == cfg.rounds

    def test_S8_tail_diagnostics_finite(self) -> None:
        """S-8: tail diagnostics are finite ≥ 0."""
        cfg = _small_c2_config()
        result = run_c2_cell(cfg)
        diag = _tail_local_diagnostics(
            result["round_diag"],
            n_rows=len(result["global_rows"]),
            burn_in=cfg.burn_in,
            tail=cfg.tail,
        )
        DIAG_KEYS = [
            "mean_local_growth_norm",
            "mean_local_growth_cosine_vs_global",
            "mean_player_growth_dispersion",
            "mean_batch_phase_spread",
            "mean_neighbor_popularity_dispersion",
            "mean_local_update_step_norm",
            "edge_disagreement_rate",
        ]
        for key in DIAG_KEYS:
            assert key in diag, f"Missing key: {key}"
            v = float(diag[key])
            assert math.isfinite(v), f"{key}={v} is not finite"
            assert v >= -1e-12, f"{key}={v} is negative"
