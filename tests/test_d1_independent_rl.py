"""Tests for D1 Independent Reinforcement Learning module + harness."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from evolution.independent_rl import (
    STRATEGY_SPACE,
    _NSTRATS,
    boltzmann_select,
    boltzmann_weights,
    cosine_similarity,
    one_hot_local_payoff,
    q_value_mean,
    rl_q_update,
    strategy_payoff_matrix,
    weight_entropy,
)
from evolution.local_graph import GraphSpec
from simulation.d1_independent_rl import (
    _cond_name,
    _ctrl_name,
    _make_graph_spec,
    _run_d1_simulation,
    _seed_metrics,
    _tail_d1_diagnostics,
    run_d1_scout,
)

# ======================================================================
# evolution/independent_rl.py — pure function tests
# ======================================================================

class TestStrategyPayoffMatrix:
    def test_diagonal_zero(self):
        mat = strategy_payoff_matrix(a=1.0, b=0.9, cross=0.2)
        for i in range(_NSTRATS):
            assert mat[i][i] == pytest.approx(0.0)

    def test_cross_coupling(self):
        mat = strategy_payoff_matrix(a=1.0, b=0.9, cross=0.2)
        # agg vs def  = a - cross = 0.8
        assert mat[0][1] == pytest.approx(0.8)
        # def vs agg  = -(b + cross) = -1.1
        assert mat[1][0] == pytest.approx(-1.1)
        # bal vs agg  = a + cross = 1.2
        assert mat[2][0] == pytest.approx(1.2)
        # bal vs def  = -(b - cross) = -0.7
        assert mat[2][1] == pytest.approx(-0.7)

    def test_no_cross(self):
        mat = strategy_payoff_matrix(a=1.0, b=0.9, cross=0.0)
        assert mat[0][1] == pytest.approx(1.0)
        assert mat[1][0] == pytest.approx(-0.9)
        assert mat[0][2] == pytest.approx(-0.9)
        assert mat[2][0] == pytest.approx(1.0)

    def test_shape(self):
        mat = strategy_payoff_matrix(a=1.0, b=0.9, cross=0.2)
        assert len(mat) == _NSTRATS
        assert all(len(row) == _NSTRATS for row in mat)


class TestOneHotLocalPayoff:
    def test_single_neighbor(self):
        mat = strategy_payoff_matrix(a=1.0, b=0.9, cross=0.2)
        # agg vs def = 0.8
        assert one_hot_local_payoff(0, [1], mat) == pytest.approx(0.8)

    def test_multiple_neighbors_mean(self):
        mat = strategy_payoff_matrix(a=1.0, b=0.9, cross=0.2)
        # agg vs [def, bal] = mean(0.8, -0.9) = -0.05
        assert one_hot_local_payoff(0, [1, 2], mat) == pytest.approx(-0.05)

    def test_same_strategy_zero(self):
        mat = strategy_payoff_matrix(a=1.0, b=0.9, cross=0.2)
        assert one_hot_local_payoff(0, [0, 0], mat) == pytest.approx(0.0)

    def test_empty_neighbors_zero(self):
        mat = strategy_payoff_matrix(a=1.0, b=0.9, cross=0.2)
        assert one_hot_local_payoff(0, [], mat) == pytest.approx(0.0)


class TestRLQUpdate:
    def test_chosen_reinforced(self):
        q = [0.0, 0.0, 0.0]
        q2 = rl_q_update(q, 1, 1.0, alpha=0.1)
        assert q2[1] == pytest.approx(0.1)
        assert q2[0] == pytest.approx(0.0)
        assert q2[2] == pytest.approx(0.0)

    def test_unchosen_decay(self):
        q = [0.5, 0.3, 0.2]
        q2 = rl_q_update(q, 0, 1.0, alpha=0.1)
        assert q2[0] == pytest.approx(0.5 * 0.9 + 0.1 * 1.0)
        assert q2[1] == pytest.approx(0.3 * 0.9)
        assert q2[2] == pytest.approx(0.2 * 0.9)

    def test_alpha_zero_no_change(self):
        q = [0.5, 0.3, 0.2]
        q2 = rl_q_update(q, 1, 10.0, alpha=0.0)
        for i in range(3):
            assert q2[i] == pytest.approx(q[i])

    def test_negative_reward(self):
        q = [0.0, 0.0, 0.0]
        q2 = rl_q_update(q, 0, -0.5, alpha=0.1)
        assert q2[0] == pytest.approx(-0.05)


class TestBoltzmannWeights:
    def test_uniform_q(self):
        w = boltzmann_weights([0.0, 0.0, 0.0], beta=5.0)
        for wi in w:
            assert wi == pytest.approx(1.0 / 3, abs=1e-6)

    def test_sum_to_one(self):
        w = boltzmann_weights([0.5, -0.2, 0.3], beta=10.0)
        assert sum(w) == pytest.approx(1.0, abs=1e-9)

    def test_higher_q_more_weight(self):
        w = boltzmann_weights([1.0, 0.0, 0.0], beta=5.0)
        assert w[0] > w[1]
        assert w[0] > w[2]

    def test_high_beta_concentrates(self):
        w_low = boltzmann_weights([1.0, 0.0, 0.0], beta=1.0)
        w_high = boltzmann_weights([1.0, 0.0, 0.0], beta=20.0)
        assert w_high[0] > w_low[0]

    def test_all_positive(self):
        w = boltzmann_weights([-5.0, -10.0, -1.0], beta=3.0)
        assert all(wi > 0 for wi in w)


class TestBoltzmannSelect:
    def test_deterministic_high_beta(self):
        rng = random.Random(42)
        q = [10.0, 0.0, 0.0]
        choices = [boltzmann_select(q, beta=100.0, rng=rng) for _ in range(50)]
        assert all(c == 0 for c in choices)

    def test_reproducible(self):
        q = [0.3, 0.2, 0.1]
        c1 = [boltzmann_select(q, beta=5.0, rng=random.Random(42)) for _ in range(20)]
        c2 = [boltzmann_select(q, beta=5.0, rng=random.Random(42)) for _ in range(20)]
        assert c1 == c2

    def test_valid_range(self):
        rng = random.Random(99)
        for _ in range(100):
            c = boltzmann_select([0.0, 0.0, 0.0], beta=5.0, rng=rng)
            assert 0 <= c < _NSTRATS


class TestWeightEntropy:
    def test_uniform(self):
        import math
        h = weight_entropy([1/3, 1/3, 1/3])
        assert h == pytest.approx(math.log(3), abs=1e-6)

    def test_degenerate(self):
        h = weight_entropy([1.0, 0.0, 0.0])
        assert h == pytest.approx(0.0)

    def test_positive(self):
        h = weight_entropy([0.5, 0.3, 0.2])
        assert h > 0.0


class TestQValueMean:
    def test_simple(self):
        assert q_value_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


class TestCosineSimilarity:
    def test_identical(self):
        assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite(self):
        assert cosine_similarity([1, 0, 0], [-1, 0, 0]) == pytest.approx(-1.0)


# ======================================================================
# simulation/d1_independent_rl.py — harness tests
# ======================================================================

class TestMakeGraphSpec:
    def test_lattice4(self):
        gs = _make_graph_spec("lattice4")
        assert gs is not None
        assert gs.topology == "lattice4"
        assert gs.degree == 4
        assert gs.lattice_rows == 15
        assert gs.lattice_cols == 20

    def test_well_mixed(self):
        assert _make_graph_spec("well_mixed") is None

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown topology"):
            _make_graph_spec("star")


class TestCondName:
    def test_active(self):
        assert _cond_name(topology="lattice4", alpha=0.05, beta=5.0) == "g2_d1_lattice4_a0p05_b5p0"

    def test_control(self):
        assert _ctrl_name("lattice4") == "g2_d1_frozen_lattice4"


class TestRunD1Simulation:
    """Smoke tests on tiny simulations."""

    TINY = dict(n_players=20, n_rounds=50, seed=42,
                a=1.0, b=0.9, cross=0.2, init_q=0.0)

    def test_returns_correct_lengths(self):
        rows, diag, fq, adj = _run_d1_simulation(
            graph_spec=None, alpha=0.05, softmax_beta=5.0, **self.TINY)
        assert len(rows) == 50
        assert len(diag) == 50
        assert len(fq) == 20
        assert adj is None  # well_mixed

    def test_lattice4_returns_adj(self):
        gs = GraphSpec(topology="lattice4", degree=4, lattice_rows=4, lattice_cols=5)
        rows, _, fq, adj = _run_d1_simulation(
            graph_spec=gs, alpha=0.05, softmax_beta=5.0, **self.TINY)
        assert adj is not None
        assert len(adj) == 20

    def test_deterministic(self):
        kw = dict(graph_spec=None, alpha=0.05, softmax_beta=5.0, **self.TINY)
        r1, _, q1, _ = _run_d1_simulation(**kw)
        r2, _, q2, _ = _run_d1_simulation(**kw)
        assert r1 == r2
        for i in range(20):
            for j in range(_NSTRATS):
                assert q1[i][j] == pytest.approx(q2[i][j])

    def test_frozen_control_q_unchanged(self):
        """alpha=0 ⇒ Q-values NEVER change from init_q."""
        rows, _, fq, _ = _run_d1_simulation(
            graph_spec=None, alpha=0.0, softmax_beta=5.0, **self.TINY)
        for player_q in fq:
            for q in player_q:
                assert q == pytest.approx(0.0)

    def test_active_q_diverge(self):
        """alpha>0 ⇒ Q not all zero after many rounds."""
        rows, _, fq, _ = _run_d1_simulation(
            graph_spec=None, alpha=0.10, softmax_beta=5.0,
            n_players=20, n_rounds=200, seed=42,
            a=1.0, b=0.9, cross=0.2, init_q=0.0,
        )
        all_qs = [q for pq in fq for q in pq]
        assert any(abs(q) > 0.001 for q in all_qs)

    def test_strategy_proportions_valid(self):
        rows, _, _, _ = _run_d1_simulation(
            graph_spec=None, alpha=0.05, softmax_beta=5.0, **self.TINY)
        for r in rows:
            s = r["p_aggressive"] + r["p_defensive"] + r["p_balanced"]
            assert s == pytest.approx(1.0, abs=1e-9)

    def test_diagnostics_keys(self):
        _, diag, _, _ = _run_d1_simulation(
            graph_spec=None, alpha=0.05, softmax_beta=5.0, **self.TINY)
        expected_keys = {
            "mean_player_weight_entropy", "weight_heterogeneity_std",
            "mean_q_value_std", "mean_neighbor_weight_cosine",
        }
        for d in diag:
            assert expected_keys.issubset(d.keys())

    def test_well_mixed_no_cosine(self):
        """well_mixed → no static adj → neighbor_weight_cosine = 0."""
        _, diag, _, _ = _run_d1_simulation(
            graph_spec=None, alpha=0.05, softmax_beta=5.0, **self.TINY)
        for d in diag:
            assert d["mean_neighbor_weight_cosine"] == pytest.approx(0.0)

    def test_lattice4_has_cosine(self):
        gs = GraphSpec(topology="lattice4", degree=4, lattice_rows=4, lattice_cols=5)
        _, diag, _, _ = _run_d1_simulation(
            graph_spec=gs, alpha=0.05, softmax_beta=5.0,
            n_players=20, n_rounds=100, seed=42,
            a=1.0, b=0.9, cross=0.2, init_q=0.0)
        # After some learning, cosine should be computable (may be ~1.0 at start)
        assert any(d["mean_neighbor_weight_cosine"] > 0 for d in diag)


class TestSeedMetrics:
    def test_returns_expected_keys(self):
        rows, _, _, _ = _run_d1_simulation(
            graph_spec=None, alpha=0.05, softmax_beta=5.0,
            n_players=20, n_rounds=100, seed=42,
            a=1.0, b=0.9, cross=0.2, init_q=0.0,
        )
        sm = _seed_metrics(rows, burn_in=30, tail=50, eta=0.55, corr_threshold=0.09)
        assert "cycle_level" in sm
        assert "env_gamma" in sm
        assert 0 <= sm["cycle_level"] <= 3


class TestTailDiagnostics:
    def test_returns_expected_keys(self):
        _, diag, fq, adj = _run_d1_simulation(
            graph_spec=None, alpha=0.05, softmax_beta=5.0,
            n_players=20, n_rounds=100, seed=42,
            a=1.0, b=0.9, cross=0.2, init_q=0.0,
        )
        td = _tail_d1_diagnostics(diag, fq, adj,
                                   n_rows=100, burn_in=30, tail=50, softmax_beta=5.0)
        assert "mean_player_weight_entropy" in td
        assert "mean_q_value_std" in td
        assert "mean_neighbor_weight_cosine" in td


class TestD1ScoutSmoke:
    """End-to-end tiny scout run."""

    def test_tiny_scout(self, tmp_path: Path):
        result = run_d1_scout(
            seeds=[42],
            topologies=["well_mixed"],
            alphas=[0.05],
            softmax_betas=[5.0],
            out_root=tmp_path / "d1",
            summary_tsv=tmp_path / "d1" / "summary.tsv",
            combined_tsv=tmp_path / "d1" / "combined.tsv",
            decision_md=tmp_path / "d1" / "decision.md",
            players=20, rounds=100, burn_in=30, tail=50,
            a=1.0, b=0.9, cross=0.2, init_q=0.0,
        )
        assert (tmp_path / "d1" / "summary.tsv").exists()
        assert (tmp_path / "d1" / "combined.tsv").exists()
        assert (tmp_path / "d1" / "decision.md").exists()
        assert "close_d1" in result

    def test_with_lattice4(self, tmp_path: Path):
        # lattice4 hardcodes 15×20=300 in _make_graph_spec
        result = run_d1_scout(
            seeds=[42],
            topologies=["lattice4"],
            alphas=[0.05],
            softmax_betas=[5.0],
            out_root=tmp_path / "d1",
            summary_tsv=tmp_path / "d1" / "summary.tsv",
            combined_tsv=tmp_path / "d1" / "combined.tsv",
            decision_md=tmp_path / "d1" / "decision.md",
            players=300, rounds=50, burn_in=15, tail=25,
            a=1.0, b=0.9, cross=0.2, init_q=0.0,
        )
        assert (tmp_path / "d1" / "decision.md").exists()

    def test_control_has_control_verdict(self, tmp_path: Path):
        result = run_d1_scout(
            seeds=[42],
            topologies=["well_mixed"],
            alphas=[0.05],
            softmax_betas=[5.0],
            out_root=tmp_path / "d1",
            summary_tsv=tmp_path / "d1" / "summary.tsv",
            combined_tsv=tmp_path / "d1" / "combined.tsv",
            decision_md=tmp_path / "d1" / "decision.md",
            players=20, rounds=50, burn_in=15, tail=25,
        )
        # Decision MD should exist and mention verdict
        txt = (tmp_path / "d1" / "decision.md").read_text()
        assert "verdict=" in txt
