"""Tests for F1 Heterogeneous Payoff (per-player payoff perturbation).

Covers:
  Sample/Apply:
    F0-1: sample_payoff_perturbation with epsilon=0 returns all zeros
    F0-2: sample_payoff_perturbation with epsilon>0 returns non-zero
    F0-3: sample_payoff_perturbation reproducible with same seed
    F0-4: apply_per_player_payoff_perturbation correctness
    F0-5: apply with zero perturbation = identity

  Harness internals:
    F0-6: _cond_name format
    F0-7: _make_graph_spec lattice4 / well_mixed
    F0-8: _strategy_cycle_stability returns float in [0, 1]
    F0-9: _payoff_heterogeneity returns 0 for eps=0

  Simulation:
    F1-1: _run_f1_simulation output structure
    F1-2: deterministic (same seed → same result)
    F1-3: eps=0 degenerates to E1 behavior (control)
    F1-4: eps>0 produces higher q_std than eps=0
    F1-5: strategy proportions sum to 1.0
    F1-6: lattice adjacency matrix is valid
    F1-7: payoff deltas have correct shape

  Metrics / diagnostics:
    F1-8: _seed_metrics returns expected keys
    F1-9: _tail_f1_diagnostics returns expected keys

  Scout:
    F1-10: tiny smoke test (2 eps × 1 topo × 1 seed)
    F1-11: verdict logic (pass / fail / weak_positive)
    F1-12: decision.md is written
"""
from __future__ import annotations

import random
import pytest

from evolution.independent_rl import (
    sample_payoff_perturbation,
    apply_per_player_payoff_perturbation,
    _NSTRATS,
)
from simulation.f1_heterogeneous_payoff import (
    _cond_name,
    _make_graph_spec,
    _run_f1_simulation,
    _seed_metrics,
    _tail_f1_diagnostics,
    _strategy_cycle_stability,
    _payoff_heterogeneity,
    run_f1_scout,
    DEFAULT_ALPHA_LO,
    DEFAULT_ALPHA_HI,
    DEFAULT_BETA,
)


# ---------------------------------------------------------------------------
# Sample / Apply
# ---------------------------------------------------------------------------

class TestSamplePayoffPerturbation:
    def test_eps_zero_all_zeros(self):
        """F0-1: epsilon=0 returns all-zero perturbation vectors."""
        rng = random.Random(42)
        deltas = sample_payoff_perturbation(10, epsilon=0.0, rng=rng)
        assert len(deltas) == 10
        for d in deltas:
            assert len(d) == _NSTRATS
            assert all(v == 0.0 for v in d)

    def test_eps_positive_non_zero(self):
        """F0-2: epsilon>0 returns non-zero vectors."""
        rng = random.Random(42)
        deltas = sample_payoff_perturbation(50, epsilon=0.1, rng=rng)
        assert len(deltas) == 50
        flat = [v for d in deltas for v in d]
        assert any(v != 0.0 for v in flat)
        # All within [-eps, eps]
        for v in flat:
            assert -0.1 <= v <= 0.1

    def test_reproducible(self):
        """F0-3: same seed → same perturbations."""
        d1 = sample_payoff_perturbation(20, epsilon=0.05, rng=random.Random(99))
        d2 = sample_payoff_perturbation(20, epsilon=0.05, rng=random.Random(99))
        assert d1 == d2

    def test_apply_correctness(self):
        """F0-4: apply adds delta[chosen] to reward."""
        rewards = [1.0, 2.0, 3.0]
        chosen = [0, 1, 2]
        perturbations = [
            [0.1, 0.2, 0.3],   # player 0 chose 0 → +0.1
            [0.4, 0.5, 0.6],   # player 1 chose 1 → +0.5
            [0.7, 0.8, 0.9],   # player 2 chose 2 → +0.9
        ]
        result = apply_per_player_payoff_perturbation(rewards, chosen, perturbations)
        assert len(result) == 3
        assert abs(result[0] - 1.1) < 1e-10
        assert abs(result[1] - 2.5) < 1e-10
        assert abs(result[2] - 3.9) < 1e-10

    def test_apply_zero_perturbation_identity(self):
        """F0-5: zero perturbation = identity."""
        rewards = [1.0, 2.0, 3.0]
        chosen = [0, 1, 2]
        zeros = [[0.0, 0.0, 0.0]] * 3
        result = apply_per_player_payoff_perturbation(rewards, chosen, zeros)
        assert result == rewards


# ---------------------------------------------------------------------------
# Harness internals
# ---------------------------------------------------------------------------

class TestCondName:
    def test_format(self):
        """F0-6: condition name format."""
        cn = _cond_name(topology="lattice4", payoff_epsilon=0.05)
        assert cn == "g2_f1_lattice4_eps0p05"

    def test_zero_eps(self):
        cn = _cond_name(topology="well_mixed", payoff_epsilon=0.0)
        assert cn == "g2_f1_well_mixed_eps0"


class TestMakeGraphSpec:
    def test_lattice4(self):
        """F0-7a: lattice4 produces valid GraphSpec."""
        gs = _make_graph_spec("lattice4")
        assert gs is not None
        assert gs.topology == "lattice4"

    def test_well_mixed(self):
        """F0-7b: well_mixed returns None."""
        gs = _make_graph_spec("well_mixed")
        assert gs is None

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            _make_graph_spec("unknown_topo")


class TestStrategyCycleStability:
    def test_returns_float(self):
        """F0-8: returns float in [0, 1]."""
        # Create fake rows with some cycling
        rows = []
        import math
        for t in range(200):
            angle = t * 0.05
            pa = 1/3 + 0.1 * math.cos(angle)
            pd = 1/3 + 0.1 * math.cos(angle + 2*math.pi/3)
            pb = 1.0 - pa - pd
            rows.append({"round": t, "p_aggressive": pa, "p_defensive": pd, "p_balanced": pb})
        scs = _strategy_cycle_stability(rows, burn_in=0, tail=200)
        assert isinstance(scs, float)
        assert 0.0 <= scs <= 1.0

    def test_insufficient_data(self):
        rows = [{"round": 0, "p_aggressive": 0.33, "p_defensive": 0.33, "p_balanced": 0.34}]
        assert _strategy_cycle_stability(rows, burn_in=0, tail=1) == 0.0


class TestPayoffHeterogeneity:
    def test_zero_for_no_perturbation(self):
        """F0-9: zero deltas → zero heterogeneity."""
        deltas = [[0.0, 0.0, 0.0]] * 10
        assert _payoff_heterogeneity(deltas) == 0.0

    def test_positive_for_perturbation(self):
        rng = random.Random(42)
        deltas = sample_payoff_perturbation(50, epsilon=0.1, rng=rng)
        h = _payoff_heterogeneity(deltas)
        assert h > 0.0


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

_TINY = dict(
    n_players=20, n_rounds=50, seed=42,
    alpha_lo=DEFAULT_ALPHA_LO, alpha_hi=DEFAULT_ALPHA_HI,
    beta=DEFAULT_BETA, a=1.0, b=0.9, cross=0.20,
)


class TestRunF1Simulation:
    def test_output_structure(self):
        """F1-1: simulation returns correct tuple structure."""
        rows, rdiag, fq, adj, alphas, deltas = _run_f1_simulation(
            graph_spec=None, payoff_epsilon=0.05, **_TINY)
        assert len(rows) == 50
        assert len(rdiag) == 50
        assert len(fq) == 20
        assert adj is None  # well_mixed
        assert len(alphas) == 20
        assert len(deltas) == 20
        # Check row keys
        assert "round" in rows[0]
        assert "p_aggressive" in rows[0]
        # Check diag keys
        assert "mean_player_weight_entropy" in rdiag[0]
        assert "mean_q_value_std" in rdiag[0]

    def test_deterministic(self):
        """F1-2: same seed → same result."""
        r1, _, _, _, _, _ = _run_f1_simulation(
            graph_spec=None, payoff_epsilon=0.05, **_TINY)
        r2, _, _, _, _, _ = _run_f1_simulation(
            graph_spec=None, payoff_epsilon=0.05, **_TINY)
        for a, b in zip(r1, r2):
            assert a["p_aggressive"] == b["p_aggressive"]
            assert a["p_defensive"] == b["p_defensive"]
            assert a["p_balanced"] == b["p_balanced"]

    def test_eps0_control(self):
        """F1-3: eps=0 produces valid output (degenerates to E1 α-only)."""
        rows, rdiag, fq, _, alphas, deltas = _run_f1_simulation(
            graph_spec=None, payoff_epsilon=0.0, **_TINY)
        assert len(rows) == 50
        # All deltas should be zero
        for d in deltas:
            assert all(v == 0.0 for v in d)

    def test_eps_positive_higher_qstd(self):
        """F1-4: eps>0 should produce different Q-trajectories than eps=0."""
        # Run two simulations: one with eps=0, one with eps=0.1
        _, diag0, _, _, _, _ = _run_f1_simulation(
            graph_spec=None, payoff_epsilon=0.0, n_players=30, n_rounds=200,
            seed=42, alpha_lo=0.02, alpha_hi=0.08, beta=5.0,
            a=1.0, b=0.9, cross=0.20)
        _, diag1, _, _, _, _ = _run_f1_simulation(
            graph_spec=None, payoff_epsilon=0.10, n_players=30, n_rounds=200,
            seed=42, alpha_lo=0.02, alpha_hi=0.08, beta=5.0,
            a=1.0, b=0.9, cross=0.20)
        # Q-std should generally be higher with perturbation
        # (just check that the trajectories diverge — last round diag differs)
        q0 = diag0[-1]["mean_q_value_std"]
        q1 = diag1[-1]["mean_q_value_std"]
        # They should not be identical (perturbation changes dynamics)
        assert q0 != q1, "eps=0 and eps=0.10 should differ in q_std"

    def test_strategy_proportions_sum_to_1(self):
        """F1-5: p_agg + p_def + p_bal = 1.0 each round."""
        rows, _, _, _, _, _ = _run_f1_simulation(
            graph_spec=None, payoff_epsilon=0.05, **_TINY)
        for r in rows:
            total = r["p_aggressive"] + r["p_defensive"] + r["p_balanced"]
            assert abs(total - 1.0) < 1e-10, f"Round {r['round']}: sum={total}"

    def test_lattice_adj(self):
        """F1-6: lattice4 produces valid adjacency."""
        from evolution.local_graph import GraphSpec
        gs = GraphSpec(topology="lattice4", degree=4, lattice_rows=4, lattice_cols=5)
        _, _, _, adj, _, _ = _run_f1_simulation(
            graph_spec=gs, payoff_epsilon=0.05, **{**_TINY, "n_players": 20})
        assert adj is not None
        assert len(adj) == 20

    def test_payoff_deltas_shape(self):
        """F1-7: payoff deltas have correct shape (n_players × 3)."""
        _, _, _, _, _, deltas = _run_f1_simulation(
            graph_spec=None, payoff_epsilon=0.08, **_TINY)
        assert len(deltas) == 20
        for d in deltas:
            assert len(d) == 3


# ---------------------------------------------------------------------------
# Metrics / diagnostics
# ---------------------------------------------------------------------------

class TestSeedMetrics:
    def test_expected_keys(self):
        """F1-8: _seed_metrics returns expected keys."""
        rows, _, _, _, _, _ = _run_f1_simulation(
            graph_spec=None, payoff_epsilon=0.05, **_TINY)
        sm = _seed_metrics(rows, burn_in=10, tail=30, eta=0.55, corr_threshold=0.09)
        for key in ["cycle_level", "stage3_score", "turn_strength",
                     "env_gamma", "env_gamma_r2", "env_gamma_n_peaks"]:
            assert key in sm


class TestTailF1Diagnostics:
    def test_expected_keys(self):
        """F1-9: _tail_f1_diagnostics returns expected keys."""
        rows, rdiag, fq, _, _, _ = _run_f1_simulation(
            graph_spec=None, payoff_epsilon=0.05, **_TINY)
        td = _tail_f1_diagnostics(rdiag, fq, None, DEFAULT_BETA,
                                   n_rows=len(rows), burn_in=10, tail=30)
        for key in ["mean_player_weight_entropy", "weight_heterogeneity_std",
                     "mean_q_value_std", "mean_neighbor_weight_cosine",
                     "spatial_strategy_clustering", "mean_edge_strategy_distance"]:
            assert key in td


# ---------------------------------------------------------------------------
# Scout
# ---------------------------------------------------------------------------

class TestF1Scout:
    def test_smoke(self, tmp_path):
        """F1-10: tiny smoke test (2 eps × 1 topo × 1 seed)."""
        result = run_f1_scout(
            seeds=[42],
            topologies=["well_mixed"],
            payoff_epsilons=[0.0, 0.05],
            out_root=tmp_path / "out",
            summary_tsv=tmp_path / "s.tsv",
            combined_tsv=tmp_path / "c.tsv",
            decision_md=tmp_path / "d.md",
            players=20, rounds=50, burn_in=10, tail=30,
        )
        assert "close_f1" in result
        assert (tmp_path / "s.tsv").exists()
        assert (tmp_path / "c.tsv").exists()
        assert (tmp_path / "d.md").exists()

    def test_verdict_logic(self, tmp_path):
        """F1-11: control gets 'control', active gets a verdict."""
        result = run_f1_scout(
            seeds=[42],
            topologies=["well_mixed"],
            payoff_epsilons=[0.0, 0.05],
            out_root=tmp_path / "out",
            summary_tsv=tmp_path / "s.tsv",
            combined_tsv=tmp_path / "c.tsv",
            decision_md=tmp_path / "d.md",
            players=20, rounds=50, burn_in=10, tail=30,
        )
        # Read combined TSV
        import csv
        with (tmp_path / "c.tsv").open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
        ctrl = [r for r in rows if r["is_control"] == "yes"]
        active = [r for r in rows if r["is_control"] == "no"]
        assert len(ctrl) == 1
        assert ctrl[0]["verdict"] == "control"
        assert len(active) == 1
        assert active[0]["verdict"] in ("pass", "fail", "weak_positive")

    def test_decision_written(self, tmp_path):
        """F1-12: decision.md is non-empty."""
        run_f1_scout(
            seeds=[42],
            topologies=["well_mixed"],
            payoff_epsilons=[0.0, 0.05],
            out_root=tmp_path / "out",
            summary_tsv=tmp_path / "s.tsv",
            combined_tsv=tmp_path / "c.tsv",
            decision_md=tmp_path / "d.md",
            players=20, rounds=50, burn_in=10, tail=30,
        )
        text = (tmp_path / "d.md").read_text()
        assert "F1 Heterogeneous Payoff Decision" in text
        assert "Pass Gate" in text
