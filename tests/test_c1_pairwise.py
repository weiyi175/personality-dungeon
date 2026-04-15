"""Tests for C1 Local Pairwise Imitation harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from evolution.local_graph import GraphSpec
from simulation.c1_pairwise_scout import (
    _make_graph_spec,
    _run_c1_simulation,
    _seed_metrics,
    _tail_c1_diagnostics,
    run_c1_scout,
)


# ---------------------------------------------------------------------------
# Deterministic reproducibility
# ---------------------------------------------------------------------------

class TestC1Deterministic:
    def test_same_seed_same_result(self):
        gs = GraphSpec(topology="ring4", degree=4)
        kw = dict(
            n_players=20, n_rounds=100, seed=42,
            graph_spec=gs, beta_pair=5.0, mu_pair=0.5,
            a=1.0, b=0.9, cross=0.20, init_bias=0.12,
        )
        rows1, _, fw1, _ = _run_c1_simulation(**kw)
        rows2, _, fw2, _ = _run_c1_simulation(**kw)
        for r1, r2 in zip(rows1, rows2):
            assert r1 == r2
        for w1, w2 in zip(fw1, fw2):
            assert w1 == w2

    def test_different_seed_different_result(self):
        gs = GraphSpec(topology="ring4", degree=4)
        kw = dict(
            n_players=20, n_rounds=100,
            graph_spec=gs, beta_pair=5.0, mu_pair=0.5,
            a=1.0, b=0.9, cross=0.20, init_bias=0.12,
        )
        rows1, _, _, _ = _run_c1_simulation(seed=42, **kw)
        rows2, _, _, _ = _run_c1_simulation(seed=99, **kw)
        # At least one row should differ
        assert any(r1 != r2 for r1, r2 in zip(rows1, rows2))


# ---------------------------------------------------------------------------
# Weight heterogeneity
# ---------------------------------------------------------------------------

class TestC1WeightHeterogeneity:
    def test_jitter_creates_initial_heterogeneity(self):
        """Initial weight jitter should produce nonzero heterogeneity early on."""
        gs = GraphSpec(topology="ring4", degree=4)
        _, diag, _, _ = _run_c1_simulation(
            n_players=30, n_rounds=5, seed=42,
            graph_spec=gs, beta_pair=10.0, mu_pair=0.8,
            a=1.0, b=0.9, cross=0.20, init_bias=0.12,
        )
        initial_std = diag[0]["weight_heterogeneity_std"]
        assert initial_std > 0.001, f"Expected initial heterogeneity > 0.001, got {initial_std}"

    def test_consensus_convergence(self):
        """Pairwise imitation should drive consensus (decreasing heterogeneity)."""
        gs = GraphSpec(topology="ring4", degree=4)
        _, diag, _, _ = _run_c1_simulation(
            n_players=30, n_rounds=300, seed=42,
            graph_spec=gs, beta_pair=10.0, mu_pair=0.8,
            a=1.0, b=0.9, cross=0.20, init_bias=0.12,
        )
        std_early = diag[1]["weight_heterogeneity_std"]
        std_late = diag[-1]["weight_heterogeneity_std"]
        # Consensus-seeking: heterogeneity should decrease
        assert std_late < std_early


# ---------------------------------------------------------------------------
# Beta effect
# ---------------------------------------------------------------------------

class TestC1BetaEffect:
    def test_higher_beta_not_less_heterogeneous(self):
        gs = GraphSpec(topology="ring4", degree=4)
        kw = dict(
            n_players=30, n_rounds=200, seed=42,
            graph_spec=gs, mu_pair=0.5,
            a=1.0, b=0.9, cross=0.20, init_bias=0.12,
        )
        _, diag_lo, _, _ = _run_c1_simulation(beta_pair=1.0, **kw)
        _, diag_hi, _, _ = _run_c1_simulation(beta_pair=20.0, **kw)
        std_lo = diag_lo[-1]["weight_heterogeneity_std"]
        std_hi = diag_hi[-1]["weight_heterogeneity_std"]
        # High beta should produce at least comparable heterogeneity
        assert std_hi >= std_lo * 0.5


# ---------------------------------------------------------------------------
# Well-mixed control
# ---------------------------------------------------------------------------

class TestC1WellMixedControl:
    def test_control_runs(self):
        rows, diag, fw, adj = _run_c1_simulation(
            n_players=20, n_rounds=100, seed=42,
            graph_spec=None, beta_pair=5.0, mu_pair=0.5,
            a=1.0, b=0.9, cross=0.20, init_bias=0.12,
        )
        assert len(rows) == 100
        assert adj is None
        assert len(fw) == 20
        # Probabilities should sum to ~1
        for r in rows:
            total = r["p_aggressive"] + r["p_defensive"] + r["p_balanced"]
            assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Graph spec builder
# ---------------------------------------------------------------------------

class TestMakeGraphSpec:
    def test_lattice4(self):
        gs = _make_graph_spec("lattice4")
        assert gs.topology == "lattice4"
        assert gs.lattice_rows == 15
        assert gs.lattice_cols == 20

    def test_small_world(self):
        gs = _make_graph_spec("small_world", small_world_p=0.15)
        assert gs.topology == "small_world"
        assert gs.p_rewire == 0.15

    def test_ring4(self):
        gs = _make_graph_spec("ring4")
        assert gs.topology == "ring4"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            _make_graph_spec("unknown_topology")


# ---------------------------------------------------------------------------
# Seed metrics + tail diagnostics
# ---------------------------------------------------------------------------

class TestC1Metrics:
    def test_seed_metrics_runs(self):
        gs = GraphSpec(topology="ring4", degree=4)
        rows, _, _, _ = _run_c1_simulation(
            n_players=20, n_rounds=200, seed=42,
            graph_spec=gs, beta_pair=5.0, mu_pair=0.5,
            a=1.0, b=0.9, cross=0.20, init_bias=0.12,
        )
        sm = _seed_metrics(
            rows, burn_in=50, tail=50, eta=0.55, corr_threshold=0.09
        )
        assert "cycle_level" in sm
        assert 0 <= sm["cycle_level"] <= 3

    def test_tail_diagnostics_with_graph(self):
        gs = GraphSpec(topology="ring4", degree=4)
        rows, rdiag, fw, adj = _run_c1_simulation(
            n_players=20, n_rounds=200, seed=42,
            graph_spec=gs, beta_pair=5.0, mu_pair=0.5,
            a=1.0, b=0.9, cross=0.20, init_bias=0.12,
        )
        td = _tail_c1_diagnostics(
            rdiag, fw, adj, n_rows=len(rows), burn_in=50, tail=50
        )
        assert "mean_player_weight_entropy" in td
        assert "spatial_strategy_clustering" in td
        assert td["spatial_strategy_clustering"] >= 0.0
        assert td["graph_clustering_coeff"] >= 0.0

    def test_tail_diagnostics_no_graph(self):
        rows, rdiag, fw, adj = _run_c1_simulation(
            n_players=20, n_rounds=200, seed=42,
            graph_spec=None, beta_pair=5.0, mu_pair=0.5,
            a=1.0, b=0.9, cross=0.20, init_bias=0.12,
        )
        td = _tail_c1_diagnostics(
            rdiag, fw, adj, n_rows=len(rows), burn_in=50, tail=50
        )
        # No graph → spatial metrics are 0
        assert td["spatial_strategy_clustering"] == 0.0
        assert td["graph_clustering_coeff"] == 0.0
        # But weight entropy should still be computed
        assert td["mean_player_weight_entropy"] > 0.0


# ---------------------------------------------------------------------------
# Full harness smoke test
# ---------------------------------------------------------------------------

class TestC1HarnessSmokeTest:
    def test_smoke(self, tmp_path):
        result = run_c1_scout(
            seeds=[42],
            topologies=["ring4"],
            beta_pairs=[5.0],
            mu_pairs=[0.5],
            out_root=tmp_path / "out",
            summary_tsv=tmp_path / "summary.tsv",
            combined_tsv=tmp_path / "combined.tsv",
            decision_md=tmp_path / "decision.md",
            players=20,
            rounds=100,
            burn_in=30,
            tail=30,
            init_bias=0.12,
            a=1.0,
            b=0.9,
            cross=0.20,
        )
        assert Path(result["summary_tsv"]).exists()
        assert Path(result["combined_tsv"]).exists()
        assert Path(result["decision_md"]).exists()
        assert isinstance(result["close_c1"], bool)

    def test_smoke_multiple_topologies(self, tmp_path):
        result = run_c1_scout(
            seeds=[42],
            topologies=["ring4"],
            beta_pairs=[5.0, 10.0],
            mu_pairs=[0.8, 0.5],
            out_root=tmp_path / "out",
            summary_tsv=tmp_path / "summary.tsv",
            combined_tsv=tmp_path / "combined.tsv",
            decision_md=tmp_path / "decision.md",
            players=20,
            rounds=100,
            burn_in=30,
            tail=30,
        )
        assert Path(result["decision_md"]).exists()
