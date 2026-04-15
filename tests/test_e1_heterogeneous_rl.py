"""Tests for E1 Heterogeneous RL module + harness."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from evolution.independent_rl import (
    _NSTRATS,
    boltzmann_weights,
    rl_q_update,
    weight_entropy,
)
from evolution.local_graph import GraphSpec
from simulation.e1_heterogeneous_rl import (
    _cond_name,
    _ctrl_name,
    _make_graph_spec,
    _parse_ranges,
    _run_e1_simulation,
    _sample_player_params,
    _seed_metrics,
    _std,
    _tail_e1_diagnostics,
    run_e1_scout,
)


# ======================================================================
# Parameter sampling
# ======================================================================

class TestSamplePlayerParams:
    def test_homo_alpha(self):
        alphas, betas = _sample_player_params(
            100, alpha_lo=0.10, alpha_hi=0.10,
            beta_lo=5.0, beta_hi=5.0, rng=random.Random(42))
        assert all(a == pytest.approx(0.10) for a in alphas)
        assert all(b == pytest.approx(5.0) for b in betas)

    def test_heterogeneous_alpha(self):
        alphas, _ = _sample_player_params(
            300, alpha_lo=0.01, alpha_hi=0.50,
            beta_lo=10.0, beta_hi=10.0, rng=random.Random(42))
        assert min(alphas) >= 0.01
        assert max(alphas) <= 0.50
        assert _std(alphas) > 0.05  # spread should be substantial

    def test_heterogeneous_beta(self):
        _, betas = _sample_player_params(
            300, alpha_lo=0.10, alpha_hi=0.10,
            beta_lo=1.0, beta_hi=40.0, rng=random.Random(42))
        assert min(betas) >= 1.0
        assert max(betas) <= 40.0
        assert _std(betas) > 5.0

    def test_clipping(self):
        alphas, betas = _sample_player_params(
            50, alpha_lo=-1.0, alpha_hi=2.0,
            beta_lo=-5.0, beta_hi=200.0, rng=random.Random(42))
        assert all(0.0 <= a <= 1.0 for a in alphas)
        assert all(0.1 <= b <= 100.0 for b in betas)

    def test_reproducible(self):
        p1 = _sample_player_params(20, alpha_lo=0.01, alpha_hi=0.50,
                                    beta_lo=1.0, beta_hi=30.0, rng=random.Random(99))
        p2 = _sample_player_params(20, alpha_lo=0.01, alpha_hi=0.50,
                                    beta_lo=1.0, beta_hi=30.0, rng=random.Random(99))
        assert p1 == p2


class TestParseRanges:
    def test_single(self):
        assert _parse_ranges("0.1:0.2") == [(0.1, 0.2)]

    def test_multiple(self):
        assert _parse_ranges("0.1:0.2,0.3:0.4") == [(0.1, 0.2), (0.3, 0.4)]

    def test_homo(self):
        assert _parse_ranges("10.0:10.0") == [(10.0, 10.0)]


# ======================================================================
# Naming
# ======================================================================

class TestCondName:
    def test_active(self):
        name = _cond_name(topology="lattice4", alpha_lo=0.02, alpha_hi=0.20,
                          beta_lo=3.0, beta_hi=20.0)
        assert "lattice4" in name
        assert "e1" in name

    def test_control(self):
        assert _ctrl_name("well_mixed") == "g2_e1_frozen_well_mixed"


class TestMakeGraphSpec:
    def test_lattice4(self):
        gs = _make_graph_spec("lattice4")
        assert gs is not None
        assert gs.topology == "lattice4"

    def test_well_mixed(self):
        assert _make_graph_spec("well_mixed") is None

    def test_unknown(self):
        with pytest.raises(ValueError):
            _make_graph_spec("star")


# ======================================================================
# Core simulation
# ======================================================================

class TestRunE1Simulation:
    TINY = dict(n_players=20, n_rounds=50, seed=42,
                a=1.0, b=0.9, cross=0.2, init_q=0.0)

    def test_returns_correct_structure(self):
        rows, diag, fq, adj, pa, pb = _run_e1_simulation(
            graph_spec=None,
            alpha_lo=0.02, alpha_hi=0.20,
            beta_lo=3.0, beta_hi=20.0, **self.TINY)
        assert len(rows) == 50
        assert len(diag) == 50
        assert len(fq) == 20
        assert adj is None
        assert len(pa) == 20
        assert len(pb) == 20

    def test_deterministic(self):
        kw = dict(graph_spec=None, alpha_lo=0.05, alpha_hi=0.15,
                  beta_lo=5.0, beta_hi=15.0, **self.TINY)
        r1, _, q1, _, a1, b1 = _run_e1_simulation(**kw)
        r2, _, q2, _, a2, b2 = _run_e1_simulation(**kw)
        assert r1 == r2
        assert a1 == a2
        assert b1 == b2

    def test_frozen_control(self):
        """alpha_lo=alpha_hi=0 → Q unchanged."""
        _, _, fq, _, _, _ = _run_e1_simulation(
            graph_spec=None,
            alpha_lo=0.0, alpha_hi=0.0,
            beta_lo=5.0, beta_hi=5.0, **self.TINY)
        for pq in fq:
            for q in pq:
                assert q == pytest.approx(0.0)

    def test_homo_degenerates_to_d1(self):
        """homo alpha/beta → all players get same params."""
        _, _, _, _, pa, pb = _run_e1_simulation(
            graph_spec=None,
            alpha_lo=0.10, alpha_hi=0.10,
            beta_lo=10.0, beta_hi=10.0, **self.TINY)
        assert all(a == pytest.approx(0.10) for a in pa)
        assert all(b == pytest.approx(10.0) for b in pb)

    def test_hetero_produces_q_divergence(self):
        """Wide α/β range → more Q diversity than homo."""
        # Homo
        _, diag_h, _, _, _, _ = _run_e1_simulation(
            graph_spec=None,
            alpha_lo=0.10, alpha_hi=0.10,
            beta_lo=10.0, beta_hi=10.0,
            n_players=50, n_rounds=500, seed=42,
            a=1.0, b=0.9, cross=0.2, init_q=0.0)
        # Hetero
        _, diag_e, _, _, _, _ = _run_e1_simulation(
            graph_spec=None,
            alpha_lo=0.005, alpha_hi=0.40,
            beta_lo=1.0, beta_hi=40.0,
            n_players=50, n_rounds=500, seed=42,
            a=1.0, b=0.9, cross=0.2, init_q=0.0)
        # Compare tail q_std
        tail_h = [d["mean_q_value_std"] for d in diag_h[-100:]]
        tail_e = [d["mean_q_value_std"] for d in diag_e[-100:]]
        mean_h = sum(tail_h) / len(tail_h)
        mean_e = sum(tail_e) / len(tail_e)
        assert mean_e > mean_h  # heterogeneous should have higher q_std

    def test_strategy_proportions_valid(self):
        rows, _, _, _, _, _ = _run_e1_simulation(
            graph_spec=None,
            alpha_lo=0.02, alpha_hi=0.20,
            beta_lo=3.0, beta_hi=20.0, **self.TINY)
        for r in rows:
            s = r["p_aggressive"] + r["p_defensive"] + r["p_balanced"]
            assert s == pytest.approx(1.0, abs=1e-9)

    def test_lattice4_returns_adj(self):
        gs = GraphSpec(topology="lattice4", degree=4, lattice_rows=4, lattice_cols=5)
        _, _, _, adj, _, _ = _run_e1_simulation(
            graph_spec=gs,
            alpha_lo=0.05, alpha_hi=0.15,
            beta_lo=5.0, beta_hi=15.0, **self.TINY)
        assert adj is not None
        assert len(adj) == 20

    def test_diagnostics_keys(self):
        _, diag, _, _, _, _ = _run_e1_simulation(
            graph_spec=None,
            alpha_lo=0.05, alpha_hi=0.15,
            beta_lo=5.0, beta_hi=15.0, **self.TINY)
        expected = {"mean_player_weight_entropy", "weight_heterogeneity_std",
                    "mean_q_value_std", "mean_neighbor_weight_cosine"}
        for d in diag:
            assert expected.issubset(d.keys())


class TestSeedMetrics:
    def test_returns_expected_keys(self):
        rows, _, _, _, _, _ = _run_e1_simulation(
            graph_spec=None,
            alpha_lo=0.05, alpha_hi=0.15,
            beta_lo=5.0, beta_hi=15.0,
            n_players=20, n_rounds=100, seed=42,
            a=1.0, b=0.9, cross=0.2, init_q=0.0)
        sm = _seed_metrics(rows, burn_in=30, tail=50, eta=0.55,
                           corr_threshold=0.09)
        assert "cycle_level" in sm
        assert 0 <= sm["cycle_level"] <= 3


class TestTailDiagnostics:
    def test_returns_expected_keys(self):
        _, diag, fq, adj, _, pb = _run_e1_simulation(
            graph_spec=None,
            alpha_lo=0.05, alpha_hi=0.15,
            beta_lo=5.0, beta_hi=15.0,
            n_players=20, n_rounds=100, seed=42,
            a=1.0, b=0.9, cross=0.2, init_q=0.0)
        td = _tail_e1_diagnostics(diag, fq, adj, pb,
                                   n_rows=100, burn_in=30, tail=50)
        assert "mean_player_weight_entropy" in td
        assert "mean_q_value_std" in td


# ======================================================================
# End-to-end scout
# ======================================================================

class TestE1ScoutSmoke:
    def test_tiny_scout(self, tmp_path: Path):
        result = run_e1_scout(
            seeds=[42],
            topologies=["well_mixed"],
            alpha_ranges=[(0.05, 0.15)],
            beta_ranges=[(5.0, 15.0)],
            out_root=tmp_path / "e1",
            summary_tsv=tmp_path / "e1" / "summary.tsv",
            combined_tsv=tmp_path / "e1" / "combined.tsv",
            decision_md=tmp_path / "e1" / "decision.md",
            players=20, rounds=100, burn_in=30, tail=50,
        )
        assert (tmp_path / "e1" / "summary.tsv").exists()
        assert (tmp_path / "e1" / "combined.tsv").exists()
        assert (tmp_path / "e1" / "decision.md").exists()
        assert "close_e1" in result

    def test_with_lattice4(self, tmp_path: Path):
        result = run_e1_scout(
            seeds=[42],
            topologies=["lattice4"],
            alpha_ranges=[(0.02, 0.20)],
            beta_ranges=[(3.0, 20.0)],
            out_root=tmp_path / "e1",
            summary_tsv=tmp_path / "e1" / "summary.tsv",
            combined_tsv=tmp_path / "e1" / "combined.tsv",
            decision_md=tmp_path / "e1" / "decision.md",
            players=300, rounds=50, burn_in=15, tail=25,
        )
        assert (tmp_path / "e1" / "decision.md").exists()

    def test_decision_has_verdict(self, tmp_path: Path):
        run_e1_scout(
            seeds=[42],
            topologies=["well_mixed"],
            alpha_ranges=[(0.10, 0.10)],
            beta_ranges=[(10.0, 10.0)],
            out_root=tmp_path / "e1",
            summary_tsv=tmp_path / "e1" / "summary.tsv",
            combined_tsv=tmp_path / "e1" / "combined.tsv",
            decision_md=tmp_path / "e1" / "decision.md",
            players=20, rounds=50, burn_in=15, tail=25,
        )
        txt = (tmp_path / "e1" / "decision.md").read_text()
        assert "verdict=" in txt
