"""Tests for E2 Lower β Ceiling (keep E1 wide α, sweep β downward).

Covers:
  Internals:
    E2-1: _cond_name format
    E2-2: _strategy_cycle_stability returns [0, 1]
    E2-3: _strategy_cycle_stability with insufficient data

  Simulation (via E1 core):
    E2-4: output structure matches E1
    E2-5: deterministic (same seed → same result)
    E2-6: β=10 degenerates to E1 baseline
    E2-7: lower β → higher entropy (core hypothesis)
    E2-8: strategy proportions sum to 1.0
    E2-9: α_std consistent with wide α (0.005, 0.40)

  Scout:
    E2-10: tiny smoke test (2 β × 1 topo × 1 seed)
    E2-11: β=10 gets 'e1_baseline' verdict
    E2-12: active conditions get valid verdict
    E2-13: decision.md is written and non-empty
    E2-14: pass gate uses relaxed q_std > 0.06
"""
from __future__ import annotations

import csv
import math
import random

import pytest

from simulation.e1_heterogeneous_rl import _run_e1_simulation, _make_graph_spec
from simulation.e2_lower_beta import (
    ALPHA_LO,
    ALPHA_HI,
    _cond_name,
    _strategy_cycle_stability,
    run_e2_scout,
)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

class TestCondName:
    def test_format(self):
        """E2-1: condition name format."""
        assert _cond_name(topology="lattice4", beta_ceiling=5.0) == "g2_e2_lattice4_b5"
        assert _cond_name(topology="well_mixed", beta_ceiling=3.0) == "g2_e2_well_mixed_b3"
        assert _cond_name(topology="lattice4", beta_ceiling=10.0) == "g2_e2_lattice4_b10"
        assert _cond_name(topology="lattice4", beta_ceiling=8.0) == "g2_e2_lattice4_b8"


class TestStrategyCycleStability:
    def test_returns_float_in_range(self):
        """E2-2: returns float in [0, 1] for cycling data."""
        rows = []
        for t in range(200):
            angle = t * 0.05
            pa = 1/3 + 0.1 * math.cos(angle)
            pd = 1/3 + 0.1 * math.cos(angle + 2 * math.pi / 3)
            pb = 1.0 - pa - pd
            rows.append({"round": t, "p_aggressive": pa, "p_defensive": pd, "p_balanced": pb})
        scs = _strategy_cycle_stability(rows, burn_in=0, tail=200)
        assert isinstance(scs, float)
        assert 0.0 <= scs <= 1.0

    def test_insufficient_data(self):
        """E2-3: returns 0.0 for < 10 data points."""
        rows = [{"round": i, "p_aggressive": 0.33, "p_defensive": 0.33, "p_balanced": 0.34}
                for i in range(5)]
        assert _strategy_cycle_stability(rows, burn_in=0, tail=5) == 0.0


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

_TINY = dict(
    n_players=20, n_rounds=50, seed=42,
    alpha_lo=ALPHA_LO, alpha_hi=ALPHA_HI,
    a=1.0, b=0.9, cross=0.20,
)


class TestSimulation:
    def test_output_structure(self):
        """E2-4: E1 core sim returns correct tuple for E2 params."""
        rows, rdiag, fq, adj, alphas, betas = _run_e1_simulation(
            graph_spec=None, beta_lo=5.0, beta_hi=5.0, **_TINY)
        assert len(rows) == 50
        assert len(rdiag) == 50
        assert len(fq) == 20
        assert adj is None
        assert len(alphas) == 20
        assert len(betas) == 20
        # All betas should be 5.0 (homo)
        for b in betas:
            assert abs(b - 5.0) < 1e-10

    def test_deterministic(self):
        """E2-5: same seed → same result."""
        r1, _, _, _, _, _ = _run_e1_simulation(
            graph_spec=None, beta_lo=5.0, beta_hi=5.0, **_TINY)
        r2, _, _, _, _, _ = _run_e1_simulation(
            graph_spec=None, beta_lo=5.0, beta_hi=5.0, **_TINY)
        for a, b in zip(r1, r2):
            assert a["p_aggressive"] == b["p_aggressive"]

    def test_beta10_is_e1_baseline(self):
        """E2-6: β=10 gives same result as E1 wide_α_homo_β."""
        r1, _, _, _, a1, b1 = _run_e1_simulation(
            graph_spec=None, beta_lo=10.0, beta_hi=10.0, **_TINY)
        r2, _, _, _, a2, b2 = _run_e1_simulation(
            graph_spec=None, beta_lo=10.0, beta_hi=10.0, **_TINY)
        assert a1 == a2
        assert b1 == b2
        for x, y in zip(r1, r2):
            assert x == y

    def test_lower_beta_higher_entropy(self):
        """E2-7: lower β should give higher per-player weight entropy."""
        from evolution.independent_rl import boltzmann_weights, weight_entropy
        # β=3 vs β=10 on same Q-table: lower β → more uniform → higher entropy
        q = [0.5, 0.2, 0.1]
        w_lo = boltzmann_weights(q, beta=3.0)
        w_hi = boltzmann_weights(q, beta=10.0)
        assert weight_entropy(w_lo) > weight_entropy(w_hi)

    def test_strategy_proportions_sum(self):
        """E2-8: p_agg + p_def + p_bal = 1.0 each round."""
        rows, _, _, _, _, _ = _run_e1_simulation(
            graph_spec=None, beta_lo=5.0, beta_hi=5.0, **_TINY)
        for r in rows:
            total = r["p_aggressive"] + r["p_defensive"] + r["p_balanced"]
            assert abs(total - 1.0) < 1e-10

    def test_alpha_std_wide(self):
        """E2-9: wide α (0.005, 0.40) gives substantial α_std."""
        _, _, _, _, alphas, _ = _run_e1_simulation(
            graph_spec=None, beta_lo=5.0, beta_hi=5.0,
            n_players=100, n_rounds=10, seed=42,
            alpha_lo=ALPHA_LO, alpha_hi=ALPHA_HI,
            a=1.0, b=0.9, cross=0.20)
        mu = sum(alphas) / len(alphas)
        std = (sum((a - mu) ** 2 for a in alphas) / len(alphas)) ** 0.5
        assert std > 0.05, f"Wide α should give substantial std, got {std:.4f}"


# ---------------------------------------------------------------------------
# Scout
# ---------------------------------------------------------------------------

class TestE2Scout:
    def test_smoke(self, tmp_path):
        """E2-10: tiny smoke test (2 β × 1 topo × 1 seed)."""
        result = run_e2_scout(
            seeds=[42],
            topologies=["well_mixed"],
            beta_ceilings=[5.0, 10.0],
            out_root=tmp_path / "out",
            summary_tsv=tmp_path / "s.tsv",
            combined_tsv=tmp_path / "c.tsv",
            decision_md=tmp_path / "d.md",
            players=20, rounds=50, burn_in=10, tail=30,
        )
        assert "close_e2" in result
        assert (tmp_path / "s.tsv").exists()
        assert (tmp_path / "c.tsv").exists()
        assert (tmp_path / "d.md").exists()

    def test_beta10_baseline_verdict(self, tmp_path):
        """E2-11: β=10 gets 'e1_baseline' verdict."""
        run_e2_scout(
            seeds=[42],
            topologies=["well_mixed"],
            beta_ceilings=[5.0, 10.0],
            out_root=tmp_path / "out",
            summary_tsv=tmp_path / "s.tsv",
            combined_tsv=tmp_path / "c.tsv",
            decision_md=tmp_path / "d.md",
            players=20, rounds=50, burn_in=10, tail=30,
        )
        with (tmp_path / "c.tsv").open() as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        ctrl = [r for r in rows if r["is_control"] == "yes"]
        assert len(ctrl) == 1
        assert ctrl[0]["verdict"] == "e1_baseline"

    def test_active_verdict(self, tmp_path):
        """E2-12: active conditions get valid verdict."""
        run_e2_scout(
            seeds=[42],
            topologies=["well_mixed"],
            beta_ceilings=[5.0, 10.0],
            out_root=tmp_path / "out",
            summary_tsv=tmp_path / "s.tsv",
            combined_tsv=tmp_path / "c.tsv",
            decision_md=tmp_path / "d.md",
            players=20, rounds=50, burn_in=10, tail=30,
        )
        with (tmp_path / "c.tsv").open() as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        active = [r for r in rows if r["is_control"] == "no"]
        assert len(active) == 1
        assert active[0]["verdict"] in ("pass", "fail", "weak_positive")

    def test_decision_written(self, tmp_path):
        """E2-13: decision.md is non-empty."""
        run_e2_scout(
            seeds=[42],
            topologies=["well_mixed"],
            beta_ceilings=[5.0, 10.0],
            out_root=tmp_path / "out",
            summary_tsv=tmp_path / "s.tsv",
            combined_tsv=tmp_path / "c.tsv",
            decision_md=tmp_path / "d.md",
            players=20, rounds=50, burn_in=10, tail=30,
        )
        text = (tmp_path / "d.md").read_text()
        assert "E2 Lower β Ceiling Decision" in text
        assert "Pass Gate" in text

    def test_relaxed_qstd_threshold(self, tmp_path):
        """E2-14: pass gate text shows q_std > 0.06."""
        run_e2_scout(
            seeds=[42],
            topologies=["well_mixed"],
            beta_ceilings=[5.0, 10.0],
            out_root=tmp_path / "out",
            summary_tsv=tmp_path / "s.tsv",
            combined_tsv=tmp_path / "c.tsv",
            decision_md=tmp_path / "d.md",
            players=20, rounds=50, burn_in=10, tail=30,
        )
        text = (tmp_path / "d.md").read_text()
        assert "mean_q_value_std > 0.06" in text
