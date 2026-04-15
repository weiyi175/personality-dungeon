"""Tests for G1 Entropy Regularization (Phase 15).

Tests cover:
1. Backward compatibility: entropy_lambda=0 → same as before
2. Entropy bonus mechanics: rewards are modified correctly
3. G1 harness basics: condition naming, seed metrics with CC1 fallback
"""

from __future__ import annotations

import random
import unittest

from evolution.independent_rl import (
    boltzmann_weights,
    rl_q_update,
    weight_entropy,
)
from simulation.e1_heterogeneous_rl import _run_e1_simulation


class TestEntropyBonusMechanics(unittest.TestCase):
    """Verify the entropy bonus computation is correct."""

    def test_entropy_of_uniform_weights(self):
        """Uniform weights → max entropy ≈ ln(3) ≈ 1.0986."""
        from math import log
        w = [1/3, 1/3, 1/3]
        h = weight_entropy(w)
        self.assertAlmostEqual(h, log(3), places=4)

    def test_entropy_of_peaked_weights(self):
        """Peaked weights → low entropy."""
        w = [0.98, 0.01, 0.01]
        h = weight_entropy(w)
        self.assertLess(h, 0.15)

    def test_entropy_bonus_increases_reward(self):
        """Adding λ·H(w) to reward gives higher effective reward."""
        q = [0.1, 0.2, 0.3]
        beta = 5.0
        w = boltzmann_weights(q, beta=beta)
        h = weight_entropy(w)
        raw_reward = 0.5
        lam = 0.1
        boosted = raw_reward + lam * h
        self.assertGreater(boosted, raw_reward)


class TestBackwardCompatibility(unittest.TestCase):
    """entropy_lambda=0 must produce identical results to pre-G1."""

    def test_lambda_zero_same_as_default(self):
        """_run_e1_simulation with entropy_lambda=0 gives same output as without."""
        kwargs = dict(
            n_players=20, n_rounds=50, seed=42,
            graph_spec=None,
            alpha_lo=0.1, alpha_hi=0.3,
            beta_lo=5.0, beta_hi=5.0,
            a=1.0, b=0.9, cross=0.2,
            init_q=0.0, control_degree=4,
        )
        # Without entropy_lambda (default = 0.0)
        rows_a, _, _, _, _, _ = _run_e1_simulation(**kwargs)
        # With explicit entropy_lambda=0.0
        rows_b, _, _, _, _, _ = _run_e1_simulation(**kwargs, entropy_lambda=0.0)

        self.assertEqual(len(rows_a), len(rows_b))
        for ra, rb in zip(rows_a, rows_b):
            for k in ra:
                self.assertAlmostEqual(float(ra[k]), float(rb[k]), places=10,
                                       msg=f"Mismatch at key={k}")


class TestEntropyRegEffect(unittest.TestCase):
    """Verify that entropy_lambda > 0 changes the dynamics."""

    def test_positive_lambda_changes_output(self):
        """entropy_lambda > 0 should produce different trajectories."""
        kwargs = dict(
            n_players=30, n_rounds=200, seed=42,
            graph_spec=None,
            alpha_lo=0.05, alpha_hi=0.2,
            beta_lo=10.0, beta_hi=10.0,
            a=1.0, b=0.9, cross=0.2,
            init_q=0.0, control_degree=4,
        )
        rows_base, _, _, _, _, _ = _run_e1_simulation(**kwargs, entropy_lambda=0.0)
        rows_reg, _, _, _, _, _ = _run_e1_simulation(**kwargs, entropy_lambda=0.1)

        # Trajectories should diverge
        diffs = 0
        for ra, rb in zip(rows_base, rows_reg):
            if abs(float(ra["p_aggressive"]) - float(rb["p_aggressive"])) > 1e-6:
                diffs += 1
        self.assertGreater(diffs, 0, "entropy_lambda should change dynamics")

    def test_lambda_affects_entropy(self):
        """Nonzero λ should change mean player entropy vs λ=0."""
        kwargs = dict(
            n_players=50, n_rounds=500, seed=42,
            graph_spec=None,
            alpha_lo=0.05, alpha_hi=0.2,
            beta_lo=10.0, beta_hi=10.0,
            a=1.0, b=0.9, cross=0.2,
            init_q=0.0, control_degree=4,
        )
        _, diag_base, _, _, _, _ = _run_e1_simulation(**kwargs, entropy_lambda=0.0)
        _, diag_reg, _, _, _, _ = _run_e1_simulation(**kwargs, entropy_lambda=0.10)

        tail_h_base = sum(d["mean_player_weight_entropy"] for d in diag_base[-100:]) / 100
        tail_h_reg = sum(d["mean_player_weight_entropy"] for d in diag_reg[-100:]) / 100
        self.assertNotAlmostEqual(tail_h_base, tail_h_reg, places=2,
                                  msg="λ>0 should change entropy dynamics")


class TestG1Harness(unittest.TestCase):
    """Test G1 harness utilities."""

    def test_cond_name_format(self):
        from simulation.g1_entropy_reg import _cond_name
        name = _cond_name(topology="well_mixed", beta_ceiling=3.0, entropy_lambda=0.05)
        self.assertEqual(name, "g2_g1_well_mixed_b3_lam0p05")

    def test_cond_name_integer_lambda(self):
        from simulation.g1_entropy_reg import _cond_name
        name = _cond_name(topology="lattice4", beta_ceiling=10.0, entropy_lambda=0.1)
        self.assertEqual(name, "g2_g1_lattice4_b10_lam0p1")

    def test_seed_metrics_with_cc1(self):
        """G1 _seed_metrics should use CC1 fallback."""
        from simulation.g1_entropy_reg import _seed_metrics
        # Generate short run
        rows, _, _, _, _, _ = _run_e1_simulation(
            n_players=20, n_rounds=100, seed=42,
            graph_spec=None,
            alpha_lo=0.1, alpha_hi=0.3,
            beta_lo=5.0, beta_hi=5.0,
            a=1.0, b=0.9, cross=0.2,
            init_q=0.0, control_degree=4,
        )
        sm = _seed_metrics(rows, burn_in=10, tail=50, eta=0.55, corr_threshold=0.09)
        self.assertIn("cycle_level", sm)
        self.assertIn("env_gamma", sm)
        self.assertIsInstance(sm["cycle_level"], int)


if __name__ == "__main__":
    unittest.main()
