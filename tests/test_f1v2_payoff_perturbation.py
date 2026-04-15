"""Tests for F1v2 Payoff Perturbation (Phase 16).

Covers:
1. Payoff perturbation backward compatibility (ε=0 → no change)
2. Perturbation effect on Q-table divergence
3. F1v2 harness condition naming & seed metrics
"""

from __future__ import annotations

import random
import unittest

from evolution.independent_rl import (
    _NSTRATS,
    apply_per_player_payoff_perturbation,
    sample_payoff_perturbation,
)


class TestPayoffPerturbationMechanics(unittest.TestCase):
    """Verify per-player payoff perturbation sampling and application."""

    def test_zero_epsilon_all_zero(self):
        deltas = sample_payoff_perturbation(10, epsilon=0.0, rng=random.Random(42))
        for d in deltas:
            self.assertEqual(d, [0.0, 0.0, 0.0])

    def test_nonzero_epsilon_bounded(self):
        eps = 0.05
        deltas = sample_payoff_perturbation(100, epsilon=eps, rng=random.Random(42))
        for d in deltas:
            for v in d:
                self.assertGreaterEqual(v, -eps)
                self.assertLessEqual(v, eps)

    def test_perturbation_applied_correctly(self):
        rewards = [1.0, 2.0, 3.0]
        chosen = [0, 1, 2]
        perturbations = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        result = apply_per_player_payoff_perturbation(rewards, chosen, perturbations)
        self.assertAlmostEqual(result[0], 1.0 + 0.1)  # player 0 chose strategy 0
        self.assertAlmostEqual(result[1], 2.0 + 0.5)  # player 1 chose strategy 1
        self.assertAlmostEqual(result[2], 3.0 + 0.9)  # player 2 chose strategy 2


class TestPayoffEpsilonBackwardCompat(unittest.TestCase):
    """Verify payoff_epsilon=0 produces bit-for-bit identical results."""

    def test_epsilon_zero_same_as_default(self):
        from simulation.e1_heterogeneous_rl import _run_e1_simulation, _make_graph_spec

        common = dict(
            n_players=20, n_rounds=50, seed=42,
            graph_spec=None,
            alpha_lo=0.005, alpha_hi=0.40,
            beta_lo=3.0, beta_hi=3.0,
            a=1.0, b=0.9, cross=0.20,
        )
        rows_a, _, _, _, _, _ = _run_e1_simulation(**common)
        rows_b, _, _, _, _, _ = _run_e1_simulation(**common, payoff_epsilon=0.0)

        self.assertEqual(len(rows_a), len(rows_b))
        for ra, rb in zip(rows_a, rows_b):
            for k in ra:
                self.assertEqual(ra[k], rb[k], f"Mismatch at key {k}")


class TestPayoffEpsilonEffect(unittest.TestCase):
    """Verify that positive epsilon changes simulation output."""

    def test_positive_epsilon_changes_output(self):
        from simulation.e1_heterogeneous_rl import _run_e1_simulation

        common = dict(
            n_players=30, n_rounds=100, seed=42,
            graph_spec=None,
            alpha_lo=0.005, alpha_hi=0.40,
            beta_lo=3.0, beta_hi=3.0,
            a=1.0, b=0.9, cross=0.20,
        )
        rows_base, _, _, _, _, _ = _run_e1_simulation(**common, payoff_epsilon=0.0)
        rows_pert, _, _, _, _, _ = _run_e1_simulation(**common, payoff_epsilon=0.08)

        # Trajectories should differ (perturbation changes rewards → different Q evolution)
        diffs = sum(
            1 for a, b in zip(rows_base, rows_pert)
            if abs(a["p_aggressive"] - b["p_aggressive"]) > 1e-10
        )
        self.assertGreater(diffs, 0, "ε=0.08 should produce different trajectory")


class TestF1v2Harness(unittest.TestCase):
    """Test harness utilities."""

    def test_cond_name_format(self):
        from simulation.f1v2_payoff_perturbation import _cond_name
        name = _cond_name(topology="lattice4", payoff_epsilon=0.04)
        self.assertEqual(name, "g2_f1v2_lattice4_eps0p04")

    def test_cond_name_zero_epsilon(self):
        from simulation.f1v2_payoff_perturbation import _cond_name
        name = _cond_name(topology="well_mixed", payoff_epsilon=0.0)
        self.assertEqual(name, "g2_f1v2_well_mixed_eps0")

    def test_seed_metrics_with_cc1(self):
        """Verify _seed_metrics invokes CC1 fallback (no crash)."""
        from simulation.f1v2_payoff_perturbation import _seed_metrics
        # Minimal synthetic data (flat → CL=1, but should not crash)
        rows = [{"p_aggressive": 0.34, "p_defensive": 0.33, "p_balanced": 0.33}
                for _ in range(100)]
        m = _seed_metrics(rows, burn_in=10, tail=50, eta=0.55, corr_threshold=0.09)
        self.assertIn("cycle_level", m)
        self.assertIn("env_gamma", m)


if __name__ == "__main__":
    unittest.main()
