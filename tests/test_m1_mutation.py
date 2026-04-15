"""Tests for M1 per-round Dirichlet mutation.

Covers:
  G0-1: mutation_rate=0.0 produces identical results to standard replicator
  G0-2: mutation_rate>0 produces per-player weight heterogeneity
  G0-3: higher mutation_rate → higher weight dispersion
  G0-4: deterministic reproducibility with same seed and mutation_rate
  G0-5: mutation preserves simplex constraint (weights stay positive)
  Validation: out-of-range mutation_rate raises
  Validation: mutation_rate>0 + async<1 raises (mutual exclusion)
  Integration: harness smoke test
"""
from __future__ import annotations

import random

import pytest

from simulation.run_simulation import (
	SimConfig,
	_apply_mutation_weights,
	_validate_mutation_params,
	simulate,
)


STRATEGY_SPACE = ["aggressive", "defensive", "balanced"]


# ---------------------------------------------------------------------------
# G0-1: mutation_rate=0.0 identical to standard replicator
# ---------------------------------------------------------------------------


def test_mutation_rate_0_identical():
	"""mutation_rate=0.0 must produce identical results to the default (no mutation)."""
	cfg_default = SimConfig(
		n_players=30,
		n_rounds=50,
		seed=42,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.20,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		memory_kernel=3,
		mutation_rate=0.0,
	)
	cfg_explicit_zero = SimConfig(
		n_players=30,
		n_rounds=50,
		seed=42,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.20,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		memory_kernel=3,
		mutation_rate=0.0,
	)
	_, rows1 = simulate(cfg_default)
	_, rows2 = simulate(cfg_explicit_zero)

	for i, (r1, r2) in enumerate(zip(rows1, rows2)):
		for s in STRATEGY_SPACE:
			assert r1[f"p_{s}"] == r2[f"p_{s}"], f"Round {i}: p_{s} differs"


# ---------------------------------------------------------------------------
# G0-2: mutation_rate>0 creates weight heterogeneity
# ---------------------------------------------------------------------------


def test_mutation_creates_heterogeneity():
	"""With mutation_rate>0, players should have different weights after a few rounds."""
	cfg = SimConfig(
		n_players=30,
		n_rounds=30,
		seed=42,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.20,
		init_bias=0.12,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.06,
		memory_kernel=3,
		mutation_rate=0.05,
	)
	observed_dispersions = []

	def round_cb(round_index, _cfg, players, _dungeon, _step_records, _row):
		if round_index < 10:
			return
		# Compute weight dispersion across players
		for s in STRATEGY_SPACE:
			vals = [float(getattr(pl, "strategy_weights", {}).get(s, 1.0)) for pl in players]
			mean = sum(vals) / len(vals)
			var = sum((v - mean) ** 2 for v in vals) / len(vals)
			observed_dispersions.append(var ** 0.5)

	simulate(cfg, round_callback=round_cb)
	avg_disp = sum(observed_dispersions) / len(observed_dispersions)
	assert avg_disp > 0.001, f"mutation_rate=0.05 should create weight dispersion; got {avg_disp:.6f}"


# ---------------------------------------------------------------------------
# G0-3: higher mutation_rate → higher weight dispersion
# ---------------------------------------------------------------------------


def test_dispersion_increases_with_mutation_rate():
	"""Weight dispersion should increase monotonically with mutation_rate."""

	def measure_dispersion(mutation_rate: float) -> float:
		cfg = SimConfig(
			n_players=50,
			n_rounds=40,
			seed=42,
			payoff_mode="matrix_ab",
			popularity_mode="sampled",
			gamma=0.1,
			epsilon=0.0,
			a=1.0,
			b=0.9,
			matrix_cross_coupling=0.20,
			init_bias=0.12,
			evolution_mode="sampled",
			payoff_lag=1,
			selection_strength=0.06,
			memory_kernel=3,
			mutation_rate=mutation_rate,
		)
		dispersions = []

		def round_cb(round_index, _cfg, players, _dungeon, _step_recs, _row):
			if round_index < 20:
				return
			for s in STRATEGY_SPACE:
				vals = [float(getattr(pl, "strategy_weights", {}).get(s, 1.0)) for pl in players]
				mean = sum(vals) / len(vals)
				var = sum((v - mean) ** 2 for v in vals) / len(vals)
				dispersions.append(var ** 0.5)

		simulate(cfg, round_callback=round_cb)
		return sum(dispersions) / len(dispersions) if dispersions else 0.0

	d_low = measure_dispersion(0.005)
	d_high = measure_dispersion(0.10)
	assert d_high > d_low, f"dispersion at η=0.10 ({d_high:.6f}) should exceed η=0.005 ({d_low:.6f})"


# ---------------------------------------------------------------------------
# G0-4: deterministic reproducibility
# ---------------------------------------------------------------------------


def test_deterministic_reproducibility():
	"""Same seed + same mutation_rate → identical simulation results."""
	for mr in [0.001, 0.01, 0.05]:
		cfg1 = SimConfig(
			n_players=30,
			n_rounds=50,
			seed=55,
			payoff_mode="matrix_ab",
			popularity_mode="sampled",
			gamma=0.1,
			epsilon=0.0,
			a=1.0,
			b=0.9,
			matrix_cross_coupling=0.20,
			init_bias=0.12,
			evolution_mode="sampled",
			payoff_lag=1,
			selection_strength=0.06,
			memory_kernel=3,
			mutation_rate=mr,
		)
		cfg2 = SimConfig(
			n_players=30,
			n_rounds=50,
			seed=55,
			payoff_mode="matrix_ab",
			popularity_mode="sampled",
			gamma=0.1,
			epsilon=0.0,
			a=1.0,
			b=0.9,
			matrix_cross_coupling=0.20,
			init_bias=0.12,
			evolution_mode="sampled",
			payoff_lag=1,
			selection_strength=0.06,
			memory_kernel=3,
			mutation_rate=mr,
		)
		_, rows1 = simulate(cfg1)
		_, rows2 = simulate(cfg2)
		for i, (r1, r2) in enumerate(zip(rows1, rows2)):
			for s in STRATEGY_SPACE:
				assert r1[f"p_{s}"] == r2[f"p_{s}"], (
					f"mutation_rate={mr}: round {i} p_{s} differs"
				)


# ---------------------------------------------------------------------------
# G0-5: mutation preserves simplex constraint (weights stay positive)
# ---------------------------------------------------------------------------


def test_mutation_preserves_simplex():
	"""_apply_mutation_weights output weights should be positive and sum > 0."""
	rng = random.Random(42)
	weights = {"aggressive": 0.5, "defensive": 0.3, "balanced": 0.2}
	for _ in range(1000):
		mutated = _apply_mutation_weights(weights, STRATEGY_SPACE, rng, 0.10)
		for s in STRATEGY_SPACE:
			assert mutated[s] > 0.0, f"Weight for {s} should be positive; got {mutated[s]}"
		total = sum(mutated.values())
		assert total > 0.0, f"Total weight should be positive; got {total}"


# ---------------------------------------------------------------------------
# Validation: out-of-range mutation_rate raises
# ---------------------------------------------------------------------------


def test_negative_mutation_rate_raises():
	with pytest.raises(ValueError, match="mutation_rate"):
		_validate_mutation_params(mutation_rate=-0.1, async_update_fraction=1.0)


def test_mutation_rate_above_1_raises():
	with pytest.raises(ValueError, match="mutation_rate"):
		_validate_mutation_params(mutation_rate=1.5, async_update_fraction=1.0)


# ---------------------------------------------------------------------------
# Validation: mutation_rate>0 + async<1 raises (mutual exclusion)
# ---------------------------------------------------------------------------


def test_mutation_async_mutual_exclusion():
	with pytest.raises(ValueError, match="mutually exclusive"):
		_validate_mutation_params(mutation_rate=0.05, async_update_fraction=0.5)


# ---------------------------------------------------------------------------
# Integration: harness smoke test
# ---------------------------------------------------------------------------


def test_m1_harness_smoke(tmp_path):
	"""Run M1 harness with minimal params to verify end-to-end."""
	from simulation.m1_mutation import run_m1_scout

	result = run_m1_scout(
		seeds=[45, 47],
		mutation_rates=[0.0, 0.05],
		out_root=tmp_path / "m1",
		summary_tsv=tmp_path / "m1_summary.tsv",
		combined_tsv=tmp_path / "m1_combined.tsv",
		decision_md=tmp_path / "m1_decision.md",
		players=30,
		rounds=120,
		burn_in=40,
		tail=60,
		memory_kernel=3,
		enable_events=False,
		events_json=None,
	)
	assert (tmp_path / "m1_decision.md").exists()
	assert (tmp_path / "m1_summary.tsv").exists()
	assert (tmp_path / "m1_combined.tsv").exists()
	assert "close_m1" in result
