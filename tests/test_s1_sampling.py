"""Tests for S1 sampling sharpness (power-law β).

Covers:
  G0-1: β=1.0 produces identical results to standard w-proportional
  G0-2: β=0 produces uniform random (all strategies equally likely)
  G0-3: β>1 sharpens distribution (dominant strategy more concentrated)
  G0-4: β<1 flattens distribution (more exploratory)
  G0-5: deterministic reproducibility with same seed
  Validation: negative β raises, async+beta combination raises
  Integration: harness smoke test
"""
from __future__ import annotations

import random
from math import exp, log

import pytest

from players.base_player import BasePlayer


# ---------------------------------------------------------------------------
# G0-1: β=1.0 identical to current w-proportional
# ---------------------------------------------------------------------------


def test_beta_1_identical_to_default():
	"""With β=1.0, choose_strategy must produce the exact same sequence as default."""
	strategy_space = ["aggressive", "defensive", "balanced"]
	weights = {"aggressive": 1.12, "defensive": 0.88, "balanced": 1.0}

	results_default = []
	rng1 = random.Random(42)
	p1 = BasePlayer(strategy_space, rng=rng1)
	p1.update_weights(weights)
	for _ in range(200):
		results_default.append(p1.choose_strategy())

	results_beta1 = []
	rng2 = random.Random(42)
	p2 = BasePlayer(strategy_space, rng=rng2)
	p2.update_weights(weights)
	p2.sampling_beta = 1.0
	for _ in range(200):
		results_beta1.append(p2.choose_strategy())

	assert results_default == results_beta1, "β=1.0 must be bit-identical to default"


# ---------------------------------------------------------------------------
# G0-2: β=0 → uniform random
# ---------------------------------------------------------------------------


def test_beta_0_uniform():
	"""With β=0, all effective weights become 1.0 → uniform distribution."""
	strategy_space = ["aggressive", "defensive", "balanced"]
	weights = {"aggressive": 5.0, "defensive": 0.1, "balanced": 1.0}

	rng = random.Random(123)
	p = BasePlayer(strategy_space, rng=rng)
	p.update_weights(weights)
	p.sampling_beta = 0.0

	counts = {"aggressive": 0, "defensive": 0, "balanced": 0}
	n = 9000
	for _ in range(n):
		s = p.choose_strategy()
		counts[s] += 1

	# With uniform sampling, each strategy should be ~1/3
	for s in strategy_space:
		share = counts[s] / n
		assert abs(share - 1.0 / 3.0) < 0.05, f"β=0: {s} share={share:.3f} should be ~0.333"


# ---------------------------------------------------------------------------
# G0-3: β>1 sharpens → dominant strategy gets higher share
# ---------------------------------------------------------------------------


def test_high_beta_sharpens():
	"""β=10 should make the dominant strategy much more likely than β=1."""
	strategy_space = ["aggressive", "defensive", "balanced"]
	weights = {"aggressive": 1.3, "defensive": 0.85, "balanced": 0.85}

	n = 5000

	# β=1 (default)
	rng1 = random.Random(77)
	p1 = BasePlayer(strategy_space, rng=rng1)
	p1.update_weights(weights)
	count_agg_1 = sum(1 for _ in range(n) if p1.choose_strategy() == "aggressive")
	share_1 = count_agg_1 / n

	# β=10 (sharp)
	rng2 = random.Random(77)
	p2 = BasePlayer(strategy_space, rng=rng2)
	p2.update_weights(weights)
	p2.sampling_beta = 10.0
	count_agg_10 = sum(1 for _ in range(n) if p2.choose_strategy() == "aggressive")
	share_10 = count_agg_10 / n

	assert share_10 > share_1 + 0.05, (
		f"β=10 should sharpen: aggressive share {share_10:.3f} should be > {share_1:.3f}"
	)


# ---------------------------------------------------------------------------
# G0-4: β<1 flattens → dominant strategy gets lower share
# ---------------------------------------------------------------------------


def test_low_beta_flattens():
	"""β=0.5 should make the dominant strategy less dominant than β=1."""
	strategy_space = ["aggressive", "defensive", "balanced"]
	weights = {"aggressive": 2.0, "defensive": 0.5, "balanced": 0.5}

	n = 5000

	# β=1
	rng1 = random.Random(99)
	p1 = BasePlayer(strategy_space, rng=rng1)
	p1.update_weights(weights)
	count_agg_1 = sum(1 for _ in range(n) if p1.choose_strategy() == "aggressive")
	share_1 = count_agg_1 / n

	# β=0.5
	rng2 = random.Random(99)
	p2 = BasePlayer(strategy_space, rng=rng2)
	p2.update_weights(weights)
	p2.sampling_beta = 0.5
	count_agg_05 = sum(1 for _ in range(n) if p2.choose_strategy() == "aggressive")
	share_05 = count_agg_05 / n

	assert share_05 < share_1 - 0.03, (
		f"β=0.5 should flatten: aggressive share {share_05:.3f} should be < {share_1:.3f}"
	)


# ---------------------------------------------------------------------------
# G0-5: deterministic reproducibility
# ---------------------------------------------------------------------------


def test_deterministic_reproducibility():
	"""Same seed + same β → identical strategy sequence."""
	strategy_space = ["aggressive", "defensive", "balanced"]
	weights = {"aggressive": 1.2, "defensive": 0.9, "balanced": 0.9}

	for beta in [0.5, 2.0, 5.0, 50.0]:
		results_a = []
		rng_a = random.Random(55)
		pa = BasePlayer(strategy_space, rng=rng_a)
		pa.update_weights(weights)
		pa.sampling_beta = beta
		for _ in range(100):
			results_a.append(pa.choose_strategy())

		results_b = []
		rng_b = random.Random(55)
		pb = BasePlayer(strategy_space, rng=rng_b)
		pb.update_weights(weights)
		pb.sampling_beta = beta
		for _ in range(100):
			results_b.append(pb.choose_strategy())

		assert results_a == results_b, f"β={beta}: reproducibility failed"


# ---------------------------------------------------------------------------
# Validation: negative β raises
# ---------------------------------------------------------------------------


def test_negative_beta_raises():
	from simulation.run_simulation import _validate_sampling_beta_params

	with pytest.raises(ValueError, match="sampling_beta"):
		_validate_sampling_beta_params(sampling_beta=-0.5, async_update_fraction=1.0)


# ---------------------------------------------------------------------------
# Validation: β ≠ 1 + async < 1 raises (mutual exclusion)
# ---------------------------------------------------------------------------


def test_beta_async_mutual_exclusion():
	from simulation.run_simulation import _validate_sampling_beta_params

	with pytest.raises(ValueError, match="mutually exclusive"):
		_validate_sampling_beta_params(sampling_beta=2.0, async_update_fraction=0.5)


# ---------------------------------------------------------------------------
# Integration: harness smoke test
# ---------------------------------------------------------------------------


def test_s1_harness_smoke(tmp_path):
	"""Run S1 harness with minimal params to verify end-to-end."""
	from simulation.s1_sampling_sharpness import run_s1_scout

	result = run_s1_scout(
		seeds=[45, 47],
		betas=[1.0, 2.0],
		out_root=tmp_path / "s1",
		summary_tsv=tmp_path / "s1_summary.tsv",
		combined_tsv=tmp_path / "s1_combined.tsv",
		decision_md=tmp_path / "s1_decision.md",
		players=30,
		rounds=120,
		burn_in=40,
		tail=60,
		memory_kernel=3,
		enable_events=False,
		events_json=None,
	)
	assert (tmp_path / "s1_decision.md").exists()
	assert (tmp_path / "s1_summary.tsv").exists()
	assert (tmp_path / "s1_combined.tsv").exists()
	assert "close_s1" in result
