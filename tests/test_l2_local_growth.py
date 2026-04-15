"""Tests for L2 local-group growth.

Covers:
  G0-1: local_group_size=0 produces identical results to standard replicator
  G0-2: local_group_size>0 creates inter-group weight divergence
  G0-3: smaller groups → more divergence
  G0-4: deterministic reproducibility with same seed and group_size
  G0-5: partition function correctness (sizes, coverage)
  Validation: negative local_group_size raises
  Validation: mutual exclusion with A1 (async) raises
  Validation: mutual exclusion with M1 (mutation) raises
  Integration: harness smoke test
"""
from __future__ import annotations

import pytest

from simulation.run_simulation import (
	SimConfig,
	_partition_players_into_groups,
	_validate_local_group_params,
	simulate,
)


STRATEGY_SPACE = ["aggressive", "defensive", "balanced"]


# ---------------------------------------------------------------------------
# G0-1: local_group_size=0 identical to standard replicator
# ---------------------------------------------------------------------------


def test_local_group_size_0_identical():
	"""local_group_size=0 must produce identical results to the default (global replicator)."""
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
		local_group_size=0,
	)
	cfg_no_param = SimConfig(
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
	)
	_, rows1 = simulate(cfg_default)
	_, rows2 = simulate(cfg_no_param)

	for i, (r1, r2) in enumerate(zip(rows1, rows2)):
		for s in STRATEGY_SPACE:
			assert r1[f"p_{s}"] == r2[f"p_{s}"], f"Round {i}: p_{s} differs"


# ---------------------------------------------------------------------------
# G0-2: local_group_size>0 creates inter-group weight divergence
# ---------------------------------------------------------------------------


def test_local_groups_create_weight_divergence():
	"""With local_group_size>0, different groups should develop different weights."""
	cfg = SimConfig(
		n_players=30,
		n_rounds=60,
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
		local_group_size=5,
	)
	observed_inter_group_stds = []

	from simulation.run_simulation import _partition_players_into_groups as _ppig
	_groups_holder = [None]
	_first = [True]

	def round_cb(round_index, _cfg, players, _dungeon, _step_records, _row):
		if _first[0]:
			_groups_holder[0] = _ppig(players, 5)
			_first[0] = False
		if round_index < 20:
			return
		groups = _groups_holder[0]
		if groups is None or len(groups) <= 1:
			return
		for s in STRATEGY_SPACE:
			group_means = []
			for grp in groups:
				vals = [float(getattr(pl, "strategy_weights", {}).get(s, 1.0)) for pl in grp]
				group_means.append(sum(vals) / len(vals) if vals else 0.0)
			mean_of_means = sum(group_means) / len(group_means)
			var = sum((gm - mean_of_means) ** 2 for gm in group_means) / len(group_means)
			observed_inter_group_stds.append(var ** 0.5)

	simulate(cfg, round_callback=round_cb)
	avg_std = sum(observed_inter_group_stds) / len(observed_inter_group_stds) if observed_inter_group_stds else 0.0
	assert avg_std > 0.0001, f"local_group_size=5 should create inter-group weight divergence; got {avg_std:.6f}"


# ---------------------------------------------------------------------------
# G0-3: smaller groups → more divergence
# ---------------------------------------------------------------------------


def test_smaller_groups_more_divergence():
	"""Smaller groups should produce more inter-group weight divergence."""

	def measure_inter_group_std(group_size: int) -> float:
		cfg = SimConfig(
			n_players=60,
			n_rounds=60,
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
			local_group_size=group_size,
		)
		from simulation.run_simulation import _partition_players_into_groups as _ppig
		_groups_holder = [None]
		_first = [True]
		stds_accum = []

		def round_cb(round_index, _cfg, players, _dungeon, _step_recs, _row):
			if _first[0]:
				_groups_holder[0] = _ppig(players, group_size)
				_first[0] = False
			if round_index < 30:
				return
			groups = _groups_holder[0]
			if groups is None or len(groups) <= 1:
				return
			for s in STRATEGY_SPACE:
				group_means = []
				for grp in groups:
					vals = [float(getattr(pl, "strategy_weights", {}).get(s, 1.0)) for pl in grp]
					group_means.append(sum(vals) / len(vals) if vals else 0.0)
				mean_of_means = sum(group_means) / len(group_means)
				var = sum((gm - mean_of_means) ** 2 for gm in group_means) / len(group_means)
				stds_accum.append(var ** 0.5)

		simulate(cfg, round_callback=round_cb)
		return sum(stds_accum) / len(stds_accum) if stds_accum else 0.0

	d_small = measure_inter_group_std(5)
	d_large = measure_inter_group_std(30)
	assert d_small > d_large, (
		f"group_size=5 std ({d_small:.6f}) should exceed group_size=30 std ({d_large:.6f})"
	)


# ---------------------------------------------------------------------------
# G0-4: deterministic reproducibility
# ---------------------------------------------------------------------------


def test_deterministic_reproducibility():
	"""Same seed + same local_group_size → identical simulation results."""
	for gs in [5, 10, 20]:
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
			local_group_size=gs,
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
			local_group_size=gs,
		)
		_, rows1 = simulate(cfg1)
		_, rows2 = simulate(cfg2)
		for i, (r1, r2) in enumerate(zip(rows1, rows2)):
			for s in STRATEGY_SPACE:
				assert r1[f"p_{s}"] == r2[f"p_{s}"], (
					f"local_group_size={gs}: round {i} p_{s} differs"
				)


# ---------------------------------------------------------------------------
# G0-5: partition function correctness
# ---------------------------------------------------------------------------


def test_partition_sizes_and_coverage():
	"""Partition should cover all players with groups of approximately the requested size."""

	class _FakePlayer:
		def __init__(self, idx: int):
			self.idx = idx

	players = [_FakePlayer(i) for i in range(100)]

	# group_size=10 → 10 groups of 10
	groups = _partition_players_into_groups(players, 10)
	assert len(groups) == 10
	assert all(len(g) == 10 for g in groups)
	all_indices = [pl.idx for g in groups for pl in g]
	assert sorted(all_indices) == list(range(100))

	# group_size=7 → 14 groups (100 // 7 = 14), remainder 2 → first 2 groups get +1
	groups = _partition_players_into_groups(players, 7)
	assert len(groups) == 14
	sizes = [len(g) for g in groups]
	assert all(s in (7, 8) for s in sizes)
	assert sum(sizes) == 100
	all_indices = [pl.idx for g in groups for pl in g]
	assert sorted(all_indices) == list(range(100))

	# group_size >= n → single group
	groups = _partition_players_into_groups(players, 100)
	assert len(groups) == 1
	assert len(groups[0]) == 100

	# group_size=0 → single group
	groups = _partition_players_into_groups(players, 0)
	assert len(groups) == 1
	assert len(groups[0]) == 100


# ---------------------------------------------------------------------------
# Validation: negative local_group_size raises
# ---------------------------------------------------------------------------


def test_negative_local_group_size_raises():
	with pytest.raises(ValueError, match="local_group_size"):
		_validate_local_group_params(
			local_group_size=-1,
			async_update_fraction=1.0,
			sampling_beta=1.0,
			mutation_rate=0.0,
			tangential_alpha=0.0,
		)


# ---------------------------------------------------------------------------
# Validation: mutual exclusion with A1 (async) raises
# ---------------------------------------------------------------------------


def test_l2_async_mutual_exclusion():
	with pytest.raises(ValueError, match="mutually exclusive"):
		_validate_local_group_params(
			local_group_size=10,
			async_update_fraction=0.5,
			sampling_beta=1.0,
			mutation_rate=0.0,
			tangential_alpha=0.0,
		)


# ---------------------------------------------------------------------------
# Validation: mutual exclusion with M1 (mutation) raises
# ---------------------------------------------------------------------------


def test_l2_mutation_mutual_exclusion():
	with pytest.raises(ValueError, match="mutually exclusive"):
		_validate_local_group_params(
			local_group_size=10,
			async_update_fraction=1.0,
			sampling_beta=1.0,
			mutation_rate=0.05,
			tangential_alpha=0.0,
		)


# ---------------------------------------------------------------------------
# Validation: mutual exclusion with S1 (sampling_beta) raises
# ---------------------------------------------------------------------------


def test_l2_sampling_beta_mutual_exclusion():
	with pytest.raises(ValueError, match="mutually exclusive"):
		_validate_local_group_params(
			local_group_size=10,
			async_update_fraction=1.0,
			sampling_beta=2.0,
			mutation_rate=0.0,
			tangential_alpha=0.0,
		)


# ---------------------------------------------------------------------------
# Validation: mutual exclusion with B1 (tangential) raises
# ---------------------------------------------------------------------------


def test_l2_tangential_mutual_exclusion():
	with pytest.raises(ValueError, match="mutually exclusive"):
		_validate_local_group_params(
			local_group_size=10,
			async_update_fraction=1.0,
			sampling_beta=1.0,
			mutation_rate=0.0,
			tangential_alpha=0.5,
		)


# ---------------------------------------------------------------------------
# Integration: harness smoke test
# ---------------------------------------------------------------------------


def test_l2_harness_smoke(tmp_path):
	"""Run L2 harness with minimal params to verify end-to-end."""
	from simulation.l2_local_growth import run_l2_scout

	result = run_l2_scout(
		seeds=[45, 47],
		local_group_sizes=[0, 10],
		out_root=tmp_path / "l2",
		summary_tsv=tmp_path / "l2_summary.tsv",
		combined_tsv=tmp_path / "l2_combined.tsv",
		decision_md=tmp_path / "l2_decision.md",
		players=30,
		rounds=120,
		burn_in=40,
		tail=60,
		memory_kernel=3,
		enable_events=False,
		events_json=None,
	)
	assert (tmp_path / "l2_decision.md").exists()
	assert (tmp_path / "l2_summary.tsv").exists()
	assert (tmp_path / "l2_combined.tsv").exists()
	assert "close_l2" in result
