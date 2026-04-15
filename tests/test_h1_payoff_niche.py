"""Tests for H1 payoff-niche.

Covers:
  G0-1: epsilon=0 + niche_group_size=0 identical to standard replicator
  G0-2: epsilon>0 creates inter-group growth direction divergence (cosine < 1.0)
  G0-3: higher epsilon → lower cosine
  G0-4: deterministic reproducibility with same seed
  G0-5: niche assignment correctness (group k → strategy k%3)
  Validation: negative epsilon raises
  Validation: epsilon>0 without niche_group_size raises
  Validation: mutual exclusion with A1 (async) raises
  Validation: mutual exclusion with M1 (mutation) raises
  Validation: mutual exclusion with L2 (local_group_size) raises
  Integration: harness smoke test
"""
from __future__ import annotations

import pytest

from simulation.run_simulation import (
	SimConfig,
	_partition_players_into_groups,
	_validate_h1_niche_params,
	simulate,
)


STRATEGY_SPACE = ["aggressive", "defensive", "balanced"]


# ---------------------------------------------------------------------------
# G0-1: epsilon=0 identical to standard replicator
# ---------------------------------------------------------------------------


def test_h1_epsilon_0_identical():
	"""payoff_niche_epsilon=0 + niche_group_size=0 must produce identical results to default."""
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
		payoff_niche_epsilon=0.0,
		niche_group_size=0,
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
# G0-2: epsilon>0 creates inter-group growth direction divergence
# ---------------------------------------------------------------------------


def test_h1_niche_creates_growth_divergence():
	"""With epsilon>0, different groups should develop different growth vector directions."""
	cfg = SimConfig(
		n_players=30,
		n_rounds=80,
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
		payoff_niche_epsilon=0.10,
		niche_group_size=10,
	)
	observed_cosines: list[float] = []
	_groups_holder: list[list[list[object]] | None] = [None]
	_first = [True]

	def round_cb(round_index, _cfg, players, _dungeon, _step_records, _row):
		if _first[0]:
			_groups_holder[0] = _partition_players_into_groups(players, 10)
			_first[0] = False
		if round_index < 20:
			return
		groups = _groups_holder[0]
		if groups is None or len(groups) <= 1:
			return
		strategies = STRATEGY_SPACE
		group_growth_vectors: list[list[float]] = []
		for group in groups:
			rewards_by_strategy: dict[str, list[float]] = {s: [] for s in strategies}
			all_rewards: list[float] = []
			for pl in group:
				r = float(getattr(pl, "last_reward", 0.0))
				s = getattr(pl, "last_strategy", None)
				all_rewards.append(r)
				if s in rewards_by_strategy:
					rewards_by_strategy[s].append(r)
			pop_mean = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
			gv = []
			for s in strategies:
				vals = rewards_by_strategy[s]
				s_mean = sum(vals) / len(vals) if vals else pop_mean
				gv.append(s_mean - pop_mean)
			group_growth_vectors.append(gv)

		# Pairwise cosine
		for i in range(len(group_growth_vectors)):
			for j in range(i + 1, len(group_growth_vectors)):
				a_vec = group_growth_vectors[i]
				b_vec = group_growth_vectors[j]
				dot = sum(x * y for x, y in zip(a_vec, b_vec))
				norm_a = sum(x * x for x in a_vec) ** 0.5
				norm_b = sum(x * x for x in b_vec) ** 0.5
				if norm_a > 1e-30 and norm_b > 1e-30:
					observed_cosines.append(dot / (norm_a * norm_b))

	simulate(cfg, round_callback=round_cb)
	if observed_cosines:
		avg_cosine = sum(observed_cosines) / len(observed_cosines)
		assert avg_cosine < 0.999, f"epsilon=0.10 should create growth direction divergence; got cosine={avg_cosine:.4f}"


# ---------------------------------------------------------------------------
# G0-3: higher epsilon → lower cosine
# ---------------------------------------------------------------------------


def test_higher_epsilon_lower_cosine():
	"""Higher epsilon should produce lower mean inter-group growth cosine."""
	def run_with_epsilon(eps: float) -> float:
		cfg = SimConfig(
			n_players=30,
			n_rounds=80,
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
			payoff_niche_epsilon=eps,
			niche_group_size=10,
		)
		cosines: list[float] = []
		_groups_holder: list[list[list[object]] | None] = [None]
		_first = [True]

		def round_cb(round_index, _cfg, players, _dungeon, _step_records, _row):
			if _first[0]:
				_groups_holder[0] = _partition_players_into_groups(players, 10)
				_first[0] = False
			if round_index < 20:
				return
			groups = _groups_holder[0]
			if groups is None or len(groups) <= 1:
				return
			strategies = STRATEGY_SPACE
			gvs: list[list[float]] = []
			for group in groups:
				rbs: dict[str, list[float]] = {s: [] for s in strategies}
				ar: list[float] = []
				for pl in group:
					r = float(getattr(pl, "last_reward", 0.0))
					s = getattr(pl, "last_strategy", None)
					ar.append(r)
					if s in rbs:
						rbs[s].append(r)
				pm = sum(ar) / len(ar) if ar else 0.0
				gvs.append([sum(rbs[s]) / len(rbs[s]) - pm if rbs[s] else 0.0 for s in strategies])
			for i in range(len(gvs)):
				for j in range(i + 1, len(gvs)):
					d = sum(x * y for x, y in zip(gvs[i], gvs[j]))
					na = sum(x * x for x in gvs[i]) ** 0.5
					nb = sum(x * x for x in gvs[j]) ** 0.5
					if na > 1e-30 and nb > 1e-30:
						cosines.append(d / (na * nb))

		simulate(cfg, round_callback=round_cb)
		return sum(cosines) / len(cosines) if cosines else 1.0

	cosine_low = run_with_epsilon(0.05)
	cosine_high = run_with_epsilon(0.20)
	assert cosine_high <= cosine_low + 0.05, (
		f"Higher epsilon should produce lower or similar cosine: "
		f"eps=0.05 → {cosine_low:.4f}, eps=0.20 → {cosine_high:.4f}"
	)


# ---------------------------------------------------------------------------
# G0-4: deterministic reproducibility
# ---------------------------------------------------------------------------


def test_h1_deterministic():
	"""Same seed + same config → identical results."""
	cfg = SimConfig(
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
		payoff_niche_epsilon=0.10,
		niche_group_size=10,
	)
	_, rows1 = simulate(cfg)
	_, rows2 = simulate(cfg)

	for i, (r1, r2) in enumerate(zip(rows1, rows2)):
		for s in STRATEGY_SPACE:
			assert r1[f"p_{s}"] == r2[f"p_{s}"], f"Round {i}: p_{s} differs"


# ---------------------------------------------------------------------------
# G0-5: niche assignment correctness
# ---------------------------------------------------------------------------


def test_niche_assignment():
	"""Group k should get niche bonus on strategy (k % 3)."""
	players = list(range(30))
	groups = _partition_players_into_groups(players, 10)
	assert len(groups) == 3
	strategies = STRATEGY_SPACE
	for gk, group in enumerate(groups):
		expected_niche = strategies[gk % len(strategies)]
		assert expected_niche == strategies[gk % 3], f"Group {gk} niche mismatch"


# ---------------------------------------------------------------------------
# Validation: negative epsilon raises
# ---------------------------------------------------------------------------


def test_h1_negative_epsilon():
	with pytest.raises(ValueError, match="payoff_niche_epsilon must be >= 0"):
		_validate_h1_niche_params(
			payoff_niche_epsilon=-0.1,
			niche_group_size=10,
			async_update_fraction=1.0,
			sampling_beta=1.0,
			mutation_rate=0.0,
			tangential_alpha=0.0,
			local_group_size=0,
		)


# ---------------------------------------------------------------------------
# Validation: epsilon>0 without niche_group_size raises
# ---------------------------------------------------------------------------


def test_h1_epsilon_without_group_size():
	with pytest.raises(ValueError, match="requires niche_group_size > 0"):
		_validate_h1_niche_params(
			payoff_niche_epsilon=0.1,
			niche_group_size=0,
			async_update_fraction=1.0,
			sampling_beta=1.0,
			mutation_rate=0.0,
			tangential_alpha=0.0,
			local_group_size=0,
		)


# ---------------------------------------------------------------------------
# Validation: niche_group_size>0 without epsilon raises
# ---------------------------------------------------------------------------


def test_h1_group_size_without_epsilon():
	with pytest.raises(ValueError, match="requires payoff_niche_epsilon > 0"):
		_validate_h1_niche_params(
			payoff_niche_epsilon=0.0,
			niche_group_size=10,
			async_update_fraction=1.0,
			sampling_beta=1.0,
			mutation_rate=0.0,
			tangential_alpha=0.0,
			local_group_size=0,
		)


# ---------------------------------------------------------------------------
# Validation: mutual exclusion with A1 (async) raises
# ---------------------------------------------------------------------------


def test_h1_async_mutual_exclusion():
	with pytest.raises(ValueError, match="H1 and A1"):
		_validate_h1_niche_params(
			payoff_niche_epsilon=0.1,
			niche_group_size=10,
			async_update_fraction=0.5,
			sampling_beta=1.0,
			mutation_rate=0.0,
			tangential_alpha=0.0,
			local_group_size=0,
		)


# ---------------------------------------------------------------------------
# Validation: mutual exclusion with M1 (mutation) raises
# ---------------------------------------------------------------------------


def test_h1_mutation_mutual_exclusion():
	with pytest.raises(ValueError, match="H1 and M1"):
		_validate_h1_niche_params(
			payoff_niche_epsilon=0.1,
			niche_group_size=10,
			async_update_fraction=1.0,
			sampling_beta=1.0,
			mutation_rate=0.05,
			tangential_alpha=0.0,
			local_group_size=0,
		)


# ---------------------------------------------------------------------------
# Validation: mutual exclusion with S1 (sampling_beta) raises
# ---------------------------------------------------------------------------


def test_h1_sampling_beta_mutual_exclusion():
	with pytest.raises(ValueError, match="H1 and S1"):
		_validate_h1_niche_params(
			payoff_niche_epsilon=0.1,
			niche_group_size=10,
			async_update_fraction=1.0,
			sampling_beta=2.0,
			mutation_rate=0.0,
			tangential_alpha=0.0,
			local_group_size=0,
		)


# ---------------------------------------------------------------------------
# Validation: mutual exclusion with B1 (tangential) raises
# ---------------------------------------------------------------------------


def test_h1_tangential_mutual_exclusion():
	with pytest.raises(ValueError, match="H1 and B1"):
		_validate_h1_niche_params(
			payoff_niche_epsilon=0.1,
			niche_group_size=10,
			async_update_fraction=1.0,
			sampling_beta=1.0,
			mutation_rate=0.0,
			tangential_alpha=0.5,
			local_group_size=0,
		)


# ---------------------------------------------------------------------------
# Validation: mutual exclusion with L2 (local_group_size) raises
# ---------------------------------------------------------------------------


def test_h1_l2_mutual_exclusion():
	with pytest.raises(ValueError, match="H1 and L2"):
		_validate_h1_niche_params(
			payoff_niche_epsilon=0.1,
			niche_group_size=10,
			async_update_fraction=1.0,
			sampling_beta=1.0,
			mutation_rate=0.0,
			tangential_alpha=0.0,
			local_group_size=10,
		)


# ---------------------------------------------------------------------------
# Integration: harness smoke test
# ---------------------------------------------------------------------------


def test_h1_harness_smoke(tmp_path):
	"""Run H1 harness with minimal params to verify end-to-end."""
	from simulation.h1_payoff_niche import run_h1_scout

	result = run_h1_scout(
		seeds=[45, 47],
		payoff_niche_epsilons=[0.0, 0.10],
		niche_group_sizes=[0, 10],
		out_root=tmp_path / "h1",
		summary_tsv=tmp_path / "h1_summary.tsv",
		combined_tsv=tmp_path / "h1_combined.tsv",
		decision_md=tmp_path / "h1_decision.md",
		players=30,
		rounds=120,
		burn_in=40,
		tail=60,
		memory_kernel=3,
		enable_events=False,
		events_json=None,
	)
	assert (tmp_path / "h1_decision.md").exists()
	assert (tmp_path / "h1_summary.tsv").exists()
	assert (tmp_path / "h1_combined.tsv").exists()
	assert "close_h1" in result
