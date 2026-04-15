"""Tests for A1 asynchronous replicator (SDD §2.7 A1).

G0 degrade invariants:
  G0-1: fraction=1.0 + same initial weights → bit-identical to replicator_step()
  G0-2: fraction=0.0 → no weights change
  G0-3: weights always positive and mean=1 per player after update
  G0-4: deterministic with same seed + round_index → reproducible
  G0-5: partial update creates weight divergence

Validation: negative/out-of-range fraction raises.
Integration: harness smoke test with small params.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from evolution.replicator_dynamics import async_replicator_step, replicator_step


STRATEGY_SPACE = ["aggressive", "defensive", "balanced"]


def _make_players(n: int = 30, seed: int = 42, *, uniform: bool = True) -> list[object]:
	"""Create a list of minimal player objects for testing."""
	import random

	rng = random.Random(seed)

	class _FakePlayer:
		def __init__(self, last_reward: float, strategy_weights: dict[str, float], last_strategy: str) -> None:
			self.last_reward = last_reward
			self.strategy_weights = dict(strategy_weights)
			self.last_strategy = last_strategy

		def update_weights(self, w: dict[str, float]) -> None:
			self.strategy_weights = dict(w)

	players = []
	for _ in range(n):
		if uniform:
			w = {s: 1.0 for s in STRATEGY_SPACE}
		else:
			w = {s: rng.random() for s in STRATEGY_SPACE}
			total = sum(w.values())
			w = {s: v / total * 3.0 for s, v in w.items()}
		s_choice = rng.choice(STRATEGY_SPACE)
		players.append(_FakePlayer(
			last_reward=rng.gauss(0.5, 0.2),
			strategy_weights=w,
			last_strategy=s_choice,
		))
	return players


# ----------------------------------------------------------------
# G0-1: fraction=1.0 bit-identical to replicator_step()
# ----------------------------------------------------------------

def test_g0_fraction_one_bit_identical() -> None:
	players_sync = _make_players(n=30, seed=42)
	players_async = _make_players(n=30, seed=42)

	expected = replicator_step(
		players_sync,
		STRATEGY_SPACE,
		selection_strength=0.06,
	)
	mean_w, diag = async_replicator_step(
		players_async,
		STRATEGY_SPACE,
		selection_strength=0.06,
		async_update_fraction=1.0,
		seed=99,
		round_index=0,
	)
	# All players should have the new weights
	for pl in players_async:
		for s in STRATEGY_SPACE:
			assert pl.strategy_weights[s] == expected[s], (
				f"Player weight mismatch at {s}: {pl.strategy_weights[s]} != {expected[s]}"
			)
	# Mean weights should match replicator_step output
	for s in STRATEGY_SPACE:
		assert abs(mean_w[s] - expected[s]) < 1e-12, (
			f"Mean weight mismatch at {s}: {mean_w[s]} != {expected[s]}"
		)
	assert diag["n_updated"] == 30
	assert abs(diag["fraction_updated"] - 1.0) < 1e-12


# ----------------------------------------------------------------
# G0-2: fraction=0.0 → no weights change
# ----------------------------------------------------------------

def test_g0_fraction_zero_no_change() -> None:
	players = _make_players(n=20, seed=42, uniform=False)
	original_weights = [dict(pl.strategy_weights) for pl in players]

	mean_w, diag = async_replicator_step(
		players,
		STRATEGY_SPACE,
		selection_strength=0.06,
		async_update_fraction=0.0,
		seed=99,
		round_index=0,
	)
	for i, pl in enumerate(players):
		for s in STRATEGY_SPACE:
			assert pl.strategy_weights[s] == original_weights[i][s], (
				f"Player {i} weight changed at {s}"
			)
	assert diag["n_updated"] == 0
	assert abs(diag["fraction_updated"]) < 1e-12


# ----------------------------------------------------------------
# G0-3: weights always positive and mean=1
# ----------------------------------------------------------------

def test_g0_weights_positive_and_mean_one() -> None:
	for frac in [0.1, 0.3, 0.5, 0.8, 1.0]:
		players = _make_players(n=50, seed=99)
		async_replicator_step(
			players,
			STRATEGY_SPACE,
			selection_strength=0.06,
			async_update_fraction=frac,
			seed=42,
			round_index=7,
		)
		for pl in players:
			for s in STRATEGY_SPACE:
				assert pl.strategy_weights[s] > 0, (
					f"weight {s}={pl.strategy_weights[s]} not positive at frac={frac}"
				)
			mean_w = sum(pl.strategy_weights[s] for s in STRATEGY_SPACE) / len(STRATEGY_SPACE)
			assert abs(mean_w - 1.0) < 1e-10, (
				f"mean weight={mean_w} at frac={frac}"
			)


# ----------------------------------------------------------------
# G0-4: deterministic reproducibility
# ----------------------------------------------------------------

def test_g0_deterministic_reproducibility() -> None:
	for _ in range(3):
		players1 = _make_players(n=30, seed=42)
		players2 = _make_players(n=30, seed=42)
		mw1, d1 = async_replicator_step(
			players1, STRATEGY_SPACE,
			selection_strength=0.06,
			async_update_fraction=0.3,
			seed=77,
			round_index=5,
		)
		mw2, d2 = async_replicator_step(
			players2, STRATEGY_SPACE,
			selection_strength=0.06,
			async_update_fraction=0.3,
			seed=77,
			round_index=5,
		)
		for s in STRATEGY_SPACE:
			assert mw1[s] == mw2[s]
		assert d1["n_updated"] == d2["n_updated"]
		for i in range(len(players1)):
			for s in STRATEGY_SPACE:
				assert players1[i].strategy_weights[s] == players2[i].strategy_weights[s]


# ----------------------------------------------------------------
# G0-5: partial update creates weight divergence
# ----------------------------------------------------------------

def test_g0_partial_update_creates_divergence() -> None:
	players = _make_players(n=50, seed=42)
	_, diag = async_replicator_step(
		players,
		STRATEGY_SPACE,
		selection_strength=0.06,
		async_update_fraction=0.5,
		seed=42,
		round_index=0,
	)
	# Some players should have been updated, others not → weight_dispersion > 0
	assert 0 < diag["n_updated"] < 50
	assert diag["weight_dispersion"] > 0.0


# ----------------------------------------------------------------
# Validation: out-of-range fraction raises
# ----------------------------------------------------------------

def test_negative_fraction_raises() -> None:
	players = _make_players(n=10)
	with pytest.raises(ValueError, match="async_update_fraction"):
		async_replicator_step(
			players, STRATEGY_SPACE, async_update_fraction=-0.1,
		)


def test_fraction_above_one_raises() -> None:
	players = _make_players(n=10)
	with pytest.raises(ValueError, match="async_update_fraction"):
		async_replicator_step(
			players, STRATEGY_SPACE, async_update_fraction=1.5,
		)


# ----------------------------------------------------------------
# Integration: A1 harness smoke test
# ----------------------------------------------------------------

def test_a1_harness_smoke(tmp_path: Path) -> None:
	from simulation.a1_async import run_a1_scout

	out_root = tmp_path / "a1"
	summary_tsv = tmp_path / "a1_summary.tsv"
	combined_tsv = tmp_path / "a1_combined.tsv"
	decision_md = tmp_path / "a1_decision.md"

	result = run_a1_scout(
		seeds=[45, 47],
		async_fractions=[0.5, 1.0],
		out_root=out_root,
		summary_tsv=summary_tsv,
		combined_tsv=combined_tsv,
		decision_md=decision_md,
		players=30,
		rounds=120,
		burn_in=30,
		tail=60,
		memory_kernel=3,
		enable_events=False,
	)

	assert Path(result["summary_tsv"]).exists()
	assert Path(result["combined_tsv"]).exists()
	assert Path(result["decision_md"]).exists()

	with summary_tsv.open(newline="", encoding="utf-8") as handle:
		summary_rows = list(csv.DictReader(handle, delimiter="\t"))
		assert summary_rows
		for row in summary_rows:
			assert row["mean_weight_dispersion"] != ""
			assert row["mean_fraction_updated"] != ""

	with combined_tsv.open(newline="", encoding="utf-8") as handle:
		combined_rows = list(csv.DictReader(handle, delimiter="\t"))
		# Should have 2 conditions (frac=0.5 and frac=1.0)
		assert len(combined_rows) == 2
		ctrl = [r for r in combined_rows if r["is_control"] == "yes"]
		assert len(ctrl) == 1
		assert ctrl[0]["verdict"] == "control"
