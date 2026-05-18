from __future__ import annotations

import math

import pytest

from scripts.frontier_locator import (
	assign_labels,
	calculate_fitness,
	crowding_distance,
	dedupe_candidates,
	non_dominated_sort,
	select_diverse_candidates,
)


def test_select_diverse_candidates_prefers_crowding_extremes_and_dedupes() -> None:
	candidates = [
		{
			"trial_id": 1,
			"seed": 45,
			"score": {"center_drift": 0.10, "rotational_energy": 10.0, "min_x": 1.0},
			"traits_vector": {"randomness": -0.10},
		},
		{
			"trial_id": 2,
			"seed": 45,
			"score": {"center_drift": 0.20, "rotational_energy": 8.0, "min_x": 4.0},
			"traits_vector": {"randomness": -0.20},
		},
		{
			"trial_id": 3,
			"seed": 45,
			"score": {"center_drift": 0.30, "rotational_energy": 6.0, "min_x": 7.0},
			"traits_vector": {"randomness": -0.30},
		},
		{
			"trial_id": 4,
			"seed": 45,
			"score": {"center_drift": 0.40, "rotational_energy": 4.0, "min_x": 10.0},
			"traits_vector": {"randomness": -0.40},
		},
		{
			"trial_id": 5,
			"seed": 45,
			"score": {"center_drift": 0.40, "rotational_energy": 4.0, "min_x": 10.0},
			"traits_vector": {"randomness": -0.40},
		},
	]
	deduped = dedupe_candidates(candidates, trait_keys=["randomness"], precision=6)
	assert len(deduped) == 4

	selected = select_diverse_candidates(deduped, limit=2)
	trial_ids = {cand["trial_id"] for cand in selected}
	assert trial_ids == {1, 4}


def test_non_dominated_sort_and_crowding_distance_basic() -> None:
	items = [
		{"objectives": (0.10, -10.0, -1.0)},
		{"objectives": (0.20, -8.0, -4.0)},
		{"objectives": (0.30, -6.0, -7.0)},
		{"objectives": (0.40, -4.0, -10.0)},
	]
	fronts = non_dominated_sort(items)
	assert fronts[0] == [0, 1, 2, 3]
	dmap = crowding_distance(fronts[0], items)
	assert math.isinf(dmap[0])
	assert math.isinf(dmap[3])
	assert dmap[1] > 0.0
	assert dmap[2] > 0.0


def test_calculate_fitness_applies_log_k_correction() -> None:
	metrics = {
		"sweetspot_width_axis": 0.20,
		"drift_p95": 0.10,
		"resonance_mean": 0.40,
		"slr": 0.25,
		"peak_re": 8.0,
	}
	base = calculate_fitness(metrics, fitness_k=math.e, peak_reference=8.0, scan_span=1.0)
	boosted = calculate_fitness(metrics, fitness_k=math.e**2, peak_reference=8.0, scan_span=1.0)
	assert base == pytest.approx(
		((0.20 / 1.0) + (1 / 1.10) + 0.40 + (1 / 1.25) + 1.0) / 5.0,
	)
	assert boosted == pytest.approx(base * 2.0)


def test_assign_labels_marks_survivor_pulsar_and_alpha() -> None:
	thresholds = {
		"width_p90": 0.20,
		"width_p50": 0.10,
		"peak_p90": 8000.0,
		"resonance_p50": 0.30,
		"drift_p50": 0.05,
		"slr_p50": 0.25,
	}
	metrics = {
		"sweetspot_width_axis": 0.25,
		"drift_p95": 0.01,
		"peak_re": 9000.0,
		"resonance_mean": 0.60,
	}
	labels = assign_labels(metrics, thresholds)
	assert labels == ["[The Survivor]", "[The Pulsar]", "[Alpha Seed]"]
