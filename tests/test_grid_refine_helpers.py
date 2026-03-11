from __future__ import annotations

from simulation.grid_refine import build_refine_points, select_candidates


def test_build_refine_points_dedupes_overlap() -> None:
	cands = [(0.10, 0.20), (0.1000000001, 0.2000000001)]
	pts = build_refine_points(cands, span_a=0.0, span_b=0.0, step=0.01, dedupe_decimals=6)
	assert pts == [(0.10, 0.20)]


def test_select_candidates_by_score_threshold() -> None:
	rows = [
		{
			"a": 0.1,
			"b": 0.2,
			"mean_rot_score_max": 0.61,
			"mean_rot_turn_strength_max": 0.01,
			"mean_max_amp": 0.02,
			"mean_max_corr": 0.3,
		},
		{
			"a": 0.2,
			"b": 0.3,
			"mean_rot_score_max": 0.59,
			"mean_rot_turn_strength_max": 0.99,
			"mean_max_amp": 0.02,
			"mean_max_corr": 0.3,
		},
		{
			"a": 0.3,
			"b": 0.4,
			"mean_rot_score_max": 0.70,
			"mean_rot_turn_strength_max": 0.0,
			"mean_max_amp": 0.02,
			"mean_max_corr": 0.3,
		},
	]
	c1 = select_candidates(rows, rot_score_threshold=0.60, min_turn_strength=0.0)
	assert (0.1, 0.2) in c1
	assert (0.3, 0.4) in c1
	assert (0.2, 0.3) not in c1

	c2 = select_candidates(rows, rot_score_threshold=0.60, min_turn_strength=0.005)
	assert c2 == [(0.1, 0.2)]

	c3 = select_candidates(rows, rot_score_threshold=0.60, min_turn_strength=0.0, min_max_corr=0.31)
	assert c3 == []
