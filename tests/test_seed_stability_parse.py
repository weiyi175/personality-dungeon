import pytest

from simulation.seed_stability import _parse_float_grid, _parse_seeds


def test_parse_seeds_range_inclusive():
	assert _parse_seeds("0:2") == [0, 1, 2]
	assert _parse_seeds("2:0:-1") == [2, 1, 0]


def test_parse_float_grid_list_and_range():
	assert _parse_float_grid("0.5,1.0") == [0.5, 1.0]
	vals = _parse_float_grid("0.5:1.0:0.25")
	assert vals == pytest.approx([0.5, 0.75, 1.0])


def test_parse_float_grid_default_step():
	vals = _parse_float_grid("0.0:0.2")
	# default step=0.1 => 0.0,0.1,0.2
	assert vals == pytest.approx([0.0, 0.1, 0.2])
