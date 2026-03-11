import math

from analysis.cycle_metrics import normalize_simplex_timeseries
from analysis.simplex import normalize_simplex3


def test_normalize_simplex3_clips_negative_and_renormalizes():
	res = normalize_simplex3((-1e-12, 0.3, 0.5), clip_negative=True, on_invalid="zeros")
	x1, x2, x3 = res.x
	assert res.clipped_negative is True
	assert x1 >= 0.0
	assert x2 >= 0.0
	assert x3 >= 0.0
	assert math.isclose(x1 + x2 + x3, 1.0, rel_tol=0.0, abs_tol=1e-12)
	assert math.isclose(x1, 0.0, rel_tol=0.0, abs_tol=1e-12)
	assert math.isclose(x2, 0.375, rel_tol=0.0, abs_tol=1e-12)
	assert math.isclose(x3, 0.625, rel_tol=0.0, abs_tol=1e-12)


def test_normalize_simplex3_rejects_nonfinite_to_zeros():
	res = normalize_simplex3((math.nan, 1.0, 1.0), clip_negative=True, on_invalid="zeros")
	assert res.x == (0.0, 0.0, 0.0)


def test_normalize_simplex3_uniform_fallback_on_nonpositive_sum():
	res = normalize_simplex3((0.0, 0.0, 0.0), clip_negative=True, on_invalid="uniform")
	u = 1.0 / 3.0
	assert math.isclose(res.x[0], u, rel_tol=0.0, abs_tol=1e-15)
	assert math.isclose(res.x[1], u, rel_tol=0.0, abs_tol=1e-15)
	assert math.isclose(res.x[2], u, rel_tol=0.0, abs_tol=1e-15)


def test_normalize_simplex_timeseries_clips_negative_per_timestep():
	series = {
		"aggressive": [-1e-12, 0.2],
		"defensive": [0.3, 0.3],
		"balanced": [0.5, 0.5],
	}
	out = normalize_simplex_timeseries(series)
	assert set(out.keys()) == {"aggressive", "defensive", "balanced"}
	assert len(out["aggressive"]) == 2
	assert math.isclose(out["aggressive"][0], 0.0, rel_tol=0.0, abs_tol=1e-12)
	assert math.isclose(out["defensive"][0] + out["balanced"][0], 1.0, rel_tol=0.0, abs_tol=1e-12)
	assert math.isclose(out["aggressive"][1] + out["defensive"][1] + out["balanced"][1], 1.0, rel_tol=0.0, abs_tol=1e-12)
