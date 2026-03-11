from __future__ import annotations

import math

from analysis.decay_rate import estimate_decay_gamma


def _make_series(gamma: float, *, n: int = 400, period: int = 20) -> dict[str, list[float]]:
	"""Synthetic rotating trajectory around center with exponential envelope."""
	c = 1.0 / 3.0
	u: list[float] = []
	v: list[float] = []
	for t in range(int(n)):
		A = math.exp(float(gamma) * float(t))
		theta = 2.0 * math.pi * float(t) / float(period)
		u.append(c + 0.05 * A * math.cos(theta))
		v.append(c + 0.05 * A * math.sin(theta))
	return {"aggressive": u, "defensive": v, "balanced": [c] * int(n)}


def test_estimate_decay_gamma_recovers_slope_sign_and_scale() -> None:
	# Choose a gentle decay so it doesn't hit machine epsilon too fast.
	true_gamma = -0.002
	series_map = _make_series(true_gamma, n=600, period=24)
	fit = estimate_decay_gamma(series_map, series_kind="p", metric="linf", min_peaks=10, min_peak_distance=8)
	assert fit is not None
	# Sign should match; magnitude should be close.
	assert fit.gamma < 0
	assert abs(fit.gamma - true_gamma) < 5e-4
	assert fit.n_peaks >= 10
	assert fit.r2 > 0.95
