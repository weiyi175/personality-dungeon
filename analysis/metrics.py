"""Analysis metrics.

This module keeps the legacy player-summary helpers for simulation while
adding frontier-positioning metrics that operate on ndarrays only.
"""

import numpy as np
from scipy import signal


_EPS = 1e-12
_SWEETSPOT_SMOOTH_WINDOW = 3
_Z_DRIFT_WINDOW = 50


def _sanitize_1d(values):
	array = np.asarray(values, dtype=float).reshape(-1)
	return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _sanitize_2d(values, *, n_cols):
	array = np.asarray(values, dtype=float)
	if array.ndim != 2 or array.shape[1] != n_cols:
		raise ValueError(f"expected a 2D array with shape (T, {n_cols})")
	return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _trailing_moving_average_3(values):
	if values.size == 0:
		return values.astype(float, copy=True)
	if values.size == 1:
		return values.astype(float, copy=True)
	if _SWEETSPOT_SMOOTH_WINDOW != 3:
		raise ValueError("sweetspot smoothing window must remain 3")
	window = _SWEETSPOT_SMOOTH_WINDOW
	cumsum = np.cumsum(values, dtype=float)
	smoothed = np.empty(values.size, dtype=float)
	if values.size < window:
		counts = np.arange(1, values.size + 1, dtype=float)
		smoothed[:] = cumsum / counts
		return smoothed
	smoothed[: window - 1] = cumsum[: window - 1] / np.arange(1, window, dtype=float)
	previous = np.concatenate(([0.0], cumsum[:-window]))
	smoothed[window - 1 :] = (cumsum[window - 1 :] - previous) / float(window)
	return smoothed


def _segment_corrcoef(a, b):
	if a.size < 2 or b.size < 2:
		return 0.0
	a = np.asarray(a, dtype=float) - float(np.mean(a))
	b = np.asarray(b, dtype=float) - float(np.mean(b))
	denom = float(np.linalg.norm(a) * np.linalg.norm(b))
	if denom <= _EPS:
		return 0.0
	return float(np.dot(a, b) / denom)


def calculate_peak_relative_width(re_array, peak_fraction=0.95, gap_tolerance=1) -> int:
	"""Return the widest RE interval that stays above a peak-relative threshold.

	The RE trace is first smoothed by a 3-point trailing moving average, then
	thresholded with `RE >= peak_fraction * re_peak` where `re_peak` is the raw
	peak of the trace. Intervals may contain up to `gap_tolerance` consecutive
	sub-threshold steps inside the span.
	"""
	peak_fraction = float(peak_fraction)
	if not 0.0 < peak_fraction <= 1.0:
		raise ValueError("peak_fraction must be in (0, 1]")
	if gap_tolerance < 0:
		raise ValueError("gap_tolerance must be >= 0")
	values = _sanitize_1d(re_array)
	if values.size == 0:
		return 0

	peak = float(np.max(values))
	if peak <= 0.0:
		return 0

	smoothed = _trailing_moving_average_3(values)
	above = smoothed >= (peak_fraction * peak)

	best_width = 0
	start_idx = None
	last_true_idx = None
	gap_len = 0

	for idx, is_above in enumerate(above):
		if bool(is_above):
			if start_idx is None:
				start_idx = idx
			last_true_idx = idx
			gap_len = 0
			continue

		if start_idx is None:
			continue

		gap_len += 1
		if gap_len > gap_tolerance:
			best_width = max(best_width, last_true_idx - start_idx + 1)
			start_idx = None
			last_true_idx = None
			gap_len = 0

	if start_idx is not None and last_true_idx is not None:
		best_width = max(best_width, last_true_idx - start_idx + 1)

	return int(best_width)


def calculate_sweetspot_width(re_array, threshold=0.95, gap_tolerance=1) -> int:
	"""Legacy alias for calculate_peak_relative_width.

	The `threshold` argument is preserved for backward compatibility and ignored.
	"""
	return calculate_peak_relative_width(re_array, peak_fraction=0.95, gap_tolerance=gap_tolerance)


def calculate_z_drift(z_history):
	"""Return the median and p95 of a 50-step sliding sigma over a Z trajectory.

	The per-window sigma is the mean of the three component-wise standard
	deviations, which isolates temporal drift from static axis offsets.
	"""
	z = _sanitize_2d(z_history, n_cols=3)
	total_steps = int(z.shape[0])
	if total_steps == 0:
		return 0.0, 0.0

	window_n = min(_Z_DRIFT_WINDOW, total_steps)
	window_sigmas = np.empty(total_steps - window_n + 1, dtype=float)

	for start in range(window_sigmas.size):
		window = z[start:start + window_n, :]
		per_axis_sigma = np.std(window, axis=0, ddof=0)
		window_sigmas[start] = float(np.mean(per_axis_sigma))

	return float(np.median(window_sigmas)), float(np.quantile(window_sigmas, 0.95))


def calculate_resonance_score(re_array, z_history, max_lag=20):
	"""Return the max absolute lagged correlation between RE and each Z component.

	The RE series and each Z component are detrended with
	scipy.signal.detrend before the lag scan.
	"""
	if max_lag < 0:
		raise ValueError("max_lag must be >= 0")
	re = _sanitize_1d(re_array)
	z = _sanitize_2d(z_history, n_cols=3)
	if re.size == 0 or z.shape[0] == 0:
		return 0.0, 0.0, 0.0

	n = min(int(re.size), int(z.shape[0]))
	re = signal.detrend(re[-n:], type="linear")
	z = signal.detrend(z[-n:, :], axis=0, type="linear")
	max_lag = min(int(max_lag), n - 1)

	def _best_for_component(component):
		best = 0.0
		for lag in range(-max_lag, max_lag + 1):
			if lag >= 0:
				re_seg = re[: n - lag]
				z_seg = component[lag:]
			else:
				shift = -lag
				re_seg = re[shift:]
				z_seg = component[: n - shift]
			corr = abs(_segment_corrcoef(re_seg, z_seg))
			if corr > best:
				best = corr
		return float(best)

	return (
		_best_for_component(z[:, 0]),
		_best_for_component(z[:, 1]),
		_best_for_component(z[:, 2]),
	)


def calculate_slr(sensitivity_base, sensitivity_asy, epsilon=1e-9):
	"""Sensitivity loss rate under ASY weighting.

	Formula: max(0, (S_base - S_asy) / (|S_base| + epsilon))
	"""
	epsilon = float(epsilon)
	if epsilon <= 0.0:
		raise ValueError("epsilon must be > 0")
	base = np.asarray(sensitivity_base, dtype=float)
	asy = np.asarray(sensitivity_asy, dtype=float)
	slr = np.maximum(0.0, (base - asy) / (np.abs(base) + epsilon))
	if slr.ndim == 0:
		return float(slr)
	return slr


def strategy_distribution(
	players,
	strategy_space,
	*,
	attr="last_strategy",
):
	"""Compute distribution over strategies in [0, 1].

	Counts values from each player's `attr` (default: last_strategy). Missing/None ignored.
	"""

	counts = {s: 0 for s in strategy_space}
	n = 0
	for p in players:
		s = getattr(p, attr, None)
		if s is None or s not in counts:
			continue
		counts[s] += 1
		n += 1

	if n == 0:
		return {s: 0.0 for s in strategy_space}

	return {s: (counts[s] / n) for s in strategy_space}


def average_utility(players, *, attr="utility"):
	"""Compute average utility across players."""

	total = 0.0
	n = 0
	for p in players:
		u = getattr(p, attr, None)
		if u is None:
			continue
		total += float(u)
		n += 1
	return total / n if n else 0.0


def average_reward(players, *, attr="last_reward"):
	"""Compute average reward across players for the last step.

	Returns None if no rewards are found.
	"""

	total = 0.0
	n = 0
	for p in players:
		r = getattr(p, attr, None)
		if r is None:
			continue
		total += float(r)
		n += 1
	return (total / n) if n else None

