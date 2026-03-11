"""Cycle metrics (research / SDD).

原則
- 純計算、無 I/O
- 僅使用標準庫（避免研究環境被依賴卡住）

本檔提供兩個你目前選定的指標：
1) 振幅下限（peak-to-peak amplitude）
2) 主頻存在（以簡化自相關找 dominant lag）

注意
- 這些指標不是嚴格的動力系統「證明」，而是可回歸、可比較的實驗驗收準則。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, sqrt
from random import Random
from typing import Literal, Mapping, Optional, Sequence

from .simplex import normalize_simplex3


def _slice_series(
	series: Sequence[float],
	*,
	burn_in: int = 0,
	tail: int | None = None,
) -> list[float]:
	if burn_in < 0:
		raise ValueError("burn_in must be >= 0")
	if tail is not None and tail < 0:
		raise ValueError("tail must be >= 0")

	n = len(series)
	start = min(max(burn_in, 0), n)
	if tail is None:
		seg = series[start:]
	else:
		seg_start = max(start, n - tail)
		seg = series[seg_start:]

	# Filter out non-finite values (e.g., NaN in avg_reward column).
	out: list[float] = []
	for v in seg:
		fv = float(v)
		if isfinite(fv):
			out.append(fv)
	return out


def peak_to_peak_amplitude(
	series: Sequence[float],
	*,
	burn_in: int = 0,
	tail: int | None = None,
) -> float:
	"""Peak-to-peak amplitude over a time window.

	Returns 0.0 for empty / all-nonfinite segments.
	"""
	seg = _slice_series(series, burn_in=burn_in, tail=tail)
	if not seg:
		return 0.0
	return float(max(seg) - min(seg))


@dataclass(frozen=True)
class DominantFrequency:
	"""Dominant periodicity proxy from autocorrelation.

	- lag: the step lag with maximum autocorrelation within the searched band
	- corr: correlation score in [-1, 1] (larger is more periodic)
	- freq: cycles per step (= 1/lag) if lag is not None
	"""

	lag: Optional[int]
	corr: float
	freq: Optional[float]


@dataclass(frozen=True)
class Stage1AmplitudeResult:
	passed: bool
	threshold: float
	aggregation: str
	amplitudes: dict[str, float]
	normalization: str = "none"
	threshold_factor: Optional[float] = None
	control_reference: Optional[float] = None
	normalized_amplitudes: Optional[dict[str, float]] = None


@dataclass(frozen=True)
class Stage2FrequencyResult:
	passed: bool
	corr_threshold: float
	aggregation: str
	frequencies: dict[str, DominantFrequency]
	method: str = "autocorr_threshold"
	statistic_name: Optional[str] = None
	statistic: Optional[float] = None
	effective_window_n: int = 0
	stage2_prefilter: bool = True
	prefilter_passed: Optional[bool] = None
	per_strategy_statistic: Optional[dict[str, float]] = None
	power_ratio_kappa: Optional[float] = None
	permutation_alpha: Optional[float] = None
	permutation_resamples: Optional[int] = None
	permutation_seed: Optional[int] = None
	per_strategy_p_value: Optional[dict[str, float]] = None
	per_strategy_power_ratio: Optional[dict[str, float]] = None
	per_strategy_prefilter: Optional[dict[str, bool]] = None
	per_strategy_pass: Optional[dict[str, bool]] = None
	window_n_by_strategy: Optional[dict[str, int]] = None


def _effective_window_n(
	proportions: Mapping[str, Sequence[float]],
	*,
	burn_in: int,
	tail: int | None,
) -> tuple[int, dict[str, int]]:
	window_n_by_strategy: dict[str, int] = {
		name: len(_slice_series(series, burn_in=burn_in, tail=tail))
		for name, series in proportions.items()
	}
	if not window_n_by_strategy:
		return 0, window_n_by_strategy
	return int(min(window_n_by_strategy.values())), window_n_by_strategy


def _goertzel_power_at_bin(seg: Sequence[float], *, k: int) -> float:
	"""Compute |DFT_k|^2 for a real-valued sequence using Goertzel.

	DFT convention: X_k = sum_{t=0}^{n-1} x_t * exp(-i*2*pi*k*t/n)
	Returns |X_k|^2.
	"""
	n = len(seg)
	if n <= 0:
		return 0.0
	if k <= 0 or k >= n:
		return 0.0
	omega = 2.0 * pi * (float(k) / float(n))
	coeff = 2.0 * cos(omega)
	s_prev = 0.0
	s_prev2 = 0.0
	for v in seg:
		s = float(v) + coeff * s_prev - s_prev2
		s_prev2 = s_prev
		s_prev = s
	# power = s_prev2^2 + s_prev^2 - coeff*s_prev*s_prev2
	power = (s_prev2 * s_prev2) + (s_prev * s_prev) - (coeff * s_prev * s_prev2)
	return float(power) if isfinite(power) and power > 0.0 else 0.0


def _fft_power_ratio_from_autocorr_peak(
	series: Sequence[float],
	*,
	burn_in: int,
	tail: int | None,
	min_lag: int,
	max_lag: int,
) -> tuple[float, DominantFrequency, int]:
	"""Estimate power ratio P(f*)/mean(P) using the dominant autocorr lag.

	- f* is inferred from the dominant autocorrelation lag.
	- Power at f* is computed via Goertzel (O(n)).
	- Mean power is approximated from Parseval's identity.

	Returns (power_ratio, dominant_frequency, window_n).
	"""
	seg = _slice_series(series, burn_in=burn_in, tail=tail)
	n = len(seg)
	if n < 8:
		return 0.0, DominantFrequency(lag=None, corr=0.0, freq=None), n

	# Mean-center.
	mean = sum(seg) / float(n)
	y = [float(v) - float(mean) for v in seg]
	# If near-constant after centering, no meaningful spectrum.
	if (max(y) - min(y)) <= 1e-12:
		return 0.0, DominantFrequency(lag=None, corr=0.0, freq=None), n

	df = dominant_frequency_autocorr(y, burn_in=0, tail=None, min_lag=min_lag, max_lag=max_lag)
	if df.lag is None or df.lag <= 0:
		return 0.0, df, n

	# Map lag -> nearest positive DFT bin.
	m = n // 2
	if m < 1:
		return 0.0, df, n
	k_bin = int(round(float(n) / float(df.lag)))
	k_bin = max(1, min(m, k_bin))

	p_star = _goertzel_power_at_bin(y, k=k_bin)
	energy = sum(float(v) * float(v) for v in y)
	if not isfinite(energy) or energy <= 0.0:
		return 0.0, df, n
	# Parseval: sum_k |X_k|^2 = n * sum_t y_t^2.
	# For real y, positive frequencies contain ~ half the energy (excluding DC).
	total_power_pos = (float(n) * float(energy)) / 2.0
	mean_power = total_power_pos / float(m) if m > 0 else 0.0
	if mean_power <= 0.0:
		return 0.0, df, n
	ratio = float(p_star) / float(mean_power)
	if not isfinite(ratio) or ratio < 0.0:
		ratio = 0.0
	return float(ratio), df, n


def _permutation_p_value_autocorr(
	series: Sequence[float],
	*,
	burn_in: int,
	tail: int | None,
	min_lag: int,
	max_lag: int,
	resamples: int,
	seed: int | None,
) -> tuple[float, DominantFrequency, int]:
	"""Permutation test for dominant autocorr (one-sided; larger corr is more periodic).

	Statistic: best_corr from dominant_frequency_autocorr.
	Returns (p_value, observed_df, window_n).
	"""
	seg = _slice_series(series, burn_in=burn_in, tail=tail)
	n = len(seg)
	if n < (min_lag + 2):
		return 1.0, DominantFrequency(lag=None, corr=0.0, freq=None), n

	obs = dominant_frequency_autocorr(seg, burn_in=0, tail=None, min_lag=min_lag, max_lag=max_lag)
	obs_stat = float(obs.corr) if (obs.lag is not None and isfinite(obs.corr)) else 0.0
	if obs.lag is None:
		return 1.0, obs, n

	rng = Random(seed)
	arr = list(seg)
	count_ge = 0
	R = int(resamples)
	for _ in range(R):
		rng.shuffle(arr)
		perm = dominant_frequency_autocorr(arr, burn_in=0, tail=None, min_lag=min_lag, max_lag=max_lag)
		stat = float(perm.corr) if (perm.lag is not None and isfinite(perm.corr)) else 0.0
		if stat >= obs_stat - 1e-12:
			count_ge += 1
	# add-one smoothing for stability
	p = (count_ge + 1.0) / (float(R) + 1.0)
	return float(p), obs, n


@dataclass(frozen=True)
class PhaseDirectionResult:
	"""Phase rotation direction consistency around the simplex centroid.

	- direction: +1 (ccw), -1 (cw), 0 (no consistent direction)
	- score: abs(mean(sign(z_t))) in [0, 1]
	"""

	direction: int
	score: float
	turn_strength: float
	eta: float
	min_turn_strength: float
	passed: bool


@dataclass(frozen=True)
class CycleLevelResult:
	"""Formal graded cycle assessment (Level 0..3).

	Level 0: none
	Level 1: oscillation (amplitude)
	Level 2: periodic oscillation (amplitude + dominant frequency)
	Level 3: structured cycle (Level 2 + consistent phase direction)
	"""

	level: int
	stage1: Stage1AmplitudeResult
	stage2: Optional[Stage2FrequencyResult]
	stage3: Optional[PhaseDirectionResult]


def normalize_simplex_timeseries(
	series: Mapping[str, Sequence[float]],
	*,
	strategies: Sequence[str] = ("aggressive", "defensive", "balanced"),
) -> dict[str, list[float]]:
	"""Normalize per-time-step values into a simplex trajectory.

	Given multiple strategy series (e.g. weights), produce x_i(t) = v_i(t) / sum_j v_j(t).

	Assumptions
	- Intended for non-negative series (weights or proportions), but clips tiny
	  negatives to 0 as a numerical drift guard.
	- If at some t the total is <= 0 after clipping (or values are non-finite),
	  returns 0 for all strategies at that t.
	"""

	keys = list(strategies)
	# Determine length by the shortest available series among requested strategies.
	lengths = [len(series.get(k, ())) for k in keys]
	n = min(lengths) if lengths else 0
	out = {k: [0.0] * n for k in keys}
	if n == 0:
		return out

	for t in range(n):
		vals = [float(series.get(k, [0.0] * n)[t]) for k in keys]
		if len(keys) == 3:
			res = normalize_simplex3(vals, clip_negative=True, on_invalid="zeros")
			x1, x2, x3 = res.x
			if (x1 + x2 + x3) <= 0.0:
				continue
			out[keys[0]][t] = float(x1)
			out[keys[1]][t] = float(x2)
			out[keys[2]][t] = float(x3)
		else:
			total = float(sum(vals))
			if total <= 0.0 or not isfinite(total):
				continue
			for k, v in zip(keys, vals):
				out[k][t] = float(v / total)
	return out


def smooth_moving_average(series: Sequence[float], *, window: int) -> list[float]:
	"""Simple centered moving-average smoothing.

	- window=1 returns a copy of the input.
	- Uses edge clamping (smaller window near boundaries).
	- Intended as a denoising option for phase-direction metrics.
	"""
	w = int(window)
	if w <= 1:
		return [float(x) for x in series]
	if w % 2 == 0:
		raise ValueError("window must be odd (e.g., 3,5,7)")
	h = w // 2
	out: list[float] = []
	n = len(series)
	for i in range(n):
		lo = max(0, i - h)
		hi = min(n, i + h + 1)
		seg = series[lo:hi]
		s = 0.0
		c = 0
		for v in seg:
			fv = float(v)
			if not isfinite(fv):
				continue
			s += fv
			c += 1
		out.append((s / c) if c else 0.0)
	return out


@dataclass(frozen=True)
class Stage3DirectionResult:
	"""Stage 3: direction consistency (phase rotation).

	Computed from a 3D simplex trajectory projected to 2D. The metric reports:
	- net_rotation: signed sum of local turning cross products (Δp_t × Δp_{t+1})
	- direction: "ccw" / "cw" if net_rotation != 0, else None
	- consistency: fraction of steps whose local rotation sign matches net sign
	
	Operationally this is a regression-friendly proxy for "single-direction cycling".
	"""

	passed: bool
	min_abs_net_rotation: float
	consistency_threshold: float
	net_rotation: float
	direction: Optional[str]
	consistency: float
	used_steps: int
	used_points: int
	order: tuple[str, str, str]


def _window_range(n: int, *, burn_in: int = 0, tail: int | None = None) -> tuple[int, int]:
	if burn_in < 0:
		raise ValueError("burn_in must be >= 0")
	if tail is not None and tail < 0:
		raise ValueError("tail must be >= 0")
	if n <= 0:
		return (0, 0)
	start = min(max(int(burn_in), 0), n)
	if tail is None:
		return (start, n)
	end = n
	begin = max(start, end - int(tail))
	return (begin, end)


def _simplex3_to_2d(x1: float, x2: float, x3: float) -> tuple[float, float]:
	"""Project (x1,x2,x3) on simplex to 2D coordinates.

	Uses an equilateral-triangle embedding:
	- v1 = (0,0), v2 = (1,0), v3 = (1/2, sqrt(3)/2)
	- point = x1*v1 + x2*v2 + x3*v3
	"""
	# x1 is implicitly weighted on origin.
	return (x2 + 0.5 * x3, (sqrt(3.0) * 0.5) * x3)


def assess_stage3_direction(
	series3: Mapping[str, Sequence[float]],
	*,
	order: tuple[str, str, str] = ("aggressive", "defensive", "balanced"),
	burn_in: int = 0,
	tail: int | None = None,
	min_abs_net_rotation: float = 0.005,
	consistency_threshold: float = 0.65,
	eps: float = 1e-15,
) -> Stage3DirectionResult:
	"""Stage 3: verify single-direction rotation in phase space.

	Notes
	- Intended input is a 3-strategy time series such as `w_*` (weights) or `p_*`.
	- For regression stability under finite-population sampling, prefer `w_*`.
	"""
	key1, key2, key3 = order
	if key1 not in series3 or key2 not in series3 or key3 not in series3:
		raise ValueError(f"series3 must contain keys {order!r}")

	s1 = series3[key1]
	s2 = series3[key2]
	s3 = series3[key3]
	n = min(len(s1), len(s2), len(s3))
	start, end = _window_range(n, burn_in=burn_in, tail=tail)

	points: list[tuple[float, float]] = []
	for i in range(start, end):
		v1 = float(s1[i])
		v2 = float(s2[i])
		v3 = float(s3[i])
		res = normalize_simplex3((v1, v2, v3), clip_negative=True, on_invalid="zeros")
		x1, x2, x3 = res.x
		if (x1 + x2 + x3) <= 0.0:
			continue
		points.append(_simplex3_to_2d(x1, x2, x3))

	if len(points) < 6:
		return Stage3DirectionResult(
			passed=False,
			min_abs_net_rotation=float(min_abs_net_rotation),
			consistency_threshold=float(consistency_threshold),
			net_rotation=0.0,
			direction=None,
			consistency=0.0,
			used_steps=0,
			used_points=len(points),
			order=order,
		)

	# Use local turning direction: cross(Δp_t, Δp_{t+1}).
	# This is robust for "orbit-like" motion and naturally rejects 1D back-and-forth.
	crosses: list[float] = []
	for t in range(len(points) - 2):
		x0, y0 = points[t]
		x1, y1 = points[t + 1]
		x2, y2 = points[t + 2]
		d1x = x1 - x0
		d1y = y1 - y0
		d2x = x2 - x1
		d2y = y2 - y1
		cross = (d1x * d2y) - (d1y * d2x)
		if abs(cross) > float(eps):
			crosses.append(cross)

	if not crosses:
		return Stage3DirectionResult(
			passed=False,
			min_abs_net_rotation=float(min_abs_net_rotation),
			consistency_threshold=float(consistency_threshold),
			net_rotation=0.0,
			direction=None,
			consistency=0.0,
			used_steps=0,
			used_points=len(points),
			order=order,
		)

	net = float(sum(crosses))
	if abs(net) <= float(eps):
		return Stage3DirectionResult(
			passed=False,
			min_abs_net_rotation=float(min_abs_net_rotation),
			consistency_threshold=float(consistency_threshold),
			net_rotation=0.0,
			direction=None,
			consistency=0.0,
			used_steps=len(crosses),
			used_points=len(points),
			order=order,
		)

	net_positive = net > 0.0
	direction = "ccw" if net_positive else "cw"
	agree = sum(1 for c in crosses if (c > 0.0) == net_positive)
	consistency = float(agree) / float(len(crosses))
	passed = (abs(net) >= float(min_abs_net_rotation)) and (
		consistency >= float(consistency_threshold)
	)

	return Stage3DirectionResult(
		passed=bool(passed),
		min_abs_net_rotation=float(min_abs_net_rotation),
		consistency_threshold=float(consistency_threshold),
		net_rotation=float(net),
		direction=direction,
		consistency=float(consistency),
		used_steps=len(crosses),
		used_points=len(points),
		order=order,
	)


def dominant_frequency_autocorr(
	series: Sequence[float],
	*,
	burn_in: int = 0,
	tail: int | None = None,
	min_lag: int = 2,
	max_lag: int = 500,
) -> DominantFrequency:
	"""Estimate whether a non-zero dominant period exists via autocorrelation.

	Implementation notes
	- Uses mean-centered autocorrelation score: corr(lag) = sum(y_t y_{t+lag}) / sum(y_t^2)
	- Searches lags in [min_lag, max_lag] capped by (n-1)
	- Pure Python O(T * max_lag): choose max_lag conservatively for long series.
	"""

	if min_lag < 1:
		raise ValueError("min_lag must be >= 1")
	if max_lag < min_lag:
		raise ValueError("max_lag must be >= min_lag")

	seg = _slice_series(series, burn_in=burn_in, tail=tail)
	n = len(seg)
	if n < (min_lag + 2):
		return DominantFrequency(lag=None, corr=0.0, freq=None)

	# Guard: treat near-constant as no periodicity (robust against float rounding).
	if (max(seg) - min(seg)) <= 1e-12:
		return DominantFrequency(lag=None, corr=0.0, freq=None)

	mean = sum(seg) / n
	y = [v - mean for v in seg]

	# Guard against degenerate autocorrelation at extreme lags.
	# - If the overlap window is too small (e.g., length 1), Pearson-like corr(lag)
	#   becomes ±1 by construction and produces severe false positives.
	# - If lag is too large relative to the window, you cannot observe multiple
	#   cycles, and the "dominant" lag tends to overfit drift/trends.
	min_overlap = 10
	min_cycles = 3
	max_lag_eff = min(max_lag, n - min_overlap, n // min_cycles)
	if max_lag_eff < min_lag:
		return DominantFrequency(lag=None, corr=0.0, freq=None)

	best_lag: Optional[int] = None
	best_corr = float("-inf")
	# Tiebreak: if corr is equal within eps, prefer smaller lag.
	eps = 1e-12
	for lag in range(min_lag, max_lag_eff + 1):
		num = 0.0
		# Pearson-like normalized autocorrelation over the overlapping window.
		# corr(lag) = <y0, y1> / (||y0|| * ||y1||)
		# where y0 = y[0:n-lag], y1 = y[lag:n]
		limit = n - lag
		den0 = 0.0
		den1 = 0.0
		for i in range(limit):
			v0 = y[i]
			v1 = y[i + lag]
			num += v0 * v1
			den0 += v0 * v0
			den1 += v1 * v1
		den = sqrt(den0 * den1) if (den0 > 0.0 and den1 > 0.0) else 0.0
		corr = (num / den) if den > 0.0 else 0.0
		if corr > best_corr + eps or (
			abs(corr - best_corr) <= eps and (best_lag is None or lag < best_lag)
		):
			best_corr = corr
			best_lag = lag

	if best_lag is None or not isfinite(best_corr):
		return DominantFrequency(lag=None, corr=0.0, freq=None)

	return DominantFrequency(lag=int(best_lag), corr=float(best_corr), freq=(1.0 / float(best_lag)))


def has_dominant_frequency(
	series: Sequence[float],
	*,
	burn_in: int = 0,
	tail: int | None = None,
	min_lag: int = 2,
	max_lag: int = 500,
	corr_threshold: float = 0.3,
) -> bool:
	"""Binary check for "主頻存在".

	This is an operational (research) criterion: returns True iff the dominant
	autocorrelation within the searched lag range exceeds a threshold.
	"""
	res = dominant_frequency_autocorr(
		series,
		burn_in=burn_in,
		tail=tail,
		min_lag=min_lag,
		max_lag=max_lag,
	)
	return res.lag is not None and res.corr >= float(corr_threshold)


def assess_stage1_amplitude(
	proportions: Mapping[str, Sequence[float]],
	*,
	burn_in: int = 0,
	tail: int | None = None,
	threshold: float = 0.02,
	aggregation: str = "any",
	normalization: Literal["none", "control_mean", "control_std"] = "none",
	control_reference: float | None = None,
	threshold_factor: float = 3.0,
	eps: float = 1e-12,
) -> Stage1AmplitudeResult:
	"""Stage 1: amplitude screen across multiple strategies.

	Amplitude normalization (publication-grade option)
	- normalization="none": use absolute threshold `threshold` (legacy behavior)
	- normalization in {"control_mean","control_std"}:
		- require `control_reference` (>0) and `threshold_factor` (>0)
		- effective threshold (raw units) = threshold_factor * control_reference
		- also returns per-strategy normalized amplitudes A/ control_reference

	aggregation:
	- "any": pass if any strategy amplitude >= threshold
	- "all": pass if all strategies amplitude >= threshold
	- "mean": pass if mean amplitude >= threshold
	- "min": pass if min amplitude >= threshold (equivalent to "all" but numeric)
	"""

	amps = {
		name: peak_to_peak_amplitude(series, burn_in=burn_in, tail=tail)
		for name, series in proportions.items()
	}
	vals = list(amps.values())

	norm = str(normalization)
	norm_amps: Optional[dict[str, float]] = None
	thr = float(threshold)
	factor_out: Optional[float] = None
	control_out: Optional[float] = None
	if norm != "none":
		ref = float(control_reference) if control_reference is not None else None
		if ref is None:
			raise ValueError("control_reference is required when normalization != 'none'")
		if not isfinite(ref) or ref <= float(eps):
			raise ValueError("control_reference must be finite and > eps")
		fac = float(threshold_factor)
		if not isfinite(fac) or fac <= 0.0:
			raise ValueError("threshold_factor must be finite and > 0")
		thr = fac * ref
		norm_amps = {k: (float(v) / ref) for k, v in amps.items()}
		factor_out = fac
		control_out = ref

	if not vals:
		passed = False
	else:
		agg = str(aggregation)
		if agg == "any":
			passed = any(v >= thr for v in vals)
		elif agg == "all":
			passed = all(v >= thr for v in vals)
		elif agg == "mean":
			passed = (sum(vals) / len(vals)) >= thr
		elif agg == "min":
			passed = min(vals) >= thr
		else:
			raise ValueError(f"Unknown aggregation: {aggregation!r}")

	return Stage1AmplitudeResult(
		passed=bool(passed),
		threshold=thr,
		aggregation=str(aggregation),
		amplitudes=amps,
		normalization=norm,
		threshold_factor=factor_out,
		control_reference=control_out,
		normalized_amplitudes=norm_amps,
	)


def assess_stage2_frequency(
	proportions: Mapping[str, Sequence[float]],
	*,
	burn_in: int = 0,
	tail: int | None = None,
	min_lag: int = 2,
	max_lag: int = 500,
	corr_threshold: float = 0.3,
	aggregation: str = "any",
	stage2_method: Literal["autocorr_threshold", "fft_power_ratio", "permutation_p"] = "autocorr_threshold",
	stage2_prefilter: bool = True,
	power_ratio_kappa: float = 8.0,
	permutation_alpha: float = 0.05,
	permutation_resamples: int = 200,
	permutation_seed: int | None = None,
) -> Stage2FrequencyResult:
	"""Stage 2: frequency significance across multiple strategies.

	Supported methods
	- autocorr_threshold (legacy): pass if dominant autocorr corr >= corr_threshold
	- fft_power_ratio: pass if power_ratio >= power_ratio_kappa
		(power_ratio is computed at the dominant autocorr lag and normalized by mean spectral power)
	- permutation_p: pass if permutation p-value <= permutation_alpha
		(test statistic = dominant autocorr corr; resamples controlled by permutation_resamples)

	Prefilter
	- If stage2_prefilter is True and method != autocorr_threshold, we require a
	  cheap autocorr pass (corr >= corr_threshold) before running the heavier
	  method on that strategy.
	"""

	# Always compute dominant autocorr outputs (used for reporting and for prefilter).
	results = {
		name: dominant_frequency_autocorr(
			series,
			burn_in=burn_in,
			tail=tail,
			min_lag=min_lag,
			max_lag=max_lag,
		)
		for name, series in proportions.items()
	}
	thr = float(corr_threshold)
	method = str(stage2_method)
	win_n, win_by = _effective_window_n(proportions, burn_in=burn_in, tail=tail)

	# Prefilter by autocorr threshold (per-strategy).
	prefilter_flags: dict[str, bool] = {
		name: (df.lag is not None and float(df.corr) >= thr)
		for name, df in results.items()
	}

	per_strategy_pass: dict[str, bool] = {}
	per_strategy_stat: dict[str, float] = {}
	per_strategy_p: dict[str, float] = {}
	per_strategy_ratio: dict[str, float] = {}
	stat_name: Optional[str] = None

	if method == "autocorr_threshold":
		stat_name = "best_corr"
		for name, df in results.items():
			corr = float(df.corr) if (df.lag is not None and isfinite(df.corr)) else 0.0
			per_strategy_stat[name] = float(corr)
			per_strategy_pass[name] = bool(df.lag is not None and corr >= thr)
	elif method == "fft_power_ratio":
		stat_name = "power_ratio"
		kappa = float(power_ratio_kappa)
		if not isfinite(kappa) or kappa <= 0.0:
			raise ValueError("power_ratio_kappa must be finite and > 0")
		for name, series in proportions.items():
			if bool(stage2_prefilter) and not prefilter_flags.get(name, False):
				per_strategy_ratio[name] = 0.0
				per_strategy_pass[name] = False
				continue
			ratio, _df, _n = _fft_power_ratio_from_autocorr_peak(
				series,
				burn_in=burn_in,
				tail=tail,
				min_lag=min_lag,
				max_lag=max_lag,
			)
			per_strategy_ratio[name] = float(ratio)
			per_strategy_pass[name] = bool(float(ratio) >= kappa)
			per_strategy_stat[name] = float(ratio)
	elif method == "permutation_p":
		stat_name = "p_value"
		alpha = float(permutation_alpha)
		if not isfinite(alpha) or not (0.0 < alpha <= 1.0):
			raise ValueError("permutation_alpha must be in (0, 1]")
		R = int(permutation_resamples)
		if R < 1:
			raise ValueError("permutation_resamples must be >= 1")
		for name, series in proportions.items():
			if bool(stage2_prefilter) and not prefilter_flags.get(name, False):
				per_strategy_p[name] = 1.0
				per_strategy_pass[name] = False
				continue
			pval, _obs, _n = _permutation_p_value_autocorr(
				series,
				burn_in=burn_in,
				tail=tail,
				min_lag=min_lag,
				max_lag=max_lag,
				resamples=R,
				seed=permutation_seed,
			)
			per_strategy_p[name] = float(pval)
			per_strategy_pass[name] = bool(float(pval) <= alpha)
			per_strategy_stat[name] = float(pval)
	else:
		raise ValueError(f"Unknown stage2_method: {stage2_method!r}")

	flags = list(per_strategy_pass.values())
	if not flags:
		passed = False
	else:
		agg = str(aggregation)
		if agg == "any":
			passed = any(flags)
		elif agg == "all":
			passed = all(flags)
		else:
			raise ValueError(f"Unknown aggregation: {aggregation!r}")

	# Aggregate statistic for reporting.
	stat_vals = list(per_strategy_stat.values())
	stat_agg: Optional[float]
	if not stat_vals:
		stat_agg = None
	elif method == "permutation_p":
		# smaller is better
		stat_agg = float(min(stat_vals)) if str(aggregation) == "any" else float(max(stat_vals))
	else:
		# larger is better
		stat_agg = float(max(stat_vals)) if str(aggregation) == "any" else float(min(stat_vals))

	return Stage2FrequencyResult(
		passed=bool(passed),
		corr_threshold=thr,
		aggregation=str(aggregation),
		frequencies=results,
		method=method,
		statistic_name=str(stat_name) if stat_name is not None else None,
		statistic=stat_agg,
		effective_window_n=int(win_n),
		stage2_prefilter=bool(stage2_prefilter),
		prefilter_passed=(bool(any(prefilter_flags.values())) if prefilter_flags else None),
		per_strategy_statistic=(per_strategy_stat if per_strategy_stat else None),
		power_ratio_kappa=(float(power_ratio_kappa) if method == "fft_power_ratio" else None),
		permutation_alpha=(float(permutation_alpha) if method == "permutation_p" else None),
		permutation_resamples=(int(permutation_resamples) if method == "permutation_p" else None),
		permutation_seed=(int(permutation_seed) if (method == "permutation_p" and permutation_seed is not None) else None),
		per_strategy_p_value=(per_strategy_p if per_strategy_p else None),
		per_strategy_power_ratio=(per_strategy_ratio if per_strategy_ratio else None),
		per_strategy_prefilter=(prefilter_flags if prefilter_flags else None),
		per_strategy_pass=(per_strategy_pass if per_strategy_pass else None),
		window_n_by_strategy=(win_by if win_by else None),
	)


def phase_direction_consistency(
	proportions: Mapping[str, Sequence[float]],
	*,
	strategies: Sequence[str] = ("aggressive", "defensive", "balanced"),
	burn_in: int = 0,
	tail: int | None = None,
	eta: float = 0.6,
	min_turn_strength: float = 0.0,
) -> PhaseDirectionResult:
	"""Stage 3: direction consistency of rotation in (pA, pD) plane.

	We exploit the simplex constraint pB = 1 - pA - pD and use a fixed
	centroid c = (1/3, 1/3) in the (pA, pD) coordinates.

	For each step t we compute:
		z_t = cross( (x_t - c), (x_{t+1} - c) )
	and aggregate score = abs(mean(sign(z_t))).
	"""

	if eta < 0.0 or eta > 1.0:
		raise ValueError("eta must be in [0, 1]")
	if min_turn_strength < 0.0:
		raise ValueError("min_turn_strength must be >= 0")

	if len(strategies) != 3:
		raise ValueError("strategies must have length 3")

	name_a, name_d, _name_b = strategies
	if name_a not in proportions or name_d not in proportions:
		return PhaseDirectionResult(
			direction=0,
			score=0.0,
			turn_strength=0.0,
			eta=float(eta),
			min_turn_strength=float(min_turn_strength),
			passed=False,
		)

	a = _slice_series(proportions[name_a], burn_in=burn_in, tail=tail)
	d = _slice_series(proportions[name_d], burn_in=burn_in, tail=tail)
	n = min(len(a), len(d))
	if n < 3:
		return PhaseDirectionResult(
			direction=0,
			score=0.0,
			turn_strength=0.0,
			eta=float(eta),
			min_turn_strength=float(min_turn_strength),
			passed=False,
		)

	# Fixed centroid in (pA, pD)
	cx = 1.0 / 3.0
	cy = 1.0 / 3.0

	signs_sum = 0
	count = 0
	abs_z_sum = 0.0
	# Numerical deadzone to avoid flipping signs from tiny noise.
	deadzone = 1e-12
	for i in range(n - 1):
		x0 = float(a[i]) - cx
		y0 = float(d[i]) - cy
		x1 = float(a[i + 1]) - cx
		y1 = float(d[i + 1]) - cy
		z = x0 * y1 - y0 * x1
		if abs(z) <= deadzone:
			continue
		abs_z_sum += abs(z)
		signs_sum += 1 if z > 0.0 else -1
		count += 1

	if count == 0:
		return PhaseDirectionResult(
			direction=0,
			score=0.0,
			turn_strength=0.0,
			eta=float(eta),
			min_turn_strength=float(min_turn_strength),
			passed=False,
		)

	mean_sign = signs_sum / float(count)
	score = abs(mean_sign)
	turn_strength = abs_z_sum / float(count)
	direction = 0
	passed = (score >= float(eta)) and (turn_strength >= float(min_turn_strength))
	if passed:
		direction = 1 if mean_sign > 0.0 else -1

	return PhaseDirectionResult(
		direction=int(direction),
		score=float(score),
		turn_strength=float(turn_strength),
		eta=float(eta),
		min_turn_strength=float(min_turn_strength),
		passed=bool(passed and direction != 0),
	)


def phase_direction_consistency_turning(
	series3: Mapping[str, Sequence[float]],
	*,
	strategies: Sequence[str] = ("aggressive", "defensive", "balanced"),
	burn_in: int = 0,
	tail: int | None = None,
	eta: float = 0.6,
	min_turn_strength: float = 0.0,
	phase_smoothing: int = 1,
	eps: float = 1e-15,
) -> PhaseDirectionResult:
	"""Stage 3 (v2): direction consistency via local turning (3-point metric).

	Compared to `phase_direction_consistency` (centroid 2-point cross), this metric:
	- Uses consecutive displacements (Δp_t × Δp_{t+1}), which rejects 1D back-and-forth.
	- Is closer to `assess_stage3_direction` but returns `PhaseDirectionResult` for
	  compatibility with `classify_cycle_level` and CSV reports.

	Returned fields
	- score: consistency fraction in [0,1]
	- turn_strength: mean(|cross|) over used steps
	"""
	if eta < 0.0 or eta > 1.0:
		raise ValueError("eta must be in [0, 1]")
	if min_turn_strength < 0.0:
		raise ValueError("min_turn_strength must be >= 0")
	if len(strategies) != 3:
		raise ValueError("strategies must have length 3")

	key1, key2, key3 = strategies
	if key1 not in series3 or key2 not in series3 or key3 not in series3:
		return PhaseDirectionResult(
			direction=0,
			score=0.0,
			turn_strength=0.0,
			eta=float(eta),
			min_turn_strength=float(min_turn_strength),
			passed=False,
		)

	s1 = _slice_series(series3[key1], burn_in=burn_in, tail=tail)
	s2 = _slice_series(series3[key2], burn_in=burn_in, tail=tail)
	s3 = _slice_series(series3[key3], burn_in=burn_in, tail=tail)
	n = min(len(s1), len(s2), len(s3))
	if n < 6:
		return PhaseDirectionResult(
			direction=0,
			score=0.0,
			turn_strength=0.0,
			eta=float(eta),
			min_turn_strength=float(min_turn_strength),
			passed=False,
		)

	if phase_smoothing != 1:
		w = int(phase_smoothing)
		s1 = smooth_moving_average(s1, window=w)
		s2 = smooth_moving_average(s2, window=w)
		s3 = smooth_moving_average(s3, window=w)

	points: list[tuple[float, float]] = []
	for i in range(n):
		v1 = float(s1[i])
		v2 = float(s2[i])
		v3 = float(s3[i])
		res = normalize_simplex3((v1, v2, v3), clip_negative=True, on_invalid="zeros")
		x1, x2, x3 = res.x
		if (x1 + x2 + x3) <= 0.0:
			continue
		points.append(_simplex3_to_2d(x1, x2, x3))

	if len(points) < 6:
		return PhaseDirectionResult(
			direction=0,
			score=0.0,
			turn_strength=0.0,
			eta=float(eta),
			min_turn_strength=float(min_turn_strength),
			passed=False,
		)

	deadzone = float(eps)
	crosses: list[float] = []
	for t in range(len(points) - 2):
		x0, y0 = points[t]
		x1, y1 = points[t + 1]
		x2, y2 = points[t + 2]
		d1x = x1 - x0
		d1y = y1 - y0
		d2x = x2 - x1
		d2y = y2 - y1
		cross = (d1x * d2y) - (d1y * d2x)
		if abs(cross) <= deadzone:
			continue
		crosses.append(float(cross))

	if not crosses:
		return PhaseDirectionResult(
			direction=0,
			score=0.0,
			turn_strength=0.0,
			eta=float(eta),
			min_turn_strength=float(min_turn_strength),
			passed=False,
		)

	net = float(sum(crosses))
	if abs(net) <= deadzone:
		return PhaseDirectionResult(
			direction=0,
			score=0.0,
			turn_strength=0.0,
			eta=float(eta),
			min_turn_strength=float(min_turn_strength),
			passed=False,
		)

	net_positive = net > 0.0
	agree = sum(1 for c in crosses if (c > 0.0) == net_positive)
	consistency = float(agree) / float(len(crosses))
	turn_strength = sum(abs(c) for c in crosses) / float(len(crosses))
	passed = (consistency >= float(eta)) and (turn_strength >= float(min_turn_strength))
	if not passed:
		return PhaseDirectionResult(
			direction=0,
			score=float(consistency),
			turn_strength=float(turn_strength),
			eta=float(eta),
			min_turn_strength=float(min_turn_strength),
			passed=False,
		)
	return PhaseDirectionResult(
		direction=(1 if net_positive else -1),
		score=float(consistency),
		turn_strength=float(turn_strength),
		eta=float(eta),
		min_turn_strength=float(min_turn_strength),
		passed=True,
	)


def _quantile(values: Sequence[float], *, q: float) -> float:
	if not values:
		return 0.0
	qq = float(q)
	if qq < 0.0 or qq > 1.0:
		raise ValueError("q must be in [0, 1]")
	sv = sorted(float(v) for v in values)
	n = len(sv)
	idx = int(qq * float(n - 1))
	idx = max(0, min(n - 1, idx))
	return float(sv[idx])


def phase_direction_consistency_turning_windowed(
	series3: Mapping[str, Sequence[float]],
	*,
	strategies: Sequence[str] = ("aggressive", "defensive", "balanced"),
	burn_in: int = 0,
	tail: int | None = None,
	eta: float = 0.6,
	min_turn_strength: float = 0.0,
	phase_smoothing: int = 1,
	window: int = 80,
	step: int = 20,
	quantile: float = 0.75,
) -> PhaseDirectionResult:
	"""Stage 3 (v2-windowed): aggregate turning-metric over sliding windows.

	Motivation
	- A single long tail can dilute intermittent but real rotation.
	- Coarse search prefers recall (max), but for detection we can use a robust
	  quantile (default p75) across windows.

	Aggregation
	- score_agg = quantile({score_w}, q)
	- strength_agg = quantile({turn_strength_w}, q)
	- direction is based on the weighted vote of window directions.
	"""
	if window <= 0:
		raise ValueError("window must be > 0")
	if step <= 0:
		raise ValueError("step must be > 0")
	if len(strategies) != 3:
		raise ValueError("strategies must have length 3")

	key1, key2, key3 = strategies
	if key1 not in series3 or key2 not in series3 or key3 not in series3:
		return PhaseDirectionResult(
			direction=0,
			score=0.0,
			turn_strength=0.0,
			eta=float(eta),
			min_turn_strength=float(min_turn_strength),
			passed=False,
		)

	s1 = _slice_series(series3[key1], burn_in=burn_in, tail=tail)
	s2 = _slice_series(series3[key2], burn_in=burn_in, tail=tail)
	s3 = _slice_series(series3[key3], burn_in=burn_in, tail=tail)
	n = min(len(s1), len(s2), len(s3))
	if n < 6:
		return PhaseDirectionResult(
			direction=0,
			score=0.0,
			turn_strength=0.0,
			eta=float(eta),
			min_turn_strength=float(min_turn_strength),
			passed=False,
		)

	w = int(window)
	if w > n:
		w = n
	s = int(step)

	window_scores: list[float] = []
	window_strengths: list[float] = []
	window_dirs: list[int] = []

	start = 0
	while start + w <= n:
		seg = {
			key1: s1[start : start + w],
			key2: s2[start : start + w],
			key3: s3[start : start + w],
		}
		ph = phase_direction_consistency_turning(
			seg,
			strategies=strategies,
			burn_in=0,
			tail=None,
			eta=float(eta),
			min_turn_strength=float(min_turn_strength),
			phase_smoothing=int(phase_smoothing),
		)
		window_scores.append(float(ph.score))
		window_strengths.append(float(ph.turn_strength))
		window_dirs.append(int(ph.direction))
		start += s

	if not window_scores:
		ph = phase_direction_consistency_turning(
			{key1: s1, key2: s2, key3: s3},
			strategies=strategies,
			burn_in=0,
			tail=None,
			eta=float(eta),
			min_turn_strength=float(min_turn_strength),
			phase_smoothing=int(phase_smoothing),
		)
		return ph

	score_agg = _quantile(window_scores, q=float(quantile))
	strength_agg = _quantile(window_strengths, q=float(quantile))

	# Weighted direction vote: direction * score * strength.
	vote = 0.0
	for d, sc, st in zip(window_dirs, window_scores, window_strengths):
		vote += float(d) * float(sc) * float(st)
	direction = 0
	if vote > 0.0:
		direction = 1
	elif vote < 0.0:
		direction = -1

	passed = (score_agg >= float(eta)) and (strength_agg >= float(min_turn_strength)) and (direction != 0)
	return PhaseDirectionResult(
		direction=int(direction),
		score=float(score_agg),
		turn_strength=float(strength_agg),
		eta=float(eta),
		min_turn_strength=float(min_turn_strength),
		passed=bool(passed),
	)


def classify_cycle_level(
	proportions: Mapping[str, Sequence[float]],
	*,
	# shared windowing (critical for consistency)
	burn_in: int = 0,
	tail: int | None = None,
	# stage 1
	amplitude_threshold: float = 0.02,
	amplitude_aggregation: str = "any",
	amplitude_normalization: Literal["none", "control_mean", "control_std"] = "none",
	amplitude_control_reference: float | None = None,
	amplitude_threshold_factor: float = 3.0,
	# stage 2
	min_lag: int = 2,
	max_lag: int = 500,
	corr_threshold: float = 0.3,
	freq_aggregation: str = "any",
	stage2_method: Literal["autocorr_threshold", "fft_power_ratio", "permutation_p"] = "autocorr_threshold",
	stage2_prefilter: bool = True,
	power_ratio_kappa: float = 8.0,
	permutation_alpha: float = 0.05,
	permutation_resamples: int = 200,
	permutation_seed: int | None = None,
	# stage 3
	eta: float = 0.6,
	min_turn_strength: float = 0.0,
	strategies: Sequence[str] = ("aggressive", "defensive", "balanced"),
	normalize_for_phase: bool = False,
	stage3_method: Literal["centroid", "turning"] = "turning",
	phase_smoothing: int = 1,
	stage3_window: int | None = None,
	stage3_step: int = 20,
	stage3_quantile: float = 0.75,
) -> CycleLevelResult:
	"""Classify cycle strength into Level 0..3 using a graded framework."""

	stage1 = assess_stage1_amplitude(
		proportions,
		burn_in=burn_in,
		tail=tail,
		threshold=amplitude_threshold,
		aggregation=amplitude_aggregation,
		normalization=str(amplitude_normalization),
		control_reference=amplitude_control_reference,
		threshold_factor=float(amplitude_threshold_factor),
	)
	if not stage1.passed:
		return CycleLevelResult(level=0, stage1=stage1, stage2=None, stage3=None)

	stage2 = assess_stage2_frequency(
		proportions,
		burn_in=burn_in,
		tail=tail,
		min_lag=min_lag,
		max_lag=max_lag,
		corr_threshold=corr_threshold,
		aggregation=freq_aggregation,
		stage2_method=str(stage2_method),
		stage2_prefilter=bool(stage2_prefilter),
		power_ratio_kappa=float(power_ratio_kappa),
		permutation_alpha=float(permutation_alpha),
		permutation_resamples=int(permutation_resamples),
		permutation_seed=(int(permutation_seed) if permutation_seed is not None else None),
	)
	if not stage2.passed:
		return CycleLevelResult(level=1, stage1=stage1, stage2=stage2, stage3=None)

	phase_input = proportions
	if normalize_for_phase:
		phase_input = normalize_simplex_timeseries(proportions, strategies=strategies)

	method = str(stage3_method)
	if method == "centroid":
		stage3 = phase_direction_consistency(
			phase_input,
			strategies=strategies,
			burn_in=burn_in,
			tail=tail,
			eta=eta,
			min_turn_strength=min_turn_strength,
		)
	elif method == "turning":
		if stage3_window is None:
			stage3 = phase_direction_consistency_turning(
				phase_input,
				strategies=strategies,
				burn_in=burn_in,
				tail=tail,
				eta=eta,
				min_turn_strength=min_turn_strength,
				phase_smoothing=int(phase_smoothing),
			)
		else:
			stage3 = phase_direction_consistency_turning_windowed(
				phase_input,
				strategies=strategies,
				burn_in=burn_in,
				tail=tail,
				eta=eta,
				min_turn_strength=min_turn_strength,
				phase_smoothing=int(phase_smoothing),
				window=int(stage3_window),
				step=int(stage3_step),
				quantile=float(stage3_quantile),
			)
	else:
		raise ValueError("stage3_method must be 'centroid' or 'turning'")
	if not stage3.passed:
		return CycleLevelResult(level=2, stage1=stage1, stage2=stage2, stage3=stage3)

	return CycleLevelResult(level=3, stage1=stage1, stage2=stage2, stage3=stage3)
