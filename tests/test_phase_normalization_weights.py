import math

from analysis.cycle_metrics import classify_cycle_level


def test_stage3_weight_series_needs_simplex_normalization():
	# Construct a 3-strategy trajectory that is a clean rotation on the simplex.
	# Then multiply all components by a time-varying global factor to mimic "weights"
	# that are not on the simplex (but still represent the same proportions).
	# With normalize_for_phase=True, Stage3 should still pass.
	T = 240
	burn_in = 0

	# Simplex-like rotating proportions (all positive and sum to 1).
	p0: list[float] = []
	p1: list[float] = []
	p2: list[float] = []
	for t in range(T):
		angle = 2.0 * math.pi * (t / 40.0)
		# Small oscillations around 1/3 with phase shifts.
		base = 1.0 / 3.0
		amp = 0.08
		a = base + amp * math.sin(angle)
		b = base + amp * math.sin(angle + 2.0 * math.pi / 3.0)
		c = base + amp * math.sin(angle + 4.0 * math.pi / 3.0)
		# Numerical safety: keep strictly positive.
		a = max(a, 1e-6)
		b = max(b, 1e-6)
		c = max(c, 1e-6)
		s = a + b + c
		p0.append(a / s)
		p1.append(b / s)
		p2.append(c / s)

	# Make "weights" by applying a shared global scaling per timestep.
	s = [1.0 + 0.6 * math.sin(2.0 * math.pi * (t / 30.0)) for t in range(T)]
	w0 = [p0[t] * s[t] for t in range(T)]
	w1 = [p1[t] * s[t] for t in range(T)]
	w2 = [p2[t] * s[t] for t in range(T)]

	# Thresholds set so Stage1/2 pass easily for this clean periodic signal.
	kwargs = dict(
		burn_in=burn_in,
		amplitude_threshold=0.02,
		corr_threshold=0.5,
		eta=0.55,
		min_turn_strength=0.0,
	)

	proportions = {"aggressive": p0, "defensive": p1, "balanced": p2}
	weights = {"aggressive": w0, "defensive": w1, "balanced": w2}

	res_p = classify_cycle_level(proportions, normalize_for_phase=False, **kwargs)
	res_w_norm = classify_cycle_level(weights, normalize_for_phase=True, **kwargs)

	assert res_p.level == 3
	assert res_w_norm.level == 3


def test_stage3_proportion_series_ok_without_normalization():
	# Same construction but using already-normalized proportions should still pass.
	T = 240
	p0: list[float] = []
	p1: list[float] = []
	p2: list[float] = []
	for t in range(T):
		angle = 2.0 * math.pi * (t / 40.0)
		base = 1.0 / 3.0
		amp = 0.08
		a = base + amp * math.sin(angle)
		b = base + amp * math.sin(angle + 2.0 * math.pi / 3.0)
		c = base + amp * math.sin(angle + 4.0 * math.pi / 3.0)
		a = max(a, 1e-6)
		b = max(b, 1e-6)
		c = max(c, 1e-6)
		s = a + b + c
		p0.append(a / s)
		p1.append(b / s)
		p2.append(c / s)

	res = classify_cycle_level(
		{"aggressive": p0, "defensive": p1, "balanced": p2},
		burn_in=0,
		amplitude_threshold=0.02,
		corr_threshold=0.5,
		eta=0.55,
		min_turn_strength=0.0,
		normalize_for_phase=False,
	)
	assert res.level == 3

	# Normalization should not change a simplex series.
	res_norm = classify_cycle_level(
		{"aggressive": p0, "defensive": p1, "balanced": p2},
		burn_in=0,
		amplitude_threshold=0.02,
		corr_threshold=0.5,
		eta=0.55,
		min_turn_strength=0.0,
		normalize_for_phase=True,
	)
	assert res_norm.level == 3
