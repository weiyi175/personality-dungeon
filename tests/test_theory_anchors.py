import math

from analysis.cycle_metrics import classify_cycle_level


def _core_control_constant(n: int = 3000):
	w = [1.0] * int(n)
	return {"aggressive": w, "defensive": w, "balanced": w}


def _core_synthetic_cycle(n: int = 4000, cycles: int = 20, r: float = 0.05):
	pa = []
	pd = []
	for i in range(int(n)):
		theta = 2.0 * math.pi * (float(i) / float(n)) * float(cycles)
		pa.append((1.0 / 3.0) + float(r) * math.cos(theta))
		pd.append((1.0 / 3.0) + float(r) * math.sin(theta))
	pb = [1.0 - a - d for a, d in zip(pa, pd)]
	return {"aggressive": pa, "defensive": pd, "balanced": pb}


def test_theory_anchor_control_constant_never_reaches_level2():
	props = _core_control_constant()
	for method in ("autocorr_threshold", "fft_power_ratio", "permutation_p"):
		res = classify_cycle_level(
			props,
			tail=2000,
			amplitude_threshold=0.02,
			corr_threshold=0.3,
			eta=0.8,
			stage2_method=method,
			stage2_prefilter=True,
			power_ratio_kappa=8.0,
			permutation_alpha=0.05,
			permutation_resamples=30,
			permutation_seed=123,
			stage3_method="turning",
			normalize_for_phase=True,
		)
		assert res.level < 2


def test_theory_anchor_synthetic_cycle_reaches_level3_for_all_stage2_methods():
	props = _core_synthetic_cycle()
	for method in ("autocorr_threshold", "fft_power_ratio", "permutation_p"):
		res = classify_cycle_level(
			props,
			tail=2000,
			amplitude_threshold=0.02,
			corr_threshold=0.3,
			eta=0.8,
			stage2_method=method,
			stage2_prefilter=True,
			power_ratio_kappa=8.0,
			permutation_alpha=0.05,
			permutation_resamples=80,
			permutation_seed=123,
			stage3_method="turning",
			normalize_for_phase=False,
		)
		assert res.level == 3
		assert res.stage1.passed is True
		assert res.stage2 is not None and res.stage2.passed is True
		assert res.stage3 is not None and res.stage3.passed is True
