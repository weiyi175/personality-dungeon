import pytest

from dungeon.dungeon_ai import DungeonAI


def test_matrix_ab_payoff_matches_expected_U_equals_Ax():
	# strategy order: aggressive, defensive, balanced
	cycle = ["aggressive", "defensive", "balanced"]
	d = DungeonAI(payoff_mode="matrix_ab", a=1.0, b=2.0, base_reward=10.0, strategy_cycle=cycle)

	# x = (0.2, 0.3, 0.5) encoded as counts
	d.popularity = {"aggressive": 2, "defensive": 3, "balanced": 5}

	# A = [[0, a, -b], [-b, 0, a], [a, -b, 0]]
	# U_A = a*x_D - b*x_B = 1*0.3 - 2*0.5 = -0.7
	assert d.evaluate("aggressive") == pytest.approx(9.3)
	# U_D = -b*x_A + a*x_B = -2*0.2 + 1*0.5 = 0.1
	assert d.evaluate("defensive") == pytest.approx(10.1)
	# U_B = a*x_A - b*x_D = 1*0.2 - 2*0.3 = -0.4
	assert d.evaluate("balanced") == pytest.approx(9.6)


def test_matrix_ab_cross_coupling_penalizes_a_d_and_rewards_b():
	cycle = ["aggressive", "defensive", "balanced"]
	d = DungeonAI(
		payoff_mode="matrix_ab",
		a=1.0,
		b=2.0,
		matrix_cross_coupling=0.5,
		base_reward=10.0,
		strategy_cycle=cycle,
	)

	# x = (0.2, 0.3, 0.5)
	d.popularity = {"aggressive": 2, "defensive": 3, "balanced": 5}

	# Base U = (-0.7, 0.1, -0.4)
	# Cross = (-0.5*0.3, -0.5*0.2, 0.5*(0.2+0.3)) = (-0.15, -0.1, 0.25)
	assert d.evaluate("aggressive") == pytest.approx(9.15)
	assert d.evaluate("defensive") == pytest.approx(10.0)
	assert d.evaluate("balanced") == pytest.approx(9.85)


def test_matrix_ab_requires_three_strategies():
	with pytest.raises(ValueError):
		DungeonAI(payoff_mode="matrix_ab", a=1.0, b=1.0, strategy_cycle=["a", "b"])  # len != 3


def test_unknown_payoff_mode_raises():
	with pytest.raises(ValueError):
		DungeonAI(payoff_mode="nope")


def test_count_cycle_still_works_with_gamma_and_epsilon():
	cycle = ["aggressive", "defensive", "balanced"]
	d = DungeonAI(payoff_mode="count_cycle", gamma=0.1, epsilon=0.2, base_reward=10.0, strategy_cycle=cycle)
	d.popularity = {"aggressive": 10, "defensive": 20, "balanced": 30}

	# For aggressive: penalty = 0.1*10=1.0; bonus=0.2*(n_prev - n_next) = 0.2*(balanced - defensive)=0.2*(30-20)=2.0
	assert d.evaluate("aggressive") == pytest.approx(11.0)


def test_matrix_ab_memory_kernel_averages_recent_popularity_prefix():
	cycle = ["aggressive", "defensive", "balanced"]
	d = DungeonAI(
		payoff_mode="matrix_ab",
		a=1.0,
		b=2.0,
		base_reward=10.0,
		strategy_cycle=cycle,
		memory_kernel=3,
	)

	d.set_popularity({"aggressive": 9, "defensive": 0, "balanced": 0})
	d.set_popularity({"aggressive": 0, "defensive": 9, "balanced": 0})

	# Prefix average over two available histories => x = (0.5, 0.5, 0.0)
	# U_A = a*x_D - b*x_B = 0.5 - 0 = 0.5
	assert d.evaluate("aggressive") == pytest.approx(10.5)
	# U_B = a*x_A - b*x_D = 0.5 - 1.0 = -0.5
	assert d.evaluate("balanced") == pytest.approx(9.5)


def test_threshold_ab_noop_matches_matrix_ab_payoff():
	cycle = ["aggressive", "defensive", "balanced"]
	base = DungeonAI(
		payoff_mode="matrix_ab",
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		base_reward=10.0,
		strategy_cycle=cycle,
		memory_kernel=3,
	)
	thr = DungeonAI(
		payoff_mode="threshold_ab",
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.2,
		threshold_theta=0.40,
		threshold_a_hi=1.0,
		threshold_b_hi=0.9,
		base_reward=10.0,
		strategy_cycle=cycle,
		memory_kernel=3,
	)

	for state in (
		{"aggressive": 9, "defensive": 0, "balanced": 0},
		{"aggressive": 0, "defensive": 9, "balanced": 0},
		{"aggressive": 1, "defensive": 1, "balanced": 8},
	):
		base.set_popularity(state)
		thr.set_popularity(state)

	for strategy in cycle:
		assert thr.evaluate(strategy) == pytest.approx(base.evaluate(strategy), abs=0.0)


def test_threshold_ab_uses_high_regime_when_q_ad_hits_theta():
	cycle = ["aggressive", "defensive", "balanced"]
	d = DungeonAI(
		payoff_mode="threshold_ab",
		a=1.0,
		b=0.9,
		threshold_theta=0.40,
		threshold_a_hi=1.2,
		threshold_b_hi=0.7,
		base_reward=10.0,
		strategy_cycle=cycle,
	)
	d.set_popularity({"aggressive": 2, "defensive": 2, "balanced": 6})

	# q_AD = 0.4, so tie-break must select the high regime.
	# U_A = a_hi*x_D - b_hi*x_B = 1.2*0.2 - 0.7*0.6 = -0.18
	assert d.evaluate("aggressive") == pytest.approx(9.82)


def test_threshold_ab_hysteresis_band_preserves_previous_high_regime():
	cycle = ["aggressive", "defensive", "balanced"]
	d = DungeonAI(
		payoff_mode="threshold_ab",
		a=1.0,
		b=0.9,
		threshold_theta=0.55,
		threshold_theta_low=0.45,
		threshold_theta_high=0.65,
		threshold_a_hi=1.2,
		threshold_b_hi=0.7,
		base_reward=10.0,
		strategy_cycle=cycle,
	)
	# First push into high regime.
	d.set_popularity({"aggressive": 4, "defensive": 3, "balanced": 3})
	assert d.evaluate("aggressive") == pytest.approx(10.15)
	# Then move inside the hysteresis band; regime should stay high.
	d.set_popularity({"aggressive": 3, "defensive": 2, "balanced": 5})
	# q_AD = 0.5 lies inside the band, so A_hi still applies.
	assert d.evaluate("aggressive") == pytest.approx(9.89)


def test_threshold_ab_hysteresis_band_initializes_to_low_inside_band():
	cycle = ["aggressive", "defensive", "balanced"]
	d = DungeonAI(
		payoff_mode="threshold_ab",
		a=1.0,
		b=0.9,
		threshold_theta=0.55,
		threshold_theta_low=0.45,
		threshold_theta_high=0.65,
		threshold_a_hi=1.2,
		threshold_b_hi=0.7,
		base_reward=10.0,
		strategy_cycle=cycle,
	)
	d.set_popularity({"aggressive": 3, "defensive": 2, "balanced": 5})
	# Initial state lies inside the band, so hysteresis must initialize to low.
	# U_A = a*x_D - b*x_B = 1.0*0.2 - 0.9*0.5 = -0.25
	assert d.evaluate("aggressive") == pytest.approx(9.75)


def test_threshold_ab_ad_product_trigger_selects_high_regime_on_overlap_peak():
	cycle = ["aggressive", "defensive", "balanced"]
	d = DungeonAI(
		payoff_mode="threshold_ab",
		a=1.0,
		b=0.9,
		threshold_theta=0.60,
		threshold_trigger="ad_product",
		threshold_a_hi=1.2,
		threshold_b_hi=0.7,
		base_reward=10.0,
		strategy_cycle=cycle,
	)
	d.set_popularity({"aggressive": 5, "defensive": 5, "balanced": 0})

	# ad_product = 4 * 0.5 * 0.5 = 1.0, so the high regime must activate.
	# U_A = a_hi*x_D - b_hi*x_B = 1.2*0.5 - 0.7*0.0 = 0.6
	assert d.evaluate("aggressive") == pytest.approx(10.6)