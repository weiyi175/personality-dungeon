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