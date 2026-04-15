from __future__ import annotations

import pytest

from simulation.personality_coupling import personality_state_k_factor, resolve_personality_coupling
from simulation.run_simulation import SimConfig, simulate


def test_personality_state_k_factor_defaults_to_one_without_state_signal() -> None:
	assert personality_state_k_factor(state_dominance=None, beta_state_k=0.6) == pytest.approx(1.0)
	assert personality_state_k_factor(state_dominance=1.0, beta_state_k=0.0) == pytest.approx(1.0)


def test_personality_state_k_factor_is_stronger_near_center_than_edge() -> None:
	assert personality_state_k_factor(state_dominance=1.0 / 3.0, beta_state_k=0.6) == pytest.approx(1.0)
	edge_factor = personality_state_k_factor(state_dominance=1.0, beta_state_k=0.6)
	assert edge_factor == pytest.approx(0.6)
	assert edge_factor < 1.0


def test_resolve_personality_coupling_applies_linear_state_modulation_before_clamp() -> None:
	params = resolve_personality_coupling(
		{},
		mu_base=0.0,
		lambda_mu=0.0,
		k_base=0.06,
		lambda_k=0.0,
		state_dominance=1.0,
		beta_state_k=0.6,
	)
	assert params["mu"] == pytest.approx(0.0)
	assert params["k"] == pytest.approx(0.036)


def test_beta_state_k_requires_personality_coupled_mode() -> None:
	with pytest.raises(ValueError, match="beta_state_k"):
		simulate(
			SimConfig(
				n_players=12,
				n_rounds=8,
				seed=45,
				payoff_mode="matrix_ab",
				popularity_mode="sampled",
				gamma=0.1,
				epsilon=0.0,
				a=1.0,
				b=0.9,
				matrix_cross_coupling=0.2,
				init_bias=0.12,
				evolution_mode="sampled",
				payoff_lag=1,
				selection_strength=0.06,
				personality_coupling_beta_state_k=0.6,
			),
		)