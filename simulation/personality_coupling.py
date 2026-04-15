from __future__ import annotations

from typing import Iterable, Mapping


def clamp(value: float, lower: float, upper: float) -> float:
	return max(lower, min(upper, value))


def personality_signal_mu(personality: Mapping[str, float]) -> float:
	stability = float(personality.get("stability_seeking", 0.0))
	patience = float(personality.get("patience", 0.0))
	impulsiveness = float(personality.get("impulsiveness", 0.0))
	return 0.5 * (stability + patience) - 0.5 * impulsiveness


def personality_signal_k(personality: Mapping[str, float]) -> float:
	ambition = float(personality.get("ambition", 0.0))
	greed = float(personality.get("greed", 0.0))
	caution = float(personality.get("caution", 0.0))
	fearfulness = float(personality.get("fearfulness", 0.0))
	return 0.5 * (ambition + greed) - 0.5 * (caution + fearfulness)


def personality_state_k_factor(*, state_dominance: float | None, beta_state_k: float) -> float:
	if state_dominance is None or float(beta_state_k) <= 0.0:
		return 1.0
	dominance = clamp(float(state_dominance), 1.0 / 3.0, 1.0)
	return 1.0 + float(beta_state_k) * (1.0 / (3.0 * dominance) - 1.0)


def resolve_personality_coupling(
	personality: Mapping[str, float],
	*,
	mu_base: float,
	lambda_mu: float,
	k_base: float,
	lambda_k: float,
	state_dominance: float | None = None,
	beta_state_k: float = 0.0,
) -> dict[str, float]:
	signal_mu = float(personality_signal_mu(personality))
	signal_k = float(personality_signal_k(personality))
	mu_value = clamp(float(mu_base) + float(lambda_mu) * signal_mu, 0.0, 0.60)
	k_raw = float(k_base) * (1.0 + float(lambda_k) * signal_k)
	k_raw *= personality_state_k_factor(
		state_dominance=state_dominance,
		beta_state_k=float(beta_state_k),
	)
	k_value = clamp(k_raw, 0.03, 0.09)
	return {
		"signal_mu": signal_mu,
		"signal_k": signal_k,
		"mu": mu_value,
		"k": k_value,
	}


def summarize_personality_coupling(
	personalities: Iterable[Mapping[str, float]],
	*,
	mu_base: float,
	lambda_mu: float,
	k_base: float,
	lambda_k: float,
	state_dominance: float | None = None,
	beta_state_k: float = 0.0,
) -> dict[str, float]:
	resolved = [
		resolve_personality_coupling(
			personality,
			mu_base=float(mu_base),
			lambda_mu=float(lambda_mu),
			k_base=float(k_base),
			lambda_k=float(lambda_k),
			state_dominance=state_dominance,
			beta_state_k=float(beta_state_k),
		)
		for personality in personalities
	]
	if not resolved:
		return {
			"signal_mu_min": 0.0,
			"signal_mu_max": 0.0,
			"signal_mu_mean": 0.0,
			"signal_k_min": 0.0,
			"signal_k_max": 0.0,
			"signal_k_mean": 0.0,
			"mu_min": 0.0,
			"mu_max": 0.0,
			"mu_mean": 0.0,
			"k_min": 0.0,
			"k_max": 0.0,
			"k_mean": 0.0,
		}

	def _triple(key: str) -> dict[str, float]:
		values = [float(item[key]) for item in resolved]
		return {
			f"{key}_min": min(values),
			f"{key}_max": max(values),
			f"{key}_mean": sum(values) / float(len(values)),
		}

	stats = {}
	stats.update(_triple("signal_mu"))
	stats.update(_triple("signal_k"))
	stats.update(_triple("mu"))
	stats.update(_triple("k"))
	return stats