"""Minimal replicator dynamics (MVP).

設計重點
- 只依賴 players 的公開欄位（例如 last_strategy、last_reward）。
- 回傳的是「策略權重」(weights)，由 BasePlayer 決定如何抽樣。
- 不在這裡做任何 I/O。

為了穩定性，採用近似 replicator：
	w_s = exp(k * (u_s - u_bar))
可避免 1 + k * growth 變成負權重的問題。
"""

from __future__ import annotations

import random as _random_module
from math import atan2
from math import exp
from math import isfinite
from math import sqrt
from typing import Dict, Iterable, List


def _growth_vector_from_records(
	records: list[tuple[str, float]],
	strategy_space: List[str],
) -> Dict[str, float]:
	totals = {s: 0.0 for s in strategy_space}
	counts = {s: 0 for s in strategy_space}
	rewards: List[float] = []
	for strategy, reward in records:
		if strategy not in totals:
			continue
		totals[strategy] += float(reward)
		counts[strategy] += 1
		rewards.append(float(reward))
	avg_r = (sum(rewards) / len(rewards)) if rewards else 0.0
	return {
		s: float((totals[s] / counts[s]) if counts[s] > 0 else avg_r) - float(avg_r)
		for s in strategy_space
	}


def anchored_subgroup_weight_pull(
	strategy_space: List[str],
	*,
	adaptive_weights: Dict[str, float],
	anchor_weights: Dict[str, float],
	anchor_pull_strength: float,
) -> Dict[str, float]:
	"""Blend subgroup weights toward an anchor and renormalize to mean=1.

	When anchor_pull_strength=1, the subgroup becomes fully frozen at the anchor.
	When anchor_pull_strength=0, the subgroup follows the ordinary adaptive update.
	"""
	pull = float(anchor_pull_strength)
	if not isfinite(pull) or not (0.0 <= pull <= 1.0):
		raise ValueError("anchor_pull_strength must be finite and lie in [0,1]")
	if not strategy_space:
		return {}
	if pull <= 0.0:
		return {s: float(adaptive_weights.get(s, 1.0)) for s in strategy_space}
	if pull >= 1.0:
		return {s: float(anchor_weights.get(s, 1.0)) for s in strategy_space}
	blended = {
		s: ((1.0 - pull) * float(adaptive_weights.get(s, 1.0)) + pull * float(anchor_weights.get(s, 1.0)))
		for s in strategy_space
	}
	mean_w = sum(float(v) for v in blended.values()) / float(len(strategy_space))
	if mean_w <= 0.0:
		return {s: 1.0 for s in strategy_space}
	return {s: (float(blended[s]) / float(mean_w)) for s in strategy_space}


def anchored_subgroup_payoff_shift(
	strategy_space: List[str],
	*,
	a: float,
	b: float,
	anchor_simplex: Dict[str, float],
	adaptive_simplex: Dict[str, float],
) -> Dict[str, float]:
	"""Compute the H3.2 payoff shift induced by a fixed subgroup anchor.

	The shift is defined as A(a,b) @ (x_fix - x_ad), where A(a,b) is the same
	cyclic matrix used by matrix_ab without the extra cross-coupling term.
	"""
	if len(strategy_space) != 3:
		raise ValueError("anchored_subgroup_payoff_shift requires exactly 3 strategies")
	gap0 = float(anchor_simplex.get(strategy_space[0], 0.0)) - float(adaptive_simplex.get(strategy_space[0], 0.0))
	gap1 = float(anchor_simplex.get(strategy_space[1], 0.0)) - float(adaptive_simplex.get(strategy_space[1], 0.0))
	gap2 = float(anchor_simplex.get(strategy_space[2], 0.0)) - float(adaptive_simplex.get(strategy_space[2], 0.0))
	return {
		strategy_space[0]: float(float(a) * gap1 - float(b) * gap2),
		strategy_space[1]: float(-float(b) * gap0 + float(a) * gap2),
		strategy_space[2]: float(float(a) * gap0 - float(b) * gap1),
	}


def bidirectional_subgroup_payoff_shifts(
	strategy_space: List[str],
	*,
	a: float,
	b: float,
	fixed_simplex: Dict[str, float],
	adaptive_simplex: Dict[str, float],
	coupling_strength: float,
) -> tuple[Dict[str, float], Dict[str, float]]:
	"""Compute equal-and-opposite H3.5 payoff shifts for two subgroups."""
	strength = float(coupling_strength)
	if not isfinite(strength) or strength < 0.0:
		raise ValueError("coupling_strength must be finite and >= 0")
	base_shift = anchored_subgroup_payoff_shift(
		strategy_space,
		a=float(a),
		b=float(b),
		anchor_simplex=fixed_simplex,
		adaptive_simplex=adaptive_simplex,
	)
	adaptive_shift = {s: float(strength * float(base_shift[s])) for s in strategy_space}
	fixed_shift = {s: float(-adaptive_shift[s]) for s in strategy_space}
	return fixed_shift, adaptive_shift


def state_dependent_anchored_subgroup_payoff_shift(
	strategy_space: List[str],
	*,
	a: float,
	b: float,
	anchor_simplex: Dict[str, float],
	adaptive_simplex: Dict[str, float],
	base_coupling_strength: float,
	beta: float,
	theta: float,
	signal: str = "gap_norm",
) -> tuple[Dict[str, float], float, float]:
	"""Compute the H3.3B payoff shift with a sigmoid-gated state signal.

	The first H3.3B version only allows the subgroup-local signal
	z(t) = ||x_fix - x_ad||_2, with a sigmoid gate
	lambda(t) = lambda_0 * sigma(beta * (z - theta)).
	"""
	strength = float(base_coupling_strength)
	if not isfinite(strength) or strength < 0.0:
		raise ValueError("base_coupling_strength must be finite and >= 0")
	beta_f = float(beta)
	if not isfinite(beta_f) or beta_f <= 0.0:
		raise ValueError("beta must be finite and > 0")
	theta_f = float(theta)
	if not isfinite(theta_f) or theta_f < 0.0:
		raise ValueError("theta must be finite and >= 0")
	if str(signal) != "gap_norm":
		raise ValueError("signal must be 'gap_norm'")
	if len(strategy_space) != 3:
		raise ValueError("state_dependent_anchored_subgroup_payoff_shift requires exactly 3 strategies")
	gap0 = float(anchor_simplex.get(strategy_space[0], 0.0)) - float(adaptive_simplex.get(strategy_space[0], 0.0))
	gap1 = float(anchor_simplex.get(strategy_space[1], 0.0)) - float(adaptive_simplex.get(strategy_space[1], 0.0))
	gap2 = float(anchor_simplex.get(strategy_space[2], 0.0)) - float(adaptive_simplex.get(strategy_space[2], 0.0))
	gap_norm = float(sqrt(gap0 * gap0 + gap1 * gap1 + gap2 * gap2))
	gate = 1.0 / (1.0 + exp(-beta_f * (gap_norm - theta_f)))
	base_shift = anchored_subgroup_payoff_shift(
		strategy_space,
		a=float(a),
		b=float(b),
		anchor_simplex=anchor_simplex,
		adaptive_simplex=adaptive_simplex,
	)
	return (
		{s: float(strength * gate * float(base_shift[s])) for s in strategy_space},
		float(gap_norm),
		float(gate),
	)


def _resolve_strategy_selection_strengths(
	strategy_space: List[str],
	*,
	selection_strength: float,
	strategy_selection_strengths: List[float] | tuple[float, ...] | None,
) -> Dict[str, float]:
	base = float(selection_strength)
	if base < 0:
		raise ValueError("selection_strength must be >= 0")
	if strategy_selection_strengths is None:
		return {s: base for s in strategy_space}
	if len(strategy_selection_strengths) != len(strategy_space):
		raise ValueError("strategy_selection_strengths must match strategy_space length")
	out: Dict[str, float] = {}
	for s, k in zip(strategy_space, strategy_selection_strengths, strict=True):
		kk = float(k)
		if not isfinite(kk) or kk < 0.0:
			raise ValueError("strategy_selection_strengths must contain finite values >= 0")
		out[s] = kk
	return out


def _vector_l2_norm(values: Iterable[float]) -> float:
	return float(sqrt(sum(float(value) * float(value) for value in values)))


def _ordered_simplex_vector(
	simplex: Dict[str, float] | Iterable[float],
	strategy_space: List[str],
) -> list[float]:
	if len(strategy_space) != 3:
		raise ValueError("tangential drift currently requires exactly 3 strategies")
	if isinstance(simplex, dict):
		vector = [float(simplex.get(strategy, 0.0)) for strategy in strategy_space]
	else:
		vector = [float(value) for value in simplex]
	if len(vector) != len(strategy_space):
		raise ValueError("simplex vector length must match strategy_space length")
	if not all(isfinite(value) for value in vector):
		raise ValueError("simplex vector must contain finite values")
	return vector


def _sampled_simplex_vector(
	players: Iterable[object],
	strategy_space: List[str],
) -> list[float]:
	counts = {strategy: 0.0 for strategy in strategy_space}
	total = 0.0
	for player in players:
		strategy = getattr(player, "last_strategy", None)
		if strategy not in counts:
			continue
		counts[str(strategy)] += 1.0
		total += 1.0
	if total <= 0.0:
		return [(1.0 / float(len(strategy_space))) for _ in strategy_space]
	return [float(counts[strategy]) / float(total) for strategy in strategy_space]


def tangential_drift_vector(
	simplex: Dict[str, float] | Iterable[float],
	strategy_space: List[str],
	delta: float,
) -> Dict[str, float]:
	"""Return a delta-normalized tangential drift vector on the 3-simplex."""
	delta_f = float(delta)
	if not isfinite(delta_f) or delta_f < 0.0:
		raise ValueError("delta must be finite and >= 0")
	if delta_f == 0.0:
		return {strategy: 0.0 for strategy in strategy_space}
	x = _ordered_simplex_vector(simplex, strategy_space)
	center = 1.0 / 3.0
	r = [float(value) - center for value in x]
	if _vector_l2_norm(r) < 1e-12:
		return {strategy: 0.0 for strategy in strategy_space}
	tau = [
		float(-r[1] + r[2]),
		float(r[0] - r[2]),
		float(-r[0] + r[1]),
	]
	tau_norm = _vector_l2_norm(tau)
	if tau_norm < 1e-12:
		return {strategy: 0.0 for strategy in strategy_space}
	scale = float(delta_f / tau_norm)
	return {
		strategy: float(scale * tau_component)
		for strategy, tau_component in zip(strategy_space, tau, strict=True)
	}


def apply_tangential_drift(
	growth_vector: Dict[str, float],
	strategy_space: List[str],
	*,
	simplex: Dict[str, float] | Iterable[float],
	delta: float,
) -> tuple[Dict[str, float], Dict[str, float]]:
	"""Add tangential drift to a growth vector and project back to zero-mean."""
	base_growth = {
		strategy: float(growth_vector.get(strategy, 0.0))
		for strategy in strategy_space
	}
	drift = tangential_drift_vector(simplex, strategy_space, float(delta))
	drift_norm = _vector_l2_norm(drift.values())
	growth_norm = _vector_l2_norm(base_growth.values())
	if float(delta) == 0.0 or drift_norm <= 0.0:
		ratio = 0.0 if growth_norm <= 1e-12 else float(drift_norm / growth_norm)
		return base_growth, {
			"drift_norm": float(drift_norm),
			"growth_norm": float(growth_norm),
			"effective_delta_growth_ratio": float(ratio),
		}
	combined = {
		strategy: float(base_growth[strategy] + drift[strategy])
		for strategy in strategy_space
	}
	mean_combined = sum(float(value) for value in combined.values()) / float(len(strategy_space))
	drifted = {
		strategy: float(combined[strategy] - mean_combined)
		for strategy in strategy_space
	}
	ratio = float("inf") if growth_norm <= 1e-12 else float(drift_norm / growth_norm)
	return drifted, {
		"drift_norm": float(drift_norm),
		"growth_norm": float(growth_norm),
		"effective_delta_growth_ratio": float(ratio),
	}


def deterministic_replicator_step(
	current_weights: Dict[str, float],
	strategy_space: List[str],
	*,
	payoff_vector: Dict[str, float],
	selection_strength: float = 0.05,
	strategy_selection_strengths: List[float] | tuple[float, ...] | None = None,
	tangential_drift_simplex: Dict[str, float] | Iterable[float] | None = None,
	tangential_drift_delta: float = 0.0,
) -> Dict[str, float]:
	"""Update one weight vector using deterministic expected payoffs."""

	strengths = _resolve_strategy_selection_strengths(
		strategy_space,
		selection_strength=float(selection_strength),
		strategy_selection_strengths=strategy_selection_strengths,
	)
	weights = {s: max(1e-12, float(current_weights.get(s, 1.0))) for s in strategy_space}
	total = sum(float(weights[s]) for s in strategy_space)
	if total <= 0.0:
		simplex = {s: (1.0 / float(len(strategy_space))) for s in strategy_space} if strategy_space else {}
	else:
		simplex = {s: (float(weights[s]) / float(total)) for s in strategy_space}
	u_bar = sum(float(simplex[s]) * float(payoff_vector.get(s, 0.0)) for s in strategy_space)
	growth = {
		s: float(payoff_vector.get(s, 0.0)) - float(u_bar)
		for s in strategy_space
	}
	if float(tangential_drift_delta) > 0.0:
		growth, _diagnostics = apply_tangential_drift(
			growth,
			strategy_space,
			simplex=(tangential_drift_simplex if tangential_drift_simplex is not None else simplex),
			delta=float(tangential_drift_delta),
		)
	out = {
		s: max(1e-6, float(weights[s]) * exp(float(strengths[s]) * float(growth[s])))
		for s in strategy_space
	}
	mean_w = sum(float(out[s]) for s in strategy_space) / float(len(strategy_space)) if strategy_space else 1.0
	if mean_w <= 0.0:
		return {s: 1.0 for s in strategy_space}
	return {s: (float(out[s]) / float(mean_w)) for s in strategy_space}


def inertial_deterministic_replicator_step(
	current_weights: Dict[str, float],
	strategy_space: List[str],
	*,
	payoff_vector: Dict[str, float],
	previous_velocity: Dict[str, float] | None = None,
	inertia: float = 0.0,
	selection_strength: float = 0.05,
	strategy_selection_strengths: List[float] | tuple[float, ...] | None = None,
) -> tuple[Dict[str, float], Dict[str, float]]:
	"""Update one weight vector using deterministic payoffs with one-step inertia."""

	momentum = float(inertia)
	if not isfinite(momentum) or not (0.0 <= momentum < 1.0):
		raise ValueError("inertia must be finite and lie in [0,1)")
	strengths = _resolve_strategy_selection_strengths(
		strategy_space,
		selection_strength=float(selection_strength),
		strategy_selection_strengths=strategy_selection_strengths,
	)
	weights = {s: max(1e-12, float(current_weights.get(s, 1.0))) for s in strategy_space}
	total = sum(float(weights[s]) for s in strategy_space)
	if total <= 0.0:
		simplex = {s: (1.0 / float(len(strategy_space))) for s in strategy_space} if strategy_space else {}
	else:
		simplex = {s: (float(weights[s]) / float(total)) for s in strategy_space}
	u_bar = sum(float(simplex[s]) * float(payoff_vector.get(s, 0.0)) for s in strategy_space)
	prev = previous_velocity or {}
	velocity = {
		s: (
			float(momentum) * float(prev.get(s, 0.0))
			+ float(strengths[s]) * (float(payoff_vector.get(s, 0.0)) - float(u_bar))
		)
		for s in strategy_space
	}
	out = {
		s: max(1e-6, float(weights[s]) * exp(float(velocity[s])))
		for s in strategy_space
	}
	mean_w = sum(float(out[s]) for s in strategy_space) / float(len(strategy_space)) if strategy_space else 1.0
	if mean_w <= 0.0:
		return ({s: 1.0 for s in strategy_space}, {s: 0.0 for s in strategy_space})
	return (
		{s: (float(out[s]) / float(mean_w)) for s in strategy_space},
		{s: float(velocity[s]) for s in strategy_space},
	)


def inertial_sampled_replicator_step(
	players: Iterable[object],
	strategy_space: List[str],
	*,
	previous_velocity: Dict[str, float] | None = None,
	inertia: float = 0.0,
	selection_strength: float = 0.05,
	strategy_selection_strengths: List[float] | tuple[float, ...] | None = None,
) -> tuple[Dict[str, float], Dict[str, float]]:
	"""Compute sampled replicator weights with one-step inertia on the sampled operator."""

	momentum = float(inertia)
	if not isfinite(momentum) or not (0.0 <= momentum < 1.0):
		raise ValueError("inertia must be finite and lie in [0,1)")
	strengths = _resolve_strategy_selection_strengths(
		strategy_space,
		selection_strength=float(selection_strength),
		strategy_selection_strengths=strategy_selection_strengths,
	)

	growth = sampled_growth_vector(players, strategy_space)
	prev = previous_velocity or {}
	velocity = {}
	weights = {}
	for s in strategy_space:
		velocity[s] = float(momentum) * float(prev.get(s, 0.0)) + float(strengths[s]) * float(growth[s])
		weights[s] = max(1e-6, exp(float(velocity[s])))

	mean_w = sum(weights.values()) / len(weights) if weights else 1.0
	if mean_w <= 0.0:
		return ({s: 1.0 for s in strategy_space}, {s: 0.0 for s in strategy_space})
	return (
		{s: (float(weights[s]) / float(mean_w)) for s in strategy_space},
		{s: float(velocity[s]) for s in strategy_space},
	)


def sampled_growth_vector(
	players: Iterable[object],
	strategy_space: List[str],
) -> Dict[str, float]:
	"""Compute sampled per-strategy growth terms against the population-average reward."""
	records: list[tuple[str, float]] = []
	for p in players:
		s = getattr(p, "last_strategy", None)
		r = getattr(p, "last_reward", None)
		if s is None or r is None:
			continue
		records.append((str(s), float(r)))
	return _growth_vector_from_records(records, strategy_space)


def stratified_growth_vector(
	players: Iterable[object],
	strategy_space: List[str],
	*,
	strata_key: str = "stratum",
	n_strata: int = 1,
) -> Dict[str, float]:
	"""Compute sampled growth after first aggregating rewards within fixed strata."""
	strata = int(n_strata)
	if strata <= 0:
		raise ValueError("n_strata must be >= 1")
	player_list = list(players)
	if strata == 1:
		return sampled_growth_vector(player_list, strategy_space)
	buckets: list[list[tuple[str, float]]] = [[] for _ in range(strata)]
	for player in player_list:
		strategy = getattr(player, "last_strategy", None)
		reward = getattr(player, "last_reward", None)
		if strategy is None or reward is None:
			continue
		bucket = int(getattr(player, strata_key, 0)) % strata
		buckets[bucket].append((str(strategy), float(reward)))
	total_records = sum(len(bucket) for bucket in buckets)
	if total_records <= 0:
		return {s: 0.0 for s in strategy_space}
	combined = {s: 0.0 for s in strategy_space}
	for bucket in buckets:
		if not bucket:
			continue
		bucket_growth = _growth_vector_from_records(bucket, strategy_space)
		weight = float(len(bucket)) / float(total_records)
		for strategy in strategy_space:
			combined[strategy] += float(weight) * float(bucket_growth[strategy])
	return combined


def inertial_growth_step(
	current_weights: Dict[str, float],
	strategy_space: List[str],
	*,
	growth_vector: Dict[str, float],
	previous_velocity: Dict[str, float] | None = None,
	inertia: float = 0.0,
	selection_strength: float = 0.05,
	strategy_selection_strengths: List[float] | tuple[float, ...] | None = None,
) -> tuple[Dict[str, float], Dict[str, float]]:
	"""Update one player's weights from a shared sampled growth vector."""
	momentum = float(inertia)
	if not isfinite(momentum) or not (0.0 <= momentum < 1.0):
		raise ValueError("inertia must be finite and lie in [0,1)")
	strengths = _resolve_strategy_selection_strengths(
		strategy_space,
		selection_strength=float(selection_strength),
		strategy_selection_strengths=strategy_selection_strengths,
	)
	weights = {s: max(1e-12, float(current_weights.get(s, 1.0))) for s in strategy_space}
	prev = previous_velocity or {}
	velocity = {
		s: float(momentum) * float(prev.get(s, 0.0)) + float(strengths[s]) * float(growth_vector.get(s, 0.0))
		for s in strategy_space
	}
	out = {s: max(1e-6, float(weights[s]) * exp(float(velocity[s]))) for s in strategy_space}
	mean_w = sum(float(out[s]) for s in strategy_space) / float(len(strategy_space)) if strategy_space else 1.0
	if mean_w <= 0.0:
		return ({s: 1.0 for s in strategy_space}, {s: 0.0 for s in strategy_space})
	return (
		{s: (float(out[s]) / float(mean_w)) for s in strategy_space},
		{s: float(velocity[s]) for s in strategy_space},
	)


def _tangential_projection(
	growth: Dict[str, float],
	simplex: Dict[str, float] | Iterable[float],
	strategy_space: List[str],
	alpha: float,
) -> tuple[Dict[str, float], Dict[str, float]]:
	"""Decompose *growth* into radial + tangential on the 3-simplex and amplify tangential.

	Returns (modified_growth, diagnostics).
	"""
	alpha_f = float(alpha)
	x = _ordered_simplex_vector(simplex, strategy_space)
	g = [float(growth.get(s, 0.0)) for s in strategy_space]
	center = 1.0 / 3.0
	r = [float(xi) - center for xi in x]
	r_dot_r = sum(ri * ri for ri in r)

	if r_dot_r < 1e-24:
		# At centroid: tangential decomposition undefined → pass through unchanged
		g_norm = _vector_l2_norm(g)
		return dict(zip(strategy_space, g)), {
			"radial_norm": 0.0,
			"tangential_norm": float(g_norm),
			"alpha_effective": 0.0,
			"tangential_ratio": 1.0 if g_norm > 1e-12 else 0.0,
			"growth_angle_rad": 0.0,
		}

	# Project g onto r to get radial component
	g_dot_r = sum(gi * ri for gi, ri in zip(g, r))
	scale = float(g_dot_r / r_dot_r)
	g_r = [scale * ri for ri in r]
	g_tau = [gi - gri for gi, gri in zip(g, g_r)]

	radial_norm = _vector_l2_norm(g_r)
	tangential_norm = _vector_l2_norm(g_tau)

	# Modified growth: g' = g_r + (1 + alpha) * g_tau
	g_prime = [gri + (1.0 + alpha_f) * gtau_i for gri, gtau_i in zip(g_r, g_tau)]

	# Re-center to maintain zero-mean (numerical hygiene)
	mean_g = sum(g_prime) / float(len(strategy_space))
	g_prime = [gi - mean_g for gi in g_prime]

	g_prime_norm = _vector_l2_norm(g_prime)
	angle = atan2(tangential_norm, radial_norm) if (radial_norm > 1e-12 or tangential_norm > 1e-12) else 0.0

	total = radial_norm + tangential_norm
	t_ratio = float(tangential_norm / total) if total > 1e-12 else 0.0

	return dict(zip(strategy_space, g_prime)), {
		"radial_norm": float(radial_norm),
		"tangential_norm": float(tangential_norm),
		"alpha_effective": float(alpha_f),
		"tangential_ratio": float(t_ratio),
		"growth_angle_rad": float(angle),
	}


def tangential_projection_replicator_step(
	players: Iterable[object],
	strategy_space: List[str],
	selection_strength: float = 0.05,
	*,
	tangential_alpha: float = 0.0,
	sampled_growth_n_strata: int = 1,
	sampled_growth_strata_key: str = "stratum",
) -> tuple[Dict[str, float], Dict[str, float]]:
	"""B1: Replicator step with tangential amplification on the 3-simplex.

	When tangential_alpha=0, this is bit-identical to replicator_step().

	Returns (weights, diagnostics).
	"""
	alpha_f = float(tangential_alpha)
	if not isfinite(alpha_f) or alpha_f < 0.0:
		raise ValueError("tangential_alpha must be finite and >= 0")

	player_list = list(players)

	growth = stratified_growth_vector(
		player_list,
		strategy_space,
		strata_key=str(sampled_growth_strata_key),
		n_strata=int(sampled_growth_n_strata),
	)
	if not growth:
		return ({s: 1.0 for s in strategy_space}, {
			"radial_norm": 0.0, "tangential_norm": 0.0,
			"alpha_effective": 0.0, "tangential_ratio": 0.0,
			"growth_angle_rad": 0.0,
		})

	if alpha_f == 0.0:
		# Exact pass-through to standard replicator (bit-identical guarantee)
		strengths = _resolve_strategy_selection_strengths(
			strategy_space,
			selection_strength=float(selection_strength),
			strategy_selection_strengths=None,
		)
		weights: Dict[str, float] = {}
		for s in strategy_space:
			w = exp(float(strengths[s]) * float(growth.get(s, 0.0)))
			weights[s] = max(1e-6, w)
		mean_w = sum(weights.values()) / len(weights)
		if mean_w <= 0:
			weights = {s: 1.0 for s in strategy_space}
		else:
			weights = {s: (w / mean_w) for s, w in weights.items()}

		# Compute diagnostics even at alpha=0 for consistent schema
		simplex = _sampled_simplex_vector(player_list, strategy_space)
		_, diag = _tangential_projection(growth, simplex, strategy_space, 0.0)
		return weights, diag

	# alpha > 0: apply tangential projection
	simplex = _sampled_simplex_vector(player_list, strategy_space)
	projected_growth, diag = _tangential_projection(growth, simplex, strategy_space, alpha_f)

	strengths = _resolve_strategy_selection_strengths(
		strategy_space,
		selection_strength=float(selection_strength),
		strategy_selection_strengths=None,
	)
	weights = {}
	for s in strategy_space:
		w = exp(float(strengths[s]) * float(projected_growth.get(s, 0.0)))
		weights[s] = max(1e-6, w)
	mean_w = sum(weights.values()) / len(weights)
	if mean_w <= 0:
		return ({s: 1.0 for s in strategy_space}, diag)
	return ({s: (w / mean_w) for s, w in weights.items()}, diag)


def async_replicator_step(
	players: list[object],
	strategy_space: List[str],
	*,
	selection_strength: float = 0.05,
	async_update_fraction: float = 1.0,
	seed: int = 0,
	round_index: int = 0,
) -> tuple[Dict[str, float], Dict[str, float]]:
	"""A1: Stochastic asynchronous replicator step.

	Each player independently updates with probability *async_update_fraction*.
	When async_update_fraction >= 1.0 and all players share the same weights,
	this is bit-identical to synchronous :func:`replicator_step`.

	Unlike :func:`replicator_step` which returns shared weights applied later,
	this function updates selected players **in-place** (via ``update_weights``).
	The caller should skip the usual weight-application loop for these players.

	Returns ``(mean_weights, diagnostics)``.
	"""
	frac = float(async_update_fraction)
	if not isfinite(frac) or frac < 0.0 or frac > 1.0:
		raise ValueError("async_update_fraction must be finite and in [0, 1]")
	k = float(selection_strength)

	player_list = list(players)
	n_total = len(player_list)

	# --- shared growth vector (same as synchronous) ---
	growth = sampled_growth_vector(player_list, strategy_space)
	if not growth:
		mean_w = {s: 1.0 for s in strategy_space}
		return mean_w, {"n_updated": 0, "fraction_updated": 0.0, "weight_dispersion": 0.0}

	# --- compute new weights (same formula as replicator_step) ---
	new_weights: Dict[str, float] = {}
	for s in strategy_space:
		w = exp(k * float(growth.get(s, 0.0)))
		new_weights[s] = max(1e-6, w)
	mean_nw = sum(new_weights.values()) / float(len(new_weights)) if new_weights else 1.0
	if mean_nw <= 0.0:
		new_weights = {s: 1.0 for s in strategy_space}
	else:
		new_weights = {s: (float(new_weights[s]) / float(mean_nw)) for s in strategy_space}

	# --- select which players update ---
	if frac >= 1.0:
		# All update → identical to sync
		for pl in player_list:
			pl.update_weights(new_weights)
		n_updated = n_total
	elif frac <= 0.0:
		# None update
		n_updated = 0
	else:
		rng = _random_module.Random(int(seed) * 1000003 + int(round_index))
		n_updated = 0
		for pl in player_list:
			if rng.random() < frac:
				pl.update_weights(new_weights)
				n_updated += 1

	# --- compute mean weights across all players ---
	if n_total == 0:
		mean_w = {s: 1.0 for s in strategy_space}
	else:
		mean_w = {s: 0.0 for s in strategy_space}
		for pl in player_list:
			pw = getattr(pl, "strategy_weights", {})
			for s in strategy_space:
				mean_w[s] += float(pw.get(s, 1.0))
		mean_w = {s: (float(mean_w[s]) / float(n_total)) for s in strategy_space}

	# --- diagnostics: weight dispersion ---
	weight_dispersion = 0.0
	if n_total > 1:
		per_strategy_std: list[float] = []
		for s in strategy_space:
			vals = [float(getattr(pl, "strategy_weights", {}).get(s, 1.0)) for pl in player_list]
			m = sum(vals) / float(len(vals))
			var = sum((v - m) ** 2 for v in vals) / float(len(vals))
			per_strategy_std.append(var ** 0.5)
		weight_dispersion = sum(per_strategy_std) / float(len(per_strategy_std))

	diagnostics = {
		"n_updated": int(n_updated),
		"fraction_updated": float(n_updated) / float(n_total) if n_total > 0 else 0.0,
		"weight_dispersion": float(weight_dispersion),
	}
	return mean_w, diagnostics


def replicator_step(
	players: Iterable[object],
	strategy_space: List[str],
	selection_strength: float = 0.05,
	strategy_selection_strengths: List[float] | tuple[float, ...] | None = None,
	*,
	sampled_growth_n_strata: int = 1,
	sampled_growth_strata_key: str = "stratum",
	tangential_drift_simplex: Dict[str, float] | Iterable[float] | None = None,
	tangential_drift_delta: float = 0.0,
) -> Dict[str, float]:
	"""Compute new strategy weights based on per-strategy average reward.

	Parameters
	- players: iterable of objects with `.last_strategy` and `.last_reward`.
	- strategy_space: list of strategy names.
	- selection_strength: higher => stronger selection pressure.

	Returns
	- dict: strategy -> positive weight (normalized to mean=1).
	"""

	strengths = _resolve_strategy_selection_strengths(
		strategy_space,
		selection_strength=float(selection_strength),
		strategy_selection_strengths=strategy_selection_strengths,
	)
	player_list = list(players)

	growth = stratified_growth_vector(
		player_list,
		strategy_space,
		strata_key=str(sampled_growth_strata_key),
		n_strata=int(sampled_growth_n_strata),
	)
	if not growth:
		return {s: 1.0 for s in strategy_space}
	if float(tangential_drift_delta) > 0.0:
		growth, _diagnostics = apply_tangential_drift(
			growth,
			strategy_space,
			simplex=(
				tangential_drift_simplex
				if tangential_drift_simplex is not None
				else _sampled_simplex_vector(player_list, strategy_space)
			),
			delta=float(tangential_drift_delta),
		)

	weights: Dict[str, float] = {}
	for s in strategy_space:
		w = exp(float(strengths[s]) * float(growth.get(s, 0.0)))
		weights[s] = max(1e-6, w)

	mean_w = sum(weights.values()) / len(weights)
	if mean_w <= 0:
		return {s: 1.0 for s in strategy_space}

	return {s: (w / mean_w) for s, w in weights.items()}

