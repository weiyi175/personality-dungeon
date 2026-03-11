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

from math import exp
from typing import Dict, Iterable, List


def replicator_step(
	players: Iterable[object],
	strategy_space: List[str],
	selection_strength: float = 0.05,
) -> Dict[str, float]:
	"""Compute new strategy weights based on per-strategy average reward.

	Parameters
	- players: iterable of objects with `.last_strategy` and `.last_reward`.
	- strategy_space: list of strategy names.
	- selection_strength: higher => stronger selection pressure.

	Returns
	- dict: strategy -> positive weight (normalized to mean=1).
	"""

	if selection_strength < 0:
		raise ValueError("selection_strength must be >= 0")

	totals = {s: 0.0 for s in strategy_space}
	counts = {s: 0 for s in strategy_space}

	rewards: List[float] = []
	for p in players:
		s = getattr(p, "last_strategy", None)
		r = getattr(p, "last_reward", None)
		if s is None or r is None:
			continue
		if s not in totals:
			continue
		totals[s] += float(r)
		counts[s] += 1
		rewards.append(float(r))

	if not rewards:
		return {s: 1.0 for s in strategy_space}

	avg_r = sum(rewards) / len(rewards)

	weights: Dict[str, float] = {}
	for s in strategy_space:
		strat_avg = totals[s] / counts[s] if counts[s] > 0 else avg_r
		growth = strat_avg - avg_r
		w = exp(selection_strength * growth)
		weights[s] = max(1e-6, w)

	mean_w = sum(weights.values()) / len(weights)
	if mean_w <= 0:
		return {s: 1.0 for s in strategy_space}

	return {s: (w / mean_w) for s, w in weights.items()}

