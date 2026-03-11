"""Analysis metrics (MVP).

原則：
- 純計算、無 I/O，避免破壞分層。
- 由 simulation 呼叫且把結果記錄成時間序列。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional


def strategy_distribution(
	players: Iterable[object],
	strategy_space: List[str],
	*,
	attr: str = "last_strategy",
) -> Dict[str, float]:
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


def average_utility(players: Iterable[object], *, attr: str = "utility") -> float:
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


def average_reward(players: Iterable[object], *, attr: str = "last_reward") -> Optional[float]:
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

