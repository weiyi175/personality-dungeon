from __future__ import annotations

from typing import Iterable, Literal


class GameEngine:
	def __init__(
		self,
		players: Iterable[object],
		dungeon: object,
		*,
		popularity_mode: Literal["sampled", "expected"] = "sampled",
	):
		self.players = list(players)
		self.dungeon = dungeon
		self.popularity_mode = str(popularity_mode)
		if self.popularity_mode not in ("sampled", "expected"):
			raise ValueError("popularity_mode must be 'sampled' or 'expected'")

	def _expected_popularity(self) -> dict[str, float]:
		"""Compute expected next-round popularity from players' current strategy weights.

		This reduces finite-population sampling noise by using the mean sampling weights
		as a proxy for the expected strategy distribution.
		"""
		# Aggregate mean weight per strategy across players.
		totals: dict[str, float] = {}
		n = 0
		for p in self.players:
			w = getattr(p, "strategy_weights", None)
			if not isinstance(w, dict) or not w:
				continue
			for k, v in w.items():
				totals[k] = totals.get(k, 0.0) + float(v)
			n += 1
		if n <= 0 or not totals:
			return {}
		# Convert mean weights into a probability-like mass (will be normalized downstream).
		return {k: (v / float(n)) for k, v in totals.items()}

	def step(self) -> list[dict[str, object]]:
		chosen_strategies: list[str] = []
		step_records: list[dict[str, object]] = []

		# 玩家選策略並獲得 reward
		for player in self.players:
			strategy = player.choose_strategy()
			chosen_strategies.append(strategy)

			if hasattr(self.dungeon, "resolve_player_outcome"):
				outcome = self.dungeon.resolve_player_outcome(player, strategy)
				reward = outcome["reward"]
				event_result = outcome.get("event_result")
				step_records.append(
					{
						"strategy": strategy,
						"reward": float(reward),
						"base_reward": float(outcome.get("base_reward", reward)),
						"event_result": event_result,
					}
				)
			elif hasattr(self.dungeon, "evaluate_player"):
				reward = self.dungeon.evaluate_player(player, strategy)
				step_records.append({"strategy": strategy, "reward": float(reward), "base_reward": float(reward), "event_result": None})
			else:
				reward = self.dungeon.evaluate(strategy)
				step_records.append({"strategy": strategy, "reward": float(reward), "base_reward": float(reward), "event_result": None})
			player.update_utility(reward)

		# 更新地下城 popularity 統計
		if self.popularity_mode == "sampled":
			self.dungeon.update_popularity(chosen_strategies)
		else:  # expected
			self.dungeon.popularity = self._expected_popularity()

		return step_records

