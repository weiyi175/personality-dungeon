import random


class BasePlayer:
	def __init__(self, strategy_space, *, rng=None):
		self.strategy_space = list(strategy_space)
		self.strategy_weights = {s: 1.0 for s in self.strategy_space}
		self.utility = 0.0
		self.last_strategy = None
		self.last_reward = None
		self._rng = rng if rng is not None else random

	def choose_strategy(self):
		strategies = list(self.strategy_weights.keys())
		weights = list(self.strategy_weights.values())
		self.last_strategy = self._rng.choices(strategies, weights=weights, k=1)[0]
		return self.last_strategy

	def update_utility(self, reward):
		self.last_reward = float(reward)
		self.utility += float(reward)

	def update_weights(self, new_weights):
		# 複製一份避免所有玩家共用同一個 dict 參考造成意外 side-effect。
		self.strategy_weights = dict(new_weights)

