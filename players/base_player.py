import random
from math import exp


DEFAULT_PERSONALITY_KEYS = [
	"impulsiveness",
	"caution",
	"greed",
	"optimism",
	"suspicion",
	"persistence",
	"randomness",
	"stability_seeking",
	"ambition",
	"patience",
	"curiosity",
	"fearfulness",
]


class BasePlayer:
	def __init__(self, strategy_space, *, rng=None, personality=None, state=None):
		self.strategy_space = list(strategy_space)
		self.strategy_weights = {s: 1.0 for s in self.strategy_space}
		self.strategy_biases = {s: 0.0 for s in self.strategy_space}
		self.utility = 0.0
		self.last_strategy = None
		self.last_reward = None
		self._rng = rng if rng is not None else random
		self.personality = {key: 0.0 for key in DEFAULT_PERSONALITY_KEYS}
		if personality:
			for key, value in personality.items():
				if key in self.personality:
					self.personality[key] = max(-1.0, min(1.0, float(value)))
		self.state = {
			"risk": 0.0,
			"stress": 0.0,
			"noise": 0.0,
			"risk_drift": 0.0,
			"health": 1.0,
			"intel": 0.0,
		}
		if state:
			for key, value in state.items():
				self.state[key] = float(value)

	def choose_strategy(self):
		strategies = list(self.strategy_weights.keys())
		weights = []
		for strategy in strategies:
			base_weight = max(1e-9, float(self.strategy_weights.get(strategy, 1.0)))
			bias = max(-6.0, min(6.0, float(self.strategy_biases.get(strategy, 0.0))))
			weights.append(base_weight * exp(bias))
		self.last_strategy = self._rng.choices(strategies, weights=weights, k=1)[0]
		return self.last_strategy

	def update_utility(self, reward):
		self.last_reward = float(reward)
		self.utility += float(reward)

	def update_weights(self, new_weights):
		# 複製一份避免所有玩家共用同一個 dict 參考造成意外 side-effect。
		self.strategy_weights = dict(new_weights)
		for strategy in self.strategy_space:
			self.strategy_biases.setdefault(strategy, 0.0)

	def apply_trait_deltas(self, deltas):
		for key, delta in dict(deltas).items():
			if key not in self.personality:
				continue
			current = float(self.personality[key])
			self.personality[key] = max(-1.0, min(1.0, current + float(delta)))

	def apply_state_deltas(self, deltas):
		for key, delta in dict(deltas).items():
			target_key = key[:-6] if str(key).endswith("_delta") else key
			current = float(self.state.get(target_key, 0.0))
			self.state[target_key] = current + float(delta)

	def apply_popularity_shift(self, shift):
		for strategy, delta in dict(shift).items():
			if strategy not in self.strategy_biases:
				continue
			current = float(self.strategy_biases[strategy])
			self.strategy_biases[strategy] = max(-2.0, min(2.0, current + float(delta)))

