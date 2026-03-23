import random


class DungeonAI:
	"""Dungeon payoff with popularity penalty and optional cyclic cross-payoff.

	Base (MVP):
		reward = base_reward - gamma * n_i
	
	Cyclic advantage (research):
		reward = base_reward - gamma * n_i + epsilon * (n_prev - n_next)
	
	Where counts n_* are from the previous round's popularity histogram.
	"""

	def __init__(
		self,
		*,
		payoff_mode: str = "count_cycle",
		gamma: float = 0.2,
		epsilon: float = 0.0,
		a: float = 0.0,
		b: float = 0.0,
		matrix_cross_coupling: float = 0.0,
		strategy_cycle: list[str] | None = None,
		base_reward: float = 10.0,
		event_loader: object | None = None,
		event_rng: random.Random | None = None,
	):
		self.payoff_mode = str(payoff_mode)
		self.gamma = float(gamma)
		self.epsilon = float(epsilon)
		self.a = float(a)
		self.b = float(b)
		self.matrix_cross_coupling = float(matrix_cross_coupling)
		self.base_reward = float(base_reward)
		self.event_loader = event_loader
		self._event_rng = event_rng if event_rng is not None else random.Random()
		# cycle defines (prev, next) neighbors in a directed ring.
		self.strategy_cycle = list(strategy_cycle) if strategy_cycle is not None else []
		self.popularity: dict[str, int] = {}

		allowed = {"count_cycle", "matrix_ab"}
		if self.payoff_mode not in allowed:
			raise ValueError(f"Unknown payoff_mode: {self.payoff_mode!r}. Allowed: {sorted(allowed)}")
		if self.payoff_mode == "matrix_ab" and len(self.strategy_cycle) != 3:
			raise ValueError("payoff_mode='matrix_ab' requires exactly 3 strategies in strategy_cycle")

	def _proportions(self) -> dict[str, float]:
		total = sum(float(v) for v in self.popularity.values())
		if total <= 0:
			return {s: 0.0 for s in self.strategy_cycle}
		return {s: (float(self.popularity.get(s, 0)) / float(total)) for s in self.strategy_cycle}

	def evaluate(self, strategy: str) -> float:
		if self.payoff_mode == "matrix_ab":
			# Research mode: expected payoff U = A x, with
			# A = [[0, a, -b],
			#      [-b, 0, a],
			#      [a, -b, 0]]
			# Optional cross coupling c_AD penalizes aggressive-defensive coexistence
			# and transfers that pressure to balanced.
			# where x is last-round strategy proportions in the order of strategy_cycle.
			if strategy not in self.strategy_cycle:
				return float(self.base_reward)
			idx = self.strategy_cycle.index(strategy)
			p = self._proportions()
			x = [p[s] for s in self.strategy_cycle]
			A = (
				(0.0, self.a, -self.b),
				(-self.b, 0.0, self.a),
				(self.a, -self.b, 0.0),
			)
			u = A[idx][0] * x[0] + A[idx][1] * x[1] + A[idx][2] * x[2]
			c = self.matrix_cross_coupling
			if c != 0.0:
				cross = (-c * x[1], -c * x[0], c * (x[0] + x[1]))
				u += cross[idx]
			return float(self.base_reward + u)

		n_i = float(self.popularity.get(strategy, 0))
		penalty = self.gamma * n_i

		bonus = 0.0
		if self.epsilon != 0.0 and self.strategy_cycle and strategy in self.strategy_cycle:
			idx = self.strategy_cycle.index(strategy)
			prev_s = self.strategy_cycle[(idx - 1) % len(self.strategy_cycle)]
			next_s = self.strategy_cycle[(idx + 1) % len(self.strategy_cycle)]
			n_prev = float(self.popularity.get(prev_s, 0))
			n_next = float(self.popularity.get(next_s, 0))
			bonus = self.epsilon * (n_prev - n_next)

		return self.base_reward - penalty + bonus

	def update_popularity(self, chosen_strategies: list[str]) -> None:
		self.popularity = {}
		for s in chosen_strategies:
			self.popularity[s] = self.popularity.get(s, 0) + 1

	def resolve_player_outcome(self, player: object, strategy: str) -> dict[str, object]:
		base_reward = float(self.evaluate(strategy))
		if self.event_loader is None:
			return {
				"reward": base_reward,
				"base_reward": base_reward,
				"event_result": None,
			}
		event_result = self.event_loader.process_turn(player, rng=self._event_rng, strategy=strategy)
		total_reward = base_reward + float(event_result.get("utility_delta", 0.0))
		return {
			"reward": total_reward,
			"base_reward": base_reward,
			"event_result": event_result,
		}

	def evaluate_player(self, player: object, strategy: str) -> float:
		return float(self.resolve_player_outcome(player, strategy)["reward"])

