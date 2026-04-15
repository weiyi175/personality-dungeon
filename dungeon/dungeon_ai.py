import random
from collections import deque
from math import isfinite


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
		memory_kernel: int = 1,
		threshold_theta: float = 0.40,
		threshold_theta_low: float | None = None,
		threshold_theta_high: float | None = None,
		threshold_trigger: str = "ad_share",
		threshold_state_alpha: float = 1.0,
		threshold_a_hi: float | None = None,
		threshold_b_hi: float | None = None,
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
		self.memory_kernel = int(memory_kernel)
		self.threshold_theta = float(threshold_theta)
		self.threshold_theta_low = float(threshold_theta_low) if threshold_theta_low is not None else None
		self.threshold_theta_high = float(threshold_theta_high) if threshold_theta_high is not None else None
		self.threshold_trigger = str(threshold_trigger)
		self.threshold_state_alpha = float(threshold_state_alpha)
		self.threshold_a_hi = float(threshold_a_hi) if threshold_a_hi is not None else None
		self.threshold_b_hi = float(threshold_b_hi) if threshold_b_hi is not None else None
		self.base_reward = float(base_reward)
		self.event_loader = event_loader
		self._event_rng = event_rng if event_rng is not None else random.Random()
		# cycle defines (prev, next) neighbors in a directed ring.
		self.strategy_cycle = list(strategy_cycle) if strategy_cycle is not None else []
		self.popularity: dict[str, int] = {}
		self._popularity_history: deque[dict[str, float]] = deque(maxlen=max(1, self.memory_kernel))
		self._threshold_regime_hi: bool | None = None
		self._threshold_state_value: float | None = None

		allowed = {"count_cycle", "matrix_ab", "threshold_ab"}
		if self.payoff_mode not in allowed:
			raise ValueError(f"Unknown payoff_mode: {self.payoff_mode!r}. Allowed: {sorted(allowed)}")
		if self.payoff_mode in {"matrix_ab", "threshold_ab"} and len(self.strategy_cycle) != 3:
			raise ValueError("matrix-like payoff modes require exactly 3 strategies in strategy_cycle")
		if self.memory_kernel <= 0 or (self.memory_kernel % 2) == 0:
			raise ValueError("memory_kernel must be a positive odd integer")
		if not isfinite(self.threshold_theta) or not (0.0 <= self.threshold_theta <= 1.0):
			raise ValueError("threshold_theta must be finite and lie in [0,1]")
		if self.threshold_theta_low is not None and (not isfinite(self.threshold_theta_low) or not (0.0 <= self.threshold_theta_low <= 1.0)):
			raise ValueError("threshold_theta_low must be finite and lie in [0,1]")
		if self.threshold_theta_high is not None and (not isfinite(self.threshold_theta_high) or not (0.0 <= self.threshold_theta_high <= 1.0)):
			raise ValueError("threshold_theta_high must be finite and lie in [0,1]")
		lo, hi = self._threshold_band()
		if lo > hi:
			raise ValueError("threshold_theta_low must be <= threshold_theta_high")
		if self.threshold_trigger not in {"ad_share", "ad_product"}:
			raise ValueError("threshold_trigger must be one of {'ad_share', 'ad_product'}")
		if not isfinite(self.threshold_state_alpha) or not (0.0 < self.threshold_state_alpha <= 1.0):
			raise ValueError("threshold_state_alpha must be finite and lie in (0,1]")
		if self.threshold_a_hi is not None and not isfinite(self.threshold_a_hi):
			raise ValueError("threshold_a_hi must be finite")
		if self.threshold_b_hi is not None and not isfinite(self.threshold_b_hi):
			raise ValueError("threshold_b_hi must be finite")

	def _threshold_band(self) -> tuple[float, float]:
		lo = float(self.threshold_theta if self.threshold_theta_low is None else self.threshold_theta_low)
		hi = float(self.threshold_theta if self.threshold_theta_high is None else self.threshold_theta_high)
		return lo, hi

	def _threshold_trigger_value(self, p: dict[str, float]) -> float:
		x_a = float(p.get(self.strategy_cycle[0], 0.0))
		x_d = float(p.get(self.strategy_cycle[1], 0.0))
		if self.threshold_trigger == "ad_product":
			return float(4.0 * x_a * x_d)
		return float(x_a + x_d)

	def _threshold_state(self, trigger_value: float) -> float:
		alpha = float(self.threshold_state_alpha)
		if self._threshold_state_value is None or alpha >= 1.0:
			self._threshold_state_value = float(trigger_value)
		else:
			self._threshold_state_value = alpha * float(trigger_value) + (1.0 - alpha) * float(self._threshold_state_value)
		return float(self._threshold_state_value)

	def _append_popularity_history(self, popularity: dict[str, float]) -> None:
		self._popularity_history.append({str(k): float(v) for k, v in popularity.items()})

	def set_popularity(self, popularity: dict[str, float]) -> None:
		self.popularity = dict(popularity)
		self._append_popularity_history(self.popularity)

	def _proportions(self) -> dict[str, float]:
		if self._popularity_history:
			merged = {s: 0.0 for s in self.strategy_cycle}
			for state in self._popularity_history:
				for s in self.strategy_cycle:
					merged[s] += float(state.get(s, 0.0))
			denom = float(len(self._popularity_history))
			avg_popularity = {s: (merged[s] / denom) for s in self.strategy_cycle}
		else:
			avg_popularity = {s: float(self.popularity.get(s, 0.0)) for s in self.strategy_cycle}
		total = sum(float(v) for v in avg_popularity.values())
		if total <= 0:
			return {s: 0.0 for s in self.strategy_cycle}
		return {s: (float(avg_popularity.get(s, 0.0)) / float(total)) for s in self.strategy_cycle}

	def _effective_matrix_params(self, p: dict[str, float]) -> tuple[float, float]:
		if self.payoff_mode != "threshold_ab":
			return float(self.a), float(self.b)
		trigger_value = self._threshold_trigger_value(p)
		state_value = self._threshold_state(trigger_value)
		lo, hi = self._threshold_band()
		if state_value >= hi:
			self._threshold_regime_hi = True
		elif state_value <= lo:
			self._threshold_regime_hi = False
		elif self._threshold_regime_hi is None:
			self._threshold_regime_hi = False
		if self._threshold_regime_hi:
			return (
				float(self.a if self.threshold_a_hi is None else self.threshold_a_hi),
				float(self.b if self.threshold_b_hi is None else self.threshold_b_hi),
			)
		return float(self.a), float(self.b)

	def evaluate(self, strategy: str) -> float:
		if self.payoff_mode in {"matrix_ab", "threshold_ab"}:
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
			a_eff, b_eff = self._effective_matrix_params(p)
			A = (
				(0.0, a_eff, -b_eff),
				(-b_eff, 0.0, a_eff),
				(a_eff, -b_eff, 0.0),
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
		self._append_popularity_history(self.popularity)

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

