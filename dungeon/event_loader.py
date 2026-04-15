from __future__ import annotations

import json
import inspect
import math
import random
from pathlib import Path
from typing import Any, Callable, Mapping


class EventSchemaError(ValueError):
	pass


def clamp(value: float, lower: float, upper: float) -> float:
	return max(lower, min(upper, value))


def logistic(value: float) -> float:
	return 1.0 / (1.0 + math.exp(-value))


def _success_model_linear_risk_complement(final_risk: float, **_kwargs: Any) -> float:
	"""Baseline success model: success probability is 1 minus final risk."""
	return 1.0 - float(final_risk)


def _success_model_logistic(
	final_risk: float,
	*,
	success_bias: float = 1.0,
	steepness: float = 6.0,
	**_kwargs: Any,
) -> float:
	"""Smooth nonlinear success curve controlled by bias and steepness."""
	return logistic(float(success_bias) - float(steepness) * float(final_risk))


def _success_model_linear_clip(
	final_risk: float,
	*,
	slope: float = -1.0,
	offset: float = 1.0,
	**_kwargs: Any,
) -> float:
	"""Affine success curve with clipping applied by the caller."""
	return float(slope) * float(final_risk) + float(offset)


def _success_model_threshold(
	final_risk: float,
	*,
	threshold: float = 0.5,
	success_if_low: float = 1.0,
	success_if_high: float = 0.0,
	**_kwargs: Any,
) -> float:
	"""Piecewise success model that switches at a risk threshold."""
	if float(final_risk) <= float(threshold):
		return float(success_if_low)
	return float(success_if_high)


SUCCESS_MODEL_REGISTRY: dict[str, Callable[..., float]] = {
	"linear_risk_complement": _success_model_linear_risk_complement,
	"logistic": _success_model_logistic,
	"linear_clip": _success_model_linear_clip,
	"threshold": _success_model_threshold,
}


LEGACY_LOGISTIC_MARKERS = ("logistic", "success_steepness", "success_bias")


class EventLoader:
	def __init__(
		self,
		json_path: str | Path,
		*,
		apply_trait_deltas: bool = True,
		failure_threshold_override: float | None = None,
		health_penalty_coefficient: float = 0.10,
		stress_risk_coefficient: float = 0.10,
		risk_ma_alpha: float = 0.0,
		risk_ma_multiplier: float = 0.0,
		stress_decay_c: float = 0.0,
		stress_decay_beta: float = 2.0,
	):
		self.json_path = Path(json_path)
		self.data = json.loads(self.json_path.read_text())
		self.dimensions_order = list(self.data.get("dimensions_order", []))
		self.utility_weight_policy = dict(self.data.get("utility_weight_policy", {}))
		self.risk_policy = dict(self.data.get("risk_policy", {}))
		self.success_policy = dict(self.data.get("success_policy", {}))
		self.state_policy = dict(self.data.get("state_policy", {}))
		self.reward_policy = dict(self.data.get("reward_policy", {}))
		self.templates = list(self.data.get("templates", []))
		self.template_by_id = {
			str(template["event_id"]): template for template in self.templates
		}
		self.event_types = [str(event_type) for event_type in self.data.get("event_types", [])]
		self._apply_trait_deltas_enabled = bool(apply_trait_deltas)
		# Runtime overrides (can be set via CLI; None means "use per-action JSON values").
		self._failure_threshold_override: float | None = (
			clamp(float(failure_threshold_override), 0.0, 1.0)
			if failure_threshold_override is not None
			else None
		)
		self._health_penalty_coefficient: float = clamp(float(health_penalty_coefficient), 0.0, 1.0)
		self._stress_risk_coefficient: float = clamp(float(stress_risk_coefficient), 0.0, 1.0)
		self._risk_ma_alpha: float = clamp(float(risk_ma_alpha), 0.0, 1.0)
		self._risk_ma_multiplier: float = max(0.0, float(risk_ma_multiplier))
		self._stress_decay_c: float = max(0.0, float(stress_decay_c))
		self._stress_decay_beta: float = max(0.0, float(stress_decay_beta))
		self._event_type_weights: dict[str, float] = {
			str(event_type): 1.0 for event_type in self.event_types
		}
		self._event_type_risk_multipliers: dict[str, float] = {
			str(event_type): 1.0 for event_type in self.event_types
		}
		self._event_type_reward_multipliers: dict[str, float] = {
			str(event_type): 1.0 for event_type in self.event_types
		}
		self._event_type_trait_delta_multipliers: dict[str, float] = {
			str(event_type): 1.0 for event_type in self.event_types
		}
		self.validate_schema()

	def set_failure_threshold_override(self, ft: float) -> None:
		"""Dynamically override the failure threshold (used by adaptive rule-mutation)."""
		self._failure_threshold_override = clamp(float(ft), 0.0, 1.0)

	def set_event_type_weights(self, weights: Mapping[str, float]) -> None:
		resolved: dict[str, float] = {}
		for event_type in self.event_types:
			value = float(weights.get(event_type, 1.0)) if event_type in weights else 1.0
			if not math.isfinite(value):
				value = 1.0
			resolved[str(event_type)] = max(0.0, float(value))
		self._event_type_weights = resolved

	def _resolve_family_multipliers(self, multipliers: Mapping[str, float]) -> dict[str, float]:
		resolved: dict[str, float] = {}
		for event_type in self.event_types:
			value = float(multipliers.get(event_type, 1.0)) if event_type in multipliers else 1.0
			if not math.isfinite(value):
				value = 1.0
			resolved[str(event_type)] = clamp(float(value), 0.5, 2.0)
		return resolved

	def set_event_type_risk_multipliers(self, multipliers: Mapping[str, float]) -> None:
		self._event_type_risk_multipliers = self._resolve_family_multipliers(multipliers)

	def set_event_type_reward_multipliers(self, multipliers: Mapping[str, float]) -> None:
		self._event_type_reward_multipliers = self._resolve_family_multipliers(multipliers)

	def set_event_type_trait_delta_multipliers(self, multipliers: Mapping[str, float]) -> None:
		self._event_type_trait_delta_multipliers = self._resolve_family_multipliers(multipliers)

	def get_world_adjustments(self) -> dict[str, dict[str, float]]:
		return {
			"event_type_weights": dict(self._event_type_weights),
			"risk_multipliers": dict(self._event_type_risk_multipliers),
			"reward_multipliers": dict(self._event_type_reward_multipliers),
			"trait_delta_multipliers": dict(self._event_type_trait_delta_multipliers),
		}

	def set_event_parameter_multipliers(
		self,
		*,
		threat_risk_multiplier: float = 1.0,
		resource_reward_multiplier: float = 1.0,
		noise_penalty_multiplier: float = 1.0,
		intel_reward_multiplier: float = 1.0,
	) -> None:
		self.set_event_type_risk_multipliers({"Threat": float(threat_risk_multiplier)})
		self.set_event_type_reward_multipliers(
			{
				"Resource": float(resource_reward_multiplier),
				"Uncertainty": float(noise_penalty_multiplier),
				"Internal": float(noise_penalty_multiplier),
				"Navigation": float(intel_reward_multiplier),
			}
		)

	def _event_type_weight(self, template: Mapping[str, Any]) -> float:
		event_type = str(template.get("type", ""))
		return max(0.0, float(self._event_type_weights.get(event_type, 1.0)))

	def validate_schema(self) -> None:
		if not self.dimensions_order:
			raise EventSchemaError("dimensions_order must be non-empty")
		allowed_state_variables = set(self.state_policy.get("state_variables", []))
		if not allowed_state_variables:
			raise EventSchemaError("state_policy.state_variables must be non-empty")
		self._validate_state_effect_mapping(
			self.state_policy.get("default_state_effects", {}).get("on_success", {}),
			allowed_state_variables=allowed_state_variables,
			context="state_policy.default_state_effects.on_success",
		)
		self._validate_state_effect_mapping(
			self.state_policy.get("default_state_effects", {}).get("on_failure", {}),
			allowed_state_variables=allowed_state_variables,
			context="state_policy.default_state_effects.on_failure",
		)
		required_reward_keys = set(self.reward_policy.get("required_reward_keys", []))
		seen_event_ids: set[str] = set()
		for template in self.templates:
			if "event_id" not in template or "actions" not in template:
				raise EventSchemaError("each template must include event_id and actions")
			event_id = str(template["event_id"])
			if event_id in seen_event_ids:
				raise EventSchemaError(f"duplicate event_id: {event_id}")
			seen_event_ids.add(event_id)
			for action in template["actions"]:
				weights = action.get("weights")
				if not isinstance(weights, list) or len(weights) != len(self.dimensions_order):
					raise EventSchemaError(
						f"action {action.get('name')} in {event_id} has invalid weights length"
					)
				reward_effects = action.get("reward_effects")
				if not isinstance(reward_effects, dict) or not required_reward_keys.issubset(reward_effects):
					raise EventSchemaError(
						f"action {action.get('name')} in {event_id} is missing required reward keys"
					)
				failure_outcomes = action.get("failure_outcomes")
				if not isinstance(failure_outcomes, list) or not failure_outcomes:
					raise EventSchemaError(
						f"action {action.get('name')} in {event_id} must declare failure_outcomes"
					)
				self._validate_success_model(action, event_id=event_id)
				state_effects = dict(action.get("state_effects", {}))
				for branch in ("on_success", "on_failure"):
					self._validate_state_effect_mapping(
						state_effects.get(branch, {}),
						allowed_state_variables=allowed_state_variables,
						context=f"{event_id}.{action.get('name')}.state_effects.{branch}",
					)

	def _validate_state_effect_mapping(
		self,
		mapping: Mapping[str, Any],
		*,
		allowed_state_variables: set[str],
		context: str,
	) -> None:
		if not isinstance(mapping, Mapping):
			raise EventSchemaError(f"{context} must be a mapping")
		for key, value in mapping.items():
			target_key = str(key[:-6] if str(key).endswith("_delta") else key)
			if target_key not in allowed_state_variables:
				raise EventSchemaError(f"{context} uses unknown state variable: {key}")
			try:
				float(value)
			except (TypeError, ValueError) as exc:
				raise EventSchemaError(f"{context} contains non-numeric state delta: {key}") from exc

	def _infer_success_model_name(self, formula: str | None) -> str:
		text = str(formula or "").strip().lower()
		if any(marker in text for marker in LEGACY_LOGISTIC_MARKERS):
			return "logistic"
		return "linear_risk_complement"

	def _default_success_model(self) -> tuple[str, dict[str, float]]:
		default_name = str(
			self.success_policy.get(
				"default_model",
				self._infer_success_model_name(self.success_policy.get("default_probability_formula")),
			)
		)
		kwargs = {
			key: float(value)
			for key, value in dict(self.success_policy.get("default_model_kwargs", {})).items()
		}
		if default_name == "logistic":
			kwargs.setdefault("success_bias", float(self.success_policy.get("default_success_bias", 1.0)))
			kwargs.setdefault("steepness", float(self.success_policy.get("default_success_steepness", 6.0)))
		return default_name, kwargs

	def _coerce_success_model_kwargs(self, raw_kwargs: Mapping[str, Any]) -> dict[str, float]:
		kwargs: dict[str, float] = {}
		for key, value in raw_kwargs.items():
			normalized_key = "steepness" if key == "success_steepness" else str(key)
			kwargs[normalized_key] = float(value)
		return kwargs

	def _success_model_callable(self, model_name: str) -> Callable[..., float]:
		try:
			return SUCCESS_MODEL_REGISTRY[model_name]
		except KeyError as exc:
			raise EventSchemaError(f"unknown success_model: {model_name}") from exc

	def _validate_success_model_kwargs(self, model_name: str, kwargs: Mapping[str, Any]) -> None:
		allowed_kwargs = self._allowed_success_model_kwargs(model_name)
		unexpected = sorted(set(kwargs) - allowed_kwargs)
		if unexpected:
			raise EventSchemaError(
				f"success_model '{model_name}' received unexpected kwargs: {', '.join(unexpected)}"
			)

	def _allowed_success_model_kwargs(self, model_name: str) -> set[str]:
		fn = self._success_model_callable(model_name)
		signature = inspect.signature(fn)
		return {
			name
			for name, parameter in signature.parameters.items()
			if name != "final_risk" and parameter.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
		}

	def _validate_success_model(self, action: Mapping[str, Any], *, event_id: str) -> None:
		model = self.resolve_success_model(action)
		self._validate_success_model_kwargs(str(model["name"]), dict(model["kwargs"]))
		fn = self._success_model_callable(model["name"])
		try:
			probe = float(fn(0.5, **dict(model["kwargs"])))
		except TypeError as exc:
			raise EventSchemaError(
				f"action {action.get('name')} in {event_id} has invalid success_model kwargs"
			) from exc
		if not math.isfinite(probe):
			raise EventSchemaError(
				f"action {action.get('name')} in {event_id} produced non-finite success probability"
			)

	def normalize_weights(self, weights: list[float]) -> list[float]:
		eps = float(self.utility_weight_policy.get("eps", 1e-9))
		values = []
		for value in weights:
			f = float(value)
			if not math.isfinite(f):
				return [0.0 for _ in weights]
			values.append(clamp(f, -1.0, 1.0))
		denom = sum(abs(v) for v in values)
		if denom <= eps:
			return [0.0 for _ in values]
		return [v / denom for v in values]

	def normalize_personality(self, personality: Mapping[str, float]) -> dict[str, float]:
		return {
			key: clamp(float(personality.get(key, 0.0)), -1.0, 1.0)
			for key in self.dimensions_order
		}

	def normalize_state(self, state: Mapping[str, float] | None) -> dict[str, float]:
		state = state or {}
		return {
			"risk": float(state.get("risk", 0.0)),
			"stress": float(state.get("stress", 0.0)),
			"noise": float(state.get("noise", 0.0)),
			"risk_drift": float(state.get("risk_drift", 0.0)),
			"health": float(state.get("health", 1.0)),
			"intel": float(state.get("intel", 0.0)),
			"risk_ma": float(state.get("risk_ma", 0.0)),
		}

	def get_event_template(self, event_id: str) -> dict[str, Any]:
		try:
			return self.template_by_id[event_id]
		except KeyError as exc:
			raise EventSchemaError(f"unknown event_id: {event_id}") from exc

	def sample_event_template(self, rng: random.Random | None = None) -> dict[str, Any]:
		rng = rng if rng is not None else random.Random()
		weights = [self._event_type_weight(template) for template in self.templates]
		if sum(weights) <= 0.0:
			return rng.choice(self.templates)
		return rng.choices(self.templates, weights=weights, k=1)[0]

	def _scale_state_effects(self, *, event_type: str, state_effects: Mapping[str, float]) -> dict[str, float]:
		resolved = {str(key): float(value) for key, value in state_effects.items()}
		reward_multiplier = float(self._event_type_reward_multipliers.get(str(event_type), 1.0))
		if event_type == "Navigation":
			intel_key = "intel_delta"
			if intel_key in resolved and resolved[intel_key] > 0.0:
				resolved[intel_key] = float(resolved[intel_key]) * reward_multiplier
		elif event_type in {"Uncertainty", "Internal"}:
			for key in ("stress_delta", "noise_delta", "risk_drift_delta"):
				if key in resolved and resolved[key] > 0.0:
					resolved[key] = float(resolved[key]) * reward_multiplier
		return resolved

	def _scale_trait_deltas(self, *, event_type: str, trait_deltas: Mapping[str, float]) -> dict[str, float]:
		multiplier = float(self._event_type_trait_delta_multipliers.get(str(event_type), 1.0))
		return {
			str(key): float(value) * multiplier
			for key, value in trait_deltas.items()
		}

	def _scale_payload(
		self,
		*,
		event_type: str,
		payload: Mapping[str, Any],
	) -> dict[str, Any]:
		resolved = dict(payload)
		reward_multiplier = float(self._event_type_reward_multipliers.get(str(event_type), 1.0))
		resolved["utility_delta"] = float(resolved.get("utility_delta", 0.0)) * reward_multiplier
		resolved["risk_delta"] = float(resolved.get("risk_delta", 0.0)) * reward_multiplier
		return resolved

	def resolve_success_model(self, action: Mapping[str, Any]) -> dict[str, Any]:
		default_name, default_kwargs = self._default_success_model()
		model_name = default_name
		model_kwargs = dict(default_kwargs)
		raw_model = action.get("success_model")
		if isinstance(raw_model, str):
			model_name = str(raw_model)
		elif isinstance(raw_model, Mapping):
			raw_mapping = dict(raw_model)
			model_name = str(raw_mapping.get("name", self._infer_success_model_name(raw_mapping.get("probability_formula"))))
			allowed_legacy_keys = self._allowed_success_model_kwargs(model_name)
			if "kwargs" in raw_mapping:
				candidate_kwargs = self._coerce_success_model_kwargs(dict(raw_mapping.get("kwargs", {})))
				model_kwargs.update({key: value for key, value in candidate_kwargs.items() if key in allowed_legacy_keys})
			legacy_kwargs = {
				key: raw_mapping[key]
				for key in (
					"success_bias",
					"success_steepness",
					"steepness",
					"slope",
					"offset",
					"threshold",
					"success_if_low",
					"success_if_high",
				)
				if key in raw_mapping and ("steepness" if key == "success_steepness" else key) in allowed_legacy_keys
			}
			model_kwargs.update(self._coerce_success_model_kwargs(legacy_kwargs))
		candidate_action_kwargs = self._coerce_success_model_kwargs(dict(action.get("success_model_kwargs", {})))
		model_kwargs.update(candidate_action_kwargs)
		return {
			"name": model_name,
			"kwargs": model_kwargs,
			"fallback": str(self.success_policy.get("fallback", "clamp(0, 1, 1 - base_risk)")),
		}

	def resolve_state_effects(self, action: Mapping[str, Any]) -> dict[str, dict[str, float]]:
		defaults = self.state_policy.get("default_state_effects", {})
		state_effects = dict(action.get("state_effects", {}))
		resolved: dict[str, dict[str, float]] = {}
		for branch in ("on_success", "on_failure"):
			merged = dict(defaults.get(branch, {}))
			merged.update(state_effects.get(branch, {}))
			resolved[branch] = {key: float(value) for key, value in merged.items()}
		return resolved

	def compute_action_utility(
		self,
		action: Mapping[str, Any],
		personality: Mapping[str, float],
		state: Mapping[str, float] | None = None,
		*,
		rng: random.Random | None = None,
	) -> float:
		traits = self.normalize_personality(personality)
		state_values = self.normalize_state(state)
		weights = self.normalize_weights(list(action["weights"]))
		utility = 0.0
		for key, weight in zip(self.dimensions_order, weights):
			utility += weight * float(traits[key])
		if rng is not None:
			randomness = max(0.0, float(traits.get("randomness", 0.0)))
			noise_multiplier = clamp(1.0 + 0.2 * float(state_values.get("noise", 0.0)), 0.0, 2.0)
			utility += rng.uniform(-1.0, 1.0) * 0.1 * randomness * noise_multiplier
		return float(utility)

	def choose_action(
		self,
		event: Mapping[str, Any],
		personality: Mapping[str, float],
		state: Mapping[str, float] | None = None,
		*,
		rng: random.Random | None = None,
	) -> dict[str, Any]:
		return max(
			event["actions"],
			key=lambda action: self.compute_action_utility(action, personality, state=state, rng=rng),
		)

	def compute_final_risk(
		self,
		action: Mapping[str, Any],
		personality: Mapping[str, float],
		state: Mapping[str, float] | None = None,
		*,
		event_type: str | None = None,
	) -> float:
		risk_multiplier = 1.0
		if event_type is not None:
			risk_multiplier = float(self._event_type_risk_multipliers.get(str(event_type), 1.0))
		base_risk = float(action.get("base_risk", 0.0)) * risk_multiplier
		risk_model = dict(action.get("risk_model", {}))
		risk = base_risk + float(risk_model.get("risk_bias", 0.0)) * risk_multiplier
		traits = self.normalize_personality(personality)
		state_values = self.normalize_state(state)
		for key, weight in dict(risk_model.get("trait_weights", {})).items():
			risk += float(weight) * float(traits.get(key, 0.0))
		risk += float(state_values.get("risk", 0.0))
		risk += float(state_values.get("risk_drift", 0.0))
		risk += self._stress_risk_coefficient * float(state_values.get("stress", 0.0))
		risk += self._risk_ma_multiplier * float(state_values.get("risk_ma", 0.0))
		risk += clamp(1.0 - float(state_values.get("health", 1.0)), 0.0, 1.0) * self._health_penalty_coefficient
		if not math.isfinite(risk):
			return clamp(base_risk, 0.0, 1.0)
		clamp_range = risk_model.get("clamp", self.risk_policy.get("clamp_range", [0.0, 1.0]))
		return clamp(float(risk), float(clamp_range[0]), float(clamp_range[1]))

	def compute_success_prob(
		self,
		action: Mapping[str, Any],
		final_risk: float,
		state: Mapping[str, float] | None = None,
	) -> float:
		# Hard gate: if final_risk >= action-level (or global) failure_threshold, force failure.
		risk_model = dict(action.get("risk_model", {}))
		default_threshold = float(self.risk_policy.get("default_failure_threshold", 0.5))
		if self._failure_threshold_override is not None:
			failure_threshold = self._failure_threshold_override
		else:
			failure_threshold = float(risk_model.get("failure_threshold", default_threshold))
		if float(final_risk) >= failure_threshold:
			return 0.0
		model = self.resolve_success_model(action)
		model_name = str(model["name"])
		kwargs = dict(model["kwargs"])
		state_values = self.normalize_state(state)
		self._validate_success_model_kwargs(model_name, kwargs)
		fn = self._success_model_callable(model_name)
		success_prob = fn(float(final_risk), **kwargs)
		success_prob += clamp(0.05 * float(state_values.get("intel", 0.0)), -0.25, 0.25)
		if not math.isfinite(success_prob):
			fallback = 1.0 - clamp(float(action.get("base_risk", 0.0)), 0.0, 1.0)
			clamp_range = self.success_policy.get("clamp_range", [0.0, 1.0])
			return clamp(fallback, float(clamp_range[0]), float(clamp_range[1]))
		clamp_range = self.success_policy.get("clamp_range", [0.0, 1.0])
		return clamp(float(success_prob), float(clamp_range[0]), float(clamp_range[1]))

	def _apply_state_decay(self, player: object) -> None:
		"""Exponential decay for accumulating state variables; passive health regen per turn."""
		decay_cfg = dict(self.state_policy.get("decay_rates", {}))
		state = player.state
		for var, default_rate in (
			("stress", 0.92),
			("noise", 0.92),
			("risk_drift", 0.90),
			("risk", 0.95),
		):
			rate = float(decay_cfg.get(var, default_rate))
			if var == "stress" and self._stress_decay_c > 0.0:
				# Nonlinear asymmetry: stress decays slower when already high.
				# effective_rate = 1 - (1 - base_rate) / (1 + c * stress^beta)
				stress_val = float(state.get("stress", 0.0))
				nonlinear_denom = 1.0 + self._stress_decay_c * (stress_val ** self._stress_decay_beta)
				rate = 1.0 - (1.0 - rate) / nonlinear_denom
			state[var] = float(state.get(var, 0.0)) * rate
		regen = float(decay_cfg.get("health_regen", 0.02))
		state["health"] = clamp(float(state.get("health", 1.0)) + regen, 0.0, 1.0)
		if self._risk_ma_alpha > 0.0:
			state["risk_ma"] = (
				self._risk_ma_alpha * float(state.get("risk_ma", 0.0))
				+ (1.0 - self._risk_ma_alpha) * float(state.get("risk", 0.0))
			)

	def choose_failure_outcome(
		self,
		failure_outcomes: list[dict[str, Any]],
		*,
		rng: random.Random | None = None,
	) -> dict[str, Any]:
		rng = rng if rng is not None else random.Random()
		if len(failure_outcomes) == 1:
			return failure_outcomes[0]
		weights = [max(0.0, float(item.get("probability", 0.0))) for item in failure_outcomes]
		if sum(weights) <= 0:
			return failure_outcomes[0]
		return rng.choices(failure_outcomes, weights=weights, k=1)[0]

	def _ensure_player_containers(self, player: object) -> None:
		if not hasattr(player, "personality") or not isinstance(getattr(player, "personality"), dict):
			setattr(player, "personality", {key: 0.0 for key in self.dimensions_order})
		if not hasattr(player, "state") or not isinstance(getattr(player, "state"), dict):
			setattr(
				player,
				"state",
				{
					"risk": 0.0,
					"stress": 0.0,
					"noise": 0.0,
					"risk_drift": 0.0,
					"health": 1.0,
					"intel": 0.0,
				},
			)

	def _apply_trait_deltas(self, player: object, deltas: Mapping[str, float]) -> None:
		if not self._apply_trait_deltas_enabled:
			return
		if hasattr(player, "apply_trait_deltas"):
			player.apply_trait_deltas(deltas)
			return
		for key, delta in deltas.items():
			current = float(player.personality.get(key, 0.0))
			player.personality[key] = clamp(current + float(delta), -1.0, 1.0)

	def _apply_state_deltas(self, player: object, deltas: Mapping[str, float]) -> None:
		if hasattr(player, "apply_state_deltas"):
			player.apply_state_deltas(deltas)
			return
		for key, delta in deltas.items():
			target_key = key[:-6] if key.endswith("_delta") else key
			current = float(player.state.get(target_key, 0.0))
			player.state[target_key] = current + float(delta)

	def _apply_popularity_shift(self, player: object, shift: Mapping[str, float]) -> None:
		if hasattr(player, "apply_popularity_shift"):
			player.apply_popularity_shift(shift)
			return
		biases = player.state.setdefault("popularity_shift_buffer", {})
		for key, delta in shift.items():
			biases[key] = float(biases.get(key, 0.0)) + float(delta)

	def _apply_reward_payload(
		self,
		player: object,
		payload: Mapping[str, Any],
		state_effects: Mapping[str, float],
		*,
		event_type: str,
		success: bool,
	) -> dict[str, Any]:
		scaled_payload = self._scale_payload(event_type=event_type, payload=payload)
		scaled_state_effects = self._scale_state_effects(event_type=event_type, state_effects=state_effects)
		utility_delta = float(scaled_payload.get("utility_delta", 0.0))
		risk_delta = float(scaled_payload.get("risk_delta", 0.0))
		player.state["risk"] = float(player.state.get("risk", 0.0)) + risk_delta
		popularity_shift = dict(scaled_payload.get("popularity_shift", {}))
		trait_deltas = dict(scaled_payload.get("trait_deltas", {})) if self._apply_trait_deltas_enabled else {}
		trait_deltas = self._scale_trait_deltas(event_type=event_type, trait_deltas=trait_deltas)
		self._apply_popularity_shift(player, popularity_shift)
		self._apply_trait_deltas(player, trait_deltas)
		self._apply_state_deltas(player, scaled_state_effects)
		player.state["last_event_success"] = bool(success)
		player.state["last_event_tags"] = list(scaled_payload.get("state_tags", []))
		player.state["last_sample_quality"] = scaled_payload.get("sample_quality")
		return {
			"utility_delta": utility_delta,
			"risk_delta": risk_delta,
			"popularity_shift": popularity_shift,
			"trait_deltas": trait_deltas,
			"state_effects": dict(scaled_state_effects),
			"state_tags": list(scaled_payload.get("state_tags", [])),
			"sample_quality": scaled_payload.get("sample_quality"),
		}

	def process_turn(
		self,
		player: object,
		*,
		event_id: str | None = None,
		action_name: str | None = None,
		rng: random.Random | None = None,
		strategy: str | None = None,
	) -> dict[str, Any]:
		self._ensure_player_containers(player)
		rng = rng if rng is not None else getattr(player, "_rng", None) or random.Random()
		event = self.get_event_template(event_id) if event_id is not None else self.sample_event_template(rng=rng)
		if action_name is not None:
			action = next(action for action in event["actions"] if action["name"] == action_name)
		else:
			action = self.choose_action(event, player.personality, state=player.state, rng=rng)

		event_type = str(event.get("type", ""))
		final_risk = self.compute_final_risk(
			action,
			player.personality,
			state=player.state,
			event_type=event_type,
		)
		success_prob = self.compute_success_prob(action, final_risk, state=player.state)
		success = rng.random() < success_prob
		resolved_state_effects = self.resolve_state_effects(action)

		if success:
			applied_payload = self._apply_reward_payload(
				player,
				dict(action.get("reward_effects", {})),
				resolved_state_effects["on_success"],
				event_type=event_type,
				success=True,
			)
			result_kind = "success"
		else:
			failure = self.choose_failure_outcome(list(action.get("failure_outcomes", [])), rng=rng)
			applied_payload = self._apply_reward_payload(
				player,
				failure,
				resolved_state_effects["on_failure"],
				event_type=event_type,
				success=False,
			)
			result_kind = str(failure.get("kind", "failure"))

		player.state["last_event_id"] = event["event_id"]
		player.state["last_action_name"] = action["name"]
		player.state["last_strategy_context"] = strategy
		player.state["last_final_risk"] = final_risk
		player.state["last_success_prob"] = success_prob
		self._apply_state_decay(player)

		return {
			"event_id": event["event_id"],
			"event_type": event_type,
			"action_name": action["name"],
			"success": success,
			"result_kind": result_kind,
			"final_risk": final_risk,
			"success_prob": success_prob,
			"utility_delta": applied_payload["utility_delta"],
			"risk_delta": applied_payload["risk_delta"],
			"trait_deltas": applied_payload["trait_deltas"],
			"popularity_shift": applied_payload["popularity_shift"],
			"state_effects": applied_payload["state_effects"],
			"state_tags": applied_payload["state_tags"],
			"sample_quality": applied_payload["sample_quality"],
		}