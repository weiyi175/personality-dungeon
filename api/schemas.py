"""Formal API schema for the core/frontend contract.

This module defines the stable JSON envelope used by the Python core and any
frontend client. The schema is intentionally strict on field names and value
ranges, while still allowing empty placeholder objects for sections that are
not populated yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Final, Literal, cast

API_VERSION: Final[str] = "1.0"
PERSONALITY_BASIS: Final[tuple[str, ...]] = (
	# --- 擴張組 The Drivers ---
	"impulsiveness",
	"assertiveness",
	"optimism",
	# --- 防禦組 The Stabilizers ---
	"risk_aversion",
	"suspicion",
	"endurance",
	# --- 擾動組 The Explorers ---
	"randomness",
	"stability_seeking",
	"curiosity",
)
RESPONSE_KINDS: Final[tuple[str, ...]] = ("reset", "step", "snapshot", "error")
STRATEGIES: Final[tuple[str, ...]] = ("aggressive", "defensive", "balanced", "custom")

ResponseKind = Literal["reset", "step", "snapshot", "error"]
StrategyName = Literal["aggressive", "defensive", "balanced", "custom"]


class SchemaError(ValueError):
	"""Raised when a payload does not satisfy the public API schema."""


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
	if isinstance(value, Mapping):
		return value
	raise SchemaError(f"{field_name} must be a mapping")


def _require_str(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
	if not isinstance(value, str):
		raise SchemaError(f"{field_name} must be a string")
	if not allow_empty and not value:
		raise SchemaError(f"{field_name} must not be empty")
	return value


def _require_bool(value: Any, field_name: str) -> bool:
	if isinstance(value, bool):
		return value
	raise SchemaError(f"{field_name} must be a boolean")


def _require_int(value: Any, field_name: str, *, minimum: int | None = None) -> int:
	if isinstance(value, bool) or not isinstance(value, int):
		raise SchemaError(f"{field_name} must be an integer")
	if minimum is not None and value < minimum:
		raise SchemaError(f"{field_name} must be >= {minimum}")
	return value


def _require_float(
	value: Any,
	field_name: str,
	*,
	minimum: float | None = None,
	maximum: float | None = None,
) -> float:
	if isinstance(value, bool) or not isinstance(value, (int, float)):
		raise SchemaError(f"{field_name} must be a number")
	result = float(value)
	if not isfinite(result):
		raise SchemaError(f"{field_name} must be finite")
	if minimum is not None and result < minimum:
		raise SchemaError(f"{field_name} must be >= {minimum}")
	if maximum is not None and result > maximum:
		raise SchemaError(f"{field_name} must be <= {maximum}")
	return result


def _require_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
	if value is None:
		return ()
	if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
		raise SchemaError(f"{field_name} must be a sequence of strings")
	return tuple(_require_str(item, f"{field_name}[{index}]", allow_empty=False) for index, item in enumerate(value))


def _require_unknown_keys(mapping: Mapping[str, Any], allowed_keys: set[str], field_name: str) -> None:
	unknown = set(mapping.keys()) - allowed_keys
	if unknown:
		joined = ", ".join(sorted(unknown))
		raise SchemaError(f"{field_name} contains unknown keys: {joined}")


def _ordered_sparse_personality_vector(mapping: Mapping[str, Any], field_name: str) -> dict[str, float]:
	_require_unknown_keys(mapping, set(PERSONALITY_BASIS), field_name)
	ordered: dict[str, float] = {}
	for name in PERSONALITY_BASIS:
		if name in mapping:
			ordered[name] = _require_float(mapping[name], f"{field_name}.{name}", minimum=-1.0, maximum=1.0)
	return ordered


@dataclass(frozen=True, slots=True)
class PersonalitySnapshot:
	# --- 擴張組 The Drivers ---
	impulsiveness: float
	assertiveness: float
	optimism: float
	# --- 防禦組 The Stabilizers ---
	risk_aversion: float
	suspicion: float
	endurance: float
	# --- 擾動組 The Explorers ---
	randomness: float
	stability_seeking: float
	curiosity: float

	def __post_init__(self) -> None:
		for key in PERSONALITY_BASIS:
			_require_float(getattr(self, key), f"personality.{key}", minimum=-1.0, maximum=1.0)

	@classmethod
	def from_mapping(cls, payload: Mapping[str, Any] | PersonalitySnapshot) -> PersonalitySnapshot:
		if isinstance(payload, cls):
			return payload
		mapping = _require_mapping(payload, "personality")
		_require_unknown_keys(mapping, set(PERSONALITY_BASIS), "personality")
		return cls(
			impulsiveness=_require_float(mapping["impulsiveness"], "personality.impulsiveness", minimum=-1.0, maximum=1.0),
			assertiveness=_require_float(mapping["assertiveness"], "personality.assertiveness", minimum=-1.0, maximum=1.0),
			optimism=_require_float(mapping["optimism"], "personality.optimism", minimum=-1.0, maximum=1.0),
			risk_aversion=_require_float(mapping["risk_aversion"], "personality.risk_aversion", minimum=-1.0, maximum=1.0),
			suspicion=_require_float(mapping["suspicion"], "personality.suspicion", minimum=-1.0, maximum=1.0),
			endurance=_require_float(mapping["endurance"], "personality.endurance", minimum=-1.0, maximum=1.0),
			randomness=_require_float(mapping["randomness"], "personality.randomness", minimum=-1.0, maximum=1.0),
			stability_seeking=_require_float(mapping["stability_seeking"], "personality.stability_seeking", minimum=-1.0, maximum=1.0),
			curiosity=_require_float(mapping["curiosity"], "personality.curiosity", minimum=-1.0, maximum=1.0),
		)

	def to_dict(self) -> dict[str, float]:
		return {
			"impulsiveness": float(self.impulsiveness),
			"assertiveness": float(self.assertiveness),
			"optimism": float(self.optimism),
			"risk_aversion": float(self.risk_aversion),
			"suspicion": float(self.suspicion),
			"endurance": float(self.endurance),
			"randomness": float(self.randomness),
			"stability_seeking": float(self.stability_seeking),
			"curiosity": float(self.curiosity),
		}


@dataclass(frozen=True, slots=True)
class PersonalityDelta:
	vector: dict[str, float] = field(default_factory=dict)
	norm: float | None = None
	confidence: float | None = None

	def __post_init__(self) -> None:
		if self.vector:
			if not isinstance(self.vector, Mapping):
				raise SchemaError("personality_delta.vector must be a mapping")
			ordered = _ordered_sparse_personality_vector(self.vector, "personality_delta.vector")
			object.__setattr__(self, "vector", ordered)
			_require_float(self.norm, "personality_delta.norm", minimum=0.0)
			_require_float(self.confidence, "personality_delta.confidence", minimum=0.0, maximum=1.0)
		else:
			if self.norm is not None or self.confidence is not None:
				raise SchemaError("personality_delta must be empty when vector is empty")

	@classmethod
	def from_mapping(cls, payload: Mapping[str, Any] | PersonalityDelta | None) -> PersonalityDelta:
		if payload is None:
			return cls()
		if isinstance(payload, cls):
			return payload
		mapping = _require_mapping(payload, "personality_delta")
		if not mapping:
			return cls()
		_require_unknown_keys(mapping, {"vector", "norm", "confidence"}, "personality_delta")
		vector = mapping.get("vector")
		if vector is None:
			raise SchemaError("personality_delta.vector is required")
		vector_mapping = _require_mapping(vector, "personality_delta.vector")
		return cls(
			vector=_ordered_sparse_personality_vector(vector_mapping, "personality_delta.vector"),
			norm=_require_float(mapping.get("norm"), "personality_delta.norm", minimum=0.0),
			confidence=_require_float(mapping.get("confidence"), "personality_delta.confidence", minimum=0.0, maximum=1.0),
		)

	def to_dict(self) -> dict[str, Any]:
		if not self.vector:
			return {}
		return {
			"vector": {key: float(value) for key, value in self.vector.items()},
			"norm": float(self.norm if self.norm is not None else 0.0),
			"confidence": float(self.confidence if self.confidence is not None else 0.0),
		}


@dataclass(frozen=True, slots=True)
class ChoicePreview:
	expected_reward: float
	expected_risk: float
	confidence: float
	expected_personality_delta: PersonalityDelta | None = None

	def __post_init__(self) -> None:
		_require_float(self.expected_reward, "choice_preview.expected_reward")
		_require_float(self.expected_risk, "choice_preview.expected_risk")
		_require_float(self.confidence, "choice_preview.confidence", minimum=0.0, maximum=1.0)
		if self.expected_personality_delta is not None and not isinstance(self.expected_personality_delta, PersonalityDelta):
			raise SchemaError("choice_preview.expected_personality_delta must be a PersonalityDelta or None")

	@classmethod
	def from_mapping(cls, payload: Mapping[str, Any] | ChoicePreview | None) -> ChoicePreview | None:
		if payload is None:
			return None
		if isinstance(payload, cls):
			return payload
		mapping = _require_mapping(payload, "choice_preview")
		if not mapping:
			return None
		_require_unknown_keys(mapping, {"expected_reward", "expected_risk", "expected_personality_delta", "confidence"}, "choice_preview")
		return cls(
			expected_reward=_require_float(mapping["expected_reward"], "choice_preview.expected_reward"),
			expected_risk=_require_float(mapping["expected_risk"], "choice_preview.expected_risk"),
			confidence=_require_float(mapping["confidence"], "choice_preview.confidence", minimum=0.0, maximum=1.0),
			expected_personality_delta=None
			if mapping.get("expected_personality_delta") is None
			else PersonalityDelta.from_mapping(mapping.get("expected_personality_delta")),
		)

	def to_dict(self) -> dict[str, Any]:
		return {
			"expected_reward": float(self.expected_reward),
			"expected_risk": float(self.expected_risk),
			"expected_personality_delta": None if self.expected_personality_delta is None else self.expected_personality_delta.to_dict(),
			"confidence": float(self.confidence),
		}


@dataclass(frozen=True, slots=True)
class EventChoice:
	choice_id: str
	label: str
	strategy: StrategyName
	enabled: bool
	selected: bool
	description: str | None = None
	preview: ChoicePreview | None = None

	def __post_init__(self) -> None:
		_require_str(self.choice_id, "choice.choice_id")
		_require_str(self.label, "choice.label")
		strategy = _require_str(self.strategy, "choice.strategy")
		if strategy not in STRATEGIES:
			raise SchemaError(f"choice.strategy must be one of {', '.join(STRATEGIES)}")
		_require_bool(self.enabled, "choice.enabled")
		_require_bool(self.selected, "choice.selected")
		if self.description is not None:
			_require_str(self.description, "choice.description", allow_empty=True)
		if self.preview is not None and not isinstance(self.preview, ChoicePreview):
			raise SchemaError("choice.preview must be a ChoicePreview or None")

	@classmethod
	def from_mapping(cls, payload: Mapping[str, Any] | EventChoice) -> EventChoice:
		if isinstance(payload, cls):
			return payload
		mapping = _require_mapping(payload, "choice")
		_require_unknown_keys(mapping, {"choice_id", "label", "strategy", "enabled", "selected", "description", "preview"}, "choice")
		preview = ChoicePreview.from_mapping(mapping.get("preview"))
		return cls(
			choice_id=_require_str(mapping["choice_id"], "choice.choice_id"),
			label=_require_str(mapping["label"], "choice.label"),
			strategy=cast(StrategyName, _require_str(mapping["strategy"], "choice.strategy")),
			enabled=_require_bool(mapping["enabled"], "choice.enabled"),
			selected=_require_bool(mapping["selected"], "choice.selected"),
			description=None if mapping.get("description") is None else _require_str(mapping["description"], "choice.description", allow_empty=True),
			preview=preview,
		)

	def to_dict(self) -> dict[str, Any]:
		return {
			"choice_id": self.choice_id,
			"label": self.label,
			"strategy": self.strategy,
			"enabled": bool(self.enabled),
			"selected": bool(self.selected),
			"description": self.description,
			"preview": None if self.preview is None else self.preview.to_dict(),
		}


@dataclass(frozen=True, slots=True)
class EventPayload:
	event_id: str
	event_type: str
	title: str
	summary: str
	tags: tuple[str, ...] = field(default_factory=tuple)
	severity: float = 0.0
	phase: str | None = None

	def __post_init__(self) -> None:
		_require_str(self.event_id, "event.event_id")
		_require_str(self.event_type, "event.event_type")
		_require_str(self.title, "event.title")
		_require_str(self.summary, "event.summary")
		object.__setattr__(self, "tags", _require_string_tuple(self.tags, "event.tags"))
		_require_float(self.severity, "event.severity", minimum=0.0, maximum=1.0)
		if self.phase is not None:
			_require_str(self.phase, "event.phase", allow_empty=True)

	@classmethod
	def from_mapping(cls, payload: Mapping[str, Any] | EventPayload) -> EventPayload:
		if isinstance(payload, cls):
			return payload
		mapping = _require_mapping(payload, "event")
		_require_unknown_keys(mapping, {"event_id", "event_type", "title", "summary", "tags", "severity", "phase"}, "event")
		return cls(
			event_id=_require_str(mapping["event_id"], "event.event_id"),
			event_type=_require_str(mapping["event_type"], "event.event_type"),
			title=_require_str(mapping["title"], "event.title"),
			summary=_require_str(mapping["summary"], "event.summary"),
			tags=_require_string_tuple(mapping.get("tags"), "event.tags"),
			severity=_require_float(mapping["severity"], "event.severity", minimum=0.0, maximum=1.0),
			phase=None if mapping.get("phase") is None else _require_str(mapping["phase"], "event.phase", allow_empty=True),
		)

	def to_dict(self) -> dict[str, Any]:
		return {
			"event_id": self.event_id,
			"event_type": self.event_type,
			"title": self.title,
			"summary": self.summary,
			"tags": list(self.tags),
			"severity": float(self.severity),
			"phase": self.phase,
		}


@dataclass(frozen=True, slots=True)
class EmotionState:
	valence: float | None = None
	arousal: float | None = None
	dominance: float | None = None
	intensity: float | None = None

	def __post_init__(self) -> None:
		values = (self.valence, self.arousal, self.dominance, self.intensity)
		if all(value is None for value in values):
			return
		if any(value is None for value in values):
			raise SchemaError("emotion must be either empty or fully populated")
		_require_float(self.valence, "emotion.valence", minimum=-1.0, maximum=1.0)
		_require_float(self.arousal, "emotion.arousal", minimum=0.0, maximum=1.0)
		_require_float(self.dominance, "emotion.dominance", minimum=0.0, maximum=1.0)
		_require_float(self.intensity, "emotion.intensity", minimum=0.0, maximum=1.0)

	@classmethod
	def from_mapping(cls, payload: Mapping[str, Any] | EmotionState | None) -> EmotionState:
		if payload is None:
			return cls()
		if isinstance(payload, cls):
			return payload
		mapping = _require_mapping(payload, "emotion")
		if not mapping:
			return cls()
		_require_unknown_keys(mapping, {"valence", "arousal", "dominance", "intensity"}, "emotion")
		return cls(
			valence=_require_float(mapping["valence"], "emotion.valence", minimum=-1.0, maximum=1.0),
			arousal=_require_float(mapping["arousal"], "emotion.arousal", minimum=0.0, maximum=1.0),
			dominance=_require_float(mapping["dominance"], "emotion.dominance", minimum=0.0, maximum=1.0),
			intensity=_require_float(mapping["intensity"], "emotion.intensity", minimum=0.0, maximum=1.0),
		)

	def to_dict(self) -> dict[str, Any]:
		if self.valence is None and self.arousal is None and self.dominance is None and self.intensity is None:
			return {}
		return {
			"valence": float(self.valence if self.valence is not None else 0.0),
			"arousal": float(self.arousal if self.arousal is not None else 0.0),
			"dominance": float(self.dominance if self.dominance is not None else 0.0),
			"intensity": float(self.intensity if self.intensity is not None else 0.0),
		}


@dataclass(frozen=True, slots=True)
class WorldState:
	scarcity: float | None = None
	threat: float | None = None
	noise: float | None = None
	intel: float | None = None

	def __post_init__(self) -> None:
		values = (self.scarcity, self.threat, self.noise, self.intel)
		if all(value is None for value in values):
			return
		if any(value is None for value in values):
			raise SchemaError("world_state must be either empty or fully populated")
		_require_float(self.scarcity, "world_state.scarcity", minimum=0.0, maximum=1.0)
		_require_float(self.threat, "world_state.threat", minimum=0.0, maximum=1.0)
		_require_float(self.noise, "world_state.noise", minimum=0.0, maximum=1.0)
		_require_float(self.intel, "world_state.intel", minimum=0.0, maximum=1.0)

	@classmethod
	def from_mapping(cls, payload: Mapping[str, Any] | WorldState | None) -> WorldState:
		if payload is None:
			return cls()
		if isinstance(payload, cls):
			return payload
		mapping = _require_mapping(payload, "world_state")
		if not mapping:
			return cls()
		_require_unknown_keys(mapping, {"scarcity", "threat", "noise", "intel"}, "world_state")
		return cls(
			scarcity=_require_float(mapping["scarcity"], "world_state.scarcity", minimum=0.0, maximum=1.0),
			threat=_require_float(mapping["threat"], "world_state.threat", minimum=0.0, maximum=1.0),
			noise=_require_float(mapping["noise"], "world_state.noise", minimum=0.0, maximum=1.0),
			intel=_require_float(mapping["intel"], "world_state.intel", minimum=0.0, maximum=1.0),
		)

	def to_dict(self) -> dict[str, Any]:
		if self.scarcity is None and self.threat is None and self.noise is None and self.intel is None:
			return {}
		return {
			"scarcity": float(self.scarcity if self.scarcity is not None else 0.0),
			"threat": float(self.threat if self.threat is not None else 0.0),
			"noise": float(self.noise if self.noise is not None else 0.0),
			"intel": float(self.intel if self.intel is not None else 0.0),
		}


@dataclass(frozen=True, slots=True)
class PlayerState:
	health: float | None = None
	stamina: float | None = None
	stress: float | None = None
	utility: float | None = None
	alive: bool | None = None
	personality: PersonalitySnapshot | None = None
	memory_load: float | None = None

	def __post_init__(self) -> None:
		values = (self.health, self.stamina, self.stress, self.utility, self.alive, self.personality)
		if all(value is None for value in values):
			return
		if any(value is None for value in values):
			raise SchemaError("player_state must be either empty or fully populated")
		_require_float(self.health, "player_state.health", minimum=0.0, maximum=1.0)
		_require_float(self.stamina, "player_state.stamina", minimum=0.0, maximum=1.0)
		_require_float(self.stress, "player_state.stress", minimum=0.0, maximum=1.0)
		_require_float(self.utility, "player_state.utility")
		_require_bool(self.alive, "player_state.alive")
		if not isinstance(self.personality, PersonalitySnapshot):
			raise SchemaError("player_state.personality must be a PersonalitySnapshot")
		if self.memory_load is not None:
			_require_float(self.memory_load, "player_state.memory_load", minimum=0.0, maximum=1.0)

	@classmethod
	def from_mapping(cls, payload: Mapping[str, Any] | PlayerState | None) -> PlayerState:
		if payload is None:
			return cls()
		if isinstance(payload, cls):
			return payload
		mapping = _require_mapping(payload, "player_state")
		if not mapping:
			return cls()
		_require_unknown_keys(mapping, {"health", "stamina", "stress", "utility", "alive", "personality", "memory_load"}, "player_state")
		return cls(
			health=_require_float(mapping["health"], "player_state.health", minimum=0.0, maximum=1.0),
			stamina=_require_float(mapping["stamina"], "player_state.stamina", minimum=0.0, maximum=1.0),
			stress=_require_float(mapping["stress"], "player_state.stress", minimum=0.0, maximum=1.0),
			utility=_require_float(mapping["utility"], "player_state.utility"),
			alive=_require_bool(mapping["alive"], "player_state.alive"),
			personality=PersonalitySnapshot.from_mapping(_require_mapping(mapping["personality"], "player_state.personality")),
			memory_load=None if mapping.get("memory_load") is None else _require_float(mapping["memory_load"], "player_state.memory_load", minimum=0.0, maximum=1.0),
		)

	def to_dict(self) -> dict[str, Any]:
		if self.health is None and self.stamina is None and self.stress is None and self.utility is None and self.alive is None and self.personality is None and self.memory_load is None:
			return {}
		return {
			"health": float(self.health if self.health is not None else 0.0),
			"stamina": float(self.stamina if self.stamina is not None else 0.0),
			"stress": float(self.stress if self.stress is not None else 0.0),
			"utility": float(self.utility if self.utility is not None else 0.0),
			"alive": bool(self.alive),
			"personality": None if self.personality is None else self.personality.to_dict(),
			"memory_load": None if self.memory_load is None else float(self.memory_load),
		}


@dataclass(frozen=True, slots=True)
class ResultState:
	selected_choice_id: str | None
	reward: float
	utility_delta: float
	risk_delta: float
	terminated: bool
	termination_reason: str | None = None
	state_delta: Mapping[str, Any] | None = None

	def __post_init__(self) -> None:
		if self.selected_choice_id is not None:
			_require_str(self.selected_choice_id, "result.selected_choice_id")
		_require_float(self.reward, "result.reward")
		_require_float(self.utility_delta, "result.utility_delta")
		_require_float(self.risk_delta, "result.risk_delta")
		_require_bool(self.terminated, "result.terminated")
		if self.termination_reason is not None:
			_require_str(self.termination_reason, "result.termination_reason", allow_empty=True)
		if self.state_delta is not None:
			_require_mapping(self.state_delta, "result.state_delta")

	@classmethod
	def from_mapping(cls, payload: Mapping[str, Any] | ResultState | None) -> ResultState | None:
		if payload is None:
			return None
		if isinstance(payload, cls):
			return payload
		mapping = _require_mapping(payload, "result")
		if not mapping:
			raise SchemaError("result cannot be an empty mapping")
		_require_unknown_keys(mapping, {"selected_choice_id", "reward", "utility_delta", "risk_delta", "terminated", "termination_reason", "state_delta"}, "result")
		state_delta = mapping.get("state_delta")
		if state_delta is not None:
			state_delta = dict(_require_mapping(state_delta, "result.state_delta"))
		return cls(
			selected_choice_id=None if mapping.get("selected_choice_id") is None else _require_str(mapping["selected_choice_id"], "result.selected_choice_id"),
			reward=_require_float(mapping["reward"], "result.reward"),
			utility_delta=_require_float(mapping["utility_delta"], "result.utility_delta"),
			risk_delta=_require_float(mapping["risk_delta"], "result.risk_delta"),
			terminated=_require_bool(mapping["terminated"], "result.terminated"),
			termination_reason=None if mapping.get("termination_reason") is None else _require_str(mapping["termination_reason"], "result.termination_reason", allow_empty=True),
			state_delta=state_delta,
		)

	def to_dict(self) -> dict[str, Any]:
		return {
			"selected_choice_id": self.selected_choice_id,
			"reward": float(self.reward),
			"utility_delta": float(self.utility_delta),
			"risk_delta": float(self.risk_delta),
			"terminated": bool(self.terminated),
			"termination_reason": self.termination_reason,
			"state_delta": None if self.state_delta is None else dict(self.state_delta),
		}


@dataclass(frozen=True, slots=True)
class ErrorState:
	code: str
	message: str
	retryable: bool
	details: Mapping[str, Any] | None = None

	def __post_init__(self) -> None:
		_require_str(self.code, "error.code")
		_require_str(self.message, "error.message")
		_require_bool(self.retryable, "error.retryable")
		if self.details is not None:
			_require_mapping(self.details, "error.details")

	@classmethod
	def from_mapping(cls, payload: Mapping[str, Any] | ErrorState | None) -> ErrorState | None:
		if payload is None:
			return None
		if isinstance(payload, cls):
			return payload
		mapping = _require_mapping(payload, "error")
		if not mapping:
			raise SchemaError("error cannot be an empty mapping")
		_require_unknown_keys(mapping, {"code", "message", "retryable", "details"}, "error")
		details = mapping.get("details")
		if details is not None:
			details = dict(_require_mapping(details, "error.details"))
		return cls(
			code=_require_str(mapping["code"], "error.code"),
			message=_require_str(mapping["message"], "error.message"),
			retryable=_require_bool(mapping["retryable"], "error.retryable"),
			details=details,
		)

	def to_dict(self) -> dict[str, Any]:
		return {
			"code": self.code,
			"message": self.message,
			"retryable": bool(self.retryable),
			"details": None if self.details is None else dict(self.details),
		}


@dataclass(frozen=True, slots=True)
class ResponseEnvelope:
	api_version: str
	kind: ResponseKind
	ok: bool
	session_id: str
	tick: int
	state_hash: str | None = None
	event: EventPayload | None = None
	choices: tuple[EventChoice, ...] = field(default_factory=tuple)
	personality_delta: PersonalityDelta = field(default_factory=PersonalityDelta)
	emotion: EmotionState = field(default_factory=EmotionState)
	world_state: WorldState = field(default_factory=WorldState)
	player_state: PlayerState = field(default_factory=PlayerState)
	result: ResultState | None = None
	warnings: tuple[str, ...] = field(default_factory=tuple)
	error: ErrorState | None = None
	extensions: dict[str, Any] = field(default_factory=dict)

	def __post_init__(self) -> None:
		api_version = _require_str(self.api_version, "api_version")
		if api_version != API_VERSION:
			raise SchemaError(f"api_version must be {API_VERSION!r}")
		kind = _require_str(self.kind, "kind")
		if kind not in RESPONSE_KINDS:
			raise SchemaError(f"kind must be one of {', '.join(RESPONSE_KINDS)}")
		_require_bool(self.ok, "ok")
		_require_str(self.session_id, "session_id")
		_require_int(self.tick, "tick", minimum=0)
		if self.state_hash is not None:
			_require_str(self.state_hash, "state_hash")
		if self.event is not None and not isinstance(self.event, EventPayload):
			raise SchemaError("event must be an EventPayload or None")
		if isinstance(self.choices, (str, bytes)) or not isinstance(self.choices, Sequence):
			raise SchemaError("choices must be a sequence")
		if not isinstance(self.personality_delta, PersonalityDelta):
			raise SchemaError("personality_delta must be a PersonalityDelta")
		if not isinstance(self.emotion, EmotionState):
			raise SchemaError("emotion must be an EmotionState")
		if not isinstance(self.world_state, WorldState):
			raise SchemaError("world_state must be a WorldState")
		if not isinstance(self.player_state, PlayerState):
			raise SchemaError("player_state must be a PlayerState")
		if self.result is not None and not isinstance(self.result, ResultState):
			raise SchemaError("result must be a ResultState or None")
		if self.error is not None and not isinstance(self.error, ErrorState):
			raise SchemaError("error must be an ErrorState or None")
		object.__setattr__(self, "choices", tuple(EventChoice.from_mapping(choice) for choice in self.choices))
		object.__setattr__(self, "warnings", _require_string_tuple(self.warnings, "warnings"))
		if not isinstance(self.extensions, Mapping):
			raise SchemaError("extensions must be a mapping")
		object.__setattr__(self, "extensions", dict(self.extensions))
		for key in self.extensions:
			_require_str(key, "extensions key")
		choice_ids = [choice.choice_id for choice in self.choices]
		if len(choice_ids) != len(set(choice_ids)):
			raise SchemaError("choices must have unique choice_id values")
		if self.kind == "error" and self.error is None:
			raise SchemaError("error responses must include an error payload")
		if not self.ok and self.kind != "error":
			raise SchemaError("failed responses must use kind='error'")
		if self.ok and self.kind == "error":
			raise SchemaError("kind='error' requires ok=False")
		if self.error is not None and self.ok:
			raise SchemaError("error payloads require ok=False")

	@classmethod
	def from_mapping(cls, payload: Mapping[str, Any] | ResponseEnvelope) -> ResponseEnvelope:
		if isinstance(payload, cls):
			return payload
		mapping = _require_mapping(payload, "response")
		_require_unknown_keys(
			mapping,
			{
				"api_version",
				"kind",
				"ok",
				"session_id",
				"tick",
				"state_hash",
				"event",
				"choices",
				"personality_delta",
				"emotion",
				"world_state",
				"player_state",
				"result",
				"warnings",
				"error",
				"extensions",
			},
			"response",
		)
		choices_data = mapping.get("choices") or []
		if isinstance(choices_data, (str, bytes)) or not isinstance(choices_data, Sequence):
			raise SchemaError("choices must be a sequence")
		extensions = mapping.get("extensions") or {}
		if not isinstance(extensions, Mapping):
			raise SchemaError("extensions must be a mapping")
		return cls(
			api_version=_require_str(mapping["api_version"], "api_version"),
			kind=cast(ResponseKind, _require_str(mapping["kind"], "kind")),
			ok=_require_bool(mapping["ok"], "ok"),
			session_id=_require_str(mapping["session_id"], "session_id"),
			tick=_require_int(mapping["tick"], "tick", minimum=0),
			state_hash=None if mapping.get("state_hash") is None else _require_str(mapping["state_hash"], "state_hash"),
			event=None if mapping.get("event") is None else EventPayload.from_mapping(mapping["event"]),
			choices=tuple(EventChoice.from_mapping(choice) for choice in choices_data),
			personality_delta=PersonalityDelta.from_mapping(mapping.get("personality_delta")),
			emotion=EmotionState.from_mapping(mapping.get("emotion")),
			world_state=WorldState.from_mapping(mapping.get("world_state")),
			player_state=PlayerState.from_mapping(mapping.get("player_state")),
			result=ResultState.from_mapping(mapping.get("result")),
			warnings=_require_string_tuple(mapping.get("warnings"), "warnings"),
			error=ErrorState.from_mapping(mapping.get("error")),
			extensions=dict(extensions),
		)

	def to_dict(self) -> dict[str, Any]:
		return {
			"api_version": self.api_version,
			"kind": self.kind,
			"ok": bool(self.ok),
			"session_id": self.session_id,
			"tick": int(self.tick),
			"state_hash": self.state_hash,
			"event": None if self.event is None else self.event.to_dict(),
			"choices": [choice.to_dict() for choice in self.choices],
			"personality_delta": self.personality_delta.to_dict(),
			"emotion": self.emotion.to_dict(),
			"world_state": self.world_state.to_dict(),
			"player_state": self.player_state.to_dict(),
			"result": None if self.result is None else self.result.to_dict(),
			"warnings": list(self.warnings),
			"error": None if self.error is None else self.error.to_dict(),
			"extensions": dict(self.extensions),
		}


ResetResponse = ResponseEnvelope
StepResponse = ResponseEnvelope
SnapshotResponse = ResponseEnvelope
ErrorResponse = ResponseEnvelope


def normalize_response_envelope(payload: Mapping[str, Any] | ResponseEnvelope) -> ResponseEnvelope:
	"""Coerce a loose mapping into a validated response envelope."""

	return ResponseEnvelope.from_mapping(payload)


__all__ = [
	"API_VERSION",
	"PERSONALITY_BASIS",
	"RESPONSE_KINDS",
	"STRATEGIES",
	"SchemaError",
	"PersonalitySnapshot",
	"PersonalityDelta",
	"ChoicePreview",
	"EventChoice",
	"EventPayload",
	"EmotionState",
	"WorldState",
	"PlayerState",
	"ResultState",
	"ErrorState",
	"ResponseEnvelope",
	"ResetResponse",
	"StepResponse",
	"SnapshotResponse",
	"ErrorResponse",
	"normalize_response_envelope",
]


