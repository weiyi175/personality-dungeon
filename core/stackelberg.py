from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class StackelbergCommitment:
	condition: str
	leader_action: str
	a: float
	b: float
	matrix_cross_coupling: float
	description: str = ""


@dataclass(frozen=True)
class FollowerResponse:
	condition: str
	level3_seed_count: int
	mean_stage3_score: float
	mean_env_gamma: float
	mean_turn_strength: float


def response_from_mapping(row: Mapping[str, Any]) -> FollowerResponse:
	return FollowerResponse(
		condition=str(row.get("condition", "")),
		level3_seed_count=int(row.get("level3_seed_count", 0)),
		mean_stage3_score=float(row.get("mean_stage3_score", 0.0)),
		mean_env_gamma=float(row.get("mean_env_gamma", 0.0)),
		mean_turn_strength=float(row.get("mean_turn_strength", 0.0)),
	)


def response_rank_key(response: FollowerResponse) -> tuple[int, float, float, float]:
	return (
		int(response.level3_seed_count),
		float(response.mean_stage3_score),
		float(response.mean_env_gamma),
		float(response.mean_turn_strength),
	)


def leader_prefers(candidate: Mapping[str, Any] | FollowerResponse, baseline: Mapping[str, Any] | FollowerResponse) -> bool:
	left = candidate if isinstance(candidate, FollowerResponse) else response_from_mapping(candidate)
	right = baseline if isinstance(baseline, FollowerResponse) else response_from_mapping(baseline)
	return response_rank_key(left) > response_rank_key(right)


def dominance_gap_from_shares(*, aggressive: float, defensive: float, balanced: float) -> float:
	values = [float(aggressive), float(defensive), float(balanced)]
	return max(values) - min(values)


def ema_step(*, previous: float | None, current: float, alpha: float) -> float:
	if previous is None:
		return float(current)
	weight = float(alpha)
	return (1.0 - weight) * float(previous) + weight * float(current)


def dominant_strategy_from_shares(*, aggressive: float, defensive: float, balanced: float) -> str:
	candidates = [
		("aggressive", float(aggressive), 0),
		("defensive", float(defensive), 1),
		("balanced", float(balanced), 2),
	]
	return max(candidates, key=lambda item: (item[1], -item[2]))[0]


def hysteresis_gate(*, value: float, current_active: bool, theta_low: float, theta_high: float) -> bool:
	metric = float(value)
	if metric >= float(theta_high):
		return True
	if metric <= float(theta_low):
		return False
	return bool(current_active)


def select_best_commitment(rows: Sequence[Mapping[str, Any]]) -> str | None:
	candidates = [row for row in rows if str(row.get("is_control", "no")) != "yes"]
	if not candidates:
		return None
	best_row = max(candidates, key=lambda row: response_rank_key(response_from_mapping(row)))
	return str(best_row.get("condition", "")) or None

