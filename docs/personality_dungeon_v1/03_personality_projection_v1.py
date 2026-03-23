from __future__ import annotations

from math import exp
from typing import Dict, Iterable, List, Mapping


DIMENSIONS: List[str] = [
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


PRIMARY_GROUPS: Dict[str, List[str]] = {
    "aggressive": ["impulsiveness", "greed", "ambition"],
    "defensive": ["caution", "suspicion", "fearfulness"],
    "balanced": ["persistence", "stability_seeking", "patience"],
}


SECONDARY_MODIFIERS: Dict[str, Dict[str, float]] = {
    "aggressive": {"optimism": 0.15, "curiosity": 0.10, "randomness": 0.05},
    "defensive": {"optimism": -0.05, "curiosity": 0.05, "randomness": -0.05},
    "balanced": {"optimism": 0.05, "curiosity": 0.10, "randomness": -0.10},
}


def clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def normalize_personality(personality: Mapping[str, float]) -> Dict[str, float]:
    normalized = {key: 0.0 for key in DIMENSIONS}
    for key, value in personality.items():
        if key in normalized:
            normalized[key] = clamp(float(value))
    return normalized


def group_score(personality: Mapping[str, float], group: str) -> float:
    keys = PRIMARY_GROUPS[group]
    base = sum(float(personality[key]) for key in keys) / len(keys)
    modifier = sum(
        float(personality[key]) * weight
        for key, weight in SECONDARY_MODIFIERS[group].items()
    )
    return base + modifier


def softmax(values: Iterable[float]) -> List[float]:
    values = list(values)
    max_value = max(values)
    exps = [exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def project_to_simplex(personality: Mapping[str, float]) -> Dict[str, float]:
    normalized = normalize_personality(personality)
    raw_scores = {
        group: group_score(normalized, group) for group in PRIMARY_GROUPS
    }
    probs = softmax(raw_scores.values())
    return dict(zip(raw_scores.keys(), probs))


def project_vector(values: List[float]) -> Dict[str, float]:
    if len(values) != len(DIMENSIONS):
        raise ValueError(f"Expected {len(DIMENSIONS)} dimensions, got {len(values)}")
    personality = dict(zip(DIMENSIONS, values))
    return project_to_simplex(personality)


def prompt_delta_template() -> Dict[str, float]:
    return {key: 0.0 for key in DIMENSIONS}