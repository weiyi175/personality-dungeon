from __future__ import annotations

from typing import Dict, Mapping


DEFAULT_A = 0.4
DEFAULT_B = 0.24254
BMA_K_INF = 0.9407

K50_TABLE = {
    50: 0.7560,
    100: 1.0949,
    200: 0.9535,
    300: 0.8580,
    500: 0.9109,
    1000: 0.9237,
}


def interpolate_k50(players: int) -> float:
    ordered = sorted(K50_TABLE.items())
    if players <= ordered[0][0]:
        return ordered[0][1]
    if players >= ordered[-1][0]:
        return BMA_K_INF

    for (left_n, left_k), (right_n, right_k) in zip(ordered, ordered[1:]):
        if left_n <= players <= right_n:
            span = right_n - left_n
            weight = (players - left_n) / span
            return left_k + weight * (right_k - left_k)
    return BMA_K_INF


def dominant_strategy(global_p: Mapping[str, float]) -> str:
    return max(global_p.items(), key=lambda item: item[1])[0]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def generate_adaptive_dungeon(
    global_p: Mapping[str, float],
    selection_strength: float,
    players: int,
    base_a: float = DEFAULT_A,
    base_b: float = DEFAULT_B,
) -> Dict[str, float | str]:
    required = {"aggressive", "defensive", "balanced"}
    if set(global_p.keys()) != required:
        raise ValueError(f"global_p must contain exactly {sorted(required)}")

    p_a = float(global_p["aggressive"])
    p_d = float(global_p["defensive"])
    p_b = float(global_p["balanced"])

    local_k50 = interpolate_k50(players)
    bias = max(p_a, p_d, p_b) - (1.0 / 3.0)
    pressure = (selection_strength / max(local_k50, 1e-6)) * max(bias, 0.0)
    dom = dominant_strategy(global_p)

    a_new = base_a
    b_new = base_b
    event_type = "balanced"

    if bias < 0.02:
        event_type = "mixed"
    elif dom == "aggressive":
        a_new = base_a + 0.18 * pressure
        b_new = base_b + 0.10 * pressure
        event_type = "Threat"
    elif dom == "defensive":
        a_new = base_a - 0.12 * pressure
        b_new = base_b + 0.22 * pressure
        event_type = "Resource"
    else:
        a_new = base_a + 0.15 * pressure
        b_new = base_b - 0.15 * pressure
        event_type = "Uncertainty"

    a_new = clamp(a_new, 0.1, 1.2)
    b_new = clamp(b_new, 0.05, 1.2)

    risk_drift = clamp(0.001 * selection_strength * max(bias, 0.0), 0.0, 0.05)
    threshold_multiplier = clamp(1.0 + 0.3 * max(bias, 0.0), 1.0, 1.5)

    return {
        "players": players,
        "selection_strength": round(selection_strength, 4),
        "local_k50_anchor": round(local_k50, 4),
        "bma_k_inf": BMA_K_INF,
        "dominant_strategy": dom,
        "dominance_bias": round(bias, 4),
        "pressure": round(pressure, 4),
        "a": round(a_new, 4),
        "b": round(b_new, 4),
        "event_type": event_type,
        "risk_drift": round(risk_drift, 5),
        "threshold_multiplier": round(threshold_multiplier, 4),
    }