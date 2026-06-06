"""
Event generator for bifurcation-driven personality control.

Design logic (P7-H Application Design):
- Events are aligned with the primary sensitive direction (v1).
- Event magnitude scales inversely with bifurcation proximity:
  closer to the critical point → weaker nudge needed.
- Sequences of events accumulate to push the trajectory past ε_c.
"""

from __future__ import annotations

import numpy as np

from simulation.bifurcation_detector import (
    BASELINE_ATTRACTOR,
    EPSILON_C,
    EPSILON_C_APP,
    FEATURE_NAMES,
    compute_bifurcation_distance,
    compute_sensitive_direction,
)

# ── Calibrated defaults (P7-H Phase II verification, N=50 trials) ────────────
# Minimum cost to reach critical proximity (>0.8) with 100% success rate:
#   n_steps=3, intensity_scale=1.0  → 100% success, mean_peak=1.000, adv=+0.298
# Alignment advantage over random direction: +0.17 average (larger for n_steps≤5)
RECOMMENDED_N_STEPS: int = 3
RECOMMENDED_INTENSITY_SCALE: float = 1.0

# ── Event type catalogue ─────────────────────────────────────────────────────

# Each entry maps an event name to a 9D personality-space displacement vector
# (signs chosen from P7-F SVD interpretation).
_EVENT_TEMPLATES: dict[str, np.ndarray] = {
    # Push toward curiosity / impulsiveness end of v1
    "explore_risk": np.array([
        0.3, 0.1, 0.2, -0.4, -0.1, -0.1, 0.3, -0.2, 0.4,
    ]),
    # Push toward risk_aversion end of v1
    "enforce_caution": np.array([
        -0.3, -0.1, -0.2, 0.4, 0.1, 0.1, -0.3, 0.2, -0.4,
    ]),
    # v2 axis: reduce suspicion, increase endurance (韌性路線)
    "build_resilience": np.array([
        0.1, 0.1, 0.1, 0.1, -0.5, -0.4, 0.1, 0.1, 0.1,
    ]),
    # v2 axis: increase suspicion, decrease endurance (崩潰路線)
    "induce_distrust": np.array([
        -0.1, -0.1, -0.1, -0.1, 0.5, 0.4, -0.1, -0.1, -0.1,
    ]),
    # Generic: push along primary sensitive direction
    "personality_shift": None,  # filled dynamically from sensitive direction
}


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ── Core API ─────────────────────────────────────────────────────────────────

def design_bifurcation_event(
    personality_vector: list[float] | np.ndarray,
    target: str = "personality_shift",
    intensity_scale: float = 1.0,
    app_calibrated: bool = False,
    direction_mode: str = "aligned",
    rng: np.random.RandomState | None = None,
) -> dict:
    """
    Design a single game event to nudge the personality toward bifurcation.

    Parameters
    ----------
    personality_vector : 9D array of current personality traits
    target             : event type key in _EVENT_TEMPLATES
    intensity_scale    : multiplier applied on top of automatic modulation
    direction_mode :
        "aligned" – EXPERIMENT group. Perturb along the primary sensitive
                    direction v1 (or the requested template).
        "random"  – CONTROL group. Perturb along a uniformly random unit
                    direction of the SAME magnitude, so the two groups differ
                    only in direction, never in force. This is the actual A/B
                    manipulation — the control arm must use this mode.
    rng :
        Random source for ``direction_mode="random"`` (reproducibility). A fresh
        default RandomState is used when None.

    Returns
    -------
    dict with keys:
      type            – event type string
      direction       – unit 9D vector of the perturbation
      magnitude       – scalar displacement magnitude
      duration_rounds – suggested simulation rounds to apply the event
      proximity       – bifurcation proximity at design time
      displacement    – full displacement vector (direction × magnitude)
      feature_deltas  – per-feature displacement dict (for readability)
    """
    pv = np.asarray(personality_vector, dtype=float)
    # App profile: projection-based proximity (v1-v2 plane, ε_c_app) so events
    # operate on the realistic personality scale instead of saturating at 0.007.
    _mode = "projection" if app_calibrated else "euclidean"
    _scale = EPSILON_C_APP if app_calibrated else EPSILON_C
    bifurc = compute_bifurcation_distance(pv, mode=_mode)
    sens = compute_sensitive_direction(pv)
    proximity = bifurc["bifurcation_proximity"]

    # Closer to bifurcation → less force required
    auto_magnitude = intensity_modulation(proximity, scale=_scale) * intensity_scale

    if direction_mode == "random":
        # Control arm: random unit direction, identical magnitude to experiment.
        _rng = rng if rng is not None else np.random.RandomState()
        raw = _rng.randn(9)
        direction = _unit(raw)
    elif target == "personality_shift" or target not in _EVENT_TEMPLATES:
        direction = np.array(sens["direction"])
    else:
        template = _EVENT_TEMPLATES[target]
        direction = _unit(template)

    displacement = direction * auto_magnitude
    duration = max(1, int(10 * (1.0 - proximity)) + 1)

    return {
        "type": target,
        "direction_mode": direction_mode,
        "direction": direction.tolist(),
        "magnitude": float(auto_magnitude),
        "duration_rounds": duration,
        "proximity": proximity,
        "displacement": displacement.tolist(),
        "feature_deltas": dict(zip(FEATURE_NAMES, displacement.tolist())),
    }


def event_sequence_planner(
    personality_vector: list[float] | np.ndarray,
    n_steps: int = 5,
    target: str = "personality_shift",
    app_calibrated: bool = False,
    direction_mode: str = "aligned",
    seed: int | None = None,
) -> dict:
    """
    Plan a sequence of events to guide a trajectory toward bifurcation.

    Each step re-evaluates proximity so that event strength decreases
    smoothly as the personality moves closer to the critical point.

    Set ``app_calibrated=True`` to use the application-layer proximity
    (v1-v2 projection, ε_c_app=0.11) — required for realistic experiments.

    ``direction_mode`` selects the A/B arm: "aligned" (experiment, along v1)
    or "random" (control, random unit direction of equal magnitude). A per-step
    fresh random direction is drawn for the control arm; pass ``seed`` for a
    reproducible sequence.

    Returns
    -------
    dict with:
      events   – list of event dicts (one per step)
      total_displacement – summed displacement over all steps
      predicted_final_proximity – estimated proximity after sequence
    """
    pv = np.asarray(personality_vector, dtype=float).copy()
    events = []
    cumulative = np.zeros(9)
    _mode = "projection" if app_calibrated else "euclidean"
    rng = np.random.RandomState(seed) if direction_mode == "random" else None

    for step in range(n_steps):
        event = design_bifurcation_event(
            pv, target=target, app_calibrated=app_calibrated,
            direction_mode=direction_mode, rng=rng,
        )
        events.append({**event, "step": step})
        disp = np.array(event["displacement"])
        pv = pv + disp
        cumulative += disp

    final_bifurc = compute_bifurcation_distance(pv, mode=_mode)

    return {
        "events": events,
        "total_displacement": cumulative.tolist(),
        "predicted_final_proximity": final_bifurc["bifurcation_proximity"],
        "n_steps": n_steps,
        "target": target,
        "direction_mode": direction_mode,
    }


def intensity_modulation(proximity: float, scale: float | None = None) -> float:
    """
    Compute event magnitude from bifurcation proximity.

    Strategy:
    - Far from bifurcation (proximity → 0): larger nudge to close the gap
    - Near bifurcation (proximity → 1): tiny nudge is sufficient
    - Minimum magnitude guarantees a non-zero push even at proximity = 1

    ``scale`` sets the critical-distance reference: EPSILON_C (dynamics, default)
    or EPSILON_C_APP (application). The max nudge is half this scale.
    """
    proximity = float(np.clip(proximity, 0.0, 1.0))
    eps = EPSILON_C if scale is None else scale
    min_magnitude = 0.0005   # always apply at least a micro-nudge
    max_magnitude = eps * 0.5  # cap at half the critical distance

    # Linear interpolation: large when far, small when near
    magnitude = max_magnitude * (1.0 - proximity) + min_magnitude
    return float(magnitude)


# ── Edge-case handling ────────────────────────────────────────────────────────

# Minimum proximity below which a bifurcation event has negligible effect.
# Verified: n_steps=1, intensity_scale=0.5 achieves only 25% success from
# proximity=0. Below this threshold, the plain dynamics dominate.
NON_CRITICAL_THRESHOLD: float = 0.3


def is_in_effective_zone(
    personality_vector: list[float] | np.ndarray,
    app_calibrated: bool = False,
) -> dict:
    """
    Check whether the current personality is in the effective event zone.

    Returns:
      in_zone        – True if bifurcation_proximity >= NON_CRITICAL_THRESHOLD
      proximity      – current bifurcation proximity
      recommendation – suggested action
    """
    pv = np.asarray(personality_vector, dtype=float)
    _mode = "projection" if app_calibrated else "euclidean"
    bifurc = compute_bifurcation_distance(pv, mode=_mode)
    prox = bifurc["bifurcation_proximity"]
    in_zone = prox >= NON_CRITICAL_THRESHOLD

    if in_zone:
        recommendation = "proceed_with_event"
    else:
        recommendation = "pre_condition_first"  # need n_steps≥3 to reach zone

    return {
        "in_zone": in_zone,
        "proximity": prox,
        "threshold": NON_CRITICAL_THRESHOLD,
        "recommendation": recommendation,
    }


def rollback_distance(
    personality_vector: list[float] | np.ndarray,
    app_calibrated: bool = False,
) -> dict:
    """
    Estimate how many free-dynamics rounds are needed to return proximity < 0.1.

    Uses the exponential decay model (decay_rate=0.95, calibrated to P7-H).
    Provides a safety check: if the personality is near-critical and the game
    wants to cancel an event sequence, this tells the designer how long recovery
    will take naturally.
    """
    _DECAY = 0.95
    _mode = "projection" if app_calibrated else "euclidean"
    pv = np.asarray(personality_vector, dtype=float).copy()
    rounds = 0
    while True:
        prox = compute_bifurcation_distance(pv, mode=_mode)["bifurcation_proximity"]
        if prox < 0.1:
            break
        pv = _DECAY * pv + (1.0 - _DECAY) * BASELINE_ATTRACTOR
        rounds += 1
        if rounds >= 500:
            break  # safety cap

    return {
        "rounds_to_recover": rounds,
        "starting_proximity": compute_bifurcation_distance(
            np.asarray(personality_vector, dtype=float), mode=_mode
        )["bifurcation_proximity"],
        "recovery_threshold": 0.1,
    }
