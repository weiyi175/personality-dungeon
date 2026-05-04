"""Per-player RL state for the Personality Dungeon runtime.

Bridge spec §3.2: holds per-player RL state only.
Architecture: players/ layer — no I/O, no plotting, no simulation logic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Mapping


_PERSONALITY_KEYS = [
    # --- 擴張組 The Drivers ---
    "impulsiveness", "assertiveness", "optimism",
    # --- 防禦組 The Stabilizers ---
    "risk_aversion", "suspicion", "endurance",
    # --- 擾動組 The Explorers ---
    "randomness", "stability_seeking", "curiosity",
]

_NSTRATS = 3


@dataclass
class RLPlayer:
    """Minimal per-player RL state (bridge spec §3.2)."""

    player_id: int
    personality: dict[str, float]
    q_values: list[float]
    alpha: float
    beta: float
    strategy_alpha_multipliers: list[float]
    payoff_bias: list[float]
    cumulative_utility: float = 0.0
    cumulative_risk: float = 0.0
    stress: float = 0.0
    risk_sensitivity: float = 0.0


# -------------------------------------------------------------------
# Personality → RL parameter mapping (blueprint §4.1)
# -------------------------------------------------------------------

def personality_latent_signals(p: Mapping[str, float]) -> dict[str, float]:
    """Compute 3 latent signals from 9D personality vector (9-trait Enneagram design).

    z_expanding  = (impulsiveness + assertiveness + optimism)   / 3
    z_contracting= (risk_aversion + suspicion    + endurance)   / 3
    z_exploring  = (randomness   + stability_seeking + curiosity) / 3

    Each signal is normalised to [-1, 1] by dividing by 3.
    """
    def _g(k: str) -> float:
        return float(p.get(k, 0.0))

    return {
        "z_expanding":   (_g("impulsiveness") + _g("assertiveness") + _g("optimism"))        / 3.0,
        "z_contracting": (_g("risk_aversion") + _g("suspicion")    + _g("endurance"))        / 3.0,
        "z_exploring":   (_g("randomness")    + _g("stability_seeking") + _g("curiosity"))   / 3.0,
    }


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def init_rl_player(
    player_id: int,
    personality: Mapping[str, float],
    *,
    alpha_base: float,
    beta_base: float,
    strategy_alpha_multipliers: list[float],
    payoff_bias: list[float],
    lambda_alpha: float = 0.0,
    lambda_beta: float = 0.0,
    lambda_r: float = 0.0,
    lambda_risk: float = 0.0,
    lambda_beta_comp: float = 0.0,
    init_q: float = 0.0,
) -> RLPlayer:
    """Create an RLPlayer with personality-modulated RL parameters.

    With all lambda values at 0 the result degenerates to pure BL2
    (no personality influence).

    Mapping (9-trait Enneagram design):
      alpha_i = alpha_base × (1 + λ_α · clamp(z_α))
        where z_α = z_expanding − 0.5·z_exploring − 0.5·z_contracting
      beta_i  = beta_base × (1 + λ_β · clamp(z_β))
        where z_β = z_contracting + z_exploring − z_expanding
      r_s_i   = r_s_baseline + λ_r · clamp(z_s)
        where z_A=z_expanding, z_D=z_contracting, z_B=z_exploring
      risk_sensitivity_i = λ_risk · clamp((z_contracting + risk_aversion − optimism)/2)
    """
    sig = personality_latent_signals(personality)

    # alpha_i: fast/slow learner heterogeneity
    z_alpha = sig["z_expanding"] - 0.5 * sig["z_exploring"] - 0.5 * sig["z_contracting"]
    alpha_mod = alpha_base * (1.0 + lambda_alpha * _clamp(z_alpha, -1.0, 1.0))
    alpha_mod = _clamp(alpha_mod, 0.001, 1.0)

    # beta_i: exploit/explore tension
    z_beta = sig["z_contracting"] + sig["z_exploring"] - sig["z_expanding"]
    beta_mod = beta_base * (1.0 + lambda_beta * _clamp(z_beta, -1.0, 1.0))
    # Adaptive β compensation for extreme low-z_β players (§4.2.H)
    if lambda_beta_comp > 0.0 and z_beta < 0.0:
        beta_mod += lambda_beta_comp * beta_base * (-_clamp(z_beta, -1.0, 0.0))
    beta_mod = _clamp(beta_mod, 0.1, 100.0)

    # r_A, r_D, r_B: per-player strategy multiplier offsets
    # aggressive→z_expanding, defensive→z_contracting, balanced→z_exploring
    sam = list(strategy_alpha_multipliers)
    sam[0] += lambda_r * _clamp(sig["z_expanding"],   -1.0, 1.0)
    sam[1] += lambda_r * _clamp(sig["z_contracting"], -1.0, 1.0)
    sam[2] += lambda_r * _clamp(sig["z_exploring"],   -1.0, 1.0)
    sam = [_clamp(s, 0.3, 2.5) for s in sam]

    # risk_sensitivity_i
    def _g(k: str) -> float:
        return float(personality.get(k, 0.0))
    z_risk = sig["z_contracting"] + _g("risk_aversion") - _g("optimism")
    risk_sens = lambda_risk * _clamp(z_risk / 2.0, -1.0, 1.0)

    p_dict = {k: float(personality.get(k, 0.0)) for k in _PERSONALITY_KEYS}

    return RLPlayer(
        player_id=player_id,
        personality=p_dict,
        q_values=[init_q] * _NSTRATS,
        alpha=alpha_mod,
        beta=beta_mod,
        strategy_alpha_multipliers=sam,
        payoff_bias=list(payoff_bias),
        risk_sensitivity=risk_sens,
    )


def sample_personality(*, rng: random.Random) -> dict[str, float]:
    """Sample a random personality vector with values in [-1, 1]."""
    return {k: rng.uniform(-1.0, 1.0) for k in _PERSONALITY_KEYS}
