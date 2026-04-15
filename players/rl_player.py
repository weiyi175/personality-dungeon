"""Per-player RL state for the Personality Dungeon runtime.

Bridge spec §3.2: holds per-player RL state only.
Architecture: players/ layer — no I/O, no plotting, no simulation logic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Mapping


_PERSONALITY_KEYS = [
    "impulsiveness", "caution", "greed", "optimism", "suspicion",
    "persistence", "randomness", "stability_seeking", "ambition",
    "patience", "curiosity", "fearfulness",
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
    """Compute 4 latent signals from 12D personality vector.

    z_drive   = 0.4·impulsiveness + 0.3·greed + 0.3·ambition
    z_guard   = 0.4·caution + 0.3·fearfulness + 0.3·suspicion
    z_temporal= 0.4·patience + 0.3·persistence + 0.3·stability_seeking
    z_noise   = 0.4·randomness + 0.3·curiosity + 0.3·optimism
    """
    def _g(k: str) -> float:
        return float(p.get(k, 0.0))

    return {
        "z_drive": 0.4 * _g("impulsiveness") + 0.3 * _g("greed") + 0.3 * _g("ambition"),
        "z_guard": 0.4 * _g("caution") + 0.3 * _g("fearfulness") + 0.3 * _g("suspicion"),
        "z_temporal": 0.4 * _g("patience") + 0.3 * _g("persistence") + 0.3 * _g("stability_seeking"),
        "z_noise": 0.4 * _g("randomness") + 0.3 * _g("curiosity") + 0.3 * _g("optimism"),
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

    Mapping (blueprint §4.2):
      alpha_i = alpha_base × (1 + λ_α · clamp(z_α))
        where z_α = z_drive − 0.5·z_temporal − 0.5·z_guard
      beta_i  = beta_base × (1 + λ_β · clamp(z_β))
        where z_β = z_guard + z_temporal − z_noise
      r_s_i   = r_s_baseline + λ_r · clamp(z_s)
        where z_A=z_drive, z_D=z_guard, z_B=z_temporal
      risk_sensitivity_i = λ_risk · clamp((z_guard + fearfulness − optimism)/2)
    """
    sig = personality_latent_signals(personality)

    # alpha_i: fast/slow learner heterogeneity
    z_alpha = sig["z_drive"] - 0.5 * sig["z_temporal"] - 0.5 * sig["z_guard"]
    alpha_mod = alpha_base * (1.0 + lambda_alpha * _clamp(z_alpha, -1.0, 1.0))
    alpha_mod = _clamp(alpha_mod, 0.001, 1.0)

    # beta_i: exploit/explore tension (blueprint §4.2: z_guard + z_temporal - z_noise)
    z_beta = sig["z_guard"] + sig["z_temporal"] - sig["z_noise"]
    beta_mod = beta_base * (1.0 + lambda_beta * _clamp(z_beta, -1.0, 1.0))
    # Adaptive β compensation for extreme low-z_β players (§4.2.H)
    if lambda_beta_comp > 0.0 and z_beta < 0.0:
        beta_mod += lambda_beta_comp * beta_base * (-_clamp(z_beta, -1.0, 0.0))
    beta_mod = _clamp(beta_mod, 0.1, 100.0)

    # r_A, r_D, r_B: per-player strategy multiplier offsets (blueprint §4.3)
    sam = list(strategy_alpha_multipliers)
    sam[0] += lambda_r * _clamp(sig["z_drive"], -1.0, 1.0)
    sam[1] += lambda_r * _clamp(sig["z_guard"], -1.0, 1.0)
    sam[2] += lambda_r * _clamp(sig["z_temporal"], -1.0, 1.0)
    sam = [_clamp(s, 0.3, 2.5) for s in sam]

    # risk_sensitivity_i (blueprint §4.2)
    def _g(k: str) -> float:
        return float(personality.get(k, 0.0))
    z_risk = sig["z_guard"] + _g("fearfulness") - _g("optimism")
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
