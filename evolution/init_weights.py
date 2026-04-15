"""Player weight initialisation helpers for T-series experiments.

Provides random Dirichlet initialisation as an alternative to the
deterministic init_bias used in C-family.

Architecture invariants
-----------------------
- No I/O (no file reading/writing).
- No dependency on plotting, simulation, or analysis layers.
- All public functions are pure.
"""

from __future__ import annotations

import random
from math import log


STRATEGY_SPACE = ["aggressive", "defensive", "balanced"]


def random_simplex_weights(
    strategy_space: list[str],
    *,
    seed: int,
    player_index: int,
    alpha: float = 1.0,
) -> dict[str, float]:
    """Generate a random simplex weight vector from Dirichlet(alpha, ..., alpha).

    Uses the gamma-variate method: sample independent Gamma(alpha, 1) variates
    and normalise.  For alpha=1.0 this gives a uniform distribution on the simplex.

    Parameters
    ----------
    strategy_space :
        List of strategy names (order matters).
    seed :
        Base random seed.
    player_index :
        Player index (combined with seed for per-player determinism).
    alpha :
        Dirichlet concentration parameter (1.0 = uniform on simplex).

    Returns
    -------
    dict[str, float]
        {strategy_name: weight} with all weights > 0 and sum = 1.
    """
    rng = random.Random(int(seed) * 10000 + int(player_index))
    k = len(strategy_space)
    # Gamma(alpha, 1) via inverse-CDF for alpha=1 (exponential), else via rejection
    if alpha == 1.0:
        # Exponential distribution: -log(U)
        raw = [-log(max(1e-300, rng.random())) for _ in range(k)]
    else:
        raw = [rng.gammavariate(float(alpha), 1.0) for _ in range(k)]
    total = sum(raw)
    normed = [v / total for v in raw]
    return dict(zip(strategy_space, normed))


def init_weight_dispersion(
    all_weights_simplex: list[list[float]],
) -> float:
    """Compute the mean L2 distance of each player's weights from the global mean.

    Returns 0.0 if all players have identical weights (uniform init).
    Positive value indicates heterogeneity (random init).
    """
    n = len(all_weights_simplex)
    if n == 0:
        return 0.0
    nstrats = len(all_weights_simplex[0])
    gmean = [0.0] * nstrats
    for wi in all_weights_simplex:
        for k in range(nstrats):
            gmean[k] += wi[k]
    gmean = [v / n for v in gmean]
    total = 0.0
    for wi in all_weights_simplex:
        total += sum((wi[k] - gmean[k]) ** 2 for k in range(nstrats)) ** 0.5
    return float(total / n)
