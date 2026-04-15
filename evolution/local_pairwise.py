"""C1 local pairwise imitation helper.

Implements the per-player update semantics defined in blueprint §5.2.

Update rule (synchronous, per player i each round):
    1.  Sample model neighbour j ~ Uniform(N(i))
    2.  Compute payoff π_i, π_j:
            π_i = (1/|N(i)|) Σ_{k∈N(i)} w_i · A·s_k
        where s_k is the one-hot strategy vector of neighbour k this round.
    3.  Adoption probability:
            q_ij = 1 / (1 + exp(-β·(π_j - π_i)))
    4.  Convex imitation update:
            w_i' = normalise((1 - μ·q_ij)·w_i + μ·q_ij·w_j)

Architecture invariants
-----------------------
- No I/O (no file reading/writing).
- No dependency on plotting, simulation, or analysis layers.
- All public functions are pure.
"""

from __future__ import annotations

from math import exp, isfinite


STRATEGY_SPACE = ["aggressive", "defensive", "balanced"]
_NSTRATS = 3

# ---------------------------------------------------------------------------
# Payoff helpers
# ---------------------------------------------------------------------------

def local_payoff(
    weights_i: list[float],
    neighbor_strategies: list[str],
    *,
    a: float,
    b: float,
    cross: float,
) -> float:
    """Compute player i's local payoff against its neighbours.

    π_i = (1/|N(i)|) Σ_{k∈N(i)}  w_i · A · s_k

    where s_k is the one-hot vector of neighbour k's sampled strategy.

    Parameters
    ----------
    weights_i:
        [w_agg, w_def, w_bal] for player i (simplex, sum≈1).
    neighbor_strategies:
        Sampled strategy string for each neighbour.
    a, b, cross:
        Payoff matrix parameters.  A = [[0,a,-b],[-b,0,a],[a,-b,0]] +
        cross-coupling term.

    Returns
    -------
    float
        Mean payoff of player i against the local neighbourhood.
    """
    if not neighbor_strategies:
        return 0.0
    n = len(neighbor_strategies)
    # One-hot encoding map
    _oh = {"aggressive": (1, 0, 0), "defensive": (0, 1, 0), "balanced": (0, 0, 1)}
    total = 0.0
    w0, w1, w2 = float(weights_i[0]), float(weights_i[1]), float(weights_i[2])
    a, b, cross = float(a), float(b), float(cross)
    for strat in neighbor_strategies:
        s = _oh.get(strat, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
        s0, s1, s2 = float(s[0]), float(s[1]), float(s[2])
        # u_i = A @ s_k  (3-vector)
        u0 = a * s1 - b * s2
        u1 = -b * s0 + a * s2
        u2 = a * s0 - b * s1
        # cross term
        u0 += -cross * s1
        u1 += -cross * s0
        u2 += cross * (s0 + s1)
        # payoff = w_i · u
        total += w0 * u0 + w1 * u1 + w2 * u2
    return float(total / n)


# ---------------------------------------------------------------------------
# Adoption probability
# ---------------------------------------------------------------------------

def pairwise_adoption_probability(
    payoff_i: float,
    payoff_j: float,
    *,
    beta_pair: float,
) -> float:
    """Fermi / logistic adoption probability.

        q_ij = 1 / (1 + exp(-β·(π_j - π_i)))

    Returns a value in (0, 1).  Numerically stable for large |π_j - π_i|.

    Parameters
    ----------
    payoff_i:
        Focal player i payoff.
    payoff_j:
        Model player j payoff.
    beta_pair:
        Comparison sharpness (must be > 0).
    """
    beta = float(beta_pair)
    diff = float(payoff_j) - float(payoff_i)
    # Clip to avoid overflow in exp().
    # exp(|x|) for |x| > 700 is ±inf in IEEE 754, so clip at ±500.
    clipped = max(-500.0, min(500.0, beta * diff))
    return float(1.0 / (1.0 + exp(-clipped)))


# ---------------------------------------------------------------------------
# Imitation update
# ---------------------------------------------------------------------------

def pairwise_imitation_update(
    weights_i: list[float],
    weights_j: list[float],
    *,
    mu: float,
    q: float,
) -> list[float]:
    """Convex imitation step towards model player j.

        w_i' = normalise((1 - μ·q)·w_i + μ·q·w_j)

    Parameters
    ----------
    weights_i, weights_j:
        Raw (not necessarily unit-sum) weight vectors of length 3.
    mu:
        Imitation strength ∈ [0, 1].
    q:
        Adoption probability ∈ (0, 1) from pairwise_adoption_probability().

    Returns
    -------
    list[float]
        Normalised weight vector (sum = 1, all entries > 0).

    Raises
    ------
    ValueError
        If any resulting weight is non-positive before normalisation (only
        possible if inputs were already degenerate).
    """
    mu = float(mu)
    q = float(q)
    alpha = mu * q  # imitation coefficient ∈ [0, 1]

    new_w = [
        (1.0 - alpha) * float(weights_i[k]) + alpha * float(weights_j[k])
        for k in range(_NSTRATS)
    ]
    return _normalise(new_w)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalise(w: list[float]) -> list[float]:
    """Normalise w so that sum = 1, all entries > 0.

    Uses simplex (proportion) normalisation: w_k / sum(w).
    If any raw entry is ≤ 0 this is a precondition violation and a
    ValueError is raised (prevents silent NaN propagation).
    """
    total = sum(float(v) for v in w)
    if not isfinite(total) or total <= 0.0:
        raise ValueError(
            f"Cannot normalise weight vector with non-positive sum={total}: {w}"
        )
    result = [float(v) / total for v in w]
    # Guard: all must be strictly positive
    for k, v in enumerate(result):
        if not isfinite(v) or v <= 0.0:
            raise ValueError(
                f"Normalised weight[{k}]={v} is not positive. "
                f"Input weights may be degenerate: {w}"
            )
    return result


# ---------------------------------------------------------------------------
# Weights ↔ list conversion helpers (used by harness)
# ---------------------------------------------------------------------------

def weights_to_list(w: dict[str, float]) -> list[float]:
    """Convert {strategy: weight} dict to ordered [w_agg, w_def, w_bal]."""
    return [float(w.get(s, 1.0)) for s in STRATEGY_SPACE]


def weights_to_simplex(w: dict[str, float]) -> list[float]:
    """Convert {strategy: weight} dict to normalised simplex list."""
    raw = weights_to_list(w)
    total = sum(raw)
    if total <= 0.0:
        return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    return [v / total for v in raw]


def list_to_weights(lst: list[float]) -> dict[str, float]:
    """Convert ordered [w_agg, w_def, w_bal] back to {strategy: weight} dict."""
    return {s: float(lst[k]) for k, s in enumerate(STRATEGY_SPACE)}
