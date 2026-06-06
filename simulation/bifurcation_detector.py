"""
Bifurcation detector for personality dynamics.

Based on P7-H findings:
- Marginal stability everywhere (λ_max ≈ 1.29e-10)
- Full 9D attractor (Kaplan-Yorke dim = 9.0)
- Critical perturbation distance ε_c ≈ 0.007 (from P7-G)
- Principal sensitive directions: v1, v2 (from P7-F SVD)
"""

from __future__ import annotations

import numpy as np

# ── Constants from P7 series ────────────────────────────────────────────────

FEATURE_NAMES = [
    "impulsiveness", "assertiveness", "optimism", "risk_aversion",
    "suspicion", "endurance", "randomness", "stability_seeking", "curiosity",
]

# Baseline attractor at α=0.2, mean over 3 seeds (P7-F)
BASELINE_ATTRACTOR = np.array([
    0.7720228727912913,   # impulsiveness
    0.7241232093429576,   # assertiveness
    0.6112562210994299,   # optimism
   -0.9225868280982982,   # risk_aversion
   -0.4316208098909589,   # suspicion
    0.2889985351128046,   # endurance
    0.6970402693133876,   # randomness
   -0.4706744249916079,   # stability_seeking
    0.8112134130435522,   # curiosity
])

# SVD principal directions (P7-F, σ₁=9.195 covers 99.93% variance)
V1 = np.array([
    -0.38472310395015424, -0.36085415090501716, -0.30908290079298933,
     0.45975085405928484,  0.22261191374202280, -0.13952439532859778,
    -0.35181484861689950,  0.23455770751654462, -0.40872386864028465,
])

V2 = np.array([
    -0.21240226471286897, -0.19944779534572710,  0.22205220807677992,
     0.25312230336041250, -0.70616726639739160, -0.46552620815102236,
     0.20246049452147202,  0.13090254311112020,  0.16797376302519700,
])

# Critical perturbation distance — DYNAMICS layer (P7-G).
# This is the micro-perturbation that triggers bifurcation near the attractor.
EPSILON_C: float = 0.007

# Critical distance — APPLICATION layer (P7-H Phase B1 calibration).
# The dynamics-layer ε_c (0.007) is ~7× smaller than the typical spread of real
# personalities (the 21 P7-F attractors span up to 0.147 from baseline), so using
# it as the proximity scale saturates proximity at 1.0 instantly for any real
# player. For the application/experiment we use the SECONDARY-ATTRACTOR distance
# (≈0.11, from P7-F multi-attractor structure / design doc) as the critical scale,
# measured in the v1-v2 sensitive subspace. This makes proximity vary smoothly
# across the realistic operating range. See dev log "問題 2".
EPSILON_C_APP: float = 0.11

# Orthonormal basis of the sensitive subspace (v1, v2). Verified: ‖v1‖=‖v2‖=1,
# v1·v2≈1.7e-15. Projecting Δ onto this 2D plane isolates the directions that
# actually drive bifurcation, instead of diluting the signal across all 9D.
SENSITIVE_BASIS = np.vstack([V1, V2])  # shape (2, 9)

# Lyapunov max exponent (P7-H Phase 3)
LAMBDA_MAX: float = 1.29e-10

# Proximity threshold above which the system is considered critical
CRITICAL_PROXIMITY: float = 0.8


# ── Core detection functions ─────────────────────────────────────────────────

def compute_bifurcation_distance(
    personality_vector: np.ndarray,
    mode: str = "euclidean",
    epsilon: float | None = None,
) -> dict:
    """
    Compute bifurcation proximity for a personality vector.

    Parameters
    ----------
    personality_vector : 9D personality coordinates
    mode :
        "euclidean"  – DYNAMICS layer (default). Full 9D distance to baseline,
                       scaled by ε_c=0.007. Backward-compatible behaviour.
        "projection" – APPLICATION layer. Distance measured in the v1-v2
                       sensitive subspace, scaled by ε_c_app=0.11. Use this for
                       the experiment so proximity does not saturate instantly.
    epsilon :
        Override the scale. Defaults to EPSILON_C (euclidean) or
        EPSILON_C_APP (projection).

    Returns a dict with:
      distance            – distance to baseline (full 9D or v1-v2 projection)
      bifurcation_proximity – normalised [0,1]; 1 = at/past critical point
      is_critical         – True when proximity > CRITICAL_PROXIMITY
      v1_projection       – projection onto primary sensitive axis v1
      v2_projection       – projection onto secondary axis v2
      mode                – the proximity mode used
    """
    pv = np.asarray(personality_vector, dtype=float)
    delta = pv - BASELINE_ATTRACTOR
    v1_proj = float(np.dot(delta, V1))
    v2_proj = float(np.dot(delta, V2))

    if mode == "projection":
        # Distance within the 2D sensitive plane (v1-v2)
        distance = float(np.hypot(v1_proj, v2_proj))
        eps = EPSILON_C_APP if epsilon is None else epsilon
    else:  # "euclidean" (dynamics layer, default)
        distance = float(np.linalg.norm(delta))
        eps = EPSILON_C if epsilon is None else epsilon

    proximity = min(1.0, distance / eps)

    return {
        "distance": distance,
        "bifurcation_proximity": proximity,
        "is_critical": proximity > CRITICAL_PROXIMITY,
        "v1_projection": v1_proj,
        "v2_projection": v2_proj,
        "mode": mode,
    }


def compute_sensitive_direction(
    personality_vector: np.ndarray,
) -> dict:
    """
    Return the most sensitive perturbation direction at the given point.

    Because λ ≈ 0⁺ everywhere (P7-H Phase 3), the marginal stability
    signature is uniform.  The primary sensitive direction is v1 (99.93%
    of variance), with v2 as the secondary direction.

    Returns a dict with:
      direction           – unit vector (9D) for the primary sensitive axis
      secondary_direction – unit vector for the secondary sensitive axis
      eigenvalue          – characteristic λ_max (marginal stability)
      sensitivity_strength – |eigenvalue|
    """
    pv = np.asarray(personality_vector, dtype=float)
    delta = pv - BASELINE_ATTRACTOR

    # Choose v1 as primary direction; flip sign to point away from baseline
    # so the returned vector always points toward potential bifurcation.
    primary = V1.copy()
    if np.dot(delta, primary) < 0:
        primary = -primary

    return {
        "direction": primary.tolist(),
        "secondary_direction": V2.tolist(),
        "eigenvalue": LAMBDA_MAX,
        "sensitivity_strength": abs(LAMBDA_MAX),
    }


def get_jacobian(alpha: float = 0.2) -> np.ndarray:
    """
    Return a linearised Jacobian at the baseline attractor.

    The Jacobian is approximated via the outer product of the two principal
    singular vectors scaled by their singular values (rank-2 reconstruction
    from P7-F SVD).  This captures 99.99% of the system's linear structure.
    """
    # Singular values from P7-F
    sigma1 = 9.195296792430849
    sigma2 = 0.23484  # σ₂ ≈ 0.235

    J = sigma1 * np.outer(V1, V1) + sigma2 * np.outer(V2, V2)
    return J
