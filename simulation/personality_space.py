"""Space A ↔ Space B personality coordinate bijection (P7-H).

Background
----------
The P7-H apparatus confused two 9D personality parameterisations that share the
same trait names/order but live on different numeric scales:

  Space A — the RL-engine / SVD coordinates. ``player.personality`` values
            cluster around ``BASELINE_ATTRACTOR`` (e.g. impulsiveness≈0.77,
            risk_aversion≈-0.92). All bifurcation geometry (BASELINE_ATTRACTOR,
            v1, v2, ε_c) is derived here (P7-F/P7-G).

  Space B — a normalised "display / operating" coordinate. The frontend used an
            ad-hoc mapping from population action proportions (p_aggressive …);
            its scale did not match Space A, so events DESIGNED in Space A were
            APPLIED and MEASURED on a different scale → proximity saturated and
            event nudges failed to persist.

This module pins down ONE canonical, exactly-invertible transform between the
two spaces so the loop is coordinate-consistent:

    design event in A  →  a_to_b()  →  apply in B  →  b_to_a()  →  measure DV in A

Definition
----------
An affine recentre + isotropic rescale:

    b = (a − BASELINE_ATTRACTOR) / SPACE_B_SCALE
    a =  b · SPACE_B_SCALE + BASELINE_ATTRACTOR

so the attractor sits at the Space-B origin and one ``SPACE_B_SCALE`` of motion
in Space A equals one unit in Space B. ``SPACE_B_SCALE`` is the application-layer
critical distance ``EPSILON_C_APP`` (≈0.11): the realistic personality spread
(~0.11–0.15 from baseline) then maps to magnitude ~1 in Space B instead of
saturating against the dynamics-layer ε_c (0.007). The map is a diagonal affine
with a non-zero scale, hence a bijection; ``a_to_b`` and ``b_to_a`` are exact
inverses (see tests/test_personality_space.py round-trip checks).

NOTE: ``SPACE_B_SCALE`` / centre are the canonical backend definition. When the
Godot frontend is available, its display mapping must be reconciled to THIS
transform (or replaced by it) so both halves of the loop agree.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from simulation.bifurcation_detector import (
    BASELINE_ATTRACTOR,
    EPSILON_C_APP,
    FEATURE_NAMES,
)

# Canonical Space-B scale: one unit of Space B == EPSILON_C_APP of Space A.
SPACE_B_SCALE: float = EPSILON_C_APP

# Centre of the transform (Space A). Space B places this at the origin.
SPACE_B_CENTER: np.ndarray = BASELINE_ATTRACTOR

_DIM = len(FEATURE_NAMES)


def _as_vec(v: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    if arr.shape != (_DIM,):
        raise ValueError(f"personality vector must have {_DIM} elements, got {arr.shape}")
    return arr


def a_to_b(personality_a: Sequence[float] | np.ndarray) -> np.ndarray:
    """Map a Space-A personality vector to Space-B (normalised, baseline-centred)."""
    a = _as_vec(personality_a)
    return (a - SPACE_B_CENTER) / SPACE_B_SCALE


def b_to_a(personality_b: Sequence[float] | np.ndarray) -> np.ndarray:
    """Map a Space-B personality vector back to Space-A (inverse of a_to_b)."""
    b = _as_vec(personality_b)
    return b * SPACE_B_SCALE + SPACE_B_CENTER


def displacement_a_to_b(displacement_a: Sequence[float] | np.ndarray) -> np.ndarray:
    """Map a Space-A *displacement* (a difference vector) to Space-B.

    Displacements are linear, not affine: only the scale applies, not the
    recentre. Use this for event displacement vectors (which are deltas, not
    absolute positions).
    """
    return _as_vec(displacement_a) / SPACE_B_SCALE


def displacement_b_to_a(displacement_b: Sequence[float] | np.ndarray) -> np.ndarray:
    """Map a Space-B displacement back to Space-A (inverse of displacement_a_to_b)."""
    return _as_vec(displacement_b) * SPACE_B_SCALE
