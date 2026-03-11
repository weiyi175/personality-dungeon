from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SimplexNormalizeResult:
    x: tuple[float, float, float]
    clipped_negative: bool


def normalize_simplex3(
    v: Sequence[float] | Iterable[float],
    *,
    clip_negative: bool = True,
    on_invalid: str = "zeros",
) -> SimplexNormalizeResult:
    """Normalize a 3-vector onto the probability simplex.

    Numerical drift guard:
    - Rejects non-finite inputs (NaN/Inf)
    - Optionally clips negatives to 0 then renormalizes
    - If the sum is non-positive after clipping, returns either zeros or uniform

    Args:
        v: 3 values.
        clip_negative: If True, negative values are clipped to 0 before normalizing.
        on_invalid: Behavior when normalization is impossible. One of:
            - "zeros": return (0,0,0)
            - "uniform": return (1/3,1/3,1/3)

    Returns:
        SimplexNormalizeResult with normalized x and whether clipping happened.
    """

    values = tuple(float(x) for x in v)
    if len(values) != 3:
        raise ValueError(f"Expected 3 values, got {len(values)}")

    if on_invalid not in {"zeros", "uniform"}:
        raise ValueError(f"on_invalid must be 'zeros' or 'uniform', got {on_invalid!r}")

    if not all(isfinite(x) for x in values):
        return SimplexNormalizeResult(_invalid_simplex(on_invalid), clipped_negative=False)

    clipped = False
    if clip_negative:
        clipped_values = []
        for x in values:
            if x < 0.0:
                clipped = True
                clipped_values.append(0.0)
            else:
                clipped_values.append(x)
        values = tuple(clipped_values)  # type: ignore[assignment]

    s = values[0] + values[1] + values[2]
    if s <= 0.0:
        return SimplexNormalizeResult(_invalid_simplex(on_invalid), clipped_negative=clipped)

    x1, x2, x3 = (values[0] / s, values[1] / s, values[2] / s)

    if not all(isfinite(x) for x in (x1, x2, x3)):
        return SimplexNormalizeResult(_invalid_simplex(on_invalid), clipped_negative=clipped)

    # Guard against tiny negative drift after arithmetic.
    if clip_negative:
        y1, y2, y3 = (max(0.0, x1), max(0.0, x2), max(0.0, x3))
        ys = y1 + y2 + y3
        if ys > 0.0 and (y1, y2, y3) != (x1, x2, x3):
            clipped = True
            x1, x2, x3 = (y1 / ys, y2 / ys, y3 / ys)

    return SimplexNormalizeResult((x1, x2, x3), clipped_negative=clipped)


def _invalid_simplex(on_invalid: str) -> tuple[float, float, float]:
    if on_invalid == "uniform":
        u = 1.0 / 3.0
        return (u, u, u)
    return (0.0, 0.0, 0.0)
