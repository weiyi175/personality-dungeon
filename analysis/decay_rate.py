"""Estimate amplitude envelope decay/growth rate (gamma) from simulation time series.

Motivation
- Stage3 needs sustained rotation with non-vanishing amplitude.
- Local linear indicators (ODE alpha, lagged rho) are useful, but the most direct
  engine-faithful indicator is the *empirical envelope slope*.

We estimate gamma by:
1) Convert a 3-strategy series into a 2D trajectory (u,v) using two components.
2) Compute radial deviation r_t = ||(u_t,v_t) - c|| from the center c.
3) Extract local maxima (peaks) of r_t.
4) Fit log(r_peak) ~ intercept + gamma * t_peak via least squares.

gamma < 0: decaying envelope (stable focus / damped spiral)
gamma ~ 0: near-neutral envelope (persistent rotation possible)
gamma > 0: growing envelope (unstable until nonlinear saturation)

No third-party dependencies.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DecayFit:
    gamma: float
    intercept: float
    r2: float
    n_peaks: int
    first_t: int
    last_t: int


def _amplitude_uv(
    series_map: dict[str, list[float]],
    *,
    u_key: str,
    v_key: str,
    center_u: float,
    center_v: float,
    metric: str,
) -> list[float]:
    u = series_map.get(u_key)
    v = series_map.get(v_key)
    if u is None or v is None:
        raise ValueError(f"series_map must contain keys: {u_key!r}, {v_key!r}")
    if len(u) != len(v):
        raise ValueError("u and v series must have same length")
    cu = float(center_u)
    cv = float(center_v)
    m = str(metric)
    if m not in ("linf", "radius"):
        raise ValueError("metric must be 'linf' or 'radius'")

    out: list[float] = []
    for uu, vv in zip(u, v):
        du = float(uu) - cu
        dv = float(vv) - cv
        if m == "radius":
            out.append(math.hypot(du, dv))
        else:
            out.append(max(abs(du), abs(dv)))
    return out


def _find_peaks(values: list[float], *, min_value: float = 1e-12, min_distance: int = 2) -> list[tuple[int, float]]:
    """Return list of (index, value) for simple local maxima.

    Peak definition: v[i-1] < v[i] >= v[i+1] and v[i] >= min_value.
    A simple min_distance suppression is applied (keep higher peak).
    """

    if len(values) < 3:
        return []

    mv = float(min_value)
    md = int(min_distance)
    if md < 1:
        md = 1

    candidates: list[tuple[int, float]] = []
    for i in range(1, len(values) - 1):
        vi = float(values[i])
        if vi < mv:
            continue
        if float(values[i - 1]) < vi and vi >= float(values[i + 1]):
            candidates.append((i, vi))

    if not candidates:
        return []

    # Suppress peaks closer than min_distance by keeping the largest.
    candidates.sort(key=lambda p: p[0])
    kept: list[tuple[int, float]] = []
    for idx, val in candidates:
        if not kept:
            kept.append((idx, val))
            continue
        last_idx, last_val = kept[-1]
        if idx - last_idx >= md:
            kept.append((idx, val))
            continue
        # too close: keep the larger one
        if val > last_val:
            kept[-1] = (idx, val)
    return kept


def _ols_fit(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Return (slope, intercept, r2) for y ~ intercept + slope*x."""

    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 points")

    mean_x = sum(x) / float(n)
    mean_y = sum(y) / float(n)

    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    for xi, yi in zip(x, y):
        dx = float(xi) - mean_x
        dy = float(yi) - mean_y
        sxx += dx * dx
        sxy += dx * dy
        syy += dy * dy

    if sxx == 0.0:
        slope = 0.0
    else:
        slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    # r2
    if syy == 0.0:
        r2 = 1.0
    else:
        # SSE = sum (y - yhat)^2
        sse = 0.0
        for xi, yi in zip(x, y):
            yhat = intercept + slope * float(xi)
            err = float(yi) - yhat
            sse += err * err
        r2 = 1.0 - (sse / syy)

    return (float(slope), float(intercept), float(r2))


def estimate_decay_gamma(
    series_map: dict[str, list[float]],
    *,
    series_kind: str,
    u_key: str = "aggressive",
    v_key: str = "defensive",
    metric: str = "linf",
    min_peaks: int = 6,
    min_peak_value: float = 1e-12,
    min_peak_distance: int = 2,
) -> DecayFit | None:
    """Estimate empirical envelope slope gamma.

    Parameters
    - series_kind: "p" uses center (1/3,1/3); "w" uses center (1,1).

    Returns None when peaks are insufficient or unusable.
    """

    kind = str(series_kind)
    if kind not in ("p", "w"):
        raise ValueError("series_kind must be 'p' or 'w'")

    if kind == "p":
        cu = 1.0 / 3.0
        cv = 1.0 / 3.0
    else:
        cu = 1.0
        cv = 1.0

    amp = _amplitude_uv(
        series_map,
        u_key=u_key,
        v_key=v_key,
        center_u=cu,
        center_v=cv,
        metric=str(metric),
    )
    peaks = _find_peaks(amp, min_value=float(min_peak_value), min_distance=int(min_peak_distance))
    if len(peaks) < int(min_peaks):
        return None

    # log of peak amplitude
    xs: list[float] = []
    ys: list[float] = []
    for t, amp in peaks:
        a = float(amp)
        if a <= 0.0:
            continue
        xs.append(float(t))
        ys.append(float(math.log(a)))

    if len(xs) < int(min_peaks):
        return None

    slope, intercept, r2 = _ols_fit(xs, ys)
    return DecayFit(
        gamma=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        n_peaks=int(len(xs)),
        first_t=int(xs[0]),
        last_t=int(xs[-1]),
    )


def _read_timeseries_csv(path: Path, *, series: str) -> dict[str, list[float]]:
    """Load series_map from a simulation timeseries CSV."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    prefix = "p_" if str(series) == "p" else "w_"
    out: dict[str, list[float]] = {}
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        series_fields = [fn for fn in reader.fieldnames if fn.startswith(prefix)]
        if not series_fields:
            raise ValueError(f"CSV has no fields starting with {prefix!r}")

        for row in reader:
            for fn in series_fields:
                name = fn[len(prefix) :]
                out.setdefault(name, []).append(float(row[fn]))

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate empirical envelope decay rate gamma from a timeseries CSV")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--series", choices=["p", "w"], default="p")
    ap.add_argument("--metric", choices=["linf", "radius"], default="linf")
    ap.add_argument("--min-peaks", type=int, default=6)
    ap.add_argument("--min-peak-distance", type=int, default=2)
    ap.add_argument("--min-peak-value", type=float, default=1e-12)
    args = ap.parse_args()

    series_map = _read_timeseries_csv(Path(args.csv), series=str(args.series))
    fit = estimate_decay_gamma(
        series_map,
        series_kind=str(args.series),
        metric=str(args.metric),
        min_peaks=int(args.min_peaks),
        min_peak_distance=int(args.min_peak_distance),
        min_peak_value=float(args.min_peak_value),
    )

    if fit is None:
        raise SystemExit("insufficient peaks to estimate gamma")

    print("=== decay_rate (empirical envelope fit) ===")
    print(f"csv={args.csv}")
    print(f"series={args.series}")
    print(f"gamma={fit.gamma:+.6g}  r2={fit.r2:.4f}  n_peaks={fit.n_peaks}  t=[{fit.first_t},{fit.last_t}]")


if __name__ == "__main__":
    main()
