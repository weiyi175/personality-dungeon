"""Quantify collapse-score sensitivity over (k_inf, beta) grids.

This is a lightweight companion to analysis.rho_curve_viz.
- No plots; outputs a CSV/JSON table for paper/report sensitivity claims.
- Stdlib-only; reads the same rho_curve sweep CSVs referenced by a compose script.

Collapse definition
- Rescale x = (k - k_inf) * N^beta.
- Bin x over the common x-range and compute the mean cross-N variance of
  P(Level3) within bins.
- Lower score => better collapse (heuristic).

Design constraints
- Lives in analysis/ and does not import simulation/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _try_float(v: object) -> Optional[float]:
	try:
		if v is None:
			return None
		s = str(v).strip()
		if s == "":
			return None
		return float(s)
	except Exception:
		return None


def _try_int(v: object) -> Optional[int]:
	try:
		if v is None:
			return None
		s = str(v).strip()
		if s == "":
			return None
		return int(float(s))
	except Exception:
		return None


def _extract_in_paths_from_compose(compose_sh: Path) -> List[Path]:
	text = compose_sh.read_text(encoding="utf-8", errors="replace")
	paths = re.findall(r"--in '([^']+)'", text)
	out: List[Path] = []
	seen: set[str] = set()
	for p in paths:
		pp = str(p).strip()
		if not pp or pp in seen:
			continue
		seen.add(pp)
		out.append(Path(pp))
	return out


def _read_sweeps(paths: Sequence[Path]) -> Dict[int, List[Tuple[float, float]]]:
	by: Dict[int, List[Tuple[float, float]]] = {}
	for path in paths:
		if not path.exists():
			continue
		with path.open("r", newline="") as f:
			r = csv.DictReader(f)
			for row in r:
				players = _try_int(row.get("players"))
				k = _try_float(row.get("selection_strength"))
				p3 = _try_float(row.get("p_level_3"))
				if players is None or k is None or p3 is None:
					continue
				by.setdefault(int(players), []).append((float(k), float(p3)))
	# sort & de-dup by k (keep last)
	for n, pts in list(by.items()):
		pts.sort(key=lambda kv: kv[0])
		uniq: Dict[float, float] = {}
		for k, p3 in pts:
			uniq[float(k)] = float(p3)
		by[n] = sorted(uniq.items(), key=lambda kv: kv[0])
	return by


def _collapse_score_xbins(
	by: Dict[int, List[Tuple[float, float]]],
	*,
	k_inf: float,
	beta: float,
	bins: int,
	p3_min: float = 0.0,
	p3_max: float = 1.0,
) -> float | None:
	pts: list[tuple[float, float, int]] = []
	lo = float(p3_min)
	hi = float(p3_max)
	for n, seq in by.items():
		if not seq:
			continue
		for k, p3 in seq:
			pp = float(p3)
			if pp < lo or pp > hi:
				continue
			x = (float(k) - float(k_inf)) * (float(n) ** float(beta))
			if math.isfinite(x) and math.isfinite(pp):
				pts.append((float(x), pp, int(n)))
	if len(pts) < 20:
		return None
	xs = [p[0] for p in pts]
	x_lo = min(xs)
	x_hi = max(xs)
	if not (math.isfinite(x_lo) and math.isfinite(x_hi)):
		return None
	if abs(x_hi - x_lo) <= 1e-12:
		return None

	b = max(8, int(bins))
	step = (x_hi - x_lo) / float(b)
	acc: list[dict[int, list[float]]] = [dict() for _ in range(b)]
	for x, y, n in pts:
		i = int((x - x_lo) / step)
		if i < 0:
			continue
		if i >= b:
			i = b - 1
		acc[i].setdefault(int(n), []).append(float(y))

	vars_: list[float] = []
	for mp in acc:
		if len(mp) < 2:
			continue
		means = []
		for ys in mp.values():
			if ys:
				means.append(sum(ys) / float(len(ys)))
		if len(means) < 2:
			continue
		m = sum(means) / float(len(means))
		v = sum((a - m) * (a - m) for a in means) / float(len(means) - 1)
		vars_.append(float(v))
	if len(vars_) < 3:
		return None
	return float(sum(vars_) / float(len(vars_)))


def _interp_k_at_p(seq: Sequence[Tuple[float, float]], y: float) -> float | None:
	"""Interpolate k at which p(y)=P(Level3) reaches y.

	Returns None if y is not bracketed by any adjacent points.
	If multiple crossings exist due to noise, picks the tightest bracket
	(minimize |p0-y|+|p1-y|).
	"""
	if not seq:
		return None
	y = float(y)
	best = None
	best_span = None
	for (k0, p0), (k1, p1) in zip(seq, seq[1:]):
		p0 = float(p0)
		p1 = float(p1)
		if not (math.isfinite(p0) and math.isfinite(p1)):
			continue
		if (p0 - y) == 0.0:
			return float(k0)
		if (p1 - y) == 0.0:
			return float(k1)
		# bracket check
		if (p0 - y) * (p1 - y) > 0.0:
			continue
		den = (p1 - p0)
		if abs(den) <= 1e-15:
			continue
		t = (y - p0) / den
		if not math.isfinite(t):
			continue
		kk = float(k0) + float(t) * (float(k1) - float(k0))
		span = abs(p0 - y) + abs(p1 - y)
		if best is None or float(span) < float(best_span):
			best = float(kk)
			best_span = float(span)
	return best


def _collapse_score_ygrid(
	by: Dict[int, List[Tuple[float, float]]],
	*,
	k_inf: float,
	beta: float,
	y_grid: Sequence[float],
	p3_min: float = 0.0,
	p3_max: float = 1.0,
	min_ns_per_y: int = 2,
) -> float | None:
	"""Interpolation-based collapse score.

	For each y in y_grid, compute x_N(y) = (k_N(y) - k_inf) * N^beta via
	linear interpolation along each N's (k,p) curve, then score as the mean
	cross-N sample variance of x_N(y) over y values.

	This avoids the fixed-bins / x-range inflation artifact of x-binning.
	"""
	lo = float(p3_min)
	hi = float(p3_max)
	grid = [float(y) for y in y_grid if lo <= float(y) <= hi and math.isfinite(float(y))]
	if len(grid) < 3:
		return None

	# Pre-filter per N to transition region (keep points slightly outside by 1e-12).
	filtered: Dict[int, List[Tuple[float, float]]] = {}
	for n, seq in by.items():
		if not seq:
			continue
		pts = [(float(k), float(p)) for (k, p) in seq if math.isfinite(float(k)) and math.isfinite(float(p))]
		if not pts:
			continue
		# ensure sorted by k
		pts.sort(key=lambda kv: kv[0])
		pts2 = [(k, p) for (k, p) in pts if (p >= lo - 1e-12 and p <= hi + 1e-12)]
		if len(pts2) >= 2:
			filtered[int(n)] = pts2

	if len(filtered) < 2:
		return None

	vars_: list[float] = []
	for y in grid:
		xs: list[float] = []
		for n, seq in filtered.items():
			kk = _interp_k_at_p(seq, y)
			if kk is None:
				continue
			x = (float(kk) - float(k_inf)) * (float(n) ** float(beta))
			if math.isfinite(x):
				xs.append(float(x))
		if len(xs) < int(min_ns_per_y):
			continue
		m = sum(xs) / float(len(xs))
		v = sum((a - m) * (a - m) for a in xs) / float(len(xs) - 1) if len(xs) > 1 else 0.0
		if math.isfinite(v):
			vars_.append(float(v))
	if len(vars_) < 3:
		return None
	return float(sum(vars_) / float(len(vars_)))


def _interp_y_at_x(seq_xy: Sequence[Tuple[float, float]], x: float) -> float | None:
	"""Interpolate y at given x for a monotone seq of (x,y)."""
	if not seq_xy:
		return None
	x = float(x)
	# quick bounds
	x0 = float(seq_xy[0][0])
	x1 = float(seq_xy[-1][0])
	if x < min(x0, x1) - 1e-12 or x > max(x0, x1) + 1e-12:
		return None
	for (xa, ya), (xb, yb) in zip(seq_xy, seq_xy[1:]):
		xa = float(xa)
		xb = float(xb)
		if (xa - x) == 0.0:
			return float(ya)
		if (xb - x) == 0.0:
			return float(yb)
		if (xa - x) * (xb - x) > 0.0:
			continue
		den = (xb - xa)
		if abs(den) <= 1e-15:
			continue
		t = (x - xa) / den
		if not math.isfinite(t):
			continue
		yy = float(ya) + float(t) * (float(yb) - float(ya))
		return float(yy)
	return None


def _collapse_score_xgrid(
	by: Dict[int, List[Tuple[float, float]]],
	*,
	k_inf: float,
	beta: float,
	x_grid_n: int = 41,
	p3_min: float = 0.0,
	p3_max: float = 1.0,
	min_ns_per_x: int = 2,
) -> float | None:
	"""Interpolation-based collapse score on a common x-grid.

	For each N, build (x,y) with x=(k-k_inf)N^beta and y=P(Level3) filtered to
	transition region. Choose a robust global x-window based on pooled x
	percentiles, then interpolate y_N(x) on a fixed x-grid. Score is mean
	cross-N variance of y.
	"""
	lo = float(p3_min)
	hi = float(p3_max)
	# use interior endpoints for defining a stable x-window (avoid requiring curves to hit extremes)
	span = float(hi - lo)
	win_lo = float(lo + 0.10 * span)
	win_hi = float(hi - 0.10 * span)
	if not (win_hi > win_lo + 1e-12):
		win_lo = lo
		win_hi = hi
	seqs: Dict[int, List[Tuple[float, float]]] = {}
	# endpoints for defining a stable transition-focused x-window
	x_at_lo: list[float] = []
	x_at_hi: list[float] = []
	for n, seq in by.items():
		if not seq:
			continue
		# For endpoint interpolation, keep a sorted (k,p) seq (slightly extended bounds).
		kp = [(float(k), float(p)) for (k, p) in seq if math.isfinite(float(k)) and math.isfinite(float(p))]
		if len(kp) < 2:
			continue
		kp.sort(key=lambda kv: kv[0])
		kp2 = [(k, p) for (k, p) in kp if (p >= lo - 1e-12 and p <= hi + 1e-12)]
		if len(kp2) >= 2:
			k_lo = _interp_k_at_p(kp2, win_lo)
			k_hi = _interp_k_at_p(kp2, win_hi)
			if k_lo is not None and k_hi is not None:
				x_at_lo.append((float(k_lo) - float(k_inf)) * (float(n) ** float(beta)))
				x_at_hi.append((float(k_hi) - float(k_inf)) * (float(n) ** float(beta)))
		pts = []
		for k, p in seq:
			pp = float(p)
			if not (pp >= lo and pp <= hi):
				continue
			x = (float(k) - float(k_inf)) * (float(n) ** float(beta))
			if math.isfinite(x) and math.isfinite(pp):
				pts.append((float(x), float(pp)))
		if len(pts) < 2:
			continue
		# sort by x, de-dup x (keep last)
		pts.sort(key=lambda kv: kv[0])
		uniq: Dict[float, float] = {}
		for x, y in pts:
			uniq[float(x)] = float(y)
		pts2 = sorted(uniq.items(), key=lambda kv: kv[0])
		if len(pts2) < 2:
			continue
		seqs[int(n)] = [(float(x), float(y)) for (x, y) in pts2]

	if len(seqs) < 2:
		return None
	if len(x_at_lo) < 3 or len(x_at_hi) < 3:
		return None
	x_at_lo.sort()
	x_at_hi.sort()
	def _pct(xs: list[float], q: float) -> float:
		q = max(0.0, min(1.0, float(q)))
		idx = int(round(q * float(len(xs) - 1)))
		idx = max(0, min(len(xs) - 1, idx))
		return float(xs[idx])
	# robust overlap: try increasingly relaxed quantile pairs
	q_pairs = [(0.80, 0.20), (0.70, 0.30), (0.65, 0.35), (0.60, 0.40), (0.55, 0.45), (0.50, 0.50)]
	x_lo = None
	x_hi = None
	for qlo, qhi in q_pairs:
		lo_q = _pct(x_at_lo, qlo)
		hi_q = _pct(x_at_hi, qhi)
		if math.isfinite(lo_q) and math.isfinite(hi_q) and hi_q > lo_q + 1e-12:
			x_lo = float(lo_q)
			x_hi = float(hi_q)
			break
	if x_lo is None or x_hi is None:
		return None
	if not (math.isfinite(x_lo) and math.isfinite(x_hi)):
		return None
	if x_hi <= x_lo + 1e-12:
		return None

	nx = max(11, int(x_grid_n))
	step = (x_hi - x_lo) / float(nx - 1)
	vars_: list[float] = []
	for i in range(nx):
		x = x_lo + float(i) * step
		ys: list[float] = []
		for _n, sxy in seqs.items():
			y = _interp_y_at_x(sxy, x)
			if y is None:
				continue
			if math.isfinite(float(y)):
				ys.append(float(y))
		if len(ys) < int(min_ns_per_x):
			continue
		m = sum(ys) / float(len(ys))
		v = sum((a - m) * (a - m) for a in ys) / float(len(ys) - 1) if len(ys) > 1 else 0.0
		if math.isfinite(v):
			vars_.append(float(v))
	if len(vars_) < max(3, int(0.3 * float(nx))):
		return None
	return float(sum(vars_) / float(len(vars_)))


def _parse_grid_spec(spec: str) -> list[float]:
	"""Parse either comma list or lo:hi:step spec."""
	s = str(spec).strip()
	if ":" in s:
		return _parse_range(s)
	return _parse_float_list(s)


def _parse_float_list(spec: str) -> list[float]:
	out: list[float] = []
	for part in str(spec).split(","):
		part = part.strip()
		if not part:
			continue
		out.append(float(part))
	return out


def _parse_range(spec: str) -> list[float]:
	# lo:hi:step
	parts = [p.strip() for p in str(spec).split(":")]
	if len(parts) != 3:
		raise ValueError("range must be lo:hi:step")
	lo = float(parts[0])
	hi = float(parts[1])
	st = float(parts[2])
	if st <= 0:
		raise ValueError("step must be >0")
	out: list[float] = []
	k = 0
	x = lo
	while x <= hi + 1e-12 and k < 100000:
		out.append(float(x))
		k += 1
		x = lo + k * st
	return out


def main(argv: Optional[Sequence[str]] = None) -> None:
	p = argparse.ArgumentParser(description="Collapse-score sensitivity over (k_inf, beta) grids")
	p.add_argument("--compose", required=True, help="Compose .sh produced by analysis.compose_rho_curve_commands")
	p.add_argument("--out", required=True, help="Output CSV path")
	p.add_argument("--out-json", default=None, help="Optional JSON summary path")
	p.add_argument("--k-inf-grid", default="0.92:0.96:0.002", help="k_inf grid as lo:hi:step")
	p.add_argument("--betas", default="0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.5", help="Comma-separated beta list")
	p.add_argument("--bins", type=int, default=35, help="Number of x-bins for score (default: 35)")
	p.add_argument(
		"--score-method",
		choices=["xbins", "xgrid", "ygrid"],
		default="xbins",
		help="Collapse score method: xbins (legacy), xgrid (robust), or ygrid (experimental) (default: xbins)",
	)
	p.add_argument("--x-grid-n", type=int, default=41, help="For xgrid method: number of x grid points (default: 41)")
	p.add_argument(
		"--y-grid",
		default=None,
		help="For ygrid method: y values as lo:hi:step or comma list (default: use p3-min..p3-max step 0.02)",
	)
	p.add_argument("--p3-min", type=float, default=0.1, help="Only score points with P(Level3) >= this (default: 0.1)")
	p.add_argument("--p3-max", type=float, default=0.9, help="Only score points with P(Level3) <= this (default: 0.9)")
	args = p.parse_args(list(argv) if argv is not None else None)

	in_paths = _extract_in_paths_from_compose(Path(str(args.compose)))
	if not in_paths:
		raise SystemExit("No --in paths found in compose")
	by = _read_sweeps(in_paths)

	k_infs = _parse_range(str(args.k_inf_grid))
	betas = _parse_float_list(str(args.betas))
	score_method = str(args.score_method)
	if score_method == "ygrid":
		if args.y_grid is None:
			# default: 0.02 step over [p3_min,p3_max]
			y_grid = _parse_range(f"{float(args.p3_min)}:{float(args.p3_max)}:0.02")
			y_grid_spec = f"{float(args.p3_min)}:{float(args.p3_max)}:0.02"
		else:
			y_grid = _parse_grid_spec(str(args.y_grid))
			y_grid_spec = str(args.y_grid)
	else:
		y_grid = []
		y_grid_spec = None

	rows: list[dict] = []
	best = None
	for k_inf in k_infs:
		for beta in betas:
			if score_method == "ygrid":
				s = _collapse_score_ygrid(
					by,
					k_inf=float(k_inf),
					beta=float(beta),
					y_grid=y_grid,
					p3_min=float(args.p3_min),
					p3_max=float(args.p3_max),
				)
			elif score_method == "xgrid":
				s = _collapse_score_xgrid(
					by,
					k_inf=float(k_inf),
					beta=float(beta),
					x_grid_n=int(args.x_grid_n),
					p3_min=float(args.p3_min),
					p3_max=float(args.p3_max),
				)
			else:
				s = _collapse_score_xbins(
					by,
					k_inf=float(k_inf),
					beta=float(beta),
					bins=int(args.bins),
					p3_min=float(args.p3_min),
					p3_max=float(args.p3_max),
				)
			if s is None:
				continue
			row = {
				"k_inf": float(k_inf),
				"beta": float(beta),
				"score": float(s),
				"bins": int(args.bins),
				"p3_min": float(args.p3_min),
				"p3_max": float(args.p3_max),
			}
			rows.append(row)
			if best is None or float(s) < float(best["score"]):
				best = dict(row)

	out_path = Path(str(args.out))
	out_path.parent.mkdir(parents=True, exist_ok=True)
	rows_sorted = sorted(rows, key=lambda r: (r["score"], r["k_inf"], r["beta"]))
	with out_path.open("w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=["k_inf", "beta", "score", "bins", "p3_min", "p3_max"])
		w.writeheader()
		for r in rows_sorted:
			w.writerow(r)

	if args.out_json:
		j = {
			"compose": str(args.compose),
			"n_N": len(by),
			"k_inf_grid": str(args.k_inf_grid),
			"betas": betas,
			"bins": int(args.bins),
			"score_method": score_method,
			"x_grid_n": int(args.x_grid_n),
			"y_grid": y_grid if score_method == "ygrid" else None,
			"y_grid_spec": y_grid_spec,
			"p3_min": float(args.p3_min),
			"p3_max": float(args.p3_max),
			"best": best,
			"top10": rows_sorted[:10],
		}
		Path(str(args.out_json)).write_text(json.dumps(j, indent=2), encoding="utf-8")

	print(f"Wrote: {out_path}")
	if args.out_json:
		print(f"Wrote: {args.out_json}")
	if best:
		print(f"Best: k_inf={best['k_inf']:.6g} beta={best['beta']:.6g} score={best['score']:.6g}")


if __name__ == "__main__":
	main()
