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


def _collapse_score(
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
	p.add_argument("--p3-min", type=float, default=0.1, help="Only score points with P(Level3) >= this (default: 0.1)")
	p.add_argument("--p3-max", type=float, default=0.9, help="Only score points with P(Level3) <= this (default: 0.9)")
	args = p.parse_args(list(argv) if argv is not None else None)

	in_paths = _extract_in_paths_from_compose(Path(str(args.compose)))
	if not in_paths:
		raise SystemExit("No --in paths found in compose")
	by = _read_sweeps(in_paths)

	k_infs = _parse_range(str(args.k_inf_grid))
	betas = _parse_float_list(str(args.betas))

	rows: list[dict] = []
	best = None
	for k_inf in k_infs:
		for beta in betas:
			s = _collapse_score(
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
