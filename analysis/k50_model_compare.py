"""Compare candidate finite-size models for k50(N) with AIC/BIC.

Purpose
- When k50(N) is non-monotone or 1/N fits poorly, we want a small, reproducible
  tool to compare model families without changing any CSV contracts.

Inputs
- A *_summary.csv produced by analysis.rho_curve_scaling.
  Uses columns: players, k50, k50_censored_low, and optionally Bayesian CI.

Models (default)
- A: k(N) = k_inf + c / N
- B: k(N) = k_inf + c / N^beta  (beta searched over a grid)
- C: k(N) = k_inf + c / sqrt(N) + d / N
- D: piecewise-A split at N < split vs N >= split (two independent A fits)

Outputs
- <outdir>/<prefix>_k50_model_compare.csv
- <outdir>/<prefix>_k50_model_compare.json
- Optional plot: <outdir>/figs/<prefix>_k50_model_compare.png (needs matplotlib)

Design constraints
- Lives in analysis/ and does not import simulation/.
- Stdlib-first; plotting is optional.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class Point:
	N: int
	k50: float
	w: float  # weight (inverse variance)


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


def _read_points(summary_csv: Path, *, use_bayes_mean: bool, use_bayes_ci_weights: bool) -> list[Point]:
	pts: list[Point] = []
	with summary_csv.open("r", newline="") as f:
		r = csv.DictReader(f)
		for row in r:
			N = _try_int(row.get("players"))
			if N is None or N <= 0:
				continue
			censored = str(row.get("k50_censored_low") or "").strip().lower() in ("1", "true", "yes")
			if censored:
				continue

			k = None
			if use_bayes_mean:
				k = _try_float(row.get("k50_bayes_mean"))
			if k is None:
				k = _try_float(row.get("k50"))
			if k is None or not math.isfinite(float(k)):
				continue

			w = 1.0
			if use_bayes_ci_weights:
				lo = _try_float(row.get("k50_bayes_ci_low"))
				hi = _try_float(row.get("k50_bayes_ci_high"))
				if lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi) and hi > lo:
					# Approx sigma from 95% CI width: hi-lo ~= 3.92*sigma
					sigma = (float(hi) - float(lo)) / 3.92
					if sigma > 0 and math.isfinite(sigma):
						w = 1.0 / (sigma * sigma)
			pts.append(Point(N=int(N), k50=float(k), w=float(w)))

	pts.sort(key=lambda p: p.N)
	return pts


def _wls_fit_intercept_slope(xs: Sequence[float], ys: Sequence[float], ws: Sequence[float]) -> tuple[float, float]:
	# Fit y = a + b x with weights w.
	S = sum(ws)
	Sx = sum(w * x for w, x in zip(ws, xs))
	Sy = sum(w * y for w, y in zip(ws, ys))
	Sxx = sum(w * x * x for w, x in zip(ws, xs))
	Sxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))
	d = S * Sxx - Sx * Sx
	if not math.isfinite(d) or abs(d) <= 1e-18:
		raise ValueError("Singular WLS for intercept+slope")
	b = (S * Sxy - Sx * Sy) / d
	a = (Sy - b * Sx) / S
	return float(a), float(b)


def _solve_3x3(A: list[list[float]], b: list[float]) -> list[float]:
	# Gaussian elimination with partial pivoting.
	M = [row[:] + [rhs] for row, rhs in zip(A, b)]
	for col in range(3):
		# pivot
		pivot = max(range(col, 3), key=lambda r: abs(M[r][col]))
		if abs(M[pivot][col]) <= 1e-18:
			raise ValueError("Singular 3x3 system")
		if pivot != col:
			M[col], M[pivot] = M[pivot], M[col]
		# eliminate
		for r in range(col + 1, 3):
			fac = M[r][col] / M[col][col]
			for c in range(col, 4):
				M[r][c] -= fac * M[col][c]
	# back-substitute
	x = [0.0, 0.0, 0.0]
	for r in reversed(range(3)):
		s = M[r][3] - sum(M[r][c] * x[c] for c in range(r + 1, 3))
		x[r] = s / M[r][r]
	return [float(v) for v in x]


def _wls_fit_3params(X1: Sequence[float], X2: Sequence[float], ys: Sequence[float], ws: Sequence[float]) -> tuple[float, float, float]:
	# Fit y = a + b*X1 + c*X2 with weights.
	S = sum(ws)
	S1 = sum(w * x1 for w, x1 in zip(ws, X1))
	S2 = sum(w * x2 for w, x2 in zip(ws, X2))
	S11 = sum(w * x1 * x1 for w, x1 in zip(ws, X1))
	S22 = sum(w * x2 * x2 for w, x2 in zip(ws, X2))
	S12 = sum(w * x1 * x2 for w, x1, x2 in zip(ws, X1, X2))
	Sy = sum(w * y for w, y in zip(ws, ys))
	S1y = sum(w * x1 * y for w, x1, y in zip(ws, X1, ys))
	S2y = sum(w * x2 * y for w, x2, y in zip(ws, X2, ys))
	A = [
		[float(S), float(S1), float(S2)],
		[float(S1), float(S11), float(S12)],
		[float(S2), float(S12), float(S22)],
	]
	b = [float(Sy), float(S1y), float(S2y)]
	a, bb, cc = _solve_3x3(A, b)
	return float(a), float(bb), float(cc)


def _sse(y_hat: Iterable[float], ys: Iterable[float], ws: Iterable[float]) -> float:
	return float(sum(w * (y - yh) * (y - yh) for yh, y, w in zip(y_hat, ys, ws)))


def _aic_bic(*, n: int, sse: float, p: int) -> tuple[float, float]:
	# Gaussian SSE-based AIC/BIC up to additive constant.
	if n <= 0:
		raise ValueError("n must be >0")
	sse2 = max(1e-18, float(sse))
	aic = float(n) * math.log(sse2 / float(n)) + 2.0 * float(p)
	bic = float(n) * math.log(sse2 / float(n)) + float(p) * math.log(float(n))
	return float(aic), float(bic)


def _parse_grid(spec: str) -> list[float]:
	# "a:b:step" inclusive-ish.
	parts = [p.strip() for p in str(spec).split(":")]
	if len(parts) != 3:
		raise ValueError("beta grid must be like lo:hi:step")
	lo = float(parts[0])
	hi = float(parts[1])
	st = float(parts[2])
	if st <= 0:
		raise ValueError("step must be >0")
	out: list[float] = []
	x = lo
	# avoid floating drift
	k = 0
	while x <= hi + 1e-12 and k < 100000:
		out.append(float(x))
		x = lo + (k + 1) * st
		k += 1
	return out


def _maybe_plot(outpath: Path, *, pts: list[Point], best: dict, curves: dict[str, dict]) -> None:
	try:
		import matplotlib.pyplot as plt
	except ModuleNotFoundError:
		return

	Ns = [p.N for p in pts]
	Ks = [p.k50 for p in pts]

	plt.figure(figsize=(7.0, 4.2))
	plt.scatter(Ns, Ks, color="black", label="k50")
	plt.xscale("log")
	plt.xlabel("N (log scale)")
	plt.ylabel("k50")
	plt.grid(True, alpha=0.25)

	# Plot best-fit curve on a dense N grid
	name = str(best.get("model"))
	params = curves.get(name, {})
	if params:
		Ns_dense = sorted(set(Ns + [int(min(Ns) * (1.2 ** i)) for i in range(0, 20)]))
		Ns_dense = [n for n in Ns_dense if n > 0]
		yhat = []
		if name == "A":
			k_inf = float(params["k_inf"])
			c = float(params["c"])
			yhat = [k_inf + c / n for n in Ns_dense]
		elif name == "B":
			k_inf = float(params["k_inf"])
			c = float(params["c"])
			beta = float(params["beta"])
			yhat = [k_inf + c / (n ** beta) for n in Ns_dense]
		elif name == "C":
			k_inf = float(params["k_inf"])
			c = float(params["c"])
			d = float(params["d"])
			yhat = [k_inf + c / math.sqrt(n) + d / n for n in Ns_dense]
		elif name == "D":
			split = int(params["split"])
			k_inf_lo = float(params["k_inf_lo"])
			c_lo = float(params["c_lo"])
			k_inf_hi = float(params["k_inf_hi"])
			c_hi = float(params["c_hi"])
			yhat = [
				(k_inf_lo + c_lo / n) if n < split else (k_inf_hi + c_hi / n)
				for n in Ns_dense
			]
		if yhat:
			plt.plot(Ns_dense, yhat, linewidth=2, label=f"best model {name}")

	plt.legend()
	outpath.parent.mkdir(parents=True, exist_ok=True)
	plt.tight_layout()
	plt.savefig(outpath, dpi=170)
	plt.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
	p = argparse.ArgumentParser(description="Compare k50(N) model families with AIC/BIC")
	p.add_argument("--summary", required=True, help="Path to *_summary.csv")
	p.add_argument("--outdir", default="outputs/analysis/rho_curve", help="Output directory")
	p.add_argument("--prefix", required=True, help="Prefix for output files")
	p.add_argument("--use-bayes-mean", type=int, choices=[0, 1], default=1, help="If 1, prefer k50_bayes_mean when available")
	p.add_argument(
		"--use-bayes-ci-weights",
		type=int,
		choices=[0, 1],
		default=1,
		help="If 1, weight points by inverse variance estimated from Bayesian CI width (when available)",
	)
	p.add_argument("--beta-grid", default="0.2:2.0:0.01", help="Grid search for beta in model B: lo:hi:step")
	p.add_argument("--piecewise-split", type=int, default=300, help="Split N for piecewise model D (N < split vs >= split)")
	args = p.parse_args(list(argv) if argv is not None else None)

	summary_csv = Path(str(args.summary))
	outdir = Path(str(args.outdir))
	prefix = str(args.prefix)

	pts = _read_points(summary_csv, use_bayes_mean=bool(int(args.use_bayes_mean) == 1), use_bayes_ci_weights=bool(int(args.use_bayes_ci_weights) == 1))
	if len(pts) < 4:
		raise SystemExit("Need at least 4 usable (non-censored) N points for model comparison")

	Ns = [p.N for p in pts]
	Ys = [p.k50 for p in pts]
	Ws = [p.w for p in pts]
	n = len(pts)

	rows: list[dict] = []
	curves: dict[str, dict] = {}

	# Model A
	xA = [1.0 / float(N) for N in Ns]
	k_inf_A, c_A = _wls_fit_intercept_slope(xA, Ys, Ws)
	yhat_A = [k_inf_A + c_A * x for x in xA]
	sse_A = _sse(yhat_A, Ys, Ws)
	aic_A, bic_A = _aic_bic(n=n, sse=sse_A, p=2)
	rows.append({"model": "A", "params": {"k_inf": k_inf_A, "c": c_A}, "sse": sse_A, "aic": aic_A, "bic": bic_A})
	curves["A"] = {"k_inf": k_inf_A, "c": c_A}

	# Model B: grid beta
	best_B = None
	for beta in _parse_grid(str(args.beta_grid)):
		xB = [1.0 / (float(N) ** float(beta)) for N in Ns]
		k_inf_B, c_B = _wls_fit_intercept_slope(xB, Ys, Ws)
		yhat_B = [k_inf_B + c_B * x for x in xB]
		sse_B = _sse(yhat_B, Ys, Ws)
		if best_B is None or sse_B < best_B["sse"]:
			best_B = {"beta": float(beta), "k_inf": float(k_inf_B), "c": float(c_B), "sse": float(sse_B)}
	if best_B is None:
		raise SystemExit("Model B beta grid search failed")
	aic_B, bic_B = _aic_bic(n=n, sse=float(best_B["sse"]), p=3)
	rows.append({"model": "B", "params": {"k_inf": best_B["k_inf"], "c": best_B["c"], "beta": best_B["beta"]}, "sse": best_B["sse"], "aic": aic_B, "bic": bic_B})
	curves["B"] = {"k_inf": best_B["k_inf"], "c": best_B["c"], "beta": best_B["beta"]}

	# Model C
	xC1 = [1.0 / math.sqrt(float(N)) for N in Ns]
	xC2 = [1.0 / float(N) for N in Ns]
	k_inf_C, c_C, d_C = _wls_fit_3params(xC1, xC2, Ys, Ws)
	yhat_C = [k_inf_C + c_C * a + d_C * b for a, b in zip(xC1, xC2)]
	sse_C = _sse(yhat_C, Ys, Ws)
	aic_C, bic_C = _aic_bic(n=n, sse=sse_C, p=3)
	rows.append({"model": "C", "params": {"k_inf": k_inf_C, "c": c_C, "d": d_C}, "sse": sse_C, "aic": aic_C, "bic": bic_C})
	curves["C"] = {"k_inf": k_inf_C, "c": c_C, "d": d_C}

	# Model D: piecewise-A
	split = int(args.piecewise_split)
	idx_lo = [i for i, N in enumerate(Ns) if N < split]
	idx_hi = [i for i, N in enumerate(Ns) if N >= split]
	if len(idx_lo) >= 2 and len(idx_hi) >= 2:
		Ns_lo = [Ns[i] for i in idx_lo]
		Ys_lo = [Ys[i] for i in idx_lo]
		Ws_lo = [Ws[i] for i in idx_lo]
		Ns_hi = [Ns[i] for i in idx_hi]
		Ys_hi = [Ys[i] for i in idx_hi]
		Ws_hi = [Ws[i] for i in idx_hi]
		x_lo = [1.0 / float(N) for N in Ns_lo]
		x_hi = [1.0 / float(N) for N in Ns_hi]
		k_inf_lo, c_lo = _wls_fit_intercept_slope(x_lo, Ys_lo, Ws_lo)
		k_inf_hi, c_hi = _wls_fit_intercept_slope(x_hi, Ys_hi, Ws_hi)
		yhat_D = []
		for N in Ns:
			if N < split:
				yhat_D.append(k_inf_lo + c_lo / float(N))
			else:
				yhat_D.append(k_inf_hi + c_hi / float(N))
		sse_D = _sse(yhat_D, Ys, Ws)
		aic_D, bic_D = _aic_bic(n=n, sse=sse_D, p=4)
		rows.append(
			{
				"model": "D",
				"params": {"split": split, "k_inf_lo": k_inf_lo, "c_lo": c_lo, "k_inf_hi": k_inf_hi, "c_hi": c_hi},
				"sse": sse_D,
				"aic": aic_D,
				"bic": bic_D,
			}
		)
		curves["D"] = {"split": split, "k_inf_lo": k_inf_lo, "c_lo": c_lo, "k_inf_hi": k_inf_hi, "c_hi": c_hi}

	# Pick best by BIC (default)
	rows_sorted = sorted(rows, key=lambda r: float(r["bic"]))
	best = rows_sorted[0]

	# Write CSV
	out_csv = outdir / f"{prefix}_k50_model_compare.csv"
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	with out_csv.open("w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=["model", "bic", "aic", "sse", "params_json"])
		w.writeheader()
		for r in rows_sorted:
			w.writerow({"model": r["model"], "bic": r["bic"], "aic": r["aic"], "sse": r["sse"], "params_json": json.dumps(r["params"], sort_keys=True)})

	# Write JSON
	out_json = outdir / f"{prefix}_k50_model_compare.json"
	out_json.write_text(
		json.dumps(
			{
				"summary": str(summary_csv),
				"n_points": n,
				"Ns": Ns,
				"use_bayes_mean": bool(int(args.use_bayes_mean) == 1),
				"use_bayes_ci_weights": bool(int(args.use_bayes_ci_weights) == 1),
				"models": rows_sorted,
				"best_by_bic": best,
			},
			indent=2,
		),
		encoding="utf-8",
	)

	# Plot
	fig_path = outdir / "figs" / f"{prefix}_k50_model_compare.png"
	_maybe_plot(fig_path, pts=pts, best=best.get("best_by_bic") or best, curves=curves)

	print(f"Wrote: {out_csv}")
	print(f"Wrote: {out_json}")
	if fig_path.exists():
		print(f"Wrote: {fig_path}")
	print(f"Best by BIC: model={best['model']} bic={best['bic']:.6g} params={best['params']}")


if __name__ == "__main__":
	main()
