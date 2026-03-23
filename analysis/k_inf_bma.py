"""Bayesian model averaging (BMA) for k_inf from k50(N) fits.

Goal
- Produce a defensible posterior distribution for k_inf without new simulation.
- Use existing per-N Bayesian uncertainty (k50_bayes_mean/std) to generate
  parameter uncertainty via Monte Carlo refits.
- Use model-compare BIC values to compute approximate model posterior weights:
    w_m \propto exp(-0.5 * (BIC_m - min_BIC))

Inputs
- A bayes-fit JSON produced by analysis.rho_curve_scaling / rho_curve_viz:
    *_bayes_fit.json
  Contains per_N entries with players, k50_bayes_mean, k50_bayes_std.
- A model-compare JSON produced by analysis.k50_model_compare:
    *_k50_model_compare.json

Models
- A: k(N) = k_inf + c/N
- B: k(N) = k_inf + c/N^beta (beta fixed to the best params in model-compare)
- C: k(N) = k_inf + c/sqrt(N) + d/N
- D: piecewise A with fixed split (from model-compare); report k_inf_hi by default.

Outputs
- JSON with:
  - per-model weights, per-model posterior summaries, and BMA mixture summaries.
- Optional CSV with posterior draws.

Design constraints
- Lives in analysis/ and does not import simulation/.
- Stdlib-only.

Notes
- This is an approximate BMA: BIC-based weights + Normal approximation for per-N
  k50 posteriors + refit-based parameter sampling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class K50Point:
	N: int
	mu: float
	sigma: float


def _pct(xs: Sequence[float], q: float) -> float:
	if not xs:
		raise ValueError("empty")
	q = float(q)
	q = max(0.0, min(1.0, q))
	ys = sorted(float(x) for x in xs)
	idx = int(round(q * float(len(ys) - 1)))
	idx = max(0, min(len(ys) - 1, idx))
	return float(ys[idx])


def _summarize(xs: Sequence[float]) -> dict:
	ys = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
	if not ys:
		return {"n": 0}
	mean = sum(ys) / float(len(ys))
	var = 0.0
	if len(ys) > 1:
		var = sum((x - mean) * (x - mean) for x in ys) / float(len(ys) - 1)
	std = math.sqrt(max(0.0, var))
	return {
		"n": int(len(ys)),
		"mean": float(mean),
		"std": float(std),
		"p025": float(_pct(ys, 0.025)),
		"p50": float(_pct(ys, 0.50)),
		"p975": float(_pct(ys, 0.975)),
	}


def _wls_fit_intercept_slope(xs: Sequence[float], ys: Sequence[float], ws: Sequence[float]) -> tuple[float, float]:
	"""Fit y = a + b x with weights w."""
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
	M = [row[:] + [rhs] for row, rhs in zip(A, b)]
	for col in range(3):
		pivot = max(range(col, 3), key=lambda r: abs(M[r][col]))
		if abs(M[pivot][col]) <= 1e-18:
			raise ValueError("Singular 3x3 system")
		if pivot != col:
			M[col], M[pivot] = M[pivot], M[col]
		for r in range(col + 1, 3):
			fac = M[r][col] / M[col][col]
			for c in range(col, 4):
				M[r][c] -= fac * M[col][c]
	x = [0.0, 0.0, 0.0]
	for r in reversed(range(3)):
		s = M[r][3] - sum(M[r][c] * x[c] for c in range(r + 1, 3))
		x[r] = s / M[r][r]
	return [float(v) for v in x]


def _wls_fit_3params(X1: Sequence[float], X2: Sequence[float], ys: Sequence[float], ws: Sequence[float]) -> tuple[float, float, float]:
	"""Fit y = a + b*X1 + c*X2 with weights."""
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
	bb = [float(Sy), float(S1y), float(S2y)]
	a, b1, b2 = _solve_3x3(A, bb)
	return float(a), float(b1), float(b2)


def _read_bayes_fit(path: Path) -> list[K50Point]:
	obj = json.loads(path.read_text(encoding="utf-8"))
	per = obj.get("per_N")
	if not isinstance(per, list) or not per:
		raise ValueError("bayes-fit json missing per_N")
	pts: list[K50Point] = []
	for row in per:
		N = int(row.get("players"))
		mu = float(row.get("k50_bayes_mean"))
		sigma = float(row.get("k50_bayes_std"))
		if N <= 0 or not math.isfinite(mu) or not math.isfinite(sigma) or sigma <= 0:
			continue
		pts.append(K50Point(N=N, mu=mu, sigma=sigma))
	pts.sort(key=lambda p: p.N)
	if len(pts) < 3:
		raise ValueError("need >=3 k50 points")
	return pts


def _read_model_compare(path: Path) -> dict:
	obj = json.loads(path.read_text(encoding="utf-8"))
	models = obj.get("models")
	if not isinstance(models, list) or not models:
		raise ValueError("model-compare json missing models")
	out: dict = {"models": {}}
	for m in models:
		name = str(m.get("model"))
		bic = float(m.get("bic"))
		params = m.get("params") or {}
		out["models"][name] = {"bic": bic, "params": params}
	return out


def _bic_weights(model_info: dict) -> dict:
	bics = {k: float(v["bic"]) for k, v in model_info["models"].items() if math.isfinite(float(v["bic"]))}
	if not bics:
		raise ValueError("no bics")
	min_bic = min(bics.values())
	unn = {k: math.exp(-0.5 * (v - min_bic)) for k, v in bics.items()}
	Z = sum(unn.values())
	if not math.isfinite(Z) or Z <= 0:
		raise ValueError("bad weight normalization")
	w = {k: float(v / Z) for k, v in unn.items()}
	return {"min_bic": float(min_bic), "weights": w}


def _fit_model_A(Ns: Sequence[int], ys: Sequence[float], ws: Sequence[float]) -> float:
	x = [1.0 / float(n) for n in Ns]
	k_inf, _c = _wls_fit_intercept_slope(x, ys, ws)
	return float(k_inf)


def _fit_model_B(Ns: Sequence[int], ys: Sequence[float], ws: Sequence[float], *, beta: float) -> float:
	x = [1.0 / (float(n) ** float(beta)) for n in Ns]
	k_inf, _c = _wls_fit_intercept_slope(x, ys, ws)
	return float(k_inf)


def _fit_model_C(Ns: Sequence[int], ys: Sequence[float], ws: Sequence[float]) -> float:
	X1 = [1.0 / math.sqrt(float(n)) for n in Ns]
	X2 = [1.0 / float(n) for n in Ns]
	k_inf, _c, _d = _wls_fit_3params(X1, X2, ys, ws)
	return float(k_inf)


def _fit_model_D(
	Ns: Sequence[int],
	ys: Sequence[float],
	ws: Sequence[float],
	*,
	split: int,
	which: str,
) -> float:
	if which not in ("lo", "hi"):
		raise ValueError("which must be lo or hi")
	idx = [i for i, n in enumerate(Ns) if (n < split if which == "lo" else n >= split)]
	if len(idx) < 2:
		raise ValueError("not enough points for piecewise fit")
	n2 = [Ns[i] for i in idx]
	y2 = [ys[i] for i in idx]
	w2 = [ws[i] for i in idx]
	return _fit_model_A(n2, y2, w2)


def main(argv: Optional[Sequence[str]] = None) -> None:
	p = argparse.ArgumentParser(description="Bayesian model averaging for k_inf")
	p.add_argument("--bayes-fit-json", required=True, help="Path to *_bayes_fit.json")
	p.add_argument("--model-compare-json", required=True, help="Path to *_k50_model_compare.json")
	p.add_argument("--out-json", required=True, help="Output JSON path")
	p.add_argument("--out-csv", default=None, help="Optional output CSV with draws")
	p.add_argument("--draws", type=int, default=5000, help="Number of Monte Carlo refit draws (default: 5000)")
	p.add_argument("--seed", type=int, default=123, help="RNG seed (default: 123)")
	p.add_argument("--piecewise-target", choices=["hi", "lo"], default="hi", help="For model D, report k_inf_hi or k_inf_lo (default: hi)")
	args = p.parse_args(list(argv) if argv is not None else None)

	pts = _read_bayes_fit(Path(str(args.bayes_fit_json)))
	info = _read_model_compare(Path(str(args.model_compare_json)))
	winfo = _bic_weights(info)
	weights: Dict[str, float] = dict(winfo["weights"])

	Ns = [p.N for p in pts]
	mus = [p.mu for p in pts]
	sigmas = [p.sigma for p in pts]
	ws = [1.0 / (s * s) for s in sigmas]

	# Fixed hyper-params from model-compare best fits.
	beta_B = float((info["models"].get("B") or {}).get("params", {}).get("beta", 2.0))
	split_D = int((info["models"].get("D") or {}).get("params", {}).get("split", 300))
	which_D = str(args.piecewise_target)

	need = [m for m in ("A", "B", "C", "D") if m in weights]
	if not need:
		raise SystemExit("No supported models found in model-compare json")

	rng = random.Random(int(args.seed))

	draws_by_model: Dict[str, List[float]] = {m: [] for m in need}
	mix_draws: List[float] = []
	mix_model: List[str] = []

	# Prepare cumulative weights for categorical sampling.
	ordered = sorted(need)
	cum = []
	acc = 0.0
	for m in ordered:
		acc += float(weights[m])
		cum.append((acc, m))
	# normalize in case of rounding
	cum[-1] = (1.0, cum[-1][1])

	def pick_model() -> str:
		u = rng.random()
		for c, m in cum:
			if u <= c:
				return m
		return cum[-1][1]

	for _i in range(int(args.draws)):
		# sample y per N from Normal(mu, sigma)
		ys = [rng.gauss(mu, s) for mu, s in zip(mus, sigmas)]
		for m in need:
			try:
				if m == "A":
					k = _fit_model_A(Ns, ys, ws)
				elif m == "B":
					k = _fit_model_B(Ns, ys, ws, beta=beta_B)
				elif m == "C":
					k = _fit_model_C(Ns, ys, ws)
				elif m == "D":
					k = _fit_model_D(Ns, ys, ws, split=split_D, which=which_D)
				else:
					continue
				draws_by_model[m].append(float(k))
			except Exception:
				# keep sample sizes aligned by skipping; mixture will still work with remaining
				pass
		# mixture draw (use same ys sample to preserve uncertainty propagation)
		m = pick_model()
		if draws_by_model[m]:
			mix_draws.append(float(draws_by_model[m][-1]))
			mix_model.append(m)

	out = {
		"inputs": {
			"bayes_fit_json": str(args.bayes_fit_json),
			"model_compare_json": str(args.model_compare_json),
			"draws": int(args.draws),
			"seed": int(args.seed),
			"piecewise_target": which_D,
			"fixed_beta_B": float(beta_B),
			"fixed_split_D": int(split_D),
		},
		"bic": {
			"min_bic": float(winfo["min_bic"]),
			"weights": weights,
		},
		"per_model": {m: {"summary": _summarize(draws_by_model[m])} for m in need},
		"bma": {
			"summary": _summarize(mix_draws),
			"n_mix": int(len(mix_draws)),
			"mix_model_counts": {m: int(mix_model.count(m)) for m in need},
		},
	}

	out_path = Path(str(args.out_json))
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

	if args.out_csv:
		csv_path = Path(str(args.out_csv))
		csv_path.parent.mkdir(parents=True, exist_ok=True)
		with csv_path.open("w", newline="") as f:
			w = csv.writer(f)
			header = ["draw", "bma_k_inf", "bma_model"] + [f"k_inf_{m}" for m in need]
			w.writerow(header)
			for i in range(min(len(mix_draws), *(len(draws_by_model[m]) for m in need))):
				row = [i, mix_draws[i], mix_model[i]] + [draws_by_model[m][i] for m in need]
				w.writerow(row)


if __name__ == "__main__":
	main()
