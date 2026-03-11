from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SummaryRow:
	players: int
	k50: float
	k50_censored_low: bool
	k50_boot_ci_low: float | None
	k50_boot_ci_high: float | None
	k50_bayes_mean: float | None
	k50_bayes_std: float | None
	k50_bayes_ci_low: float | None
	k50_bayes_ci_high: float | None


def _parse_bool(v: str | None) -> bool:
	if v is None:
		return False
	s = str(v).strip().lower()
	return s in {"1", "true", "yes", "y"}


def read_summary(path: Path) -> dict[int, SummaryRow]:
	rows: dict[int, SummaryRow] = {}
	with path.open(newline="") as f:
		for r in csv.DictReader(f):
			p_raw = r.get("players")
			if not p_raw:
				continue
			players = int(float(p_raw))
			k50_raw = r.get("k50")
			if not k50_raw:
				continue
			k50 = float(k50_raw)

			def fopt(key: str) -> float | None:
				v = r.get(key)
				if v is None or str(v).strip() == "":
					return None
				return float(v)

			row = SummaryRow(
				players=players,
				k50=k50,
				k50_censored_low=_parse_bool(r.get("k50_censored_low")),
				k50_boot_ci_low=fopt("k50_boot_ci_low"),
				k50_boot_ci_high=fopt("k50_boot_ci_high"),
				k50_bayes_mean=fopt("k50_bayes_mean"),
				k50_bayes_std=fopt("k50_bayes_std"),
				k50_bayes_ci_low=fopt("k50_bayes_ci_low"),
				k50_bayes_ci_high=fopt("k50_bayes_ci_high"),
			)
			rows[int(players)] = row
	return rows


def _ci_width(lo: float | None, hi: float | None) -> float | None:
	if lo is None or hi is None:
		return None
	return float(hi) - float(lo)


def _fmt(x: float | None, *, digits: int = 4) -> str:
	if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
		return ""
	return f"{float(x):.{digits}g}"


def _merge_p3_points(paths: list[Path]) -> dict[tuple[int, float], tuple[float, int]]:
	"""Return (players,k)->(p3_weighted_mean,total_seeds)."""
	acc: dict[tuple[int, float], list[float]] = {}
	# store as [sum_p3_seeds, sum_seeds]
	for path in paths:
		with path.open(newline="") as f:
			for r in csv.DictReader(f):
				try:
					players = int(float(r.get("players") or ""))
					k = float(r.get("selection_strength") or "")
					p3 = float(r.get("p_level_3") or "")
					n_seeds = int(float(r.get("n_seeds") or "0"))
				except Exception:
					continue
				key = (players, round(k, 10))
				if key not in acc:
					acc[key] = [0.0, 0.0]
				acc[key][0] += float(p3) * float(n_seeds)
				acc[key][1] += float(n_seeds)

	out: dict[tuple[int, float], tuple[float, int]] = {}
	for key, (s_p3, s_n) in acc.items():
		if s_n <= 0:
			continue
		out[key] = (float(s_p3) / float(s_n), int(s_n))
	return out


def monotonicity_violations(
	paths: list[Path],
	*,
	players: int,
	k_center: float,
	band: float,
	tol: float = 1e-12,
) -> tuple[int, int]:
	"""Count adjacent decreases of p3 within [center-band, center+band].

	Returns (violations, points_used).
	"""
	points = _merge_p3_points(paths)
	series: list[tuple[float, float]] = []
	k_lo = float(k_center) - float(band)
	k_hi = float(k_center) + float(band)
	for (p, k), (p3, _n) in points.items():
		if int(p) != int(players):
			continue
		if k < k_lo - 1e-12 or k > k_hi + 1e-12:
			continue
		series.append((float(k), float(p3)))
	series.sort(key=lambda t: t[0])
	viol = 0
	for i in range(1, len(series)):
		prev = series[i - 1][1]
		cur = series[i][1]
		if cur + float(tol) < prev:
			viol += 1
	return viol, len(series)


def _viol_rate(viol: int, points: int) -> float | None:
	adj = int(points) - 1
	if adj <= 0:
		return None
	return float(viol) / float(adj)


def fit_model_A(Ns: list[int], ys: list[float], ws: list[float]) -> tuple[float, float, float]:
	"""k50 = k_inf + c/N. Returns (k_inf, c, sse)."""
	xs = [1.0 / float(n) for n in Ns]
	S = sum(ws)
	Sx = sum(w * x for w, x in zip(ws, xs))
	Sy = sum(w * y for w, y in zip(ws, ys))
	Sxx = sum(w * x * x for w, x in zip(ws, xs))
	Sxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))
	den = S * Sxx - Sx * Sx
	if den == 0.0:
		return float("nan"), float("nan"), float("inf")
	c = (S * Sxy - Sx * Sy) / den
	k_inf = (Sy - c * Sx) / S
	sse = sum(w * (y - (k_inf + c * x)) ** 2 for w, x, y in zip(ws, xs, ys))
	return float(k_inf), float(c), float(sse)


def fit_model_C(Ns: list[int], ys: list[float], ws: list[float]) -> tuple[float, float, float, float]:
	"""k50 = k_inf + c/N + d/N^2. Returns (k_inf, c, d, sse)."""
	x1 = [1.0 / float(n) for n in Ns]
	x2 = [x * x for x in x1]

	# Weighted normal equations for [a,b,c] in y=a + b x1 + c x2
	S00 = sum(ws)
	S01 = sum(w * u for w, u in zip(ws, x1))
	S02 = sum(w * u for w, u in zip(ws, x2))
	S11 = sum(w * u * u for w, u in zip(ws, x1))
	S12 = sum(w * u * v for w, u, v in zip(ws, x1, x2))
	S22 = sum(w * v * v for w, v in zip(ws, x2))
	T0 = sum(w * y for w, y in zip(ws, ys))
	T1 = sum(w * u * y for w, u, y in zip(ws, x1, ys))
	T2 = sum(w * v * y for w, v, y in zip(ws, x2, ys))

	A = [
		[S00, S01, S02],
		[S01, S11, S12],
		[S02, S12, S22],
	]
	b = [T0, T1, T2]

	# Gaussian elimination (3x3)
	for i in range(3):
		# pivot
		pivot = i
		for j in range(i + 1, 3):
			if abs(A[j][i]) > abs(A[pivot][i]):
				pivot = j
		if abs(A[pivot][i]) < 1e-18:
			return float("nan"), float("nan"), float("nan"), float("inf")
		if pivot != i:
			A[i], A[pivot] = A[pivot], A[i]
			b[i], b[pivot] = b[pivot], b[i]
		# eliminate
		inv = 1.0 / A[i][i]
		for k in range(i, 3):
			A[i][k] *= inv
		b[i] *= inv
		for j in range(3):
			if j == i:
				continue
			factor = A[j][i]
			if factor == 0.0:
				continue
			for k in range(i, 3):
				A[j][k] -= factor * A[i][k]
			b[j] -= factor * b[i]

	a, c1, d = b
	sse = sum(w * (y - (a + c1 * u + d * v)) ** 2 for w, u, v, y in zip(ws, x1, x2, ys))
	return float(a), float(c1), float(d), float(sse)


def fit_model_B(
	Ns: list[int],
	ys: list[float],
	ws: list[float],
	*,
	beta_min: float = 0.2,
	beta_max: float = 2.0,
	beta_step: float = 0.01,
) -> tuple[float, float, float, float]:
	"""k50 = k_inf + c/N^beta. Returns (k_inf, c, beta, sse) by grid search."""
	best = (float("nan"), float("nan"), float("nan"), float("inf"))
	beta = beta_min
	while beta <= beta_max + 1e-12:
		xs = [1.0 / (float(n) ** float(beta)) for n in Ns]
		S = sum(ws)
		Sx = sum(w * x for w, x in zip(ws, xs))
		Sy = sum(w * y for w, y in zip(ws, ys))
		Sxx = sum(w * x * x for w, x in zip(ws, xs))
		Sxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))
		den = S * Sxx - Sx * Sx
		if den != 0.0:
			c = (S * Sxy - Sx * Sy) / den
			k_inf = (Sy - c * Sx) / S
			sse = sum(w * (y - (k_inf + c * x)) ** 2 for w, x, y in zip(ws, xs, ys))
			if sse < best[3]:
				best = (float(k_inf), float(c), float(beta), float(sse))
		beta += beta_step
	return best


def aic_bic(*, sse: float, n: int, k: int) -> tuple[float, float]:
	# Use Gaussian RSS-based criteria; treat SSE as RSS proxy.
	# Not perfect under weighting, but useful for relative comparison.
	if n <= 0 or not math.isfinite(sse) or sse <= 0.0:
		return float("nan"), float("nan")
	aic = float(n) * math.log(float(sse) / float(n)) + 2.0 * float(k)
	bic = float(n) * math.log(float(sse) / float(n)) + float(k) * math.log(float(n))
	return float(aic), float(bic)


def main() -> None:
	ap = argparse.ArgumentParser(description="Refinement before/after review + finite-size model comparison")
	ap.add_argument("--before-summary", type=Path, required=True)
	ap.add_argument("--after-summary", type=Path, required=True)
	ap.add_argument("--before-sweeps", type=Path, action="append", default=[])
	ap.add_argument("--after-sweeps", type=Path, action="append", default=[])
	ap.add_argument("--band", type=float, default=0.05)
	ap.add_argument("--out", type=Path, required=True)
	args = ap.parse_args()

	before = read_summary(Path(args.before_summary))
	after = read_summary(Path(args.after_summary))

	players_common = sorted(set(before.keys()) & set(after.keys()))

	lines: list[str] = []
	lines.append("# Refinement before/after review (k50 uncertainty + monotonicity + model comparison)\n")
	lines.append(f"- before: {args.before_summary}")
	lines.append(f"- after: {args.after_summary}")
	lines.append("")

	# 1) Uncertainty widths
	lines.append("## 1) k50 uncertainty widths (bootstrap CI / Bayes CrI)\n")
	lines.append("| N | boot_CI_width_before | boot_CI_width_after | Δboot | bayes_CrI_width_before | bayes_CrI_width_after | Δbayes |")
	lines.append("| --- | --- | --- | --- | --- | --- | --- |")
	for n in players_common:
		b = before[n]
		a = after[n]
		wbb = _ci_width(b.k50_boot_ci_low, b.k50_boot_ci_high)
		waa = _ci_width(a.k50_boot_ci_low, a.k50_boot_ci_high)
		db = (waa - wbb) if (wbb is not None and waa is not None) else None
		wby = _ci_width(b.k50_bayes_ci_low, b.k50_bayes_ci_high)
		way = _ci_width(a.k50_bayes_ci_low, a.k50_bayes_ci_high)
		dy = (way - wby) if (wby is not None and way is not None) else None
		lines.append(
			"| "
			+ " | ".join(
				[
					str(n),
					_fmt(wbb),
					_fmt(waa),
					_fmt(db),
					_fmt(wby),
					_fmt(way),
					_fmt(dy),
				]
			)
			+ " |"
		)
	lines.append("")

	# 2) Monotonicity violations (within band around k50)
	if args.before_sweeps and args.after_sweeps:
		lines.append("## 2) Crossing monotonicity (adjacent decreases of P(L3) within k50±band)\n")
		lines.append("| N | points_before | viol_before | viol_rate_before | points_after | viol_after | viol_rate_after |")
		lines.append("| --- | --- | --- | --- | --- | --- | --- |")
		for n in players_common:
			b = before[n]
			a = after[n]
			vb, pb = monotonicity_violations(list(args.before_sweeps), players=n, k_center=b.k50, band=float(args.band))
			va, pa = monotonicity_violations(list(args.after_sweeps), players=n, k_center=a.k50, band=float(args.band))
			rb = _viol_rate(vb, pb)
			ra = _viol_rate(va, pa)
			lines.append(f"| {n} | {pb} | {vb} | {_fmt(rb, digits=3)} | {pa} | {va} | {_fmt(ra, digits=3)} |")
		lines.append("")

	# 3) Model comparison A/B/C using Bayes mean + Bayes std as weights
	lines.append("## 3) Finite-size model comparison (SDD 4.6.7; weighted by Bayes std)\n")
	Ns: list[int] = []
	ys: list[float] = []
	ws: list[float] = []
	for n in players_common:
		row = after[n]
		if row.k50_censored_low:
			continue
		if row.k50_bayes_mean is None or row.k50_bayes_std is None:
			continue
		sig = float(row.k50_bayes_std)
		if sig <= 0.0 or not math.isfinite(sig):
			continue
		Ns.append(int(n))
		ys.append(float(row.k50_bayes_mean))
		ws.append(1.0 / (sig * sig))

	lines.append(f"Data used (after): n={len(Ns)} (excluded censored or missing Bayes stats)")
	lines.append("")

	if len(Ns) >= 3:
		kA, cA, sseA = fit_model_A(Ns, ys, ws)
		aicA, bicA = aic_bic(sse=sseA, n=len(Ns), k=2)

		kB, cB, betaB, sseB = fit_model_B(Ns, ys, ws)
		aicB, bicB = aic_bic(sse=sseB, n=len(Ns), k=3)

		kC, cC, dC, sseC = fit_model_C(Ns, ys, ws)
		aicC, bicC = aic_bic(sse=sseC, n=len(Ns), k=3)

		lines.append("| model | form | params | SSE_w | AIC | BIC |")
		lines.append("| --- | --- | --- | --- | --- | --- |")
		lines.append(
			"| A | k50 = k_inf + c/N | "
			+ f"k_inf={_fmt(kA)}, c={_fmt(cA)} | {_fmt(sseA)} | {_fmt(aicA)} | {_fmt(bicA)} |"
		)
		lines.append(
			"| B | k50 = k_inf + c/N^β | "
			+ f"k_inf={_fmt(kB)}, c={_fmt(cB)}, β={_fmt(betaB, digits=3)} | {_fmt(sseB)} | {_fmt(aicB)} | {_fmt(bicB)} |"
		)
		lines.append(
			"| C | k50 = k_inf + c/N + d/N^2 | "
			+ f"k_inf={_fmt(kC)}, c={_fmt(cC)}, d={_fmt(dC)} | {_fmt(sseC)} | {_fmt(aicC)} | {_fmt(bicC)} |"
		)
		lines.append("")
		lines.append("Notes:")
		lines.append("- AIC/BIC here use RSS-style formulas on weighted SSE; treat as heuristic ranking.")
		lines.append("- If A/B/C are close, prefer the simpler model unless you have more N or tighter uncertainty.")
	else:
		lines.append("Not enough N points with Bayes stats to fit models A/B/C robustly.")
		lines.append("")

	args.out.parent.mkdir(parents=True, exist_ok=True)
	args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
	print(f"Wrote: {args.out}")


if __name__ == "__main__":
	main()
