"""Finite-size scaling / collapse analysis for rho_curve sweeps.

This script lives in analysis/ (does not import simulation/) and reads the CSV
outputs produced by `python -m simulation.rho_curve`.

Outputs
- A summary CSV with k50/k90/peak/plateau per N.
- Plots:
  - k50 vs 1/N (with a simple linear fit if possible)
  - P(Level3) vs (rho-1) (curve collapse by N)
  - mean_env_gamma vs (rho-1) (envelope criticality proxy)

Dependencies
- Standard library for CSV parsing.
- matplotlib/numpy are treated as optional; plots require matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _extract_in_paths_from_compose(compose_sh: Path) -> list[Path]:
	"""Extract --in '...csv' paths from a compose .sh.

	This is a convenience for reproducible pipelines that generate stable compose
	scripts via `analysis.compose_rho_curve_commands`.
	"""
	text = compose_sh.read_text(encoding="utf-8", errors="replace")
	paths = re.findall(r"--in '([^']+)'", text)
	out: list[Path] = []
	seen: set[str] = set()
	for p in paths:
		pp = str(p).strip()
		if not pp:
			continue
		if pp in seen:
			continue
		seen.add(pp)
		out.append(Path(pp))
	return out


def _sigmoid(x: float) -> float:
	# Numerically stable sigmoid.
	xf = float(x)
	if xf >= 0.0:
		z = math.exp(-xf)
		return 1.0 / (1.0 + z)
	z = math.exp(xf)
	return z / (1.0 + z)


def _log_sigmoid(x: float) -> float:
	# log(sigmoid(x)) stably.
	xf = float(x)
	if xf >= 0.0:
		return -math.log1p(math.exp(-xf))
	return xf - math.log1p(math.exp(xf))


def _log1m_sigmoid(x: float) -> float:
	# log(1 - sigmoid(x)) stably.
	xf = float(x)
	if xf >= 0.0:
		return -xf - math.log1p(math.exp(-xf))
	return -math.log1p(math.exp(xf))


def _inv_2x2(a11: float, a12: float, a21: float, a22: float) -> tuple[float, float, float, float] | None:
	det = float(a11) * float(a22) - float(a12) * float(a21)
	if not math.isfinite(det) or abs(det) <= 1e-18:
		return None
	inv_det = 1.0 / det
	return (
		float(a22) * inv_det,
		-float(a12) * inv_det,
		-float(a21) * inv_det,
		float(a11) * inv_det,
	)


def _cholesky_2x2(c11: float, c12: float, c21: float, c22: float) -> tuple[float, float, float] | None:
	"""Return (l11, l21, l22) such that L=[[l11,0],[l21,l22]] and C=L L^T."""
	if not (math.isfinite(c11) and math.isfinite(c12) and math.isfinite(c21) and math.isfinite(c22)):
		return None
	if c11 <= 0.0:
		return None
	l11 = math.sqrt(c11)
	l21 = c21 / l11
	rem = c22 - l21 * l21
	if rem <= 0.0:
		return None
	l22 = math.sqrt(rem)
	return float(l11), float(l21), float(l22)


def _log_posterior_binomial_logit(
	ks: Sequence[float],
	ys: Sequence[int],
	ns: Sequence[int],
	*,
	alpha: float,
	beta: float,
	prior_sigma: float | None,
) -> float:
	"""Log posterior up to additive constant."""
	ll = 0.0
	for k, y, n in zip(ks, ys, ns):
		if n <= 0:
			continue
		x = float(alpha) + float(beta) * float(k)
		ll += float(y) * _log_sigmoid(x) + float(n - y) * _log1m_sigmoid(x)
	if prior_sigma is not None:
		s = float(prior_sigma)
		if s > 0.0 and math.isfinite(s):
			inv = 1.0 / (s * s)
			ll += -0.5 * inv * (float(alpha) * float(alpha) + float(beta) * float(beta))
	return float(ll)


def _fit_binomial_logit_laplace(
	ks: Sequence[float],
	ys: Sequence[int],
	ns: Sequence[int],
	*,
	prior_sigma: float | None = 10.0,
	max_iter: int = 50,
	tol: float = 1e-8,
) -> tuple[tuple[float, float], tuple[float, float, float, float]]:
	"""Fit MAP for Binomial-logit model and return (mean, cov).

	Model:
	  y_i ~ Binomial(n_i, sigmoid(alpha + beta*k_i))
	Prior (optional):
	  alpha, beta ~ Normal(0, prior_sigma^2)

	Returns:
	  mean=(alpha, beta)
	  cov=(c11,c12,c21,c22) approx posterior covariance via Laplace (inv(-H)).
	"""
	if len(ks) != len(ys) or len(ks) != len(ns):
		raise ValueError("ks/ys/ns length mismatch")
	if len(ks) < 3:
		raise ValueError("Need at least 3 k-points for Bayesian fit")

	# Guard against degenerate data (complete separation): all 0 or all n.
	tot_y = sum(int(y) for y in ys)
	tot_n = sum(int(n) for n in ns)
	if tot_n <= 0:
		raise ValueError("No trials (n_seeds) available")
	if tot_y == 0 or tot_y == tot_n:
		raise ValueError("Degenerate counts (all-successes or all-failures)")

	# Initialize near the empirical mean; start with beta>0.
	p0 = (float(tot_y) + 0.5) / (float(tot_n) + 1.0)
	p0 = min(1.0 - 1e-6, max(1e-6, p0))
	alpha = math.log(p0 / (1.0 - p0))
	beta = 1.0

	prior = prior_sigma
	inv_prior_var = 0.0
	if prior is not None:
		s = float(prior)
		if not math.isfinite(s) or s <= 0.0:
			raise ValueError("prior_sigma must be finite and >0")
		inv_prior_var = 1.0 / (s * s)

	cur_lp = _log_posterior_binomial_logit(ks, ys, ns, alpha=alpha, beta=beta, prior_sigma=prior)
	for _ in range(int(max_iter)):
		g0 = 0.0
		g1 = 0.0
		h00 = 0.0
		h01 = 0.0
		h11 = 0.0
		for k, y, n in zip(ks, ys, ns):
			if n <= 0:
				continue
			x = float(alpha) + float(beta) * float(k)
			p = _sigmoid(x)
			w = float(n) * float(p) * float(1.0 - p)
			res = float(y) - float(n) * float(p)
			g0 += res
			g1 += res * float(k)
			h00 += -w
			h01 += -w * float(k)
			h11 += -w * float(k) * float(k)
		if inv_prior_var > 0.0:
			g0 += -float(alpha) * inv_prior_var
			g1 += -float(beta) * inv_prior_var
			h00 += -inv_prior_var
			h11 += -inv_prior_var

		if max(abs(g0), abs(g1)) < float(tol):
			break

		invH = _inv_2x2(h00, h01, h01, h11)
		if invH is None:
			raise ValueError("Singular Hessian in Bayesian fit")
		i00, i01, i10, i11 = invH
		d0 = i00 * g0 + i01 * g1
		d1 = i10 * g0 + i11 * g1

		# Newton step: theta_new = theta - inv(H) * g
		step = 1.0
		accepted = False
		for _ls in range(25):
			a2 = float(alpha) - step * float(d0)
			b2 = float(beta) - step * float(d1)
			lp2 = _log_posterior_binomial_logit(ks, ys, ns, alpha=a2, beta=b2, prior_sigma=prior)
			if math.isfinite(lp2) and lp2 >= cur_lp:
				alpha, beta, cur_lp = a2, b2, lp2
				accepted = True
				break
			step *= 0.5
		if not accepted:
			# If even tiny steps don't improve, stop.
			break

	# Final Hessian for covariance.
	g0 = 0.0
	g1 = 0.0
	h00 = 0.0
	h01 = 0.0
	h11 = 0.0
	for k, y, n in zip(ks, ys, ns):
		if n <= 0:
			continue
		x = float(alpha) + float(beta) * float(k)
		p = _sigmoid(x)
		w = float(n) * float(p) * float(1.0 - p)
		res = float(y) - float(n) * float(p)
		g0 += res
		g1 += res * float(k)
		h00 += -w
		h01 += -w * float(k)
		h11 += -w * float(k) * float(k)
	if inv_prior_var > 0.0:
		g0 += -float(alpha) * inv_prior_var
		g1 += -float(beta) * inv_prior_var
		h00 += -inv_prior_var
		h11 += -inv_prior_var

	# Covariance = inv(-H)
	negH00 = -h00
	negH01 = -h01
	negH11 = -h11
	inv_negH = _inv_2x2(negH00, negH01, negH01, negH11)
	if inv_negH is None:
		raise ValueError("Non-invertible posterior curvature")
	c11, c12, c21, c22 = inv_negH
	return (float(alpha), float(beta)), (float(c11), float(c12), float(c21), float(c22))


def bayesian_k50_laplace(
	seq: Sequence[RhoCurvePoint],
	*,
	prior_sigma: float = 10.0,
	draws: int = 2000,
	seed: int | None = None,
	assume_increasing: bool = True,
) -> tuple[float | None, float | None, float | None, float | None, int]:
	"""Approx Bayesian k50 from Binomial-logit Laplace posterior.

	Uses (n_seeds, p_level_3) at each k as Binomial counts.
	Returns (mean, std, ci_low, ci_high, n_eff_draws).
	"""
	if not seq:
		return None, None, None, None, 0

	ks: list[float] = []
	ys: list[int] = []
	ns: list[int] = []
	for p in seq:
		n = int(p.n_seeds)
		if n <= 0:
			continue
		k = float(p.k)
		phat = float(p.p_level_3)
		phat = min(1.0, max(0.0, phat))
		y = int(round(phat * float(n)))
		y = max(0, min(n, y))
		ks.append(k)
		ys.append(y)
		ns.append(n)
	if len(ks) < 3:
		return None, None, None, None, 0

	(mean_a, mean_b), cov = _fit_binomial_logit_laplace(
		ks,
		ys,
		ns,
		prior_sigma=float(prior_sigma),
	)
	c11, c12, c21, c22 = cov
	chol = _cholesky_2x2(c11, c12, c21, c22)
	if chol is None:
		raise ValueError("Posterior covariance not PSD")
	l11, l21, l22 = chol

	rng = Random(seed)
	K: list[float] = []
	R = int(draws)
	if R <= 0:
		return None, None, None, None, 0
	for _ in range(R):
		z1 = rng.gauss(0.0, 1.0)
		z2 = rng.gauss(0.0, 1.0)
		a = float(mean_a) + l11 * z1
		b = float(mean_b) + l21 * z1 + l22 * z2
		if bool(assume_increasing) and b <= 0.0:
			continue
		if not math.isfinite(b) or abs(b) <= 1e-12:
			continue
		k50 = -float(a) / float(b)
		if math.isfinite(k50):
			K.append(float(k50))
	if not K:
		return None, None, None, None, 0
	mu = sum(K) / float(len(K))
	var = sum((x - mu) ** 2 for x in K) / float(len(K))
	sd = math.sqrt(max(0.0, var))
	lo = _quantile(K, 0.025)
	hi = _quantile(K, 0.975)
	return float(mu), float(sd), float(lo), float(hi), int(len(K))


def _expand_input_paths(inputs: Sequence[str]) -> List[Path]:
	"""Expand file globs in CLI inputs.

	This allows passing quoted patterns like "outputs/.../N100*.csv". If a given
	item exists as a file, it's used as-is; otherwise, if it contains glob
	metacharacters, it's expanded.
	"""
	paths: List[Path] = []
	for raw in inputs:
		p = Path(raw)
		if p.exists():
			paths.append(p)
			continue
		if any(ch in raw for ch in "*?["):
			matches = sorted(glob.glob(raw))
			if matches:
				paths.extend(Path(m) for m in matches)
				continue
		raise FileNotFoundError(f"Input not found (and no glob matches): {raw}")
	return paths


def _first_nonempty_from_sweeps(
	paths: Sequence[Path],
	*,
	keys: Sequence[str],
) -> dict[str, str]:
	"""Extract the first non-empty value for each key across sweep CSV inputs."""
	out: dict[str, str] = {}
	remaining = set(keys)
	if not paths or not remaining:
		return out
	for path in paths:
		try:
			with path.open(newline="") as f:
				r = csv.DictReader(f)
				for row in r:
					for k in list(remaining):
						v = row.get(str(k))
						if v is None:
							continue
						s = str(v).strip()
						if s == "":
							continue
						out[str(k)] = s
						remaining.remove(k)
					if not remaining:
						return out
		except Exception:
			continue
	return out


@dataclass(frozen=True)
class RhoCurvePoint:
	players: int
	k: float
	rho_minus_1: float
	p_level_3: float
	mean_env_gamma: float
	n_seeds: int


def _parse_targets(spec: str) -> List[float]:
	out: List[float] = []
	for part in spec.split(","):
		part = part.strip()
		if not part:
			continue
		out.append(float(part))
	if not out:
		raise ValueError("--targets is empty")
	return out


def load_rho_curve_csv(path: str | Path) -> List[RhoCurvePoint]:
	path = Path(path)
	with path.open(newline="") as f:
		reader = csv.DictReader(f)
		rows = list(reader)

	points: List[RhoCurvePoint] = []
	for r in rows:
		points.append(
			RhoCurvePoint(
				players=int(float(r["players"])),
				k=float(r["selection_strength"]),
				rho_minus_1=float(r["rho_minus_1"]),
				p_level_3=float(r["p_level_3"]),
				mean_env_gamma=float(r.get("mean_env_gamma", "nan")),
				n_seeds=int(float(r.get("n_seeds", "0"))),
			)
		)
	return points


def merge_points(datasets: Sequence[Sequence[RhoCurvePoint]]) -> List[RhoCurvePoint]:
	"""Merge multiple sweeps.

	Deduplicate by (players, k). If duplicates exist, keep the row with larger
	n_seeds, else keep the last seen.
	"""

	best: Dict[Tuple[int, float], RhoCurvePoint] = {}
	for ds in datasets:
		for p in ds:
			key = (p.players, p.k)
			cur = best.get(key)
			if cur is None or p.n_seeds > cur.n_seeds:
				best[key] = p
			else:
				# keep current
				pass
	return list(best.values())


def group_by_players(points: Sequence[RhoCurvePoint]) -> Dict[int, List[RhoCurvePoint]]:
	out: Dict[int, List[RhoCurvePoint]] = {}
	for p in points:
		out.setdefault(p.players, []).append(p)
	for n, seq in out.items():
		seq.sort(key=lambda x: x.k)
	return out


def _interp(a: float, b: float, t: float) -> float:
	return a + t * (b - a)


@dataclass(frozen=True)
class Crossing:
	k: float
	rho_minus_1: float
	p_level_3: float
	mean_env_gamma: float


def find_first_crossing(seq: Sequence[RhoCurvePoint], *, target: float) -> Optional[Crossing]:
	"""Find first k where p_level_3 crosses target, using linear interpolation."""
	if not seq:
		return None

	# If already above target at the first point, treat it as a crossing.
	if seq[0].p_level_3 >= target:
		p0 = seq[0]
		return Crossing(k=p0.k, rho_minus_1=p0.rho_minus_1, p_level_3=p0.p_level_3, mean_env_gamma=p0.mean_env_gamma)

	for i in range(len(seq) - 1):
		p0 = seq[i]
		p1 = seq[i + 1]
		y0 = p0.p_level_3
		y1 = p1.p_level_3
		if (y0 < target <= y1) or (y0 > target >= y1):
			if y1 == y0:
				# flat segment; choose the right endpoint
				return Crossing(k=p1.k, rho_minus_1=p1.rho_minus_1, p_level_3=p1.p_level_3, mean_env_gamma=p1.mean_env_gamma)
			t = (target - y0) / (y1 - y0)
			return Crossing(
				k=_interp(p0.k, p1.k, t),
				rho_minus_1=_interp(p0.rho_minus_1, p1.rho_minus_1, t),
				p_level_3=target,
				mean_env_gamma=_interp(p0.mean_env_gamma, p1.mean_env_gamma, t),
			)
	return None


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
	n = len(xs)
	if n < 2:
		return float("nan")
	mx = sum(xs) / n
	my = sum(ys) / n
	vx = sum((x - mx) ** 2 for x in xs)
	vy = sum((y - my) ** 2 for y in ys)
	if vx == 0.0 or vy == 0.0:
		return float("nan")
	cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
	return cov / math.sqrt(vx * vy)


@dataclass(frozen=True)
class SummaryRow:
	players: int
	k_min: float
	p3_at_k_min: float
	k50_boot_mean: float | None
	k50_boot_std: float | None
	k50_boot_ci_low: float | None
	k50_boot_ci_high: float | None
	k50_bayes_mean: float | None
	k50_bayes_std: float | None
	k50_bayes_ci_low: float | None
	k50_bayes_ci_high: float | None
	k50_bayes_n_eff: int | None
	bayes_method: str | None
	bayes_prior_sigma: float | None
	bayes_draws: int | None
	bayes_seed: int | None
	k50: float | None
	rho_m1_50: float | None
	k50_censored_low: bool
	k90: float | None
	rho_m1_90: float | None
	p3_max: float
	k_at_max: float
	rho_m1_at_max: float
	plateau_k_start: float
	plateau_k_end: float
	corr_rho_p3: float
	k_first_positive: float | None
	rho_m1_first_positive: float | None


def summarize(seq: Sequence[RhoCurvePoint]) -> SummaryRow:
	assert seq
	k_min = float(seq[0].k)
	p3_at_k_min = float(seq[0].p_level_3)
	xs = [p.rho_minus_1 for p in seq]
	ys = [p.p_level_3 for p in seq]
	corr = pearson(xs, ys)

	peak = max(seq, key=lambda p: p.p_level_3)
	pmax = float(peak.p_level_3)
	# plateau = contiguous region at the end where p == pmax (within tol)
	tol = 1e-12
	plateau = [p for p in seq if abs(p.p_level_3 - pmax) <= tol]
	plateau.sort(key=lambda p: p.k)
	plateau_k_start = float(plateau[0].k)
	plateau_k_end = float(plateau[-1].k)

	first_pos = next((p for p in seq if p.p_level_3 > 0.0), None)
	c50 = find_first_crossing(seq, target=0.5)
	c90 = find_first_crossing(seq, target=0.9)
	k50_censored_low = bool(seq[0].p_level_3 >= 0.5)

	return SummaryRow(
		players=int(seq[0].players),
		k_min=k_min,
		p3_at_k_min=p3_at_k_min,
		k50_boot_mean=None,
		k50_boot_std=None,
		k50_boot_ci_low=None,
		k50_boot_ci_high=None,
		k50_bayes_mean=None,
		k50_bayes_std=None,
		k50_bayes_ci_low=None,
		k50_bayes_ci_high=None,
		k50_bayes_n_eff=None,
		bayes_method=None,
		bayes_prior_sigma=None,
		bayes_draws=None,
		bayes_seed=None,
		k50=(float(c50.k) if c50 is not None else None),
		rho_m1_50=(float(c50.rho_minus_1) if c50 is not None else None),
		k50_censored_low=bool(k50_censored_low),
		k90=(float(c90.k) if c90 is not None else None),
		rho_m1_90=(float(c90.rho_minus_1) if c90 is not None else None),
		p3_max=pmax,
		k_at_max=float(peak.k),
		rho_m1_at_max=float(peak.rho_minus_1),
		plateau_k_start=plateau_k_start,
		plateau_k_end=plateau_k_end,
		corr_rho_p3=float(corr),
		k_first_positive=(float(first_pos.k) if first_pos is not None else None),
		rho_m1_first_positive=(float(first_pos.rho_minus_1) if first_pos is not None else None),
	)


def _binomial(n: int, p: float, *, rng: Random) -> int:
	"""Simple binomial sampler for small n using stdlib RNG."""
	if n <= 0:
		return 0
	pp = float(p)
	if pp <= 0.0:
		return 0
	if pp >= 1.0:
		return int(n)
	x = 0
	for _ in range(int(n)):
		if rng.random() < pp:
			x += 1
	return int(x)


def _quantile(xs: Sequence[float], q: float) -> float:
	if not xs:
		return float("nan")
	qq = float(q)
	if qq <= 0.0:
		return float(min(xs))
	if qq >= 1.0:
		return float(max(xs))
	s = sorted(float(x) for x in xs)
	pos = qq * (len(s) - 1)
	lo = int(math.floor(pos))
	hi = int(math.ceil(pos))
	if lo == hi:
		return float(s[lo])
	t = pos - lo
	return float(s[lo] + t * (s[hi] - s[lo]))


def bootstrap_k50(
	seq: Sequence[RhoCurvePoint],
	*,
	resamples: int,
	seed: int | None,
) -> tuple[float | None, float | None, float | None, float | None]:
	"""Bootstrap k50 uncertainty from (p_level_3, n_seeds) at each k.

	We treat each point as an independent Binomial(n_seeds, p_level_3) draw.
	This ignores within-seed correlations across k, but provides a cheap,
	low-dependency baseline uncertainty estimate.
	"""
	R = int(resamples)
	if R <= 0:
		return None, None, None, None
	rng = Random(seed)
	ks = [float(p.k) for p in seq]
	rhos = [float(p.rho_minus_1) for p in seq]
	ps = [float(p.p_level_3) for p in seq]
	ns = [int(p.n_seeds) for p in seq]
	boot: list[float] = []
	for _ in range(R):
		points: list[RhoCurvePoint] = []
		for k, rho, p, n in zip(ks, rhos, ps, ns):
			# Resample p via a Binomial draw.
			s = _binomial(int(n), float(p), rng=rng)
			p_hat = (float(s) / float(n)) if n > 0 else 0.0
			points.append(
				RhoCurvePoint(
					players=int(seq[0].players),
					k=float(k),
					rho_minus_1=float(rho),
					p_level_3=float(p_hat),
					mean_env_gamma=float("nan"),
					n_seeds=int(n),
				)
			)
		c = find_first_crossing(points, target=0.5)
		if c is None:
			continue
		boot.append(float(c.k))
	if not boot:
		return None, None, None, None
	mean = sum(boot) / float(len(boot))
	var = sum((x - mean) ** 2 for x in boot) / float(len(boot))
	std = math.sqrt(var)
	ci_lo = _quantile(boot, 0.025)
	ci_hi = _quantile(boot, 0.975)
	return float(mean), float(std), float(ci_lo), float(ci_hi)


def write_summary_csv(
	path: Path,
	rows: Sequence[SummaryRow],
	*,
	stage2_settings: dict[str, str] | None = None,
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	stage2_settings = dict(stage2_settings or {})
	with path.open("w", newline="") as f:
		w = csv.DictWriter(
			f,
			fieldnames=[
				"stage2_method",
				"stage2_prefilter",
				"power_ratio_kappa",
				"permutation_alpha",
				"permutation_resamples",
				"permutation_seed",
				"players",
				"k_min",
				"p3_at_k_min",
				"k50_boot_mean",
				"k50_boot_std",
				"k50_boot_ci_low",
				"k50_boot_ci_high",
				"bayes_method",
				"bayes_prior_sigma",
				"bayes_draws",
				"bayes_seed",
				"k50_bayes_n_eff",
				"k50_bayes_mean",
				"k50_bayes_std",
				"k50_bayes_ci_low",
				"k50_bayes_ci_high",
				"k_first_positive",
				"rho_minus_1_first_positive",
				"k50",
				"rho_minus_1_50",
				"k50_censored_low",
				"k90",
				"rho_minus_1_90",
				"p3_max",
				"k_at_max",
				"rho_minus_1_at_max",
				"plateau_k_start",
				"plateau_k_end",
				"corr_rho_minus_1_p_level_3",
			],
		)
		w.writeheader()
		for r in rows:
			w.writerow(
				{
					"stage2_method": stage2_settings.get("stage2_method", ""),
					"stage2_prefilter": stage2_settings.get("stage2_prefilter", ""),
					"power_ratio_kappa": stage2_settings.get("power_ratio_kappa", ""),
					"permutation_alpha": stage2_settings.get("permutation_alpha", ""),
					"permutation_resamples": stage2_settings.get("permutation_resamples", ""),
					"permutation_seed": stage2_settings.get("permutation_seed", ""),
					"players": r.players,
					"k_min": r.k_min,
					"p3_at_k_min": r.p3_at_k_min,
					"k50_boot_mean": r.k50_boot_mean,
					"k50_boot_std": r.k50_boot_std,
					"k50_boot_ci_low": r.k50_boot_ci_low,
					"k50_boot_ci_high": r.k50_boot_ci_high,
					"bayes_method": (r.bayes_method or ""),
					"bayes_prior_sigma": r.bayes_prior_sigma,
					"bayes_draws": r.bayes_draws,
					"bayes_seed": r.bayes_seed,
					"k50_bayes_n_eff": r.k50_bayes_n_eff,
					"k50_bayes_mean": r.k50_bayes_mean,
					"k50_bayes_std": r.k50_bayes_std,
					"k50_bayes_ci_low": r.k50_bayes_ci_low,
					"k50_bayes_ci_high": r.k50_bayes_ci_high,
					"k_first_positive": r.k_first_positive,
					"rho_minus_1_first_positive": r.rho_m1_first_positive,
					"k50": r.k50,
					"rho_minus_1_50": r.rho_m1_50,
					"k50_censored_low": r.k50_censored_low,
					"k90": r.k90,
					"rho_minus_1_90": r.rho_m1_90,
					"p3_max": r.p3_max,
					"k_at_max": r.k_at_max,
					"rho_minus_1_at_max": r.rho_m1_at_max,
					"plateau_k_start": r.plateau_k_start,
					"plateau_k_end": r.plateau_k_end,
					"corr_rho_minus_1_p_level_3": r.corr_rho_p3,
				}
			)


def fit_k50_vs_invN(rows: Sequence[SummaryRow]) -> Dict[str, float] | None:
	pts = [
		(1.0 / r.players, r.k50)
		for r in rows
		if (r.k50 is not None and r.players > 0 and (not bool(getattr(r, "k50_censored_low", False))))
	]
	if len(pts) < 2:
		return None

	try:
		import numpy as np  # optional
		x = np.array([p[0] for p in pts], dtype=float)
		y = np.array([p[1] for p in pts], dtype=float)
		A = np.vstack([np.ones_like(x), x]).T
		beta, *_ = np.linalg.lstsq(A, y, rcond=None)
		k_inf = float(beta[0])
		c = float(beta[1])
		# R^2
		yhat = A @ beta
		ss_res = float(((y - yhat) ** 2).sum())
		ss_tot = float(((y - y.mean()) ** 2).sum())
		r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
		return {"k_inf": k_inf, "c": c, "r2": r2}
	except Exception:
		return None


def plot_k50_scaling(
	rows: Sequence[SummaryRow],
	*,
	fit: Dict[str, float] | None,
	outpath: Path,
	title: str = "Finite-size scaling: k50 vs 1/N",
) -> None:
	import matplotlib.pyplot as plt  # optional dependency

	# Split points: proper k50 estimates vs left-censored (only know k50 <= k_min).
	x_fit: list[float] = []
	y_fit: list[float] = []
	x_cens: list[float] = []
	y_cens: list[float] = []

	for r in rows:
		if r.players <= 0:
			continue
		x = 1.0 / r.players
		if r.k50 is None:
			continue
		if bool(getattr(r, "k50_censored_low", False)):
			x_cens.append(x)
			y_cens.append(float(r.k_min))
		else:
			x_fit.append(x)
			y_fit.append(float(r.k50))

	plt.figure(figsize=(6, 4))
	if x_fit:
		plt.scatter(x_fit, y_fit, color="tab:blue", label="k50 (interpolated)")
	if x_cens:
		plt.scatter(x_cens, y_cens, color="tab:red", marker="x", s=55, label="left-censored: k50 ≤ k_min")

	# Annotations
	for r in rows:
		if r.players <= 0:
			continue
		x = 1.0 / r.players
		if r.k50 is None:
			continue
		if bool(getattr(r, "k50_censored_low", False)):
			plt.annotate(
				f"N={r.players}\n≤{r.k_min:g}",
				(x, float(r.k_min)),
				textcoords="offset points",
				xytext=(5, 5),
				fontsize=8,
			)
		else:
			plt.annotate(f"N={r.players}", (x, float(r.k50)), textcoords="offset points", xytext=(5, 5), fontsize=8)

	if fit is not None and x_fit:
		k_inf = fit["k_inf"]
		c = fit["c"]
		xmin = min(x_fit)
		xmax = max(x_fit)
		x_line = [0.0, xmax * 1.05]
		y_line = [k_inf + c * x for x in x_line]
		plt.plot(x_line, y_line, color="tab:orange", label=f"fit: k50 = {k_inf:.3g} + {c:.3g}/N (R2={fit.get('r2', float('nan')):.3f})")
		plt.legend()
	else:
		plt.legend()

	plt.xlabel("1/N")
	plt.ylabel("k50 (P(Level3)=0.5)")
	plt.title(title)
	plt.grid(True, alpha=0.3)
	outpath.parent.mkdir(parents=True, exist_ok=True)
	plt.tight_layout()
	plt.savefig(outpath, dpi=160)
	plt.close()


def plot_collapse(
	byN: Dict[int, Sequence[RhoCurvePoint]],
	*,
	outpath: Path,
	title: str = "Order parameter: P(Level3) vs (rho-1)",
) -> None:
	import matplotlib.pyplot as plt  # optional dependency

	plt.figure(figsize=(6, 4))
	for n in sorted(byN.keys()):
		seq = sorted(byN[n], key=lambda p: p.rho_minus_1)
		x = [p.rho_minus_1 for p in seq]
		y = [p.p_level_3 for p in seq]
		plt.plot(x, y, marker="o", markersize=3, linewidth=1, label=f"N={n}")

	plt.axvline(0.0, color="black", linewidth=1, alpha=0.4)
	plt.xlabel("rho - 1")
	plt.ylabel("P(Level3)")
	plt.title(title)
	plt.legend()
	plt.grid(True, alpha=0.3)
	outpath.parent.mkdir(parents=True, exist_ok=True)
	plt.tight_layout()
	plt.savefig(outpath, dpi=160)
	plt.close()


def plot_gamma_vs_rho(
	byN: Dict[int, Sequence[RhoCurvePoint]],
	*,
	outpath: Path,
	title: str = "Envelope proxy: mean_env_gamma vs (rho-1)",
) -> None:
	import matplotlib.pyplot as plt  # optional dependency

	plt.figure(figsize=(6, 4))
	for n in sorted(byN.keys()):
		seq = sorted(byN[n], key=lambda p: p.rho_minus_1)
		x = [p.rho_minus_1 for p in seq]
		y = [p.mean_env_gamma for p in seq]
		plt.plot(x, y, marker="o", markersize=3, linewidth=1, label=f"N={n}")

	plt.axvline(0.0, color="black", linewidth=1, alpha=0.4)
	plt.axhline(0.0, color="black", linewidth=1, alpha=0.4)
	plt.xlabel("rho - 1")
	plt.ylabel("mean_env_gamma")
	plt.title(title)
	plt.legend()
	plt.grid(True, alpha=0.3)
	outpath.parent.mkdir(parents=True, exist_ok=True)
	plt.tight_layout()
	plt.savefig(outpath, dpi=160)
	plt.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
	p = argparse.ArgumentParser(description="Finite-size scaling analysis for rho_curve sweeps")
	p.add_argument(
		"--in",
		dest="inputs",
		action="append",
		required=False,
		default=[],
		help="Input rho_curve CSV path (repeatable)",
	)
	p.add_argument(
		"--compose",
		type=str,
		default=None,
		help="Optional compose .sh (from analysis.compose_rho_curve_commands); extracts repeated --in paths.",
	)
	p.add_argument("--outdir", type=str, default="outputs/analysis/rho_curve", help="Output directory")
	p.add_argument("--prefix", type=str, default="rho_curve", help="Output filename prefix")
	p.add_argument("--targets", type=str, default="0.5,0.9", help="Crossing targets for P(Level3), comma-separated")
	p.add_argument(
		"--bootstrap-resamples",
		type=int,
		default=0,
		help="If >0, compute a bootstrap uncertainty estimate for k50 from (p_level_3, n_seeds) at each k.",
	)
	p.add_argument(
		"--bootstrap-seed",
		type=int,
		default=None,
		help="RNG seed for bootstrap (for reproducibility).",
	)
	p.add_argument(
		"--bayes-method",
		choices=["none", "laplace"],
		default="none",
		help="Optional Bayesian uncertainty estimate for k50. 'laplace' uses a Binomial-logit Laplace approximation (stdlib-only).",
	)
	p.add_argument(
		"--bayes-draws",
		type=int,
		default=2000,
		help="Posterior draws for Bayesian k50 (Laplace sampling).",
	)
	p.add_argument(
		"--bayes-seed",
		type=int,
		default=None,
		help="RNG seed for Bayesian sampling (reproducibility).",
	)
	p.add_argument(
		"--bayes-prior-sigma",
		type=float,
		default=10.0,
		help="Weak Normal(0, sigma^2) prior scale for (alpha,beta) in Bayesian k50 (Laplace).",
	)
	p.add_argument(
		"--bayes-assume-increasing",
		type=int,
		choices=[0, 1],
		default=1,
		help="If 1, discard posterior draws with beta<=0 (enforces monotone increasing P(L3) with k).",
	)
	args = p.parse_args(list(argv) if argv is not None else None)

	in_paths: list[Path] = []
	if args.compose:
		compose_path = Path(str(args.compose))
		if not compose_path.exists():
			raise SystemExit(f"compose not found: {compose_path}")
		in_paths.extend(_extract_in_paths_from_compose(compose_path))
	in_paths.extend(_expand_input_paths(list(args.inputs or [])))
	if not in_paths:
		raise SystemExit("需要提供 --compose 或至少一個 --in")
	stage2_settings = _first_nonempty_from_sweeps(
		in_paths,
		keys=(
			"stage2_method",
			"stage2_prefilter",
			"power_ratio_kappa",
			"permutation_alpha",
			"permutation_resamples",
			"permutation_seed",
		),
	)
	datasets = [load_rho_curve_csv(path) for path in in_paths]
	points = merge_points(datasets)
	byN = group_by_players(points)

	summaries = [summarize(seq) for _, seq in sorted(byN.items(), key=lambda kv: kv[0])]
	# Optional bootstrap CI for k50 per N.
	R = int(args.bootstrap_resamples)
	updated: list[SummaryRow] = []
	bayes_method = str(args.bayes_method)
	for r in summaries:
		seq = byN.get(int(r.players), [])
		m = sd = lo = hi = None
		if R > 0:
			m, sd, lo, hi = bootstrap_k50(seq, resamples=R, seed=args.bootstrap_seed)

		bm = bsd = blo = bhi = None
		n_eff: int | None = None
		b_method_out: str | None = None
		prior_out: float | None = None
		draws_out: int | None = None
		seed_out: int | None = None
		if bayes_method != "none":
			try:
				if bayes_method != "laplace":
					raise ValueError(f"Unknown bayes method: {bayes_method!r}")
				bm, bsd, blo, bhi, n_eff = bayesian_k50_laplace(
					seq,
					prior_sigma=float(args.bayes_prior_sigma),
					draws=int(args.bayes_draws),
					seed=(int(args.bayes_seed) if args.bayes_seed is not None else None),
					assume_increasing=bool(int(args.bayes_assume_increasing) == 1),
				)
				b_method_out = "laplace"
				prior_out = float(args.bayes_prior_sigma)
				draws_out = int(args.bayes_draws)
				seed_out = (int(args.bayes_seed) if args.bayes_seed is not None else None)
			except Exception:
				# Keep summary usable even if Bayesian fit fails for a particular N.
				bm = bsd = blo = bhi = None
				n_eff = None
				b_method_out = "laplace"
				prior_out = float(args.bayes_prior_sigma)
				draws_out = int(args.bayes_draws)
				seed_out = (int(args.bayes_seed) if args.bayes_seed is not None else None)

		updated.append(
			SummaryRow(
				players=r.players,
				k_min=r.k_min,
				p3_at_k_min=r.p3_at_k_min,
				k50_boot_mean=m,
				k50_boot_std=sd,
				k50_boot_ci_low=lo,
				k50_boot_ci_high=hi,
				k50_bayes_mean=bm,
				k50_bayes_std=bsd,
				k50_bayes_ci_low=blo,
				k50_bayes_ci_high=bhi,
				k50_bayes_n_eff=n_eff,
				bayes_method=b_method_out,
				bayes_prior_sigma=prior_out,
				bayes_draws=draws_out,
				bayes_seed=seed_out,
				k50=r.k50,
				rho_m1_50=r.rho_m1_50,
				k50_censored_low=r.k50_censored_low,
				k90=r.k90,
				rho_m1_90=r.rho_m1_90,
				p3_max=r.p3_max,
				k_at_max=r.k_at_max,
				rho_m1_at_max=r.rho_m1_at_max,
				plateau_k_start=r.plateau_k_start,
				plateau_k_end=r.plateau_k_end,
				corr_rho_p3=r.corr_rho_p3,
				k_first_positive=r.k_first_positive,
				rho_m1_first_positive=r.rho_m1_first_positive,
			)
		)
	summaries = updated
	outdir = Path(args.outdir)
	outdir.mkdir(parents=True, exist_ok=True)

	summary_path = outdir / f"{args.prefix}_summary.csv"
	write_summary_csv(summary_path, summaries, stage2_settings=stage2_settings)

	# Optional provenance JSON for Bayesian k50.
	if str(args.bayes_method) != "none":
		bayes_fit_path = outdir / f"{args.prefix}_bayes_fit.json"
		bayes_obj: dict = {
			"bayes_method": str(args.bayes_method),
			"bayes_prior_sigma": float(args.bayes_prior_sigma),
			"bayes_draws": int(args.bayes_draws),
			"bayes_seed": (int(args.bayes_seed) if args.bayes_seed is not None else None),
			"bayes_assume_increasing": bool(int(args.bayes_assume_increasing) == 1),
			"per_N": [],
		}
		for r in summaries:
			bayes_obj["per_N"].append(
				{
					"players": int(r.players),
					"k50_bayes_mean": (float(r.k50_bayes_mean) if r.k50_bayes_mean is not None else None),
					"k50_bayes_std": (float(r.k50_bayes_std) if r.k50_bayes_std is not None else None),
					"k50_bayes_ci_low": (float(r.k50_bayes_ci_low) if r.k50_bayes_ci_low is not None else None),
					"k50_bayes_ci_high": (float(r.k50_bayes_ci_high) if r.k50_bayes_ci_high is not None else None),
					"k50_bayes_n_eff": (int(r.k50_bayes_n_eff) if r.k50_bayes_n_eff is not None else None),
					"k50": (float(r.k50) if r.k50 is not None else None),
					"k50_censored_low": bool(r.k50_censored_low),
				}
			)
		bayes_fit_path.write_text(json.dumps(bayes_obj, indent=2, sort_keys=True) + "\n")

	fit = fit_k50_vs_invN(summaries)
	if fit is not None:
		(outdir / f"{args.prefix}_k50_fit.json").write_text(json.dumps(fit, indent=2) + "\n")

	# Plots (matplotlib optional)
	try:
		plot_k50_scaling(summaries, fit=fit, outpath=outdir / f"{args.prefix}_k50_vs_invN.png")
		plot_collapse(byN, outpath=outdir / f"{args.prefix}_pL3_vs_rho_minus_1.png")
		plot_gamma_vs_rho(byN, outpath=outdir / f"{args.prefix}_gamma_vs_rho_minus_1.png")
	except ModuleNotFoundError:
		pass

	print(f"Wrote summary: {summary_path}")
	if str(args.bayes_method) != "none":
		print(f"Wrote bayes fit: {outdir / f'{args.prefix}_bayes_fit.json'}")
	if R > 0:
		print(f"bootstrap: resamples={R} seed={args.bootstrap_seed}")
	if str(args.bayes_method) != "none":
		print(
			"bayes: method={} draws={} seed={} prior_sigma={} assume_increasing={}".format(
				str(args.bayes_method),
				int(args.bayes_draws),
				(args.bayes_seed if args.bayes_seed is not None else ""),
				float(args.bayes_prior_sigma),
				int(args.bayes_assume_increasing),
			)
		)
	if fit is not None:
		print(f"k50 fit: k_inf={fit['k_inf']:.6g} c={fit['c']:.6g} R2={fit['r2']:.3f}")
	else:
		print("k50 fit: insufficient points")


if __name__ == "__main__":
	main()
