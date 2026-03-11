"""Generate a detailed Markdown report from rho-curve scaling outputs.

This is meant for research notes / paper drafts: it turns the summary CSV
into readable text per N, and (optionally) uses the underlying sweep CSVs to
assess crossing reliability (bracketing gap, monotonicity, k-step resolution).

Typical usage (current canonical merged dataset):
	python -m analysis.rho_curve_report \
		--summary outputs/analysis/rho_curve/rho_curve_merged_with_lowband_clean_summary.csv \
		--fit-json outputs/analysis/rho_curve/rho_curve_merged_with_lowband_clean_k50_fit.json \
		--in outputs/sweeps/rho_curve/rho_curve_a0p4_b0p2425407_lag1_sampled_eta055_R4000_S30_N50_200_1000_k0p1_1p0_s0p02.csv \
		--in outputs/sweeps/rho_curve/rho_curve_a0p4_b0p2425407_lag1_sampled_eta055_R4000_S30_N200_k1p0_1p4_s0p02.csv \
		--in outputs/sweeps/rho_curve/rho_curve_critical_band_a0p4_b0p2425407_lag1_eta055_R2500_S10_N100_300_500_2000_k0p7_1p2_s0p02.csv \
		--in outputs/sweeps/rho_curve/rho_curve_lowband_corr0p09_clean_N300_500.csv \
		--out outputs/analysis/rho_curve/rho_curve_merged_with_lowband_clean_report.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from analysis import sensitivity


def _expand_input_paths(inputs: Sequence[str]) -> List[Path]:
	"""Expand file globs in CLI inputs.

	Supports passing quoted patterns like "outputs/.../N100*.csv".
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


@dataclass(frozen=True)
class Point:
	players: int
	k: float
	p3: float
	rho_m1: float
	n_seeds: int


def _try_float(x: object) -> Optional[float]:
	try:
		return float(x)  # type: ignore[arg-type]
	except Exception:
		return None


def _try_int(x: object) -> Optional[int]:
	try:
		return int(float(x))  # type: ignore[arg-type]
	except Exception:
		return None


def _fmt(x: object, *, digits: int = 4) -> str:
	if x is None:
		return "—"
	if isinstance(x, str):
		if not x.strip():
			return "—"
		# Try numeric formatting for numeric-looking strings.
		try:
			x = float(x)
		except Exception:
			return x
	try:
		v = float(x)  # type: ignore[arg-type]
	except Exception:
		return str(x)
	if math.isnan(v) or math.isinf(v):
		return "—"
	# keep tiny values readable
	if abs(v) < 1e-12:
		v = 0.0
	# use compact formatting
	return f"{v:.{digits}g}"


def load_summary(path: Path) -> List[dict[str, str]]:
	with path.open(newline="") as f:
		rows = list(csv.DictReader(f))
	# stable sort by players
	rows.sort(key=lambda r: int(float(r.get("players", "0") or 0.0)))
	return rows


def _infer_bayes_fit_path(summary_path: Path) -> Path:
	"""Infer '${prefix}_bayes_fit.json' from '${prefix}_summary.csv' in the same directory."""
	name = summary_path.name
	suffix = "_summary.csv"
	if name.endswith(suffix):
		prefix = name[: -len(suffix)]
		return summary_path.parent / f"{prefix}_bayes_fit.json"
	# Fallback: keep stem and append.
	return summary_path.parent / f"{summary_path.stem}_bayes_fit.json"


def _load_bayes_fit_json(path: Path) -> dict | None:
	try:
		if not path.exists():
			return None
		return json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return None


def _bayes_provenance_md(*, bayes_fit_path: Path, bayes: dict) -> str:
	"""Build a compact markdown snippet for Bayesian k50 provenance."""
	method = str(bayes.get("bayes_method", "") or "")
	if not method:
		method = "unknown"
	prior = bayes.get("bayes_prior_sigma", None)
	draws = bayes.get("bayes_draws", None)
	seed = bayes.get("bayes_seed", None)
	assume_inc = bayes.get("bayes_assume_increasing", None)
	per_n = bayes.get("per_N", [])

	lines: list[str] = []
	lines.append("")
	lines.append("Bayesian k50 provenance:")
	lines.append(f"- bayes_fit_json: {bayes_fit_path}")
	lines.append(f"- method: {method}")
	if prior is not None:
		lines.append(f"- prior_sigma: {_fmt(prior)}")
	if draws is not None:
		lines.append(f"- draws: {draws}")
	if seed is not None and str(seed) != "":
		lines.append(f"- seed: {seed}")
	if assume_inc is not None:
		lines.append(f"- assume_increasing: {assume_inc}")

	# Per-N compact table.
	rows: list[list[str]] = [["N", "k50_bayes_mean", "95% CrI", "n_eff"]]
	if isinstance(per_n, list):
		for item in per_n:
			if not isinstance(item, dict):
				continue
			n = _try_int(item.get("players"))
			if n is None:
				continue
			mu = _fmt(item.get("k50_bayes_mean"))
			lo = _fmt(item.get("k50_bayes_ci_low"))
			hi = _fmt(item.get("k50_bayes_ci_high"))
			cri = "—"
			if lo != "—" and hi != "—":
				cri = f"{lo}..{hi}"
			neff = item.get("k50_bayes_n_eff")
			rows.append([str(int(n)), mu, cri, (str(neff) if neff is not None else "—")])
	if len(rows) > 1:
		lines.append("")
		lines.append(_md_table(rows).rstrip())

	return "\n".join(lines).rstrip() + "\n"


def _get_float(row: dict[str, str], key: str) -> Optional[float]:
	v = row.get(key, "")
	if v is None:
		return None
	f = _try_float(v)
	return f


def _get_bool(row: dict[str, str], key: str) -> bool:
	v = (row.get(key, "") or "").strip().lower()
	return v in {"1", "true", "yes", "y"}


def iter_points(paths: Sequence[Path]) -> Iterable[Point]:
	for path in paths:
		with path.open(newline="") as f:
			r = csv.DictReader(f)
			for row in r:
				n = _try_int(row.get("players"))
				k = _try_float(row.get("selection_strength"))
				p3 = _try_float(row.get("p_level_3"))
				rho_m1 = _try_float(row.get("rho_minus_1"))
				n_seeds = _try_int(row.get("n_seeds"))
				if n is None or k is None or p3 is None or rho_m1 is None:
					continue
				if n_seeds is None:
					n_seeds = 0
				yield Point(players=n, k=float(k), p3=float(p3), rho_m1=float(rho_m1), n_seeds=int(n_seeds))


def group_points(points: Iterable[Point]) -> Dict[int, List[Point]]:
	"""Group points by N, deduplicating by (N, k) keeping max n_seeds.

	This mirrors the scaling merge rule so diagnostics are computed on the same
	set of points that would be used by the analysis.
	"""
	best: Dict[Tuple[int, float], Point] = {}
	for p in points:
		key = (p.players, round(p.k, 10))
		cur = best.get(key)
		if cur is None or p.n_seeds > cur.n_seeds:
			best[key] = p

	out: Dict[int, List[Point]] = {}
	for p in best.values():
		out.setdefault(p.players, []).append(p)
	for n, seq in out.items():
		seq.sort(key=lambda x: x.k)
	return out


@dataclass(frozen=True)
class CrossingDiag:
	target: float
	has_bracket: bool
	k_lo: float | None
	k_hi: float | None
	p_lo: float | None
	p_hi: float | None
	gap: float | None


def crossing_diagnostics(seq: Sequence[Point], *, target: float) -> CrossingDiag:
	if not seq:
		return CrossingDiag(target=target, has_bracket=False, k_lo=None, k_hi=None, p_lo=None, p_hi=None, gap=None)

	# find first adjacent pair that brackets the target
	for a, b in zip(seq, seq[1:]):
		if (a.p3 < target <= b.p3) or (a.p3 > target >= b.p3):
			return CrossingDiag(
				target=target,
				has_bracket=True,
				k_lo=float(a.k),
				k_hi=float(b.k),
				p_lo=float(a.p3),
				p_hi=float(b.p3),
				gap=float(b.k - a.k),
			)
	# special case: already above target at minimum k (left-censored)
	if seq[0].p3 >= target:
		return CrossingDiag(target=target, has_bracket=False, k_lo=float(seq[0].k), k_hi=None, p_lo=float(seq[0].p3), p_hi=None, gap=None)

	return CrossingDiag(target=target, has_bracket=False, k_lo=None, k_hi=None, p_lo=None, p_hi=None, gap=None)


@dataclass(frozen=True)
class SeqQuality:
	n_points: int
	k_min: float | None
	k_max: float | None
	k_step_min: float | None
	k_step_median: float | None
	monotone_violations: int


def sequence_quality(seq: Sequence[Point]) -> SeqQuality:
	if not seq:
		return SeqQuality(n_points=0, k_min=None, k_max=None, k_step_min=None, k_step_median=None, monotone_violations=0)

	# seq is already deduplicated by k; still guard against accidental ties.
	ks = [p.k for p in seq]
	steps = [b - a for a, b in zip(ks, ks[1:]) if b > a + 1e-15]
	steps_sorted = sorted(steps)
	step_min = steps_sorted[0] if steps_sorted else None
	step_med = steps_sorted[len(steps_sorted) // 2] if steps_sorted else None

	# monotonicity is not guaranteed, but many decreases indicate noise
	viol = 0
	for a, b in zip(seq, seq[1:]):
		if b.p3 + 1e-12 < a.p3:
			viol += 1

	return SeqQuality(
		n_points=len(seq),
		k_min=float(seq[0].k),
		k_max=float(seq[-1].k),
		k_step_min=(float(step_min) if step_min is not None else None),
		k_step_median=(float(step_med) if step_med is not None else None),
		monotone_violations=int(viol),
	)


def _md_table(rows: Sequence[Sequence[str]]) -> str:
	# rows include header as first row
	if not rows:
		return ""
	header = rows[0]
	sep = ["---"] * len(header)
	out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
	for r in rows[1:]:
		out.append("| " + " | ".join(r) + " |")
	return "\n".join(out) + "\n"


def _first_nonempty_from_sweeps(
	paths: Sequence[Path],
	*,
	keys: Sequence[str],
) -> dict[str, str]:
	"""Extract the first non-empty value for each key across sweep CSV inputs.

	This is intentionally simple and stable: it does not attempt to validate
	consistency across files; it just provides a quick reproducibility summary.
	"""
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
						v = row.get(k)
						if v is None:
							continue
						s = str(v).strip()
						if s != "":
							out[str(k)] = s
							remaining.remove(k)
					if not remaining:
						return out
		except Exception:
			# Ignore unreadable/invalid CSVs; diagnostics are best-effort.
			continue

	return out


def _stage2_settings_inconsistencies(
	paths: Sequence[Path],
	*,
	keys: Sequence[str],
) -> dict[str, dict[str, list[Path]]]:
	"""Return per-key distinct values (non-empty) mapped to the files providing them."""
	by_key: dict[str, dict[str, list[Path]]] = {str(k): {} for k in keys}
	for path in paths:
		found_in_file: set[str] = set()
		try:
			with path.open(newline="") as f:
				r = csv.DictReader(f)
				for row in r:
					for k in keys:
						ks = str(k)
						if ks not in by_key:
							continue
						v = row.get(ks)
						if v is None:
							continue
						s = str(v).strip()
						if s == "":
							continue
						bucket = by_key[ks].setdefault(s, [])
						if not bucket or bucket[-1] != path:
							bucket.append(path)
						found_in_file.add(ks)
					# Stop early once we've seen at least one non-empty value for every key in this file.
					if len(found_in_file) >= len(keys):
						break
		except Exception:
			continue

	# Filter to only inconsistent keys.
	out: dict[str, dict[str, list[Path]]] = {}
	for k, m in by_key.items():
		# Only count distinct non-empty values.
		if len(m) > 1:
			out[k] = m
	return out


def build_notes_report(
	*,
	rows: Sequence[dict[str, str]],
	fit: dict | None,
	summary_path: Path,
	inputs: Sequence[Path],
	byN: Dict[int, List[Point]],
	generated_utc: str,
	sensitivity_md: str = "",
	bayes_provenance_md: str = "",
) -> str:
	lines: list[str] = []
	lines.append("# Rho-curve scaling report")
	lines.append("")
	lines.append(f"Generated: {generated_utc}")
	lines.append("")
	lines.append(f"Summary: {summary_path}")
	if inputs:
		lines.append("")
		lines.append("Inputs:")
		for pth in inputs:
			lines.append(f"- {pth}")
		stage2 = _first_nonempty_from_sweeps(
			inputs,
			keys=(
				"stage2_method",
				"stage2_prefilter",
				"power_ratio_kappa",
				"permutation_alpha",
				"permutation_resamples",
				"permutation_seed",
			),
		)
		if stage2:
			lines.append("")
			lines.append("Stage2 settings (from sweep CSV; first non-empty):")
			lines.append(f"- method: {_fmt(stage2.get('stage2_method'))}")
			lines.append(f"- prefilter: {_fmt(stage2.get('stage2_prefilter'))}")
			lines.append(f"- kappa (fft power ratio): {_fmt(stage2.get('power_ratio_kappa'))}")
			lines.append(f"- alpha (permutation): {_fmt(stage2.get('permutation_alpha'))}")
			lines.append(f"- resamples (permutation): {_fmt(stage2.get('permutation_resamples'))}")
			lines.append(f"- seed (permutation): {_fmt(stage2.get('permutation_seed'))}")
			incons = _stage2_settings_inconsistencies(
				inputs,
				keys=(
					"stage2_method",
					"stage2_prefilter",
					"power_ratio_kappa",
					"permutation_alpha",
					"permutation_resamples",
					"permutation_seed",
				),
			)
			if incons:
				lines.append("")
				lines.append("WARNING: inconsistent Stage2 settings across inputs:")
				rows = [["key", "value", "files"]]
				for k in sorted(incons.keys()):
					for v, files in sorted(incons[k].items(), key=lambda kv: kv[0]):
						# Keep the list short but informative.
						file_list = ", ".join(str(p) for p in files[:3])
						if len(files) > 3:
							file_list += f" (+{len(files)-3} more)"
						rows.append([str(k), str(v), file_list])
				lines.append(_md_table(rows))
	if fit is not None:
		lines.append("")
		lines.append("k50 fit (excluding left-censored points):")
		lines.append(f"- k_inf: {_fmt(fit.get('k_inf'))}")
		lines.append(f"- c: {_fmt(fit.get('c'))}")
		lines.append(f"- r2: {_fmt(fit.get('r2'), digits=3)}")
	if bayes_provenance_md.strip():
		lines.append(bayes_provenance_md.rstrip())

	lines.append("")
	lines.append("## Per-N summary")
	lines.append("")

	# Compact global table first
	table: list[list[str]] = [[
		"N",
		"k_min",
		"P3(k_min)",
		"k50",
		"k90",
		"P3_max",
		"plateau_k",
		"censored?",
		"monotone_viol",
		"Δk@50",
	]]

	for r in rows:
		n = int(float(r.get("players", "0") or 0.0))
		pk0 = r.get("plateau_k_start", "")
		pk1 = r.get("plateau_k_end", "")
		plateau = "—"
		if pk0.strip() and pk1.strip():
			plateau = f"{_fmt(pk0)}..{_fmt(pk1)}"
		cens = str(_get_bool(r, "k50_censored_low"))

		monov = "—"
		dk50 = "—"
		if inputs and n in byN:
			q = sequence_quality(byN[n])
			monov = str(q.monotone_violations)
			d50 = crossing_diagnostics(byN[n], target=0.5)
			if d50.has_bracket and d50.gap is not None:
				dk50 = _fmt(d50.gap, digits=3)

		table.append(
			[
				str(n),
				_fmt(r.get("k_min")),
				_fmt(r.get("p3_at_k_min"), digits=3),
				_fmt(r.get("k50")),
				_fmt(r.get("k90")),
				_fmt(r.get("p3_max"), digits=3),
				plateau,
				cens,
				monov,
				dk50,
			]
		)

	lines.append(_md_table(table))
	lines.append("\n## Interpretation per N\n")

	for r in rows:
		n = int(float(r.get("players", "0") or 0.0))
		lines.append(f"### N = {n}")
		lines.append("")
		lines.append("Key numbers:")
		lines.append(f"- k_min = {_fmt(r.get('k_min'))}, P(L3)@k_min = {_fmt(r.get('p3_at_k_min'), digits=3)}")
		lines.append(f"- k_first_positive = {_fmt(r.get('k_first_positive'))}")
		k50 = _fmt(r.get('k50'))
		cens_flag = str(_get_bool(r, 'k50_censored_low'))
		ci_lo = r.get("k50_boot_ci_low", "")
		ci_hi = r.get("k50_boot_ci_high", "")
		ci_std = r.get("k50_boot_std", "")
		ci_text = ""
		if (ci_lo or "").strip() and (ci_hi or "").strip():
			ci_text = f"; bootstrap 95% CI ≈ {_fmt(ci_lo)}..{_fmt(ci_hi)}"
			if (ci_std or "").strip():
				ci_text += f" (std≈{_fmt(ci_std, digits=3)})"
		
		bayes_text = ""
		b_method = str(r.get("bayes_method", "") or "").strip()
		b_lo = str(r.get("k50_bayes_ci_low", "") or "").strip()
		b_hi = str(r.get("k50_bayes_ci_high", "") or "").strip()
		b_std = str(r.get("k50_bayes_std", "") or "").strip()
		if b_lo and b_hi:
			label = f"bayes({b_method})" if b_method else "bayes"
			bayes_text = f"; {label} 95% CrI ≈ {_fmt(b_lo)}..{_fmt(b_hi)}"
			if b_std:
				bayes_text += f" (std≈{_fmt(b_std, digits=3)})"
		lines.append(f"- k50 = {k50} (censored_low = {cens_flag}{ci_text})")
		if bayes_text:
			lines[-1] = lines[-1].rstrip(")") + f"{bayes_text})"
		lines.append(f"- k90 = {_fmt(r.get('k90'))}")
		lines.append(f"- P(L3)_max = {_fmt(r.get('p3_max'), digits=3)} at k = {_fmt(r.get('k_at_max'))}")
		lines.append(f"- plateau k-range (exact max within tol) = {_fmt(r.get('plateau_k_start'))}..{_fmt(r.get('plateau_k_end'))}")
		lines.append(f"- corr(rho-1, P(L3)) = {_fmt(r.get('corr_rho_minus_1_p_level_3'), digits=3)}")

		if inputs and n in byN:
			seq = byN[n]
			q = sequence_quality(seq)
			lines.append("")
			lines.append("Reliability diagnostics (from underlying merged points):")
			lines.append(f"- n_points = {q.n_points}, k_range = {_fmt(q.k_min)}..{_fmt(q.k_max)}")
			lines.append(f"- k_step_min ≈ {_fmt(q.k_step_min, digits=3)}, k_step_median ≈ {_fmt(q.k_step_median, digits=3)}")
			lines.append(f"- monotonicity decreases count (P(L3) drops as k increases) = {q.monotone_violations}")

			for target in (0.5, 0.9):
				d = crossing_diagnostics(seq, target=target)
				if d.has_bracket:
					lines.append(
						f"- crossing@{target:g} bracket: k in [{_fmt(d.k_lo)}, {_fmt(d.k_hi)}], "
						f"P3 in [{_fmt(d.p_lo, digits=3)}, {_fmt(d.p_hi, digits=3)}], Δk = {_fmt(d.gap, digits=3)}"
					)
				else:
					if d.k_lo is not None and d.k_hi is None and d.p_lo is not None and d.p_lo >= target:
						lines.append(f"- crossing@{target:g}: already ≥ target at k_min (left-censored at low k)")
					else:
						lines.append(f"- crossing@{target:g}: no adjacent bracket found in available k grid")

			c50 = crossing_diagnostics(seq, target=0.5)
			reliability = "unknown"
			if c50.has_bracket and c50.gap is not None:
				if c50.gap <= 0.02 and q.monotone_violations == 0:
					reliability = "high (tight bracket; near-monotone)"
				elif c50.gap <= 0.02 and q.monotone_violations <= 1:
					reliability = "medium (tight bracket; minor non-monotonicity)"
				elif c50.gap <= 0.04 and q.monotone_violations <= 2:
					reliability = "medium (ok bracket; some noise)"
				else:
					reliability = "low (non-monotone / noisy around crossing)"
			elif seq and seq[0].p3 >= 0.5:
				reliability = "left-censored (only an upper bound without lower-k data)"
			else:
				reliability = "missing bracket (need more k points around crossing)"
			lines.append(f"- k50 reliability: {reliability}")

		lines.append("")

	if sensitivity_md.strip():
		lines.append("")
		lines.append(sensitivity_md.rstrip())
		lines.append("")

	return "\n".join(lines) + "\n"


def build_paper_report(
	*,
	rows: Sequence[dict[str, str]],
	fit: dict | None,
	summary_path: Path,
	inputs: Sequence[Path],
	byN: Dict[int, List[Point]],
	generated_utc: str,
	sensitivity_md: str = "",
	bayes_provenance_md: str = "",
) -> str:
	"""Paper-style narrative: trend → exceptions → limitations + a concluding summary."""
	# Extract k50 values for a trend statement.
	k50_pts: list[tuple[int, float]] = []
	censored_ns: list[int] = []
	for r in rows:
		n = int(float(r.get("players", "0") or 0.0))
		if _get_bool(r, "k50_censored_low"):
			censored_ns.append(n)
			continue
		k50 = _get_float(r, "k50")
		if k50 is not None:
			k50_pts.append((n, float(k50)))
	# Sort by N.
	k50_pts.sort(key=lambda x: x[0])

	lines: list[str] = []
	lines.append("# Rho-curve: paper-style narrative")
	lines.append("")
	lines.append(f"Generated: {generated_utc}")
	lines.append("")
	lines.append(f"Summary source: {summary_path}")
	if inputs:
		lines.append("Inputs:")
		for pth in inputs:
			lines.append(f"- {pth}")
		stage2 = _first_nonempty_from_sweeps(
			inputs,
			keys=(
				"stage2_method",
				"stage2_prefilter",
				"power_ratio_kappa",
				"permutation_alpha",
				"permutation_resamples",
				"permutation_seed",
			),
		)
		if stage2:
			lines.append("")
			lines.append("Stage2 settings (from sweep CSV; first non-empty):")
			lines.append(f"- method: {_fmt(stage2.get('stage2_method'))}")
			lines.append(f"- prefilter: {_fmt(stage2.get('stage2_prefilter'))}")
			lines.append(f"- kappa (fft power ratio): {_fmt(stage2.get('power_ratio_kappa'))}")
			lines.append(f"- alpha (permutation): {_fmt(stage2.get('permutation_alpha'))}")
			lines.append(f"- resamples (permutation): {_fmt(stage2.get('permutation_resamples'))}")
			lines.append(f"- seed (permutation): {_fmt(stage2.get('permutation_seed'))}")
			incons = _stage2_settings_inconsistencies(
				inputs,
				keys=(
					"stage2_method",
					"stage2_prefilter",
					"power_ratio_kappa",
					"permutation_alpha",
					"permutation_resamples",
					"permutation_seed",
				),
			)
			if incons:
				lines.append("")
				lines.append("WARNING: inconsistent Stage2 settings across inputs:")
				rows = [["key", "value", "files"]]
				for k in sorted(incons.keys()):
					for v, files in sorted(incons[k].items(), key=lambda kv: kv[0]):
						file_list = ", ".join(str(p) for p in files[:3])
						if len(files) > 3:
							file_list += f" (+{len(files)-3} more)"
						rows.append([str(k), str(v), file_list])
				lines.append(_md_table(rows))

	if bayes_provenance_md.strip():
		lines.append(bayes_provenance_md.rstrip())

	lines.append("")
	lines.append("## Main trend")
	lines.append("")
	if k50_pts:
		# Describe directionality cautiously.
		pairs = ", ".join([f"N={n}: k50={_fmt(k)}" for n, k in k50_pts])
		lines.append(
			"We estimate the finite-size critical proxy $k_{50}$ (defined by $P(\\mathrm{Level3})=0.5$) for each system size $N$. "
			"Across the available sizes, the estimated values are: " + pairs + "."
		)
		lines.append(
			"While $k_{50}$ shows clear size dependence, the dependence is not strictly monotone across all $N$ in this merged dataset, "
			"suggesting either nontrivial finite-size corrections and/or residual heterogeneity across sweep segments (e.g., different run lengths / seeds per segment)."
		)
	else:
		lines.append("No usable (non-censored) $k_{50}$ estimates were found in the summary.")

	if fit is not None:
		lines.append("")
		lines.append(
			"A simple linear fit of the form $k_{50}(N) \\approx k_\\infty + c/N$ (excluding left-censored points) yields "
			f"$k_\\infty\\approx{_fmt(fit.get('k_inf'))}$, $c\\approx{_fmt(fit.get('c'))}$ with $R^2\\approx{_fmt(fit.get('r2'), digits=3)}$. "
			"Given the low $R^2$, this fit should be treated as a coarse summary rather than a definitive scaling law."
		)

	if censored_ns:
		lines.append("")
		lines.append(
			"Some sizes are left-censored at the lowest sampled $k$ (i.e., $P(\\mathrm{Level3})\\ge 0.5$ already at $k_{\\min}$), so their $k_{50}$ would only provide upper bounds. "
			"(In the current run, censored sizes: " + ", ".join(map(str, sorted(censored_ns))) + ".)"
		)

	lines.append("")
	lines.append("## Per-size interpretation")
	lines.append("")
	for r in rows:
		n = int(float(r.get("players", "0") or 0.0))
		kmin = _fmt(r.get("k_min"))
		p3min = _fmt(r.get("p3_at_k_min"), digits=3)
		k50 = _fmt(r.get("k50"))
		k90 = _fmt(r.get("k90"))
		pmax = _fmt(r.get("p3_max"), digits=3)
		pks = _fmt(r.get("plateau_k_start"))
		pke = _fmt(r.get("plateau_k_end"))
		cens = _get_bool(r, "k50_censored_low")

		# Reliability statement from diagnostics if available.
		rel_sentence = ""
		if inputs and n in byN:
			seq = byN[n]
			q = sequence_quality(seq)
			c50 = crossing_diagnostics(seq, target=0.5)
			if c50.has_bracket and c50.gap is not None:
				rel_sentence = (
					f"The crossing is bracketed within $\\Delta k\\approx{_fmt(c50.gap, digits=3)}$ (k-grid resolution), "
					f"with {q.monotone_violations} observed non-monotone steps in $P(\\mathrm{{Level3}})$ across k."
				)
			elif cens:
				rel_sentence = "The lowest sampled k is already above the 0.5 threshold (left-censored), so only an upper bound on $k_{50}$ is available without extending to lower k."
			else:
				rel_sentence = "The available k-grid does not bracket the 0.5 crossing; additional k points are needed around the transition."

		lines.append(f"### N = {n}")
		lines.append("")
		lines.append(
			f"At $N={n}$, the sweep starts at $k_\\min={kmin}$ with $P(\\mathrm{{Level3}})={p3min}$. "
			f"We estimate $k_{{50}}={k50}$ and $k_{{90}}={k90}$ (when available), and observe a maximum $P(\\mathrm{{Level3}})_\\max={pmax}$ with an end-plateau over k≈{pks}..{pke}."
		)
		if rel_sentence:
			lines.append(rel_sentence)
		lines.append("")

	lines.append("## Exceptions and limitations")
	lines.append("")
	lines.append(
		"Although all merged datasets share the same high-level model settings, different sweep segments may differ in run length (rounds), seed count, and the k-range coverage. "
		"These differences can introduce additional scatter in $k_{50}$ beyond pure finite-size effects."
	)
	lines.append(
		"Non-monotonicity of $P(\\mathrm{Level3})$ as a function of k (visible as multiple monotonicity violations) indicates stochastic noise at fixed resolution $\\Delta k$, "
		"which can bias simple linear interpolation for crossings. In such cases, increasing the number of seeds and/or using a monotone regression / smoothing procedure would improve robustness."
	)
	lines.append("")

	if sensitivity_md.strip():
		lines.append("## Appendix: sensitivity analysis")
		lines.append("")
		lines.append(sensitivity_md.rstrip())
		lines.append("")

	lines.append("## Conclusion")
	lines.append("")
	lines.append(
		"In summary, the provided sweeps enable direct interpolation of $k_{50}$ and $k_{90}$ where crossings are bracketed by the sampled k-grid. "
		"The resulting $k_{50}$ values can exhibit finite-size dependence and may be non-monotone across N due to stochastic noise and finite k resolution. "
		"The simple $k_{50}(N)=k_\\infty+c/N$ fit (when available) should therefore be treated as a rough descriptive summary unless the diagnostics indicate clean bracketing and near-monotone curves."
	)

	return "\n".join(lines) + "\n"


def _demote_headings(md: str, *, levels: int) -> str:
	"""Demote markdown headings by N levels (e.g. '#' -> '##')."""
	if levels <= 0:
		return md
	out: list[str] = []
	for line in (md or "").splitlines():
		if line.startswith("#"):
			i = 0
			while i < len(line) and line[i] == "#":
				i += 1
			# Only treat as heading if followed by a space.
			if i > 0 and i < len(line) and line[i] == " ":
				out.append(("#" * (i + levels)) + line[i:])
				continue
		out.append(line)
	return "\n".join(out) + "\n"


def _build_sensitivity_embed(
	*,
	primary_summary: Path,
	extra_specs: Sequence[str],
	for_paper: bool,
) -> str:
	"""Build a sensitivity markdown snippet suitable for embedding."""
	if not extra_specs:
		return ""

	seen: set[Path] = set()
	runs: list[sensitivity.Run] = []

	# Always include the primary summary as baseline.
	base_path = primary_summary
	if base_path.exists():
		seen.add(base_path.resolve())
		runs.append(
			sensitivity.Run(
				label=base_path.stem,
				path=base_path,
				by_n=sensitivity.load_k50_map(base_path),
			)
		)

	for spec in extra_specs:
		label, path = sensitivity._parse_labeled_path(str(spec))
		if not path.exists():
			raise FileNotFoundError(path)
		resolved = path.resolve()
		if resolved in seen:
			continue
		seen.add(resolved)
		runs.append(sensitivity.Run(label=str(label), path=path, by_n=sensitivity.load_k50_map(path)))

	# Need at least 2 runs to compare.
	if len(runs) < 2:
		return ""

	md = sensitivity.build_report(runs)
	md = _demote_headings(md, levels=1)
	if for_paper:
		# Keep embedded section compact: rename the top heading.
		lines = md.splitlines()
		if lines and lines[0].startswith("## "):
			lines[0] = "### k50 variability across runs"
		md = "\n".join(lines).rstrip() + "\n"
	return md


def main(argv: Optional[Sequence[str]] = None) -> None:
	p = argparse.ArgumentParser(description="Generate a Markdown report for rho-curve scaling outputs")
	p.add_argument("--summary", type=str, required=True, help="Summary CSV path")
	p.add_argument("--fit-json", type=str, default="", help="Optional k50 fit JSON path")
	p.add_argument(
		"--bayes-fit-json",
		type=str,
		default="",
		help="Optional Bayesian k50 provenance JSON path. If not provided, inferred as '${prefix}_bayes_fit.json' next to --summary.",
	)
	p.add_argument("--in", dest="inputs", action="append", default=[], help="Optional sweep CSV inputs for reliability diagnostics")
	p.add_argument(
		"--sensitivity-summary",
		dest="sensitivity_summaries",
		action="append",
		default=[],
		help="Optional additional summary CSVs to compare (label:path or path). Repeatable. If provided, the primary --summary is included as baseline.",
	)
	p.add_argument("--out", type=str, required=True, help="Output Markdown path")
	p.add_argument("--out-paper", type=str, default="", help="Optional paper-style Markdown output path")
	args = p.parse_args(list(argv) if argv is not None else None)

	summary_path = Path(args.summary)
	rows = load_summary(summary_path)

	fit: dict | None = None
	if args.fit_json:
		fit_path = Path(args.fit_json)
		if fit_path.exists():
			fit = json.loads(fit_path.read_text())

	inputs = _expand_input_paths(list(args.inputs))
	byN: Dict[int, List[Point]] = {}
	if inputs:
		byN = group_points(iter_points(inputs))

	bayes_prov_md = ""
	bayes_fit_path = Path(args.bayes_fit_json) if args.bayes_fit_json else _infer_bayes_fit_path(summary_path)
	bayes = _load_bayes_fit_json(bayes_fit_path)
	if bayes is not None:
		bayes_prov_md = _bayes_provenance_md(bayes_fit_path=bayes_fit_path, bayes=bayes)

	now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
	sens_notes = _build_sensitivity_embed(
		primary_summary=summary_path,
		extra_specs=list(args.sensitivity_summaries),
		for_paper=False,
	)
	sens_paper = _build_sensitivity_embed(
		primary_summary=summary_path,
		extra_specs=list(args.sensitivity_summaries),
		for_paper=True,
	)
	notes_text = build_notes_report(
		rows=rows,
		fit=fit,
		summary_path=summary_path,
		inputs=inputs,
		byN=byN,
		generated_utc=now,
		sensitivity_md=sens_notes,
		bayes_provenance_md=bayes_prov_md,
	)
	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(notes_text)
	print(f"Wrote report: {out_path}")

	if args.out_paper:
		paper_text = build_paper_report(
			rows=rows,
			fit=fit,
			summary_path=summary_path,
			inputs=inputs,
			byN=byN,
			generated_utc=now,
			sensitivity_md=sens_paper,
			bayes_provenance_md=bayes_prov_md,
		)
		paper_path = Path(args.out_paper)
		paper_path.parent.mkdir(parents=True, exist_ok=True)
		paper_path.write_text(paper_text)
		print(f"Wrote paper report: {paper_path}")


if __name__ == "__main__":
	main()
