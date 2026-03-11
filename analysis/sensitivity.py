"""Sensitivity analysis for rho-curve summaries.

This script compares multiple *summary CSVs* (e.g. produced from sweeps with
slightly different hyperparameters like eta/corr_threshold) and quantifies how
k50 varies.

Design goals
- No simulation; operates purely on summary CSVs.
- Minimal dependencies (stdlib only).
- Friendly markdown output for research notes.

Usage
	python -m analysis.sensitivity \
	  --in-summary "base:outputs/analysis/rho_curve/rho_curve_base_summary.csv" \
	  --in-summary "eta060:outputs/analysis/rho_curve/rho_curve_eta060_summary.csv" \
	  --out outputs/analysis/rho_curve/sensitivity_k50.md
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _try_float(x: object) -> Optional[float]:
	try:
		v = float(x)  # type: ignore[arg-type]
		if math.isnan(v) or math.isinf(v):
			return None
		return float(v)
	except Exception:
		return None


def _fmt(x: object, *, digits: int = 4) -> str:
	if x is None:
		return "—"
	if isinstance(x, str):
		if not x.strip():
			return "—"
		f = _try_float(x)
		if f is None:
			return x
		x = f
	try:
		v = float(x)  # type: ignore[arg-type]
	except Exception:
		return str(x)
	if math.isnan(v) or math.isinf(v):
		return "—"
	if abs(v) < 1e-12:
		v = 0.0
	return f"{v:.{digits}g}"


def _parse_labeled_path(spec: str) -> tuple[str, Path]:
	"""Parse either 'label:path' or 'path'."""
	raw = str(spec)
	if ":" in raw and not raw.strip().endswith(":"):
		label, path = raw.split(":", 1)
		label = label.strip()
		path = path.strip()
		if label and path:
			return label, Path(path)
	p = Path(raw)
	return p.stem, p


@dataclass(frozen=True)
class Run:
	label: str
	path: Path
	by_n: Dict[int, float]


def load_k50_map(path: Path) -> Dict[int, float]:
	out: Dict[int, float] = {}
	with path.open(newline="") as f:
		r = csv.DictReader(f)
		for row in r:
			n = _try_float(row.get("players", ""))
			k50 = _try_float(row.get("k50", ""))
			cens = str(row.get("k50_censored_low", "") or "").strip().lower() in {"1", "true", "yes", "y"}
			if n is None or k50 is None:
				continue
			if cens:
				# Censored points are upper bounds; omit from sensitivity stats by default.
				continue
			out[int(n)] = float(k50)
	return out


def _mean(xs: Sequence[float]) -> float:
	return sum(xs) / float(len(xs)) if xs else float("nan")


def _std(xs: Sequence[float]) -> float:
	if len(xs) < 2:
		return 0.0
	m = _mean(xs)
	v = sum((x - m) ** 2 for x in xs) / float(len(xs))
	return math.sqrt(v)


def build_report(runs: Sequence[Run]) -> str:
	labels = [r.label for r in runs]
	all_ns: List[int] = sorted({n for r in runs for n in r.by_n.keys()})

	lines: List[str] = []
	lines.append("# Sensitivity report: k50 across runs")
	lines.append("")
	lines.append("Inputs:")
	for r in runs:
		lines.append(f"- {r.label}: {r.path}")

	if not all_ns:
		lines.append("")
		lines.append("No overlapping non-censored k50 entries found.")
		return "\n".join(lines) + "\n"

	lines.append("")
	lines.append("## Per-N variability")
	lines.append("")
	table: List[List[str]] = [["N", "runs", "k50_mean", "k50_std", "k50_range", "cv%", "missing"]]

	# A simple sensitivity score: std/mean across runs for each N.
	scores: List[tuple[float, int]] = []
	for n in all_ns:
		vals: List[float] = []
		missing = 0
		for r in runs:
			v = r.by_n.get(int(n))
			if v is None:
				missing += 1
			else:
				vals.append(float(v))
		m = _mean(vals)
		s = _std(vals)
		rng = (max(vals) - min(vals)) if vals else float("nan")
		cv = (100.0 * s / m) if (vals and m != 0.0 and math.isfinite(m)) else float("nan")
		table.append([str(n), str(len(vals)), _fmt(m), _fmt(s, digits=3), _fmt(rng, digits=3), _fmt(cv, digits=3), str(missing)])
		if vals and math.isfinite(cv):
			scores.append((float(cv), int(n)))

	# Render markdown table.
	def md_table(rows: Sequence[Sequence[str]]) -> str:
		h = rows[0]
		sep = ["---"] * len(h)
		out = ["| " + " | ".join(h) + " |", "| " + " | ".join(sep) + " |"]
		for r in rows[1:]:
			out.append("| " + " | ".join(r) + " |")
		return "\n".join(out) + "\n"

	lines.append(md_table(table))

	if scores:
		lines.append("## Most sensitive sizes")
		lines.append("")
		scores.sort(reverse=True)
		for cv, n in scores[: min(10, len(scores))]:
			lines.append(f"- N={n}: cv≈{_fmt(cv, digits=3)}%")

	lines.append("")
	lines.append("Notes:")
	lines.append("- This compares k50 point estimates across runs; it does not re-fit from per-seed data.")
	lines.append("- Left-censored k50 entries are omitted by default (upper bounds).")
	return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
	p = argparse.ArgumentParser(description="Sensitivity analysis across rho-curve summary CSVs")
	p.add_argument(
		"--in-summary",
		dest="inputs",
		action="append",
		required=True,
		help="Input summary CSV. Format: 'label:path' or just 'path'. Repeatable.",
	)
	p.add_argument("--out", type=Path, required=True, help="Output markdown path")
	args = p.parse_args(list(argv) if argv is not None else None)

	runs: List[Run] = []
	for spec in list(args.inputs):
		label, path = _parse_labeled_path(str(spec))
		if not path.exists():
			raise FileNotFoundError(path)
		runs.append(Run(label=str(label), path=path, by_n=load_k50_map(path)))

	text = build_report(runs)
	args.out.parent.mkdir(parents=True, exist_ok=True)
	args.out.write_text(text, encoding="utf-8")
	print(f"Wrote: {args.out}")


if __name__ == "__main__":
	main()
