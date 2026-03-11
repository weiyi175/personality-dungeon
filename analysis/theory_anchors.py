"""Theory anchors (core + extended) for regression-friendly verification.

This module provides a small, stable set of anchors to protect research
conclusions against accidental metric/threshold drift.

Core anchors are deterministic and do NOT require simulation.
Extended anchors can optionally read stored validation timeseries.

Usage
	python -m analysis.theory_anchors --out outputs/analysis/anchors.md
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from analysis.cycle_metrics import classify_cycle_level


def _fmt(x: object, *, digits: int = 4) -> str:
	if x is None:
		return "—"
	try:
		v = float(x)  # type: ignore[arg-type]
	except Exception:
		return str(x)
	if math.isnan(v) or math.isinf(v):
		return "—"
	return f"{v:.{digits}g}"


@dataclass(frozen=True)
class AnchorResult:
	name: str
	passed: bool
	level: int
	stage2_method: str
	stage2_stat_name: str
	stage2_stat: float


def _core_control_constant(*, n: int = 3000) -> Dict[str, List[float]]:
	# Control-like constant weights (e.g., a=b=0 => w tends to stay at 1).
	w = [1.0] * int(n)
	return {"aggressive": w, "defensive": w, "balanced": w}


def _core_synthetic_cycle(*, n: int = 4000, cycles: int = 20, r: float = 0.05) -> Dict[str, List[float]]:
	pa: List[float] = []
	pd: List[float] = []
	for i in range(int(n)):
		theta = 2.0 * math.pi * (float(i) / float(n)) * float(cycles)
		pa.append((1.0 / 3.0) + float(r) * math.cos(theta))
		pd.append((1.0 / 3.0) + float(r) * math.sin(theta))
	pb = [1.0 - a - d for a, d in zip(pa, pd)]
	return {"aggressive": pa, "defensive": pd, "balanced": pb}


def _extended_from_timeseries_csv(path: Path, *, n_rows: int = 12000, series: str = "p") -> Dict[str, List[float]]:
	key = "p" if series == "p" else "w"
	a: List[float] = []
	d: List[float] = []
	b: List[float] = []
	with path.open(newline="") as f:
		r = csv.DictReader(f)
		for row_i, row in enumerate(r):
			if row_i >= int(n_rows):
				break
			a.append(float(row[f"{key}_aggressive"]))
			d.append(float(row[f"{key}_defensive"]))
			b.append(float(row[f"{key}_balanced"]))
	return {"aggressive": a, "defensive": d, "balanced": b}


def run_anchor(
	name: str,
	proportions: Dict[str, Sequence[float]],
	*,
	stage2_method: str,
	expect_level_at_least: int | None = None,
	expect_level_below: int | None = None,
	expect_level_exact: int | None = None,
	**kwargs,
) -> AnchorResult:
	res = classify_cycle_level(
		proportions,
		tail=int(kwargs.get("tail", 3000)),
		amplitude_threshold=float(kwargs.get("amplitude_threshold", 0.02)),
		corr_threshold=float(kwargs.get("corr_threshold", 0.3)),
		eta=float(kwargs.get("eta", 0.8)),
		stage2_method=str(stage2_method),
		stage2_prefilter=bool(kwargs.get("stage2_prefilter", True)),
		power_ratio_kappa=float(kwargs.get("power_ratio_kappa", 8.0)),
		permutation_alpha=float(kwargs.get("permutation_alpha", 0.05)),
		permutation_resamples=int(kwargs.get("permutation_resamples", 200)),
		permutation_seed=(int(kwargs.get("permutation_seed", 123)) if kwargs.get("permutation_seed", 123) is not None else None),
		stage3_method=str(kwargs.get("stage3_method", "turning")),
		normalize_for_phase=bool(kwargs.get("normalize_for_phase", False)),
	)
	stat_name = ""
	stat = 0.0
	if res.stage2 is not None:
		stat_name = str(res.stage2.statistic_name or "")
		stat = float(res.stage2.statistic) if res.stage2.statistic is not None else 0.0
	level = int(res.level)
	passed = True
	if expect_level_exact is not None:
		passed = passed and (level == int(expect_level_exact))
	if expect_level_at_least is not None:
		passed = passed and (level >= int(expect_level_at_least))
	if expect_level_below is not None:
		passed = passed and (level < int(expect_level_below))
	return AnchorResult(
		name=str(name),
		passed=bool(passed),
		level=int(level),
		stage2_method=str(stage2_method),
		stage2_stat_name=str(stat_name),
		stage2_stat=float(stat),
	)


def build_markdown(results: Sequence[AnchorResult]) -> str:
	lines: List[str] = []
	lines.append("# Theory anchors")
	lines.append("")
	lines.append("These anchors are intended for regression-friendly verification of cycle metrics.")
	lines.append("")
	rows: List[List[str]] = [["anchor", "stage2_method", "passed", "level", "stat_name", "stat"]]
	for r in results:
		rows.append([
			r.name,
			r.stage2_method,
			"yes" if r.passed else "no",
			str(r.level),
			r.stage2_stat_name or "—",
			_fmt(r.stage2_stat, digits=4),
		])
	# markdown table
	header = rows[0]
	sep = ["---"] * len(header)
	lines.append("| " + " | ".join(header) + " |")
	lines.append("| " + " | ".join(sep) + " |")
	for rr in rows[1:]:
		lines.append("| " + " | ".join(rr) + " |")
	lines.append("")
	return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
	p = argparse.ArgumentParser(description="Run theory anchors")
	p.add_argument("--out", type=Path, required=True, help="Output markdown path")
	p.add_argument(
		"--include-extended",
		action="store_true",
		help="If set, include an extended anchor from outputs/sweeps/hotspot_validation if present.",
	)
	args = p.parse_args(list(argv) if argv is not None else None)

	results: List[AnchorResult] = []

	# Core: control-like constant series should NEVER show cycles.
	control = _core_control_constant(n=3000)
	for method in ("autocorr_threshold", "fft_power_ratio", "permutation_p"):
		results.append(
			run_anchor(
				"core_control_constant",
				control,
				stage2_method=method,
				expect_level_below=2,
			)
		)

	# Core: synthetic rotation should be detected as Level3 across methods.
	cycle = _core_synthetic_cycle(n=4000)
	for method in ("autocorr_threshold", "fft_power_ratio", "permutation_p"):
		results.append(
			run_anchor(
				"core_synthetic_cycle",
				cycle,
				stage2_method=method,
				expect_level_exact=3,
				eta=0.8,
				corr_threshold=0.3,
				power_ratio_kappa=8.0,
				permutation_resamples=200,
				permutation_seed=123,
			)
		)

	if bool(args.include_extended):
		path = Path("outputs/sweeps/hotspot_validation/val_g_0p008_eps_0p2_k_0p02_N_500_T_30000.csv")
		if path.exists():
			props = _extended_from_timeseries_csv(path, n_rows=12000, series="p")
			# This anchor is descriptive (not enforced across all methods).
			results.append(
				run_anchor(
					"ext_hotspot_validation_g008_eps02_k002_N500",
					props,
					stage2_method="autocorr_threshold",
					expect_level_at_least=3,
					eta=0.6,
					corr_threshold=0.3,
				)
			)
			results.append(
				run_anchor(
					"ext_hotspot_validation_g008_eps02_k002_N500",
					props,
					stage2_method="permutation_p",
					expect_level_at_least=3,
					eta=0.6,
					corr_threshold=0.3,
					permutation_resamples=200,
					permutation_seed=123,
				)
			)

	text = build_markdown(results)
	args.out.parent.mkdir(parents=True, exist_ok=True)
	args.out.write_text(text, encoding="utf-8")
	print(f"Wrote: {args.out}")


if __name__ == "__main__":
	main()
