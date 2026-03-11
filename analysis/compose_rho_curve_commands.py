"""Compose rho-curve scaling/report commands by auto-assembling --in lists.

Purpose
- Reduce manual errors when maintaining long --in lists across refinement iterations.
- Keep stdlib-only and remain in analysis/ (must not import simulation/).
- This script PRINTS commands only; it does not run scaling/report.

Typical use
- Provide base sweeps (main/ext/lowband) explicitly.
- Provide a refinement TAG (from scripts/run_refinement_sweep.sh) or explicit globs.
- Optionally select latest-per-N refinement file to avoid mixing multiple runs.

Example
  ./venv/bin/python -m analysis.compose_rho_curve_commands \
    --prefix rho_curve_merged_with_lowband_clean_plus_refine2 \
    --base-in outputs/sweeps/rho_curve/rho_curve_a0p4_b0p2425407_lag1_sampled_eta055_R4000_S30_N50_200_1000_k0p1_1p0_s0p02.csv \
    --base-in outputs/sweeps/rho_curve/rho_curve_a0p4_b0p2425407_lag1_sampled_eta055_R4000_S30_N200_k1p0_1p4_s0p02.csv \
    --lowband outputs/sweeps/rho_curve/rho_curve_lowband_corr0p09_clean_N300_500.csv \
    --refine-tag prov2_20260308_123456
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


_N_RE = re.compile(r"_N(?P<n>\d+)_")


def _expand_paths(items: Iterable[str]) -> list[Path]:
	out: list[Path] = []
	for raw in items:
		s = str(raw).strip()
		if not s:
			continue
		matches = sorted(glob.glob(s))
		if matches:
			out.extend(Path(m) for m in matches)
		else:
			out.append(Path(s))
	# de-dup, preserve order
	seen: set[str] = set()
	uniq: list[Path] = []
	for p in out:
		k = str(p)
		if k in seen:
			continue
		seen.add(k)
		uniq.append(p)
	return uniq


def _shell_quote(s: str) -> str:
	# Safe single-quote wrapper for bash.
	if s == "":
		return "''"
	return "'" + s.replace("'", "'\\''") + "'"


def _parse_n_from_name(path: Path) -> Optional[int]:
	m = _N_RE.search(path.name)
	if not m:
		return None
	try:
		return int(m.group("n"))
	except Exception:
		return None


@dataclass(frozen=True)
class Inputs:
	base: list[Path]
	refine: list[Path]
	lowband: Optional[Path]


def _select_latest_per_n(refine_files: list[Path]) -> list[Path]:
	# Group by parsed N when possible; choose newest mtime per N.
	by_n: dict[int, list[Path]] = {}
	no_n: list[Path] = []
	for p in refine_files:
		n = _parse_n_from_name(p)
		if n is None:
			no_n.append(p)
			continue
		by_n.setdefault(n, []).append(p)

	selected: list[Path] = []
	for n, ps in sorted(by_n.items(), key=lambda kv: kv[0]):
		ps_sorted = sorted(ps, key=lambda x: (x.stat().st_mtime, str(x)), reverse=True)
		selected.append(ps_sorted[0])

	# Keep unparsed files at the end (deterministic order).
	selected.extend(sorted(no_n, key=lambda p: str(p)))
	return selected


def _gather_inputs(
	*,
	base_in: list[str],
	lowband: Optional[str],
	refine_tag: Optional[str],
	refine_globs: list[str],
	choose: str,
) -> Inputs:
	base = _expand_paths(base_in)
	lb = Path(lowband) if lowband else None

	refine_patterns: list[str] = []
	if refine_globs:
		refine_patterns.extend(refine_globs)
	if refine_tag:
		refine_patterns.append(f"outputs/sweeps/rho_curve/rho_curve_refine_crossing_{refine_tag}_*.csv")

	refine_files = _expand_paths(refine_patterns)

	# Filter to existing files (globs should already), but keep explicit paths only if they exist.
	base_missing = [p for p in base if not p.exists()]
	if base_missing:
		raise SystemExit("Missing base inputs:\n" + "\n".join(f"- {p}" for p in base_missing))
	if lb is not None and not lb.exists():
		raise SystemExit(f"Missing lowband file: {lb}")

	refine_existing = [p for p in refine_files if p.exists()]
	if refine_patterns and not refine_existing:
		raise SystemExit(
			"No refinement CSVs matched.\n"
			+ "Patterns:\n"
			+ "\n".join(f"- {pat}" for pat in refine_patterns)
		)
	if not refine_patterns:
		raise SystemExit("Need --refine-tag or at least one --refine-glob")

	if choose == "latest-per-n":
		refine_existing = _select_latest_per_n(refine_existing)
	elif choose == "all":
		refine_existing = sorted(refine_existing, key=lambda p: str(p))
	else:
		raise SystemExit(f"Unknown choose mode: {choose}")

	return Inputs(base=base, refine=refine_existing, lowband=lb)


def _compose_in_list(inputs: Inputs) -> list[Path]:
	out: list[Path] = []
	out.extend(inputs.base)
	out.extend(inputs.refine)
	if inputs.lowband is not None:
		out.append(inputs.lowband)
	# De-dup while preserving order
	seen: set[str] = set()
	uniq: list[Path] = []
	for p in out:
		k = str(p)
		if k in seen:
			continue
		seen.add(k)
		uniq.append(p)
	return uniq


def _print_scaling(
	*,
	python_bin: str,
	module: str,
	outdir: Path,
	prefix: str,
	in_list: list[Path],
) -> list[str]:
	lines: list[str] = []
	lines.append("# (B) scaling (auto-composed --in list)")
	lines.append(f"{python_bin} -m {module} \\")
	for p in in_list:
		lines.append(f"  --in {_shell_quote(str(p))} \\")
	lines.append(f"  --outdir {_shell_quote(str(outdir))} \\")
	lines.append(f"  --prefix {_shell_quote(prefix)}")
	return lines


def _print_report(
	*,
	python_bin: str,
	module: str,
	outdir: Path,
	prefix: str,
	in_list: list[Path],
) -> list[str]:
	summary = outdir / f"{prefix}_summary.csv"
	fit_json = outdir / f"{prefix}_k50_fit.json"
	notes_out = outdir / f"{prefix}_report.md"
	paper_out = outdir / f"{prefix}_paper.md"

	lines: list[str] = []
	lines.append("# (C) report (auto-composed --in list)")
	lines.append(f"{python_bin} -m {module} \\")
	lines.append(f"  --summary {_shell_quote(str(summary))} \\")
	lines.append(f"  --fit-json {_shell_quote(str(fit_json))} \\")
	for p in in_list:
		lines.append(f"  --in {_shell_quote(str(p))} \\")
	lines.append(f"  --out {_shell_quote(str(notes_out))} \\")
	lines.append(f"  --out-paper {_shell_quote(str(paper_out))}")
	return lines


def main(argv: Optional[list[str]] = None) -> None:
	p = argparse.ArgumentParser(description="Compose rho-curve scaling/report commands (prints only)")
	p.add_argument("--prefix", type=str, required=True, help="Output prefix for scaling/report")
	p.add_argument(
		"--outdir",
		type=str,
		default="outputs/analysis/rho_curve",
		help="Directory where scaling/report outputs live",
	)
	p.add_argument(
		"--python-bin",
		type=str,
		default="./venv/bin/python",
		help="Python executable to print in composed commands",
	)
	p.add_argument(
		"--scaling-module",
		type=str,
		default="analysis.rho_curve_scaling",
		help="Module path for scaling entrypoint",
	)
	p.add_argument(
		"--report-module",
		type=str,
		default="analysis.rho_curve_report",
		help="Module path for report entrypoint",
	)
	p.add_argument(
		"--base-in",
		action="append",
		default=[],
		help="Base sweep CSV path or glob (repeatable). Order is preserved.",
	)
	p.add_argument(
		"--lowband",
		type=str,
		default=None,
		help="Optional lowband clean CSV to append at the end",
	)
	p.add_argument(
		"--refine-tag",
		type=str,
		default=None,
		help="Refinement TAG (matches outputs/sweeps/rho_curve/rho_curve_refine_crossing_<TAG>_*.csv)",
	)
	p.add_argument(
		"--refine-glob",
		action="append",
		default=[],
		help="Refinement CSV glob(s) (repeatable)",
	)
	p.add_argument(
		"--choose",
		choices=["latest-per-n", "all"],
		default="latest-per-n",
		help="How to handle multiple refinement files per N",
	)
	p.add_argument(
		"--mode",
		choices=["both", "scaling", "report", "in-list"],
		default="both",
		help="What to print",
	)
	p.add_argument(
		"--write",
		type=str,
		default=None,
		help="Optional path to write the composed commands (useful to avoid terminal line-wrapping when copy/pasting)",
	)
	args = p.parse_args(list(argv) if argv is not None else None)

	inputs = _gather_inputs(
		base_in=list(args.base_in),
		lowband=args.lowband,
		refine_tag=args.refine_tag,
		refine_globs=list(args.refine_glob),
		choose=str(args.choose),
	)
	in_list = _compose_in_list(inputs)

	outdir = Path(str(args.outdir))
	prefix = str(args.prefix)
	python_bin = str(args.python_bin)

	lines: list[str] = []
	# Headline summary
	lines.append("# === compose_rho_curve_commands ===")
	lines.append(f"# cwd={_shell_quote(os.getcwd())}")
	lines.append(f"# outdir={_shell_quote(str(outdir))}")
	lines.append(f"# prefix={_shell_quote(prefix)}")
	lines.append(f"# base={len(inputs.base)} refine={len(inputs.refine)} lowband={'1' if inputs.lowband else '0'}")
	lines.append(f"# choose={_shell_quote(str(args.choose))}")

	if args.mode in ("in-list",):
		for pth in in_list:
			lines.append(str(pth))
	else:
		if args.mode in ("both", "scaling"):
			lines.extend(
				_print_scaling(
					python_bin=python_bin,
					module=str(args.scaling_module),
					outdir=outdir,
					prefix=prefix,
					in_list=in_list,
				)
			)
			lines.append("")

		if args.mode in ("both", "report"):
			lines.extend(
				_print_report(
					python_bin=python_bin,
					module=str(args.report_module),
					outdir=outdir,
					prefix=prefix,
					in_list=in_list,
				)
			)

	text = "\n".join(lines).rstrip() + "\n"
	if args.write:
		out_path = Path(str(args.write))
		out_path.parent.mkdir(parents=True, exist_ok=True)
		out_path.write_text(text, encoding="utf-8")
	print(text, end="")


if __name__ == "__main__":
	main()
