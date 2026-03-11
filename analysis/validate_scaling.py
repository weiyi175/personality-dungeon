"""Validate and compare two rho-curve scaling runs.

This is a lightweight, stdlib-only checker to quantify improvement after a
refinement iteration.

Inputs are two prefixes that exist under an output directory:
- <prefix>_summary.csv
- <prefix>_k50_fit.json (optional; if missing, delta is skipped)

Outputs
- delta r2, delta k_inf, delta c (when fit json exists)
- per-N k50 deltas (when present)

This lives in analysis/ and does not import simulation/.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional


def _load_fit(path: Path) -> dict | None:
	if not path.exists():
		return None
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return None


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


def _load_k50_map(summary_csv: Path) -> dict[int, float]:
	out: dict[int, float] = {}
	with summary_csv.open("r", newline="") as f:
		reader = csv.DictReader(f)
		for r in reader:
			players_s = r.get("players")
			k50 = _try_float(r.get("k50"))
			if players_s is None or k50 is None:
				continue
			try:
				players = int(float(str(players_s).strip()))
			except Exception:
				continue
			out[int(players)] = float(k50)
	return out


def _load_players_present(summary_csv: Path) -> set[int]:
	players: set[int] = set()
	with summary_csv.open("r", newline="") as f:
		reader = csv.DictReader(f)
		for r in reader:
			players_s = r.get("players")
			if players_s is None:
				continue
			try:
				players.add(int(float(str(players_s).strip())))
			except Exception:
				continue
	return players


def main(argv: Optional[list[str]] = None) -> None:
	p = argparse.ArgumentParser(description="Compare two rho-curve scaling outputs")
	p.add_argument("--outdir", type=str, default="outputs/analysis/rho_curve", help="Output directory containing prefixes")
	p.add_argument("--old", type=str, required=True, help="Old prefix (baseline)")
	p.add_argument("--new", type=str, required=True, help="New prefix (candidate)")
	args = p.parse_args(list(argv) if argv is not None else None)

	outdir = Path(str(args.outdir))
	old = str(args.old)
	new = str(args.new)

	old_fit = _load_fit(outdir / f"{old}_k50_fit.json")
	new_fit = _load_fit(outdir / f"{new}_k50_fit.json")

	print("=== scaling fit delta ===")
	if old_fit is None:
		print(f"missing: {outdir / (old + '_k50_fit.json')}")
	if new_fit is None:
		print(f"missing: {outdir / (new + '_k50_fit.json')}")
	if old_fit is not None and new_fit is not None:
		for key in ("r2", "k_inf", "c"):
			o = _try_float(old_fit.get(key))
			n = _try_float(new_fit.get(key))
			if o is None or n is None:
				continue
			print(f"{key}: {o:.6g} -> {n:.6g}  (delta {n - o:+.6g})")

	print("\n=== per-N k50 delta ===")
	old_sum = outdir / f"{old}_summary.csv"
	new_sum = outdir / f"{new}_summary.csv"
	if not old_sum.exists():
		raise SystemExit(f"missing summary: {old_sum}")
	if not new_sum.exists():
		raise SystemExit(f"missing summary: {new_sum}")

	old_players_present = _load_players_present(old_sum)
	new_players_present = _load_players_present(new_sum)
	old_map = _load_k50_map(old_sum)
	new_map = _load_k50_map(new_sum)
	players_all = sorted(old_players_present | new_players_present)
	missing_old = sorted(old_players_present - set(old_map.keys()))
	missing_new = sorted(new_players_present - set(new_map.keys()))
	if missing_old or missing_new:
		print("(note) some rows have missing/blank k50 and are excluded from deltas")
		if missing_old:
			print(f"missing k50 in OLD: {missing_old}")
		if missing_new:
			print(f"missing k50 in NEW: {missing_new}")
	print("")
	print("players\tk50_old\tk50_new\tdelta")
	for n in players_all:
		o = old_map.get(n)
		nv = new_map.get(n)
		if o is None or nv is None:
			# keep output compact; missing k50 is already summarized above
			continue
		d = float(nv) - float(o)
		print(f"{n}\t{o:.6g}\t{nv:.6g}\t{d:+.6g}")


if __name__ == "__main__":
	main()
