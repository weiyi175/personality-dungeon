"""Clean and merge low-band rho-curve sweep CSVs.

Motivation
- Long sweeps may be interrupted and later resumed with append, which can leave
  truncated/misaligned rows in the CSV.
- For finite-size scaling we only need a consistent set of (players, k) points
  with the required numeric fields.

This script filters to a target players set and k-range, drops malformed rows,
deduplicates by (players, k) keeping the row with the largest n_seeds, and
writes a clean CSV containing a stable core column set.

Example
  python -m analysis.clean_rho_curve_lowband \
    --in outputs/sweeps/rho_curve/rho_curve_lowband_corr0p09_..._N300_500_...csv \
    --in outputs/sweeps/rho_curve/rho_curve_lowband_corr0p09_..._N500_...csv \
    --players 300,500 \
    --k-min 0.18 \
    --k-max-by-players 300=0.44,500=0.42 \
    --out outputs/sweeps/rho_curve/rho_curve_lowband_corr0p09_clean_N300_500.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


CORE_COLS: list[str] = [
	"players",
	"a",
	"b",
	"selection_strength",
	"rho",
	"rho_minus_1",
	"p_level_ge_2",
	"p_level_3",
	"mean_stage3_score",
	"mean_max_amp",
	"mean_max_corr",
	"mean_env_gamma",
	"mean_env_gamma_r2",
	"mean_env_gamma_n_peaks",
	"n_seeds",
]


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


def _parse_players(spec: str) -> set[int]:
	out: set[int] = set()
	for part in spec.split(","):
		part = part.strip()
		if not part:
			continue
		out.add(int(part))
	if not out:
		raise ValueError("--players is empty")
	return out


def _parse_k_max_by_players(spec: str) -> Dict[int, float]:
	"""Parse e.g. '300=0.44,500=0.42'."""
	out: Dict[int, float] = {}
	if not spec.strip():
		return out
	for part in spec.split(","):
		part = part.strip()
		if not part:
			continue
		if "=" not in part:
			raise ValueError("--k-max-by-players entries must be like 'N=kmax'")
		n_str, k_str = [x.strip() for x in part.split("=", 1)]
		n = int(n_str)
		kmax = float(k_str)
		out[n] = kmax
	return out


@dataclass(frozen=True)
class CleanRow:
	n_seeds: int
	players: int
	k: float
	data: dict[str, str]


def _iter_rows(paths: Sequence[Path]) -> Iterable[dict[str, str]]:
	for path in paths:
		with path.open(newline="") as f:
			r = csv.DictReader(f)
			for row in r:
				yield row


def clean_lowband(
	*,
	inputs: Sequence[Path],
	players_allow: set[int],
	k_min: float,
	k_max_by_players: Dict[int, float],
	default_k_max: float,
) -> List[dict[str, str]]:
	best: Dict[Tuple[int, float], CleanRow] = {}

	for row in _iter_rows(inputs):
		n = _try_int(row.get("players"))
		k = _try_float(row.get("selection_strength"))
		if n is None or k is None:
			continue
		if n not in players_allow:
			continue
		k_max = float(k_max_by_players.get(n, default_k_max))
		if not (k_min - 1e-12 <= k <= k_max + 1e-12):
			continue

		# Required numeric fields for scaling.
		if _try_float(row.get("rho_minus_1")) is None:
			continue
		if _try_float(row.get("p_level_3")) is None:
			continue

		n_seeds = _try_int(row.get("n_seeds"))
		if n_seeds is None:
			n_seeds = 0

		key = (n, round(k, 10))
		cur = best.get(key)
		if cur is None or n_seeds > cur.n_seeds:
			data = {c: str(row.get(c, "")) for c in CORE_COLS}
			data["players"] = str(n)
			data["selection_strength"] = str(round(k, 10))
			best[key] = CleanRow(n_seeds=n_seeds, players=n, k=round(k, 10), data=data)

	rows_out = [v.data for v in sorted(best.values(), key=lambda x: (x.players, x.k))]
	return rows_out


def main(argv: Optional[Sequence[str]] = None) -> None:
	p = argparse.ArgumentParser(description="Clean/merge lowband rho-curve sweep CSVs")
	p.add_argument("--in", dest="inputs", action="append", required=True, help="Input CSV path (repeatable)")
	p.add_argument("--out", type=str, required=True, help="Output CSV path")
	p.add_argument("--players", type=str, required=True, help="Allowed players list, e.g. '300,500'")
	p.add_argument("--k-min", type=float, required=True, help="Minimum k to keep")
	p.add_argument(
		"--k-max-by-players",
		type=str,
		default="",
		help="Per-players k_max overrides, e.g. '300=0.44,500=0.42'",
	)
	p.add_argument("--default-k-max", type=float, default=1e9, help="Default k_max if not specified per players")
	args = p.parse_args(list(argv) if argv is not None else None)

	inputs = [Path(x) for x in args.inputs]
	rows = clean_lowband(
		inputs=inputs,
		players_allow=_parse_players(args.players),
		k_min=float(args.k_min),
		k_max_by_players=_parse_k_max_by_players(args.k_max_by_players),
		default_k_max=float(args.default_k_max),
	)

	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open("w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=CORE_COLS)
		w.writeheader()
		w.writerows(rows)

	byN: Dict[int, int] = {}
	for r in rows:
		byN[int(float(r["players"]))] = byN.get(int(float(r["players"])), 0) + 1
	print(f"Wrote clean CSV: {out_path} (rows={len(rows)} byN={byN})")


if __name__ == "__main__":
	main()
