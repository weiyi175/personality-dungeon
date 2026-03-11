"""Select "noisy" N (players) candidates for rho-curve crossing refinement.

Goal
- Reduce manual, subjective selection of which N to refine.
- Keep stdlib-only (fits repo research constraints).

Two modes
1) Summary mode (fast): ranks by indicators available in *_summary.csv.
2) Sweep mode (richer): ranks by monotonicity/roughness computed from raw sweep CSV(s).

This lives in analysis/ and does not import simulation/.
"""

from __future__ import annotations

import argparse
import csv
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


def _expand(paths: Iterable[str]) -> list[Path]:
	out: list[Path] = []
	for p in paths:
		pp = str(p).strip()
		if not pp:
			continue
		# Support globs for convenience.
		matches = sorted(glob.glob(pp))
		if matches:
			out.extend(Path(m) for m in matches)
		else:
			out.append(Path(pp))
	# De-dup while preserving order.
	seen: set[str] = set()
	uniq: list[Path] = []
	for p in out:
		k = str(p)
		if k in seen:
			continue
		seen.add(k)
		uniq.append(p)
	return uniq


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
		# allow float-ish ints
		return int(float(s))
	except Exception:
		return None


@dataclass(frozen=True)
class SummaryScore:
	players: int
	score: float
	reason: str


def _rank_from_summary(summary_csv: Path) -> list[SummaryScore]:
	rows: list[SummaryScore] = []
	with summary_csv.open("r", newline="") as f:
		reader = csv.DictReader(f)
		for r in reader:
			players = _try_int(r.get("players"))
			if players is None:
				continue

			corr = _try_float(r.get("corr_rho_minus_1_p_level_3"))
			k50 = _try_float(r.get("k50"))
			k50_censored = str(r.get("k50_censored_low") or "").strip().lower() in ("1", "true", "yes")
			k_first_pos = _try_float(r.get("k_first_positive"))
			rho_m1_first_pos = _try_float(r.get("rho_minus_1_first_positive"))
			rho_m1_50 = _try_float(r.get("rho_minus_1_50")) or _try_float(r.get("rho_m1_50"))

			# Heuristic: "noisy" often manifests as weaker rho->P correlation and/or unstable crossings.
			# We build a score where higher means "more suspicious / worth refining".
			s = 0.0
			reasons: list[str] = []

			if corr is None:
				s += 1.0
				reasons.append("missing corr")
			else:
				# lower corr => noisier
				s += max(0.0, 1.0 - float(corr))
				reasons.append(f"corr={corr:.3g}")

			if k50 is None:
				s += 1.0
				reasons.append("missing k50")
			else:
				reasons.append(f"k50={k50:.4g}")

			if k50_censored:
				s += 1.0
				reasons.append("k50 censored")

			# If the crossing happens very close to the linear boundary (rho-1 ~ 0), small noise can flip outcomes.
			if rho_m1_50 is not None:
				s += 0.25 * (1.0 / (1e-6 + abs(float(rho_m1_50))))  # larger when very close to 0
				reasons.append(f"rho-1@50={rho_m1_50:+.3g}")
			elif rho_m1_first_pos is not None:
				s += 0.10 * (1.0 / (1e-6 + abs(float(rho_m1_first_pos))))
				reasons.append(f"rho-1@first+={rho_m1_first_pos:+.3g}")

			# If first positive occurs at k_min, it suggests the sweep starts inside the transition band.
			if k_first_pos is not None and _try_float(r.get("k_min")) is not None:
				k_min = float(r.get("k_min") or 0.0)
				if abs(float(k_first_pos) - k_min) <= 1e-12:
					s += 0.5
					reasons.append("first positive at k_min")

			rows.append(SummaryScore(players=int(players), score=float(s), reason="; ".join(reasons)))

	rows.sort(key=lambda x: x.score, reverse=True)
	return rows


@dataclass
class SweepStats:
	players: int
	monotone_violations: int
	sign_changes: int
	crossings: int
	points: int
	score: float


def _rank_from_sweeps(sweep_csvs: list[Path], *, target: float, eps: float) -> list[SweepStats]:
	# Gather per players: list of (k, p3)
	by: dict[int, list[tuple[float, float]]] = {}
	for path in sweep_csvs:
		with path.open("r", newline="") as f:
			reader = csv.DictReader(f)
			for r in reader:
				players = _try_int(r.get("players"))
				k = _try_float(r.get("selection_strength"))
				y = _try_float(r.get("p_level_3"))
				if players is None or k is None or y is None:
					continue
				by.setdefault(int(players), []).append((float(k), float(y)))

	out: list[SweepStats] = []
	for players, pts in by.items():
		pts.sort(key=lambda kv: kv[0])
		y = [p[1] for p in pts]
		if len(y) < 5:
			continue

		monov = 0
		dys: list[float] = []
		for i in range(len(y) - 1):
			dy = float(y[i + 1]) - float(y[i])
			dys.append(dy)
			if dy < -float(eps):
				monov += 1

		# Sign-change count in dy (ignoring tiny moves)
		signs: list[int] = []
		for dy in dys:
			if abs(float(dy)) <= float(eps):
				continue
			signs.append(1 if dy > 0.0 else -1)
		sc = 0
		for i in range(len(signs) - 1):
			if signs[i] != signs[i + 1]:
				sc += 1

		# Count crossings around the target value.
		cross = 0
		for i in range(len(y) - 1):
			a = float(y[i]) - float(target)
			b = float(y[i + 1]) - float(target)
			if a == 0.0:
				continue
			if (a < 0.0 and b > 0.0) or (a > 0.0 and b < 0.0):
				cross += 1

		# Composite score: more monotone violations and more oscillation around target => noisier.
		score = (2.0 * float(monov)) + (1.0 * float(cross)) + (0.25 * float(sc))
		out.append(
			SweepStats(
				players=int(players),
				monotone_violations=int(monov),
				sign_changes=int(sc),
				crossings=int(cross),
				points=int(len(y)),
				score=float(score),
			)
		)

	out.sort(key=lambda s: s.score, reverse=True)
	return out


def main(argv: Optional[list[str]] = None) -> None:
	p = argparse.ArgumentParser(description="Select noisy N(players) for rho-curve refinement")
	p.add_argument("--summary", type=str, default=None, help="Path to *_summary.csv (fast heuristic ranking)")
	p.add_argument(
		"--in",
		dest="inputs",
		action="append",
		default=[],
		help="Input sweep CSV path or glob (repeatable); enables sweep-based ranking",
	)
	p.add_argument("--top", type=int, default=3, help="How many players values to output")
	p.add_argument(
		"--mode",
		choices=["summary", "sweeps"],
		default=None,
		help="Force ranking mode; default picks sweeps if --in provided else summary",
	)
	p.add_argument("--target", type=float, default=0.5, help="Crossing target used in sweep-based crossing count")
	p.add_argument("--eps", type=float, default=1e-9, help="Tolerance for monotone/sign checks")
	p.add_argument(
		"--format",
		choices=["space", "csv", "tsv"],
		default="space",
		help="Output format for selected players (default: space-separated)",
	)
	args = p.parse_args(list(argv) if argv is not None else None)

	mode = args.mode
	in_paths = _expand(args.inputs)
	if mode is None:
		mode = "sweeps" if in_paths else "summary"

	top = max(1, int(args.top))
	selected: list[int] = []

	if mode == "sweeps":
		if not in_paths:
			raise SystemExit("mode=sweeps requires at least one --in")
		stats = _rank_from_sweeps(in_paths, target=float(args.target), eps=float(args.eps))
		selected = [s.players for s in stats[:top]]
		# Human-readable diagnostic table to stderr-like stdout (still fine for CLI use).
		print("players\tscore\tmonov\tcross\tsignchg\tpoints")
		for s in stats[: min(len(stats), max(10, top))]:
			print(f"{s.players}\t{s.score:.3g}\t{s.monotone_violations}\t{s.crossings}\t{s.sign_changes}\t{s.points}")
	else:
		if args.summary is None:
			raise SystemExit("mode=summary requires --summary")
		summary_path = Path(str(args.summary))
		rows = _rank_from_summary(summary_path)
		selected = [r.players for r in rows[:top]]
		print("players\tscore\treason")
		for r in rows[: min(len(rows), max(10, top))]:
			print(f"{r.players}\t{r.score:.3g}\t{r.reason}")

	# Final line: a shell-friendly list.
	if args.format == "csv":
		print(",".join(str(x) for x in selected))
	elif args.format == "tsv":
		print("\t".join(str(x) for x in selected))
	else:
		print(" ".join(str(x) for x in selected))


if __name__ == "__main__":
	main()
