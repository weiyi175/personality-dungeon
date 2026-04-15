#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.cycle_metrics import classify_cycle_level


def _load_series(csv_path: Path) -> dict[str, list[float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")

    required_cols = {"p_aggressive", "p_defensive", "p_balanced"}
    missing = required_cols - set(rows[0].keys())
    if missing:
        raise ValueError(f"CSV missing columns {sorted(missing)}: {csv_path}")

    return {
        "aggressive": [float(row["p_aggressive"]) for row in rows],
        "defensive": [float(row["p_defensive"]) for row in rows],
        "balanced": [float(row["p_balanced"]) for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose Level/score/turn_strength from simulation CSV files."
    )
    parser.add_argument("csv_paths", nargs="+", help="One or more CSV files to diagnose.")
    parser.add_argument("--burn-in", type=int, default=2400)
    parser.add_argument("--tail", type=int, default=1000)
    args = parser.parse_args()

    for raw_path in args.csv_paths:
        csv_path = Path(raw_path)
        if not csv_path.exists():
            print(f"{csv_path}\tERROR=file-not-found")
            continue

        try:
            series = _load_series(csv_path)
            result = classify_cycle_level(
                series,
                burn_in=args.burn_in,
                tail=args.tail,
                amplitude_threshold=0.02,
                corr_threshold=0.09,
                eta=0.55,
                stage3_method="turning",
                phase_smoothing=1,
                min_lag=2,
                max_lag=500,
            )
            stage3_score = 0.0
            stage3_turn = 0.0
            if result.stage3 is not None:
                stage3_score = float(result.stage3.score)
                stage3_turn = float(result.stage3.turn_strength)
            print(
                f"{csv_path}\tlevel={result.level}\t"
                f"score={stage3_score:.4f}\t"
                f"turn={stage3_turn:.6f}"
            )
        except Exception as exc:
            print(f"{csv_path}\tERROR={type(exc).__name__}:{exc}")


if __name__ == "__main__":
    main()
