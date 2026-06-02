#!/usr/bin/env python3
"""Run a matrix of experiments from CSV.

Expected columns: seed,w,burn_in,n_rounds
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_experiment import run_experiment


def _safe_token(value: str) -> str:
    return value.strip().replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run experiment matrix from CSV.")
    parser.add_argument("--matrix", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--log", type=str, default="reports/experiments/matrix_run_log.txt")
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with matrix_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    with log_path.open("w", encoding="utf-8") as log:
        for idx, row in enumerate(rows):
            seed = int(row.get("seed", "0"))
            n_rounds = int(row.get("n_rounds", "200"))
            burn_in = int(row.get("burn_in", "50"))
            w = row.get("w", "na")

            token = f"seed{seed}_w{_safe_token(str(w))}_burn{burn_in}_rounds{n_rounds}"
            out_path = out_dir / f"run_{token}.json"

            payload = run_experiment(seed=seed, runs=1, n_steps=n_rounds)
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            log.write(f"{idx+1}/{len(rows)} wrote {out_path}\n")

    print(f"Wrote {len(rows)} runs. Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
