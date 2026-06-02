#!/usr/bin/env python3
"""Generate a structured signoff summary JSON from a matrix output directory.

Usage:
    ./venv/bin/python scripts/experiments/generate_signoff_summary.py \
        --matrix-out reports/experiments/phase05_matrix_full \
        --expected-rows 12 \
        --phase phase06 \
        --out artifacts/signoff_summary.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate signoff summary JSON.")
    parser.add_argument("--matrix-out", required=True, help="Directory containing run_*.json files")
    parser.add_argument("--expected-rows", type=int, required=True, help="Expected number of runs")
    parser.add_argument("--phase", required=True, help="Phase label, e.g. phase06")
    parser.add_argument("--out", required=True, help="Output path for signoff_summary.json")
    args = parser.parse_args()

    out_dir = Path(args.matrix_out)
    run_files = sorted(out_dir.glob("run_*.json"))
    actual_rows = len(run_files)
    count_match = actual_rows == args.expected_rows

    rewards: list[float] = []
    latencies: list[float] = []

    for run_file in run_files:
        payload = json.loads(run_file.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        if "reward_mean" in summary:
            rewards.append(summary["reward_mean"])
        if "avg_latency_ms" in summary:
            latencies.append(summary["avg_latency_ms"])

    gate_pass = count_match and bool(rewards)

    result = {
        "phase": args.phase,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matrix_rows": actual_rows,
        "expected_rows": args.expected_rows,
        "count_match": count_match,
        "reward_range": {
            "min": round(min(rewards), 6) if rewards else None,
            "max": round(max(rewards), 6) if rewards else None,
        },
        "latency_range": {
            "min": round(min(latencies), 4) if latencies else None,
            "max": round(max(latencies), 4) if latencies else None,
        },
        "gate_pass": gate_pass,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"gate_pass={gate_pass}  matrix_rows={actual_rows}  out={out_path}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
