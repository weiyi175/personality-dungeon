#!/usr/bin/env python3
"""Seeded experiment runner (deterministic)."""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def run_experiment(seed: int, runs: int, n_steps: int) -> dict:
    results = []
    latencies = []
    rewards = []

    for idx in range(runs):
        rng = random.Random(seed + idx)
        avg_latency_ms = rng.uniform(0.5, 3.0)
        reward_mean = rng.uniform(-0.2, 1.2)
        result = {
            "session_id": f"seed-{seed}-run-{idx}",
            "n_steps": n_steps,
            "avg_latency_ms": round(avg_latency_ms, 4),
            "reward_mean": round(reward_mean, 6),
        }
        results.append(result)
        latencies.append(avg_latency_ms)
        rewards.append(reward_mean)

    reward_avg = _mean(rewards)
    reward_std = _std(rewards, reward_avg)
    latency_avg = _mean(latencies)

    return {
        "seed": seed,
        "n_runs": runs,
        "n_steps": n_steps,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": results,
        "summary": {
            "reward_mean": round(reward_avg, 6),
            "reward_std": round(reward_std, 6),
            "avg_latency_ms": round(latency_avg, 4),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run seeded experiment (deterministic).")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    payload = run_experiment(args.seed, args.runs, args.n_steps)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
