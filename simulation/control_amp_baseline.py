"""Control baseline amplitude cache builder (one-shot pipeline).

This script automates the full flow needed by SDD 4.7(1):

1) Run a control baseline (matrix_ab with a=b=0) across seeds and optionally players-grid.
2) Write a per-seed CSV that contains `players` and `max_amp`.
3) Build a JSON control cache compatible with `simulation.rho_curve --amplitude-control-cache`.

Rationale
- `analysis.amplitude_control_cache` expects a per-seed-like CSV with `max_amp`.
- `simulation.seed_stability` already computes `max_amp` correctly over a windowed segment.
- This CLI removes manual merge / multi-step glue.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from analysis.amplitude_control_cache import build_amplitude_control_cache_from_csv
from simulation.seed_stability import _auto_jobs, _parse_int_grid, _parse_seeds, _run_one


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else [
            "players",
            "seed",
            "selection_strength",
            "series",
            "burn_in",
            "tail",
            "max_amp",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build control per-seed max_amp CSV + JSON cache for amplitude normalization"
    )

    # Protocol (must match rho_curve usage; cache loader enforces these).
    p.add_argument("--series", choices=["p", "w"], required=True)
    p.add_argument("--rounds", type=int, required=True)
    p.add_argument("--seeds", type=str, required=True, help='Seed spec: "0:29" or "1,2,3"')
    p.add_argument("--players", type=int, default=None)
    p.add_argument(
        "--players-grid",
        type=str,
        default=None,
        help='Optional players grid: "100,300,1000" or "100:1000:200"',
    )
    p.add_argument("--burn-in", type=int, default=None)
    p.add_argument("--burn-in-frac", type=float, default=0.3)
    p.add_argument("--tail", type=int, default=None)
    p.add_argument(
        "--selection-strength",
        type=float,
        default=0.02,
        help="Selection strength k (kept for protocol alignment; control has U=0 so result is typically k-invariant).",
    )
    p.add_argument(
        "--popularity-mode",
        type=str,
        default="sampled",
        choices=["sampled", "expected"],
    )
    p.add_argument(
        "--evolution-mode",
        type=str,
        default="sampled",
        choices=["sampled", "mean_field"],
    )
    p.add_argument("--payoff-lag", type=int, default=1, choices=[0, 1])
    p.add_argument("--memory-kernel", type=int, default=1, choices=[1, 3, 5])
    p.add_argument("--init-bias", type=float, default=0.0)
    p.add_argument("--jobs", type=int, default=0, help="Worker processes. 0 => auto (cpu_count-1)")

    # Outputs
    p.add_argument(
        "--out-per-seed",
        type=Path,
        required=True,
        help="Output per-seed CSV path (contains players,max_amp,...).",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        required=True,
        help="Output JSON cache path (compatible with simulation.rho_curve).",
    )
    p.add_argument(
        "--ddof",
        type=int,
        default=1,
        help="Std ddof used in JSON (1=sample std, 0=population std).",
    )
    args = p.parse_args()

    rounds = int(args.rounds)
    if rounds <= 0:
        raise ValueError("--rounds must be > 0")

    seed_list = _parse_seeds(str(args.seeds))
    if not seed_list:
        raise ValueError("--seeds resolved to empty")

    burn_in = args.burn_in
    if burn_in is None:
        burn_in = int(round(float(rounds) * float(args.burn_in_frac)))
    burn_in = max(0, int(burn_in))

    tail: int | None = args.tail
    if tail is not None:
        tail = int(tail)
        if tail < 0:
            raise ValueError("--tail must be >= 0")

    if args.players_grid is not None:
        players_vals = _parse_int_grid(str(args.players_grid))
        if not players_vals:
            raise ValueError("--players-grid resolved to empty")
    else:
        if args.players is None:
            raise ValueError("Provide --players or --players-grid")
        players_vals = [int(args.players)]

    jobs = int(args.jobs)
    if jobs == 0:
        jobs = _auto_jobs()
    if jobs < 1:
        raise ValueError("--jobs must be >= 0")

    payload = {
        "payoff_mode": "matrix_ab",
        "gamma": 0.0,
        "epsilon": 0.0,
        "a": 0.0,
        "b": 0.0,
        "init_bias": float(args.init_bias),
        "popularity_mode": str(args.popularity_mode),
        "evolution_mode": str(args.evolution_mode),
        "payoff_lag": int(args.payoff_lag),
        "memory_kernel": int(args.memory_kernel),
        "key": None,
    }

    metric_kwargs: dict = {}
    out_rows: list[dict] = []

    if jobs == 1:
        for npl in players_vals:
            for sd in seed_list:
                (
                    _key,
                    lv,
                    s1,
                    s2,
                    s3,
                    s3_score,
                    s3_turn,
                    max_amp,
                    max_corr,
                    env_gamma,
                    env_r2,
                    env_peaks,
                ) = _run_one(
                    players=int(npl),
                    rounds=int(rounds),
                    seed=int(sd),
                    payload=payload,
                    series=str(args.series),
                    burn_in=int(burn_in),
                    tail=int(tail) if tail is not None else None,
                    selection_strength=float(args.selection_strength),
                    metric_kwargs=metric_kwargs,
                )
                out_rows.append(
                    {
                        "players": int(npl),
                        "seed": int(sd),
                        "selection_strength": float(args.selection_strength),
                        "series": str(args.series),
                        "burn_in": int(burn_in),
                        "tail": (int(tail) if tail is not None else ""),
                        "memory_kernel": int(args.memory_kernel),
                        "cycle_level": int(lv),
                        "stage1_passed": bool(s1),
                        "stage2_passed": bool(s2),
                        "stage3_passed": bool(s3),
                        "stage3_score": float(s3_score),
                        "stage3_turn_strength": float(s3_turn),
                        "max_amp": float(max_amp),
                        "max_corr": float(max_corr),
                        "env_gamma": float(env_gamma),
                        "env_gamma_r2": float(env_r2),
                        "env_gamma_n_peaks": int(env_peaks),
                    }
                )
    else:
        with ProcessPoolExecutor(max_workers=int(jobs)) as ex:
            futs = {}
            for npl in players_vals:
                for sd in seed_list:
                    fut = ex.submit(
                        _run_one,
                        players=int(npl),
                        rounds=int(rounds),
                        seed=int(sd),
                        payload=payload,
                        series=str(args.series),
                        burn_in=int(burn_in),
                        tail=int(tail) if tail is not None else None,
                        selection_strength=float(args.selection_strength),
                        metric_kwargs=metric_kwargs,
                    )
                    futs[fut] = (int(npl), int(sd))
            for fut in as_completed(futs):
                npl, sd = futs[fut]
                (
                    _key,
                    lv,
                    s1,
                    s2,
                    s3,
                    s3_score,
                    s3_turn,
                    max_amp,
                    max_corr,
                    env_gamma,
                    env_r2,
                    env_peaks,
                ) = fut.result()
                out_rows.append(
                    {
                        "players": int(npl),
                        "seed": int(sd),
                        "selection_strength": float(args.selection_strength),
                        "series": str(args.series),
                        "burn_in": int(burn_in),
                        "tail": (int(tail) if tail is not None else ""),
                        "memory_kernel": int(args.memory_kernel),
                        "cycle_level": int(lv),
                        "stage1_passed": bool(s1),
                        "stage2_passed": bool(s2),
                        "stage3_passed": bool(s3),
                        "stage3_score": float(s3_score),
                        "stage3_turn_strength": float(s3_turn),
                        "max_amp": float(max_amp),
                        "max_corr": float(max_corr),
                        "env_gamma": float(env_gamma),
                        "env_gamma_r2": float(env_r2),
                        "env_gamma_n_peaks": int(env_peaks),
                    }
                )

    # Deterministic row order for stable diffs.
    out_rows.sort(key=lambda r: (int(r["players"]), int(r["seed"])))

    _write_csv(Path(args.out_per_seed), out_rows)

    cache = build_amplitude_control_cache_from_csv(
        [Path(args.out_per_seed)],
        series=str(args.series),
        burn_in=int(burn_in),
        tail=int(tail) if tail is not None else None,
        selection_strength=float(args.selection_strength),
        ddof=int(args.ddof),
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_json).open("w", encoding="utf-8") as f:
        json.dump(cache.to_json_obj(), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    # Minimal human-friendly summary.
    print("=== control_amp_baseline ===")
    print(f"players: {players_vals[0]}..{players_vals[-1]} (n={len(players_vals)})")
    print(f"seeds: {seed_list[0]}..{seed_list[-1]} (n={len(seed_list)})")
    print(f"series={args.series} burn_in={burn_in} tail={tail}")
    print(f"Wrote per-seed CSV: {args.out_per_seed}")
    print(f"Wrote JSON cache:   {args.out_json}")


if __name__ == "__main__":
    main()
