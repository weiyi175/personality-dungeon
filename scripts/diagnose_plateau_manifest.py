#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.cycle_metrics import (
    assess_stage1_amplitude,
    assess_stage2_frequency,
    assess_stage3_direction,
    classify_cycle_level,
    phase_direction_consistency,
    phase_direction_consistency_turning,
)


STRATEGIES = ("aggressive", "defensive", "balanced")
CSV_COLUMNS = {
    "aggressive": "p_aggressive",
    "defensive": "p_defensive",
    "balanced": "p_balanced",
}


def _load_series(csv_path: Path) -> dict[str, list[float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")

    missing = {v for v in CSV_COLUMNS.values()} - set(rows[0].keys())
    if missing:
        raise ValueError(f"CSV missing columns {sorted(missing)}: {csv_path}")

    return {
        name: [float(row[column]) for row in rows]
        for name, column in CSV_COLUMNS.items()
    }


def _resolve_out(manifest_path: Path, out_value: str) -> Path:
    path = Path(out_value)
    if path.exists():
        return path
    candidate = manifest_path.parent / Path(out_value).name
    if candidate.exists():
        return candidate
    candidate = manifest_path.parent.parent / Path(out_value).name
    if candidate.exists():
        return candidate
    return path


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bool01(flag: bool) -> int:
    return 1 if flag else 0


def _strategy_best_corr(stage2) -> tuple[str, float, int]:
    best_name = ""
    best_corr = float("-inf")
    best_lag = 0
    for name in STRATEGIES:
        df = stage2.frequencies.get(name)
        corr = float(df.corr) if df is not None and df.lag is not None else 0.0
        lag = int(df.lag) if df is not None and df.lag is not None else 0
        if corr > best_corr:
            best_name = name
            best_corr = corr
            best_lag = lag
    if best_corr == float("-inf"):
        return "", 0.0, 0
    return best_name, float(best_corr), int(best_lag)


def _analyze_csv(csv_path: Path, args: argparse.Namespace) -> dict[str, object]:
    series = _load_series(csv_path)

    stage1 = assess_stage1_amplitude(
        series,
        burn_in=args.burn_in,
        tail=args.tail,
        threshold=args.amplitude_threshold,
        aggregation=args.amplitude_aggregation,
    )
    stage2 = assess_stage2_frequency(
        series,
        burn_in=args.burn_in,
        tail=args.tail,
        min_lag=args.min_lag,
        max_lag=args.max_lag,
        corr_threshold=args.corr_threshold,
        aggregation=args.freq_aggregation,
        stage2_method="autocorr_threshold",
    )
    classify = classify_cycle_level(
        series,
        burn_in=args.burn_in,
        tail=args.tail,
        amplitude_threshold=args.amplitude_threshold,
        amplitude_aggregation=args.amplitude_aggregation,
        min_lag=args.min_lag,
        max_lag=args.max_lag,
        corr_threshold=args.corr_threshold,
        freq_aggregation=args.freq_aggregation,
        eta=args.eta,
        min_turn_strength=args.min_turn_strength,
        stage3_method="turning",
        phase_smoothing=args.phase_smoothing,
    )
    stage3_turning = phase_direction_consistency_turning(
        series,
        burn_in=args.burn_in,
        tail=args.tail,
        eta=args.eta,
        min_turn_strength=args.min_turn_strength,
        phase_smoothing=args.phase_smoothing,
    )
    stage3_centroid = phase_direction_consistency(
        series,
        burn_in=args.burn_in,
        tail=args.tail,
        eta=args.eta,
        min_turn_strength=args.min_turn_strength,
    )
    stage3_rotation = assess_stage3_direction(
        series,
        burn_in=args.burn_in,
        tail=args.tail,
    )

    amp_values = [float(stage1.amplitudes[name]) for name in STRATEGIES]
    best_name, best_corr, best_lag = _strategy_best_corr(stage2)

    row: dict[str, object] = {
        "level": int(classify.level),
        "amp_passed": _bool01(stage1.passed),
        "amp_threshold": float(stage1.threshold),
        "amp_max": max(amp_values) if amp_values else 0.0,
        "amp_mean": _safe_mean(amp_values),
        "freq_passed": _bool01(stage2.passed),
        "freq_threshold": float(stage2.corr_threshold),
        "freq_best_strategy": best_name,
        "freq_best_corr": float(best_corr),
        "freq_best_lag": int(best_lag),
        "freq_effective_window_n": int(stage2.effective_window_n),
        "turning_passed": _bool01(stage3_turning.passed),
        "turning_direction": int(stage3_turning.direction),
        "turning_score": float(stage3_turning.score),
        "turning_strength": float(stage3_turning.turn_strength),
        "centroid_passed": _bool01(stage3_centroid.passed),
        "centroid_direction": int(stage3_centroid.direction),
        "centroid_score": float(stage3_centroid.score),
        "centroid_strength": float(stage3_centroid.turn_strength),
        "rotation_passed": _bool01(stage3_rotation.passed),
        "rotation_net": float(stage3_rotation.net_rotation),
        "rotation_direction": stage3_rotation.direction or "",
        "rotation_consistency": float(stage3_rotation.consistency),
        "rotation_used_steps": int(stage3_rotation.used_steps),
        "rotation_used_points": int(stage3_rotation.used_points),
    }

    for name in STRATEGIES:
        row[f"amp_{name}"] = float(stage1.amplitudes.get(name, 0.0))
        df = stage2.frequencies.get(name)
        row[f"corr_{name}"] = float(df.corr) if df is not None and df.lag is not None else 0.0
        row[f"lag_{name}"] = int(df.lag) if df is not None and df.lag is not None else 0

    return row


def _write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose plateau-like candidate manifests with stage1/stage2/stage3 details."
    )
    parser.add_argument("manifest", help="TSV manifest with columns seed, a, b, rounds, out")
    parser.add_argument("--out-dir", help="Directory for generated TSV reports; defaults next to the manifest")
    parser.add_argument("--burn-in", type=int, default=2400)
    parser.add_argument("--tail", type=int, default=1000)
    parser.add_argument("--amplitude-threshold", type=float, default=0.02)
    parser.add_argument("--amplitude-aggregation", default="any")
    parser.add_argument("--min-lag", type=int, default=2)
    parser.add_argument("--max-lag", type=int, default=500)
    parser.add_argument("--corr-threshold", type=float, default=0.09)
    parser.add_argument("--freq-aggregation", default="any")
    parser.add_argument("--eta", type=float, default=0.55)
    parser.add_argument("--min-turn-strength", type=float, default=0.0)
    parser.add_argument("--phase-smoothing", type=int, default=1)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    out_dir = Path(args.out_dir) if args.out_dir else manifest_path.parent / "diagnostics"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle, delimiter="\t"))

    per_csv_rows: list[dict[str, object]] = []
    per_point_acc: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    errors: list[dict[str, object]] = []

    for manifest_row in manifest_rows:
        seed = manifest_row.get("seed", "")
        aval = manifest_row.get("a", "")
        bval = manifest_row.get("b", "")
        csv_path = _resolve_out(manifest_path, manifest_row.get("out", ""))
        try:
            metrics = _analyze_csv(csv_path, args)
            row = {
                "seed": seed,
                "a": aval,
                "b": bval,
                "rounds": manifest_row.get("rounds", ""),
                "csv": str(csv_path),
                **metrics,
            }
            per_csv_rows.append(row)
            per_point_acc[(aval, bval)].append(row)
        except Exception as exc:
            errors.append(
                {
                    "seed": seed,
                    "a": aval,
                    "b": bval,
                    "rounds": manifest_row.get("rounds", ""),
                    "csv": str(csv_path),
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )

    per_point_rows: list[dict[str, object]] = []
    summary_fields = [
        "mean_level",
        "hits_level3",
        "n",
        "mean_amp_max",
        "mean_amp_mean",
        "mean_freq_best_corr",
        "mean_turning_score",
        "mean_turning_strength",
        "mean_centroid_score",
        "mean_centroid_strength",
        "mean_rotation_consistency",
        "mean_rotation_net",
        "freq_best_corr_max",
        "turning_score_max",
        "turning_strength_max",
    ]
    for (aval, bval), rows in sorted(per_point_acc.items(), key=lambda item: (float(item[0][0]), float(item[0][1]))):
        point_row: dict[str, object] = {
            "a": aval,
            "b": bval,
            "n": len(rows),
            "mean_level": _safe_mean([float(r["level"]) for r in rows]),
            "hits_level3": sum(1 for r in rows if int(r["level"]) >= 3),
            "mean_amp_max": _safe_mean([float(r["amp_max"]) for r in rows]),
            "mean_amp_mean": _safe_mean([float(r["amp_mean"]) for r in rows]),
            "mean_freq_best_corr": _safe_mean([float(r["freq_best_corr"]) for r in rows]),
            "mean_turning_score": _safe_mean([float(r["turning_score"]) for r in rows]),
            "mean_turning_strength": _safe_mean([float(r["turning_strength"]) for r in rows]),
            "mean_centroid_score": _safe_mean([float(r["centroid_score"]) for r in rows]),
            "mean_centroid_strength": _safe_mean([float(r["centroid_strength"]) for r in rows]),
            "mean_rotation_consistency": _safe_mean([float(r["rotation_consistency"]) for r in rows]),
            "mean_rotation_net": _safe_mean([float(r["rotation_net"]) for r in rows]),
            "freq_best_corr_max": max(float(r["freq_best_corr"]) for r in rows),
            "turning_score_max": max(float(r["turning_score"]) for r in rows),
            "turning_strength_max": max(float(r["turning_strength"]) for r in rows),
        }
        per_point_rows.append(point_row)

    csv_fields = [
        "seed", "a", "b", "rounds", "csv", "level",
        "amp_passed", "amp_threshold", "amp_aggressive", "amp_defensive", "amp_balanced", "amp_max", "amp_mean",
        "freq_passed", "freq_threshold", "freq_best_strategy", "freq_best_corr", "freq_best_lag", "freq_effective_window_n",
        "corr_aggressive", "lag_aggressive", "corr_defensive", "lag_defensive", "corr_balanced", "lag_balanced",
        "turning_passed", "turning_direction", "turning_score", "turning_strength",
        "centroid_passed", "centroid_direction", "centroid_score", "centroid_strength",
        "rotation_passed", "rotation_net", "rotation_direction", "rotation_consistency", "rotation_used_steps", "rotation_used_points",
    ]
    per_point_fields = ["a", "b", *summary_fields]
    error_fields = ["seed", "a", "b", "rounds", "csv", "error"]

    _write_tsv(out_dir / "per_csv_metrics.tsv", per_csv_rows, csv_fields)
    _write_tsv(out_dir / "per_point_summary.tsv", per_point_rows, per_point_fields)
    if errors:
        _write_tsv(out_dir / "errors.tsv", errors, error_fields)

    ranked = sorted(
        per_point_rows,
        key=lambda row: (
            float(row["hits_level3"]),
            float(row["mean_level"]),
            float(row["mean_turning_score"]),
            float(row["mean_freq_best_corr"]),
        ),
        reverse=True,
    )

    print(f"manifest={manifest_path}")
    print(f"per_csv={out_dir / 'per_csv_metrics.tsv'}")
    print(f"per_point={out_dir / 'per_point_summary.tsv'}")
    print(f"errors={len(errors)}")
    print("=== point summary ===")
    for row in ranked:
        print(
            f"a={row['a']} b={row['b']} | "
            f"hits={row['hits_level3']}/{row['n']} mean_level={float(row['mean_level']):.3f} "
            f"amp_max={float(row['mean_amp_max']):.4f} freq_corr={float(row['mean_freq_best_corr']):.4f} "
            f"turn_score={float(row['mean_turning_score']):.4f} turn_strength={float(row['mean_turning_strength']):.6f} "
            f"rot_consistency={float(row['mean_rotation_consistency']):.4f}"
        )


if __name__ == "__main__":
    main()