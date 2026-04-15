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

from analysis.cycle_metrics import classify_cycle_level, phase_direction_consistency_turning


CFG = dict(
    burn_in=2400,
    tail=1000,
    amplitude_threshold=0.02,
    corr_threshold=0.09,
    eta=0.55,
    stage3_method="turning",
    phase_smoothing=1,
    min_lag=2,
    max_lag=500,
)


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


def _load_series(csv_path: Path) -> dict[str, list[float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")
    return {
        "aggressive": [float(row["p_aggressive"]) for row in rows],
        "defensive": [float(row["p_defensive"]) for row in rows],
        "balanced": [float(row["p_balanced"]) for row in rows],
    }


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _round_key(value: str) -> str:
    return f"{float(value):.6g}"


def _manifest_run_tag(manifest_path: Path) -> str:
    return manifest_path.parent.name


def _manifest_candidates(path: Path) -> list[Path]:
    preferred = [
        path / "manifest_cross_coupling.tsv",
        path / "manifest_m3_timing.tsv",
    ]
    resolved = [candidate for candidate in preferred if candidate.exists()]
    if resolved:
        return resolved
    return sorted(candidate for candidate in path.glob("manifest*.tsv") if candidate.is_file())


def _collect_manifest_rows(manifest_paths: list[Path]) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    analyzed: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    for manifest_path in manifest_paths:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        for row in rows:
            csv_path = _resolve_out(manifest_path, row["out"])
            try:
                series = _load_series(csv_path)
                result = classify_cycle_level(series, **CFG)
                turning = phase_direction_consistency_turning(
                    series,
                    burn_in=CFG["burn_in"],
                    tail=CFG["tail"],
                    eta=CFG["eta"],
                    min_turn_strength=0.0,
                    phase_smoothing=CFG["phase_smoothing"],
                )
                score = float(result.stage3.score) if result.stage3 is not None else 0.0
                analyzed.append(
                    {
                        "run_tag": _manifest_run_tag(manifest_path),
                        "manifest": str(manifest_path),
                        "seed": int(row["seed"]),
                        "a": _round_key(row["a"]),
                        "b": _round_key(row["b"]),
                        "c": _round_key(row["c"]),
                        "mode": row.get("popularity_mode", ""),
                        "aps": _round_key(row.get("aps", "0")),
                        "interval": row.get("interval", ""),
                        "target": _round_key(row.get("target", "0")),
                        "level": int(result.level),
                        "score": score,
                        "turn_strength": float(turning.turn_strength),
                        "turn_score": float(turning.score),
                        "csv": str(csv_path),
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "manifest": str(manifest_path),
                        "seed": row.get("seed", ""),
                        "a": row.get("a", ""),
                        "b": row.get("b", ""),
                        "c": row.get("c", ""),
                        "mode": row.get("popularity_mode", ""),
                        "aps": row.get("aps", ""),
                        "interval": row.get("interval", ""),
                        "target": row.get("target", ""),
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
    return analyzed, errors


def _write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize mechanism-search manifests into ranked condition tables."
    )
    parser.add_argument(
        "manifest_paths",
        nargs="+",
        help="One or more manifest TSV files, or directories containing manifest*.tsv.",
    )
    parser.add_argument(
        "--out-dir",
        help="Output directory for summary TSVs. Defaults to outputs/mechanism_search/s1_summary_<timestampless>.",
    )
    parser.add_argument(
        "--baseline-manifest",
        help="Optional manifest used to compute baseline turn strength for relative uplift.",
    )
    args = parser.parse_args()

    manifest_paths: list[Path] = []
    for raw in args.manifest_paths:
        path = Path(raw)
        if path.is_dir():
            manifest_paths.extend(_manifest_candidates(path))
        elif path.exists():
            manifest_paths.append(path)
        else:
            raise SystemExit(f"Path not found: {path}")

    if not manifest_paths:
        raise SystemExit("No manifest paths resolved.")

    analyzed, errors = _collect_manifest_rows(manifest_paths)
    if not analyzed:
        raise SystemExit("No rows analyzed successfully.")

    baseline_rows: list[dict[str, object]] = []
    if args.baseline_manifest:
        baseline_rows, _ = _collect_manifest_rows([Path(args.baseline_manifest)])
    baseline_turn = _safe_mean([float(row["turn_strength"]) for row in baseline_rows]) if baseline_rows else 0.0

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "outputs" / "mechanism_search" / "s1_summary"

    per_seed_fields = [
        "run_tag", "manifest", "seed", "a", "b", "c", "mode", "aps", "interval", "target",
        "level", "score", "turn_strength", "turn_score", "csv",
    ]
    _write_tsv(out_dir / "per_seed_metrics.tsv", analyzed, per_seed_fields)

    by_condition_ab: dict[tuple[str, str, str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    by_condition: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)

    for row in analyzed:
        by_condition_ab[(row["a"], row["b"], row["c"], row["mode"], row["aps"], row["interval"], row["target"])].append(row)
        by_condition[(row["c"], row["mode"], row["aps"], row["interval"], row["target"])].append(row)

    per_ab_rows: list[dict[str, object]] = []
    for key, rows in sorted(
        by_condition_ab.items(),
        key=lambda item: (
            float(item[0][0]),
            float(item[0][1]),
            float(item[0][2]),
            item[0][3],
            float(item[0][4]),
            int(item[0][5]),
            float(item[0][6]),
        ),
    ):
        mean_turn = _safe_mean([float(r["turn_strength"]) for r in rows])
        row = {
            "a": key[0],
            "b": key[1],
            "c": key[2],
            "mode": key[3],
            "aps": key[4],
            "interval": key[5],
            "target": key[6],
            "n": len(rows),
            "hits": sum(1 for r in rows if int(r["level"]) >= 3),
            "pr_l3": sum(1 for r in rows if int(r["level"]) >= 3) / len(rows),
            "mean_level": _safe_mean([float(r["level"]) for r in rows]),
            "mean_score": _safe_mean([float(r["score"]) for r in rows]),
            "mean_turn_strength": mean_turn,
            "mean_turn_score": _safe_mean([float(r["turn_score"]) for r in rows]),
            "turn_uplift_vs_baseline": ((mean_turn / baseline_turn) - 1.0) if baseline_turn > 0.0 else 0.0,
        }
        per_ab_rows.append(row)

    per_ab_rows.sort(
        key=lambda row: (float(row["pr_l3"]), float(row["mean_turn_strength"]), float(row["mean_score"])),
        reverse=True,
    )
    _write_tsv(
        out_dir / "summary_by_ab.tsv",
        per_ab_rows,
        [
            "a", "b", "c", "mode", "aps", "interval", "target", "n", "hits", "pr_l3", "mean_level",
            "mean_score", "mean_turn_strength", "mean_turn_score", "turn_uplift_vs_baseline",
        ],
    )

    per_condition_rows: list[dict[str, object]] = []
    for key, rows in sorted(
        by_condition.items(),
        key=lambda item: (
            float(item[0][0]),
            item[0][1],
            float(item[0][2]),
            int(item[0][3]),
            float(item[0][4]),
        ),
    ):
        mean_turn = _safe_mean([float(r["turn_strength"]) for r in rows])
        row = {
            "c": key[0],
            "mode": key[1],
            "aps": key[2],
            "interval": key[3],
            "target": key[4],
            "n": len(rows),
            "ab_points": len({(r["a"], r["b"]) for r in rows}),
            "hits": sum(1 for r in rows if int(r["level"]) >= 3),
            "pr_l3": sum(1 for r in rows if int(r["level"]) >= 3) / len(rows),
            "mean_level": _safe_mean([float(r["level"]) for r in rows]),
            "mean_score": _safe_mean([float(r["score"]) for r in rows]),
            "mean_turn_strength": mean_turn,
            "mean_turn_score": _safe_mean([float(r["turn_score"]) for r in rows]),
            "turn_uplift_vs_baseline": ((mean_turn / baseline_turn) - 1.0) if baseline_turn > 0.0 else 0.0,
        }
        per_condition_rows.append(row)

    per_condition_rows.sort(
        key=lambda row: (float(row["pr_l3"]), float(row["mean_turn_strength"]), float(row["mean_score"])),
        reverse=True,
    )
    _write_tsv(
        out_dir / "summary_by_condition.tsv",
        per_condition_rows,
        [
            "c", "mode", "aps", "interval", "target", "n", "ab_points", "hits", "pr_l3", "mean_level",
            "mean_score", "mean_turn_strength", "mean_turn_score", "turn_uplift_vs_baseline",
        ],
    )

    if errors:
        _write_tsv(
            out_dir / "errors.tsv",
            errors,
            ["manifest", "seed", "a", "b", "c", "mode", "aps", "interval", "target", "error"],
        )

    print(f"manifests={len(manifest_paths)}")
    print(f"per_seed={out_dir / 'per_seed_metrics.tsv'}")
    print(f"summary_by_ab={out_dir / 'summary_by_ab.tsv'}")
    print(f"summary_by_condition={out_dir / 'summary_by_condition.tsv'}")
    print(f"errors={len(errors)}")
    if baseline_turn > 0.0:
        print(f"baseline_turn_strength={baseline_turn:.8f}")
    print("=== top conditions ===")
    for row in per_condition_rows[:10]:
        print(
            f"c={row['c']} mode={row['mode']} aps={row['aps']} interval={row['interval']} target={row['target']} | "
            f"hits={row['hits']}/{row['n']} pr_l3={float(row['pr_l3']):.4f} "
            f"mean_score={float(row['mean_score']):.4f} mean_turn={float(row['mean_turn_strength']):.6f} "
            f"uplift={float(row['turn_uplift_vs_baseline']):+.4f}"
        )


if __name__ == "__main__":
    main()