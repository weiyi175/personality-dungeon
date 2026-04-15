from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/w1"

SOURCE_DRAFT_CSV = OUT_DIR / "d2_b2_seed_draft_v1.csv"

NEW_L1_TABLE_CSV = OUT_DIR / "d3_b2_new_l1_key_table_v1.csv"
BROKE_TABLE_CSV = OUT_DIR / "d3_b2_broke_key_table_v1.csv"
RESCUED_TABLE_CSV = OUT_DIR / "d3_b2_rescued_key_table_v1.csv"

D3_SUMMARY_JSON = OUT_DIR / "d3_b2_failure_tables_summary_v1.json"
D3_NOTE_MD = OUT_DIR / "d3_b2_failure_tables_note_v1.md"

HEALTHY_THRESHOLD = 0.8


def _to_int(value: str) -> int:
    return int(float(value))


def _to_float(value: str) -> float:
    return float(value)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _normalise_row(raw: dict[str, str]) -> dict[str, Any]:
    s3 = _to_float(raw["s3"])
    return {
        "seed": _to_int(raw["seed"]),
        "primary_label": raw["primary_label"],
        "level": _to_int(raw["level"]),
        "s3": s3,
        "gamma": _to_float(raw["gamma"]),
        "baseline_healthy_flag": _to_int(raw["baseline_healthy_flag"]),
        "current_healthy_flag": _to_int(raw["current_healthy_flag"]),
        "new_l1_flag": _to_int(raw["new_l1_flag"]),
        "broke_flag": _to_int(raw["broke_flag"]),
        "rescued_flag": _to_int(raw["rescued_flag"]),
        "fairness_fail_flag": _to_int(raw["fairness_fail_flag"]),
        "dispatch_fairness_pass": _to_int(raw["dispatch_fairness_pass"]),
        "event_sync_index_mean": _to_float(raw["event_sync_index_mean"]),
        "reward_perturb_corr": _to_float(raw["reward_perturb_corr"]),
        "trap_entry_round": _to_int(raw["trap_entry_round"]) if raw.get("trap_entry_round") not in (None, "") else None,
        "source_file": raw["source_file"],
        "healthy_gap": max(0.0, HEALTHY_THRESHOLD - s3),
        "rescue_margin": max(0.0, s3 - HEALTHY_THRESHOLD),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _project_new_l1(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projected = []
    for r in rows:
        if r["new_l1_flag"] != 1:
            continue
        projected.append(
            {
                "seed": r["seed"],
                "level": r["level"],
                "s3": r["s3"],
                "healthy_gap": r["healthy_gap"],
                "gamma": r["gamma"],
                "trap_entry_round": r["trap_entry_round"],
                "event_sync_index_mean": r["event_sync_index_mean"],
                "reward_perturb_corr": r["reward_perturb_corr"],
                "priority_note": "newly dropped to level 1 from baseline healthy",
                "source_file": r["source_file"],
            }
        )
    return sorted(projected, key=lambda x: (-x["healthy_gap"], x["seed"]))


def _project_broke(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projected = []
    for r in rows:
        if r["broke_flag"] != 1:
            continue
        projected.append(
            {
                "seed": r["seed"],
                "level": r["level"],
                "s3": r["s3"],
                "healthy_gap": r["healthy_gap"],
                "gamma": r["gamma"],
                "trap_entry_round": r["trap_entry_round"],
                "event_sync_index_mean": r["event_sync_index_mean"],
                "reward_perturb_corr": r["reward_perturb_corr"],
                "priority_note": "baseline healthy but now below healthy threshold",
                "source_file": r["source_file"],
            }
        )
    return sorted(projected, key=lambda x: (-x["healthy_gap"], x["seed"]))


def _project_rescued(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projected = []
    for r in rows:
        if r["rescued_flag"] != 1:
            continue
        projected.append(
            {
                "seed": r["seed"],
                "level": r["level"],
                "s3": r["s3"],
                "rescue_margin": r["rescue_margin"],
                "gamma": r["gamma"],
                "trap_entry_round": r["trap_entry_round"],
                "event_sync_index_mean": r["event_sync_index_mean"],
                "reward_perturb_corr": r["reward_perturb_corr"],
                "priority_note": "baseline non-healthy but now recovered healthy",
                "source_file": r["source_file"],
            }
        )
    return sorted(projected, key=lambda x: (-x["rescue_margin"], x["seed"]))


def _fmt_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _write_note(new_l1: list[dict[str, Any]], broke: list[dict[str, Any]], rescued: list[dict[str, Any]]) -> None:
    def seeds(rows: list[dict[str, Any]]) -> str:
        return ",".join(str(r["seed"]) for r in rows) if rows else "-"

    lines = [
        "# W1 D3 B2 Failure Tables",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- source: {SOURCE_DRAFT_CSV.relative_to(ROOT)}",
        "",
        "## Summary",
        "",
        f"- new_l1_count: {len(new_l1)}",
        f"- broke_count: {len(broke)}",
        f"- rescued_count: {len(rescued)}",
        f"- new_l1_seeds: {seeds(new_l1)}",
        f"- broke_seeds: {seeds(broke)}",
        f"- rescued_seeds: {seeds(rescued)}",
        "",
        "## Top Severity (Broke)",
        "",
        "| seed | s3 | healthy_gap | trap_entry_round |",
        "|---:|---:|---:|---:|",
    ]

    for row in broke[:5]:
        lines.append(
            f"| {row['seed']} | {_fmt_float(row['s3'])} | {_fmt_float(row['healthy_gap'])} | {row['trap_entry_round']} |"
        )

    lines.append("")
    lines.append("## Top Rescue Margin")
    lines.append("")
    lines.append("| seed | s3 | rescue_margin | trap_entry_round |")
    lines.append("|---:|---:|---:|---:|")

    for row in rescued[:5]:
        lines.append(
            f"| {row['seed']} | {_fmt_float(row['s3'])} | {_fmt_float(row['rescue_margin'])} | {row['trap_entry_round']} |"
        )

    D3_NOTE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_rows = _read_rows(SOURCE_DRAFT_CSV)
    rows = [_normalise_row(raw) for raw in raw_rows]

    new_l1_rows = _project_new_l1(rows)
    broke_rows = _project_broke(rows)
    rescued_rows = _project_rescued(rows)

    _write_csv(NEW_L1_TABLE_CSV, new_l1_rows)
    _write_csv(BROKE_TABLE_CSV, broke_rows)
    _write_csv(RESCUED_TABLE_CSV, rescued_rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE_DRAFT_CSV.relative_to(ROOT)),
        "counts": {
            "new_l1": len(new_l1_rows),
            "broke": len(broke_rows),
            "rescued": len(rescued_rows),
        },
        "files": {
            "new_l1_csv": str(NEW_L1_TABLE_CSV.relative_to(ROOT)),
            "broke_csv": str(BROKE_TABLE_CSV.relative_to(ROOT)),
            "rescued_csv": str(RESCUED_TABLE_CSV.relative_to(ROOT)),
            "note_md": str(D3_NOTE_MD.relative_to(ROOT)),
        },
        "seed_sets": {
            "new_l1": [row["seed"] for row in new_l1_rows],
            "broke": [row["seed"] for row in broke_rows],
            "rescued": [row["seed"] for row in rescued_rows],
        },
    }

    D3_SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_note(new_l1_rows, broke_rows, rescued_rows)

    print(f"Wrote: {NEW_L1_TABLE_CSV.relative_to(ROOT)}")
    print(f"Wrote: {BROKE_TABLE_CSV.relative_to(ROOT)}")
    print(f"Wrote: {RESCUED_TABLE_CSV.relative_to(ROOT)}")
    print(f"Wrote: {D3_SUMMARY_JSON.relative_to(ROOT)}")
    print(f"Wrote: {D3_NOTE_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
