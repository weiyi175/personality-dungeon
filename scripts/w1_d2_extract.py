from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

A3_SUMMARY_PATH = ROOT / "outputs/track_a_protocol_regression_summary.json"
B2_SUMMARY_PATH = ROOT / "outputs/b2_diag_multiplicative_gate60_block42_101_fw2000_ft050_summary.json"

OUT_DIR = ROOT / "outputs/w1"
A3_TREND_ROW_CSV = OUT_DIR / "d2_a3_trend_row_v1.csv"
B2_SEED_DRAFT_CSV = OUT_DIR / "d2_b2_seed_draft_v1.csv"
B2_SEED_DRAFT_SUMMARY_JSON = OUT_DIR / "d2_b2_seed_draft_summary_v1.json"
D2_NOTE_MD = OUT_DIR / "d2_delivery_note_v1.md"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_stage(summary: dict[str, Any], name: str) -> dict[str, Any]:
    for stage in summary.get("stages", []):
        if stage.get("name") == name:
            return stage
    raise KeyError(f"Stage not found: {name}")


def _extract_pytest_pass_count(stdout: str) -> int | None:
    m = re.search(r"(\d+)\s+passed\b", stdout)
    if not m:
        return None
    return int(m.group(1))


def _resolve_baseline_non_healthy(path: Path, healthy_threshold: float) -> set[int]:
    data = _load_json(path)
    result: set[int] = set()

    for row in data.get("non_healthy", []):
        if isinstance(row, dict) and "seed" in row:
            result.add(int(row["seed"]))

    if result:
        return result

    # Fallback: if non_healthy list is absent, derive from outcomes.
    for row in data.get("outcomes", []):
        seed = int(row["seed"])
        s3 = float(row.get("s3", 0.0))
        if s3 < healthy_threshold:
            result.add(seed)

    return result


def _build_a3_trend_row() -> dict[str, Any]:
    summary = _load_json(A3_SUMMARY_PATH)
    a1_recheck = _extract_stage(summary, "A1 Gate Recheck")
    a1_compare = _extract_stage(summary, "A1 Compare")
    a2_runtime = _extract_stage(summary, "A2 Runtime Regression")

    recheck_path = Path(a1_recheck.get("details", {}).get("recheck_out_json", ""))
    if not recheck_path.is_absolute():
        recheck_path = ROOT / recheck_path
    recheck = _load_json(recheck_path)

    a2_stdout = str(a2_runtime.get("details", {}).get("stdout", "")).strip()

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "a3_timestamp_utc": summary.get("timestamp_utc"),
        "a3_overall_pass": bool(summary.get("overall_pass", False)),
        "a1_recheck_passed": bool(a1_recheck.get("passed", False)),
        "a1_compare_passed": bool(a1_compare.get("passed", False)),
        "a2_runtime_passed": bool(a2_runtime.get("passed", False)),
        "a1_l1": recheck.get("l1"),
        "a1_healthy": recheck.get("healthy"),
        "a1_mean_s3": recheck.get("mean_s3"),
        "a1_median_s3": recheck.get("median_s3"),
        "a1_p10_s3": recheck.get("p10_s3"),
        "a1_mean_gamma": recheck.get("mean_gamma"),
        "a1_non_healthy_count": len(recheck.get("non_healthy", []) or []),
        "a2_pytest_exit_code": a2_runtime.get("exit_code"),
        "a2_pytest_pass_count": _extract_pytest_pass_count(a2_stdout),
        "a2_pytest_stdout": a2_stdout,
        "a3_summary_file": str(A3_SUMMARY_PATH.relative_to(ROOT)),
        "a1_recheck_file": str(recheck_path.relative_to(ROOT)),
    }


def _build_b2_seed_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    b2 = _load_json(B2_SUMMARY_PATH)

    healthy_threshold = float(b2.get("gate", {}).get("healthy_threshold", 0.8))
    baseline_path = Path(b2.get("baseline_summary_json", "outputs/pers_cal_baseline_gate60_summary.json"))
    if not baseline_path.is_absolute():
        baseline_path = ROOT / baseline_path

    baseline_non_healthy = _resolve_baseline_non_healthy(baseline_path, healthy_threshold)
    outcomes = b2.get("outcomes", []) or []

    rows: list[dict[str, Any]] = []
    counts = {
        "total_seeds": 0,
        "stable_healthy": 0,
        "non_healthy": 0,
        "new_l1": 0,
        "broke": 0,
        "rescued": 0,
        "fairness_fail": 0,
    }

    for item in sorted(outcomes, key=lambda x: int(x["seed"])):
        seed = int(item["seed"])
        level = int(item.get("level", 0))
        s3 = float(item.get("s3", 0.0))
        gamma = float(item.get("gamma", 0.0))
        fairness_pass = bool(item.get("dispatch_fairness_pass", True))

        baseline_healthy = seed not in baseline_non_healthy
        current_healthy = s3 >= healthy_threshold

        new_l1_flag = baseline_healthy and (level == 1)
        broke_flag = baseline_healthy and (not current_healthy)
        rescued_flag = (not baseline_healthy) and current_healthy
        fairness_fail_flag = not fairness_pass

        if new_l1_flag:
            primary_label = "new_l1"
        elif broke_flag:
            primary_label = "broke"
        elif rescued_flag:
            primary_label = "rescued"
        elif not current_healthy:
            primary_label = "non_healthy_other"
        else:
            primary_label = "stable_healthy"

        if not current_healthy:
            counts["non_healthy"] += 1
        if new_l1_flag:
            counts["new_l1"] += 1
        if broke_flag:
            counts["broke"] += 1
        if rescued_flag:
            counts["rescued"] += 1
        if fairness_fail_flag:
            counts["fairness_fail"] += 1
        if primary_label == "stable_healthy":
            counts["stable_healthy"] += 1
        counts["total_seeds"] += 1

        rows.append(
            {
                "seed": seed,
                "primary_label": primary_label,
                "level": level,
                "s3": s3,
                "gamma": gamma,
                "baseline_healthy_flag": int(baseline_healthy),
                "current_healthy_flag": int(current_healthy),
                "new_l1_flag": int(new_l1_flag),
                "broke_flag": int(broke_flag),
                "rescued_flag": int(rescued_flag),
                "fairness_fail_flag": int(fairness_fail_flag),
                "dispatch_fairness_pass": int(fairness_pass),
                "event_sync_index_mean": float(item.get("event_sync_index_mean", 0.0)),
                "reward_perturb_corr": float(item.get("reward_perturb_corr", 0.0)),
                "trap_entry_round": item.get("trap_entry_round"),
                "source_file": str(B2_SUMMARY_PATH.relative_to(ROOT)),
            }
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "b2_summary": str(B2_SUMMARY_PATH.relative_to(ROOT)),
            "baseline_summary": str(baseline_path.relative_to(ROOT)),
        },
        "thresholds": {
            "healthy_threshold": healthy_threshold,
        },
        "counts": counts,
        "gate": b2.get("gate", {}),
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_note(a3_row: dict[str, Any], b2_summary: dict[str, Any]) -> None:
    c = b2_summary["counts"]
    lines = [
        "# W1 D2 Delivery Note",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- a3_overall_pass: {a3_row['a3_overall_pass']}",
        f"- a1_l1: {a3_row['a1_l1']}",
        f"- a1_healthy: {a3_row['a1_healthy']}",
        f"- a2_pytest_pass_count: {a3_row['a2_pytest_pass_count']}",
        "",
        "## B2 Seed Draft",
        "",
        f"- total_seeds: {c['total_seeds']}",
        f"- stable_healthy: {c['stable_healthy']}",
        f"- non_healthy: {c['non_healthy']}",
        f"- new_l1: {c['new_l1']}",
        f"- broke: {c['broke']}",
        f"- rescued: {c['rescued']}",
        f"- fairness_fail: {c['fairness_fail']}",
    ]
    D2_NOTE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    a3_row = _build_a3_trend_row()
    b2_rows, b2_summary = _build_b2_seed_rows()

    _write_csv(A3_TREND_ROW_CSV, [a3_row])
    _write_csv(B2_SEED_DRAFT_CSV, b2_rows)
    B2_SEED_DRAFT_SUMMARY_JSON.write_text(
        json.dumps(b2_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_note(a3_row, b2_summary)

    print(f"Wrote: {A3_TREND_ROW_CSV.relative_to(ROOT)}")
    print(f"Wrote: {B2_SEED_DRAFT_CSV.relative_to(ROOT)}")
    print(f"Wrote: {B2_SEED_DRAFT_SUMMARY_JSON.relative_to(ROOT)}")
    print(f"Wrote: {D2_NOTE_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
