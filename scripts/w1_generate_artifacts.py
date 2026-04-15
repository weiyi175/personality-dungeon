from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

A3_SUMMARY_PATH = ROOT / "outputs/track_a_protocol_regression_summary.json"
B2_SUMMARY_PATH = ROOT / "outputs/b2_diag_multiplicative_gate60_block42_101_fw2000_ft050_summary.json"
B3_SUMMARY_PATH = ROOT / "outputs/b3_diag_impact_spread_gate60_block42_101_fw2000_ft050_summary.json"

OUT_DIR = ROOT / "outputs/w1"
A3_DASHBOARD_CSV = OUT_DIR / "a3_dashboard_v1.csv"
FAILURE_CATALOG_CSV = OUT_DIR / "failure_catalog_v1.csv"
FAILURE_SUMMARY_JSON = OUT_DIR / "failure_catalog_v1_summary.json"
W1_MEMO_MD = OUT_DIR / "w1_review_memo_v1.md"


@dataclass
class FamilyResult:
    family: str
    source_file: str
    block: str
    total_seeds: int
    non_healthy: int
    fairness_fail: int
    new_l1: int
    broke: int
    rescued: int
    gate_overall_pass: bool


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
    # Example: "34 passed in 0.51s"
    m = re.search(r"(\d+)\s+passed\b", stdout)
    if not m:
        return None
    return int(m.group(1))


def _build_a3_dashboard_row(a3_summary: dict[str, Any]) -> dict[str, Any]:
    a1_recheck = _extract_stage(a3_summary, "A1 Gate Recheck")
    a1_compare = _extract_stage(a3_summary, "A1 Compare")
    a2_runtime = _extract_stage(a3_summary, "A2 Runtime Regression")

    recheck_path = Path(a1_recheck.get("details", {}).get("recheck_out_json", ""))
    if not recheck_path.is_absolute():
        recheck_path = ROOT / recheck_path
    recheck = _load_json(recheck_path)

    a2_stdout = str(a2_runtime.get("details", {}).get("stdout", ""))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "a3_timestamp_utc": a3_summary.get("timestamp_utc"),
        "a3_overall_pass": bool(a3_summary.get("overall_pass", False)),
        "a1_recheck_passed": bool(a1_recheck.get("passed", False)),
        "a1_compare_passed": bool(a1_compare.get("passed", False)),
        "a2_runtime_passed": bool(a2_runtime.get("passed", False)),
        "a1_l1": recheck.get("l1"),
        "a1_healthy": recheck.get("healthy"),
        "a1_mean_s3": recheck.get("mean_s3"),
        "a1_median_s3": recheck.get("median_s3"),
        "a1_p10_s3": recheck.get("p10_s3"),
        "a1_mean_gamma": recheck.get("mean_gamma"),
        "a1_gate_overall_pass": bool(recheck.get("gate", {}).get("overall_pass", False)),
        "a1_non_healthy_count": len(recheck.get("non_healthy", []) or []),
        "a2_pytest_exit_code": a2_runtime.get("exit_code"),
        "a2_pytest_pass_count": _extract_pytest_pass_count(a2_stdout),
        "a2_pytest_stdout": a2_stdout.strip(),
        "a3_summary_file": str(A3_SUMMARY_PATH.relative_to(ROOT)),
        "a1_recheck_file": str(recheck_path.relative_to(ROOT)),
    }


def _baseline_non_healthy_set(baseline: dict[str, Any]) -> set[int]:
    values = baseline.get("non_healthy", [])
    non_healthy: set[int] = set()
    for row in values:
        if isinstance(row, dict) and "seed" in row:
            non_healthy.add(int(row["seed"]))
    return non_healthy


def _block_str(seeds: list[int]) -> str:
    if not seeds:
        return "unknown"
    return f"{min(seeds)}..{max(seeds)}"


def _build_family_rows(
    *,
    family: str,
    family_summary: dict[str, Any],
    baseline_non_healthy: set[int],
    source_file: Path,
) -> tuple[list[dict[str, Any]], FamilyResult]:
    outcomes = family_summary.get("outcomes", []) or []
    healthy_threshold = float(family_summary.get("gate", {}).get("healthy_threshold", 0.8))
    seeds = [int(item.get("seed", -1)) for item in outcomes if "seed" in item]

    rows: list[dict[str, Any]] = []
    fairness_fail = 0
    new_l1 = 0
    broke = 0
    rescued = 0
    non_healthy = 0

    for item in outcomes:
        seed = int(item["seed"])
        level = int(item.get("level", 0))
        s3 = float(item.get("s3", 0.0))
        gamma = float(item.get("gamma", 0.0))

        baseline_is_healthy = seed not in baseline_non_healthy
        current_is_healthy = s3 >= healthy_threshold
        dispatch_fairness_pass = bool(item.get("dispatch_fairness_pass", True))

        new_l1_flag = baseline_is_healthy and level == 1
        broke_flag = baseline_is_healthy and (not current_is_healthy)
        rescued_flag = (not baseline_is_healthy) and current_is_healthy
        fairness_fail_flag = not dispatch_fairness_pass

        if not current_is_healthy:
            non_healthy += 1
        if fairness_fail_flag:
            fairness_fail += 1
        if new_l1_flag:
            new_l1 += 1
        if broke_flag:
            broke += 1
        if rescued_flag:
            rescued += 1

        failure_type = "ok"
        if new_l1_flag:
            failure_type = "new_l1"
        elif broke_flag:
            failure_type = "broke"
        elif fairness_fail_flag:
            failure_type = "fairness"
        elif not current_is_healthy:
            failure_type = "other"

        if failure_type == "ok" and not rescued_flag:
            continue

        note = []
        if rescued_flag:
            note.append("rescued_from_baseline_non_healthy")
        if new_l1_flag:
            note.append("new_level1_from_baseline_healthy")
        if broke_flag and (not new_l1_flag):
            note.append("dropped_below_healthy_threshold")

        rows.append(
            {
                "family": family,
                "block": _block_str(seeds),
                "seed": seed,
                "failure_type": failure_type,
                "new_l1_flag": int(new_l1_flag),
                "rescued_flag": int(rescued_flag),
                "broke_flag": int(broke_flag),
                "fairness_fail_flag": int(fairness_fail_flag),
                "level": level,
                "s3": s3,
                "gamma": gamma,
                "dispatch_fairness_pass": int(dispatch_fairness_pass),
                "baseline_healthy_flag": int(baseline_is_healthy),
                "current_healthy_flag": int(current_is_healthy),
                "note": ";".join(note) if note else "",
                "source_file": str(source_file.relative_to(ROOT)),
            }
        )

    summary = FamilyResult(
        family=family,
        source_file=str(source_file.relative_to(ROOT)),
        block=_block_str(seeds),
        total_seeds=len(outcomes),
        non_healthy=non_healthy,
        fairness_fail=fairness_fail,
        new_l1=new_l1,
        broke=broke,
        rescued=rescued,
        gate_overall_pass=bool(family_summary.get("gate", {}).get("overall_pass", False)),
    )
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_memo(
    *,
    a3_row: dict[str, Any],
    family_results: list[FamilyResult],
    failure_rows: list[dict[str, Any]],
) -> None:
    total_failure_rows = len(failure_rows)
    new_l1_total = sum(x.new_l1 for x in family_results)
    broke_total = sum(x.broke for x in family_results)
    rescued_total = sum(x.rescued for x in family_results)

    lines: list[str] = []
    lines.append("# W1 Review Memo v1")
    lines.append("")
    lines.append(f"- generated_at_utc: {a3_row['generated_at_utc']}")
    lines.append(f"- a3_overall_pass: {a3_row['a3_overall_pass']}")
    lines.append(f"- a1_l1: {a3_row['a1_l1']}")
    lines.append(f"- a1_healthy: {a3_row['a1_healthy']}")
    lines.append(f"- a1_mean_s3: {a3_row['a1_mean_s3']}")
    lines.append(f"- a2_pytest_pass_count: {a3_row['a2_pytest_pass_count']}")
    lines.append("")
    lines.append("## Failure Catalog v1 Snapshot")
    lines.append("")
    lines.append(f"- failure_rows: {total_failure_rows}")
    lines.append(f"- new_l1_total: {new_l1_total}")
    lines.append(f"- broke_total: {broke_total}")
    lines.append(f"- rescued_total: {rescued_total}")
    lines.append("")
    lines.append("## By Family")
    lines.append("")
    lines.append("| Family | Block | total_seeds | non_healthy | new_l1 | broke | rescued | fairness_fail | gate_overall_pass |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for item in family_results:
        lines.append(
            f"| {item.family} | {item.block} | {item.total_seeds} | {item.non_healthy} | {item.new_l1} | {item.broke} | {item.rescued} | {item.fairness_fail} | {'PASS' if item.gate_overall_pass else 'FAIL'} |"
        )
    lines.append("")
    lines.append("## Decision Note")
    lines.append("")
    lines.append("- W1 以主線穩定與失敗知識庫建置為目標，維持 Track A 優先。")
    lines.append("- 若後續 A3 fail，優先處理主線，再更新 catalog。")

    W1_MEMO_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    a3_summary = _load_json(A3_SUMMARY_PATH)
    b2_summary = _load_json(B2_SUMMARY_PATH)
    b3_summary = _load_json(B3_SUMMARY_PATH)

    baseline_path = Path(b2_summary.get("baseline_summary_json", "outputs/pers_cal_baseline_gate60_summary.json"))
    if not baseline_path.is_absolute():
        baseline_path = ROOT / baseline_path
    baseline = _load_json(baseline_path)
    baseline_non_healthy = _baseline_non_healthy_set(baseline)

    a3_row = _build_a3_dashboard_row(a3_summary)

    b2_rows, b2_result = _build_family_rows(
        family="B2",
        family_summary=b2_summary,
        baseline_non_healthy=baseline_non_healthy,
        source_file=B2_SUMMARY_PATH,
    )
    b3_rows, b3_result = _build_family_rows(
        family="B3",
        family_summary=b3_summary,
        baseline_non_healthy=baseline_non_healthy,
        source_file=B3_SUMMARY_PATH,
    )

    failure_rows = sorted(b2_rows + b3_rows, key=lambda row: (row["family"], row["seed"]))

    _write_csv(A3_DASHBOARD_CSV, [a3_row])
    _write_csv(FAILURE_CATALOG_CSV, failure_rows)

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "a3_summary": str(A3_SUMMARY_PATH.relative_to(ROOT)),
            "b2_summary": str(B2_SUMMARY_PATH.relative_to(ROOT)),
            "b3_summary": str(B3_SUMMARY_PATH.relative_to(ROOT)),
            "baseline_summary": str(baseline_path.relative_to(ROOT)),
        },
        "a3_dashboard": {
            "row_count": 1,
            "csv": str(A3_DASHBOARD_CSV.relative_to(ROOT)),
        },
        "failure_catalog": {
            "row_count": len(failure_rows),
            "csv": str(FAILURE_CATALOG_CSV.relative_to(ROOT)),
            "families": [b2_result.__dict__, b3_result.__dict__],
        },
    }
    FAILURE_SUMMARY_JSON.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_memo(a3_row=a3_row, family_results=[b2_result, b3_result], failure_rows=failure_rows)

    print(f"Wrote: {A3_DASHBOARD_CSV.relative_to(ROOT)}")
    print(f"Wrote: {FAILURE_CATALOG_CSV.relative_to(ROOT)}")
    print(f"Wrote: {FAILURE_SUMMARY_JSON.relative_to(ROOT)}")
    print(f"Wrote: {W1_MEMO_MD.relative_to(ROOT)}")
    print(f"Failure rows: {len(failure_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
