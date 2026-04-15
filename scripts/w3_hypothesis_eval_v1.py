from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/w3"

BACKLOG_JSON = OUT_DIR / "hypothesis_backlog_v1.json"
W1_SUMMARY_JSON = ROOT / "outputs/w1/d5_w1_review_summary_v1.json"
W2_REPORT_JSON = ROOT / "outputs/w2/block_risk_report_v1.json"
A3_SUMMARY_JSON = ROOT / "outputs/track_a_protocol_regression_summary.json"

OUT_JSON = OUT_DIR / "hypothesis_eval_v1.json"
OUT_MD = OUT_DIR / "hypothesis_eval_v1.md"

BLOCK_ORDER = ["42..101", "102..161", "162..221", "222..281"]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _to_int(value: Any, default: int = 0) -> int:
    if value in (None, "", "NA"):
        return default
    return int(float(value))


def _to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "NA"):
        return default
    return float(value)


def _resolve_non_healthy_set(data: dict[str, Any], healthy_threshold: float) -> set[int]:
    result: set[int] = set()
    for row in data.get("non_healthy", []) or []:
        if isinstance(row, dict) and "seed" in row:
            result.add(_to_int(row["seed"]))
        else:
            result.add(_to_int(row))

    if result:
        return result

    for row in data.get("outcomes", []) or []:
        seed = _to_int(row.get("seed"))
        s3 = _to_float(row.get("s3"))
        if s3 < healthy_threshold:
            result.add(seed)
    return result


def _focus_rows(w2_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = w2_report["focus_comparisons"]["b1_async_poisson_r008_across_blocks"]["rows"]
    order = {b: i for i, b in enumerate(BLOCK_ORDER)}
    return sorted(rows, key=lambda r: order.get(str(r.get("block", "")), 999))


def _evaluate_h1(w1_summary: dict[str, Any], w2_report: dict[str, Any]) -> dict[str, Any]:
    ref_broke_non_l1 = _to_int(w1_summary["b2_failure_profile"].get("broke_non_l1"))
    baseline_map = w2_report["inputs"]["baseline_blocks"]

    results: list[dict[str, Any]] = []

    for row in _focus_rows(w2_report):
        block = str(row["block"])
        run_path = ROOT / str(row["source"])
        baseline_path = ROOT / str(baseline_map[block])

        run_data = _read_json(run_path)
        baseline_data = _read_json(baseline_path)
        healthy_threshold = _to_float(run_data.get("gate", {}).get("healthy_threshold"), 0.8)
        baseline_non_healthy = _resolve_non_healthy_set(baseline_data, healthy_threshold)

        broke_non_l1_s3: list[float] = []
        early_trap_count = 0

        for outcome in run_data.get("outcomes", []) or []:
            seed = _to_int(outcome.get("seed"))
            s3 = _to_float(outcome.get("s3"))
            level = _to_int(outcome.get("level"))
            trap_round = outcome.get("trap_entry_round")

            baseline_healthy = seed not in baseline_non_healthy
            current_healthy = s3 >= healthy_threshold
            is_broke_non_l1 = baseline_healthy and (not current_healthy) and level != 1

            if not is_broke_non_l1:
                continue

            broke_non_l1_s3.append(s3)
            if trap_round is not None and _to_int(trap_round) <= 2000:
                early_trap_count += 1

        broke_non_l1_count = len(broke_non_l1_s3)
        near_055_count = sum(1 for s3 in broke_non_l1_s3 if abs(s3 - 0.55) <= 0.03)

        reduction_ratio = (
            1.0 - (broke_non_l1_count / ref_broke_non_l1)
            if ref_broke_non_l1 > 0
            else 0.0
        )
        early_trap_share = (
            early_trap_count / broke_non_l1_count if broke_non_l1_count > 0 else 0.0
        )

        checks = {
            "new_l1_zero": _to_int(row.get("new_l1")) == 0,
            "broke_non_l1_drop_ge_30pct": reduction_ratio >= 0.30,
            "early_trap_share_le_070": early_trap_share <= 0.70,
        }

        results.append(
            {
                "block": block,
                "source": str(run_path.relative_to(ROOT)),
                "new_l1": _to_int(row.get("new_l1")),
                "broke_non_l1_count": broke_non_l1_count,
                "broke_non_l1_mean_s3": mean(broke_non_l1_s3) if broke_non_l1_s3 else None,
                "near_055_share": (
                    near_055_count / broke_non_l1_count if broke_non_l1_count > 0 else None
                ),
                "early_trap_share": early_trap_share,
                "reduction_vs_w1_ref": reduction_ratio,
                "checks": checks,
                "pass": all(checks.values()),
            }
        )

    fail_fast_triggered = any((r["new_l1"] > 0) or (r["early_trap_share"] > 0.70) for r in results)
    passed = all(r["pass"] for r in results) and (len(results) > 0)

    return {
        "id": "H-01",
        "title": "broke_non_l1 ~ s3=0.55 閾值群聚跨 block 可重現性",
        "reference": {
            "w1_broke_non_l1": ref_broke_non_l1,
            "w1_source": str(W1_SUMMARY_JSON.relative_to(ROOT)),
        },
        "criteria": {
            "new_l1_zero": "required",
            "broke_non_l1_drop_ge_30pct": "required",
            "early_trap_share_le_070": "fail_fast_guard",
        },
        "block_results": results,
        "fail_fast_triggered": fail_fast_triggered,
        "pass": passed,
        "verdict": "pass" if passed else "fail",
    }


def _evaluate_h2(w2_report: dict[str, Any]) -> dict[str, Any]:
    focus = w2_report["focus_comparisons"]["b1_async_poisson_r008_across_blocks"]
    rows = _focus_rows(w2_report)

    pass_rate = _to_float(focus.get("pass_rate"))
    mean_new_l1 = _to_float(focus.get("mean_new_l1"))
    any_new_l1_gt0 = any(_to_int(r.get("new_l1")) > 0 for r in rows)
    any_l1_gt3 = any(_to_int(r.get("l1")) > 3 for r in rows)

    checks = {
        "pass_rate_ge_075": pass_rate >= 0.75,
        "mean_new_l1_le_025": mean_new_l1 <= 0.25,
        "no_block_new_l1_gt_0": not any_new_l1_gt0,
        "no_block_l1_gt_3": not any_l1_gt3,
    }

    fail_fast_triggered = (pass_rate < 0.75) or (mean_new_l1 > 0.25) or any_new_l1_gt0
    passed = all(checks.values())

    return {
        "id": "H-02",
        "title": "cross-block 穩健性證據補齊",
        "focus_source": str(W2_REPORT_JSON.relative_to(ROOT)),
        "metrics": {
            "pass_rate": pass_rate,
            "mean_new_l1": mean_new_l1,
            "any_block_new_l1_gt_0": any_new_l1_gt0,
            "any_block_l1_gt_3": any_l1_gt3,
            "pass_blocks": focus.get("pass_blocks", []),
            "fail_blocks": focus.get("fail_blocks", []),
        },
        "checks": checks,
        "fail_fast_triggered": fail_fast_triggered,
        "pass": passed,
        "verdict": "pass" if passed else "fail",
    }


def _evaluate_h3(w1_summary: dict[str, Any], w2_report: dict[str, Any]) -> dict[str, Any]:
    rows = _focus_rows(w2_report)
    drift = w2_report["a3_drift_detection"]

    rescued_ref = _to_int(w1_summary["b2_failure_profile"].get("rescued"))
    risk_ref = _to_int(w1_summary["b2_failure_profile"].get("broke")) + _to_int(
        w1_summary["b2_failure_profile"].get("new_l1")
    )

    rescued_now = sum(_to_int(r.get("rescued")) for r in rows)
    broke_now = sum(_to_int(r.get("broke")) for r in rows)
    new_l1_now = sum(_to_int(r.get("new_l1")) for r in rows)
    risk_now = broke_now + new_l1_now

    risk_reduction = 1.0 - (risk_now / risk_ref) if risk_ref > 0 else 0.0
    drift_guard_ok = not bool(drift.get("drift_detected", True))

    checks = {
        "rescued_count_ge_7": rescued_now >= rescued_ref,
        "risk_reduction_ge_25pct": risk_reduction >= 0.25,
        "a3_drift_guard_stable": drift_guard_ok,
    }

    fail_fast_triggered = (
        (rescued_now < rescued_ref)
        or (risk_reduction < 0.25)
        or (not drift_guard_ok)
    )
    passed = all(checks.values())

    return {
        "id": "H-03",
        "title": "rescued 與 broke/new_l1 共同改善",
        "reference": {
            "rescued_ref": rescued_ref,
            "risk_ref_broke_plus_new_l1": risk_ref,
            "w1_source": str(W1_SUMMARY_JSON.relative_to(ROOT)),
        },
        "current": {
            "rescued_now": rescued_now,
            "broke_now": broke_now,
            "new_l1_now": new_l1_now,
            "risk_now_broke_plus_new_l1": risk_now,
            "risk_reduction": risk_reduction,
            "drift_detected": bool(drift.get("drift_detected", True)),
        },
        "checks": checks,
        "fail_fast_triggered": fail_fast_triggered,
        "pass": passed,
        "verdict": "pass" if passed else "fail",
        "assumption": "H-03 uses W1 single-block baseline versus current 4-block aggregate as an operational risk trend check.",
    }


def _build_payload() -> dict[str, Any]:
    backlog = _read_json(BACKLOG_JSON)
    w1_summary = _read_json(W1_SUMMARY_JSON)
    w2_report = _read_json(W2_REPORT_JSON)
    a3_summary = _read_json(A3_SUMMARY_JSON)

    h1 = _evaluate_h1(w1_summary, w2_report)
    h2 = _evaluate_h2(w2_report)
    h3 = _evaluate_h3(w1_summary, w2_report)

    hypothesis_results = [h1, h2, h3]
    passed_ids = [h["id"] for h in hypothesis_results if h["pass"]]
    failed_ids = [h["id"] for h in hypothesis_results if not h["pass"]]

    return {
        "version": "v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "backlog": str(BACKLOG_JSON.relative_to(ROOT)),
            "w1_summary": str(W1_SUMMARY_JSON.relative_to(ROOT)),
            "w2_report": str(W2_REPORT_JSON.relative_to(ROOT)),
            "a3_summary": str(A3_SUMMARY_JSON.relative_to(ROOT)),
        },
        "context": {
            "backlog_status": backlog.get("status"),
            "a3_overall_pass": bool(a3_summary.get("overall_pass", False)),
            "w2_block_robustness": w2_report.get("w2_decision", {}).get("block_robustness"),
        },
        "hypotheses": hypothesis_results,
        "summary": {
            "passed": passed_ids,
            "failed": failed_ids,
            "all_pass": len(failed_ids) == 0,
            "overall_verdict": "pass" if len(failed_ids) == 0 else "fail",
        },
    }


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _write_markdown(payload: dict[str, Any]) -> None:
    h1, h2, h3 = payload["hypotheses"]
    summary = payload["summary"]

    lines: list[str] = []
    lines.append("# W3 Hypothesis Eval v1")
    lines.append("")
    lines.append(f"- generated_at_utc: {payload['generated_at_utc']}")
    lines.append("- run_command: ./venv/bin/python scripts/w3_hypothesis_eval_v1.py")
    lines.append(f"- overall_verdict: {summary['overall_verdict']}")
    lines.append(f"- passed: {summary['passed']}")
    lines.append(f"- failed: {summary['failed']}")
    lines.append("")

    lines.append("## H-01")
    lines.append("")
    lines.append(f"- verdict: {h1['verdict']}")
    lines.append(f"- fail_fast_triggered: {h1['fail_fast_triggered']}")
    lines.append("- block details:")
    lines.append("")
    lines.append("| block | new_l1 | broke_non_l1 | mean_s3 | near_055_share | early_trap_share | reduction_vs_w1_ref | pass |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in h1["block_results"]:
        lines.append(
            "| {block} | {new_l1} | {broke_non_l1_count} | {mean_s3} | {near055} | {early} | {reduction} | {passed} |".format(
                block=row["block"],
                new_l1=row["new_l1"],
                broke_non_l1_count=row["broke_non_l1_count"],
                mean_s3=_fmt_float(row["broke_non_l1_mean_s3"]),
                near055=_fmt_float(row["near_055_share"]),
                early=_fmt_float(row["early_trap_share"]),
                reduction=_fmt_float(row["reduction_vs_w1_ref"]),
                passed=row["pass"],
            )
        )
    lines.append("")

    lines.append("## H-02")
    lines.append("")
    lines.append(f"- verdict: {h2['verdict']}")
    lines.append(f"- fail_fast_triggered: {h2['fail_fast_triggered']}")
    lines.append(f"- pass_rate: {_fmt_float(h2['metrics']['pass_rate'])} (target >= 0.750)")
    lines.append(f"- mean_new_l1: {_fmt_float(h2['metrics']['mean_new_l1'])} (target <= 0.250)")
    lines.append(f"- any_block_new_l1_gt_0: {h2['metrics']['any_block_new_l1_gt_0']}")
    lines.append(f"- any_block_l1_gt_3: {h2['metrics']['any_block_l1_gt_3']}")
    lines.append("")

    lines.append("## H-03")
    lines.append("")
    lines.append(f"- verdict: {h3['verdict']}")
    lines.append(f"- fail_fast_triggered: {h3['fail_fast_triggered']}")
    lines.append(
        f"- rescued_now: {h3['current']['rescued_now']} (target >= {h3['reference']['rescued_ref']})"
    )
    lines.append(
        f"- risk_now(broke+new_l1): {h3['current']['risk_now_broke_plus_new_l1']} (ref={h3['reference']['risk_ref_broke_plus_new_l1']})"
    )
    lines.append(f"- risk_reduction: {_fmt_float(h3['current']['risk_reduction'])} (target >= 0.250)")
    lines.append(f"- drift_detected: {h3['current']['drift_detected']}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- {h3['assumption']}")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = _build_payload()

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload)

    print(f"Wrote: {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote: {OUT_MD.relative_to(ROOT)}")
    print(f"overall_verdict: {payload['summary']['overall_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())