from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/w2"

A3_SUMMARY_JSON = ROOT / "outputs/track_a_protocol_regression_summary.json"
D5_SUMMARY_JSON = ROOT / "outputs/w1/d5_w1_review_summary_v1.json"

BASELINE_FILES = {
    "42..101": ROOT / "outputs/pers_cal_baseline_gate60_summary.json",
    "102..161": ROOT / "outputs/pers_cal_baseline_gate60_block102_161_summary.json",
    "162..221": ROOT / "outputs/pers_cal_baseline_gate60_block162_221_summary.json",
    "222..281": ROOT / "outputs/pers_cal_baseline_gate60_block222_281_summary.json",
}

EVENT_RUNS: list[dict[str, str]] = [
    {
        "experiment_id": "b1_async_round_robin_r008",
        "family": "b1",
        "block": "42..101",
        "path": "outputs/b1_async_dispatch_rr_w2000_t050_gate60_summary.json",
    },
    {
        "experiment_id": "b1_async_poisson_r008",
        "family": "b1",
        "block": "42..101",
        "path": "outputs/b1_async_dispatch_poisson_w2000_t050_gate60_summary.json",
    },
    {
        "experiment_id": "b1_async_poisson_r006",
        "family": "b1",
        "block": "42..101",
        "path": "outputs/b1_async_dispatch_poisson_r006_w2000_t050_gate60_summary.json",
    },
    {
        "experiment_id": "b1_async_poisson_r007",
        "family": "b1",
        "block": "42..101",
        "path": "outputs/b1_async_dispatch_poisson_r007_w2000_t050_gate60_summary.json",
    },
    {
        "experiment_id": "b1_async_poisson_r009",
        "family": "b1",
        "block": "42..101",
        "path": "outputs/b1_async_dispatch_poisson_r009_w2000_t050_gate60_summary.json",
    },
    {
        "experiment_id": "b1_async_poisson_r008",
        "family": "b1",
        "block": "102..161",
        "path": "outputs/b1_async_dispatch_poisson_r008_w2000_t050_gate60_block102_161_summary.json",
    },
    {
        "experiment_id": "b1_async_poisson_r008",
        "family": "b1",
        "block": "162..221",
        "path": "outputs/b1_async_dispatch_poisson_r008_w2000_t050_gate60_block162_221_summary.json",
    },
    {
        "experiment_id": "b1_async_poisson_r008",
        "family": "b1",
        "block": "222..281",
        "path": "outputs/b1_async_dispatch_poisson_r008_w2000_t050_gate60_block222_281_summary.json",
    },
    {
        "experiment_id": "b2_multiplicative_fw2000_ft050",
        "family": "b2",
        "block": "42..101",
        "path": "outputs/b2_diag_multiplicative_gate60_block42_101_fw2000_ft050_summary.json",
    },
    {
        "experiment_id": "b3_impact_spread_fw2000_ft050",
        "family": "b3",
        "block": "42..101",
        "path": "outputs/b3_diag_impact_spread_gate60_block42_101_fw2000_ft050_summary.json",
    },
]

CSV_PATH = OUT_DIR / "block_risk_report_v1.csv"
JSON_PATH = OUT_DIR / "block_risk_report_v1.json"
MD_PATH = OUT_DIR / "block_risk_report_v1.md"


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


def _summary_metrics(data: dict[str, Any]) -> dict[str, Any]:
    gate = data.get("gate", {})
    seeds = data.get("seeds", []) or []
    seed_range = f"{min(seeds)}..{max(seeds)}" if seeds else "NA"

    return {
        "stage": str(data.get("stage", "NA")),
        "overall_pass": bool(gate.get("overall_pass", False)),
        "l1": _to_int(data.get("l1")),
        "healthy": _to_int(data.get("healthy")),
        "mean_s3": _to_float(data.get("mean_s3")),
        "new_l1": _to_int(data.get("new_l1")),
        "rescued": _to_int(data.get("rescued")),
        "broke": _to_int(data.get("broke")),
        "fairness_fail_count": _to_int(data.get("fairness_fail_count")),
        "seed_range": seed_range,
    }


def _risk_score(metrics: dict[str, Any], delta_healthy: int, delta_s3: float) -> tuple[int, str]:
    score = 0
    if not metrics["overall_pass"]:
        score += 4
    score += max(0, metrics["new_l1"]) * 2
    score += max(0, metrics["l1"] - 3)
    score += max(0, 42 - metrics["healthy"])
    if metrics["fairness_fail_count"] > 0:
        score += 3
    if delta_healthy < -2:
        score += 1
    if delta_s3 < -0.02:
        score += 1

    if score >= 8:
        tier = "high"
    elif score >= 4:
        tier = "medium"
    else:
        tier = "low"
    return score, tier


def _build_payload() -> dict[str, Any]:
    a3 = _read_json(A3_SUMMARY_JSON)
    d5 = _read_json(D5_SUMMARY_JSON)

    baselines: dict[str, dict[str, Any]] = {}
    for block, path in BASELINE_FILES.items():
        data = _read_json(path)
        metrics = _summary_metrics(data)
        baselines[block] = {
            "source": str(path.relative_to(ROOT)),
            **metrics,
        }

    reference_mainline = d5.get("mainline", {})
    current_mainline = {
        "a3_overall_pass": bool(a3.get("overall_pass", False)),
        "a1_l1": baselines["42..101"]["l1"],
        "a1_healthy": baselines["42..101"]["healthy"],
        "a1_mean_s3": baselines["42..101"]["mean_s3"],
    }

    drift_thresholds = {
        "delta_l1_abs_max": 1,
        "delta_healthy_abs_max": 3,
        "delta_mean_s3_abs_max": 0.02,
    }

    drift = {
        "delta_l1": current_mainline["a1_l1"] - _to_int(reference_mainline.get("a1_l1")),
        "delta_healthy": current_mainline["a1_healthy"] - _to_int(reference_mainline.get("a1_healthy")),
        "delta_mean_s3": current_mainline["a1_mean_s3"] - _to_float(reference_mainline.get("a1_mean_s3")),
    }

    drift_checks = {
        "a3_overall_pass": current_mainline["a3_overall_pass"],
        "delta_l1_ok": abs(drift["delta_l1"]) <= drift_thresholds["delta_l1_abs_max"],
        "delta_healthy_ok": abs(drift["delta_healthy"]) <= drift_thresholds["delta_healthy_abs_max"],
        "delta_mean_s3_ok": abs(drift["delta_mean_s3"]) <= drift_thresholds["delta_mean_s3_abs_max"],
    }

    event_rows: list[dict[str, Any]] = []
    for row in EVENT_RUNS:
        path = ROOT / row["path"]
        data = _read_json(path)
        metrics = _summary_metrics(data)

        base = baselines.get(row["block"])
        if base is None:
            raise RuntimeError(f"Missing baseline for block: {row['block']}")

        delta_l1 = metrics["l1"] - base["l1"]
        delta_healthy = metrics["healthy"] - base["healthy"]
        delta_mean_s3 = metrics["mean_s3"] - base["mean_s3"]
        risk_score, risk_tier = _risk_score(metrics, delta_healthy, delta_mean_s3)

        event_rows.append(
            {
                "experiment_id": row["experiment_id"],
                "family": row["family"],
                "block": row["block"],
                "stage": metrics["stage"],
                "overall_pass": metrics["overall_pass"],
                "l1": metrics["l1"],
                "healthy": metrics["healthy"],
                "mean_s3": metrics["mean_s3"],
                "new_l1": metrics["new_l1"],
                "rescued": metrics["rescued"],
                "broke": metrics["broke"],
                "fairness_fail_count": metrics["fairness_fail_count"],
                "seed_range": metrics["seed_range"],
                "delta_l1_vs_baseline": delta_l1,
                "delta_healthy_vs_baseline": delta_healthy,
                "delta_mean_s3_vs_baseline": delta_mean_s3,
                "risk_score": risk_score,
                "risk_tier": risk_tier,
                "source": row["path"],
            }
        )

    poisson_008_rows = [
        r
        for r in event_rows
        if r["experiment_id"] == "b1_async_poisson_r008"
        and r["block"] in {"42..101", "102..161", "162..221", "222..281"}
    ]
    poisson_008_pass_blocks = [r["block"] for r in poisson_008_rows if r["overall_pass"]]
    poisson_008_fail_blocks = [r["block"] for r in poisson_008_rows if not r["overall_pass"]]
    poisson_008_pass_rate = (
        len(poisson_008_pass_blocks) / len(poisson_008_rows) if poisson_008_rows else 0.0
    )

    w2_mainline_verdict = (
        "keep_80_20"
        if all(drift_checks.values())
        else "stabilize_first"
    )
    block_robust_verdict = (
        "block_robust"
        if poisson_008_rows
        and poisson_008_pass_rate >= 0.75
        and all(_to_int(r["new_l1"]) == 0 for r in poisson_008_rows)
        else "not_block_robust"
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "a3_summary": str(A3_SUMMARY_JSON.relative_to(ROOT)),
            "d5_summary": str(D5_SUMMARY_JSON.relative_to(ROOT)),
            "baseline_blocks": {
                block: str(path.relative_to(ROOT)) for block, path in BASELINE_FILES.items()
            },
            "event_runs": [row["path"] for row in EVENT_RUNS],
        },
        "a3_drift_detection": {
            "reference": {
                "a3_overall_pass": bool(reference_mainline.get("a3_overall_pass", False)),
                "a1_l1": _to_int(reference_mainline.get("a1_l1")),
                "a1_healthy": _to_int(reference_mainline.get("a1_healthy")),
                "a1_mean_s3": _to_float(reference_mainline.get("a1_mean_s3")),
            },
            "current": current_mainline,
            "thresholds": drift_thresholds,
            "deltas": drift,
            "checks": drift_checks,
            "drift_detected": not all(drift_checks.values()),
            "verdict": "stable" if all(drift_checks.values()) else "drift_detected",
        },
        "baseline_block_comparison": baselines,
        "event_block_comparison": event_rows,
        "focus_comparisons": {
            "b1_async_poisson_r008_across_blocks": {
                "rows": poisson_008_rows,
                "pass_blocks": poisson_008_pass_blocks,
                "fail_blocks": poisson_008_fail_blocks,
                "pass_rate": poisson_008_pass_rate,
                "mean_new_l1": mean([r["new_l1"] for r in poisson_008_rows]) if poisson_008_rows else 0.0,
                "verdict": block_robust_verdict,
            }
        },
        "w2_decision": {
            "mainline_schedule": w2_mainline_verdict,
            "block_robustness": block_robust_verdict,
            "block_risk_report": "block_risk_report_v1",
        },
    }


def _write_csv(event_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "experiment_id",
        "family",
        "block",
        "seed_range",
        "overall_pass",
        "l1",
        "healthy",
        "mean_s3",
        "new_l1",
        "rescued",
        "broke",
        "fairness_fail_count",
        "delta_l1_vs_baseline",
        "delta_healthy_vs_baseline",
        "delta_mean_s3_vs_baseline",
        "risk_score",
        "risk_tier",
        "source",
    ]

    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in event_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _write_markdown(payload: dict[str, Any]) -> None:
    drift = payload["a3_drift_detection"]
    focus = payload["focus_comparisons"]["b1_async_poisson_r008_across_blocks"]
    decision = payload["w2_decision"]

    lines: list[str] = []
    lines.append("# Block Risk Report v1 (W2)")
    lines.append("")
    lines.append(f"- generated_at_utc: {payload['generated_at_utc']}")
    lines.append("")

    lines.append("## 1) A3 Routine Regression + Drift Detection")
    lines.append("")
    lines.append(f"- verdict: {drift['verdict']}")
    lines.append(f"- drift_detected: {drift['drift_detected']}")
    lines.append(
        f"- deltas: l1={drift['deltas']['delta_l1']}, healthy={drift['deltas']['delta_healthy']}, mean_s3={drift['deltas']['delta_mean_s3']:.6f}"
    )
    lines.append(f"- checks: {drift['checks']}")
    lines.append("")

    lines.append("## 2) Block-Level Robustness Comparison")
    lines.append("")
    lines.append("| experiment_id | block | overall_pass | l1 | healthy | new_l1 | rescued | broke | risk_tier |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["event_block_comparison"]:
        lines.append(
            f"| {row['experiment_id']} | {row['block']} | {row['overall_pass']} | {row['l1']} | {row['healthy']} | {row['new_l1']} | {row['rescued']} | {row['broke']} | {row['risk_tier']} |"
        )
    lines.append("")

    lines.append("## 3) Focus Comparison: b1 async_poisson r0.08")
    lines.append("")
    lines.append(f"- pass_blocks: {focus['pass_blocks']}")
    lines.append(f"- fail_blocks: {focus['fail_blocks']}")
    lines.append(f"- pass_rate: {focus['pass_rate']:.3f}")
    lines.append(f"- mean_new_l1: {focus['mean_new_l1']:.3f}")
    lines.append(f"- verdict: {focus['verdict']}")
    lines.append("")

    lines.append("## 4) W2 Decision Input")
    lines.append("")
    lines.append(f"- mainline_schedule: {decision['mainline_schedule']}")
    lines.append(f"- block_robustness: {decision['block_robustness']}")
    lines.append("- recommendation: keep 80/20 rhythm, continue block diagnostics, do not restart event line now.")

    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = _build_payload()

    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(payload["event_block_comparison"])
    _write_markdown(payload)

    print(f"Wrote: {JSON_PATH.relative_to(ROOT)}")
    print(f"Wrote: {CSV_PATH.relative_to(ROOT)}")
    print(f"Wrote: {MD_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
