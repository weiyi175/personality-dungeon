from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/w1"

D5_SUMMARY_JSON = OUT_DIR / "d5_w1_review_summary_v1.json"
W2_DECISION_JSON = OUT_DIR / "w2_decision_input_v1.json"
W2_DECISION_MD = OUT_DIR / "w2_decision_input_v1.md"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _is_mainline_stable(mainline: dict[str, Any]) -> tuple[bool, list[str]]:
    checks = {
        "a3_overall_pass": bool(mainline.get("a3_overall_pass", False)),
        "status_green": str(mainline.get("status", "")).upper() == "GREEN",
        "a1_l1_le_3": int(mainline.get("a1_l1", 0)) <= 3,
        "a1_healthy_ge_42": int(mainline.get("a1_healthy", 0)) >= 42,
    }
    reasons = [f"{k}={v}" for k, v in checks.items()]
    return all(checks.values()), reasons


def _has_structural_evidence(payload: dict[str, Any]) -> tuple[bool, dict[str, bool], list[str]]:
    fail = payload.get("b2_failure_profile", {})
    hypotheses = payload.get("hypotheses", [])

    broke_non_l1_count = int(fail.get("broke_non_l1", 0))
    broke_non_l1_mean_s3 = float(fail.get("broke_non_l1_mean_s3") or 0.0)
    broke_early_trap_share = float(fail.get("broke_early_trap_share_le_2000") or 0.0)
    rescued = int(fail.get("rescued", 0))
    p0_seeds = fail.get("p0_seeds", []) or []

    checks = {
        "h2_cluster_size_ge_5": broke_non_l1_count >= 5,
        "h2_mean_s3_in_055_band": 0.53 <= broke_non_l1_mean_s3 <= 0.58,
        "early_trap_share_ge_060": broke_early_trap_share >= 0.60,
        "rescued_positive_control_ge_5": rescued >= 5,
        "p0_seed_non_empty": len(p0_seeds) > 0,
        "hypothesis_bundle_ge_3": len(hypotheses) >= 3,
    }

    reasons = [
        f"broke_non_l1_count={broke_non_l1_count}",
        f"broke_non_l1_mean_s3={broke_non_l1_mean_s3:.4f}",
        f"broke_early_trap_share={broke_early_trap_share:.3f}",
        f"rescued={rescued}",
        f"p0_seeds={p0_seeds}",
        f"hypotheses={len(hypotheses)}",
    ]

    return all(checks.values()), checks, reasons


def _build_payload() -> dict[str, Any]:
    d5 = _read_json(D5_SUMMARY_JSON)
    mainline = d5.get("mainline", {})
    fail = d5.get("b2_failure_profile", {})

    mainline_stable, mainline_reasons = _is_mainline_stable(mainline)
    has_structure, structure_checks, structure_reasons = _has_structural_evidence(d5)

    restart_now_ready = bool(fail.get("gate_overall_pass", False) and int(fail.get("new_l1", 0)) == 0)

    if has_structure and not restart_now_ready:
        event_verdict = "supported_for_future_hypothesis_only"
    elif has_structure and restart_now_ready:
        event_verdict = "supported_and_restart_ready"
    else:
        event_verdict = "insufficient_structure"

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "d5_summary": str(D5_SUMMARY_JSON.relative_to(ROOT)),
        },
        "decision_inputs": {
            "mainline_80_20": {
                "question": "主線是否穩定到可維持 80/20 節奏不變",
                "keep_80_20": mainline_stable,
                "verdict": "keep_80_20" if mainline_stable else "stabilize_first",
                "checks": mainline_reasons,
                "watch": {
                    "p0_seeds": fail.get("p0_seeds", []),
                    "new_l1": int(fail.get("new_l1", 0)),
                    "broke": int(fail.get("broke", 0)),
                },
            },
            "event_line_restart_hypothesis": {
                "question": "是否存在足夠結構證據支持未來 Event line 重啟假說",
                "structural_evidence_sufficient": has_structure,
                "restart_now_ready": restart_now_ready,
                "verdict": event_verdict,
                "checks": structure_checks,
                "signals": structure_reasons,
                "next_actions": [
                    "W2 first pass keeps 80/20 rhythm unchanged.",
                    "Run P0 seeds event-timeline replay before any restart proposal.",
                    "Keep rescued seeds as positive controls for mechanism validation.",
                ],
            },
        },
    }


def _write_markdown(payload: dict[str, Any]) -> None:
    d1 = payload["decision_inputs"]["mainline_80_20"]
    d2 = payload["decision_inputs"]["event_line_restart_hypothesis"]

    lines: list[str] = []
    lines.append("# W2 Decision Input v1（from W1）")
    lines.append("")
    lines.append(f"- generated_at_utc: {payload['generated_at_utc']}")
    lines.append(f"- input: {payload['input']['d5_summary']}")
    lines.append("")

    lines.append("## 1) 判斷一：主線是否維持 80/20")
    lines.append("")
    lines.append(f"- verdict: {d1['verdict']}")
    lines.append(f"- keep_80_20: {d1['keep_80_20']}")
    lines.append("- checks:")
    for item in d1["checks"]:
        lines.append(f"  - {item}")
    lines.append("- watch:")
    lines.append(f"  - p0_seeds={d1['watch']['p0_seeds']}")
    lines.append(f"  - new_l1={d1['watch']['new_l1']}")
    lines.append(f"  - broke={d1['watch']['broke']}")
    lines.append("")

    lines.append("## 2) 判斷二：是否有足夠結構證據支持未來 Event line 重啟假說")
    lines.append("")
    lines.append(f"- verdict: {d2['verdict']}")
    lines.append(f"- structural_evidence_sufficient: {d2['structural_evidence_sufficient']}")
    lines.append(f"- restart_now_ready: {d2['restart_now_ready']}")
    lines.append("- checks:")
    for k, v in d2["checks"].items():
        lines.append(f"  - {k}={v}")
    lines.append("- signals:")
    for item in d2["signals"]:
        lines.append(f"  - {item}")
    lines.append("- next_actions:")
    for item in d2["next_actions"]:
        lines.append(f"  - {item}")
    lines.append("")

    lines.append("## 3) 結論（W2 起手）")
    lines.append("")
    if d1["keep_80_20"]:
        lines.append("- 主線穩定：W2 維持 80/20 節奏不變。")
    else:
        lines.append("- 主線未穩：W2 先轉 stabilization，不維持原 80/20。")

    if d2["structural_evidence_sufficient"]:
        if d2["restart_now_ready"]:
            lines.append("- Event line：結構證據充分且已達可重啟條件。")
        else:
            lines.append("- Event line：結構證據充分，但尚未達立即重啟條件（作為 future hypothesis input）。")
    else:
        lines.append("- Event line：結構證據不足，暫不支持重啟假說。")

    W2_DECISION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = _build_payload()
    W2_DECISION_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload)

    print(f"Wrote: {W2_DECISION_JSON.relative_to(ROOT)}")
    print(f"Wrote: {W2_DECISION_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
