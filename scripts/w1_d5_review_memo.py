from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/w1"

D2_A3_TREND_ROW = OUT_DIR / "d2_a3_trend_row_v1.csv"
D2_B2_SUMMARY = OUT_DIR / "d2_b2_seed_draft_summary_v1.json"
D3_NEW_L1 = OUT_DIR / "d3_b2_new_l1_key_table_v1.csv"
D3_BROKE = OUT_DIR / "d3_b2_broke_key_table_v1.csv"
D3_RESCUED = OUT_DIR / "d3_b2_rescued_key_table_v1.csv"
D4_SUMMARY = OUT_DIR / "d4_weekly_summary_v1.json"
D4_RISK_PRIORITY = OUT_DIR / "d4_weekly_risk_priority_v1.csv"

D5_MEMO_MD = OUT_DIR / "d5_w1_review_memo_v1.md"
D5_SUMMARY_JSON = OUT_DIR / "d5_w1_review_summary_v1.json"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(v: Any) -> float | None:
    if v in (None, "", "NA"):
        return None
    return float(v)


def _to_int(v: Any) -> int | None:
    if v in (None, "", "NA"):
        return None
    return int(float(v))


def _to_bool_text(v: Any) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "NA"
    return f"{v:.{digits}f}"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def _seed_list(values: list[int], max_items: int = 12) -> str:
    if not values:
        return "無"
    return ", ".join(str(x) for x in values[:max_items])


def _build_payload() -> dict[str, Any]:
    a3_rows = _read_csv_rows(D2_A3_TREND_ROW)
    if not a3_rows:
        raise RuntimeError("D2 A3 trend row is empty")
    a3 = a3_rows[0]

    b2_summary = _read_json(D2_B2_SUMMARY)
    d4_summary = _read_json(D4_SUMMARY)

    new_l1_rows = _read_csv_rows(D3_NEW_L1)
    broke_rows = _read_csv_rows(D3_BROKE)
    rescued_rows = _read_csv_rows(D3_RESCUED)
    risk_rows = _read_csv_rows(D4_RISK_PRIORITY)

    a3_overall_pass = _to_bool_text(a3.get("a3_overall_pass"))
    a1_l1 = _to_int(a3.get("a1_l1")) or 0
    a1_healthy = _to_int(a3.get("a1_healthy")) or 0
    a2_exit_code = _to_int(a3.get("a2_pytest_exit_code"))
    if a2_exit_code is None:
        a2_exit_code = 999

    mainline_green = bool(a3_overall_pass and a1_l1 <= 3 and a1_healthy >= 42 and a2_exit_code == 0)
    b2_gate_overall = bool(b2_summary.get("gate", {}).get("overall_pass", False))

    broke_gaps = [_to_float(r.get("healthy_gap")) or 0.0 for r in broke_rows]
    broke_s3 = [_to_float(r.get("s3")) or 0.0 for r in broke_rows]
    broke_traps = [_to_int(r.get("trap_entry_round")) for r in broke_rows if _to_int(r.get("trap_entry_round")) is not None]
    broke_non_l1_rows = [r for r in broke_rows if (_to_int(r.get("level")) or 0) != 1]
    broke_non_l1_gaps = [_to_float(r.get("healthy_gap")) or 0.0 for r in broke_non_l1_rows]
    broke_non_l1_s3 = [_to_float(r.get("s3")) or 0.0 for r in broke_non_l1_rows]
    rescued_margins = [_to_float(r.get("rescue_margin")) or 0.0 for r in rescued_rows]

    broke_early_trap_count = sum(1 for x in broke_traps if x is not None and x <= 2000)
    broke_early_trap_share = (broke_early_trap_count / len(broke_traps)) if broke_traps else 0.0

    p0 = d4_summary.get("high_priority", {}).get("p0", [])
    p1 = d4_summary.get("high_priority", {}).get("p1", [])

    top5_risks = [
        {
            "rank": _to_int(r.get("rank")),
            "seed": _to_int(r.get("seed")),
            "label": r.get("label"),
            "priority": r.get("priority"),
            "risk_score": _to_float(r.get("risk_score")),
            "s3": _to_float(r.get("s3")),
            "healthy_gap": _to_float(r.get("healthy_gap")),
            "trap_entry_round": _to_int(r.get("trap_entry_round")),
        }
        for r in risk_rows[:5]
    ]

    hypotheses = [
        {
            "id": "H1",
            "title": "new_l1 屬於高風險早期崩落型態",
            "evidence": {
                "new_l1_seeds": [_to_int(r.get("seed")) for r in new_l1_rows],
                "new_l1_s3": [_to_float(r.get("s3")) for r in new_l1_rows],
                "new_l1_trap_rounds": [_to_int(r.get("trap_entry_round")) for r in new_l1_rows],
            },
        },
        {
            "id": "H2",
            "title": "broke seeds 呈現接近 0.55 的閾值邊界群聚",
            "evidence": {
                "broke_count": len(broke_rows),
                "broke_non_l1_count": len(broke_non_l1_rows),
                "broke_non_l1_mean_s3": _mean(broke_non_l1_s3),
                "broke_non_l1_mean_gap": _mean(broke_non_l1_gaps),
                "broke_early_trap_share_le_2000": broke_early_trap_share,
            },
        },
        {
            "id": "H3",
            "title": "rescued seeds 可作為正向對照，提供恢復樣態訊號",
            "evidence": {
                "rescued_count": len(rescued_rows),
                "rescued_mean_margin": _mean(rescued_margins),
                "rescued_max_margin": max(rescued_margins) if rescued_margins else None,
            },
        },
    ]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "d2_a3_trend_row": str(D2_A3_TREND_ROW.relative_to(ROOT)),
            "d2_b2_summary": str(D2_B2_SUMMARY.relative_to(ROOT)),
            "d3_new_l1": str(D3_NEW_L1.relative_to(ROOT)),
            "d3_broke": str(D3_BROKE.relative_to(ROOT)),
            "d3_rescued": str(D3_RESCUED.relative_to(ROOT)),
            "d4_summary": str(D4_SUMMARY.relative_to(ROOT)),
            "d4_risk_priority": str(D4_RISK_PRIORITY.relative_to(ROOT)),
        },
        "mainline": {
            "a3_overall_pass": a3_overall_pass,
            "a1_l1": a1_l1,
            "a1_healthy": a1_healthy,
            "a1_mean_s3": _to_float(a3.get("a1_mean_s3")),
            "a2_pytest_pass_count": _to_int(a3.get("a2_pytest_pass_count")),
            "status": "GREEN" if mainline_green else "RED",
        },
        "b2_failure_profile": {
            "gate_overall_pass": b2_gate_overall,
            "new_l1": len(new_l1_rows),
            "broke": len(broke_rows),
            "broke_non_l1": len(broke_non_l1_rows),
            "rescued": len(rescued_rows),
            "p0_seeds": p0,
            "p1_seeds": p1,
            "broke_mean_gap": _mean(broke_gaps),
            "broke_mean_s3": _mean(broke_s3),
            "broke_non_l1_mean_gap": _mean(broke_non_l1_gaps),
            "broke_non_l1_mean_s3": _mean(broke_non_l1_s3),
            "rescued_mean_margin": _mean(rescued_margins),
            "broke_early_trap_share_le_2000": broke_early_trap_share,
        },
        "top5_risks": top5_risks,
        "hypotheses": hypotheses,
    }


def _write_memo(payload: dict[str, Any]) -> None:
    mainline = payload["mainline"]
    fail = payload["b2_failure_profile"]

    lines: list[str] = []
    lines.append("# W1 Review Memo v1（D5）")
    lines.append("")
    lines.append(f"- generated_at_utc: {payload['generated_at_utc']}")
    lines.append("")

    lines.append("## 1) W1 主線健康結論")
    lines.append("")
    lines.append(f"- 主線狀態：{mainline['status']}（A3 overall_pass={mainline['a3_overall_pass']}）")
    lines.append(f"- A1 指標：L1={mainline['a1_l1']}，Healthy={mainline['a1_healthy']}，mean_s3={_fmt(mainline['a1_mean_s3'])}")
    lines.append(f"- A2 指標：pytest_pass_count={mainline['a2_pytest_pass_count']}")
    lines.append("- 結論：Track A 主線可持續，維持主線優先策略不變。")
    lines.append("")

    lines.append("## 2) 失敗樣態摘要（B2）")
    lines.append("")
    lines.append(f"- gate_overall_pass: {fail['gate_overall_pass']}")
    lines.append(f"- new_l1={fail['new_l1']}、broke={fail['broke']}（其中 non-L1={fail['broke_non_l1']}）、rescued={fail['rescued']}")
    lines.append(f"- 高優先風險：P0={_seed_list(fail['p0_seeds'])}，P1={_seed_list(fail['p1_seeds'])}")
    lines.append(f"- broke 平均 healthy_gap={_fmt(fail['broke_mean_gap'])}，平均 s3={_fmt(fail['broke_mean_s3'])}")
    lines.append(f"- broke(non-L1) 平均 healthy_gap={_fmt(fail['broke_non_l1_mean_gap'])}，平均 s3={_fmt(fail['broke_non_l1_mean_s3'])}")
    lines.append(f"- rescued 平均 rescue_margin={_fmt(fail['rescued_mean_margin'])}")
    lines.append(f"- broke 早期 trap share(<=2000)={_fmt(fail['broke_early_trap_share_le_2000'], 3)}")
    lines.append("")

    lines.append("## 3) 下一步假說")
    lines.append("")
    for h in payload["hypotheses"]:
        lines.append(f"- {h['id']}：{h['title']}")
    lines.append("")

    lines.append("## 4) 下週建議處置（W2 起手）")
    lines.append("")
    lines.append("1. 立即重跑 P0 seeds（86, 60）並做事件時點回放，確認 L1 崩落觸發條件。")
    lines.append("2. 對 broke 群做『低谷段』對照檢查，優先看 trap_entry_round 與健康閾值附近漂移。")
    lines.append("3. 把 rescued 群固定為正向對照 seed set，後續變更需同時檢查是否保留救援效果。")
    lines.append("4. 週內若新增 runtime 變更，先跑 A3，再重刷 D2-D5 產物鏈。")
    lines.append("")

    lines.append("## 5) 產物索引")
    lines.append("")
    lines.append(f"- {D5_SUMMARY_JSON.relative_to(ROOT)}")
    lines.append(f"- {D5_MEMO_MD.relative_to(ROOT)}")

    D5_MEMO_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = _build_payload()
    D5_SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_memo(payload)

    print(f"Wrote: {D5_SUMMARY_JSON.relative_to(ROOT)}")
    print(f"Wrote: {D5_MEMO_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
