from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/w1"

D2_A3_TREND_ROW = OUT_DIR / "d2_a3_trend_row_v1.csv"
D3_NEW_L1 = OUT_DIR / "d3_b2_new_l1_key_table_v1.csv"
D3_BROKE = OUT_DIR / "d3_b2_broke_key_table_v1.csv"
D3_RESCUED = OUT_DIR / "d3_b2_rescued_key_table_v1.csv"

D4_RISK_PRIORITY_CSV = OUT_DIR / "d4_weekly_risk_priority_v1.csv"
D4_SUMMARY_JSON = OUT_DIR / "d4_weekly_summary_v1.json"
D4_ONEPAGER_MD = OUT_DIR / "d4_weekly_onepager_v1.md"


@dataclass
class SeedRisk:
    seed: int
    new_l1: bool = False
    broke: bool = False
    rescued: bool = False
    level: int | None = None
    s3: float | None = None
    gamma: float | None = None
    trap_entry_round: int | None = None
    event_sync_index_mean: float | None = None
    reward_perturb_corr: float | None = None
    healthy_gap: float = 0.0
    rescue_margin: float = 0.0
    score: float = 0.0
    priority: str = "P3"
    label: str = ""
    recommendation: str = ""


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_int(value: Any) -> int | None:
    if value in (None, "", "NA"):
        return None
    return int(float(value))


def _to_float(value: Any) -> float | None:
    if value in (None, "", "NA"):
        return None
    return float(value)


def _fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _build_seed_risk_map(
    new_l1_rows: list[dict[str, str]],
    broke_rows: list[dict[str, str]],
    rescued_rows: list[dict[str, str]],
) -> dict[int, SeedRisk]:
    risk_map: dict[int, SeedRisk] = {}

    def ensure(seed: int) -> SeedRisk:
        if seed not in risk_map:
            risk_map[seed] = SeedRisk(seed=seed)
        return risk_map[seed]

    for row in new_l1_rows:
        seed = int(row["seed"])
        item = ensure(seed)
        item.new_l1 = True
        item.level = _to_int(row.get("level"))
        item.s3 = _to_float(row.get("s3"))
        item.gamma = _to_float(row.get("gamma"))
        item.trap_entry_round = _to_int(row.get("trap_entry_round"))
        item.event_sync_index_mean = _to_float(row.get("event_sync_index_mean"))
        item.reward_perturb_corr = _to_float(row.get("reward_perturb_corr"))
        item.healthy_gap = max(item.healthy_gap, _to_float(row.get("healthy_gap")) or 0.0)

    for row in broke_rows:
        seed = int(row["seed"])
        item = ensure(seed)
        item.broke = True
        item.level = _to_int(row.get("level"))
        item.s3 = _to_float(row.get("s3"))
        item.gamma = _to_float(row.get("gamma"))
        item.trap_entry_round = _to_int(row.get("trap_entry_round"))
        item.event_sync_index_mean = _to_float(row.get("event_sync_index_mean"))
        item.reward_perturb_corr = _to_float(row.get("reward_perturb_corr"))
        item.healthy_gap = max(item.healthy_gap, _to_float(row.get("healthy_gap")) or 0.0)

    for row in rescued_rows:
        seed = int(row["seed"])
        item = ensure(seed)
        item.rescued = True
        item.level = _to_int(row.get("level"))
        item.s3 = _to_float(row.get("s3"))
        item.gamma = _to_float(row.get("gamma"))
        item.trap_entry_round = _to_int(row.get("trap_entry_round"))
        item.event_sync_index_mean = _to_float(row.get("event_sync_index_mean"))
        item.reward_perturb_corr = _to_float(row.get("reward_perturb_corr"))
        item.rescue_margin = max(item.rescue_margin, _to_float(row.get("rescue_margin")) or 0.0)

    return risk_map


def _assign_risk_policy(item: SeedRisk) -> None:
    score = 0.0
    if item.new_l1:
        score += 100.0
    if item.broke:
        score += 60.0
    if item.rescued:
        score -= 20.0

    score += (item.healthy_gap * 100.0)

    if item.trap_entry_round is not None:
        if item.trap_entry_round <= 1500:
            score += 15.0
        elif item.trap_entry_round <= 2500:
            score += 8.0
        elif item.trap_entry_round <= 3500:
            score += 3.0

    if score >= 150:
        priority = "P0"
    elif score >= 100:
        priority = "P1"
    elif score >= 60:
        priority = "P2"
    else:
        priority = "P3"

    labels: list[str] = []
    if item.new_l1:
        labels.append("new_l1")
    if item.broke:
        labels.append("broke")
    if item.rescued:
        labels.append("rescued")
    item.label = "+".join(labels) if labels else "other"

    if item.new_l1:
        recommendation = "立即重跑 seed 並比對 L1 觸發條件；列入修復阻斷清單"
    elif item.broke and item.healthy_gap >= 0.24:
        recommendation = "優先檢查 s3 低谷與 trap 時點；納入下一輪 gate 觀察"
    elif item.broke:
        recommendation = "中優先排查事件/調度設定，確認是否可回到健康閾值以上"
    elif item.rescued:
        recommendation = "保留作為正向對照 seed，追蹤是否可穩定複現"
    else:
        recommendation = "持續觀察"

    item.score = score
    item.priority = priority
    item.recommendation = recommendation


def _rank_rows(items: list[SeedRisk]) -> list[dict[str, Any]]:
    ranked = sorted(items, key=lambda x: (-x.score, x.seed))
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(ranked, start=1):
        rows.append(
            {
                "rank": idx,
                "seed": item.seed,
                "label": item.label,
                "priority": item.priority,
                "risk_score": round(item.score, 3),
                "level": item.level,
                "s3": item.s3,
                "healthy_gap": round(item.healthy_gap, 6),
                "rescue_margin": round(item.rescue_margin, 6),
                "gamma": item.gamma,
                "trap_entry_round": item.trap_entry_round,
                "event_sync_index_mean": item.event_sync_index_mean,
                "reward_perturb_corr": item.reward_perturb_corr,
                "recommendation": item.recommendation,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_onepager(
    *,
    a3_row: dict[str, str],
    ranked_rows: list[dict[str, Any]],
    counts: dict[str, int],
) -> None:
    p0_p1 = [row for row in ranked_rows if row["priority"] in ("P0", "P1")]
    p0_rows = [row for row in ranked_rows if row["priority"] == "P0"]
    p1_rows = [row for row in ranked_rows if row["priority"] == "P1"]
    p2_broke_rows = [
        row for row in ranked_rows if row["priority"] == "P2" and "broke" in str(row["label"])
    ]
    rescued_rows = [row for row in ranked_rows if "rescued" in str(row["label"])]

    def _seed_list(rows: list[dict[str, Any]], *, max_items: int = 8) -> str:
        if not rows:
            return "無"
        seeds = [str(row["seed"]) for row in rows[:max_items]]
        return ", ".join(seeds)

    lines: list[str] = []
    lines.append("# W1 D4 週會版一頁摘要 v1")
    lines.append("")
    lines.append(f"- generated_at_utc: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- A3 source: {D2_A3_TREND_ROW.relative_to(ROOT)}")
    lines.append(f"- D3 sources: {D3_NEW_L1.relative_to(ROOT)}, {D3_BROKE.relative_to(ROOT)}, {D3_RESCUED.relative_to(ROOT)}")
    lines.append("")

    lines.append("## 1) 主線健康快照（A3）")
    lines.append("")
    lines.append("| overall_pass | A1 L1 | A1 Healthy | A1 mean_s3 | pytest_pass_count |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        "| {overall} | {l1} | {healthy} | {mean_s3} | {pytest_pass} |".format(
            overall=a3_row.get("a3_overall_pass"),
            l1=a3_row.get("a1_l1"),
            healthy=a3_row.get("a1_healthy"),
            mean_s3=a3_row.get("a1_mean_s3"),
            pytest_pass=a3_row.get("a2_pytest_pass_count"),
        )
    )
    lines.append("")

    lines.append("## 2) B2 失敗樣態快照")
    lines.append("")
    lines.append("| new_l1 | broke | rescued | 高優先風險(P0+P1) |")
    lines.append("|---:|---:|---:|---:|")
    lines.append(f"| {counts['new_l1']} | {counts['broke']} | {counts['rescued']} | {len(p0_p1)} |")
    lines.append("")

    lines.append("## 3) 風險優先排序（Top 10）")
    lines.append("")
    lines.append("| rank | seed | label | s3 | gap | trap | score | priority | 建議處置 |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|---|---|")
    for row in ranked_rows[:10]:
        lines.append(
            "| {rank} | {seed} | {label} | {s3} | {gap} | {trap} | {score} | {priority} | {advice} |".format(
                rank=row["rank"],
                seed=row["seed"],
                label=row["label"],
                s3=_fmt_float(_to_float(row["s3"]), 4),
                gap=_fmt_float(_to_float(row["healthy_gap"]), 4),
                trap=row["trap_entry_round"] if row["trap_entry_round"] is not None else "NA",
                score=_fmt_float(_to_float(row["risk_score"]), 1),
                priority=row["priority"],
                advice=row["recommendation"],
            )
        )
    lines.append("")

    lines.append("## 4) 本週建議處置")
    lines.append("")
    lines.append(
        f"1. P0（立即）：優先處理阻斷風險 seeds（{_seed_list(p0_rows)}），先釐清為何直接落到 L1。"
    )
    lines.append(
        f"2. P1（高）：高風險監控 seeds（{_seed_list(p1_rows)}），納入本週重點回放。"
    )
    lines.append(
        f"3. P2（中）：broke 觀察 seeds（{_seed_list(p2_broke_rows)}），檢查是否可回到健康閾值。"
    )
    lines.append(
        f"4. 正向對照：rescued seeds（{_seed_list(rescued_rows)}）作為成功樣態保留，供後續比較。"
    )

    D4_ONEPAGER_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    a3_rows = _read_csv_rows(D2_A3_TREND_ROW)
    if not a3_rows:
        raise RuntimeError(f"No rows found in {D2_A3_TREND_ROW}")
    a3_row = a3_rows[0]

    new_l1_rows = _read_csv_rows(D3_NEW_L1)
    broke_rows = _read_csv_rows(D3_BROKE)
    rescued_rows = _read_csv_rows(D3_RESCUED)

    risk_map = _build_seed_risk_map(new_l1_rows, broke_rows, rescued_rows)
    items = list(risk_map.values())
    for item in items:
        _assign_risk_policy(item)

    ranked_rows = _rank_rows(items)
    _write_csv(D4_RISK_PRIORITY_CSV, ranked_rows)

    counts = {
        "new_l1": len(new_l1_rows),
        "broke": len(broke_rows),
        "rescued": len(rescued_rows),
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "a3_trend_row": str(D2_A3_TREND_ROW.relative_to(ROOT)),
            "new_l1_table": str(D3_NEW_L1.relative_to(ROOT)),
            "broke_table": str(D3_BROKE.relative_to(ROOT)),
            "rescued_table": str(D3_RESCUED.relative_to(ROOT)),
        },
        "counts": counts,
        "high_priority": {
            "p0": [r["seed"] for r in ranked_rows if r["priority"] == "P0"],
            "p1": [r["seed"] for r in ranked_rows if r["priority"] == "P1"],
        },
        "outputs": {
            "risk_priority_csv": str(D4_RISK_PRIORITY_CSV.relative_to(ROOT)),
            "weekly_onepager_md": str(D4_ONEPAGER_MD.relative_to(ROOT)),
        },
    }
    D4_SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_onepager(a3_row=a3_row, ranked_rows=ranked_rows, counts=counts)

    print(f"Wrote: {D4_RISK_PRIORITY_CSV.relative_to(ROOT)}")
    print(f"Wrote: {D4_SUMMARY_JSON.relative_to(ROOT)}")
    print(f"Wrote: {D4_ONEPAGER_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
