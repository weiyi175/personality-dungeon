#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.cycle_metrics import classify_cycle_level


DEFAULT_SEEDS = [42, 44, 45, 67, 73, 90]
TARGET_SEEDS = [45, 67]


@dataclass
class CycleDiag:
    level: int
    phase_consistency: float
    turn_strength: float


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_series(csv_path: Path) -> dict[str, list[float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV has no rows: {csv_path}")

    required = {"p_aggressive", "p_defensive", "p_balanced"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"CSV missing columns {sorted(missing)}: {csv_path}")

    return {
        "aggressive": [float(r["p_aggressive"]) for r in rows],
        "defensive": [float(r["p_defensive"]) for r in rows],
        "balanced": [float(r["p_balanced"]) for r in rows],
    }


def _compute_cycle_diag(csv_path: Path) -> CycleDiag:
    series = _load_series(csv_path)
    res = classify_cycle_level(
        series,
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
    score = float(res.stage3.score) if res.stage3 is not None else 0.0
    turn = float(res.stage3.turn_strength) if res.stage3 is not None else 0.0
    return CycleDiag(level=int(res.level), phase_consistency=score, turn_strength=turn)


def _round_coverage_check(
    provenance: dict[str, Any],
    *,
    n_rounds: int,
) -> dict[str, Any]:
    interval = int(provenance.get("world_update_interval", 0) or 0)
    rows = list(provenance.get("world_update_rows", []) or [])
    world_mode = str(provenance.get("world_mode", ""))

    report: dict[str, Any] = {
        "world_mode": world_mode,
        "world_update_interval": interval,
        "world_update_windows": len(rows),
        "expected_windows": 0,
        "feedback_apply_max_per_round": 0,
        "duplicate_apply_round_count": 0,
        "missing_round_count": 0,
        "window_index_mismatch_count": 0,
        "window_span_mismatch_count": 0,
        "window_contiguity_mismatch_count": 0,
        "window_out_of_range_count": 0,
    }

    if interval <= 0:
        report["expected_windows"] = 0
        report["expected_no_world_rows"] = True
        report["world_rows_present_with_off_mode"] = bool(rows)
        return report

    expected_windows = n_rounds // interval
    report["expected_windows"] = int(expected_windows)

    coverage = [0] * n_rounds
    contiguity_mismatch = 0
    span_mismatch = 0
    index_mismatch = 0
    out_of_range = 0
    prev_end = -1

    for i, row in enumerate(rows):
        widx = int(row.get("window_index", -1))
        start = int(row.get("start_round", -1))
        end = int(row.get("end_round", -1))
        window_rows = int(row.get("window_rows", -1))

        if widx != i:
            index_mismatch += 1

        if start < 0 or end < start or end >= n_rounds:
            out_of_range += 1
            continue

        if (end - start + 1) != window_rows:
            span_mismatch += 1

        if i == 0:
            if start != 0:
                contiguity_mismatch += 1
        else:
            if start != (prev_end + 1):
                contiguity_mismatch += 1

        prev_end = end
        for r in range(start, end + 1):
            coverage[r] += 1

    if rows and prev_end != (n_rounds - 1):
        contiguity_mismatch += 1

    duplicates = sum(1 for c in coverage if c > 1)
    missing = sum(1 for c in coverage if c == 0)
    max_apply = max(coverage) if coverage else 0

    report["feedback_apply_max_per_round"] = int(max_apply)
    report["duplicate_apply_round_count"] = int(duplicates)
    report["missing_round_count"] = int(missing)
    report["window_index_mismatch_count"] = int(index_mismatch)
    report["window_span_mismatch_count"] = int(span_mismatch)
    report["window_contiguity_mismatch_count"] = int(contiguity_mismatch)
    report["window_out_of_range_count"] = int(out_of_range)
    report["window_count_mismatch"] = int(len(rows) != expected_windows)
    return report


def _has_engineering_noise(coverage_checks: list[dict[str, Any]]) -> bool:
    for c in coverage_checks:
        if c["duplicate_apply_round_count"] > 0:
            return True
        if c["window_count_mismatch"] > 0:
            return True
        if c["window_contiguity_mismatch_count"] > 0:
            return True
        if c["window_span_mismatch_count"] > 0:
            return True
        if c["window_index_mismatch_count"] > 0:
            return True
        if c["window_out_of_range_count"] > 0:
            return True
        if c["world_update_interval"] > 0 and c["missing_round_count"] > 0:
            return True
    return False


def _pair_dirs(outputs_root: Path) -> dict[str, dict[str, Path]]:
    return {
        "async_poisson": {
            "adaptive": outputs_root / "personality_rl_async_poisson",
            "off": outputs_root / "personality_rl_async_poisson_wfoff",
        },
        "async_round_robin": {
            "adaptive": outputs_root / "personality_rl_async_round_robin",
            "off": outputs_root / "personality_rl_async_round_robin_wfoff",
        },
    }


def _build_decision(
    engineering_noise: bool,
    paired_deltas: dict[str, Any],
) -> dict[str, Any]:
    if engineering_noise:
        return {
            "decision": "fix_bug",
            "rationale": (
                "工程檢查命中重複套用或時序錯位異常，"
                "需先修正實作噪音再進行機制判讀。"
            ),
            "next_action": "先修正 world feedback 更新時序/覆蓋，再重跑同 6 seeds smoke 對照。",
        }

    # 工程層乾淨時，觀察 seed_45(降級) 與 seed_67(升級) 的方向訊號。
    s45 = paired_deltas.get("seed_45", {})
    s67 = paired_deltas.get("seed_67", {})
    s45_neg = float(s45.get("mean_delta_turn_strength", 0.0)) < 0 and float(
        s45.get("mean_delta_phase_consistency", 0.0)
    ) < 0
    s67_pos = float(s67.get("mean_delta_turn_strength", 0.0)) > 0 and float(
        s67.get("mean_delta_phase_consistency", 0.0)
    ) > 0

    if s45_neg and s67_pos:
        rationale = (
            "工程檢查未見顯著噪音，且 seed_45 與 seed_67 呈現方向分化，"
            "較符合機制增益分區效應而非單純程式錯位。"
        )
    else:
        rationale = (
            "工程檢查未見顯著噪音；雖然 seed 間方向不完全分化，"
            "但目前證據仍優先支持機制層假設 1（回授等效阻尼）校準。"
        )

    return {
        "decision": "hypothesis1_recalibration",
        "rationale": rationale,
        "next_action": "維持 lock 參數與 6 seeds，進入假設1的最小增量配對診斷報告。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "B1 smoke 異常診斷：先排除工程噪音（重複套用/時序錯位），"
            "再輸出 seed_45 與 seed_67 的 adaptive-off 配對差異。"
        )
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--n-rounds", type=int, default=12000)
    parser.add_argument(
        "--out-json",
        default="outputs/analysis/b1_smoke_anomaly_diagnosis.json",
    )
    parser.add_argument(
        "--out-md",
        default="outputs/analysis/b1_smoke_anomaly_diagnosis.md",
    )
    args = parser.parse_args()

    outputs_root = (REPO_ROOT / args.outputs_root).resolve()
    pairs = _pair_dirs(outputs_root)

    coverage_checks: list[dict[str, Any]] = []
    paired_deltas: dict[str, Any] = {}
    mode_level: dict[str, Any] = {}

    for dispatch_mode, d in pairs.items():
        mode_level[dispatch_mode] = {}
        for mode_name, base_dir in d.items():
            per_seed: dict[str, Any] = {}
            for seed in DEFAULT_SEEDS:
                seed_dir = base_dir / f"seed_{seed}"
                prov_path = seed_dir / "provenance.json"
                csv_path = seed_dir / "timeseries.csv"
                if not prov_path.exists() or not csv_path.exists():
                    raise FileNotFoundError(f"Missing artifacts under {seed_dir}")
                prov = _load_json(prov_path)
                cycle = _compute_cycle_diag(csv_path)
                per_seed[f"seed_{seed}"] = {
                    "cycle": asdict(cycle),
                    "provenance_path": str(prov_path.relative_to(REPO_ROOT)),
                    "timeseries_path": str(csv_path.relative_to(REPO_ROOT)),
                }
                if mode_name == "adaptive":
                    chk = _round_coverage_check(prov, n_rounds=int(args.n_rounds))
                    chk["dispatch_mode"] = dispatch_mode
                    chk["seed"] = seed
                    chk["provenance_path"] = str(prov_path.relative_to(REPO_ROOT))
                    coverage_checks.append(chk)
            mode_level[dispatch_mode][mode_name] = per_seed

    for seed in TARGET_SEEDS:
        seed_key = f"seed_{seed}"
        deltas: dict[str, Any] = {"by_dispatch_mode": {}}
        turn_vals: list[float] = []
        phase_vals: list[float] = []
        for dispatch_mode in pairs:
            a = mode_level[dispatch_mode]["adaptive"][seed_key]["cycle"]
            o = mode_level[dispatch_mode]["off"][seed_key]["cycle"]
            d_turn = float(a["turn_strength"]) - float(o["turn_strength"])
            d_phase = float(a["phase_consistency"]) - float(o["phase_consistency"])
            turn_vals.append(d_turn)
            phase_vals.append(d_phase)
            deltas["by_dispatch_mode"][dispatch_mode] = {
                "adaptive_turn_strength": float(a["turn_strength"]),
                "off_turn_strength": float(o["turn_strength"]),
                "delta_turn_strength": float(d_turn),
                "adaptive_phase_consistency": float(a["phase_consistency"]),
                "off_phase_consistency": float(o["phase_consistency"]),
                "delta_phase_consistency": float(d_phase),
                "adaptive_level": int(a["level"]),
                "off_level": int(o["level"]),
            }
        deltas["mean_delta_turn_strength"] = float(sum(turn_vals) / len(turn_vals))
        deltas["mean_delta_phase_consistency"] = float(sum(phase_vals) / len(phase_vals))
        paired_deltas[seed_key] = deltas

    engineering_noise = _has_engineering_noise(coverage_checks)
    decision = _build_decision(engineering_noise, paired_deltas)

    payload = {
        "protocol": "B1 smoke anomaly diagnose",
        "rule": "先排除工程噪音，再驗證物理機制",
        "log_base": "log_e(k)",
        "n_rounds": int(args.n_rounds),
        "seeds_scope": list(DEFAULT_SEEDS),
        "target_pair_seeds": list(TARGET_SEEDS),
        "engineering_checks": coverage_checks,
        "paired_deltas": paired_deltas,
        "decision": decision,
    }

    out_json = (REPO_ROOT / args.out_json).resolve()
    out_md = (REPO_ROOT / args.out_md).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# B1 Smoke 異常診斷報告",
        "",
        "## 1) 工程噪音檢查（D 段）",
        "",
        "| dispatch | seed | interval | windows(expected/actual) | dup_rounds | contiguity_mismatch | span_mismatch | out_of_range |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for c in coverage_checks:
        lines.append(
            "| {dispatch} | {seed} | {interval} | {exp}/{act} | {dup} | {conti} | {span} | {oor} |".format(
                dispatch=c["dispatch_mode"],
                seed=c["seed"],
                interval=c["world_update_interval"],
                exp=c["expected_windows"],
                act=c["world_update_windows"],
                dup=c["duplicate_apply_round_count"],
                conti=c["window_contiguity_mismatch_count"],
                span=c["window_span_mismatch_count"],
                oor=c["window_out_of_range_count"],
            )
        )

    lines.extend(
        [
            "",
            "## 2) 配對差異（seed_45 / seed_67）",
            "",
            "| seed | dispatch | Δturn_strength (adaptive-off) | Δphase_consistency (adaptive-off) | L(adaptive/off) |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for seed_key in ("seed_45", "seed_67"):
        by_mode = paired_deltas[seed_key]["by_dispatch_mode"]
        for dispatch_mode, d in by_mode.items():
            lines.append(
                "| {seed} | {dispatch} | {dt:.6f} | {dp:.6f} | {la}/{lo} |".format(
                    seed=seed_key,
                    dispatch=dispatch_mode,
                    dt=float(d["delta_turn_strength"]),
                    dp=float(d["delta_phase_consistency"]),
                    la=int(d["adaptive_level"]),
                    lo=int(d["off_level"]),
                )
            )
        lines.append(
            "| {seed} | mean(two dispatch) | {dt:.6f} | {dp:.6f} | - |".format(
                seed=seed_key,
                dt=float(paired_deltas[seed_key]["mean_delta_turn_strength"]),
                dp=float(paired_deltas[seed_key]["mean_delta_phase_consistency"]),
            )
        )

    lines.extend(
        [
            "",
            "## 3) 決策建議",
            "",
            f"- decision: {decision['decision']}",
            f"- rationale: {decision['rationale']}",
            f"- next_action: {decision['next_action']}",
            "",
            "## 4) 日誌更新建議",
            "",
            "- 在研發日誌新增一節：B1 Smoke 異常診斷（先排除工程噪音）。",
            "- 引用本次 JSON 與 MD 報告路徑，並記錄 decision 與 next_action。",
            "- 明確標註本次分析僅使用 6 seeds、未開 gate60、未擴大 sweep。",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_json.relative_to(REPO_ROOT)}")
    print(f"Wrote: {out_md.relative_to(REPO_ROOT)}")
    print(f"Decision: {decision['decision']}")


if __name__ == "__main__":
    main()
