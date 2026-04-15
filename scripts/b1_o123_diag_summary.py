from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    (
        "42..101",
        ROOT / "outputs/b1_async_dispatch_poisson_w2000_t050_gate60_summary.json",
        ROOT / "outputs/pers_cal_baseline_gate60_summary.json",
    ),
    (
        "102..161",
        ROOT / "outputs/b1_async_dispatch_poisson_r008_w2000_t050_gate60_block102_161_summary.json",
        ROOT / "outputs/pers_cal_baseline_gate60_block102_161_summary.json",
    ),
    (
        "162..221",
        ROOT / "outputs/b1_async_dispatch_poisson_r008_w2000_t050_gate60_block162_221_summary.json",
        ROOT / "outputs/pers_cal_baseline_gate60_block162_221_summary.json",
    ),
    (
        "222..281",
        ROOT / "outputs/b1_async_dispatch_poisson_r008_w2000_t050_gate60_block222_281_summary.json",
        ROOT / "outputs/pers_cal_baseline_gate60_block222_281_summary.json",
    ),
]

EARLY_TRAP_CUTOFF = 4000
HEALTHY_THRESHOLD = 0.80


def quantile(values: list[int], prob: float) -> float | None:
    if not values:
        return None
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * prob))
    return float(arr[idx])


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def med(values: list[int]) -> float | None:
    if not values:
        return None
    return float(median(values))


def fmt(val: float | None, digits: int = 4) -> str:
    if val is None:
        return "NA"
    return f"{val:.{digits}f}"


def main() -> int:
    rows: list[dict[str, object]] = []

    for block, gate_path, baseline_path in CASES:
        if not gate_path.exists():
            raise FileNotFoundError(f"Missing gate summary: {gate_path}")
        if not baseline_path.exists():
            raise FileNotFoundError(f"Missing baseline summary: {baseline_path}")

        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

        outcomes = gate.get("outcomes", [])
        if not outcomes:
            raise RuntimeError(f"No outcomes in {gate_path}")

        seeds = [int(item["seed"]) for item in outcomes]
        lo, hi = min(seeds), max(seeds)
        if f"{lo}..{hi}" != block:
            raise RuntimeError(f"Seed block mismatch: expected {block}, got {lo}..{hi} ({gate_path})")

        baseline_healthy = {
            int(item["seed"]): (float(item.get("s3", 0.0)) >= HEALTHY_THRESHOLD)
            for item in baseline.get("outcomes", [])
        }

        new_l1_seeds = [
            int(item["seed"])
            for item in outcomes
            if baseline_healthy.get(int(item["seed"]), False) and int(item.get("level", 0)) == 1
        ]
        new_l1_set = set(new_l1_seeds)

        all_sync = [float(item.get("event_sync_index_mean", 0.0)) for item in outcomes]
        all_corr = [float(item.get("reward_perturb_corr", 0.0)) for item in outcomes]
        all_trap = [int(item["trap_entry_round"]) for item in outcomes if item.get("trap_entry_round") is not None]

        group_new = [item for item in outcomes if int(item["seed"]) in new_l1_set]
        group_other = [item for item in outcomes if int(item["seed"]) not in new_l1_set]

        new_sync = [float(item.get("event_sync_index_mean", 0.0)) for item in group_new]
        other_sync = [float(item.get("event_sync_index_mean", 0.0)) for item in group_other]
        new_corr = [float(item.get("reward_perturb_corr", 0.0)) for item in group_new]
        other_corr = [float(item.get("reward_perturb_corr", 0.0)) for item in group_other]
        new_trap = [int(item["trap_entry_round"]) for item in group_new if item.get("trap_entry_round") is not None]
        other_trap = [int(item["trap_entry_round"]) for item in group_other if item.get("trap_entry_round") is not None]

        trap_nonnull = len(all_trap)
        early_share = None
        if trap_nonnull:
            early_count = sum(1 for value in all_trap if value <= EARLY_TRAP_CUTOFF)
            early_share = early_count / trap_nonnull

        delta_sync = None if (not new_sync or not other_sync) else (mean(new_sync) - mean(other_sync))
        delta_corr = None if (not new_corr or not other_corr) else (mean(new_corr) - mean(other_corr))
        delta_trap = None if (not new_trap or not other_trap) else (med(new_trap) - med(other_trap))

        rows.append(
            {
                "block": block,
                "n": len(outcomes),
                "new_l1": int(gate.get("new_l1", 0) or 0),
                "new_l1_seeds": ",".join(str(seed) for seed in sorted(new_l1_seeds)) if new_l1_seeds else "-",
                "o1_mean_sync": mean(all_sync),
                "o1_mean_corr": mean(all_corr),
                "o1_trap_median": med(all_trap),
                "o2_delta_sync_new_minus_others": delta_sync,
                "o2_delta_corr_new_minus_others": delta_corr,
                "o2_delta_trap_median_new_minus_others": delta_trap,
                "o3_trap_p25": quantile(all_trap, 0.25),
                "o3_trap_p50": quantile(all_trap, 0.50),
                "o3_early_trap_share_le_4000": early_share,
                "o3_trap_nonnull": trap_nonnull,
                "gate_overall_pass": bool(gate.get("gate", {}).get("overall_pass", False)),
                "gate_file": str(gate_path.relative_to(ROOT)),
            }
        )

    out_csv = ROOT / "outputs/b1_o123_diagnostic_summary.csv"
    out_json = ROOT / "outputs/b1_o123_diagnostic_summary.json"
    out_md = ROOT / "outputs/b1_o123_diagnostic_summary.md"

    fieldnames = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append(
        "| Block | N | new_L1 | new_L1 seeds | mean_sync | mean_corr | trap_p50 | "
        "Delta sync(new-others) | Delta corr(new-others) | Delta trap_p50(new-others) | "
        "trap p25/p50 | early_trap_share<=4000 | trap_nonnull | Gate |"
    )
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|")
    for row in rows:
        lines.append(
            "| {block} | {n} | {new_l1} | {new_l1_seeds} | {ms} | {mc} | {tp50} | {ds} | {dc} | {dt} | {tp25}/{tp50b} | {es} | {tn} | {gate} |".format(
                block=row["block"],
                n=row["n"],
                new_l1=row["new_l1"],
                new_l1_seeds=row["new_l1_seeds"],
                ms=fmt(row["o1_mean_sync"], 4),
                mc=fmt(row["o1_mean_corr"], 4),
                tp50=fmt(row["o1_trap_median"], 1),
                ds=fmt(row["o2_delta_sync_new_minus_others"], 4),
                dc=fmt(row["o2_delta_corr_new_minus_others"], 4),
                dt=fmt(row["o2_delta_trap_median_new_minus_others"], 1),
                tp25=fmt(row["o3_trap_p25"], 1),
                tp50b=fmt(row["o3_trap_p50"], 1),
                es=fmt(row["o3_early_trap_share_le_4000"], 3),
                tn=row["o3_trap_nonnull"],
                gate="PASS" if row["gate_overall_pass"] else "FAIL",
            )
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_csv.relative_to(ROOT)}")
    print(f"Wrote: {out_json.relative_to(ROOT)}")
    print(f"Wrote: {out_md.relative_to(ROOT)}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
