"""symmetry_breaking_study.py — Exploring 組對稱破缺（Symmetry Breaking）研究

研究目標：打破 Exploring 組的 1/3 等權對稱性，測試：
  (1) curiosity（好奇心）與 randomness（隨機性）是否對 RE 有不同的物理語義
  (2) 非對稱加權能否讓 curiosity 成為主導 RE 波動的「性格引擎」
  (3) 甜點區範圍是否因此平移（更耐受 curiosity 變異）

實驗配置：
  對稱基準  (SYM):  curiosity=1/3, randomness=1/3, stability_seeking=1/3
  破缺實驗  (ASY):  curiosity=0.5,  randomness=0.3,  stability_seeking=0.2

SDD 合規性：
  - 僅 import evolution/（pure functions，無 I/O）
  - 不 import simulation/（架構不變條件）
  - 輸出：stdout + outputs/symmetry_breaking_study.json

用法：
    ./venv/bin/python -m analysis.symmetry_breaking_study
    ./venv/bin/python analysis/symmetry_breaking_study.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from evolution.replicator_dynamics import personality_guided_replicator_step
from evolution.personality_mapping import (
    WEIGHTS_SYMMETRIC,
    WEIGHTS_EXPLORING_ASYMMETRIC,
    compute_z_signals,
)
from analysis.dream_sensitivity import (
    _DREAM_CANDIDATES,
    _ALL_TRAITS,
    _TRAIT_GROUP_OF,
    _T_STEPS,
    _TAIL_START,
    _DELTA,
    _WIN,
    _LOSS,
    _MU,
    _DT,
    _simulate_re,
    _project_z,
    compute_sensitivity_matrix,
)

# ---------------------------------------------------------------------------
# 掃描常數（與 phase_transition_study.py 對齊）
# ---------------------------------------------------------------------------
_WARMUP_END  = 200
_RE_L3_THR   = 5000.0

# ---------------------------------------------------------------------------
# trial_33 固定維度（掃描 randomness，驗證甜點區平移）
# ---------------------------------------------------------------------------
_TRIAL33_BASE: dict[str, float] = {
    "impulsiveness":    -0.030906531686155037,
    "assertiveness":     0.295543436977969,
    "optimism":          0.1533032620442824,
    "risk_aversion":     0.19198316891043887,
    "suspicion":         0.6008560293248041,
    "endurance":        -0.1640099405326949,
    # randomness: SCAN TARGET
    "stability_seeking": 0.2738254878666811,
    "curiosity":         0.018102349788193417,
}

# ReLU 臨界點計算：z_exploring = 0 ↔ 加權和 = 0
# SYM: (randomness + stability_seeking + curiosity) / 3 = 0
#   → randomness = -(stability_seeking + curiosity) = -0.2919
_RELU_THRESHOLD_SYM = -(
    _TRIAL33_BASE["stability_seeking"] + _TRIAL33_BASE["curiosity"]
)  # ≈ -0.2919

# ASY: 0.5*curiosity + 0.3*randomness + 0.2*stability_seeking = 0
#   → randomness = -(0.5*curiosity + 0.2*stability_seeking) / 0.3
_w_asy = WEIGHTS_EXPLORING_ASYMMETRIC["exploring"]
_RELU_THRESHOLD_ASY = -(
    _w_asy["curiosity"] * _TRIAL33_BASE["curiosity"]
    + _w_asy["stability_seeking"] * _TRIAL33_BASE["stability_seeking"]
) / _w_asy["randomness"]  # ≈ -0.2127


# ---------------------------------------------------------------------------
# 單點模擬（帶完整診斷）
# ---------------------------------------------------------------------------
def _simulate_full(z_exp: float, z_con: float, z_expl: float) -> dict:
    simplex: dict[str, float] = {
        "expanding":   1.0 / 3,
        "contracting": 1.0 / 3,
        "exploring":   1.0 / 3,
    }
    traj_exp: list[float] = []
    mutation_strengths: list[float] = []

    for _ in range(_T_STEPS):
        simplex, diag = personality_guided_replicator_step(
            simplex,
            z_expanding=z_exp,
            z_contracting=z_con,
            z_exploring=z_expl,
            win=_WIN,
            loss=_LOSS,
            mu=_MU,
            dt=_DT,
        )
        traj_exp.append(simplex["expanding"])
        mutation_strengths.append(diag["mutation_strength"])

    arr = np.array(traj_exp)
    tail = arr[_TAIL_START:]
    rfft = np.fft.rfft(tail)
    re_tail = float(np.sum(np.abs(rfft[1:]) ** 2))

    warmup = arr[:_WARMUP_END]
    rfft_w = np.fft.rfft(warmup)
    re_warmup = float(np.sum(np.abs(rfft_w[1:]) ** 2))

    return {
        "re_tail":      re_tail,
        "re_warmup":    re_warmup,
        "amplitude":    float(np.ptp(tail)),
        "avg_mutation": float(np.mean(mutation_strengths)),
        "is_level3":    re_tail >= _RE_L3_THR,
    }


# ---------------------------------------------------------------------------
# Phase Transition Scan（支援任意 weights）
# ---------------------------------------------------------------------------
def run_phase_scan(
    weights: dict,
    label: str,
    relu_threshold: float,
    n_points: int = 81,
    verbose: bool = True,
) -> dict:
    """掃描 randomness ∈ [-1.0, +1.0]，使用指定的 weights 計算 z_exploring。"""
    scan_values = np.linspace(-1.0, 1.0, n_points)
    results: list[dict] = []

    base_z_expl = compute_z_signals(
        {**_TRIAL33_BASE, "randomness": -0.47082668289346113},
        weights,
    )[2]

    if verbose:
        print(f"\n  ReLU 臨界點（{label}）：randomness ≈ {relu_threshold:.4f}")
        print(f"  trial_33 基準 z_exploring（{label}）= {base_z_expl:.4f}\n")
        print(f"  {'randomness':>12}  {'z_exploring':>12}  {'relu':>6}  "
              f"{'RE_tail':>12}  {'amplitude':>10}  L3")
        print("  " + "-" * 72)

    prev_relu = None
    for i, rnd_val in enumerate(scan_values):
        traits = {**_TRIAL33_BASE, "randomness": float(rnd_val)}
        z_exp, z_con, z_expl = compute_z_signals(traits, weights)
        relu_active = z_expl > 0.0

        crossing = (prev_relu is not None and relu_active != prev_relu)
        prev_relu = relu_active

        res = _simulate_full(z_exp, z_con, z_expl)
        record = {
            "randomness":   float(rnd_val),
            "z_exploring":  float(z_expl),
            "relu_active":  relu_active,
            "re_tail":      res["re_tail"],
            "re_warmup":    res["re_warmup"],
            "amplitude":    res["amplitude"],
            "avg_mutation": res["avg_mutation"],
            "is_level3":    res["is_level3"],
        }
        results.append(record)

        if verbose:
            relu_str = "ON " if relu_active else "off"
            l3_str   = "✓" if res["is_level3"] else "✗"
            cross_mark = " ← CROSSING" if crossing else ""
            print(f"  {rnd_val:>12.4f}  {z_expl:>12.4f}  {relu_str:>6}  "
                  f"{res['re_tail']:>12.0f}  {res['amplitude']:>10.4f}  {l3_str}{cross_mark}")

    # 分析
    pre_relu  = [r for r in results if not r["relu_active"]]
    post_relu = [r for r in results if r["relu_active"]]
    re_pre  = float(np.mean([r["re_tail"] for r in pre_relu]))  if pre_relu  else 0.0
    re_post = float(np.mean([r["re_tail"] for r in post_relu])) if post_relu else 0.0
    max_rec = max(results, key=lambda r: r["re_tail"])

    # 甜點寬度：RE > 22,000 的連續區間
    sweetspot_rnd = [r["randomness"] for r in results if r["re_tail"] > 22000 and not r["relu_active"]]
    sweetspot_range = (
        (float(sweetspot_rnd[0]), float(sweetspot_rnd[-1]))
        if sweetspot_rnd else None
    )

    return {
        "label":          label,
        "relu_threshold": relu_threshold,
        "base_z_expl":    base_z_expl,
        "re_pre_mean":    round(re_pre, 1),
        "re_post_mean":   round(re_post, 1),
        "re_ratio":       round(re_post / re_pre, 4) if re_pre > 0 else 0.0,
        "max_re":         round(max_rec["re_tail"], 1),
        "max_re_randomness": max_rec["randomness"],
        "sweetspot_range": sweetspot_range,
        "results":        results,
    }


# ---------------------------------------------------------------------------
# 敏感度對比：SYM vs ASY
# ---------------------------------------------------------------------------
def run_sensitivity_comparison(verbose: bool = True) -> dict:
    """對三個候選分別計算 SYM 與 ASY 權重下的敏感度，並輸出對比。"""

    if verbose:
        print("\n" + "=" * 80)
        print("  Part 1 — 敏感度掃描：SYM（對稱）vs ASY（破缺）")
        print("=" * 80)

    # SYM（不傳 weights → 預設對稱）
    if verbose:
        print("\n[SYM - 對稱基準 1/3, 1/3, 1/3]")
    sym_matrix = compute_sensitivity_matrix(verbose=verbose, weights=None)

    # ASY
    if verbose:
        print("\n[ASY - 破缺：curiosity=0.5, randomness=0.3, stability_seeking=0.2]")
    asy_matrix = compute_sensitivity_matrix(
        verbose=verbose,
        weights=WEIGHTS_EXPLORING_ASYMMETRIC,
    )

    # 橫向對比表格
    if verbose:
        print("\n" + "=" * 80)
        print("  Exploring 組特質敏感度對比")
        print("  欄：|dRE/dTrait|  對稱基準 vs 破缺實驗")
        print("=" * 80)
        exploring_traits = ["randomness", "stability_seeking", "curiosity"]
        header = f"  {'Trait':<22}  {'SYM avg':>10}  {'ASY avg':>10}  {'ASY/SYM':>8}  {'主導變化'}"
        print(header)
        print("  " + "-" * 65)

        for trait in exploring_traits:
            sym_avg = float(np.mean([
                abs(c["sensitivity"][trait]) for c in sym_matrix["candidates"]
            ]))
            asy_avg = float(np.mean([
                abs(c["sensitivity"][trait]) for c in asy_matrix["candidates"]
            ]))
            ratio = asy_avg / sym_avg if sym_avg > 0 else float("inf")
            direction = "▲" if ratio > 1.1 else ("▼" if ratio < 0.9 else "≈")
            print(f"  {trait:<22}  {sym_avg:>10.0f}  {asy_avg:>10.0f}  "
                  f"{ratio:>8.3f}  {direction} ×{ratio:.2f}")

        # 排名對比（以 trial_33 為代表）
        def _get_expl_rank(matrix: dict) -> list[tuple[str, float]]:
            c33 = next(c for c in matrix["candidates"] if c["trial_id"] == 33)
            return [(t, abs(c33["sensitivity"][t])) for t in exploring_traits]

        print("\n  trial_33 Exploring 組排名（|dRE/dTrait| 降冪）：")
        sym_rank = sorted(_get_expl_rank(sym_matrix), key=lambda x: x[1], reverse=True)
        asy_rank = sorted(_get_expl_rank(asy_matrix), key=lambda x: x[1], reverse=True)
        print(f"  {'Rank':<6} {'SYM 特質':<22} {'SYM 值':>10}  {'ASY 特質':<22} {'ASY 值':>10}")
        print("  " + "-" * 75)
        for i, ((st, sv), (at, av)) in enumerate(zip(sym_rank, asy_rank), 1):
            print(f"  #{i:<5} {st:<22} {sv:>10.0f}  {at:<22} {av:>10.0f}")

    return {"sym": sym_matrix, "asy": asy_matrix}


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 80)
    print("  Symmetry Breaking Study — Exploring 組對稱破缺實驗")
    print("  SYM：[1/3, 1/3, 1/3]  →  ASY：[curiosity=0.5, randomness=0.3, stability=0.2]")
    print("=" * 80)

    t0 = time.perf_counter()

    # Part 1：敏感度對比
    sens_result = run_sensitivity_comparison(verbose=True)

    # Part 2：相變掃描對比（SYM vs ASY）
    print("\n" + "=" * 80)
    print("  Part 2 — Phase Transition Scan：SYM vs ASY（trial_33, randomness sweep）")
    print("=" * 80)

    print("\n[SYM] 掃描中…")
    sym_scan = run_phase_scan(
        weights=WEIGHTS_SYMMETRIC,
        label="SYM",
        relu_threshold=_RELU_THRESHOLD_SYM,
        n_points=81,
        verbose=False,
    )

    print("[ASY] 掃描中…")
    asy_scan = run_phase_scan(
        weights=WEIGHTS_EXPLORING_ASYMMETRIC,
        label="ASY",
        relu_threshold=_RELU_THRESHOLD_ASY,
        n_points=81,
        verbose=False,
    )

    # Part 2 對比摘要表格
    print("\n" + "=" * 80)
    print("  Phase Transition 對比摘要")
    print("=" * 80)
    print(f"\n  {'指標':<32}  {'SYM':>12}  {'ASY':>12}  {'變化'}")
    print("  " + "-" * 70)

    rows = [
        ("ReLU 臨界點 randomness",
         f"{_RELU_THRESHOLD_SYM:.4f}",
         f"{_RELU_THRESHOLD_ASY:.4f}",
         f"右移 {_RELU_THRESHOLD_ASY - _RELU_THRESHOLD_SYM:+.4f}"),
        ("trial_33 基準 z_exploring",
         f"{sym_scan['base_z_expl']:.4f}",
         f"{asy_scan['base_z_expl']:.4f}",
         ""),
        ("RE_pre_relu 平均（relu=off）",
         f"{sym_scan['re_pre_mean']:.0f}",
         f"{asy_scan['re_pre_mean']:.0f}",
         f"{asy_scan['re_pre_mean'] - sym_scan['re_pre_mean']:+.0f}"),
        ("RE_post_relu 平均（relu=on）",
         f"{sym_scan['re_post_mean']:.0f}",
         f"{asy_scan['re_post_mean']:.0f}",
         f"{asy_scan['re_post_mean'] - sym_scan['re_post_mean']:+.0f}"),
        ("後/前 RE 比值",
         f"{sym_scan['re_ratio']:.4f}",
         f"{asy_scan['re_ratio']:.4f}",
         ""),
        ("RE 峰值",
         f"{sym_scan['max_re']:.0f}",
         f"{asy_scan['max_re']:.0f}",
         f"{asy_scan['max_re'] - sym_scan['max_re']:+.0f}"),
        ("RE 峰值對應 randomness",
         f"{sym_scan['max_re_randomness']:.4f}",
         f"{asy_scan['max_re_randomness']:.4f}",
         f"右移 {asy_scan['max_re_randomness'] - sym_scan['max_re_randomness']:+.4f}"),
        ("甜點區（RE>22000，relu=off）",
         str(sym_scan["sweetspot_range"]),
         str(asy_scan["sweetspot_range"]),
         ""),
    ]
    for label, sv, av, change in rows:
        print(f"  {label:<32}  {sv:>12}  {av:>12}  {change}")

    # 物理詮釋
    print("\n" + "=" * 80)
    print("  物理詮釋")
    print("=" * 80)

    threshold_shift = _RELU_THRESHOLD_ASY - _RELU_THRESHOLD_SYM
    print(f"\n  1. ReLU 臨界點右移 {threshold_shift:.4f}：")
    print(f"     curiosity 權重提高（0.5→），使 z_exploring 對 curiosity 更敏感。")
    print(f"     隨機性（randomness）對閾值的貢獻下降（0.3 vs 1/3），")
    print(f"     達到 z_expl=0 所需的 randomness 負值更小（閾值右移）。")

    sym_peak_expl = compute_z_signals(
        {**_TRIAL33_BASE, "randomness": sym_scan["max_re_randomness"]},
        WEIGHTS_SYMMETRIC,
    )[2]
    asy_peak_expl = compute_z_signals(
        {**_TRIAL33_BASE, "randomness": asy_scan["max_re_randomness"]},
        WEIGHTS_EXPLORING_ASYMMETRIC,
    )[2]

    print(f"\n  2. RE 峰值位置比較：")
    print(f"     SYM 峰值：randomness={sym_scan['max_re_randomness']:.4f}，"
          f" z_expl={sym_peak_expl:.4f}")
    print(f"     ASY 峰值：randomness={asy_scan['max_re_randomness']:.4f}，"
          f" z_expl={asy_peak_expl:.4f}")
    peak_shift = asy_scan["max_re_randomness"] - sym_scan["max_re_randomness"]
    if peak_shift > 0.02:
        print(f"     → 甜點區右移（{peak_shift:+.4f}）：系統對 randomness 的正值更耐受。")
    elif peak_shift < -0.02:
        print(f"     → 甜點區左移（{peak_shift:+.4f}）：需要更負的 randomness 才能達到峰值。")
    else:
        print(f"     → 甜點區位置基本不變（{peak_shift:+.4f}）。")

    print(f"\n  3. RE 峰值強度：SYM={sym_scan['max_re']:.0f}  ASY={asy_scan['max_re']:.0f}")
    re_delta = asy_scan["max_re"] - sym_scan["max_re"]
    if re_delta > 200:
        print(f"     → 破缺使峰值 RE 提升 {re_delta:.0f}（curiosity 主導效應增益）。")
    elif re_delta < -200:
        print(f"     → 破缺使峰值 RE 下降 {-re_delta:.0f}（randomness 降權削弱了甜點強度）。")
    else:
        print(f"     → 峰值 RE 幾乎不變（{re_delta:+.0f}），甜點結構穩健。")

    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"\n  完成時間：{elapsed:.1f}s")

    # Artifact
    out_path = _save_artifact(sens_result, sym_scan, asy_scan)
    print(f"  Artifact：{out_path}")


def _save_artifact(
    sens_result: dict,
    sym_scan: dict,
    asy_scan: dict,
    output_dir: Path | None = None,
) -> Path:
    if output_dir is None:
        output_dir = _ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "symmetry_breaking_study.json"

    def _sens_summary(matrix: dict) -> list:
        return [
            {
                "trial_id": c["trial_id"],
                "z_exploring_baseline": round(c["z_exploring_baseline"], 6),
                "exploring_sensitivity": {
                    t: round(c["sensitivity"][t], 2)
                    for t in ["randomness", "stability_seeking", "curiosity"]
                },
                "abs_rank_top3": c["abs_rank"][:3],
            }
            for c in matrix["candidates"]
        ]

    def _scan_compact(scan: dict) -> dict:
        return {
            "label":           scan["label"],
            "relu_threshold":  round(scan["relu_threshold"], 6),
            "base_z_expl":     round(scan["base_z_expl"], 6),
            "re_pre_mean":     scan["re_pre_mean"],
            "re_post_mean":    scan["re_post_mean"],
            "re_ratio":        scan["re_ratio"],
            "max_re":          scan["max_re"],
            "max_re_randomness": scan["max_re_randomness"],
            "sweetspot_range": scan["sweetspot_range"],
            "scan_results": [
                {
                    "randomness":  round(r["randomness"], 4),
                    "z_exploring": round(r["z_exploring"], 6),
                    "relu_active": r["relu_active"],
                    "re_tail":     round(r["re_tail"], 1),
                    "is_level3":   r["is_level3"],
                }
                for r in scan["results"]
            ],
        }

    payload = {
        "schema_version": "v2.0",
        "study_type": "symmetry_breaking_exploring_weights",
        "weights": {
            "sym": {
                "randomness": round(1/3, 6),
                "stability_seeking": round(1/3, 6),
                "curiosity": round(1/3, 6),
            },
            "asy": {
                "curiosity": 0.5,
                "randomness": 0.3,
                "stability_seeking": 0.2,
            },
        },
        "sensitivity": {
            "sym": _sens_summary(sens_result["sym"]),
            "asy": _sens_summary(sens_result["asy"]),
        },
        "phase_transition": {
            "sym": _scan_compact(sym_scan),
            "asy": _scan_compact(asy_scan),
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


if __name__ == "__main__":
    main()
