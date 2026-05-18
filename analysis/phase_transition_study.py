"""phase_transition_study.py — z_exploring 相變臨界點掃描

研究假說：當 z_exploring 從負值越過 0（ReLU 開啟），RE 曲線會出現：
  (A) 「先升後降的甜點區」→ 向心力暫時強化軌道
  (B) 「直接下滑」→ 向心力作為耗散力，消耗旋轉動能

實驗設計：
  - 鎖定 trial_33（z_exploring_baseline ≈ -0.060，最接近相變閾值的候選）
  - 固定其他 8 個維度，線性掃描 randomness ∈ [-1.0, +1.0]
  - 記錄每個掃描點的：RE、mutation_strength、z_exploring、軌道特性
  - 標記 relu 臨界點（z_exploring = 0 的交叉位置）

SDD 合規性：
  - 僅 import evolution.replicator_dynamics（純函式，無 I/O）
  - 不 import simulation/（架構不變條件）
  - 輸出：stdout + outputs/phase_transition_randomness_scan.json

用法：
    ./venv/bin/python -m analysis.phase_transition_study
    ./venv/bin/python analysis/phase_transition_study.py
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

# ---------------------------------------------------------------------------
# 模擬常數（對齊 dream_seed_finder.py v1.0）
# ---------------------------------------------------------------------------
_T_STEPS     = 1000
_TAIL_START  = 500
_WARMUP_END  = 200   # 前 200 步視為暖機期
_WIN         = 1.0
_LOSS        = 1.2
_MU          = 0.05
_DT          = 1.0
_RE_L3_THR   = 5000.0

# ---------------------------------------------------------------------------
# trial_33 基準（固定 8 個維度，掃描 randomness）
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

# relu 臨界點：z_exploring = 0 ↔ randomness = -(stability_seeking + curiosity)
_RELU_THRESHOLD_RANDOMNESS = -(
    _TRIAL33_BASE["stability_seeking"] + _TRIAL33_BASE["curiosity"]
)  # ≈ -0.2919


def _project_z(traits: dict) -> tuple[float, float, float]:
    z_exp  = (traits["impulsiveness"] + traits["assertiveness"] + traits["optimism"]) / 3.0
    z_con  = (traits["risk_aversion"] + traits["suspicion"]     + traits["endurance"]) / 3.0
    z_expl = (traits["randomness"] + traits["stability_seeking"] + traits["curiosity"]) / 3.0
    return float(z_exp), float(z_con), float(z_expl)


def _simulate_full(
    z_exp: float,
    z_con: float,
    z_expl: float,
) -> dict:
    """模擬並回傳完整診斷（RE、軌道特性、mutation_strength）。"""
    simplex: dict[str, float] = {
        "expanding": 1.0 / 3,
        "contracting": 1.0 / 3,
        "exploring": 1.0 / 3,
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

    # RE（尾段 FFT）
    tail = arr[_TAIL_START:]
    rfft = np.fft.rfft(tail)
    re_tail = float(np.sum(np.abs(rfft[1:]) ** 2))

    # 暖機段 RE（前 200 步）
    warmup = arr[:_WARMUP_END]
    rfft_w = np.fft.rfft(warmup)
    re_warmup = float(np.sum(np.abs(rfft_w[1:]) ** 2))

    # 軌道振幅（peak-to-peak in tail）
    amplitude = float(np.ptp(tail))

    # 平均 mutation_strength
    avg_mutation = float(np.mean(mutation_strengths))

    return {
        "re_tail":      re_tail,
        "re_warmup":    re_warmup,
        "amplitude":    amplitude,
        "avg_mutation": avg_mutation,
        "is_level3":    re_tail >= _RE_L3_THR,
    }


def run_scan(
    n_points: int = 81,
    verbose: bool = True,
) -> dict:
    """
    掃描 randomness ∈ [-1.0, +1.0]，回傳完整掃描結果。

    參數：
        n_points: 掃描點數（預設 81 = 每 0.025 一個點）
    """
    scan_values = np.linspace(-1.0, 1.0, n_points)
    results: list[dict] = []

    # 計算基準值（trial_33 原始 z_exploring）
    base_traits = {**_TRIAL33_BASE, "randomness": _TRIAL33_BASE.get("randomness", -0.47082668289346113)}
    base_traits_full = {**_TRIAL33_BASE, "randomness": -0.47082668289346113}
    _, _, z_expl_base = _project_z(base_traits_full)

    if verbose:
        print(f"\n  固定維度：stability_seeking={_TRIAL33_BASE['stability_seeking']:.4f}, "
              f"curiosity={_TRIAL33_BASE['curiosity']:.4f}")
        print(f"  ReLU 臨界點：randomness ≈ {_RELU_THRESHOLD_RANDOMNESS:.4f}  "
              f"（此點 z_exploring = 0）")
        print(f"  trial_33 基準：randomness=-0.4708, z_exploring={z_expl_base:.4f}\n")
        print(f"  {'randomness':>12}  {'z_exploring':>12}  {'relu':>6}  "
              f"{'RE_tail':>12}  {'RE_warmup':>12}  {'amplitude':>10}  {'mutation':>10}  L3")
        print("  " + "-" * 92)

    prev_relu = None
    crossing_idx = None

    for i, rnd_val in enumerate(scan_values):
        traits = {**_TRIAL33_BASE, "randomness": float(rnd_val)}
        z_exp, z_con, z_expl = _project_z(traits)
        relu_active = z_expl > 0.0

        # 標記臨界點交叉
        if prev_relu is not None and relu_active != prev_relu:
            crossing_idx = i
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
            cross_mark = " ← CROSSING" if crossing_idx == i else ""
            print(f"  {rnd_val:>12.4f}  {z_expl:>12.4f}  {relu_str:>6}  "
                  f"{res['re_tail']:>12.0f}  {res['re_warmup']:>12.0f}  "
                  f"{res['amplitude']:>10.4f}  {res['avg_mutation']:>10.5f}  {l3_str}{cross_mark}")

    # 分析結果
    analysis = _analyze_results(results)
    if verbose:
        _print_analysis(analysis)

    return {
        "scan_target": "randomness",
        "fixed_dimensions": _TRIAL33_BASE,
        "relu_threshold_randomness": _RELU_THRESHOLD_RANDOMNESS,
        "n_points": n_points,
        "results": results,
        "analysis": analysis,
    }


def _analyze_results(results: list[dict]) -> dict:
    """從掃描結果萃取相變特性。"""
    # 1. relu 臨界點前後的平均 RE
    pre_relu  = [r for r in results if not r["relu_active"]]
    post_relu = [r for r in results if r["relu_active"]]

    re_pre_mean  = float(np.mean([r["re_tail"] for r in pre_relu]))  if pre_relu  else 0.0
    re_post_mean = float(np.mean([r["re_tail"] for r in post_relu])) if post_relu else 0.0

    # 2. Level 3 通過範圍
    l3_passes = [r for r in results if r["is_level3"]]
    l3_range  = (
        float(l3_passes[0]["randomness"]),
        float(l3_passes[-1]["randomness"]),
    ) if l3_passes else (float("nan"), float("nan"))

    # 3. RE 最高點（甜點）
    max_re_record = max(results, key=lambda r: r["re_tail"])

    # 4. 尋找相變附近是否有「先升後降」的甜點結構
    # 取 relu 臨界點附近 ±10 個點做局部分析
    crossing_vals = [r for r in results if abs(r["z_exploring"]) < 0.05]
    if crossing_vals:
        re_near_crossing = [r["re_tail"] for r in crossing_vals]
        has_sweetspot = (max(re_near_crossing) > re_pre_mean * 1.1 and
                         re_post_mean < re_pre_mean * 0.8)
    else:
        has_sweetspot = False

    # 5. 系統類型判斷
    if re_post_mean < re_pre_mean * 0.5:
        system_type = "dissipative"  # 耗散系統：向心力消耗動能
        system_note = "mutation 作為耗散力，RE 在 relu 開啟後顯著下降"
    elif re_post_mean > re_pre_mean * 1.2:
        system_type = "resonant"     # 共振系統：向心力暫時強化軌道
        system_note = "mutation 作為共振力，RE 在 relu 開啟後上升"
    else:
        system_type = "neutral"      # 中性：弱耦合
        system_note = "mutation 影響較小（弱耦合模式）"

    return {
        "re_pre_relu_mean":  round(re_pre_mean, 1),
        "re_post_relu_mean": round(re_post_mean, 1),
        "re_ratio_post_to_pre": round(re_post_mean / re_pre_mean, 4) if re_pre_mean > 0 else 0.0,
        "level3_pass_range":  l3_range,
        "level3_pass_count":  len(l3_passes),
        "max_re_at_randomness": float(max_re_record["randomness"]),
        "max_re_value":         round(max_re_record["re_tail"], 1),
        "has_sweetspot_near_crossing": has_sweetspot,
        "system_type":  system_type,
        "system_note":  system_note,
    }


def _print_analysis(analysis: dict) -> None:
    print("\n" + "=" * 80)
    print("  相變分析結果")
    print("=" * 80)
    print(f"\n  ReLU 開啟前（z_exploring < 0）：平均 RE = {analysis['re_pre_relu_mean']:.0f}")
    print(f"  ReLU 開啟後（z_exploring > 0）：平均 RE = {analysis['re_post_relu_mean']:.0f}")
    print(f"  後/前 RE 比值：{analysis['re_ratio_post_to_pre']:.4f}")
    print(f"\n  Level 3 通過點數：{analysis['level3_pass_count']}")
    if analysis['level3_pass_count'] > 0:
        print(f"  Level 3 通過範圍：randomness ∈ [{analysis['level3_pass_range'][0]:.4f}, "
              f"{analysis['level3_pass_range'][1]:.4f}]")
    print(f"\n  RE 最大值：{analysis['max_re_value']:.0f}  "
          f"（randomness = {analysis['max_re_at_randomness']:.4f}）")
    print(f"\n  臨界點附近是否存在甜點結構：{'是' if analysis['has_sweetspot_near_crossing'] else '否'}")
    print(f"\n  系統類型判讀：【{analysis['system_type'].upper()}】")
    print(f"  物理意義：{analysis['system_note']}")

    if analysis["system_type"] == "dissipative":
        print("\n  ✦ 結論：向心突變（Centripetal mutation）是耗散力。")
        print("    Level 3 旋轉是「純博弈驅動」的保守系統湧現，")
        print("    向心力介入反而破壞軌道。→ dream seeds 的 z_expl < 0 是物理必要條件。")
    elif analysis["system_type"] == "resonant":
        print("\n  ✦ 結論：向心突變是共振力。")
        print("    存在一個 z_expl > 0 的「甜點區」可提升旋轉能。")
        print("    → 可以嘗試設計 z_expl 輕微正值的新型 dream seeds。")
    else:
        print("\n  ✦ 結論：向心突變影響較弱，系統主要由 payoff matrix 驅動。")


def save_artifact(scan_result: dict, output_dir: Path | None = None) -> Path:
    if output_dir is None:
        output_dir = _ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "phase_transition_randomness_scan.json"

    payload = {
        "schema_version": "v2.0",
        "study_type":     "phase_transition_scan",
        "scan_target":    scan_result["scan_target"],
        "relu_threshold_randomness": scan_result["relu_threshold_randomness"],
        "n_points":       scan_result["n_points"],
        "analysis":       scan_result["analysis"],
        "scan_results": [
            {
                "randomness":   round(r["randomness"], 6),
                "z_exploring":  round(r["z_exploring"], 6),
                "relu_active":  r["relu_active"],
                "re_tail":      round(r["re_tail"], 1),
                "re_warmup":    round(r["re_warmup"], 1),
                "amplitude":    round(r["amplitude"], 6),
                "avg_mutation": round(r["avg_mutation"], 6),
                "is_level3":    r["is_level3"],
            }
            for r in scan_result["results"]
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def main() -> None:
    print("=" * 80)
    print("  Phase Transition Study — z_exploring 相變臨界點掃描")
    print("  研究假說：向心突變是耗散力（dissipative）還是共振力（resonant）？")
    print(f"  鎖定候選：trial_33（z_exploring_base ≈ -0.060）")
    print(f"  掃描維度：randomness ∈ [-1.0, +1.0]，81 點（步長 0.025）")
    print(f"  ReLU 臨界點：randomness ≈ {_RELU_THRESHOLD_RANDOMNESS:.4f}")
    print("=" * 80)

    t0 = time.perf_counter()
    scan_result = run_scan(n_points=81, verbose=True)
    elapsed = time.perf_counter() - t0

    out_path = save_artifact(scan_result)
    print(f"\n  完成時間：{elapsed:.1f}s")
    print(f"  Artifact：{out_path}")


if __name__ == "__main__":
    main()
