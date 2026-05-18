"""dream_sensitivity.py — 9D Trait 敏感度掃描（有限差分法）

計算三個 Dream Candidate 在 9D 特質空間的 ΔRE/ΔTrait 敏感度矩陣。

方法：中央差分（Central Difference）
    sensitivity(trait_k) = (RE(trait_k + δ) - RE(trait_k - δ)) / (2δ)
    δ = 1e-4（足夠小以近似偏導數，足夠大以避免 float64 精度截斷誤差）

SDD 合規性：
    - 僅 import evolution.replicator_dynamics（純函式，無 I/O）
    - 不 import simulation/（架構不變條件）
    - 輸出：stdout 表格 + outputs/sensitivity_9d_dream_candidates.json

用法：
    ./venv/bin/python -m analysis.dream_sensitivity
    ./venv/bin/python analysis/dream_sensitivity.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# SDD: analysis/ 可 import evolution/（純函式），不得 import simulation/
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from evolution.replicator_dynamics import personality_guided_replicator_step
from evolution.personality_mapping import (
    WEIGHTS_SYMMETRIC,
    compute_z_signals as _compute_z_signals,
)

# ---------------------------------------------------------------------------
# 模擬常數（與 dream_seed_finder.py v1.0 及 test suite 完全對齊）
# ---------------------------------------------------------------------------
_T_STEPS    = 1000
_TAIL_START = 500
_DELTA      = 1e-4   # 中央差分步長
_WIN        = 1.0
_LOSS       = 1.2
_MU         = 0.05
_DT         = 1.0

# ---------------------------------------------------------------------------
# 9D 特質分組（對應 Model v2.0 / Enneagram 三組）
# ---------------------------------------------------------------------------
_TRAIT_GROUPS: dict[str, list[str]] = {
    "expanding":   ["impulsiveness", "assertiveness", "optimism"],
    "contracting": ["risk_aversion", "suspicion",     "endurance"],
    "exploring":   ["randomness",    "stability_seeking", "curiosity"],
}
_ALL_TRAITS: list[str] = (
    _TRAIT_GROUPS["expanding"] +
    _TRAIT_GROUPS["contracting"] +
    _TRAIT_GROUPS["exploring"]
)
_TRAIT_GROUP_OF: dict[str, str] = {
    t: g for g, ts in _TRAIT_GROUPS.items() for t in ts
}

# ---------------------------------------------------------------------------
# 嵌入式 Dream Candidates（來源：dream_final/dream_seeds_v2_9d.json）
# ---------------------------------------------------------------------------
_DREAM_CANDIDATES: list[dict] = [
    {
        "trial_id": 53,
        "RE_avg": 23528.286,
        "traits": {
            "impulsiveness":    0.640624046042338,
            "assertiveness":   -0.11007001793358212,
            "optimism":        -0.07242881844613819,
            "risk_aversion":    0.5566629668518146,
            "suspicion":        0.34253363378252655,
            "endurance":       -0.5518408878010823,
            "randomness":      -0.20883432296656573,
            "stability_seeking": 0.29683774568912275,
            "curiosity":       -0.41985189235564574,
        },
    },
    {
        "trial_id": 47,
        "RE_avg": 23074.547,
        "traits": {
            "impulsiveness":    0.03275959033652663,
            "assertiveness":   -0.24766858126978053,
            "optimism":         0.5848600132357374,
            "risk_aversion":    0.16277817309737516,
            "suspicion":        0.10140099164416747,
            "endurance":        0.1121221589927027,
            "randomness":      -0.27358803874915805,
            "stability_seeking": -0.12322496315054773,
            "curiosity":        0.0863380590116847,
        },
    },
    {
        "trial_id": 33,
        "RE_avg": 22364.656,
        "traits": {
            "impulsiveness":   -0.030906531686155037,
            "assertiveness":    0.295543436977969,
            "optimism":         0.1533032620442824,
            "risk_aversion":    0.19198316891043887,
            "suspicion":        0.6008560293248041,
            "endurance":       -0.1640099405326949,
            "randomness":      -0.47082668289346113,
            "stability_seeking": 0.2738254878666811,
            "curiosity":        0.018102349788193417,
        },
    },
]


# ---------------------------------------------------------------------------
# 核心計算函式
# ---------------------------------------------------------------------------

def _project_z(
    traits: dict,
    weights: dict | None = None,
) -> tuple[float, float, float]:
    """9D traits → (z_expanding, z_contracting, z_exploring)。

    weights 省略時使用 WEIGHTS_SYMMETRIC（對稱 1/3 基準）。
    """
    return _compute_z_signals(traits, weights)


def _simulate_re(z_exp: float, z_con: float, z_expl: float) -> float:
    """純引擎模擬 → 尾段旋轉能（RE）。無 I/O、無 logging。（z 訊號已預先計算）"""
    simplex: dict[str, float] = {
        "expanding": 1.0 / 3,
        "contracting": 1.0 / 3,
        "exploring": 1.0 / 3,
    }
    traj_exp: list[float] = []
    for _ in range(_T_STEPS):
        simplex, _ = personality_guided_replicator_step(
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

    tail = np.array(traj_exp[_TAIL_START:])
    rfft = np.fft.rfft(tail)
    return float(np.sum(np.abs(rfft[1:]) ** 2))


def compute_sensitivity_row(
    traits: dict,
    delta: float = _DELTA,
    weights: dict | None = None,
) -> dict[str, float]:
    """對一個 traits 向量計算所有 9 個特質的中央差分敏感度。"""
    row: dict[str, float] = {}
    for trait in _ALL_TRAITS:
        # 正向擾動（clamp 到 [-1, 1]）
        t_plus = {**traits, trait: float(np.clip(traits[trait] + delta, -1.0, 1.0))}
        re_plus = _simulate_re(*_project_z(t_plus, weights))

        # 負向擾動
        t_minus = {**traits, trait: float(np.clip(traits[trait] - delta, -1.0, 1.0))}
        re_minus = _simulate_re(*_project_z(t_minus, weights))

        row[trait] = (re_plus - re_minus) / (2.0 * delta)
    return row


def compute_sensitivity_matrix(
    candidates: list[dict] | None = None,
    delta: float = _DELTA,
    verbose: bool = True,
    weights: dict | None = None,
) -> dict:
    """
    計算敏感度矩陣。返回結構：
    {
        "delta": δ,
        "candidates": [
            {
                "trial_id": int,
                "RE_avg_baseline": float,
                "z_exploring_baseline": float,
                "sensitivity": {trait: dRE/dTrait, ...},
                "abs_rank": [(trait, |dRE/dTrait|), ...],   # 降冪排序
            },
            ...
        ]
    }
    """
    if candidates is None:
        candidates = _DREAM_CANDIDATES

    results: list[dict] = []
    total_sims = len(candidates) * len(_ALL_TRAITS) * 2
    done = 0

    for cand in candidates:
        trial_id = cand["trial_id"]
        traits = cand["traits"]
        _, _, z_expl = _project_z(traits, weights)

        if verbose:
            print(f"\n[trial_{trial_id}] RE_avg={cand['RE_avg']:.0f}  "
                  f"z_exploring={z_expl:.4f} (relu={'off' if z_expl <= 0 else 'ON'})")
            print(f"  {'Trait':<22} {'Group':<12} {'dRE/dTrait':>14}  {'|dRE/dTrait|':>14}")
            print("  " + "-" * 68)

        row = {}
        for trait in _ALL_TRAITS:
            t_plus  = {**traits, trait: float(np.clip(traits[trait] + delta, -1.0, 1.0))}
            t_minus = {**traits, trait: float(np.clip(traits[trait] - delta, -1.0, 1.0))}
            re_plus  = _simulate_re(*_project_z(t_plus, weights))
            re_minus = _simulate_re(*_project_z(t_minus, weights))
            sensitivity = (re_plus - re_minus) / (2.0 * delta)
            row[trait] = sensitivity
            done += 2

            if verbose:
                group = _TRAIT_GROUP_OF[trait]
                sign = "+" if sensitivity >= 0 else "-"
                bar_len = min(int(abs(sensitivity) / 500), 20)
                bar = sign * bar_len
                print(f"  {trait:<22} {group:<12} {sensitivity:>14.1f}  {abs(sensitivity):>14.1f}  {bar}")

        abs_rank = sorted(row.items(), key=lambda kv: abs(kv[1]), reverse=True)

        if verbose:
            print(f"\n  Top-3 影響力排名:")
            for rank_i, (t, s) in enumerate(abs_rank[:3], 1):
                print(f"    #{rank_i} {t:<22} |dRE/dTrait| = {abs(s):.1f}")

        results.append({
            "trial_id": trial_id,
            "RE_avg_baseline": cand["RE_avg"],
            "z_exploring_baseline": z_expl,
            "relu_active": z_expl > 0,
            "sensitivity": row,
            "abs_rank": [(t, round(abs(s), 2)) for t, s in abs_rank],
        })

    return {"delta": delta, "candidates": results}


def print_heatmap(matrix: dict) -> None:
    """輸出跨候選的橫向比較 Heatmap 表格。"""
    candidates = matrix["candidates"]
    trial_ids = [f"trial_{c['trial_id']}" for c in candidates]

    print("\n" + "=" * 80)
    print("  9D Trait 敏感度 Heatmap  |dRE/dTrait|")
    print("  （數值越大 = 對 Level-3 旋轉能影響越強）")
    print("=" * 80)
    header = f"  {'Trait':<22} {'Group':<12}"
    for tid in trial_ids:
        header += f"  {tid:>12}"
    print(header)
    print("  " + "-" * (38 + 14 * len(trial_ids)))

    for trait in _ALL_TRAITS:
        group = _TRAIT_GROUP_OF[trait]
        line = f"  {trait:<22} {group:<12}"
        for c in candidates:
            val = abs(c["sensitivity"][trait])
            # 強度標記：★ > 5000, ● > 2000, · > 500
            if val > 5000:
                marker = "★"
            elif val > 2000:
                marker = "●"
            elif val > 500:
                marker = "·"
            else:
                marker = " "
            line += f"  {val:>10.0f} {marker}"
        print(line)

    print("\n  圖例：★ > 5000   ● > 2000   · > 500")


def save_artifact(matrix: dict, output_dir: Path | None = None) -> Path:
    """儲存敏感度矩陣 JSON 至 outputs/。"""
    if output_dir is None:
        output_dir = _ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "sensitivity_9d_dream_candidates.json"

    # 轉成可序列化格式
    payload = {
        "schema_version": "v2.0",
        "analysis_type": "central_difference_sensitivity",
        "delta": matrix["delta"],
        "t_steps": _T_STEPS,
        "tail_start": _TAIL_START,
        "candidates": [
            {
                "trial_id": c["trial_id"],
                "RE_avg_baseline": c["RE_avg_baseline"],
                "z_exploring_baseline": c["z_exploring_baseline"],
                "relu_active": c["relu_active"],
                "sensitivity": {k: round(v, 4) for k, v in c["sensitivity"].items()},
                "abs_rank": c["abs_rank"],
            }
            for c in matrix["candidates"]
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 80)
    print("  Dream Candidates 9D Trait 敏感度掃描")
    print(f"  方法：中央差分 δ={_DELTA:.0e}  模擬步數={_T_STEPS}  尾段={_TAIL_START}~{_T_STEPS}")
    print(f"  候選數：{len(_DREAM_CANDIDATES)}  特質數：{len(_ALL_TRAITS)}")
    print(f"  總模擬次數：{len(_DREAM_CANDIDATES) * len(_ALL_TRAITS) * 2}")
    print("=" * 80)

    t0 = time.perf_counter()
    matrix = compute_sensitivity_matrix(verbose=True)
    elapsed = time.perf_counter() - t0

    print_heatmap(matrix)
    out_path = save_artifact(matrix)

    print(f"\n  完成時間：{elapsed:.1f}s")
    print(f"  Artifact：{out_path}")


if __name__ == "__main__":
    main()
