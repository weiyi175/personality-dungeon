"""personality_mapping.py — 人格特質 → z 訊號投影配置（純函式，無 I/O）

提供：
  1. 訊號權重常數（對稱基準 / 對稱破缺實驗）
  2. compute_z_signals() — 可配置加權投影

SDD 合規性（evolution/ 層約束）：
  - 不做 I/O，不依賴 plotting，不依賴 simulation/
  - 與 personality_guided_replicator_step() 正交（僅計算輸入訊號，不做狀態更新）

版本歷史：
  v1.0 — 初版，含對稱基準與 Exploring 非對稱破缺兩個配置
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 訊號權重定義
# 約束：各組權重總和 = 1.0，保持 z 訊號的值域語意不變（最大值約 1.0）
# ---------------------------------------------------------------------------

WEIGHTS_SYMMETRIC: dict[str, dict[str, float]] = {
    "expanding": {
        "impulsiveness":  1 / 3,
        "assertiveness":  1 / 3,
        "optimism":       1 / 3,
    },
    "contracting": {
        "risk_aversion":  1 / 3,
        "suspicion":      1 / 3,
        "endurance":      1 / 3,
    },
    "exploring": {
        "randomness":        1 / 3,
        "stability_seeking": 1 / 3,
        "curiosity":         1 / 3,
    },
}
"""對稱基準（各組等權 1/3 × 1/3 × 1/3）。

此為 Model v2.0 的預設配置：
  z_X = (trait_1 + trait_2 + trait_3) / 3.0
"""

WEIGHTS_EXPLORING_ASYMMETRIC: dict[str, dict[str, float]] = {
    "expanding":   WEIGHTS_SYMMETRIC["expanding"],
    "contracting": WEIGHTS_SYMMETRIC["contracting"],
    "exploring": {
        "curiosity":         0.5,  # 好奇心：主導性格引擎（最高權重）
        "randomness":        0.3,  # 隨機性：次要貢獻
        "stability_seeking": 0.2,  # 穩定尋求：緩衝/阻尼
    },
}
"""Exploring 組對稱破缺（Symmetry Breaking）配置。

物理假說：
  curiosity（好奇心）對旋轉動能的影響語意與 randomness（隨機性）不同。
  好奇心是「定向探索」，随機性是「無方向擾動」。
  非對稱加權讓 curiosity 成為 z_exploring 的主導引擎，
  預期使 curiosity 的 dRE/dTrait 敏感度顯著上升（相對對稱基準）。

注意：expanding 與 contracting 組維持對稱，以便隔離效果。
"""


def compute_z_signals(
    traits: dict[str, float],
    weights: dict[str, dict[str, float]] | None = None,
) -> tuple[float, float, float]:
    """計算 (z_expanding, z_contracting, z_exploring)。

    Args:
        traits:  9D 特質字典 {trait_name: value}，值域 [-1, 1]
        weights: 各組加權字典，格式與 WEIGHTS_SYMMETRIC 相同。
                 省略（None）則等效使用 WEIGHTS_SYMMETRIC。

    Returns:
        (z_expanding, z_contracting, z_exploring) 三個浮點訊號。

    Examples::

        from evolution.personality_mapping import (
            compute_z_signals,
            WEIGHTS_SYMMETRIC,
            WEIGHTS_EXPLORING_ASYMMETRIC,
        )
        traits = {...}  # 9D dict
        z_exp, z_con, z_expl = compute_z_signals(traits)              # 對稱
        z_exp, z_con, z_expl = compute_z_signals(traits, WEIGHTS_EXPLORING_ASYMMETRIC)  # 破缺
    """
    if weights is None:
        weights = WEIGHTS_SYMMETRIC

    z_exp  = sum(traits[t] * w for t, w in weights["expanding"].items())
    z_con  = sum(traits[t] * w for t, w in weights["contracting"].items())
    z_expl = sum(traits[t] * w for t, w in weights["exploring"].items())

    return float(z_exp), float(z_con), float(z_expl)
