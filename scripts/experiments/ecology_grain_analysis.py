#!/usr/bin/env python3
"""Grain 分析：9D will-space → 3-archetype 投影會不會藏掉真實多樣性？

主線錨點（game-vision）：研究皇冠＝**多樣性**動力。但「多樣性」是在 **3-archetype** 層
量的（共存 vs monoculture）。若真實 will 的變異有相當一部分落在 archetype 投影**看不到**的
軸上，那生態層量到的 diversity/monoculture **不等於** will-space 的真實多樣性——這是 grain 陷阱。

用現有 N=199 真人 will SBERT-9D（session_id ∧ outcome 雙非空），跑 5 個決定性指標：
  M1 PCA 有效維度        — 9D 變異實際攤在幾個軸（與 Exp A「2-axis」對照）
  M2 GRAIN 盲區          — 9D 變異有多少**無法**從生產的 3-soft archetype 線性回收（＝被藏的多樣性）
  M3 per-feature 可見度  — 哪幾個特徵被藏（預期 randomness 最低：不在任何 score 軸）
  M4 軸對齊             — 損失是「2D 太少」還是「archetype 軸選錯方向」（archetype-visible vs PCA top-2）
  M5 grain 粗度          — hard argmax 分箱的 within/between 變異（連續變化被丟多少）

純讀 ecology_state.json，用**生產**投影 personality_to_archetype_soft（不另造映射）。
用法：python scripts/experiments/ecology_grain_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))           # evolution.* 等 repo-root 套件
sys.path.insert(0, str(_ROOT / "api"))   # ecology_tracker 當 top-level import
from ecology_tracker import (  # noqa: E402
    ARCHETYPES,
    FEATURE_NAMES,
    personality_to_archetype_soft,
)


def load_real_9d(state_path: str) -> np.ndarray:
    """真人 will SBERT-9D（與 β-instrument 同判準：session_id ∧ outcome 雙非空）。"""
    d = json.loads(Path(state_path).read_text())
    rows = [s["personality_9d"] for s in d.get("submissions", [])
            if s.get("session_id") and s.get("outcome") and len(s.get("personality_9d", [])) == 9]
    return np.asarray(rows, dtype=float)


def _ols_reconstruct(X: np.ndarray, Z: np.ndarray) -> tuple[float, np.ndarray]:
    """用 Z（含截距）線性回收 X 的每一欄。回 (整體 R², per-feature R²)。

    整體 R² = 1 − Σ_resid_var / Σ_total_var（跨特徵加總變異，反映**佔總變異的可回收比**）。
    """
    Zd = np.column_stack([np.ones(len(Z)), Z])           # 加截距
    beta, *_ = np.linalg.lstsq(Zd, X, rcond=None)        # (k+1, D)
    Xhat = Zd @ beta
    resid = X - Xhat
    Xc = X - X.mean(0, keepdims=True)
    tot_per = (Xc ** 2).sum(0)                           # 每特徵總變異
    res_per = (resid ** 2).sum(0)
    r2_per = 1.0 - res_per / np.where(tot_per > 0, tot_per, 1.0)
    r2_overall = 1.0 - res_per.sum() / tot_per.sum()
    return float(r2_overall), r2_per


def analyze(X: np.ndarray) -> list[str]:
    n, d = X.shape
    out: list[str] = []
    out.append("=" * 70)
    out.append(f"  GRAIN 分析：9D will-space → 3-archetype 投影是否藏多樣性  (N={n})")
    out.append("=" * 70)

    # ── M1 PCA 有效維度 ──────────────────────────────────────────────────────
    Xc = X - X.mean(0, keepdims=True)
    # 在 archetype 投影作用的**原始**單位空間做 PCA（不標準化，與投影一致）。
    cov = np.cov(Xc, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    evals = np.clip(evals, 0, None)
    frac = evals / evals.sum()
    cum = np.cumsum(frac)
    part_ratio = (evals.sum() ** 2) / (evals ** 2).sum()   # participation ratio
    out.append("")
    out.append("[M1] PCA 有效維度（原始單位、置中、未標準化）")
    out.append("   PC 變異占比 : " + "  ".join(f"PC{i+1}={frac[i]*100:4.1f}%" for i in range(d)))
    out.append(f"   累積 top-2 : {cum[1]*100:.1f}%   top-3 : {cum[2]*100:.1f}%")
    out.append(f"   participation ratio（有效維度）: {part_ratio:.2f} / {d}")
    # 印 PC1/PC2 主導特徵（對照 Exp A 2-axis）
    for pc in range(min(2, d)):
        load = evecs[:, pc]
        idx = np.argsort(np.abs(load))[::-1][:3]
        terms = "  ".join(f"{FEATURE_NAMES[i]}{load[i]:+.2f}" for i in idx)
        out.append(f"   PC{pc+1} 主導 : {terms}")

    # ── 生產投影：3-soft archetype ───────────────────────────────────────────
    soft = np.array([personality_to_archetype_soft(list(x)) for x in X])
    hard = soft.argmax(1)

    # ── M2 GRAIN 盲區：9D 變異有多少可從 archetype 投影回收 ────────────────────
    # 精確可見子空間：softmax 保留的資訊＝score 差，且 s_j−s_bal = tau·(log soft_j − log soft_bal)，
    # 可從生產 soft 輸出無損反推（不重寫公式）→ 回歸到這 2 個 contrast＝archetype 真正編碼的維度。
    TAU = 0.6
    lsoft = np.log(np.clip(soft, 1e-9, 1.0))
    contrasts = TAU * (lsoft[:, :2] - lsoft[:, 2:3])    # [c_agg−bal, c_def−bal]
    r2_vis, _ = _ols_reconstruct(X, contrasts)
    # 操作觀察量：生態 dynamics 實際用的 soft 權重（softmax 後、會飽和壓縮）。
    r2_soft, r2_soft_per = _ols_reconstruct(X, soft)
    out.append("")
    out.append("[M2] GRAIN 盲區：9D 變異可從 archetype 投影回收的比例")
    out.append(f"   精確可見子空間 (score 差, 2D) R² : {r2_vis*100:.1f}%   "
               f"→ 盲區 {(1-r2_vis)*100:.1f}%")
    out.append(f"   操作 soft 權重 (softmax 後) R²   : {r2_soft*100:.1f}%   "
               f"→ 盲區 {(1-r2_soft)*100:.1f}%")
    out.append(f"   ★ 隱藏多樣性（盲區，取精確值）  : {(1-r2_vis)*100:.1f}%  ← 生態層看不到的 will 變異")

    # ── M3 per-feature 可見度 ────────────────────────────────────────────────
    out.append("")
    out.append("[M3] per-feature 可見度（從 3-soft 回收各特徵的 R²；低＝被藏）")
    feat_std = X.std(0)
    for i in np.argsort(r2_soft_per):   # 低到高
        flag = "  ← 幾乎不可見" if r2_soft_per[i] < 0.10 else ""
        out.append(f"   {FEATURE_NAMES[i]:16s} R²={r2_soft_per[i]*100:5.1f}%  "
                    f"(std={feat_std[i]:.3f}){flag}")

    # ── M4 軸對齊：archetype-visible vs 最佳 2D / 3D（PCA）─────────────────────
    # archetype softmax 只用 score 差（simplex=2 DoF）。比「最佳線性 2D」量損失歸因。
    pca2 = Xc @ evecs[:, :2]
    pca3 = Xc @ evecs[:, :3]
    r2_pca2, _ = _ols_reconstruct(X, pca2)
    r2_pca3, _ = _ols_reconstruct(X, pca3)
    out.append("")
    out.append("[M4] 軸對齊：損失是「2D 太少」還是「archetype 軸選錯」（兩者皆 2D，公平比）")
    out.append(f"   最佳線性 2D (PCA top-2) 可回收     : {r2_pca2*100:.1f}%   (3D: {r2_pca3*100:.1f}%)")
    out.append(f"   archetype 可見 2D (score 差) 可回收 : {r2_vis*100:.1f}%")
    gap = r2_pca2 - r2_vis
    out.append(f"   → 軸選擇損失 (best-2D − archetype) : {gap*100:+.1f} pt（archetype 軸偏向 PC1、欠讀 PC2）"
               f"\n     降維不可逆損失 (100% − best-2D) : {(1-r2_pca2)*100:.1f} pt（4.6 維攤不進 2D）")

    # ── M5 grain 粗度：hard argmax 分箱 within/between ───────────────────────
    grand = X.mean(0, keepdims=True)
    ss_tot = ((X - grand) ** 2).sum()
    ss_within = 0.0
    sizes = []
    for k in range(len(ARCHETYPES)):
        Xk = X[hard == k]
        sizes.append(len(Xk))
        if len(Xk):
            ss_within += ((Xk - Xk.mean(0, keepdims=True)) ** 2).sum()
    within_frac = ss_within / ss_tot
    out.append("")
    out.append("[M5] grain 粗度：hard argmax 3-箱的變異拆解")
    out.append("   箱大小 : " + "  ".join(f"{ARCHETYPES[k]}={sizes[k]}" for k in range(len(ARCHETYPES))))
    out.append(f"   within-bin 變異占比 : {within_frac*100:.1f}%   "
               f"between-bin : {(1-within_frac)*100:.1f}%")
    out.append(f"   → hard 標籤丟掉 {within_frac*100:.1f}% 的 9D 變異（箱內連續差異）")

    # ── 裁決 ────────────────────────────────────────────────────────────────
    hidden = (1 - r2_vis) * 100
    out.append("")
    out.append("-" * 70)
    if hidden >= 40:
        verdict = (f"⚠ grain 藏多樣性：{hidden:.0f}% 的 will 變異落在 archetype 投影盲區。"
                   " 生態層的 diversity/monoculture ≠ will-space 真實多樣性。")
    elif hidden >= 20:
        verdict = (f"△ grain 部分藏多樣性：{hidden:.0f}% 盲區。archetype 量到的多樣性是"
                   " will-space 的有損投影，解讀 monoculture 時要記得這層損失。")
    else:
        verdict = (f"✓ grain 大致忠實：僅 {hidden:.0f}% 盲區，3-archetype 抓住了 will 變異主體。")
    out.append("裁決：" + verdict)
    out.append("=" * 70)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="9D→3-archetype grain 分析（真實多樣性是否被藏）")
    ap.add_argument("--state", default="reports/ecology/ecology_state.json")
    ap.add_argument("--out", default="reports/experiments/ecology_grain/GRAIN_ANALYSIS.md",
                    help="把報告寫到此路徑（None 則只印 stdout）")
    args = ap.parse_args()

    X = load_real_9d(args.state)
    if len(X) < 10:
        print(f"真人 9D 筆數 {len(X)} < 10，樣本不足。")
        sys.exit(1)
    lines = analyze(X)
    print("\n".join(lines))

    if args.out and args.out.lower() != "none":
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        body = "# Grain 分析報告（9D→3-archetype 是否藏多樣性）\n\n```\n" + "\n".join(lines) + "\n```\n"
        outp.write_text(body)
        print(f"\n[report] 已寫 {outp}")


if __name__ == "__main__":
    main()
