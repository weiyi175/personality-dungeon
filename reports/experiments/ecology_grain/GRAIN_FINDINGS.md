# Grain 分析 — 9D will-space → 3-archetype 投影是否藏掉真實多樣性

**日期**: 2026-06-24 ／ **資料**: N=199 真人 will SBERT-9D（`ecology_state.json`，session_id ∧ outcome 雙非空）
**腳本（可重現）**: `scripts/experiments/ecology_grain_analysis.py`（用**生產**投影 `api/ecology_tracker.py::personality_to_archetype_soft`，未另造映射）
**原始數字**: `GRAIN_ANALYSIS.md`（每次跑覆寫）

---

## TL;DR 裁決

⚠ **grain 藏掉一半以上的多樣性**。生產的 9D→3-archetype 投影，其**精確可見 2D 子空間**只回收 **46%** 的 will 變異（**54% 盲區**）；生態 dynamics 實際用的 softmax soft 權重更只回收 37%（63% 盲區）。

→ **生態層量到的 diversity / monoculture ≠ will-space 的真實多樣性。** archetype 共存是 will 多樣性的一個**有損(>50%)、偏 PC1** 的 proxy。

**但這不推翻 g\*(β)**：g\*(β)/neg-freq/RPS 這套是**對 archetype 動力**的內部陳述，在它自己的單位上成立。grain 限制的是「archetype 共存 ⟹ **人格**多樣性」這個**詮釋跳躍**。是第三個 apparatus-limit 發現（同族於：靜態 mechanism-personality、live-ecology 讀 SBERT 非 mechanism-9D）。

---

## 證據（5 指標）

**M1 — will-space 結構（PCA）**：有效維度 ≈ **4.60 / 9**（participation ratio）。top-2 = 58.3%、top-3 = 70.9%。
- PC1（38.2%）= `stability_seeking +0.54 / risk_aversion +0.51 / impulsiveness −0.48` ＝ **bold↔cautious**
- PC2（20.1%）= `optimism −0.58 / suspicion +0.54 / endurance −0.34` ＝ **optimism↔suspicion**
- → 在**不同/更大**樣本上獨立複現 Exp A 的 2-axis 結構（Exp A: PC1=38%、PC2=22%）。但 4.6 維代表它**不是乾淨 2D**，PC3–5 仍有實質變異。

**M2 — grain 盲區**：精確可見 2D（score 差，由 `s_j−s_bal = τ·log(soft_j/soft_bal)` 無損反推）R² = **46.2%**；操作 soft 權重 R² = 36.8%。兩者都 >50% 盲區 → 排除「只是 softmax 壓縮假象」。

**M3 — 哪些特徵被藏**（從 soft 回收各特徵 R²）：
- 高可見：`risk_aversion 62%`、`impulsiveness 54%`、`stability_seeking 49%`（＝ PC1 / agg↔def 軸）
- **被藏**：`endurance 7%`、`suspicion 15%`、`randomness 16%`、`optimism 20%`（＝ PC2 + randomness）
- → archetype 投影看得見 PC1，**幾乎瞎於 PC2**（optimism↔suspicion）。`randomness` 近不可見（不在任何 score 軸、只透過 extremity 懲罰間接進 balanced）。

**M4 — 損失歸因**（皆 2D、公平比）：最佳線性 2D（PCA top-2）= 58.3%；archetype 可見 2D = 46.2%。
- **軸選擇損失** = +12.1 pt（archetype 軸偏向 PC1、欠讀 PC2 — **可改善**：選對軸能多救 12 pt）
- **降維不可逆損失** = 41.7 pt（4.6 維本就攤不進 2D — **結構性**：3 archetype / 2 simplex-DoF 容不下）

**M5 — hard 分箱粗度**：箱大小 agg 88 / def 97 / **bal 14（balanced-thin 複現）**。within-bin 變異 = **74.3%**，between-bin = 25.7%。
- → 知道一個玩家的 hard archetype，只說明其 will 位置的 ~26%。

---

## 詮釋（含 layer-separation，避免 over-claim）

**這個發現是什麼**：研究皇冠（game-vision concept-2）＝**多樣性**動力。但「多樣性」全程在 **3-archetype** 層量。本分析證明：3-archetype 是 ~4.6 維 will-space 的一個**偏 PC1、>50% 盲區**的影子。
- archetype **monoculture**（feared collapse）下，will-space 仍可藏大量多樣性（74% within-bin）。
- archetype **diversity**（success）也**不**保證 will 多樣性。

**這個發現不是什麼**（layer-separation）：
- **不**推翻 g\*(β)/neg-freq/RPS-rotation：那套是**對 archetype state 的動力**，archetype 就是它選定的選擇單位，在其單位上自洽。本分析不碰那條。
- **不**是說投影「壞了」：投影按設計把 9D 壓成 3 派系供 RPS payoff 運作，這是刻意的。問題只在**外推到「人格多樣性」時的解讀**。

**連結既有發現**：
- PC2（被欠讀的軸）的錨＝`suspicion`（見 [trait-blast-radius-expA]：suspicion ANCHORS PC2、vindicates suspicion→野）。→ **若要把「野」做成真實 faction 軸，現行 archetype 映射看不到它**（與 Exp A「factions = 2-axis 投影」一致）。
- 同族於 apparatus-limits：靜態 mechanism-9D（P₀ 控制）、live ecology 讀 SBERT-varied 非 mechanism。本發現補上第三條：**diversity 引擎量的是 ~4.6D will 的 2D-ish archetype 影子**。

**選項（不單方面定，供決策）**：
1. **研究目標＝archetype 層動力（g\*(β)）** → grain 無妨，archetype 是選定單位。**只需在報告明寫**「diversity ＝ archetype-diversity，是 will 多樣性的有損 proxy」。（最低成本，建議。）
2. **想宣稱人格層多樣性** → 需更細 grain（更多 archetype／連續投影）或顯式 caveat。屬 scoping/文件決定，非急迫程式改動。
3. **可改善的 12 pt**：若仍用 3-archetype 但想少瞎於 PC2，可重校投影軸（把 optimism/suspicion 拉進 contrast）— 但這會動 payoff 校準（COIN_BRACKETS/λ 需重跑），且 off-main-line（不服務 bound β）。**不建議現在做**。

---

## 主線定位（反偏離）

本分析是 **handoff §4 P3 solo consolidation**，用**現有**資料、不需收集 → 在不靠真人 pilot 下推進了一個**對研究皇冠的誠實邊界**。它**不**是 bound β 的替代（β 仍需乙）。它的價值＝把「archetype diversity = 多樣性」這個一直默認的等號，量化成「>50% 有損 proxy」，供 capstone 與對外宣稱時誠實標注。
