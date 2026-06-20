# 地牢/遊戲設計 — 待回歸問題 Parking Lot

> **用途**：記錄討論中**已擱置、之後要回來收**的問題，避免往下鑿時遺忘。
> **更新**：2026-06-19。狀態標：🔴未決 / 🟡暫定（有預設待確認）/ 🟢已鎖（列此供追溯）。
> 相關稿：`地牢counter-policy_L0L1介面_規劃_v1.md`、`地牢AI_成長與等級_規劃_v1.md`、`人格生態評分_規劃_v1.md`。

---

## A. Counter-policy / Combat（PARKED 2026-06-19，待 build；模擬階段結束）
- 🟢 **修法家族＝C（authored 剋制表）**：H_counter 證偽（9 桶弱點方向全 v1、proximity 只量 v1-v2 平面）→ counter-matrix **不可測、必須 authored**。A/B 棄。
- 🟢 **factions＝Option 4+**：不分 trait 簇（A0 修正：bold↔cautious 主軸 PC1=38%，非 3 簇）；3 faction＝**PC1-PC2 平面 3×120° argmax**（真 partition，三派都有質量）；9 trait-argmax 桶＝連續 substrate。薄野＝gate0 PC1-only 投影 artifact，非 payoff 問題。
- 🔴 **第三派主題（authored，待 build+playtest）**：第三方向載什麼 trait 可算（PCA loading 回 trait 空間，~5 分）；但「那組 trait 是不是連貫、可命名、好玩的派系」**不可測**。「野＝多疑↔樂觀」已收回（假設非測量）。
- 🔴 **combat M 箭頭+數值（authored，待 build+playtest）**：H_counter 證偽 ⇒ 設計+試玩，非模擬可答。含結算機制（單/多回合、局內抵抗）、PVP 重用 proximity 多少、effectiveness 度量——全待 build 時定。
- 🟢 **v2 roadmap（前向相容三叉路，不可丟）**：① M 值 凍結→per-dungeon 學習；② M 列 離散→連續 9D；③ M 欄 單桶→局內軌跡。原則：v1 是 v2 粗特例、零丟棄。

## B. 地牢成長 / 等級
- 🟢 `clarity_i` ＝桶信心主導（已鎖，成長稿 §8.1）。
- 🟡 **攻方偵察**：不對稱資訊（暫定）vs 全資訊 v2 開關。
- 🟡 **學習率道具過擬合曲線**：凸+甜區（暫定），`η_sweet`+懲罰斜率待數值校準。
- 🔴 **所有權重 `w_*` 數值校準**（base/score/w_rare/quality/survival/cover/directed/λ_decay）。

## C. 經濟（Rank / 金幣）
- 🔴 **零和守恆 vs 通膨**：金幣抵銷採守恆版（A 拿 B 實掉的）——我推守恆，**待你最終拍板**。
- 🔴 **參數**：`rank_stake`、`offset_ratio`(暫 0.5)、`coin_per_rank` 兌換率——全待定不寫死。
- 🟢 **存活金幣修正（已做 2026-06-20）**：真相比「權重太低」更糟——**存活對金幣零貢獻**（CollapseScreen 的 `存活加成 rounds×0.25` 只加進**顯示分數**，coins=後端 `score_to_coins(生態 score)`、存活不在裡面）。修＝存活變**獨立 coin 來源**（`SURVIVAL_COIN_RATE=0.3`，200 回合 +60），加進 `_total_coins`、不折進生態 score（不打架 brackets），CollapseScreen 誠實拆解「💰本場＝生態X+存活Y」。k=0.3 刻意次要於生態多樣性誘因（太高→求生壓追稀缺＝反多樣性；且稀缺=balanced 經 opt+cur=高 recklessness 短命，survive-vs-追稀缺是真選擇）。前端 commit `7bf2f7e`。**待真人 Godot 驗證顯示**。
- 🟡 **生態 neg-freq 均衡支付水位**（2026-06-19）：uniform 生態下人人 ~100 coins（無稀缺＝無差別誘因，intended 且自限）；但絕對水位是 coin 通膨事——建 coin 經濟時把「均衡支付≈100/人」算進 sink 設計。現無持久錢包，無需處理，記著別丟。
- 🔴 **欄位券＝金幣 source**：要進 source/sink 表（非多樣性問題，已澄清）。
- 🟡 **加速券→「雙生券」**：改成局內雙人格平行跑（雙倍機會＋雙倍 Trace），rename + 用玩家價值正當化。**待定案**。
- 🔴 **配對規則**：自選名單 + Rank 分段 + 重複遞減 + 門票成本（草案，未定）。
- 🔴 **完整 source/sink 平衡表**：曾提議、未做。

## D. 生態層 / concept 2
- 🟢 生態＝顯示稀缺+coin 誘因（人在迴路最簡版，已鎖 2026-06-18）。
- 🟢 **開局顯示稀缺 type（已建+e2e 2026-06-20）**：寫遺言前琥珀「本週期稀缺：X型 — 選它分數金幣更高 / 想成為它：寫…」（X=argmax(weight)，n<10 隱藏）。**同時修了更深 bug**：鎖定的「獎勵稀缺」原本沒在 code（`_fitness` 是 RPS Path-B 殘留、獎勵與稀缺解耦）→ 換負頻率依賴 `1/N−q`、lam=2、重校 brackets、migration 防 load() 覆寫 lam。閉環 actionable 半邊真的閉了。**仍未 commit**（兩 repo）。
- 🟡 **archetype 命名債（2026-06-19）**：生態「balanced」型其實要靠 optimism+curiosity 才到（投影 `s_bal=opt+cur−extremity`），**不是「中庸」**（中庸→0.33 三方平手、不 argmax）。v1 用開局文案 guard 修（導「樂觀好奇、非中庸」）。真乾淨化＝改名（動 CollapseScreen/生態多檔，出 (a) 範圍）或修投影（Option 4+，parked）。記著別丟。
- 🟢 **反 whiplash**：a>b 阻尼側已在對的一邊（2026-06-19 Exp B 重驗：maxRe⟺sign(b−a)，b=a 翻轉、cr 無關穩定性；b=1.2→滅絕全域佐證）；稀缺加成遲滯/緩斜坡＝建議未實作。
- 🟢 **★ 生態 M vs combat M（已決 2026-06-19）＝不統一**：共用 taxonomy/topology、**不共用增益**；唯一約束＝生態 payoff **a≥b**（共存）；只有 combat 特別要 b>a 才需分表（「分兩張表」已收窄成「一個不等式」）。**cr＝自由平衡旋鈕**（移 q*：cr=0→[.33,.33,.33]、.5→[.17,.56,.27]，不碰穩定性，範圍≲0.8）。

## E. Meta / 全局
- 🟢 concept1 vs concept2 ＝路線 B（concept2 只動玩家面分數/金幣，concept1 吃乾淨原始 Trace）。
- 🟢 粒度＝9 桶（trait argmax，放掉 RPS 旋轉硬追求）。
- 🟢 上下輪無性格累積（lives 全解耦，已從 codebase 移除）。
- 🟡 **冷啟動**：地牢規則式 bootstrap + 交棒門檻；小火龍用現有 26 真人+296 sim+迭代 Trace 起跑。草案，未細化。
- 🔵 **concept1「非激勵學成的真智能」＝願景非機制**（路線 C 誠實降級備案）。
- ⚠ **全部設計稿皆 DRAFT、未實作、未 commit**。
