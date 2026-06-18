# P7-H 真人 Confirmatory 研究 — 最終分析報告

> **執行日期**：2026-06-18（收案封存）
> **預先註冊**：[docs/experiments/p7_online_personality_loop/P7H_PREREGISTRATION.md](../../../docs/experiments/p7_online_personality_loop/P7H_PREREGISTRATION.md)
> **裝置 sign-off**：[docs/experiments/p7_online_personality_loop/P7H_APPARATUS_SIGNOFF.md](../../../docs/experiments/p7_online_personality_loop/P7H_APPARATUS_SIGNOFF.md)
> **分析腳本**：[scripts/experiments/analyze_p7h_real_study.py](../../../scripts/experiments/analyze_p7h_real_study.py)
> **原始資料**：[p7h_player_test_sessions.json](p7h_player_test_sessions.json)、[p7h_survey_responses.json](p7h_survey_responses.json)
> **裝置除錯史**：見 [研發日誌.md](../../../研發日誌.md) 2026-06-06 ~ 2026-06-09 條目

---

## 1. 結論（一句話）

在乾淨的 **26 control / 26 experiment** 真人 confirmatory 樣本上，預先註冊的唯一 confirmatory 假設 **H1（定向人格事件比隨機事件更快把族群人格推向分岔臨界）達成**：主要客觀 DV `max_proximity` 之 **Cohen's d = 3.25、p(單尾) = 2.0×10⁻¹²**，方向與量級與裝置驗證階段的模擬預測（d ≈ 3.8–4.4）一致。崩壞率 experiment 26/26（100%）vs control 12/26（46%）。**探索性問卷（H2）無乾淨 P7-H 資料、不予報告**（見 §5）。**研究在此 N 封存。**

---

## 2. 樣本與資料清洗（關鍵）

### 2.1 原始檔案的表面狀態（誤導）

`p7h_player_test_sessions.json` 內 `is_human=true` 的 session 共 **239 筆**（control 30 / experiment 209）。這個 1:7 的臂別失衡**不是** P7-H 的真實樣本——它被**人格迭代研究** session 污染。

### 2.2 污染源（已釐清）

人格迭代研究與 P7-H 共用同一 player-test store。迭代研究的配臂邏輯把帶 `run_id` 的 session **強制標為 `group="experiment"`**（見 [研發日誌.md](../../../研發日誌.md) 2026-06-14 條、`api/ab_test_manager.py` 後續移除記錄）。於是迭代研究的所有 session 灌進了 P7-H 的 experiment 臂：

| 來源（experiment 臂，帶 `run_id`） | 筆數 | 性質 |
|---|---|---|
| `participant_id="dev"` | 89 | 迭代研究**實驗者**自玩（非 naive） |
| `participant_id="EXP_PREPILOT"` | 45 | 迭代研究 pre-pilot 實驗者自玩 |
| `P04`–`P31`（含 `EXCL_*`） | 48 | 迭代研究 **naive pilot 受試者**（屬另一研究） |
| **小計（帶 run_id）** | **182** | 全部排除於 P7-H |

另有 3 筆 control 亦帶 `run_id`，同樣排除。生態層 headless 煙測也可能寫入 `is_human=true` 的 session（[研發日誌.md](../../../研發日誌.md) 2026-06-12 生態條的已知污染坑）。

### 2.3 乾淨 P7-H confirmatory 過濾規則（已落入分析腳本）

P7-H confirmatory session **不帶 `run_id`**（P7-H 早於 `run_id`/`participant_id` 欄位）。乾淨過濾：

```
is_human == true
AND run_id 為空              ← 排除迭代研究（本次新增於 _load_sessions）
AND ended_at 非空
AND len(trajectory) >= 10    （_N_MIN）
AND RT 中位數在 [500, ∞) 或 RT 全為 0（自動遊玩，見裝置除錯史）
```

`scripts/experiments/analyze_p7h_real_study.py` 的 `_load_sessions` 本次新增 `run_id` 排除（2026-06-18）；修補前該腳本會把 208 筆污染 experiment session 一併納入，得出虛高的 d=4.60（n=29/208）。

### 2.4 乾淨樣本

| 臂 | n | participant_id | collapse_reason 分佈 |
|---|---|---|---|
| control | **26** | 全 `dev`（預設值，見 caveat C1） | `max_rounds` 14（右截斷）/ `proximity+passive_failure` 12 |
| experiment | **26** | 全 `dev`（預設值） | `proximity+passive_failure` 26 |

平衡的 26/26 與 count-balance A/B 配臂（session-id-based）一致。

---

## 3. H1 — 主要 confirmatory 結果（客觀）

### 3.1 max_proximity（PRIMARY DV，Welch 單尾）

| 臂 | n | mean | 　 |
|---|---|---|---|
| control | 26 | 0.7468 | 　 |
| experiment | 26 | 0.9922 | 　 |

- **Welch t = 11.72，p(單尾) = 1.97×10⁻¹²　✅ 顯著**
- **Cohen's d = 3.25，95% CI [2.42, 4.08]**
- achieved power = 1.000

定向（v1-aligned）事件把族群平均人格在 v1-v2 敏感面的投影距離單調推近分岔臨界（proximity → 1.0 飽和）；隨機事件則正負交錯、停在較低 proximity。此即預先註冊的 H1 主張，**達成**。

### 3.2 存活（rounds-to-collapse，次要客觀，ctrl > exp）

| 臂 | n | mean rounds | 崩壞 |
|---|---|---|---|
| control | 26 | 163.5 | 12/26（14 筆 `max_rounds` 右截斷） |
| experiment | 26 | 72.5 | 26/26 |

- **Welch t = −10.04，p(單尾) = 2.84×10⁻¹¹　✅ 顯著**
- **Cohen's d = 2.78，95% CI [2.02, 3.55]**
- ⚠ control 含 14 筆右截斷（未崩壞、貢獻最終回合）；Welch 為近似。N≥30/組宜改 Kaplan-Meier + log-rank 正確處理截斷。崩壞率對比（100% vs 46%）方向與日誌 2026-06-09 的 15/15 vs 6/15 一致。

### 3.3 診斷量（不作判定，混淆已知）

`H1b` 淨位移、路徑長度、`n_critical_crossings` 全部**反向或退化**（d < 0 或零變異），這是**預期且已記錄**的量測假象：事件強度受 proximity 調制（越近臨界施力越小），experiment 臂快速飽和後自我節流，故任何「位移量級」DV 都偏袒 control。此即裝置驗證階段（commit 80e1916）翻案、把主要 DV 從淨位移改為 `max_proximity` 的原因。**這些診斷量不可作為效應方向的證據**。

---

## 4. 裝置可信度（為何此 d 可信而非假象）

P7-H 的 d 之所以可信，建立在收案前完成的裝置工程鏈（見裝置除錯史）：

1. **自變項真的被操弄**：早期 pilot 的 control 也收到 v1-aligned 事件（`apply_event` 硬編 `"experiment"`）→ 已修為讀 `PlayerTestSessionClient.group`（control→random、experiment→aligned）。
2. **事件有持久效果**：proximity 飽和 + 每 round 整包覆寫人格的兩個 bug 已修；事件在 Space A 擾動全族群並跨 round 持久。
3. **DV 座標正確**：proximity 改用 v1-v2 敏感子空間投影（`compute_bifurcation_distance(mode="projection")`），解決歐氏距離對任何真人瞬間飽和的問題；Space A/B 雙射 round-trip gap = 0。
4. **主要 DV 選擇**：`max_proximity`（連續、無 proximity-調制混淆）取代淨位移；穩健性掃描（回饋 on/off × intensity × 8 seeds）全部顯著 d=2.44–5.26。
5. **回歸測試**保護裝置行為（`tests/test_p7h_apparatus_regression.py`：exp>ctrl、d>1、control 未飽和守衛）。

---

## 5. H2（探索性問卷）— 無乾淨 P7-H 資料，不予報告

`p7h_survey_responses.json` 共 62 筆問卷。以 `session_id` join 回 session store：

| join 目標 | 筆數 |
|---|---|
| 乾淨 P7-H session（無 run_id） | **0** |
| 迭代研究 session（有 run_id） | 42 |
| 未匹配（legacy/已刪 session） | 20 |

**零筆問卷對應乾淨 P7-H session**——全部問卷皆來自迭代研究或 legacy。分析腳本印出的 H2 數字（control=16.4 / experiment=8.4、q1–q3 Holm 全 n.s.）**完全是非 P7-H 來源的混合，不得作為 P7-H 結果引用**。

這**不影響** confirmatory 結論：H2 在預先註冊中已由共主要**降為探索性**（只報效應量、不做 go/no-go；P7H_PREREGISTRATION §3/§4/§7/§10），P7-H 的唯一 confirmatory 是客觀 H1。問卷管線在迭代研究上線後即被該研究接管，P7-H 端未獨立收集到對應問卷——這是程序性缺口，記錄於此，**非 H1 的威脅**。

---

## 6. 三個誠實 caveat（引用時必帶）

- **C1（provenance 粒度）**：乾淨 26/26 session 的 `participant_id` 全為預設值 `"dev"`——因 P7-H confirmatory 早於 `participant_id` 欄位（該欄 2026-06-13 才為迭代研究新增）。故 **P7-H 層級無 per-participant 來源碼**，無法在資料層面排除「某些 `dev` session 為實驗者自玩、而非獨立 naive 受試者」。可信度改由三點支撐：(a) `is_human=true` 由 Godot 前端在真人遊玩時設定；(b) 平衡的 26/26 A/B 分割與隨機配臂一致；(c) 效應方向與量級重現裝置模擬預測。**最強誠實宣稱＝「在 is_human-flagged、A/B-balanced 樣本上 H1 成立」**，而非「26/26 全為已驗證身分的 naive 受試者」。
- **C2（存活截斷）**：control 14/26 右截斷，§3.2 的 Welch d 為近似；嚴格存活推論需 log-rank。崩壞率對比（100% vs 46%）不受此影響。
- **C3（H1 過度檢定力）**：d=3.25 下 N=26/組遠超 80% power（裝置本就為「不飽和、可區分」校準）。真正的 N 約束是 H2（主觀 UX），而 H2 在 P7-H 端無乾淨資料；若日後要做 UX confirmatory，須獨立重收問卷並 pre-register SESOI。

---

## 7. 封存決定

- **H1 confirmatory 達成，P7-H 在 N=26/26 封存**，不再續收（d=3.25 已遠超所需 power；續收只增穩健性、不改結論）。
- **分析腳本已修補** `run_id` 排除（2026-06-18），使官方腳本可重現本報告的乾淨 26/26（修補前為污染的 29/208）。
- **後續若做 H1 複製研究**：獨立樣本（不同遺言文字池、不同 RL seed）、且**P7-H 自有 participant_id 來源碼 + 自有問卷管線**，與迭代研究 store 物理隔離（避免本次的 run_id 污染重演）。

---

## 7b. 補充：真人遺言 SIM 穩健性複製（非 confirmatory）

被迭代研究 `run_id` 污染的 188 筆 session 內含 **148 個不重複真人遺言**（naive P04-P31 + 實驗者）。這些遺言文字是真資料，被救出做 **H1 的 sim 穩健性複製**：用每個遺言儲存的原始 SBERT 向量重跑 P7-H apparatus，**每個遺言 paired 跑兩臂**（control + experiment，同 RL seed），量 `max_proximity`。

- **腳本**：[scripts/experiments/run_p7h_will_replay.py](../../../scripts/experiments/run_p7h_will_replay.py)——**只打無狀態端點** `/rl_sessions/{initialize,step,apply-event}`、**完全不碰 `/player-test/*`**，結果寫獨立檔 [p7h_will_replay_sim.json](p7h_will_replay_sim.json)。**confirmatory 主檔零變更**。
- **結果（n=148 paired）**：control max_proximity 0.734 vs experiment 0.864；**paired Δ = +0.130 ± 0.108，paired t = 14.71、p = 1.1×10⁻³⁰、Cohen's dz = 1.21**；131/148（89%）遺言 exp>ctrl；崩壞 experiment 148/148 vs control 56/148。
- **provenance（誠實分層）**：每筆 run 同時帶 `is_human=false`（**session 層＝sim replay，非真人玩**）與 `will_author_is_human=true` + `will_source="iteration_study"` + 原 `src_participant_id`（**遺言文字＝真人寫**）。兩者刻意分開：保留「真人遺言」的真價值，又杜絕把 sim run 誤當人類 session。
- **為何仍不可併入 confirmatory 26/26**（綁定理由，與 is_human 標籤無關）：① 非在 P7-H pre-reg 協定下收集（遺言寫於 iteration 研究脈絡）；② 非獨立（148 遺言來自 ~27 作者，其中 dev+EXP_PREPILOT=110 為實驗者，naive 僅 P04-P31 ~48）；③ paired 重用（每遺言跑兩臂，無真人被分配）。
- **定位（誠實）**：`is_human=false` 的 **sim 複製**，**不可併入 confirmatory 26/26**（pre-reg 把 sim 排除於 H1）。它證明 H1 方向在真實玩家遺言文字分佈上穩健，且 recklessness→intensity/cadence 映射在生態上合理（謹慎遺言 R 低、衝動遺言 R 高，分離可見）。sim 的 max_proximity 比 live（exp 0.992）壓縮，因 N=4 player + COLLAPSE_PROXIMITY=0.8 較早封頂；**方向與分離一致**。

---

## 8. 關鍵數字速查

| 指標 | control | experiment | 統計 |
|---|---|---|---|
| n（乾淨） | 26 | 26 | 平衡 A/B |
| max_proximity（PRIMARY） | 0.747 | 0.992 | d=3.25, p=2.0e-12 ✅ |
| rounds-to-collapse | 163.5 | 72.5 | d=2.78, p=2.8e-11 ✅ |
| 崩壞率 | 12/26 (46%) | 26/26 (100%) | — |
| H2 問卷 | — | — | 無乾淨 P7-H 資料 |
</content>
</invoke>
