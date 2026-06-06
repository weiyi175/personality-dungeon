# P7-H 確認性研究 — 預先註冊（Pre-registration）

> **鎖定狀態**：本文件於收案開始前定稿。一旦開始收集真人資料，下列假設、依變項定義、檢定方法、樣本量與停止規則 **不得修改**。任何偏離須在最終報告的「Deviations」段落明列。
>
> **定稿日期**：2026-06-05
> **研究代號**：P7-H（線上人格迴圈之分岔應用層 / bifurcation application layer）
> **分析腳本（鎖定）**：[scripts/experiments/analyze_p7h_real_study.py](../../../scripts/experiments/analyze_p7h_real_study.py)

---

## 1. 背景與目的

P7-F/P7-G 已在動力學層辨識出人格軌跡的分岔結構（敏感方向 v1/v2、臨界尺度）。P7-H 將其轉為**遊戲應用層**：實驗組的事件沿敏感方向 v1 設計（bifurcation-optimised），對照組事件採隨機方向（baseline）。

**研究問題**：沿敏感方向設計的事件，是否能（H1）在客觀上更有效地推動玩家人格位移、（H2）在主觀上提升遊戲體驗、（H3）使兩者相關。

離線階段已用 N=210 模擬資料跑通整條分析鏈作為管線驗證；本預先註冊針對**真人收案**（N=212）。

---

## 2. 設計

- **型態**：雙臂、組間（between-subjects）、玩家對組別**單盲**（玩家不知自己屬哪一組）。
- **分組**：
  - `control`（對照）：事件採隨機方向。
  - `experiment`（實驗）：事件沿敏感方向 v1。
- **分派機制**：計數平衡（count-balanced），新場次指派到目前人數較少的組，保證兩臂相等；平手時以 `sha256(session_id+seed)` 確定性決定。實作見 [api/ab_test_manager.py](../../../api/ab_test_manager.py) `assign_session`。
- **每位玩家流程**：`/bifurcation/ab-test/assign` → `/player-test/start` → 多次 `/player-test/step`（遊戲動作）→ `/player-test/end` → `/survey/submit`。每人約 20–30 分鐘。
- **匿名化**：僅記錄 `player_alias`（匿名識別碼），不收集可識別個資。

---

## 3. 樣本量與檢定力

- **目標 N = 212**（`control` 106 / `experiment` 106）。
- **檢定力依據**：主要客觀 DV（`max_proximity`）在模擬下效應量極大（d≈+3.8~+4.4），所需 N 遠低於 64/組；目標 N 主要由**共主要主觀 DV（H2）**與 106/106 計數平衡決定。保守以 Cohen's d ≈ 0.5 估算：雙臂 Welch t（單尾 α=0.05、power=0.80）所需約 **64/組**（公式見 [analyze_p7h_real_study.py](../../../scripts/experiments/analyze_p7h_real_study.py) `n_per_group_for_power`）。106/組相對 64/組留有充足餘裕。
- 分析腳本另會回報**在觀測 N 下達成的檢定力**與**達 80% 所需 N**。

---

## 4. 假設與檢定（逐字對齊分析腳本）

### H1 — 主要假設（客觀）
- **依變項（主要，2026-06-06 修訂）**：`max_proximity` ＝ 場次中 9D 人格軌跡達到的最大 bifurcation proximity（定義見 [api/player_test_tracker.py](../../../api/player_test_tracker.py) `record_step`／`max_proximity`）。
- **檢定**：Welch 兩樣本 t 檢定，**單尾**（H1: experiment > control）。
- **效應量**：Cohen's d 及 95% CI（pooled SD）。
- **顯著判準**：單尾 p < 0.05。
- **改採此 DV 的理由**：事件強度受 proximity 調制（越近分岔施力越小），對齊組接近臨界後自我節流、隨機組永遠拿全力，故**任何位移量級 DV（淨位移、路徑長度）都系統性偏袒對照組**，並在序列夠長時 null／反轉。`max_proximity` 直接量測操作所瞄準的目標（抵達分岔），無此混淆，跨操作區間穩健（模擬 d≈+3.8~+4.4, p<1e-14；詳見 [REGIME_FINDING.md](../../../reports/experiments/p7h_engine_sim/REGIME_FINDING.md)）。此亦回歸原先 crossing 類 DV 的精神（見 §4 H3 偏差說明）。

### H1b — 次級假設（客觀，位移幅度）
- **依變項**：`total_displacement` ＝ 場次首尾 9D 人格向量的歐氏距離 `‖P_final − P_initial‖`（[api/player_test_tracker.py](../../../api/player_test_tracker.py) `end_session`）。
- **檢定／判準**：同 H1（Welch 單尾、Cohen's d、p < 0.05）。
- **⚠ 有效性條件**：僅在實驗組未飽和（max proximity < ~0.95）時有效；遊戲設計須將事件序列收到此範圍（現行 `intensity_scale=1.0` 下約 ≤ 4 個分岔事件/場次，取代 §6 原「30 步」預期）。否則此 DV 預期 null／反轉，**不得**據以推翻 H1 主結論。

### H2 — 共主要假設（主觀）
- **依變項**：問卷 UX composite ＝ `q1_naturalness + q2_fun + q3_replay`（每題 1–10）。
- **主檢定**：Mann-Whitney U，**單尾**（experiment > control）於 composite。
- **次級（逐題）**：q1/q2/q3 各自 Mann-Whitney U（單尾），三題以 **Holm-Bonferroni** 校正。
- **顯著判準**：composite 單尾 p < 0.05；逐題以 Holm 校正後 p < 0.05。

### H3 — 探索性假設
- **變項（原始預先登記）**：`n_critical_crossings`（proximity 由 < 0.8 跨越到 ≥ 0.8 的次數）對上 UX composite，依 `session_id` 配對。
- **檢定**：Spearman 等級相關，**雙尾**。
- **顯著判準**：雙尾 p < 0.05。為探索性，不納入主要結論的多重比較校正族。

> **⚠ 預先登記偏差記錄（2026-06-06）**：
> 試跑後發現 bifurcation detector 對所有人格向量均回傳 `bifurcation_proximity = 1.0`（所有向量皆已超過臨界閾值），導致 `n_critical_crossings` 在所有 sessions 中結構性為 0，無法作為 x 變項。
> **實際分析改以 `total_displacement`（同為 H1 的客觀 DV）替代 `n_critical_crossings` 作為 x 變項**，檢定問題變為「人格位移幅度與 UX 主觀體驗是否相關」。
> 此偏差已在分析程式碼（`analyze_p7h_real_study.py`）及輸出 JSON 中以 `deviation_note` 欄位標注。

---

## 5. 問卷題目（鎖定，1–10 量表）

| 題號 | 題目 | 量表 |
|---|---|---|
| q1_naturalness | 人格轉變是否感到自然？ | 1（完全不自然）— 10（非常自然） |
| q2_fun | 邊界控制是否增加遊戲樂趣？ | 1（完全沒有）— 10（非常有趣） |
| q3_replay | 是否願意繼續遊玩？ | 1（絕對不會）— 10（非常想繼續） |

題目定義鎖定於 [api/survey_manager.py](../../../api/survey_manager.py) `QUESTIONS`。

---

## 6. 納入 / 排除條件

- **納入分析**：場次須 `ended_at` 非空（已正常結束）。分析腳本自動過濾未結束場次（[analyze_p7h_real_study.py](../../../scripts/experiments/analyze_p7h_real_study.py) `_load_sessions`）。
- **最少動作數**：場次軌跡須至少 **10 個動作步驟**才入分析（N_min = 10，確保玩家至少觸發數次分岔事件並排除極早中離場次）。
- **事件序列上限（2026-06-06 新增）**：為使次級 DV H1b（淨位移）有效並避免飽和假象，事件序列須使**實驗組 max proximity 維持 < ~0.95**，現行 `intensity_scale=1.0` 下約 **≤ 4 個分岔事件/場次**。此取代先前「遊戲設計預期 30 步（≈10 事件）」之描述——30 步正落在飽和反轉區。主要 DV `max_proximity` 不受此限制影響，但序列上限仍用於 H1b 與遊戲體驗設計。
- **排除**：明顯逾時/中離（未呼叫 `/player-test/end`，即 `ended_at` 為空）之場次自動排除。反應時間異常排除：場次各步驟 `response_time_ms` 的**中位數** < 500 ms（自動化/機器人行為）或 > 180,000 ms（長時間無操作 AFK；即每步平均逾 3 分鐘）者排除。最終報告須記錄排除場次數與各排除理由。

---

## 7. 停止規則

- 達 **212 個合格完成場次**（且 `control` 106 / `experiment` 106 皆滿）即停止收案。
- **不做期中偷看（no interim peeking）**：收案完成前不執行 H1/H2/H3 之顯著性檢定。進度監控僅看各組 `n_completed` 計數（`GET /player-test/summary`），不看效應或 p 值。

---

## 8. 知情同意

- 收案前呈現同意頁，說明：研究目的、約 20–30 分鐘時長、資料匿名收集與用途、可隨時退出。
- 取得同意後才指派組別與開始遊玩。

---

## 9. 資料與分析執行

- **資料落地**：真人資料寫入 `reports/experiments/p7h_real_study/`（後端以 `P7H_OUT_DIR` 設定，預設即此路徑），與模擬 `p7h_player_test/` 分流，互不污染。每次 `end_session` / `submit` / `assign` 自動存檔，並於伺服器啟動時回填，重啟不丟資料。
- **分析指令**（收滿後執行）：
  ```bash
  ./venv/bin/python scripts/experiments/analyze_p7h_real_study.py \
    --sessions reports/experiments/p7h_real_study/p7h_player_test_sessions.json \
    --survey   reports/experiments/p7h_real_study/p7h_survey_responses.json \
    --out      reports/experiments/p7h_real_study
  ```
- **輸出**：`reports/experiments/p7h_real_study/p7h_real_study_analysis.json` 與 console 之 H1/H2/H3 報告。

---

## 10. 偏離記錄（Deviations）

> 收案/分析過程中任何與本文件不符之處，於此逐條記錄（事件、原因、影響）。預設為空。

> **⚠ 收案前發現 — 事件序列長度會反轉 H1 效果（2026-06-06，待使用者確認）**
> 以重設計後的 RL 引擎 Space-A 事件路徑做後端模擬驗證（`run_p7h_engine_sim.py`，N=60）後發現：
> H1 的效果**方向**取決於每場次的事件數，因為事件強度受 proximity 調制（越近分岔施力越小）。實驗組沿 v1 很快讓 proximity 飽和（→1.0），其後續對齊事件量級退化為 min nudge、淨位移停滯；對照組停在較低 proximity，持續取得全幅事件，長序列下淨位移反而超越實驗組。
>
> | 動作數 | 事件數 | 實驗組 max proximity | Cohen's d |
> |---|---|---|---|
> | 6 | 2 | 0.77 | +0.56 |
> | 12 | 4 | 0.94 | **+0.86**（峰值，Welch p=1.4e-3，power 0.885） |
> | 18 | 6 | 0.99 | +0.03 |
> | 30 | 10 | 1.00 | **−0.64（反轉）** |
>
> **此與 §6 line 82「遊戲設計預期為 30 步」直接衝突** —— 30 步（約 10 事件）正落在飽和反轉區。重現了歷史「效果反轉」，確認其為 **proximity 調制 DV 在飽和時的量測假象**，非科學發現。
>
> **DV 對照測試（在同一份資料上跑四種 DV）**：根因是 DV 家族選錯，不是序列長度。事件強度受 proximity 調制 → 對齊組接近臨界後自我節流、隨機組永遠拿全力，故**任何位移量級 DV 都偏袒對照組**。
>
> | DV | 最佳區(12 步) | 反轉區(30 步) | 抗飽和 |
> |---|---|---|---|
> | 淨位移 total_displacement | d=+0.82 ✅ | d=−0.64 ✗ | 否（僅飽和前） |
> | 路徑長度 path_displacement | d=−2.40 ✗ | d=−4.90 ✗ | **否（更糟）** |
> | **max_proximity** | **d=+4.44（p=4e-17）** | **d=+3.78（p=3e-15）** | **是** |
> | n_critical_crossings | exp 1.0/ctrl 0.0（p≈0） | d=+2.82 ✅ | 是（計數，var=0 時 d 退化） |
>
> **更正**：先前建議的「路徑長度 DV」經實測**證明錯誤（反而放大反轉）**。正確建議：
> **① 將主要客觀 DV 改為 `max_proximity`**（連續、無混淆、跨區間穩健 d≈+3.8~+4.4）。此亦復活了**原始**預先登記的 crossing 類 DV（`n_critical_crossings`，當初棄用僅因舊裝置全飽和成 1.0，Space A/B 修復後已不再）；優先用 `max_proximity` 以避免零變異退化。
> **② 若保留淨位移為次級 DV**，須把事件序列收到實驗組 max proximity < ~0.95（`intensity_scale=1.0` 下約 ≤ 4 事件/場次），它僅在飽和前有效。
> 詳見 [reports/experiments/p7h_engine_sim/REGIME_FINDING.md](../../../reports/experiments/p7h_engine_sim/REGIME_FINDING.md)。
