# 高耦合失敗診斷提問框架 (High-Coupling Analysis Prompt Template)

**版本**：1.0  
**日期**：2026-04-17  
**用途**：標準化提問格式，用於 AI 分析、論文寫作、實驗設計中；確保邏輯一致、邊界清晰  

---

## 核心提問

請根據以下實驗結論，分析「低耦合世界」與「高耦合世界」在因果結構上的差異，並指出為何低耦合 MVP 能穩定到達 Level 3，而高耦合世界卻穩定卡在 Level 2。

---

## 研究重點（5 大柱子）

### 1. 低耦合世界的成功關鍵
- **定義**：哪些**事件**與**虛擬數據來源**是 MVP 達到 L3 的必要條件？
- **輸出格式**：表格，包含（Anchor | 規格 | 為何關鍵 | 移除後會怎樣）
- **例題清單**：
  - 固定事件模板集（來源：`02_event_templates_v1.json`，不受世界狀態污染）
  - 固定 risk/reward formula（deterministic，無動態參數）
  - bounded memory cap（20-sentence，責任邊界清晰）
  - 控制組 baseline stats（做 uplift 對照）
  - 單一 cycle_level gate（定義不滑動）
  - 單一人格表示（12D vector，語意明確）

### 2. 虛擬錨點被替換的代價
- **定義**：在高耦合版本中，上述哪些虛擬錨點被替換為「真實交互」或「動態狀態」，引入了什麼複雜度？
- **輸出格式**：表格，包含（虛擬資料源 | 低耦合值 | 高耦合值 | 新複雜度）
- **預期結論**：每一項替換都有「局部作用」，但最後都被層層均化吃掉

### 3. 橋接事件與破壞機制
- **定義**：在高耦合世界中，新增的哪些 bridge events（world_state, policy, health, testimony_carryover）最可能破壞 L3 gate？
- **輸出格式**：表格，包含（橋接層 | 機制 | 預期影響 | 實際發生 | 診斷）
- **例題清單**：W1 (world feedback) / W2.1R (death+testament) / W3.1–W3.3 (leader policy) / B2–B5 (local patches)

### 4. 失敗機制的統一歸納
- **定義**：所有高耦合失敗是否可歸納為「**三層均化效應**」？
  - Layer 1: Sampling noise（O(1/√N) 高頻抖動）
  - Layer 2: Popularity averaging（移動平均平滑）
  - Layer 3: Synchronized replicator（所有玩家同相位推進 → 旋轉訊號消滅）
- **輸出格式**：
  - 物理圖像（ASCII 或 Mermaid）
  - 定量診斷表：各項 W/B 實驗對每層的「攻擊」與「失效原因」
  - 結論：exponential 級耗散（非 linear），局部修補無效
- **預期數據支撐**：
  - B5 (tangential drift)：deterministic gate ✓（0.55–0.60），sampled gate ✗（0.51–0.53），差值 0.05 完全被採樣雜訊吃掉
  - B2 (island deme)：inter-deme phase 1.93 rad 但全局採樣立刻均化
  - W3.x (policy)：policy 啟動但無法改寫 attractor basin

### 5. Phase-based Coupling 實驗方案
- **定義**：提出一套逐步從低耦合過渡到高耦合的驗證方案，目標是找到「第一個破壞 L3 的橋接點」
- **約束條件**：
  - 一次只增加一條 bridge（e.g., Phase 1: read-only world state）
  - 必須保留的虛擬錨點：
    - 固定事件模板
    - 固定 reward/risk formula
    - bounded memory cap（hard constraint）
    - control baseline
    - 單一 cycle_level gate
  - 每個 Phase 必須驗證「L3 rate 是否保持」
- **輸出格式**：
  - Phase 0–3 的階層表（Phase | 做什麼 | 保留的 anchor | 驗證標準）
  - Phase 1 詳細設計（core_changes, control dict, verdict_criteria）
  - 虛擬錨點保留清單（表格，列出各 anchor 在各 phase 的狀態：✅/⚠/❌）
- **預期里程碑**：
  - Phase 0：確認 L3 基線穩定（6/6）
  - Phase 1：檢查 read-only world state 本身是否破壞（預期：PASS）
  - Phase 2：檢查 difficulty modulation（預期：可能 FAIL）
  - Phase 3：完整高耦合（預期：FAIL，驗證假說）

---

## 輸出格式要求

### 必填項

1. **比較表格**：邊界清晰，包含「虛擬數據錨點」、「真實交互來源」、「新增橋接事件」的三欄
2. **機制診斷表**：W/B series 對三層均化的發動與失效
3. **Phase 設計表**：Phase 0–3 的驗證路線圖
4. **虛擬錨點清單**：Phase 跨度下的保留狀態

### 選填項

- ASCII 或 Mermaid 圖解（因果鏈、均化物理）
- 定量數據（上傳或引用 outputs/ 中的實際實驗結果）
- 數學公式（如三層均化的衰減率計算）

---

## 既有文件關聯

本框架已在以下文件中完整實現：

- **[docs/high_coupling_failure_diagnosis_spec.md](./high_coupling_failure_diagnosis_spec.md)**：獨立規格文件，包含完整的 5 大柱分析
- **[研發日誌.md § Phase 99](../研發日誌.md#phase-99)**：集成到研發日誌，作為正式診斷章節

---

## 未來使用場景

### 場景 1：論文寫作
提交給合作者：「根據這個提問框架，補充您的章節」→ 確保邏輯一致

### 場景 2：AI 對話補強
「請根據這個框架，在以下方面深入分析...」→ 快速聚焦

### 場景 3：實驗設計驗證
啟動 Phase 1 前：按照框架檢查清單，確保所有 anchor 都被正確識別

### 場景 4：負結果總結
論文 negative results section：直接引用框架的「已排除路線表」

---

## 檢查清單

在填充這個框架時，確保：

- [ ] 5 個研究重點都有對應的表格或分析
- [ ] 虛擬錨點與真實交互來源明確區分
- [ ] 三層均化的指數級耗散有定量證據（不是定性描述）
- [ ] Phase-based coupling 的每個 phase 都有明確的 PASS/FAIL 判決標準
- [ ] 所有引用都能回溯到 SDD.md、研發日誌.md 或 outputs/ 中的實驗結果
- [ ] 沒有與 SDD.md 既有定義（BL1/BL2/RL-CLOSE）發生矛盾

---

**簽核**：框架發佈，可用於後續實驗 / 論文 / 對話中  
**下一步**：Phase 1 實驗設計完成後，反過來驗證本框架的「預期結論」是否準確
