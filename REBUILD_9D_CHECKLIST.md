# 9D 重建清單（12D 污染修復追蹤）

每完成一項重建，在對應項目加上 `[x]`、填入日期與簽核人，再繼續下一項。
**未簽核前不得視為完成。**

---

## 重建項目

### `[ ]` #1 — `docs/event_json_format_guide.md`

- **問題**：已封存版本中 weights 範例為 12D（`// 12D 個性權重`，12 個數值），
  與現行 9D event template（`02_event_templates_smoke_v1.json`）的 9-weight schema 不符。
- **重建方向**：
  1. 以 `docs/personality_dungeon_v1/02_event_templates_smoke_v1.json` 的實際 JSON 格式為基礎
  2. weights 陣列改為 9 個元素，對應順序：
     `[impulsiveness, assertiveness, optimism, risk_aversion, suspicion, endurance, randomness, stability_seeking, curiosity]`
  3. 欄位說明更新（刪除「12D」字眼，補充三組語意：Drivers / Stabilizers / Explorers）
- **參考**：`simulation/personality_rl_runtime.py::_PERSONALITY_KEYS_ORDERED`
- **簽核**：＿＿＿＿＿＿ 日期：＿＿＿＿＿＿

---

### `[ ]` #2 — `docs/core_frontend_api_contract_v1.md`

- **問題**：已封存版本含 13 處 12D 引用，personality API 欄位（`dimensions_order`、
  weights vector、trait 範例）均為 12D 定義。
- **重建方向**：
  1. 以 `api/` 中現行 FastAPI endpoints 為準（`/sessions/initialize`、`/sessions/{id}/step` 等）
  2. personality 欄位改為 9D Enneagram：
     ```json
     "personality": {
       "impulsiveness": 0.5, "assertiveness": 0.3, "optimism": 0.2,
       "risk_aversion": -0.1, "suspicion": 0.0, "endurance": 0.4,
       "randomness": 0.2, "stability_seeking": 0.1, "curiosity": 0.3
     }
     ```
  3. 保留原有 `ResponseEnvelope v1` 契約結構（additive-only 原則）
- **參考**：`api/`、`SDD.md §12`
- **簽核**：＿＿＿＿＿＿ 日期：＿＿＿＿＿＿

---

### `[ ]` #3 — `docs/personality_rl_runtime_bridge_v1.md`

- **問題**：已封存版本的程式碼範例含 12D 特性名稱（`caution`、`ambition`、
  `fearfulness`），且 L93 標注 `// 12D 個性權重`、L309 標注 `# 12D 向量`。
  文件主體（RL bridge 架構說明）本身屬 9D 設計，可更新後保留。
- **重建方向**：
  1. 保留架構說明主體（`run_simulation.py` vs E1/BL2 per-player Q-table 的差異分析）
  2. 所有程式碼範例換成 9D trait 名稱（參考 `simulation/personality_coupling.py`）：
     ```python
     personality = {
         "impulsiveness": 0.5, "assertiveness": 0.3, "optimism": 0.2,
         "risk_aversion": -0.1, "suspicion": 0.0, "endurance": 0.4,
         "randomness": 0.2, "stability_seeking": 0.1, "curiosity": 0.3
     }
     ```
  3. 刪除「12D」標注字眼
- **參考**：`simulation/personality_coupling.py`、`simulation/personality_rl_runtime.py`
- **簽核**：＿＿＿＿＿＿ 日期：＿＿＿＿＿＿

---

### `[ ]` #4 — `docs/personality_dungeon_v1/00_world_framework_v1.md`

- **問題**：已封存版本的世界框架以 12D 人格維度定義人物行為，與現行
  9D Enneagram 三組（Drivers / Stabilizers / Explorers）不符。
- **重建方向**：
  1. 三大策略組（aggressive / defensive / balanced）對應：
     - **Drivers**（aggressive）：impulsiveness、assertiveness、optimism
     - **Stabilizers**（defensive）：risk_aversion、suspicion、endurance
     - **Explorers**（balanced）：randomness、stability_seeking、curiosity
  2. 以 `docs/personality_dungeon_v1/03_personality_projection_v1.py::PRIMARY_GROUPS` 為正式定義
  3. 不得引用 greed / ambition / caution / fearfulness / patience / persistence
- **參考**：`docs/personality_dungeon_v1/03_personality_projection_v1.py`、`SDD.md §3`
- **簽核**：＿＿＿＿＿＿ 日期：＿＿＿＿＿＿

---

### `[ ]` #5 — `outputs/b1_async_dispatch_poisson_r006/r007` 參數掃描驗證

- **問題**：以下 4 個 summary 檔案在 EventBridge fix（2026-05-18）前執行，
  彼時 `_PERSONALITY_KEYS_ORDERED` 仍含 12D 舊鍵，導致 4/9 維度在
  `compute_reward_risk()` 中靜默歸零（`mean_reward_multiplier_*` 偏低）：
  - `b1_async_dispatch_poisson_r006_w2000_t050_gate60_summary.json`
  - `b1_async_dispatch_poisson_r006_w2000_t050_smoke_summary.json`
  - `b1_async_dispatch_poisson_r007_w2000_t050_gate60_summary.json`
  - `b1_async_dispatch_poisson_r007_w2000_t050_smoke_summary.json`

- **現況確認**：
  - 上述 summary **不含 `mean_reward_multiplier_*` 欄位**（r006/r007 為早期
    純參數掃描，結論為「選 r008」）
  - 主要判斷依據是 `l1/l2/l3/healthy/fairness_fail_count`，與 reward_risk 無直接關聯
  - 因此 **不影響「r008 是最佳事件率」的已確立結論**

- **重建方向**：
  - 若未來有研究需引用 r006/r007 的 personality coupling 效果，
    需以 r006/r007 參數重跑並加 `_9d_` 後綴
  - 現階段**不需要立即重跑**，保留現有檔案作為歷史參照即可
  - 需在 SDD.md 或研究報告中標記：「r006/r007 掃描為 pre-fix 執行，
    reward coupling 數值不可用，僅 cycle 指標（l1/l2/l3）有效」

- **簽核**：＿＿＿＿＿＿ 日期：＿＿＿＿＿＿

---

## 進度摘要

| # | 項目 | 狀態 |
|---|------|------|
| 1 | `docs/event_json_format_guide.md` | `[ ]` 待重建 |
| 2 | `docs/core_frontend_api_contract_v1.md` | `[ ]` 待重建 |
| 3 | `docs/personality_rl_runtime_bridge_v1.md` | `[ ]` 待重建 |
| 4 | `docs/personality_dungeon_v1/00_world_framework_v1.md` | `[ ]` 待重建 |
| 5 | B1 r006/r007 outputs 驗證 | `[ ]` 待確認標記 |

---

*本清單由 migration 2026-05-26 生成。每次簽核請填入 `git commit hash` 或日期。*
