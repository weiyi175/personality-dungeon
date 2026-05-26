# SDD 12D 封存遷移藍圖

**建立日期**: 2026-05-25  
**執行者**: GitHub Copilot  
**授權**: 用戶授予完整執行權限  

---

## 目標

將 SDD.md 中所有 **12 維人格模型**相關實驗記錄封存至 `SDD_12D_備份/SDD_12D.md`，
並從 SDD.md 中刪除這些段落，以確保 9D（Enneagram）為唯一主線。

**硬性限制**：
- ⛔ 絕對不得觸碰 `研發日誌.md`
- ⛔ 不移動 §7.4 以後的 9D 規格（`7.4 文本 → 9 維人格推斷` 起為 9D）
- ⛔ 不移動仍供 9D 實驗使用的 `02_event_templates_*.json`
- ✅ W1 主線（L3660-L3730）KEEP（世界狀態維度，非人格維度）
- ✅ EXP-1.2（L3790+）KEEP（明確引用 9D `DOMINANT_TEMPLATES`）
- ✅ H1-H5.5R（payoff/threshold/memory 系列）KEEP（與人格維度無關）

---

## 精確行號邊界（已驗證）

| # | 段落 | 起始行 | 結束行 | 行數 | 動作 |
|---|------|--------|--------|------|------|
| A | §4.7 item 0 "Personality Event Schema 加嚴（12維）" | 1879 | 1951 | 73 | ARCHIVE |
| B | §7.2 "12 維 Personality Vector" | 2230 | 2236 | 7 | ARCHIVE |
| C | §7.3 "Personality/Event 新主線最小 smoke 契約（2026-04-01）" | 2237 | 2279 | 43 | ARCHIVE |
| D | H6 "完整 Personality + Event 世界模型主線" | 2708 | 2737 | 30 | ARCHIVE |
| E | H7 header + H7.1-H7.5 | 2738 | 3023 | 286 | ARCHIVE |
| F | H7.6 "Noise Amplitude Sweep" | 3024 | 3110 | 87 | ARCHIVE |
| G | H7.7 "Corner Escape Work" | 3111 | 3207 | 97 | ARCHIVE |
| H | H8 系列（H8.0-H8.4 + 最終總結） | 3208 | 3658 | 451 | ARCHIVE |
| I | W2.1 12D testament 契約（含 P_i∈[-1,1]^12） | 3739 | 3789 | 51 | ARCHIVE |

**總計封存行數：~1125 行**

---

## docs/ 檔案封存清單

| 原路徑 | 決定 | 原因 |
|--------|------|------|
| `docs/personality_dungeon_v1/01_personality_dimensions_v1.json` | **KEEP IN PLACE** | `players/SDD.md` + `README.md` 仍引用；移動需同步更新多個文件 |
| `docs/personality_dungeon_v1/03_personality_projection_v1.py` | **KEEP IN PLACE** | `simulation/personality_gate0.py` line 99 **live importlib call** — 移走即破壞管線 |
| `docs/personality_dungeon_v1/05_little_dragon_v1.py` | **KEEP IN PLACE** | `simulation/personality_h6.py` line 542 **live importlib call** — 移走即破壞管線 |

> ⚠️ **決策說明**: 上述三個檔案的遷移需要同步修改 simulation 代碼（`personality_gate0.py` 的 `PROJECTION_MODULE_PATH`、`personality_h6.py` 的 `LITTLE_DRAGON_MODULE_PATH`），以及更新 `README.md`/`docs/FILES.md`/`players/SDD.md`。此工作量超出本次封存遷移的既定範圍，且有破壞研究管線的風險，故暫緩，待單獨 Track B 清理時一併處理。

**不移動**（9D 仍在使用）：`02_event_templates_*.json`, `02_event_templates_field_matrix.md`, `00_world_framework_v1.md`, `04_economy_rules_v1.json`


---

## SDD.md 小文字修正

| 行號 | 原文 | 改為 |
|------|------|------|
| ~49 | `players/`：RL 玩家定義（**之後擴充 12 維 personality vector**） | 刪除「（之後擴充 12 維 personality vector）」|
| ~69 | 不是要一次做完 **12 維人格 + 事件系統 + Boss stackelberg** | 改為「不是要一次做完人格 + 事件系統 + Boss stackelberg」|

---

## 執行清單（逐步簽核）

### Phase 0：準備
- [x] 確認所有12D段落精確行號（✅ 已完成）
- [x] 建立 `SDD_12D_備份/` 目錄（✅ 已完成）
- [x] 建立本藍圖文件（✅ 已完成）

### Phase 1：建立封存文件
- [ ] 1.1 建立 `SDD_12D_備份/SDD_12D.md` 封存標頭
- [ ] 1.2 複製 §4.7 item 0 (L1879-L1951) 至 SDD_12D.md
- [ ] 1.3 複製 §7.2 (L2230-L2236) 至 SDD_12D.md
- [ ] 1.4 複製 §7.3 (L2237-L2279) 至 SDD_12D.md
- [ ] 1.5 複製 H6 (L2708-L2737) 至 SDD_12D.md
- [ ] 1.6 複製 H7 header + H7.1-H7.5 (L2738-L3023) 至 SDD_12D.md
- [ ] 1.7 複製 H7.6 (L3024-L3110) 至 SDD_12D.md
- [ ] 1.8 複製 H7.7 (L3111-L3207) 至 SDD_12D.md
- [ ] 1.9 複製 H8 系列 (L3208-L3658) 至 SDD_12D.md
- [ ] 1.10 複製 W2.1 12D content (L3739-L3789) 至 SDD_12D.md

### Phase 2：從 SDD.md 刪除 12D 段落（大到小順序，避免行號偏移）
- [ ] 2.1 刪除 W2.1 12D content (L3739-L3789)
- [ ] 2.2 刪除 H8 系列 (L3208-L3658)
- [ ] 2.3 刪除 H7.7 (L3111-L3207)
- [ ] 2.4 刪除 H7.6 (L3024-L3110)
- [ ] 2.5 刪除 H7 header + H7.1-H7.5 (L2738-L3023)
- [ ] 2.6 刪除 H6 (L2708-L2737)
- [ ] 2.7 刪除 §7.3 (L2237-L2279)
- [ ] 2.8 刪除 §7.2 (L2230-L2236)
- [ ] 2.9 刪除 §4.7 item 0 (L1879-L1951)

### Phase 3：SDD.md 文字修正
- [ ] 3.1 修正 `players/` 描述（移除「12 維 personality vector」提及）
- [ ] 3.2 修正非目標聲明（移除「12 維人格」字樣）

### Phase 4：docs/ 檔案搬移
- [x] 4.1 驗證 `00_world_framework_v1.md` → KEEP IN PLACE（README 引用）
- [x] 4.2 驗證 `04_economy_rules_v1.json` → KEEP IN PLACE（FILES.md + event_templates_field_matrix 引用）
- [x] 4.3 `01_personality_dimensions_v1.json` → **KEEP IN PLACE**（多文件引用）
- [x] 4.4 `03_personality_projection_v1.py` → **KEEP IN PLACE**（`personality_gate0.py` live importlib）
- [x] 4.5 `05_little_dragon_v1.py` → **KEEP IN PLACE**（`personality_h6.py` live importlib）
- [x] 4.6 docs 遷移暫緩：待 Track B 清理時一併處理

### Phase 5：最終驗收
- [x] 5.1 grep 確認 SDD.md 中已無 "H7.6/H7.7/H8 系列" headers — ✅ 0 matches
- [x] 5.2 grep 確認 SDD.md 中已無 `P_i∈[-1,1]^{12}` — ✅ 0 matches
- [x] 5.3 確認 `SDD_12D_備份/SDD_12D.md` 完整（82347 bytes, 1125+ archived lines）
- [x] 5.4 確認 `SDD_12D_備份/` 結構正確（MIGRATION_BLUEPRINT.md + SDD_12D.md + SDD.md.bak）
- [x] 5.5 確認 SDD.md 行數合理：5154 行（原 6282，刪除 1128 行）
- [x] 5.6 `研發日誌.md` 未被觸碰 — ✅ md5: `04c0e5d4af322c6deb79ed1e405ca98b`
- [x] 5.7 `players/` 描述已更新為「9D Enneagram；trait set 已鎖定」
- [x] 5.8 B4/B3.2 `z_k` 公式已更新為 `assertiveness - risk_aversion`（9D）

---

## 完成時間

**執行完成時間**：2026-05-25  
**執行結果**：✅ 全部簽核通過

### 殘留注意事項（待 Track B 處理）
1. `simulation/personality_h6.py` + `personality_gate0.py` 仍 importlib 呼叫 12D era docs（`03_personality_projection_v1.py` / `05_little_dragon_v1.py`）— 代碼需更新但超出本次範圍
2. `docs/FILES.md` + `README.md` 仍列出 12D 文件（01/03/05）— 待 Track B 統一清理
3. `players/SDD.md` 仍引用 H6 Gate 2 / 03_projection_v1 路徑 — 待另行確認
4. `EventBridge._PERSONALITY_KEYS_ORDERED` 殘留 12D 舊鍵 — TODO item L5147 已記錄

---

## 封存原則聲明

> 本封存依據 `players/rl_player.py` commit `355e481`（2026-05-04）正式從 12D 切換
> 至 9D（Enneagram）。所有 12D 實驗已由 9D 重驗，封存內容僅供歷史查閱。
> 
> 封存後：
> - 禁止再以 `SDD_12D.md` 的 H6/H7/H8 規格作為主線參照
> - 主線 Spec 唯一來源：`SDD.md`
> - 9D trait 集合：`impulsiveness, assertiveness, optimism`（Drivers）；
>   `risk_aversion, suspicion, endurance`（Stabilizers）；
>   `randomness, stability_seeking, curiosity`（Explorers）
