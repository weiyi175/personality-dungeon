# ✅ Phase 2 Manual Test Checklist (Today — 10 min)

**目標**: 驗證 PlayableLoopScene 在 Godot 中正常工作，init → 5 steps → ending 完整迴圈無錯誤。

---

## 準備 (2 min)

- [ ] **API 伺服器運行**
  ```bash
  cd /home/user/personality-dungeon
  ./venv/bin/python -m api.server
  # Expected: "Uvicorn running on http://0.0.0.0:8000"
  ```

- [ ] **Godot 編輯器已開啟**
  - Project: `/mnt/c/Users/n1166/personality-dungeon/`

---

## 場景建置 (3 min)

按照 [VERIFICATION_GUIDE.md](../../docs/progress/05_phase_2/VERIFICATION_GUIDE.md) **方法 1** 建立 PlayableLoopScene：

```
PlayableLoopScene (Control, script: res://src/ui/PlayableLoopScene.gd)
├── PlayableLoopController (Node, script: res://src/core/PlayableLoopController.gd)
└── VBoxContainer (VBoxContainer)
  ├── StatusPanel (PanelContainer, script: res://src/ui/SessionStatusPanel.gd)
  │   └── VBoxContainer (VBoxContainer)
  │       ├── StatusLabel (Label)
  │       ├── SessionIDLabel (Label)
  │       ├── RoundLabel (Label)
  │       ├── PhaseLabel (Label)
  │       ├── RiskLabel (Label)
  │       └── DetailLabel (Label)
  ├── RadarChart (Control, script: res://src/ui/RadarChart.gd)
  └── ButtonsHBox (HBoxContainer)
    ├── InitButton (Button, text="開始")
    ├── StepButton (Button, text="下一步")
    └── ResetButton (Button, text="重新開始")
```

> 命名務必完全一致（尤其是 `ButtonsHBox`），避免 [PlayableLoopScene.gd](../../../src/ui/PlayableLoopScene.gd) 路徑綁定失敗。

### 物件查找對照表（Name / Type / Script）

| Node Name | Type | Script | 檔案存在 | 路徑 |
|---|---|---|---|---|
| PlayableLoopScene | Control | PlayableLoopScene.gd | Yes | res://src/ui/PlayableLoopScene.gd |
| PlayableLoopController | Node | PlayableLoopController.gd | Yes | res://src/core/PlayableLoopController.gd |
| StatusPanel | PanelContainer | SessionStatusPanel.gd | Yes | res://src/ui/SessionStatusPanel.gd |
| RadarChart | Control | RadarChart.gd | Yes | res://src/ui/RadarChart.gd |
| ButtonsHBox | HBoxContainer | None | N/A | scene node only |
| InitButton | Button | None | N/A | scene node only |
| StepButton | Button | None | N/A | scene node only |
| ResetButton | Button | None | N/A | scene node only |
| StatusLabel | Label | None | N/A | scene node only |
| SessionIDLabel | Label | None | N/A | scene node only |
| RoundLabel | Label | None | N/A | scene node only |
| PhaseLabel | Label | None | N/A | scene node only |
| RiskLabel | Label | None | N/A | scene node only |
| DetailLabel | Label | None | N/A | scene node only |

補充：`RLSessionAPIClient.gd` 不需要手動放在場景樹，會由 [PlayableLoopController.gd](../../../src/core/PlayableLoopController.gd) 在 `_ready()` 動態 `load + add_child`。

- [ ] **保存場景** → `res://src/ui/PlayableLoopScene.tscn`
- [ ] **在編輯器中預覽** (F5 或 Play button)
- [ ] **檢查日誌** (Output 面板，應見)
  ```
  [PlayableLoopController] RLSessionAPIClient 已載入，base_url=http://localhost:8000
  [PlayableLoopScene] 場景連接完成
  ```

---

## 功能測試 (5 min)

### Test 1: 初始化（1 min）
- [ ] **按下「開始」按鈕**
  - 預期日誌：
    ```
    [PlayableLoopController] 狀態轉換: 1 (INITIALIZING)
    [PlayableLoopController] 啟動會話，payload={...}
    [PlayableLoopController] 會話初始化完成：xxxxxxxx (phase=burn-in)
    ```
  - 預期 UI 更新：
    - SessionID: `xxxxxxxx`
    - Round: `0`
    - Phase: `burn-in`
    - Risk: `0.0000`

### Test 2: Step 循環（3 min）
- [ ] **連按「下一步」5 次**
  - 每次預期日誌：`回合 N 完成 (phase=burn-in, risk_mean=...)`
  - 預期 UI 更新：
    - Round: `1`, `2`, `3`, `4`, `5` (遞增)
    - RadarChart：雷達圖頂點位置改變

- [ ] **檢查無 HTTP 錯誤**
  - API 伺服器日誌應顯示：
    ```
    INFO:     127.0.0.1:xxxxx - "POST /rl_sessions/initialize HTTP/1.1" 200 OK
    INFO:     127.0.0.1:xxxxx - "POST /rl_sessions/{id}/step HTTP/1.1" 200 OK
    ```

### Test 3: 最短收尾驗證（ending / reset）
- [x] **繼續按「下一步」直到 phase 變成 `"ended"` 或 `"final"`**
  - 已 live 驗證：看到 `會話已結束`
- [x] **確認收尾狀態正確**
  - 已 live 驗證：StatusPanel 顯示「✅ 會話已結束」
  - 已 live 驗證：「下一步」按鈕 disabled
  - 已 live 驗證：「重新開始」按鈕可按
- [x] **按下「重新開始」**
  - 已 live 驗證：回到 IDLE，並可再次按「開始」

---

## 驗證結果

### ✅ Pass 標準
- [x] 場景載入無 error（live verified）
- [x] 「開始」後 session_id 生成，phase=burn-in（live verified）
- [x] 5 次 step 全部回傳 200 OK（live verified）
- [x] Round 遞增 1–5（live verified）
- [x] RadarChart 更新（視覺化變化）（live verified）
- [x] 結局時「下一步」變灰，「重新開始」啟用（live verified）
- [x] 整個流程無 HTTP 4xx/5xx 錯誤（live verified）

### ❌ Fail 標準（停止並通知）
- [ ] 場景載入 error（缺少節點或腳本）
- [ ] HTTP 404/500 錯誤
- [ ] UI 未更新（Round 不變或 RadarChart 不動）
- [ ] Godot exception 出現在 Output

---

## 如測試通過

### 立即做（今天）
1. ✅ 保存 Godot 場景
2. ✅ 截圖或簡短視頻（可選，用於明天提交）

### 明天要做
1. ✅ 更新 `docs/progress/04_phase_review/milestone_review.md`：
   - P2-SC-01: Status → `Done`
   - P2-RLC-01: Status → `Done`
   - 添加 Evidence 連結：`[src/ui/PlayableLoopScene.tscn](../../../src/ui/PlayableLoopScene.tscn)`

2. ✅ 更新 `docs/progress/04_phase_review/signoff_checklist.md`：
   - 新增 Phase 2 Signoff 表格（Template: [signoff_checklist.md](signoff_checklist.md) Phase 2 section TBD）

3. ✅ 標記 **P2-PL-01** as `Done`（完整迴圈驗證）

### 本週內
- ✅ 啟動 Phase 3 規劃執行
  - 文檔已準備：[docs/progress/06_phase_3/instrumentation_plan.md](../../docs/progress/06_phase_3/instrumentation_plan.md)

---

## 故障排查（快速版）

| 症狀 | 檢查項 |
|---|---|
| 場景載入 error | 檢查節點名稱是否與 `_find_ui_components()` 路徑匹配 |
| HTTP 404 | 確認 API 伺服器正運行：`curl http://localhost:8000/rl_sessions/initialize -X POST ...` |
| UI 不更新 | 檢查 RadarChart 和 SessionStatusPanel 是否正確附加腳本和連接到 controller signal |
| Godot 異常 | 檢查 Output 面板，複製錯誤訊息並參考 [VERIFICATION_GUIDE.md 故障排查表](../../docs/progress/05_phase_2/VERIFICATION_GUIDE.md) |

---

**預計總時間**: 10 分鐘（若無重大問題）  
**下一步**: [Tomorrow's Update Guide](../../docs/progress/05_phase_2/TOMORROW_UPDATE_GUIDE.md)
