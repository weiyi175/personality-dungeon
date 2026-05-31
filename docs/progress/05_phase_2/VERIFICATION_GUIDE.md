# Phase 2 Playable Loop 驗證指南

## 快速開始（手動測試）

### 前置準備
1. ✅ Python API 伺服器已在 http://localhost:8000 運行
   - 檢查：`curl http://localhost:8000/rl_sessions/initialize -X POST -H "Content-Type: application/json" -d '{"n_players": 4, "n_rounds": 200, "burn_in": 50, "seed": 42}'`
   - 應回傳 200 OK + session_id

2. ✅ Godot 已安裝以下腳本檔案：
  - [src/core/RLSessionAPIClient.gd](../../../src/core/RLSessionAPIClient.gd)
  - [src/core/PlayableLoopController.gd](../../../src/core/PlayableLoopController.gd)
  - [src/ui/PlayableLoopScene.gd](../../../src/ui/PlayableLoopScene.gd)
  - [src/ui/SessionStatusPanel.gd](../../../src/ui/SessionStatusPanel.gd)
  - [src/ui/RadarChart.gd](../../../src/ui/RadarChart.gd) (已更新 update_from_snapshot)

### 在 Godot 編輯器中建立場景

**方法 1：手動建立（建議新手）**

1. 在 Godot 編輯器中新增場景：File → New Scene
2. 根節點選 `Control`（命名為 `PlayableLoopScene`）
3. 添加佈局（使用 FileSystem 或直接在編輯器中）：
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

   補充：`RLSessionAPIClient.gd` 不需要手動放在場景樹，會由 [PlayableLoopController.gd](../../../src/core/PlayableLoopController.gd) 在 `_ready()` 動態 `load + add_child`。

4. 保存場景為 `res://src/ui/PlayableLoopScene.tscn`

5. 在主場景中載入或直接在編輯器預覽

**方法 2：自動建立（GDScript 方式）**

如果想避免手動設置，可以在 [src/ui/PlayableLoopScene.gd](../../../src/ui/PlayableLoopScene.gd) 中添加動態建立邏輯（目前留作 TODO）。

### 驗證流程（5 步測試）

#### Step 1: 檢查場景能否載入
- 在 Godot 編輯器中開啟 `PlayableLoopScene.tscn`
- 觀察輸出日誌：
  ```
  [PlayableLoopController] RLSessionAPIClient 已載入，base_url=http://localhost:8000
  [PlayableLoopScene] 場景連接完成
  ```
- ✅ 預期：無 error，UI 元件全部可見

#### Step 2: 按下「開始」按鈕
- 在編輯器預覽中（F5 或 Play 按鈕）
- 觀察 PlayableLoopController 輸出：
  ```
  [PlayableLoopController] 狀態轉換: 1 (INITIALIZING)
  [PlayableLoopController] 啟動會話，payload={...}
  ```
- 等待 ~1-2 秒
- 預期輸出：
  ```
  [PlayableLoopController] 會話初始化完成：xxxxxxxx (phase=burn-in)
  [PlayableLoopController] 狀態轉換: 2 (WAITING_FOR_STEP)
  ```
- ✅ UI 應顯示：
  - SessionID: `xxxxxxxx` (前 8 字)
  - Round: `0`
  - Phase: `burn-in`
  - Risk: `0.0000`

#### Step 3: 按下「下一步」按鈕（連 5 次）
- 每次點擊觀察輸出：
  ```
  [PlayableLoopController] 狀態轉換: 3 (STEPPING)
  [PlayableLoopController] 觸發步驟...
  [PlayableLoopController] 回合 1 完成 (phase=burn-in, risk_mean=XXXX)
  [PlayableLoopController] 狀態轉換: 2 (WAITING_FOR_STEP)
  ```
- ✅ UI 應實時更新：
  - Round: `1`, `2`, `3`, `4`, `5`
  - RadarChart 雷達圖應跟著更新（頂點位置改變）

#### Step 4: 驗證 5 步後的狀態
- 檢查 RadarChart 是否根據 snapshot 數據重繪
- 檢查 SessionStatusPanel 是否顯示正確的統計資訊
- ✅ 預期：
  - 無 exception 或 HTTP 錯誤
  - 所有 5 步都回傳 200 OK
  - Snapshot 欄位完整非空

#### Step 5: 驗證結局偵測
- 繼續按「下一步」直到 phase 變為 `"ended"` 或 `"final"`
- ✅ 預期行為：
  - 當偵測到結局：
    ```
    [PlayableLoopController] 會話已結束
    [PlayableLoopController] 狀態轉換: 4 (ENDED)
    ```
  - UI 變化：
    - StatusPanel 顯示「✅ 會話已結束」
    - 「下一步」按鈕變灰（disabled）
    - 「重新開始」按鈕啟用
  - 點擊「重新開始」後恢復 IDLE 狀態

### 預期的 API 呼叫順序

```
User Click "開始"
  ↓ (HTTP)
POST http://localhost:8000/rl_sessions/initialize
Content-Type: application/json
{
  "n_players": 4,
  "n_rounds": 200,
  "burn_in": 50,
  "seed": RANDOM,
  "personality_mode": "balanced"
}
  ↓ (Response 200 OK)
{
  "session_id": "uuid-xxxx",
  "initial_snapshot": {
    "session_id": "uuid-xxxx",
    "round": 0,
    "tick": 0,
    "warm": false,
    "phase": "burn-in",
    "p_aggressive": 0.33,
    "p_defensive": 0.33,
    "p_balanced": 0.34,
    ...28 fields...
  }
}
  ↓ PlayableLoopController 發出 initialize_completed signal
  ↓ UI 更新 (round=0, phase=burn-in)

---

User Click "下一步" (5x)
  ↓ (HTTP)
POST http://localhost:8000/rl_sessions/{session_id}/step
Content-Type: application/json
{}
  ↓ (Response 200 OK)
{
  "session_id": "uuid-xxxx",
  "snapshot": {
    "session_id": "uuid-xxxx",
    "round": 1,
    "tick": 1,
    "warm": false,
    "phase": "burn-in",
    ...
  }
}
  ↓ PlayableLoopController 發出 step_completed signal
  ↓ UI 更新 (round=1)
  ↓ RadarChart.update_from_snapshot() 重繪
```

## 故障排查

| 問題 | 可能原因 | 解決 |
|---|---|---|
| 場景載入失敗，報告「找不到節點」 | 場景結構或命名不符 | 優先比對 [QUICK_TEST_CHECKLIST.md](QUICK_TEST_CHECKLIST.md) 的節點名稱（特別是 `ButtonsHBox`）與 [PlayableLoopScene.gd](../../../src/ui/PlayableLoopScene.gd) 綁定路徑 |
| 按下「開始」無反應 | API 伺服器未運行 | 確認 http://localhost:8000 可訪問 |
| 「開始」後顯示 HTTP 錯誤 | API payload 格式不對或欄位缺失 | 檢查 RLSessionAPIClient.initialize_session() 的驗證邏輯 |
| RadarChart 不更新 | update_from_snapshot() 未被呼叫或 method_missing | 確認 RadarChart 已附加腳本，PlayableLoopController 已呼叫 |
| 結局不被偵測 | phase 欄位名稱不同 | 檢查 snapshot 中實際的 phase 值（應為 "ended" 或 "final"） |

## 後續步驟（如無問題）

1. ✅ 完成 5-step smoke test 無例外 → Mark P2-PL-01 as Done
2. ✅ 拍攝視頻或截圖作為 evidence → 存入 [video/playable_5step.mp4](video/playable_5step.mp4)
3. ✅ 更新 [docs/progress/02_demo_review/playable_loop_review.md](docs/progress/02_demo_review/playable_loop_review.md)
4. ✅ 啟動 Phase 3（Instrumentation）規劃

## 參考檔案

- 架構規劃：[docs/progress/05_phase_2/playable_loop_plan.md](../05_phase_2/playable_loop_plan.md)
- Smoke 日誌：[logs/smoke_init.json](logs/smoke_init.json), [logs/smoke_steps.json](logs/smoke_steps.json)
- API 契約：[docs/progress/01_runtime_bridge/api_contract_status.md](../01_runtime_bridge/api_contract_status.md)
