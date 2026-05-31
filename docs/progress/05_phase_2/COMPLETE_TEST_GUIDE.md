# 🎮 完整 Phase 2 測試指南（含 HTTP 伺服器啟動）

**目標**: 在 Godot 編輯器中驗證可玩迴圈：初始化 → 5 步 → 結局  
**預計時間**: 15 分鐘（包含伺服器啟動）  
**必備環境**: Godot 4.x, Python 3.10, venv 已激活

---

## ❶ 啟動 HTTP 伺服器（2-3 分鐘）

### 步驟 1.1: 開啟終端，進入專案目錄

```bash
cd /home/user/personality-dungeon
```

### 步驟 1.2: 確認 venv 已激活

檢查命令列前是否顯示 `(venv)`：

```bash
# 如果沒有 (venv) 前綴，執行：
source ./venv/bin/activate

# 預期輸出：
# (venv) user@DESKTOP-LJ1NKO3:~/personality-dungeon$
```

### 步驟 1.3: 啟動 API 伺服器

```bash
./venv/bin/python -m api.server
```

**預期輸出**（等待 ~3 秒）:
```
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**✅ 確認伺服器已啟動**：在**新的終端視窗**中測試

```bash
# 新開一個終端（不要關閉上面的伺服器終端）
curl -X POST http://localhost:8000/rl_sessions/initialize \
  -H "Content-Type: application/json" \
  -d '{"n_players": 4, "n_rounds": 200, "burn_in": 50, "seed": 42}'

# 預期輸出（JSON）：
# {
#   "session_id": "uuid-xxxx-xxxx-xxxx",
#   "initial_snapshot": { ... }
# }
```

若收到 JSON 回應 → **伺服器成功啟動 ✅**  
若無回應或 error → 參考 [故障排查](#故障排查)

### 步驟 1.4: WSL API + Windows Godot 的 URL 設定（重要）

若你是「API 跑在 WSL、Godot 跑 Windows 可執行檔」，`localhost` 可能不會指到同一個服務。

在 WSL 取得 IP：

```bash
hostname -I | awk '{print $1}'
# 範例: 172.31.143.82
```

然後在 Godot 的 `PlayableLoopController` 節點，將 `api_base_url` 改成：

```text
http://<WSL_IP>:8000
```

例如：`http://172.31.143.82:8000`

---

## ❷ 在 Godot 中建立 PlayableLoopScene（5-7 分鐘）

### 步驟 2.1: 開啟 Godot 編輯器

1. 啟動 Godot
2. 開啟專案: `/mnt/c/Users/n1166/personality-dungeon/`
3. 等待編輯器加載完成（~10 秒）

### 步驟 2.2: 建立新場景

在 Godot 編輯器中：
1. **Scene → New Scene** (或按 Ctrl+N)
2. 選擇 **Control** 作為根節點
3. **Scene → Save Scene** 
4. 儲存為: `res://src/ui/PlayableLoopScene.tscn`

### 步驟 2.3: 附加根節點腳本

1. 選擇根節點 `PlayableLoopScene` (在 Scene tree 左側)
2. **Attach Script** (右鍵 → Attach Script)
3. 選項：
   - **Path**: `res://src/ui/PlayableLoopScene.gd`
   - **Built-in Script** 勾選：**關閉**（我們使用現有檔案）
4. 按 **Create**

### 步驟 2.4: 建立子節點層級

在 `PlayableLoopScene` 下方建立以下節點結構（使用右鍵 → Add Child Node）：

```
PlayableLoopScene (Control) ← 根節點，已附加 PlayableLoopScene.gd
├─ PlayableLoopController (Node)
│  └─ 右鍵 → Attach Script → res://src/core/PlayableLoopController.gd
│
└─ VBoxContainer
   ├─ StatusPanel (PanelContainer)
   │  └─ 右鍵 → Attach Script → res://src/ui/SessionStatusPanel.gd
   │  └─ VBoxContainer (子層)
   │     ├─ StatusLabel (Label)
   │     ├─ SessionIDLabel (Label)
   │     ├─ RoundLabel (Label)
   │     ├─ PhaseLabel (Label)
   │     ├─ RiskLabel (Label)
   │     └─ DetailLabel (Label)
   │
   ├─ RadarChart (Control)
   │  └─ 右鍵 → Attach Script → res://src/ui/RadarChart.gd
   │
   └─ HBoxContainer (用於按鈕排列)
      ├─ InitButton (Button, text="開始")
      ├─ StepButton (Button, text="下一步")
      └─ ResetButton (Button, text="重新開始")
```

### 步驟 2.5: 設定 VBoxContainer 佈局

1. 選擇 `VBoxContainer`
2. **Inspector → Layout**
   - **Separation**: 8
   - **Expand Mode**: Ignore
3. **Size Flags**
   - **Vertical**: Fill
   - **Horizontal**: Fill

### 步驟 2.6: 設定 RadarChart 大小

1. 選擇 `RadarChart`
2. **Size Flags**
   - **Vertical**: Expand Fill
   - **Horizontal**: Expand Fill
3. **Custom Minimum Size**: 300x300 (可選，方便預覽)

### 步驟 2.7: 設定 HBoxContainer (按鈕)

1. 選擇 `HBoxContainer`
2. **Alignment**: Center

### ✅ 場景結構完成

儲存場景 (**Ctrl+S**)，應看到:
```
res://src/ui/PlayableLoopScene.tscn
```

---

## ❸ 執行測試（5 分鐘）

### 步驟 3.1: 在編輯器中預覽場景

1. 在 Scene tree 中選擇根節點 `PlayableLoopScene`
2. **Scene → Play Scene** (或按 **F6**)
   - 或按編輯器右上角的「播放」按鈕

**預期**：
- 視窗彈出，顯示 UI（按鈕、標籤、雷達圖）
- Godot Output 面板顯示：
  ```
  [PlayableLoopController] RLSessionAPIClient 已載入，base_url=http://localhost:8000
  [PlayableLoopScene] 場景連接完成
  ```

### 步驟 3.2: 點擊「開始」按鈕

1. 在彈出的預覽視窗中，點擊「**開始**」按鈕
2. 等待 ~1-2 秒

**預期日誌**（在 Output 面板觀察）:
```
[PlayableLoopController] 狀態轉換: 1 (INITIALIZING)
[PlayableLoopController] 啟動會話，payload={...}
```

等待 ~1-2 秒後：
```
[PlayableLoopController] 會話初始化完成：xxxxxxxx (phase=burn-in)
[PlayableLoopController] 狀態轉換: 2 (WAITING_FOR_STEP)
```

**預期 UI 更新**：
```
SessionID:  xxxxxxxx        (8 字元)
Round:      0
Phase:      burn-in
Risk:       0.0000
```

**預期 API 伺服器日誌**（在終端 1 中觀察）:
```
INFO:     127.0.0.1:XXXXX - "POST /rl_sessions/initialize HTTP/1.1" 200 OK
```

✅ **Checkpoint 1 Pass**：Session 初始化成功

### 步驟 3.3: 連按「下一步」5 次

1. 點擊「**下一步**」按鈕
2. 觀察 Output 日誌
3. 重複 4 次（共 5 次）

**每次的預期日誌**:
```
[PlayableLoopController] 狀態轉換: 3 (STEPPING)
[PlayableLoopController] 觸發步驟...
[PlayableLoopController] 回合 1 完成 (phase=burn-in, risk_mean=0.0000)
[PlayableLoopController] 狀態轉換: 2 (WAITING_FOR_STEP)
```

**每次的預期 UI 更新**:
```
Round:      1 → 2 → 3 → 4 → 5 (遞增)
Phase:      burn-in
Risk:       0.0000 (可能變化)
```

**預期 RadarChart 變化**：
- 雷達圖的 9 個頂點應隨著 step 而移動（視覺化變化）

**預期 API 伺服器日誌**（每次一行）:
```
INFO:     127.0.0.1:XXXXX - "POST /rl_sessions/{id}/step HTTP/1.1" 200 OK
```

✅ **Checkpoint 2 Pass**：所有 5 步都成功

### 步驟 3.4: 驗證結局偵測

1. **繼續按「下一步」**（可能需要 10-20 次）
2. 觀察 Output 日誌和 UI

**當達到結局時的預期日誌**:
```
[PlayableLoopController] 會話已結束
[PlayableLoopController] 狀態轉換: 4 (ENDED)
```

**預期 UI 變化**：
- StatusPanel 頂部顯示：「✅ 會話已結束」
- Phase：通常為「tail」（若後端版本不同，也可能是「ended」或「final」）
- 「**下一步**」按鈕變灰（disabled）
- 「**重新開始**」按鈕變亮（enabled）

✅ **Checkpoint 3 Pass**：結局正確偵測

### 步驟 3.5: 點擊「重新開始」按鈕

1. 點擊「**重新開始**」
2. 觀察 Output 日誌

**預期日誌**:
```
[PlayableLoopController] 重新開始
```

**預期 UI 重置**：
```
SessionID:  ---
Round:      0
Phase:      ---
Risk:       0.0
```

**預期按鈕狀態**：
- 「**開始**」按鈕啟用
- 「**下一步**」按鈕變灰
- 「**重新開始**」按鈕變灰

✅ **Checkpoint 4 Pass**：重新開始功能正常

---

## ✅ 完全通過標準

若以上 4 個 Checkpoints 都通過，則測試 **PASS**：

- [ ] Checkpoint 1: Session 初始化成功（phase=burn-in）
- [ ] Checkpoint 2: 5 步全部成功，HTTP 200 OK
- [ ] Checkpoint 3: 結局被正確偵測（phase=tail 或 ended/final）
- [ ] Checkpoint 4: 重新開始功能正常

**若全部 ✅ PASS**：
1. **停止 Godot 預覽** (按 ESC 或編輯器停止按鈕)
2. 記錄完成時間
3. 進行 [TOMORROW_UPDATE_GUIDE.md](TOMORROW_UPDATE_GUIDE.md)

---

## ❌ 故障排查

### 問題 1: 無法連接到 API 伺服器

**症狀**: 按「開始」後看到錯誤：
```
[PlayableLoopController] 會話失敗：...
```

**檢查**:
```bash
# 在新終端中測試伺服器
curl http://localhost:8000/rl_sessions/initialize \
  -X POST -H "Content-Type: application/json" \
  -d '{"n_players": 4, "n_rounds": 200, "burn_in": 50, "seed": 42}'

# 若無回應或 "connection refused"：
# → 伺服器未啟動，返回步驟 ❶
```

若你是 WSL + Windows Godot 混合環境，請改測：

```bash
WSL_IP=$(hostname -I | awk '{print $1}')
curl -X POST "http://$WSL_IP:8000/rl_sessions/initialize" \
   -H "Content-Type: application/json" \
   -d '{"n_players": 4, "n_rounds": 200, "burn_in": 50, "seed": 42}'
```

### 問題 2: 場景無法載入（找不到節點）

**症狀**: Preview 彈出視窗但無 UI 內容，或 error 訊息

**檢查**:
1. Scene tree 結構是否完全（按步驟 2.4 檢查）
2. 節點名稱是否完全一致（特別是 `PlayableLoopController`, `StatusPanel`, 等)
3. 重新附加腳本（右鍵 → Attach Script）

### 問題 3: 按鈕無反應

**症狀**: 點擊「開始」沒有任何反應

**檢查**:
1. Output 有無錯誤訊息？
2. `PlayableLoopController.gd` 是否正確附加到 Node？
3. 試試手動編輯 `PlayableLoopController.gd`，添加 debug 輸出：
   ```gdscript
   func _on_init_button_pressed() -> void:
       print("[DEBUG] Init button pressed")
       # ... rest of code
   ```

### 問題 4: RadarChart 不更新

**症狀**: Step 後 UI 標籤更新，但雷達圖不變

**檢查**:
1. `RadarChart.gd` 是否附加到 `RadarChart` 節點？
2. `update_from_snapshot()` 方法是否存在？
3. 在 `PlayableLoopController.gd` 中確認有呼叫：
   ```gdscript
   if _radar_chart and _radar_chart.has_method("update_from_snapshot"):
       _radar_chart.update_from_snapshot(snapshot)
   ```

### 問題 5: HTTP 4xx/5xx 錯誤

**症狀**: 伺服器回傳 404 或 500

**檢查日誌**:
1. API 伺服器終端中的完整錯誤訊息
2. Godot Output 中的錯誤詳情
3. 確認 API 端點 URL 正確：`http://localhost:8000/rl_sessions/...`

---

## 📊 測試結果紀錄

測試完成後，填入以下表格：

| 檢查項 | Status | Notes |
|---|---|---|
| API 伺服器啟動 | ✅ / ❌ | URL: http://localhost:8000 |
| Godot 場景載入 | ✅ / ❌ | 無 error |
| Session 初始化 | ✅ / ❌ | phase=burn-in |
| Step ×5 完成 | ✅ / ❌ | Round: 1-5 |
| RadarChart 更新 | ✅ / ❌ | 視覺化變化 |
| 結局偵測 | ✅ / ❌ | phase=tail（或 ended/final） |
| 重新開始 | ✅ / ❌ | UI 重置 |
| **總體結果** | **✅ PASS** / ❌ FAIL | |

---

## 🎬 下一步

### 若全部 ✅ PASS

1. **停止 Godot** (ESC)
2. **停止 API 伺服器** (終端 1，Ctrl+C)
3. **執行 [TOMORROW_UPDATE_GUIDE.md](TOMORROW_UPDATE_GUIDE.md)**
   - 更新 `milestone_review.md`
   - 添加 Phase 2 簽核表

### 若任何 ❌ FAIL

1. 參考 [故障排查](#故障排查)
2. 修復問題
3. **重新測試**（返回步驟 ❸）

---

## 📞 支援文檔

- **Godot 編輯器問題**: [VERIFICATION_GUIDE.md](VERIFICATION_GUIDE.md#故障排查)
- **API 合約問題**: [api_contract_status.md](../../docs/progress/01_runtime_bridge/api_contract_status.md)
- **架構深入理解**: [playable_loop_plan.md](playable_loop_plan.md)

---

**準備好開始？ ✅ 從步驟 ❶ 開始！**
