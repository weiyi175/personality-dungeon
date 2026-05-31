# ⚡ 快速參考：5 分鐘啟動

## 終端 1：啟動 HTTP 伺服器 (2 min)

```bash
cd /home/user/personality-dungeon
source ./venv/bin/activate
./venv/bin/python -m api.server

# 預期：Uvicorn running on http://0.0.0.0:8000
```

## 終端 2：驗證伺服器正常 (1 min)

```bash
curl -X POST http://localhost:8000/rl_sessions/initialize \
  -H "Content-Type: application/json" \
  -d '{"n_players": 4, "n_rounds": 200, "burn_in": 50, "seed": 42}'

# 應回傳 JSON with session_id
```

## Godot：建立場景 (5 min)

### 快速建置 (複製貼上節點結構)

```
PlayableLoopScene (Control)
  ↳ Attach Script: res://src/ui/PlayableLoopScene.gd
  
├─ PlayableLoopController (Node)
│  ↳ Attach Script: res://src/core/PlayableLoopController.gd
│
└─ VBoxContainer
   ├─ StatusPanel (PanelContainer)
   │  ↳ Attach Script: res://src/ui/SessionStatusPanel.gd
   │  └─ VBoxContainer
   │     ├─ StatusLabel (Label)
   │     ├─ SessionIDLabel (Label)
   │     ├─ RoundLabel (Label)
   │     ├─ PhaseLabel (Label)
   │     ├─ RiskLabel (Label)
   │     └─ DetailLabel (Label)
   │
   ├─ RadarChart (Control)
   │  ↳ Attach Script: res://src/ui/RadarChart.gd
   │
   └─ HBoxContainer
      ├─ InitButton (Button, text="開始")
      ├─ StepButton (Button, text="下一步")
      └─ ResetButton (Button, text="重新開始")
```

### 重要設置

| 節點 | 設置 | 值 |
|---|---|---|
| VBoxContainer | Separation | 8 |
| | Vertical Size Flag | Fill |
| | Horizontal Size Flag | Fill |
| RadarChart | Vertical Size Flag | Expand Fill |
| | Horizontal Size Flag | Expand Fill |
| HBoxContainer | Alignment | Center |

## Godot：執行測試 (5 min)

**F6 預覽**，依序執行：

1. **按「開始」**
   - 預期：Output 顯示 `會話初始化完成`
   - UI 顯示 SessionID, Round=0, Phase=burn-in

2. **連按「下一步」5 次**
   - 預期：Round 遞增 1→5
   - 每次 API 伺服器日誌顯示 `200 OK`

3. **繼續按「下一步」至結局**
   - 預期：Phase 變為 `ended`
   - 「下一步」按鈕變灰
   - 「重新開始」啟用

4. **按「重新開始」**
   - 預期：UI 重置，可再次「開始」

---

## ✅ Pass 標準 (全部 ✓)

- [ ] API 伺服器成功啟動 (Uvicorn)
- [ ] curl 測試回傳 JSON session_id
- [ ] Godot 場景無 error
- [ ] Session 初始化成功
- [ ] 5 步全部 HTTP 200 OK
- [ ] RadarChart 視覺化變化
- [ ] 結局偵測正常
- [ ] 重新開始功能正常

---

## 詳細版本

👉 [COMPLETE_TEST_GUIDE.md](COMPLETE_TEST_GUIDE.md) (完整 15 分鐘指南，含故障排查)

---

**開始: `cd /home/user/personality-dungeon && source ./venv/bin/activate && ./venv/bin/python -m api.server`**
