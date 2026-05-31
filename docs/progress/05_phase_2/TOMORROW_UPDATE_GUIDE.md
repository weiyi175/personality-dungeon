# 📋 Tomorrow's Update Guide (If Test Passes)

**日期**: 2026-05-29（測試通過後執行）  
**前置條件**: [QUICK_TEST_CHECKLIST.md](QUICK_TEST_CHECKLIST.md) 全部 ✅ Pass

---

## 更新步驟 (15 min)

### Step 1: 更新 milestone_review.md

**檔案**: `docs/progress/04_phase_review/milestone_review.md`

找到 Phase 2 subtasks 表格，將以下行的 Status 從 `In Progress` 改為 `Done`：

```markdown
| P2-SC-01 | Godot Scaffold | ... | Status | ... | Done |
| P2-RLC-01 | API Integration | ... | Status | ... | Done |
```

同時填入 Evidence 連結：

```markdown
| P2-SC-01 | ... | Evidence | [src/ui/PlayableLoopScene.tscn](../../../src/ui/PlayableLoopScene.tscn) + logs | ... | Done |
| P2-RLC-01 | ... | Evidence | logs/playable_test_YYYY-MM-DD.log | ... | Done |
```

**複製貼上範本**:
```markdown
| P2-SC-01 | Godot Scaffold | PlayableLoopController + SessionStatusPanel + PlayableLoopScene | 1) 場景載入無 error 2) Button click handlers exist | [src/ui/PlayableLoopScene.tscn](../../../src/ui/PlayableLoopScene.tscn) + [PlayableLoopController.gd](../../../src/core/PlayableLoopController.gd) | Integration Lead | Done |
| P2-RLC-01 | API Integration | 連接 RLSessionAPIClient signal handlers | 1) initialize_completed 被呼叫 2) step_completed 被呼叫 | logs/playable_test_2026-05-29.log | Backend Lead | Done |
```

### Step 2: 更新 signoff_checklist.md

**檔案**: `docs/progress/04_phase_review/signoff_checklist.md`

在 Phase 1 Signoff 表格後添加 **Phase 2 Signoff (Draft, 2026-05-29)**：

```markdown
### Phase 2 Signoff (Draft, 2026-05-29)

- Phase: Phase 2 — Playable Loop & UI Integration
- Checklist:
	- [x] PlayableLoopScene.tscn 場景建置完成
	- [x] 連接 RLSessionAPIClient signal handlers
	- [x] UI (RadarChart + SessionStatusPanel) 與 snapshot 同步
	- [x] 1 完整迴圈 (init → 5 steps → ending) 無例外
	- [x] 所有測試通過，無 HTTP 4xx/5xx 錯誤
- Evidence:
	1) [src/ui/PlayableLoopScene.tscn](../../../src/ui/PlayableLoopScene.tscn)
	2) logs/playable_test_2026-05-29.log
	3) [screenshot/playable_5step.png](screenshot/playable_5step.png) (可選)
- Signoff Table:

| Role | Name | Decision | Date | Notes |
|---|---|---|---|---|
| Integration Lead |  | Approve/Reject | YYYY-MM-DD | PlayableLoopScene 場景建置 ✓ |
| Backend Lead |  | Approve/Reject | YYYY-MM-DD | API integration 正常 ✓ |
| UX Lead |  | Approve/Reject | YYYY-MM-DD | UI 更新流暢無卡頓 ✓ |
| QA Lead |  | Approve/Reject | YYYY-MM-DD | 5-step smoke 全通過 ✓ |
```

### Step 3: 更新三軸進度摘要（可選）

**檔案**: `docs/progress/04_phase_review/milestone_review.md` (Milestone Summary 表格)

更新 Product Progress 和 Integration Progress：

```markdown
| Dimension | Previous | Current | Delta | Notes |
|---|---:|---:|---:|---|
| Research Progress | 85 | 85 | -- | Phase 1 frozen, Phase 3 planning |
| Product Progress | 45 | 55 | +10 | Playable loop UI 完成，可互動流程就位 |
| Integration Progress | 52 | 62 | +10 | API ↔ Godot bridge 驗證完成 |
```

### Step 4: 更新行動項目狀態（可選）

**檔案**: `docs/progress/04_phase_review/milestone_review.md` (決策與行動表格)

更新 MA-002：

```markdown
| MA-002 | 完成 step->UI 映射一輪可玩流程 | Integration Lead | 2026-06-12 | RB-02, PL-003 | Done |
```

---

## 驗證更新

更新完成後，檢查以下項目：

- [ ] milestone_review.md 中 P2-SC-01/RLC-01 Status = Done
- [ ] Evidence 連結有效（檔案存在）
- [ ] signoff_checklist.md 中 Phase 2 Signoff 表格完整
- [ ] 無 markdown 語法錯誤（檔案可正常渲染）

---

## 如未通過測試

如果 [QUICK_TEST_CHECKLIST.md](QUICK_TEST_CHECKLIST.md) 中任何項目 ❌ Fail：

1. **記錄錯誤信息** → 複製 Godot Output 日誌
2. **參考故障排查**  
   - [VERIFICATION_GUIDE.md 故障排查表](VERIFICATION_GUIDE.md#故障排查)
   - 或 [PlayableLoopController.gd 源碼註釋](../../../src/core/PlayableLoopController.gd)
3. **修復後重新測試** → 若修復成功再執行上述更新步驟

---

## Phase 3 準備（本週內）

Phase 2 Signoff 完成後，立即啟動 Phase 3：

**文檔**: [docs/progress/06_phase_3/instrumentation_plan.md](../../docs/progress/06_phase_3/instrumentation_plan.md)

**目標**: 為 session 與 step 埋點，建立指標收集與告警規則

**Timeline**: 
- S1: 埋點核心 (2026-06-05)
- S2: Metrics backend (2026-06-08)
- S3: Dashboard/報告 (2026-06-12)
- S4: 集成與簽核 (2026-06-19)

---

**檢查清單完成時間**: ~15 分鐘  
**預計提交時間**: 明日上午
