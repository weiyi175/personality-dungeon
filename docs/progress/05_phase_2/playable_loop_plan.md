# Phase 2 — Playable Loop Implementation Plan

## Metadata

| Field | Value |
|---|---|
| Phase | Phase 2 |
| Goal | 從 Godot UI 啟動 RL session，執行 step 迴圈，顯示結果 |
| Start Date | 2026-05-28 |
| Target Completion | 2026-06-12 |
| Dependencies | Phase 1 signoff ✅ |

## Phase 2 Objectives

### Primary Goal
實裝**一個完整的可玩迴圈**：
1. **Init**: 按下「開始」按鈕 → 呼叫 `/rl_sessions/initialize`
2. **Step Loop**: 顯示每回合的狀態 → 自動 or 手動觸發 step
3. **Display**: 更新 UI（RadarChart、phase label、risk meter）
4. **Ending**: 達到終局後顯示結果卡

### Scope
- **In Scope**: 
  - Godot UI (DebugPanel/RadarChart) 與 RLSessionAPIClient 整合
  - Step 迴圈自動播放或手動觸發
  - 實時顯示 snapshot.phase, snapshot.risk_mean, snapshot.round
  - 結局時停止並顯示統計
- **Out of Scope**:
  - 教學 UI / 結局解釋卡 (Phase 4)
  - Metrics instrumentation (Phase 3)
  - Adaptive/fate 研究驗證 (Phase 5)

## Technical Architecture

### 1. Godot Scene Structure

```
PlayableLoopScene
├── PlayableLoopController.gd (NEW)
│   ├── Manages RLSessionAPIClient lifecycle
│   ├── Handles init/step/display state machine
│   └── Emits signals for UI updates
├── UI Container
│   ├── SessionStatusPanel (NEW)
│   │   ├── Session ID display
│   │   ├── Round counter
│   │   ├── Phase label (burn-in / exploit / etc)
│   │   └── Risk meter
│   ├── RadarChart (EXISTING, reused)
│   │   ├── Display 9D trait vector from snapshot
│   │   ├── Connect to PlayableLoopController.snapshot_updated signal
│   ├── Control Buttons (NEW)
│   │   ├── "開始" (init_session button)
│   │   ├── "下一步" (step button, hidden after ending)
│   │   ├── "重新開始" (reset button)
└── RLSessionAPIClient (EMBEDDED)
    ├── initialize_completed signal → PlayableLoopController
    ├── step_completed signal → PlayableLoopController
    └── session_failed signal → PlayableLoopController

```

### 2. State Machine (PlayableLoopController.gd)

```
IDLE
  ↓ [Click "開始"]
INITIALIZING
  ↓ [initialize_completed]
WAITING_FOR_STEP (or AUTO_STEPPING)
  ↓ [Click "下一步" or auto-advance]
STEPPING
  ↓ [step_completed]
  ├─ [phase != final] → WAITING_FOR_STEP
  └─ [phase == final] → ENDED
ENDED
  ↓ [Click "重新開始"]
IDLE

```

### 3. Signal Flow

```
User clicks "開始"
  ↓
PlayableLoopController._on_init_button_pressed()
  ↓
RLSessionAPIClient.initialize_session(payload)
  ↓ (HTTP request)
/rl_sessions/initialize (API)
  ↓ (response with initial_snapshot)
RLSessionAPIClient.initialize_completed.emit(session_id, snapshot)
  ↓
PlayableLoopController._on_session_initialized(session_id, snapshot)
  ├─ Store session_id
  ├─ Update UI: phase_label, risk_meter, round counter
  └─ Emit snapshot_updated signal
    ↓
SessionStatusPanel, RadarChart subscribe & update display
  ↓
User clicks "下一步"
  ↓
PlayableLoopController._on_step_button_pressed()
  ↓
RLSessionAPIClient.step_session(session_id)
  ↓ (HTTP request)
/rl_sessions/{id}/step (API)
  ↓ (response with new snapshot)
RLSessionAPIClient.step_completed.emit(session_id, snapshot)
  ↓
PlayableLoopController._on_step_completed(session_id, snapshot)
  ├─ Increment round counter
  ├─ Update UI: phase_label, risk_meter
  ├─ Check if snapshot.phase == "ended"
  │  ├─ If no → Enable "下一步" button, emit snapshot_updated
  │  └─ If yes → Disable buttons, show "已結束" panel
  └─ Emit snapshot_updated signal (for RadarChart)
```

## Implementation Checklist

### Phase 2A: Core Scaffold (S1, Due 2026-06-02)
- [ ] Create PlayableLoopController.gd with state machine skeleton
- [ ] Create SessionStatusPanel.gd for round/phase/risk display
- [ ] Create PlayableLoopScene.tscn that embeds above + RadarChart
- [ ] Wire up init/step button click handlers (no HTTP yet)
- [ ] Verify scene loads in Godot editor without errors

### Phase 2B: RLSessionAPIClient Integration (S2, Due 2026-06-05)
- [ ] Create new DungeonSim instance in PlayableLoopController (or use injected)
- [ ] Connect RLSessionAPIClient.initialize_completed → _on_session_initialized()
- [ ] Connect RLSessionAPIClient.step_completed → _on_step_completed()
- [ ] Connect RLSessionAPIClient.session_failed → _on_session_failed()
- [ ] Test: Press "開始" → inspect logs for `/rl_sessions/initialize` call

### Phase 2C: UI Binding (S3, Due 2026-06-08)
- [ ] Bind snapshot.phase → SessionStatusPanel.phase_label.text
- [ ] Bind snapshot.risk_mean → SessionStatusPanel.risk_meter (progress bar or text)
- [ ] Bind snapshot.round → SessionStatusPanel.round_label.text
- [ ] Bind snapshot traits (p_aggressive, p_defensive, etc.) → RadarChart.update_from_snapshot()
- [ ] Test: After step, verify UI updates with new values

### Phase 2D: State Transitions & Ending (S4, Due 2026-06-10)
- [ ] Implement ending detection: `if snapshot.phase == "ended"`
- [ ] On ending: disable "下一步" button, show "已結束" label, show summary stats
- [ ] Implement "重新開始" button to reset state machine to IDLE
- [ ] Test: Run full loop init → 5 steps → verify ending panel appears

### Phase 2E: Verification & Signoff (S5, Due 2026-06-12)
- [ ] Manual play test: 1 full loop (init → steps → ending) without errors
- [ ] Verify snapshot values propagate correctly to UI
- [ ] Verify no duplicate requests or race conditions
- [ ] Capture video or screenshots of playable loop
- [ ] Update playable_loop_review.md with Actual results and Match status
- [ ] Mark P2-PL-01 as Done

## Evidence & Acceptance Criteria

### Verification Template

| Scenario | Expected | Actual | Match | Evidence |
|---|---|---|---|---|
| Init button clicked | session_id generated, phase=burn-in | TBD | TBD | [logs/playable_init.log](logs/playable_init.log) |
| Step button clicked | round increments, snapshot updated | TBD | TBD | [logs/playable_step_1.log](logs/playable_step_1.log) |
| 5-step loop | no exceptions, final phase != null | TBD | TBD | [video/playable_5step.mp4](video/playable_5step.mp4) |
| Ending detection | "已結束" panel appears, buttons disabled | TBD | TBD | [screenshot/ending_panel.png](screenshot/ending_panel.png) |

### Pass/Fail Criteria
- ✅ **Pass**: All scenarios match expected; no HTTP errors; UI updates correctly; 1 full loop completes without hang
- ❌ **Fail**: Any scenario mismatch; HTTP 4xx/5xx errors; UI frozen or out-of-sync; exceptions in logs

## Phase 2 Subtasks

| Subtask ID | Scope | PR 交付物 | Verification Steps | Evidence | Owner | Status |
|---|---|---|---|---|---|---|
| P2-SC-01 | Godot Scaffold | PlayableLoopController + SessionStatusPanel + PlayableLoopScene | 1) 場景載入無 error 2) Button click handlers exist | [src/core/PlayableLoopController.gd](../../../src/core/PlayableLoopController.gd) + [src/ui/SessionStatusPanel.gd](../../../src/ui/SessionStatusPanel.gd) + [src/ui/PlayableLoopScene.tscn](../../../src/ui/PlayableLoopScene.tscn) | Integration Lead | Open |
| P2-RLC-01 | API Integration | 連接 RLSessionAPIClient signal handlers | 1) initialize_completed 被呼叫 2) step_completed 被呼叫 | [logs/playable_init.log](logs/playable_init.log) + [logs/playable_step.log](logs/playable_step.log) | Backend Lead | Open |
| P2-UI-01 | UI Binding | RadarChart + SessionStatusPanel 與 snapshot 同步 | 1) snapshot 欄位映射正確 2) UI 實時更新 | [screenshot/playable_ui_update.png](screenshot/playable_ui_update.png) | UX Lead | Open |
| P2-PL-01 | Playable Loop | 1 完整迴圈（init → steps → ending） | 1) 無 exception 2) 結局正確偵測 | [video/playable_5step.mp4](video/playable_5step.mp4) + [logs/playable_full_loop.log](logs/playable_full_loop.log) | QA Lead | Open |

## Cross-References

- Upstream: [phase_gate_checklist.md](../04_phase_review/phase_gate_checklist.md) Phase 2 acceptance template
- Upstream: [playable_loop_review.md](../02_demo_review/playable_loop_review.md) validation checklist
- Downstream (Phase 3): instrumentation_plan.md (TBD)
- Downstream (Phase 4): demo_harden_plan.md (TBD)

## Revision History

| Date | Author | Change |
|---|---|---|
| 2026-05-28 | Copilot | Initial Phase 2 playable loop plan |
| 2026-05-28 | Copilot | Added architecture, state machine, checklists |

## Future Expansion

- Auto-step 模式：自動每隔 N 秒觸發 step（取代手動點擊）
- 逐幀調速：slider 控制 step 速率
- 多會話管理：同時執行多個 session 進行比較
