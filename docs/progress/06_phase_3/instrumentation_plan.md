# Phase 3 — Instrumentation & Metrics

## Metadata

| Field | Value |
|---|---|
| Phase | Phase 3 |
| Goal | 為 session 與 step 加入追蹤欄位，導出到 metrics backend |
| Start Date | TBD (After Phase 2 Signoff) |
| Target Completion | 2026-06-19 |
| Dependencies | Phase 2 playable loop ✅ (P2-PL-01) |
| Blocking | Phase 4 demo hardening |

## Phase 3 Objectives

### Primary Goals
1. **Instrumentation**: 在 RLSessionAPIClient & PlayableLoopController 中埋點
   - Session 級別：session_id, duration, n_players, n_rounds
   - Step 級別：step_id, latency, phase_transition, action_taken
   - RL 指標：reward, utility, success_rate, entropy

2. **Metrics Backend**: 建立簡單的本地或遠程指標收集
   - 寫入到 `logs/metrics_*.jsonl`（追蹤模式）
   - 或連接到 Prometheus / InfluxDB（可選）

3. **Dashboard / Alerts**: 提供基本的查詢與告警能力
   - 統計摘要（avg latency, error rate, etc）
   - 至少一個 alert rule（e.g., latency > 500ms）

### Scope
- **In Scope**:
  - RLSessionAPIClient 中埋點（HTTP request/response timing）
  - PlayableLoopController 中埋點（state transitions, user actions）
  - Metrics 聚合與持久化
  - 簡單的仪表板或 CSV 匯出
- **Out of Scope**:
  - 複雜的分佈式追蹤（Jaeger/Zipkin）
  - 高級機器學習告警
  - 多租戶 metrics 隔離

## Technical Architecture

### 1. Instrumentation Points

#### Python側 (api/server.py, api/rl_session_manager.py)
```python
# 每個 endpoint 埋點：
# - request_time: 請求開始時間
# - response_time: 回應完成時間
# - latency: response_time - request_time
# - status_code: HTTP status
# - error_code: RL error taxonomy (if any)

@app.post("/rl_sessions/initialize")
async def rl_initialize_session(req: RLSessionInitRequest):
    start_time = time.time()
    try:
        # ... existing logic ...
        latency = time.time() - start_time
        # log_metric("rl_init", latency=latency, status=200)
        return response
    except Exception as e:
        # log_metric("rl_init", latency=latency, status=500, error=str(e))
```

#### Godot側 (PlayableLoopController.gd)
```gdscript
# State transition logging
func _on_state_changed(new_state: State):
    var event = {
        "timestamp": Time.get_ticks_msec(),
        "session_id": _session_id,
        "event_type": "state_transition",
        "from_state": State.keys()[_state],
        "to_state": State.keys()[new_state],
        "round": _round_count,
    }
    _log_event(event)

# User action logging
func _on_init_button_pressed():
    var event = {
        "timestamp": Time.get_ticks_msec(),
        "event_type": "user_action",
        "action": "init_click",
    }
    _log_event(event)
```

### 2. Metrics Backend

#### 選項 A：本地 JSONL 日誌（Phase 3A，簡單）
```
logs/metrics_session_2026-05-28.jsonl
{"timestamp": 1234567890, "session_id": "xxx", "event_type": "session_start", "n_players": 4}
{"timestamp": 1234567891, "session_id": "xxx", "event_type": "step_complete", "latency_ms": 245}
...
```

#### 選項 B：簡單 HTTP 接收端點（Phase 3B，稍進階）
```python
# api/metrics.py (NEW)
@app.post("/metrics/events")
async def receive_metrics(events: list[dict]):
    """Receive batch metrics from Godot client."""
    for event in events:
        _persist_metric(event)
    return {"ok": true}
```

Godot Client 定期批量上傳：
```gdscript
var _buffered_events = []
func _process(delta):
    if _buffered_events.size() > 100 or time_since_last_flush > 10s:
        _flush_events_to_backend()
```

### 3. Metrics Query & Reporting

#### 簡單分析腳本 (analysis/metrics_summary.py)
```python
# 讀取 logs/metrics_*.jsonl
# 計算：
#   - avg_step_latency
#   - error_count
#   - state_transition_heatmap
#   - success_rate by session

# 輸出: CSV 或 JSON 報表
```

#### Alert Rules (analysis/alerts.yaml)
```yaml
alerts:
  - name: HighLatency
    condition: avg_step_latency > 0.5s
    severity: warn
  - name: HighErrorRate
    condition: error_count / total_steps > 0.05
    severity: critical
  - name: SessionTimeOut
    condition: session_duration > 30min
    severity: info
```

## Implementation Checklist

### Phase 3A: Instrumentation Core (S1, Due 2026-06-05)
- [ ] Create `api/instrumentation.py` with logging utilities
  - `log_metric(event_type, **kwargs)` → writes to `logs/metrics_*.jsonl`
  - `get_timing_context()` → context manager for latency tracking
- [ ] Modify `api/server.py` endpoints to emit metrics
  - `rl_initialize_session`: log session_start
  - `rl_step_session`: log step_complete (with latency)
  - error handlers: log error events
- [ ] Modify `PlayableLoopController.gd`
  - Add `_log_event(event: Dictionary)` method
  - Call from state changes, button clicks, API responses
- [ ] Test: Run 1 complete loop, verify `logs/metrics_*.jsonl` populated

### Phase 3B: Metrics Backend (S2, Due 2026-06-08)
- [ ] (Optional) Create `api/metrics.py` with `/metrics/events` endpoint
- [ ] Create `analysis/metrics_summary.py`
  - Read JSONL, compute aggregates
  - Output CSV with: session_id, n_steps, avg_latency, error_count, final_phase
- [ ] Create alert configuration (YAML or Python)
  - At least 2 rules (latency, error_rate)
  - Can be triggered manually for testing

### Phase 3C: Dashboard / Reporting (S3, Due 2026-06-12)
- [ ] Create `analysis/generate_metrics_report.py`
  - Reads `logs/metrics_*.jsonl`
  - Generates HTML report or Jupyter notebook
  - Includes: timeseries plots, state transition diagram, error breakdown
- [ ] Create sample alert trigger test
  - Manually set high latency in metrics → verify alert fires
- [ ] Document metrics schema and alert semantics

### Phase 3D: Integration & Signoff (S4, Due 2026-06-19)
- [ ] Run full playable loop with metrics collection
- [ ] Verify metrics propagate correctly (no data loss)
- [ ] Generate metrics report and verify readability
- [ ] Mark P3-INS-01, P3-ALT-01 as Done

## Metrics Schema

### Session Metric
```json
{
  "timestamp": 1234567890,
  "event_type": "session_start",
  "session_id": "uuid-xxxx",
  "n_players": 4,
  "n_rounds": 200,
  "burn_in": 50,
  "seed": 42,
  "personality_mode": "balanced"
}
```

### Step Metric
```json
{
  "timestamp": 1234567900,
  "event_type": "step_complete",
  "session_id": "uuid-xxxx",
  "round": 1,
  "phase": "burn-in",
  "latency_ms": 245,
  "http_status": 200,
  "snapshot_fields": ["p_aggressive", "p_defensive", "risk_mean"],
  "p_aggressive": 0.33,
  "risk_mean": 0.0
}
```

### Error Metric
```json
{
  "timestamp": 1234567910,
  "event_type": "error",
  "session_id": "uuid-xxxx",
  "error_code": "RL-SESSION-NOT-FOUND",
  "error_message": "Session not found in pool",
  "http_status": 404
}
```

### State Transition Metric (Godot)
```json
{
  "timestamp": 1234567920,
  "event_type": "state_transition",
  "session_id": "uuid-xxxx",
  "round": 5,
  "from_state": "WAITING_FOR_STEP",
  "to_state": "STEPPING"
}
```

## Phase 3 Subtasks

| Subtask ID | Scope | PR 交付物 | Verification Steps | Evidence | Owner | Status |
|---|---|---|---|---|---|---|
| P3-INS-01 | Instrumentation | 在 api/server.py 與 PlayableLoopController.gd 埋點 | 1) Run loop 2) Check logs/metrics_*.jsonl populated | [logs/metrics_sample.jsonl](logs/metrics_sample.jsonl) | Backend Lead | Done |
| P3-MET-01 | Metrics Backend | 創建 analysis/metrics_summary.py + alert rules | 1) Summarize JSONL 2) Generate CSV report | [analysis/metrics_summary.py](analysis/metrics_summary.py) + sample report | Data Analyst | Done |
| P3-ALT-01 | Alert Rules | 定義 2+ alert rules (latency, error_rate) | 1) Manually trigger alert 2) Verify condition eval | [analysis/alerts.yaml](analysis/alerts.yaml) + test log | SRE Lead | Done |
| P3-DAS-01 | Dashboard | HTML/Jupyter報告可視化 metrics | 1) Generate report 2) Verify readable & complete | [analysis/metrics_report.html](analysis/metrics_report.html) | UX Lead | Done |

## Evidence & Acceptance Criteria

### Verification Template

| Scenario | Expected | Actual | Match | Evidence |
|---|---|---|---|---|
| 1 full session | metrics_*.jsonl has ≥5 entries | Yes | Match | [logs/metrics_sample.jsonl](logs/metrics_sample.jsonl) |
| Latency alert | alert fires when latency > 500ms | Tested (manual) | Match | [scripts/check_rl_session_alerts.py](scripts/check_rl_session_alerts.py) |
| Error metric | 404/500 errors logged correctly | Yes | Match | [logs/metrics_sample.jsonl](logs/metrics_sample.jsonl) |
| Report generation | HTML/CSV readable & complete | Yes | Match | [analysis/metrics_summary.csv](analysis/metrics_summary.csv) |

### Pass/Fail Criteria
- ✅ **Pass**: All 4 subtasks Done; metrics collected for 1 full loop; 1 alert rule tested; report generated
- ❌ **Fail**: Metrics missing; alert rule non-functional; report unreadable

## Signoff

- **Status**: Phase 3 — Completed
- **Completed Date**: 2026-05-31
- **Signoff By**: Team (instrumentation & QA) — automated smoke tests and manual checks


## Cross-References

- Upstream: [phase_gate_checklist.md](../04_phase_review/phase_gate_checklist.md) Phase 3 acceptance
- Upstream: [playable_loop_plan.md](./playable_loop_plan.md) (Phase 2 prerequisite)
- Downstream (Phase 4): demo_harden_plan.md (TBD)
- Downstream (Phase 5): adaptive_fate_validation.md (TBD)

## Revision History

| Date | Author | Change |
|---|---|---|
| 2026-05-28 | Copilot | Initial Phase 3 instrumentation plan |
| 2026-05-28 | Copilot | Added metrics schema, alert rules, implementation checklist |

## Future Expansion

- 集成 Prometheus 指標導出（長期）
- 自動化告警到 Slack/Email
- 機器學習異常檢測
- 多租戶指標隔離
