"""Event JSON 格式整理與實現檢查

本文檔整理 Event 事件系統的 JSON 格式、現有實現與使用方式。
"""

# ============================================================
# 1. EventPayload API Schema（響應契約）
# ============================================================

## 位置
- 實現: [api/schemas.py](../api/schemas.py) 行 314-360
- 契約文件: [docs/core_frontend_api_contract_v1.md](../core_frontend_api_contract_v1.md)

## 結構（Python dataclass）

```python
@dataclass(frozen=True, slots=True)
class EventPayload:
    event_id: str           # 事件唯一識別碼
    event_type: str         # 事件類型（如："Threat", "Resource", "Uncertainty"）
    title: str              # 事件標題
    summary: str            # 事件描述
    tags: tuple[str, ...] = field(default_factory=tuple)        # 標籤列表（可選）
    severity: float = 0.0   # 嚴重程度 [0.0-1.0]
    phase: str | None = None    # 遊戲階段標記（可選）
```

## JSON 序列化例子

```json
{
    "event_id": "threat_shadow_stalker",
    "event_type": "Threat",
    "title": "Shadow Stalker",
    "summary": "A shadow stalker is circling at the edge of torchlight.",
    "tags": ["danger", "perception_check"],
    "severity": 0.72,
    "phase": "combat"
}
```

## 驗證規則

- `event_id`, `event_type`, `title`, `summary`: 必填，非空字串
- `severity`: 浮點數，必須在 [0.0, 1.0] 範圍內
- `tags`: 字串元組，可省略（預設為空元組）
- `phase`: 字串或 None，允許空字串

---

# ============================================================
# 2. Event Template JSON 文件格式
# ============================================================

## 現有文件

| 文件名 | 位置 | 用途 |
|--------|------|------|
| `02_event_templates_v1.json` | [docs/personality_dungeon_v1/](../docs/personality_dungeon_v1/) | 完整 event 模板庫 |
| `02_event_templates_smoke_v1.json` | [docs/personality_dungeon_v1/](../docs/personality_dungeon_v1/) | Smoke 測試用（簡化版） |

## 頂層結構

```json
{
    "version": "v1.0",
    "schema_status": "research_ready_draft",
    "event_types": ["Threat", "Resource", "Uncertainty", "Navigation", "Internal"],
    "dimensions_order": [...personality basis 12D...],
    "utility_weight_policy": {...},
    "risk_policy": {...},
    "success_policy": {...},
    "state_policy": {...},
    "reward_policy": {...},
    "templates": [
        {event_template_1},
        {event_template_2},
        ...
    ]
}
```

## Event Template 詳細結構

```json
{
    "event_id": "threat_shadow_stalker",
    "type": "Threat",
    "description": "A shadow stalker is circling at the edge of torchlight.",
    "actions": [
        {
            "name": "attack",
            "weights": [0.9, -0.7, 0.0, 0.2, ...],  // 12D 個性權重
            "base_risk": 0.72,
            "risk_model": {
                "risk_bias": 0.0,
                "trait_weights": {
                    "impulsiveness": 0.18,
                    "ambition": 0.12,
                    "caution": -0.1
                },
                "clamp": [0.0, 1.0],
                "failure_threshold": 0.82
            },
            "success_model": {
                "probability_formula": "clamp(0, 1, 1 - final_risk)",
                "success_bias": 1.0,
                "success_steepness": 6.0,
                "fallback": "clamp(0, 1, 1 - base_risk)"
            },
            "reward_effects": {
                "utility_delta": 0.26,
                "risk_delta": -0.06,
                "popularity_shift": {
                    "aggressive": 0.05,
                    "defensive": -0.03,
                    "balanced": -0.02
                },
                "trait_deltas": {
                    "ambition": 0.03,
                    "fearfulness": -0.02
                },
                "sample_quality": "rare",
                "state_tags": ["clear_path", "dominance_signal"]
            },
            "state_effects": {
                "on_success": {
                    "stress_delta": -0.08,
                    "noise_delta": -0.02,
                    "risk_drift_delta": -0.01,
                    "health_delta": 0.0,
                    "intel_delta": 0.02
                },
                "on_failure": {
                    "stress_delta": 0.18,
                    "noise_delta": 0.06,
                    "risk_drift_delta": 0.02,
                    "health_delta": -0.12,
                    "intel_delta": -0.01
                }
            },
            "failure_outcomes": [
                {
                    "kind": "counter_hit",
                    "probability": 1.0,
                    "utility_delta": -0.28,
                    "risk_delta": 0.22,
                    "popularity_shift": {...},
                    "trait_deltas": {...},
                    "state_tags": ["exposed", "tempo_loss"]
                }
            ]
        },
        {
            "name": "observe",
            "weights": [...],
            "base_risk": 0.18,
            ...
        },
        ...
    ]
}
```

---

# ============================================================
# 3. 現有實現清單
# ============================================================

## ✅ 已實現部分

### 3.1 API Schema（api/schemas.py）
- ✅ **EventPayload** dataclass（行 314-360）
  - 欄位驗證（from_mapping / to_dict）
  - 類型檢查（event_id, event_type 必填）
  - 範圍檢查（severity [0.0, 1.0]）
  
- ✅ **ResponseEnvelope** 集成
  - 行 613-730：ResponseEnvelope 包含 event 欄位
  - event 欄位是 `EventPayload | None`
  - 在 from_mapping() 中正確解析 event

### 3.2 Event 模板與採樣
- ✅ **EventBridge**（simulation/personality_rl_runtime.py 行 152-192）
  - 讀取 JSON 模板檔
  - sample_event() 隨機採樣事件
  - compute_reward_risk() 計算 personality-aligned reward/risk

- ✅ **Event 模板檔案**
  - 02_event_templates_smoke_v1.json（已有）
  - 02_event_templates_v1.json（已有）
  - 完整的 event_types / dimensions_order / actions 定義

### 3.3 Event 在遊戲中的整合
- ✅ **Event 事件在 simulation 中的應用**
  - simulation/personality_rl_runtime.py：PersonalityRLConfig 支援 events_json 參數
  - event_rate 控制事件觸發機率
  - event_reward_scale / event_modulation_mode 等配置

- ✅ **Event 可追溯性（Provenance）**
  - analysis/event_provenance_summary.py（行 417-600）
  - 從 CSV 提取 event 統計信息
  - 生成 event_type / event_id / action 的彙總

- ✅ **測試框架**
  - tests/test_personality_rl_runtime.py（行 491-550）
  - TestEventBridge 類
  - 測試：加載 → 採樣 → reward_risk 計算 → no NaN

---

# ============================================================
# 4. ResponseEnvelope 中的 Event 字段
# ============================================================

當 API 伺服器返回事件時，格式如下：

```python
# api/server.py 中的端點會返回
response = {
    "api_version": "1.0",
    "kind": "step",
    "ok": True,
    "session_id": "...",
    "tick": 5,
    "state_hash": "...",
    
    # ⬇️ Event 字段（可能為 None 或 EventPayload）
    "event": {
        "event_id": "threat_shadow_stalker",
        "event_type": "Threat",
        "title": "Shadow Stalker",
        "summary": "A shadow stalker is circling...",
        "tags": ["danger", "perception"],
        "severity": 0.72,
        "phase": "combat"
    },
    
    # 其他字段...
    "choices": [...],
    "emotion": {...},
    "world_state": {...},
    "player_state": {...},
    "result": {...}
}
```

---

# ============================================================
# 5. 如何在 Server 中返回 Event
# ============================================================

在 [api/server.py](../api/server.py) 中，若要在 step 響應中包含事件：

```python
from api.schemas import EventPayload, normalize_response_envelope

# 建立 event 物件
event_payload = EventPayload(
    event_id="threat_shadow_stalker",
    event_type="Threat",
    title="Shadow Stalker",
    summary="A shadow stalker is circling...",
    tags=("danger", "perception"),
    severity=0.72,
    phase="combat"
)

# 在響應中包含
envelope_data = {
    "api_version": "1.0",
    "kind": "step",
    "ok": True,
    "session_id": session_id,
    "tick": session.tick,
    "state_hash": session.compute_state_hash(),
    "event": event_payload.to_dict(),  # ⬅️ 包含事件
    "world_state": session.world_state,
    "result": result.to_dict(),
}

envelope = normalize_response_envelope(envelope_data)
return envelope.to_dict()
```

---

# ============================================================
# 6. 使用 EventBridge 來採樣事件（Simulation）
# ============================================================

```python
from simulation.personality_rl_runtime import EventBridge
from pathlib import Path
import random

# 加載 event 模板
events_json = Path("docs/personality_dungeon_v1/02_event_templates_smoke_v1.json")
bridge = EventBridge(str(events_json))

# 隨機採樣事件
rng = random.Random(42)
event_dict = bridge.sample_event(rng=rng)
# event_dict: {"event_id": "...", "type": "...", "description": "...", "actions": [...]}

# 計算 personality-aligned reward/risk
personality = {"impulsiveness": 0.5, "caution": -0.3, ...}  # 12D 向量
reward_modifier, risk_modifier = bridge.compute_reward_risk(
    event=event_dict,
    personality=personality,
    scale=0.01,
    rng=rng
)
```

---

# ============================================================
# 7. Event 在測試中的使用
# ============================================================

見 [tests/test_personality_rl_runtime.py](../tests/test_personality_rl_runtime.py) 行 491-550：

```python
@pytest.mark.skipif(not EVENT_JSON.exists(), reason="event templates JSON not found")
def test_load_and_sample(self):
    bridge = EventBridge(EVENT_JSON)
    assert len(bridge.events) >= 1
    rng = random.Random(7)
    ev = bridge.sample_event(rng=rng)
    assert "type" in ev or "event_type" in ev

@pytest.mark.skipif(not EVENT_JSON.exists(), reason="event templates JSON not found")
def test_reward_risk_not_nan(self):
    bridge = EventBridge(EVENT_JSON)
    rng = random.Random(7)
    ev = bridge.sample_event(rng=rng)
    pers = sample_personality(rng=rng)
    rm, rk = bridge.compute_reward_risk(ev, pers, scale=0.01, rng=rng)
    assert rm == rm  # not NaN
    assert rk == rk
```

---

# ============================================================
# 8. 完整檢查表
# ============================================================

| 功能 | 狀態 | 位置 |
|------|------|------|
| ✅ EventPayload dataclass | 實現完成 | api/schemas.py:314-360 |
| ✅ Event 驗證 (from_mapping/to_dict) | 實現完成 | api/schemas.py:332-357 |
| ✅ EventBridge 採樣器 | 實現完成 | simulation/personality_rl_runtime.py:152-192 |
| ✅ Event Template JSON（v1） | 已有檔案 | docs/personality_dungeon_v1/02_event_templates_v1.json |
| ✅ Event Template JSON（smoke） | 已有檔案 | docs/personality_dungeon_v1/02_event_templates_smoke_v1.json |
| ✅ ResponseEnvelope 集成 | 實現完成 | api/schemas.py:613-730 |
| ✅ Event 可追溯性分析 | 實現完成 | analysis/event_provenance_summary.py:417-600 |
| ✅ Event 測試框架 | 實現完成 | tests/test_personality_rl_runtime.py:491-550 |
| ⏳ Server 端點集成 | 部分實現 | api/server.py（需手動添加事件邏輯） |

---

# ============================================================
# 9. 下一步建議
# ============================================================

若要在 API server 中完整支援事件：

1. **Server 端點整合**
   - 在 `/sessions/{session_id}/step` 中使用 EventBridge
   - 根據 event_rate 決定是否觸發事件
   - 在 ResponseEnvelope 中填充 event 欄位

2. **Session 狀態追蹤**
   - 在 GameSession 中添加事件隊列
   - 追蹤正在進行的事件及其出現

3. **Client 端 UI 組件**
   - 解析 ResponseEnvelope 中的 event 物件
   - 顯示事件標題、描述、嚴重程度
   - 渲染可用的 choices（event actions）

---

## 參考檔案

- Schema 定義：[api/schemas.py](../api/schemas.py)
- Server 實作：[api/server.py](../api/server.py)
- Event 模板：[docs/personality_dungeon_v1/](../docs/personality_dungeon_v1/)
- Simulation 整合：[simulation/personality_rl_runtime.py](../simulation/personality_rl_runtime.py)
- 測試：[tests/test_personality_rl_runtime.py](../tests/test_personality_rl_runtime.py)
- 可追溯性：[analysis/event_provenance_summary.py](../analysis/event_provenance_summary.py)
