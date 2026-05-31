# Module Dictionary

## Metadata

| Field | Value |
|---|---|
| Document ID | PGM-02 |
| Owner | Architecture Lead |
| Audience | 全體開發與評審 |
| Update Frequency | Bi-weekly |
| Dependency | system_layer_overview.md |

## Cross References

- ./system_layer_overview.md
- ./module_dependency_graph.md
- ../01_runtime_bridge/api_contract_status.md

## 欄位字典（治理層）

| 欄位 | 說明 |
|---|---|
| Module Name | 模組名稱 |
| Layer | 所屬系統層 |
| Responsibility | 單一責任定義 |
| Inputs | 主要輸入 |
| Outputs | 主要輸出 |
| Depends On | 上游依賴 |
| Used By | 下游使用者 |
| Status | Concept/Designed/Prototype/Integrated/Playable/Stable/Deprecated |
| Owner | 主責人 |
| Backup Owner | 備援負責人 |

## 模組清單（Baseline 2026-05-27）

| Module Name | Layer | Responsibility | Inputs | Outputs | Depends On | Used By | Status | Owner | Backup Owner |
|---|---|---|---|---|---|---|---|---|---|
| LastWill Input | L1 | 收集玩家遺言文本 | UI text | canonical testament payload | UI form | Personality infer | Integrated | Gameplay Engineer | UI Engineer |
| Personality Infer | L2 | 將文本映射人格向量 | testament payload | 9D vector | LastWill Input, infer model | Player state, UI radar | Integrated | Backend Lead | Research Engineer |
| Event Loader | L3 | 生成候選事件集 | state snapshot | event candidates | dungeon schema | AI policy, risk engine | Prototype | Simulation Engineer | Backend Lead |
| RL Session Engine | L4 | 控制 session init/step | init/step payload | next state, rewards | Player/Event modules | API routes, Godot bridge | Integrated | Backend Lead | Integration Lead |
| Runtime API Bridge | L4 | 將 session 對外暴露為可用契約 | session state | JSON response | RL Session Engine | Godot client | Prototype | Integration Lead | Backend Lead |
| Demo Orchestrator | L5 | 串接展示流程 | scenario script | guided demo loop | Runtime bridge | Demo review | Designed | PM + UX | QA Lead |

## Baseline Evidence

| Module Name | Evidence |
|---|---|
| LastWill Input | [DebugPanel input flow](../../src/ui/DebugPanel.gd#L24) |
| Personality Infer | [infer endpoint](../../api/server.py#L782) |
| Event Loader | [event loader implementation](../../dungeon/event_loader.py) |
| RL Session Engine | [rl session engine](../../simulation/rl_session_engine.py) |
| Runtime API Bridge | [session initialize route](../../api/server.py#L389) |
| Demo Orchestrator | [demo checklist](../02_demo_review/demo_checklist.md) |

## 責任邊界原則

1. 一個模組僅負責一種核心能力。
2. UI 模組不承載核心演算。
3. Runtime Bridge 不應改寫核心規則，只做契約轉換與傳遞。

## Revision History

| Date | Author | Change |
|---|---|---|
| 2026-05-27 | Copilot | Initial module dictionary and boundary rules |
| 2026-05-27 | Copilot | Added baseline owners and module evidence links |

## Future Expansion

- 加入 module maturity index (MMI) 自動計算欄位。
- 加入模組生命週期遷移歷程。
