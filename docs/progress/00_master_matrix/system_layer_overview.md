# System Layer Overview

## Metadata

| Field | Value |
|---|---|
| Document ID | PGM-01 |
| Owner | Architecture Lead |
| Audience | Tech Lead, PM, Integration Team |
| Update Frequency | Weekly |
| Dependency | project_progress_matrix.md |

## Cross References

- ../README.md
- ./project_progress_matrix.md
- ./module_dictionary.md
- ./module_dependency_graph.md

## 系統層定義

| Layer ID | 系統層 | 說明 |
|---|---|---|
| L1 | 玩家系統 | 玩家輸入、狀態、回合互動 |
| L2 | Personality Engine | 人格推斷、人格動態 |
| L3 | 地牢系統 | 事件生成、風險演算、結果套用 |
| L4 | Runtime Bridge | Python/API/Godot 資料流與會話橋接 |
| L5 | Demo 與體驗層 | 展示、教學、可理解性 |
| L6 | 治理與驗證層 | 研究驗證、Phase Gate、簽核 |

## Layer Status Board

| Layer | Core Modules | Current Status | Research | Product | Integration | Key Risk | Owner | Target Gate |
|---|---|---|---:|---:|---:|---|---|---|
| L1 玩家系統 | 遺言輸入, 玩家狀態 | Prototype | 80 | 55 | 50 | 回合閉環不足 | Gameplay Engineer | Phase-Playable |
| L2 Personality | 推斷, 迴圈更新 | Integrated | 85 | 65 | 60 | 可解釋呈現不足 | Backend + UX | Phase-Bridge |
| L3 地牢系統 | 事件抽樣, 風險演算 | Prototype | 80 | 45 | 50 | 前端映射未對齊 | Gameplay + Integration | Phase-Playable |
| L4 Runtime Bridge | Session init/step | Prototype | 75 | 40 | 45 | mock flow 混用 | Integration Lead | Phase-Integration |
| L5 Demo 體驗 | Demo Loop, 教學引導 | Designed | 60 | 35 | 30 | 展示流程不連續 | PM + UX | Phase-Demo |
| L6 治理驗證 | Gate, Signoff | Designed | 70 | 50 | 55 | 證據追蹤尚未自動化 | PMO | Phase-Review |

## Baseline Evidence (2026-05-27)

| Layer | Evidence |
|---|---|
| L1 | [DebugPanel client load](../../src/ui/DebugPanel.gd#L24), [PlayerManager signal](../../src/core/PlayerManager.gd#L6) |
| L2 | [infer route](../../api/server.py#L782), [set personality from api](../../src/core/PlayerManager.gd#L35) |
| L3 | [adaptive counter block](../../dungeon/dungeon_ai.py#L17), [event loader](../../dungeon/event_loader.py) |
| L4 | [initialize route](../../api/server.py#L389), [step route](../../api/server.py#L469), [MockDungeon](../../api/server.py#L53) |
| L5 | [Debug UI scene](../../src/ui/DebugUI.tscn#L38) |
| L6 | [phase gate checklist](../04_phase_review/phase_gate_checklist.md) |

## 系統層驗證規範

| Layer | 必備驗證 | 最低門檻 |
|---|---|---|
| L1 | 玩家可從輸入走到至少一個回合結果 | Pass/Fail |
| L2 | 推斷與更新結果可重現 | 固定 seed 一致 |
| L3 | 事件結果與風險演算一致 | schema + replay 一致 |
| L4 | API 與 Godot payload 對齊 | contract drift = 0 阻塞項 |
| L5 | 新玩家可在 5 分鐘內理解核心機制 | Demo review pass |
| L6 | Gate 條件全部過線 | Signoff 完整 |

## Revision History

| Date | Author | Change |
|---|---|---|
| 2026-05-27 | Copilot | Initial layer mapping and status board |
| 2026-05-27 | Copilot | Added owner roles and baseline evidence links |

## Future Expansion

- 增加每層 KPI 與量測腳本連結。
- 新增跨層故障應急流程欄位。
