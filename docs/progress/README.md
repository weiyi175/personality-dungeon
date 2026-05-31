# Project Governance Docs

## 0. 文件定位

本目錄提供 Personality Dungeon 的長期治理文件系統，目標是同時追蹤：

- 研究完成度
- 產品完成度
- 系統整合度
- Demo readiness
- 技術債風險
- Phase gate 與簽核狀態

## 1. 目錄結構

```text
docs/progress/
├── 00_master_matrix/
│   ├── project_progress_matrix.md
│   ├── system_layer_overview.md
│   ├── module_dictionary.md
│   └── module_dependency_graph.md
├── 01_runtime_bridge/
│   ├── runtime_bridge_matrix.md
│   ├── api_contract_status.md
│   └── godot_python_sync.md
├── 02_demo_review/
│   ├── demo_checklist.md
│   ├── demo_gate.md
│   └── playable_loop_review.md
├── 03_technical_debt/
│   ├── technical_debt_matrix.md
│   ├── mock_tracking.md
│   └── refactor_risk.md
├── 04_phase_review/
│   ├── phase_gate_checklist.md
│   ├── milestone_review.md
│   └── signoff_checklist.md
└── archive/
    └── README.md
```

## 2. 統一命名規則

- 檔名採 lower_snake_case。
- 所有治理文件需包含以下章節：
  - Metadata
  - Cross References
  - Main Matrix or Checklist
  - Revision History
  - Future Expansion

## 3. 文件總覽與依賴

| 文件 | 用途 | 適用對象 | 更新頻率 | 主要依賴 |
|---|---|---|---|---|
| 00_master_matrix/project_progress_matrix.md | 專案全域進度主矩陣 | PM, Tech Lead, Research Lead | Weekly | 其餘全部治理檔 |
| 00_master_matrix/system_layer_overview.md | 系統層狀態總覽 | 架構負責人, 組長 | Weekly | project_progress_matrix |
| 00_master_matrix/module_dictionary.md | 模組字典與責任邊界 | 全體開發 | Bi-weekly | SDD, 系統層文件 |
| 00_master_matrix/module_dependency_graph.md | 模組依賴與耦合圖 | 架構/整合工程師 | Bi-weekly | module_dictionary |
| 01_runtime_bridge/runtime_bridge_matrix.md | Python-API-Godot 串接追蹤 | Backend, Frontend, Integration | 2-3 days | api_contract_status, godot_python_sync |
| 01_runtime_bridge/api_contract_status.md | API 契約成熟度與破壞風險 | Backend, QA | 2-3 days | server routes, schema |
| 01_runtime_bridge/godot_python_sync.md | 欄位同步與漂移風險 | Godot, Backend | 2-3 days | api_contract_status |
| 02_demo_review/demo_checklist.md | Demo 可展示性清單 | PM, Demo Owner | Before each demo | demo_gate |
| 02_demo_review/demo_gate.md | Demo Gate 準入規則 | PM, Reviewer | Before each phase review | demo_checklist |
| 02_demo_review/playable_loop_review.md | 實際可玩流程審查 | UX, QA, Gameplay | Weekly | demo_checklist |
| 03_technical_debt/technical_debt_matrix.md | 技術債總帳 | Tech Lead, Maintainer | Weekly | mock_tracking, refactor_risk |
| 03_technical_debt/mock_tracking.md | Mock/placeholder 專項追蹤 | Integration, Backend | Weekly | runtime_bridge_matrix |
| 03_technical_debt/refactor_risk.md | 重構風險評估 | Tech Lead | Weekly | technical_debt_matrix |
| 04_phase_review/phase_gate_checklist.md | Phase Gate 達標清單 | PM, Reviewer | At gate | all summary docs |
| 04_phase_review/milestone_review.md | 里程碑回顧報告模板 | PM, Research Lead | At milestone | gate + matrix |
| 04_phase_review/signoff_checklist.md | 正式簽核文件 | Product Owner, Leads | At signoff | phase_gate_checklist |

## 4. 統一狀態字典

| 狀態 | 定義 |
|---|---|
| Concept | 僅有理念，尚未定義可執行規格 |
| Designed | 規格已定義，尚未實作 |
| Prototype | 已有實驗版或雛形，可局部驗證 |
| Integrated | 已接入主流程，但未達產品穩定 |
| Playable | 使用者可完整操作核心流程 |
| Stable | 功能穩定且具回歸保證 |
| Deprecated | 已淘汰或替代，僅保留歷史參考 |

## 5. 統一進度維度

| 維度 | 說明 | 量化範圍 |
|---|---|---|
| Research Progress | 理論、模型、實驗證據成熟度 | 0-100 |
| Product Progress | 玩家可用性與功能可交付程度 | 0-100 |
| Integration Progress | 跨層串接完整度與一致性 | 0-100 |
| Demo Ready | 當前可展示性 | Yes/Partial/No |
| Technical Debt Risk | 技術債風險等級 | Low/Medium/High/Critical |

## 6. 維護流程建議

1. 先更新 Runtime 與 Debt，再更新 Master Matrix。
2. 任何 Phase Review 前，必須同步更新 Demo Gate 與 Signoff。
3. 禁止僅改百分比不附驗證證據。

## 7. Baseline (2026-05-27)

| 類型 | 當前評估 |
|---|---|
| Research Progress | 高 (約 85) |
| Product Progress | 中低 (約 45) |
| Integration Progress | 中 (約 52) |
| Demo Ready | Partial |
| Technical Debt Risk | High |

快速入口：

- 全域總覽：[00_master_matrix/project_progress_matrix.md](00_master_matrix/project_progress_matrix.md)
- 串接瓶頸：[01_runtime_bridge/runtime_bridge_matrix.md](01_runtime_bridge/runtime_bridge_matrix.md)
- 展示準備：[02_demo_review/demo_gate.md](02_demo_review/demo_gate.md)
- 債務阻塞：[03_technical_debt/technical_debt_matrix.md](03_technical_debt/technical_debt_matrix.md)
- Gate 簽核：[04_phase_review/signoff_checklist.md](04_phase_review/signoff_checklist.md)

## Revision History

| Date | Author | Change |
|---|---|---|
| 2026-05-27 | Copilot | Initial governance doc system scaffold |
| 2026-05-27 | Copilot | Added baseline summary and quick-entry links |

## Future Expansion

- 新增自動化匯總腳本，從各矩陣生成 weekly status digest。
- 新增 machine-readable 匯出格式 (YAML/JSON) 供 dashboard 使用。
