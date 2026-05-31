# Demo Checklist

## Metadata

| Field | Value |
|---|---|
| Document ID | DM-00 |
| Owner | Demo Owner |
| Audience | PM, QA, Presenter |
| Update Frequency | Before each demo |
| Dependency | demo_gate.md |

## Cross References

- ./demo_gate.md
- ./playable_loop_review.md
- ../01_runtime_bridge/runtime_bridge_matrix.md

## 玩家導向驗收清單（Acceptance Checklist）

每個驗收項目請填寫：Goal、Pass/Fail 判準、Required Evidence（至少 3 項）、Verification Steps、Owner、Signoff。

示例 Table（請以此為模板填寫每項實驗/演示）

| Item ID | Goal | Pass/Fail 判準 | Required Evidence | Verification Steps | Owner | Signoff |
|---|---|---|---|---|---|---|
| DM-C01 | 玩家理解人格如何影響結果 | UI 顯示人格向量、至少 3 個案例對照說明 | 1) UI 截圖 2) 3 組 Run logs 3) PR with mapping | 1) run infer -> 2) start session -> 3) record UI | UX Lead | [ ] |
| DM-C02 | 玩家能完成 5 回合以上 | 5 次連續回合無例外 | 1) smoke logs 2) video 3) step API logs | run smoke script x5 | QA Lead | [ ] |

## Baseline Score (2026-05-27)

| 指標 | 分數 |
|---|---:|
| 可展示性 | 48 |
| 可理解性 | 52 |
| 可一輪完結 | 35 |

## Demo 風險追蹤

| Risk ID | 風險描述 | Severity | Mitigation | Owner | Due |
|---|---|---|---|---|---|
| DM-R01 | runtime step 中斷 | High | fallback scenario + canned run | Integration Lead | 2026-06-12 |
| DM-R02 | 玩家無法理解結局判定 | High | 補結局解釋卡與教學提示 | UX Lead | 2026-06-14 |

## Verification Template（檔案附帶範例命令）

```bash
# 在專案 venv 中執行 smoke 測試（範例）
./venv/bin/python -m scripts/run_smoke.py --repeats 5 --out logs/smoke_demo.log
```

驗證時請附上 logs、影片截圖、PR 與任何測試指令。

## Revision History

| Date | Author | Change |
|---|---|---|
| 2026-05-27 | Copilot | Initial demo checklist template |
| 2026-05-27 | Copilot | Added baseline demo readiness assessment |

## Future Expansion

- 新增觀眾回饋欄位與量化分數。
