# 08_06-01 — 設計 regression + smoke CI jobs

**Owner:** n1166  
**狀態:** 封卷 ✅  
**建立日期:** 2026-06-01

---

## 目標

為 Phase 05 實驗管線建立兩條 CI job：
1. `ci-smoke.yml` — PR 觸發，執行 smoke matrix（3 rows），COUNT_MATCH 驗證
2. `ci-regression.yml` — 手動觸發 + nightly，執行 full matrix（12 rows），產出 signoff artifact

## 行為契約（Repro Contract）

| ID | 規格 |
|---|---|
| C6-01 | ci-smoke 必須在 2 分鐘內完成（含 venv 建立）|
| C6-02 | ci-smoke COUNT_MATCH 失敗時整個 job 應以非零 exit code 結束 |
| C6-03 | ci-regression 成功後必須上傳 `artifacts/signoff_summary.json` |
| C6-04 | 兩個 workflow 都使用 `./venv/bin/python`，不依賴系統 python |

## 產物

- `.github/workflows/ci-smoke.yml`
- `.github/workflows/ci-regression.yml`

## 執行步驟

- [x] R6-01 建立 `ci-smoke.yml`：checkout → venv → run_matrix（matrix.csv，3 rows）→ count_match 驗證 → 失敗時 exit 1
- [x] R6-02 建立 `ci-regression.yml`：checkout → venv → run_matrix（matrix_phase05_full.csv，12 rows）→ generate_signoff_summary → upload-artifact
- [x] R6-03 本地 dry-run：`./venv/bin/python scripts/experiments/run_matrix.py ...` 確認兩個指令可正常執行
- [x] R6-04 確認 workflow YAML 語法有效（no parse errors）

## 驗證標準（Gate）

| Check | 標準 |
|---|---|
| R6-01 | ci-smoke.yml 建立，包含 count_match 邏輯 | **PASS (2026-06-01)** |
| R6-02 | ci-regression.yml 建立，包含 upload-artifact step | **PASS (2026-06-01)** |
| R6-03 | 本地 dry-run COUNT_MATCH=True | **PASS (2026-06-01)** |
| R6-04 | YAML 可解析，無語法錯誤 | **PASS (2026-06-01)** |

## 證據索引

- E6-01: `.github/workflows/ci-smoke.yml`
- E6-02: `.github/workflows/ci-regression.yml`
- E6-03: dry-run 執行 log

## Estimated Effort

3–7 days（已排入 2026-06-01）
