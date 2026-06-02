# 08_06-02 — 自動產生 signoff artifact

**Owner:** n1166  
**狀態:** 封卷 ✅  
**建立日期:** 2026-06-01

---

## 目標

在 CI regression 成功後，自動執行 `generate_signoff_summary.py`，
產生結構化的 `artifacts/signoff_summary.json`，供簽核/審核流程使用。

## 行為契約

| ID | 規格 |
|---|---|
| C6-05 | signoff_summary.json 必須包含：`phase`, `generated_at`, `matrix_rows`, `count_match`, `reward_range`, `latency_range`, `gate_pass` |
| C6-06 | `gate_pass` 為 boolean；COUNT_MATCH=False 或 matrix_rows < 12 時必須為 false |
| C6-07 | 腳本必須接受 `--out-dir` 與 `--matrix-out` 參數 |

## 產物

- `scripts/experiments/generate_signoff_summary.py`
- `artifacts/signoff_summary.json`（本地執行樣本）

## 執行步驟

- [x] M6-01 建立 `generate_signoff_summary.py`：讀取 matrix output dir，彙整 reward/latency 範圍，輸出 JSON
- [x] M6-02 本地執行，以 phase05_matrix_full 為輸入，產生 `artifacts/signoff_summary.json`
- [x] M6-03 驗證輸出 JSON 符合 C6-05 schema
- [x] M6-04 確認 `gate_pass=true`（12 rows，COUNT_MATCH=True）

## 驗證標準（Gate）

| Check | 標準 |
|---|---|
| M6-01 | 腳本建立，接受參數 | **PASS (2026-06-01)** |
| M6-02 | 本地成功執行，產出 JSON | **PASS (2026-06-01)** |
| M6-03 | JSON 含所有必要欄位 | **PASS (2026-06-01)** |
| M6-04 | gate_pass=true | **PASS (2026-06-01)** |

## 證據索引

- E6-04: `scripts/experiments/generate_signoff_summary.py`
- E6-05: `artifacts/signoff_summary.json`（樣本輸出）

## Estimated Effort

1–3 days（已排入 2026-06-01）
