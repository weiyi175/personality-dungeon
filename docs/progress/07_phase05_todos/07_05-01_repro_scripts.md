# 07_05-01 — 建立可重現實驗腳本（seeded）

## Metadata

| Field | Value |
|---|---|
| Owner | Research Lead（待填） |
| Target date | 2026-06-03 |
| Scope | Phase 5 實驗可重現腳本與輸出契約 |
| Dependency | SDD.md、scripts/experiments/run_experiment.py、scripts/experiments/run_matrix.py |

## 目標

- 建立可重現執行流程：同一組 seed 與參數下，統計結果必須一致。
- 明確定義輸出路徑與欄位契約，避免後續分析腳本對不到資料。
- 形成可直接交給 CI 的最小 smoke 命令集合。

## 產物（Deliverables）

- 可用腳本：scripts/experiments/run_experiment.py（單 seed）、scripts/experiments/run_matrix.py（矩陣批次）。
- 輸出目錄：reports/experiments/（run JSON、matrix log）。
- 匯總檔：analysis/experiments_summary.csv。
- 簽核紀錄：本文件的驗證結果表格。

## 可重現契約（Repro Contract）

- Python 執行一律使用 ./venv/bin/python。
- 單次 run 固定 seed，且 seed 規則不得隱式更動。
- JSON 檔中的 generated_at 屬於時間戳，不納入 deterministic 比對。
- 可重現判準以 runs 與 summary 數值為主，採完全一致（float 字串一致）。

## 執行步驟

1. 建立輸出資料夾。

```bash
mkdir -p reports/experiments analysis
```

2. 執行單 seed 10 runs（基準樣本）。

```bash
./venv/bin/python scripts/experiments/run_experiment.py \
	--seed 42 \
	--runs 10 \
	--n-steps 200 \
	--out reports/experiments/run_seed42_r10.json
```

3. 連續執行兩次同參數，驗證可重現（忽略 generated_at）。

```bash
./venv/bin/python scripts/experiments/run_experiment.py --seed 42 --runs 10 --n-steps 200 --out reports/experiments/repro_a.json
./venv/bin/python scripts/experiments/run_experiment.py --seed 42 --runs 10 --n-steps 200 --out reports/experiments/repro_b.json
./venv/bin/python - <<'PY'
import json
from pathlib import Path

def normalized(p: str):
		d = json.loads(Path(p).read_text(encoding='utf-8'))
		d.pop('generated_at', None)
		return d

a = normalized('reports/experiments/repro_a.json')
b = normalized('reports/experiments/repro_b.json')
print('REPRO_PASS' if a == b else 'REPRO_FAIL')
PY
```

## 驗證清單

| Check ID | 檢查項目 | Pass 條件 | Status | Evidence |
|---|---|---|---|---|
| R5-01 | 單 seed 腳本可執行 | exit code=0 且寫出 JSON | PASS (2026-06-01) | reports/experiments/run_seed42_r10.json |
| R5-02 | 同參數可重現 | REPRO_PASS | PASS (2026-06-01) | reports/experiments/repro_a.json, reports/experiments/repro_b.json |
| R5-03 | JSON 欄位完整 | seed, n_runs, n_steps, runs, summary 都存在 | PASS (2026-06-01) | reports/experiments/repro_a.json |
| R5-04 | venv 環境一致 | 所有命令都由 ./venv/bin/python 啟動 | PASS (2026-06-01) | 2026-06-01 終端執行紀錄 |

## 執行紀錄摘要（2026-06-01）

- Smoke 參數：seed=42、runs=10、n_steps=200。
- Repro 檢查：repro_a 與 repro_b 在排除 generated_at 後完全一致（REPRO_PASS）。
- summary：reward_mean=0.345291、reward_std=0.278632、avg_latency_ms=1.4887。

## 風險與對策

| Risk | 影響 | 對策 |
|---|---|---|
| generated_at 造成檔案位元不同 | 誤判為不可重現 | 比對時排除 generated_at |
| 參數散落於手動命令 | 無法回歸同條件 | 以 matrix.csv 固定條件並版本控管 |
| 輸出路徑混亂 | 分析腳本找不到檔案 | 固定寫入 reports/experiments/ |

## 預估工時

- 3-7 days
