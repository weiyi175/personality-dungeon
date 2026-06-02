# 07_05-02 — 執行實驗矩陣並收集結果

## Metadata

| Field | Value |
|---|---|
| Owner | Data Analyst（待填） |
| Target date | 2026-06-05 |
| Scope | 批次執行 matrix 並輸出可分析彙總 |
| Dependency | scripts/experiments/matrix.csv、scripts/experiments/run_matrix.py |

## 實驗矩陣定義

目前矩陣欄位契約（CSV header）：

- seed
- w
- burn_in
- n_rounds

建議分兩層執行：

| Layer | 用途 | 建議規模 |
|---|---|---|
| smoke | 檢查流程可跑通 | 3 rows |
| full | 正式收斂觀察 | 12-30 rows |

## 基準執行命令

```bash
mkdir -p reports/experiments analysis
./venv/bin/python scripts/experiments/run_matrix.py \
  --matrix scripts/experiments/matrix.csv \
  --out reports/experiments \
  --log reports/experiments/matrix_run_log.txt
```

## 匯總 CSV 產生

執行後將所有 run JSON 匯總成 analysis/experiments_summary.csv：

```bash
./venv/bin/python - <<'PY'
import csv
import json
from pathlib import Path

out_csv = Path('analysis/experiments_summary.csv')
rows = []
for p in sorted(Path('reports/experiments').glob('run_*.json')):
	d = json.loads(p.read_text(encoding='utf-8'))
	s = d.get('summary', {})
	rows.append({
		'file': p.name,
		'seed': d.get('seed'),
		'n_runs': d.get('n_runs'),
		'n_steps': d.get('n_steps'),
		'reward_mean': s.get('reward_mean'),
		'reward_std': s.get('reward_std'),
		'avg_latency_ms': s.get('avg_latency_ms'),
	})

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open('w', encoding='utf-8', newline='') as f:
	w = csv.DictWriter(f, fieldnames=[
		'file', 'seed', 'n_runs', 'n_steps',
		'reward_mean', 'reward_std', 'avg_latency_ms'
	])
	w.writeheader()
	w.writerows(rows)

print(f'Wrote {out_csv} rows={len(rows)}')
PY
```

## 驗證清單

| Check ID | 檢查項目 | Pass 條件 | Status | Evidence |
|---|---|---|---|---|
| M5-01 | matrix 全列執行完成 | log 含「wrote N runs」且 N=CSV 列數 | PASS (2026-06-01) | reports/experiments/phase05_matrix_smoke/matrix_run_log.txt |
| M5-02 | run JSON 完整落地 | run_*.json 檔案數 = CSV 列數 | PASS (2026-06-01) | reports/experiments/phase05_matrix_smoke/ |
| M5-03 | summary CSV 產生成功 | analysis/experiments_summary.csv 存在且列數>0 | PASS (2026-06-01) | analysis/experiments_summary.csv |
| M5-04 | 欄位契約一致 | CSV 欄位符合規格 | PASS (2026-06-01) | analysis/experiments_summary.csv（header 驗證） |

## 本次執行備註（2026-06-01）

- 為避免既有 run JSON 汙染計數，本次 smoke 使用隔離輸出目錄：reports/experiments/phase05_matrix_smoke/。
- 計數結果：matrix_rows=3、run_json_count=3、summary_rows=3、count_match=True。
- summary 目前對應 smoke 矩陣（3 rows），可直接作為 Phase 05 Gate 證據。

## Full Matrix 推進結果（2026-06-01）

- 已執行 full matrix（12 rows）：scripts/experiments/matrix_phase05_full.csv。
- 輸出目錄：reports/experiments/phase05_matrix_full/（隔離執行，避免覆蓋 smoke 證據）。
- full summary：analysis/experiments_summary_full.csv。
- 計數結果：matrix_rows=12、run_json_count=12、summary_rows=12、count_match=True。
- 指標範圍：reward_mean ∈ [-0.164985, 1.046194]，avg_latency_ms ∈ [0.5964, 2.9459]。

## 追蹤指標（Research / Product / Integration）

| Dimension | 指標 | 門檻 |
|---|---|---|
| Research | reward_mean 跨 seed 分布可解釋 | 需有摘要與異常註記 |
| Product | 執行時間可接受 | smoke < 5 分鐘 |
| Integration | 輸出可被後續報告讀取 | summary CSV 欄位固定 |

## 預估工時

- 2-5 days（依矩陣大小）
