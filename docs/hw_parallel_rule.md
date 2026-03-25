# 硬體並行最佳化規則（i9-9900KF + 3080 Ti + 32 GB）

> **適用對象**：所有 simulation 腳本設計者
> **位置**：`docs/hw_parallel_rule.md`
> **更新**：每次硬體或模擬記憶體估算有重大變動時同步修改

---

## 1. 硬體規格速查

| 資源 | 規格 | 對模擬的意義 |
|------|------|-------------|
| CPU  | Intel i9-9900KF — **8 物理核 / 16 邏輯執行緒** (HT) | Python CPU-bound 每 process 占 1 core；**最多 8 parallel processes** |
| RAM  | 32 GB DDR4 | 每個 simulation job 約 200–600 MB → 最多 ~50 個 job 同時跑不超記憶體 |
| GPU  | RTX 3080 Ti 12 GB VRAM | 目前模擬為純 Python/NumPy，不使用 GPU；未來 CuPy / PyTorch 港口後可加速 loop |
| 磁碟 | （依實際 SSD 速度）| CSV 寫入約 5–20 MB/job；I/O 不是瓶頸 |

---

## 2. 並行數（J）計算公式

```
J_cpu = N_PHYSICAL_CORES          # 8（i9-9900KF 的物理核數）
J_ram = floor(RAM_GB / MEM_PER_JOB_GB)
J     = min(J_cpu, J_ram)
```

### 本機預設值

| 每個 job 估計記憶體 | 建議 J |
|--------------------|--------|
| ≤ 500 MB | **8**（CPU 瓶頸） |
| 500 MB – 2 GB | **min(8, floor(32/MEM))** |
| > 4 GB | 先量測，再決定 |

> **Hyper-Threading 注意**：HT 在 CPU-bound Python 任務上增益有限（共用 L1/L2 與執行埠）。
> 若任務有大量 NumPy/Cython（GIL 釋放），可試 J=12 觀察吞吐量是否提升。

---

## 3. Bash 腳本並行樣板

### 3.1 小批（≤ 8 jobs）— 全部同時啟

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-./venv/bin/python}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs logs

J_MAX=8          # i9-9900KF 物理核數；需要時可降低
PIDS=()

# 啟動 job 的 helper
launch() {
  local label="$1" out="$2"; shift 2
  local logfile="logs/_$(basename "$out" .csv).log"
  # nohup: 避免 HUP signal 中斷長跑
  nohup "$PYTHON_BIN" -m simulation.run_simulation "$@" --out "$out" \
    > "$logfile" 2>&1 &
  local pid=$!
  PIDS+=("$pid")
  echo "[LAUNCH] $label  pid=$pid  log=$logfile"
}

# ── 在此列出所有 jobs ──────────────────────────────────
launch "B-light s45" "outputs/blight_seed45.csv" \
  --seed 45 --rounds 8000 --players 300 ...

launch "B-light s48" "outputs/blight_seed48.csv" \
  --seed 48 --rounds 8000 --players 300 ...
# ...（最多 J_MAX 個同時）

# ── 等待全部完成，收集失敗 ──────────────────────────────
FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "⚠️  pid=$pid 失敗" >&2
    FAILED=$((FAILED + 1))
  fi
done
[[ $FAILED -eq 0 ]] || { echo "❌ $FAILED 個 job 失敗，請檢查 logs/"; exit 1; }
echo "🎉 所有 job 完成"
```

### 3.2 大批（> 8 jobs）— 用 GNU Parallel 限速

```bash
# 前提：parallel 已安裝（sudo apt install parallel）
parallel --jobs "$J_MAX" --joblog "logs/parallel.log" \
  "$PYTHON_BIN -m simulation.run_simulation \
    --seed {} --rounds 8000 --players 300 \
    --out outputs/run_seed{}.csv \
    ... > logs/_run_seed{}.log 2>&1" \
  ::: 45 46 47 48 49 50 51 52
```

> 若 GNU parallel 未安裝，可改用 xargs：
> ```bash
> printf '%s\n' 45 46 47 48 | xargs -P 8 -I{} bash -c \
>   "$PYTHON_BIN -m simulation.run_simulation --seed {} --out outputs/s{}.csv ..."
> ```

---

## 4. 診斷在並行完成後才跑

```bash
# 全部模擬完成後，再並行診斷（診斷很快，J 可拉高）
diagnose_all() {
  local J_DIAG=8
  for csv in outputs/*.csv; do
    echo "$csv"
  done | xargs -P "$J_DIAG" -I{} \
    "$PYTHON_BIN" find_lv3/scripts/diagnose_cycle.py "{}"
}
```

**不要把診斷插在模擬迴圈中間**（會讓 CPU 空轉等診斷完才能啟下一個 job）。

---

## 5. 系統資源監控命令

```bash
# 即時 CPU / RAM 使用
htop

# 只看 simulation 相關 process
watch -n 2 "ps aux --sort=-%cpu | grep run_simulation | head -20"

# 預測完成時間（jobs 數 × 平均秒數 / J）
# GPU（未來用途）
watch -n 2 nvidia-smi

# 確認每個 job 使用多少記憶體
ps -o pid,rss,comm -p $(pgrep -f run_simulation | tr '\n' ',') | \
  awk 'NR==1{print} NR>1{print $1, $2/1024 " MB", $3}'
```

---

## 6. 長跑（10 000+ rounds）注意事項

1. **一律加 `nohup`**：防止 SSH 斷線或 terminal 關閉中斷 job。
2. **log 導向檔案**：`> logs/xxx.log 2>&1`；避免 stdout 占記憶體。
3. **不要超過 J_MAX**：長跑 job 較耗記憶體，出發前先 `free -h` 確認可用 RAM。

| Rounds | 每 job 估計 RAM | 建議 J |
|--------|----------------|--------|
| 4000   | ~200 MB | 8 |
| 8000   | ~300 MB | 8 |
| 10000  | ~400 MB | 8 |
| 50000+ | 先量測 | ≤ 8 |

---

## 7. GPU 加速（未來路線圖）

目前 `simulation.run_simulation` 為純 Python，不使用 GPU。
若要啟用：

- **NumPy → CuPy 替換**：適用 replicator dynamics / payoff matrix 的 batch 更新。
- **平行種子套跑**：在一張 3080 Ti (12 GB) 上，可一次 batch 數十個種子的矩陣運算（需重構 `evolution/` 內核心迴圈）。
- 在此之前，GPU 閒置；模擬瓶頸在 Python 單執行緒 + GIL。

---

## 8. 快速清單（每次寫新腳本前對照）

- [ ] 設定 `J_MAX=8`（或依公式計算）
- [ ] `PIDS=()` 收集所有背景 PID
- [ ] 每個 job 輸出 log 到 `logs/`（不插入 stdout）
- [ ] 模擬全完成後才跑診斷（`wait` 之後）
- [ ] 用 `wait $pid` 個別收集失敗，統一回報
- [ ] 長跑加 `nohup`
- [ ] 腳本頭部印出 ROOT_DIR / PYTHON_BIN / J_MAX 供除錯
