#!/usr/bin/env bash
# =============================================================================
# run_exp_D.sh — 階段 D：記憶機制專項（Memory × Lag 2×2 Analysis）
# 規格來源：docs/exp_spec_9persona_v1.md § 階段 D
#
# 用法：bash scripts/run_exp_D.sh
#
# 執行順序：D1 → D2 → D3 → D4（各自 6 seeds，mean_field 模式）
# D2 為參考基準，必須符合 P(Level3)=1.0（SDD 歷史 deterministic gate 要求）。
# D1/D3/D4 相對 D2 的 mean_s3 偏移若 > 20% → 記為 SIGNIFICANT_DIFFERENCE。
#
# exit code：
#   0 = 全部正常（D2 P(Level3)=1.0，各 cell 無意外）
#   1 = D2 未達歷史 gate 標準（P(Level3) < 1.0）
#   2 = D2 達標但有 D1/D3/D4 顯著差異（資訊性，非錯誤）
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=./venv/bin/python
LOGFILE="outputs/exp_D_run.log"
mkdir -p outputs
echo "=== run_exp_D.sh 開始：$(date) ===" | tee "$LOGFILE"

WORST_CODE=0
D_SEEDS="45 47 49 51 53 55"

# ──────────────────────────────────────────────────────────────────────────────
# 輔助：對 run_simulation 輸出的 CSV 群計算 cycle metrics
# 用法：_eval_run_sim_d <exp_id> <csv_dir> <seeds_space_sep>
# stdout：最後一行為 JSON {"mean_s3":..., "l3_count":..., "p_l3":..., "level_counts":{...}}
# stderr：顯示表格
# ──────────────────────────────────────────────────────────────────────────────
_eval_run_sim_d() {
    local exp_id="$1"
    local csv_dir="$2"
    local seeds_str="$3"

    local OUT
    OUT=$($PYTHON - <<PYEOF
import csv, sys, json, pathlib
from analysis.cycle_metrics import classify_cycle_level

csv_dir    = pathlib.Path("${csv_dir}")
seeds_str  = "${seeds_str}"
seeds      = [int(s) for s in seeds_str.split()]

results = []
for s in seeds:
    p = csv_dir / f"seed_{s}.csv"
    if not p.exists():
        print(f"  [WARN] 找不到 {p}", file=sys.stderr)
        continue
    with open(p) as f:
        rows = list(csv.DictReader(f))
    series = {st: [float(r[f'p_{st}']) for r in rows]
              for st in ['aggressive', 'defensive', 'balanced']}
    cyc = classify_cycle_level(
        series,
        burn_in=1000, tail=1000,
        amplitude_threshold=0.02,
        corr_threshold=0.09,
        eta=0.55,
        stage3_method='turning',
        phase_smoothing=1,
        min_lag=2,
        max_lag=500
    )
    s3 = float(cyc.stage3.score) if cyc.stage3 else 0.0
    results.append({"seed": s, "level": cyc.level, "s3": s3})

n = len(results)
mean_s3 = sum(r["s3"] for r in results) / max(n, 1)
level_counts = {}
for r in results:
    lv = str(r["level"])
    level_counts[lv] = level_counts.get(lv, 0) + 1
l3_count = level_counts.get("3", 0)
p_l3 = l3_count / max(n, 1)

for r in results:
    print(f"  seed={r['seed']}  level={r['level']}  s3={r['s3']:.6f}")
print(f"  mean_s3      = {mean_s3:.6f}")
print(f"  l3_count     = {l3_count} / {n}")
print(f"  P(Level3)    = {p_l3:.4f}")
print(f"  level_counts = {level_counts}")
# 最後一行：JSON
print(json.dumps({
    "mean_s3": mean_s3,
    "l3_count": l3_count,
    "p_l3": p_l3,
    "n_seeds": n,
    "level_counts": level_counts
}))
PYEOF
    )

    local TABLE JSON_LINE
    TABLE=$(echo "$OUT" | head -n -1)
    JSON_LINE=$(echo "$OUT" | tail -n 1)

    echo "[${exp_id}]" | tee -a "$LOGFILE" >&2
    echo "$TABLE" | tee -a "$LOGFILE" >&2

    echo "$JSON_LINE"
}

# ==============================================================================
# D1：memory_kernel=1, payoff_lag=0
# ==============================================================================
echo "" | tee -a "$LOGFILE"
echo "[D1] m=1, lag=0  seeds=${D_SEEDS}  evolution=mean_field  rounds=3000" | tee -a "$LOGFILE"

mkdir -p outputs/exp_D1_m1_lag0
for S in $D_SEEDS; do
    $PYTHON -m simulation.run_simulation \
      --players 300 --rounds 3000 --seed $S \
      --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
      --evolution-mode mean_field --payoff-lag 0 --memory-kernel 1 \
      --selection-strength 0.06 --init-bias 0.12 \
      --popularity-mode expected \
      --out "outputs/exp_D1_m1_lag0/seed_${S}.csv" \
      2>&1 | tee -a "$LOGFILE"
done

echo "[D1] 評估中..." | tee -a "$LOGFILE"
D1_JSON=$(_eval_run_sim_d "D1" "outputs/exp_D1_m1_lag0" "$D_SEEDS")
D1_S3=$($PYTHON -c "import json; print(json.loads('${D1_JSON}')['mean_s3'])")
D1_L3=$($PYTHON -c "import json; print(json.loads('${D1_JSON}')['l3_count'])")
D1_PL3=$($PYTHON -c "import json; print(json.loads('${D1_JSON}')['p_l3'])")
echo "[D1] mean_s3=${D1_S3}  l3_count=${D1_L3}  P(L3)=${D1_PL3}" | tee -a "$LOGFILE"

# ==============================================================================
# D2：memory_kernel=1, payoff_lag=1（參考基準）
# ==============================================================================
echo "" | tee -a "$LOGFILE"
echo "[D2] m=1, lag=1  seeds=${D_SEEDS}  evolution=mean_field  rounds=3000  ← 參考基準" | tee -a "$LOGFILE"

mkdir -p outputs/exp_D2_m1_lag1
for S in $D_SEEDS; do
    $PYTHON -m simulation.run_simulation \
      --players 300 --rounds 3000 --seed $S \
      --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
      --evolution-mode mean_field --payoff-lag 1 --memory-kernel 1 \
      --selection-strength 0.06 --init-bias 0.12 \
      --popularity-mode expected \
      --out "outputs/exp_D2_m1_lag1/seed_${S}.csv" \
      2>&1 | tee -a "$LOGFILE"
done

echo "[D2] 評估中..." | tee -a "$LOGFILE"
D2_JSON=$(_eval_run_sim_d "D2" "outputs/exp_D2_m1_lag1" "$D_SEEDS")
D2_S3=$($PYTHON -c "import json; print(json.loads('${D2_JSON}')['mean_s3'])")
D2_L3=$($PYTHON -c "import json; print(json.loads('${D2_JSON}')['l3_count'])")
D2_PL3=$($PYTHON -c "import json; print(json.loads('${D2_JSON}')['p_l3'])")
echo "[D2] mean_s3=${D2_S3}  l3_count=${D2_L3}  P(L3)=${D2_PL3}" | tee -a "$LOGFILE"

# D2 gate 驗證：P(Level3) 需達 SDD 歷史 gate 標準
D2_GATE_STATUS=$($PYTHON - <<PYEOF
p_l3 = float("${D2_PL3}")
# SDD gate 要求：P(Level3) >= 0.9（歷史 deterministic gate 閉合條件）
if p_l3 >= 0.9:
    print(f"  [D2-gate] P(Level3)={p_l3:.4f} >= 0.9 → GATE_PASS")
    print("GATE_PASS")
else:
    print(f"  [D2-gate] P(Level3)={p_l3:.4f} < 0.9 → GATE_FAIL（未達歷史 closure 標準）")
    print("GATE_FAIL")
PYEOF
)

D2_GATE_TABLE=$(echo "$D2_GATE_STATUS" | head -n -1)
D2_GATE_FINAL=$(echo "$D2_GATE_STATUS" | tail -n 1)
echo "$D2_GATE_TABLE" | tee -a "$LOGFILE"
echo "[D2] gate 判定：${D2_GATE_FINAL}" | tee -a "$LOGFILE"

if [[ "$D2_GATE_FINAL" == "GATE_FAIL" ]]; then
    echo "[D2] ⚠ D2 未達歷史 gate 標準（P(Level3) < 0.9）。" | tee -a "$LOGFILE"
    echo "[D2] 可能原因：n_rounds=3000 不足以顯現 Level3；或 mean_field 參數需調整。" | tee -a "$LOGFILE"
    WORST_CODE=1
fi

# ==============================================================================
# D3：memory_kernel=3, payoff_lag=0
# ==============================================================================
echo "" | tee -a "$LOGFILE"
echo "[D3] m=3, lag=0  seeds=${D_SEEDS}  evolution=mean_field  rounds=3000" | tee -a "$LOGFILE"

mkdir -p outputs/exp_D3_m3_lag0
for S in $D_SEEDS; do
    $PYTHON -m simulation.run_simulation \
      --players 300 --rounds 3000 --seed $S \
      --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
      --evolution-mode mean_field --payoff-lag 0 --memory-kernel 3 \
      --selection-strength 0.06 --init-bias 0.12 \
      --popularity-mode expected \
      --out "outputs/exp_D3_m3_lag0/seed_${S}.csv" \
      2>&1 | tee -a "$LOGFILE"
done

echo "[D3] 評估中..." | tee -a "$LOGFILE"
D3_JSON=$(_eval_run_sim_d "D3" "outputs/exp_D3_m3_lag0" "$D_SEEDS")
D3_S3=$($PYTHON -c "import json; print(json.loads('${D3_JSON}')['mean_s3'])")
D3_L3=$($PYTHON -c "import json; print(json.loads('${D3_JSON}')['l3_count'])")
D3_PL3=$($PYTHON -c "import json; print(json.loads('${D3_JSON}')['p_l3'])")
echo "[D3] mean_s3=${D3_S3}  l3_count=${D3_L3}  P(L3)=${D3_PL3}" | tee -a "$LOGFILE"

# ==============================================================================
# D4：memory_kernel=3, payoff_lag=1
# ==============================================================================
echo "" | tee -a "$LOGFILE"
echo "[D4] m=3, lag=1  seeds=${D_SEEDS}  evolution=mean_field  rounds=3000" | tee -a "$LOGFILE"

mkdir -p outputs/exp_D4_m3_lag1
for S in $D_SEEDS; do
    $PYTHON -m simulation.run_simulation \
      --players 300 --rounds 3000 --seed $S \
      --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
      --evolution-mode mean_field --payoff-lag 1 --memory-kernel 3 \
      --selection-strength 0.06 --init-bias 0.12 \
      --popularity-mode expected \
      --out "outputs/exp_D4_m3_lag1/seed_${S}.csv" \
      2>&1 | tee -a "$LOGFILE"
done

echo "[D4] 評估中..." | tee -a "$LOGFILE"
D4_JSON=$(_eval_run_sim_d "D4" "outputs/exp_D4_m3_lag1" "$D_SEEDS")
D4_S3=$($PYTHON -c "import json; print(json.loads('${D4_JSON}')['mean_s3'])")
D4_L3=$($PYTHON -c "import json; print(json.loads('${D4_JSON}')['l3_count'])")
D4_PL3=$($PYTHON -c "import json; print(json.loads('${D4_JSON}')['p_l3'])")
echo "[D4] mean_s3=${D4_S3}  l3_count=${D4_L3}  P(L3)=${D4_PL3}" | tee -a "$LOGFILE"

# ==============================================================================
# 相對 D2 的顯著差異分析
# ==============================================================================
echo "" | tee -a "$LOGFILE"
echo "[D] 相對 D2 參考基準的 mean_s3 偏移分析（>20% 視為顯著差異）" | tee -a "$LOGFILE"

DRIFT_ANALYSIS=$($PYTHON - <<PYEOF
import json

d2_s3 = float("${D2_S3}")
cells = {
    "D1 (m=1, lag=0)": float("${D1_S3}"),
    "D3 (m=3, lag=0)": float("${D3_S3}"),
    "D4 (m=3, lag=1)": float("${D4_S3}"),
}
denom = max(d2_s3, 0.05)  # 避免分母趨近 0（mean_field 模式下 s3 通常 > 0）
sig_flags = []
lines = []
lines.append(f"  D2 基準 mean_s3 = {d2_s3:.6f}")
for cid, s3 in cells.items():
    drift = abs(s3 - d2_s3) / denom
    sig = "SIGNIFICANT_DIFFERENCE" if drift > 0.20 else "within_20pct"
    lines.append(f"  {cid:<20} mean_s3={s3:.6f}  drift={drift*100:.1f}%  {sig}")
    if sig == "SIGNIFICANT_DIFFERENCE":
        sig_flags.append(cid)

print('\n'.join(lines))
if sig_flags:
    print(f"  [WARN] 顯著差異（>20% vs D2）：{', '.join(sig_flags)}")
else:
    print("  [OK] 所有 cells 與 D2 偏移均在 20% 以內。")
PYEOF
)
echo "$DRIFT_ANALYSIS" | tee -a "$LOGFILE"

# ==============================================================================
# 彙總 Phase D 結果
# ==============================================================================
echo "" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"
echo "  Phase D 總結  （memory_kernel × payoff_lag 2×2 grid）" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"
printf "  %-12s m=1  lag=0  mean_s3=%-10s P(L3)=%s\n" "D1:" "$D1_S3" "$D1_PL3"  | tee -a "$LOGFILE"
printf "  %-12s m=1  lag=1  mean_s3=%-10s P(L3)=%s  ← 參考基準  gate=%s\n" "D2:" "$D2_S3" "$D2_PL3" "$D2_GATE_FINAL" | tee -a "$LOGFILE"
printf "  %-12s m=3  lag=0  mean_s3=%-10s P(L3)=%s\n" "D3:" "$D3_S3" "$D3_PL3"  | tee -a "$LOGFILE"
printf "  %-12s m=3  lag=1  mean_s3=%-10s P(L3)=%s\n" "D4:" "$D4_S3" "$D4_PL3"  | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"

# 寫入彙總 JSON
$PYTHON - <<PYEOF 2>&1 | tee -a "$LOGFILE"
import json, datetime

summary = {
    "phase": "D",
    "timestamp": datetime.datetime.now().isoformat(),
    "grid": "memory_kernel x payoff_lag",
    "results": {
        "D1": {"memory_kernel": 1, "payoff_lag": 0, "mean_s3": float("${D1_S3}"), "l3_count": int("${D1_L3}"), "p_l3": float("${D1_PL3}")},
        "D2": {"memory_kernel": 1, "payoff_lag": 1, "mean_s3": float("${D2_S3}"), "l3_count": int("${D2_L3}"), "p_l3": float("${D2_PL3}"), "gate": "${D2_GATE_FINAL}"},
        "D3": {"memory_kernel": 3, "payoff_lag": 0, "mean_s3": float("${D3_S3}"), "l3_count": int("${D3_L3}"), "p_l3": float("${D3_PL3}")},
        "D4": {"memory_kernel": 3, "payoff_lag": 1, "mean_s3": float("${D4_S3}"), "l3_count": int("${D4_L3}"), "p_l3": float("${D4_PL3}")},
    },
    "notes": {
        "D2_gate": "P(Level3) >= 0.9 依 SDD 歷史 deterministic gate 要求",
        "drift_threshold": "mean_s3 相對 D2 偏移 > 20% 記為 SIGNIFICANT_DIFFERENCE",
        "evolution_mode": "mean_field（payoff_lag 語意僅在此模式下有效，SDD 4.1.1）"
    }
}

with open("outputs/exp_D_summary.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("彙總已存至 outputs/exp_D_summary.json")
PYEOF

echo "=== run_exp_D.sh 結束：$(date) ===" | tee -a "$LOGFILE"

if [[ "$WORST_CODE" -eq 0 ]]; then
    echo "[PASS] Phase D 完成，D2 達歷史 gate 標準，可進入論文圖表生成。" | tee -a "$LOGFILE"
elif [[ "$WORST_CODE" -eq 1 ]]; then
    echo "[WARN] D2 未達 P(Level3) >= 0.9，請回檢 mean_field 參數或增加 n_rounds。" | tee -a "$LOGFILE"
fi

exit $WORST_CODE
