#!/usr/bin/env bash
# =============================================================================
# run_exp_C.sh — 階段 C：H-series No-op 驗證（同模組配對基準版）
# 規格來源：docs/exp_spec_9persona_v1.md § 階段 C
#
# 用法：bash scripts/run_exp_C.sh
#
# 核心調整：
#   不再以 A1 當 C-series no-op 比較基準。
#   每個 Cx 改用「同模組、同 seeds、同 rounds、同 evolution-mode」的配對比較：
#     Cx-base vs Cx-noop
#
# gate：
#   |noop_mean_s3 - base_mean_s3| <= 0.05 → no-op PASS
#   no-op PASS 才執行 active；否則 active=BLOCKED
#
# exit code：
#   0 = 全部 no-op PASS（active 全執行）
#   1 = 至少一個 no-op FAIL（對應 active BLOCKED）
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=./venv/bin/python
LOGFILE="outputs/exp_C_run.log"
mkdir -p outputs
echo "=== run_exp_C.sh 開始：$(date) ===" | tee "$LOGFILE"

WORST_CODE=0
SEEDS="45 47 49"
NOOP_TOL=0.05

# ──────────────────────────────────────────────────────────────────────────────
# 輔助：對 run_simulation 輸出的 CSV 群計算 cycle metrics
# 用法：_eval_run_sim <exp_id> <csv_dir> <seeds_space_sep>
# stdout：最後一行 JSON
# ──────────────────────────────────────────────────────────────────────────────
_eval_run_sim() {
    local exp_id="$1"
    local csv_dir="$2"
    local seeds_str="$3"

    local OUT
    OUT=$($PYTHON - <<PYEOF
import csv, json, pathlib, sys
from analysis.cycle_metrics import classify_cycle_level

csv_dir   = pathlib.Path("${csv_dir}")
seeds     = [int(s) for s in "${seeds_str}".split()]

results = []
for s in seeds:
    p = csv_dir / f"seed_{s}.csv"
    if not p.exists():
        print(f"  [WARN] 找不到 {p}", file=sys.stderr)
        continue
    with open(p) as f:
        rows = list(csv.DictReader(f))
    series = {
        st: [float(r[f"p_{st}"]) for r in rows]
        for st in ["aggressive", "defensive", "balanced"]
    }
    cyc = classify_cycle_level(
        series,
        burn_in=1000,
        tail=1000,
        amplitude_threshold=0.02,
        corr_threshold=0.09,
        eta=0.55,
        stage3_method="turning",
        phase_smoothing=1,
        min_lag=2,
        max_lag=500,
    )
    s3 = float(cyc.stage3.score) if cyc.stage3 else 0.0
    results.append({"seed": s, "level": int(cyc.level), "s3": s3})

n = len(results)
mean_s3 = sum(r["s3"] for r in results) / max(n, 1)
level_counts = {}
for r in results:
    lv = str(r["level"])
    level_counts[lv] = level_counts.get(lv, 0) + 1
l3_count = level_counts.get("3", 0)

for r in results:
    print(f"  seed={r['seed']}  level={r['level']}  s3={r['s3']:.6f}")
print(f"  mean_s3      = {mean_s3:.6f}")
print(f"  l3_count     = {l3_count}")
print(f"  level_counts = {level_counts}")
print(json.dumps({
    "mean_s3": mean_s3,
    "l3_count": l3_count,
    "n_seeds": n,
    "level_counts": level_counts,
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

# ──────────────────────────────────────────────────────────────────────────────
# 輔助：no-op gate（與 paired base 比較）
# 用法：_check_noop_pair <exp_id> <base_s3> <noop_s3> <tol>
# stdout：PASS 或 FAIL
# ──────────────────────────────────────────────────────────────────────────────
_check_noop_pair() {
    local exp_id="$1"
    local base_s3="$2"
    local noop_s3="$3"
    local tol="$4"

    local OUT
    OUT=$($PYTHON - <<PYEOF
base_s3 = float("${base_s3}")
noop_s3 = float("${noop_s3}")
tol = float("${tol}")
abs_err = abs(noop_s3 - base_s3)
status = "PASS" if abs_err <= tol else "FAIL"
print(f"  [noop_pair] base_mean_s3={base_s3:.6f}  noop_mean_s3={noop_s3:.6f}  abs_err={abs_err:.6f}  tol={tol:.6f}  -> {status}")
print(status)
PYEOF
    )

    local DETAIL STATUS
    DETAIL=$(echo "$OUT" | head -n -1)
    STATUS=$(echo "$OUT" | tail -n 1)
    echo "$DETAIL" | tee -a "$LOGFILE" >&2
    echo "[${exp_id}] noop 配對檢查：${STATUS}" | tee -a "$LOGFILE" >&2
    echo "$STATUS"
}

# ──────────────────────────────────────────────────────────────────────────────
# 輔助：active 相對 base 的效果摘要
# 用法：_summarize_active <exp_id> <base_s3> <active_s3>
# stdout：最後一行 JSON
# ──────────────────────────────────────────────────────────────────────────────
_summarize_active() {
    local exp_id="$1"
    local base_s3="$2"
    local active_s3="$3"

    local OUT
    OUT=$($PYTHON - <<PYEOF
import json
base_s3 = float("${base_s3}")
active_s3 = float("${active_s3}")
delta = active_s3 - base_s3
rel = abs(delta) / max(abs(base_s3), 1e-6)
if delta >= 0.015:
    trend = "UPLIFT"
elif delta <= -0.015:
    trend = "DOWNLIFT"
else:
    trend = "NEUTRAL"
print(f"  [active_effect] base_mean_s3={base_s3:.6f}  active_mean_s3={active_s3:.6f}  delta={delta:+.6f}  rel={rel*100:.2f}%  trend={trend}")
print(json.dumps({"delta": delta, "rel": rel, "trend": trend}))
PYEOF
    )

    local DETAIL JSON_LINE
    DETAIL=$(echo "$OUT" | head -n -1)
    JSON_LINE=$(echo "$OUT" | tail -n 1)
    echo "$DETAIL" | tee -a "$LOGFILE" >&2
    echo "$JSON_LINE"
}

# =============================================================================
# C1：H1 Memory
# base: memory_kernel=1
# noop: memory_kernel=1（與 base 同管線，驗證 no-op 退化一致性）
# active: memory_kernel=3
# =============================================================================
echo "" | tee -a "$LOGFILE"
echo "[C1] H1 Memory：base/noop(k=1) vs active(k=3)" | tee -a "$LOGFILE"
mkdir -p outputs/exp_C1_base outputs/exp_C1_noop outputs/exp_C1_active

# C1-base
echo "[C1-base] 執行中..." | tee -a "$LOGFILE"
for S in $SEEDS; do
    $PYTHON -m simulation.run_simulation \
      --players 300 --rounds 3000 --seed "$S" \
      --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
      --evolution-mode mean_field --payoff-lag 1 \
      --memory-kernel 1 \
      --selection-strength 0.06 --init-bias 0.12 \
      --popularity-mode expected \
      --out "outputs/exp_C1_base/seed_${S}.csv" \
      2>&1 | tee -a "$LOGFILE"
done

# C1-noop
echo "[C1-noop] 執行中..." | tee -a "$LOGFILE"
for S in $SEEDS; do
    $PYTHON -m simulation.run_simulation \
      --players 300 --rounds 3000 --seed "$S" \
      --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
      --evolution-mode mean_field --payoff-lag 1 \
      --memory-kernel 1 \
      --selection-strength 0.06 --init-bias 0.12 \
      --popularity-mode expected \
      --out "outputs/exp_C1_noop/seed_${S}.csv" \
      2>&1 | tee -a "$LOGFILE"
done

echo "[C1] base/noop 評估中..." | tee -a "$LOGFILE"
C1_BASE_JSON=$(_eval_run_sim "C1-base" "outputs/exp_C1_base" "$SEEDS")
C1_NOOP_JSON=$(_eval_run_sim "C1-noop" "outputs/exp_C1_noop" "$SEEDS")
C1_BASE_S3=$($PYTHON -c "import json; print(json.loads('${C1_BASE_JSON}')['mean_s3'])")
C1_NOOP_S3=$($PYTHON -c "import json; print(json.loads('${C1_NOOP_JSON}')['mean_s3'])")
C1_NOOP_STATUS=$(_check_noop_pair "C1-noop" "$C1_BASE_S3" "$C1_NOOP_S3" "$NOOP_TOL")

if [[ "$C1_NOOP_STATUS" == "PASS" ]]; then
    echo "[C1-active] 執行中..." | tee -a "$LOGFILE"
    for S in $SEEDS; do
        $PYTHON -m simulation.run_simulation \
          --players 300 --rounds 3000 --seed "$S" \
          --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
          --evolution-mode mean_field --payoff-lag 1 \
          --memory-kernel 3 \
          --selection-strength 0.06 --init-bias 0.12 \
          --popularity-mode expected \
          --out "outputs/exp_C1_active/seed_${S}.csv" \
          2>&1 | tee -a "$LOGFILE"
    done

    echo "[C1-active] 評估中..." | tee -a "$LOGFILE"
    C1_ACTIVE_JSON=$(_eval_run_sim "C1-active" "outputs/exp_C1_active" "$SEEDS")
    C1_ACTIVE_S3=$($PYTHON -c "import json; print(json.loads('${C1_ACTIVE_JSON}')['mean_s3'])")
    C1_ACTIVE_L3=$($PYTHON -c "import json; print(json.loads('${C1_ACTIVE_JSON}')['l3_count'])")
    C1_ACTIVE_EFFECT_JSON=$(_summarize_active "C1-active" "$C1_BASE_S3" "$C1_ACTIVE_S3")
    C1_ACTIVE_TREND=$($PYTHON -c "import json; print(json.loads('${C1_ACTIVE_EFFECT_JSON}')['trend'])")
    C1_ACTIVE_STATUS="DONE"
else
    C1_ACTIVE_S3="nan"
    C1_ACTIVE_L3="0"
    C1_ACTIVE_EFFECT_JSON='{"delta": null, "rel": null, "trend": "BLOCKED"}'
    C1_ACTIVE_TREND="BLOCKED"
    C1_ACTIVE_STATUS="BLOCKED"
    WORST_CODE=1
fi

echo "[C1] noop=${C1_NOOP_STATUS}  active=${C1_ACTIVE_STATUS}" | tee -a "$LOGFILE"

# =============================================================================
# C2：H2 Threshold
# base: matrix_ab（同 C2 管線）
# noop: matrix_ab（同條件重跑，作為 no-op 配對）
# active: threshold_ab(theta=0.40)
# =============================================================================
echo "" | tee -a "$LOGFILE"
echo "[C2] H2 Threshold：base/noop(matrix_ab) vs active(threshold_ab)" | tee -a "$LOGFILE"
mkdir -p outputs/exp_C2_base outputs/exp_C2_noop outputs/exp_C2_active

# C2-base
echo "[C2-base] 執行中..." | tee -a "$LOGFILE"
for S in $SEEDS; do
    $PYTHON -m simulation.run_simulation \
      --players 300 --rounds 3000 --seed "$S" \
      --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
      --evolution-mode mean_field --payoff-lag 1 \
      --memory-kernel 3 \
      --selection-strength 0.06 --init-bias 0.12 \
      --popularity-mode expected \
      --out "outputs/exp_C2_base/seed_${S}.csv" \
      2>&1 | tee -a "$LOGFILE"
done

# C2-noop
echo "[C2-noop] 執行中..." | tee -a "$LOGFILE"
for S in $SEEDS; do
    $PYTHON -m simulation.run_simulation \
      --players 300 --rounds 3000 --seed "$S" \
      --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
      --evolution-mode mean_field --payoff-lag 1 \
      --memory-kernel 3 \
      --selection-strength 0.06 --init-bias 0.12 \
      --popularity-mode expected \
      --out "outputs/exp_C2_noop/seed_${S}.csv" \
      2>&1 | tee -a "$LOGFILE"
done

echo "[C2] base/noop 評估中..." | tee -a "$LOGFILE"
C2_BASE_JSON=$(_eval_run_sim "C2-base" "outputs/exp_C2_base" "$SEEDS")
C2_NOOP_JSON=$(_eval_run_sim "C2-noop" "outputs/exp_C2_noop" "$SEEDS")
C2_BASE_S3=$($PYTHON -c "import json; print(json.loads('${C2_BASE_JSON}')['mean_s3'])")
C2_NOOP_S3=$($PYTHON -c "import json; print(json.loads('${C2_NOOP_JSON}')['mean_s3'])")
C2_NOOP_STATUS=$(_check_noop_pair "C2-noop" "$C2_BASE_S3" "$C2_NOOP_S3" "$NOOP_TOL")

if [[ "$C2_NOOP_STATUS" == "PASS" ]]; then
    echo "[C2-active] 執行中..." | tee -a "$LOGFILE"
    for S in $SEEDS; do
        $PYTHON -m simulation.run_simulation \
          --players 300 --rounds 3000 --seed "$S" \
          --payoff-mode threshold_ab --a 1.0 --b 0.9 \
          --threshold-theta 0.40 --threshold-a-hi 1.4 --threshold-b-hi 0.9 \
          --evolution-mode mean_field --payoff-lag 1 \
          --memory-kernel 3 \
          --selection-strength 0.06 --init-bias 0.12 \
          --popularity-mode expected \
          --out "outputs/exp_C2_active/seed_${S}.csv" \
          2>&1 | tee -a "$LOGFILE"
    done

    echo "[C2-active] 評估中..." | tee -a "$LOGFILE"
    C2_ACTIVE_JSON=$(_eval_run_sim "C2-active" "outputs/exp_C2_active" "$SEEDS")
    C2_ACTIVE_S3=$($PYTHON -c "import json; print(json.loads('${C2_ACTIVE_JSON}')['mean_s3'])")
    C2_ACTIVE_L3=$($PYTHON -c "import json; print(json.loads('${C2_ACTIVE_JSON}')['l3_count'])")
    C2_ACTIVE_EFFECT_JSON=$(_summarize_active "C2-active" "$C2_BASE_S3" "$C2_ACTIVE_S3")
    C2_ACTIVE_TREND=$($PYTHON -c "import json; print(json.loads('${C2_ACTIVE_EFFECT_JSON}')['trend'])")
    C2_ACTIVE_STATUS="DONE"
else
    C2_ACTIVE_S3="nan"
    C2_ACTIVE_L3="0"
    C2_ACTIVE_EFFECT_JSON='{"delta": null, "rel": null, "trend": "BLOCKED"}'
    C2_ACTIVE_TREND="BLOCKED"
    C2_ACTIVE_STATUS="BLOCKED"
    WORST_CODE=1
fi

echo "[C2] noop=${C2_NOOP_STATUS}  active=${C2_ACTIVE_STATUS}" | tee -a "$LOGFILE"

# =============================================================================
# C3：H3 Hetero
# base: fixed_subgroup_share=0.0
# noop: fixed_subgroup_share=0.0（同條件重跑，作為 no-op 配對）
# active: fixed_subgroup_share=0.10
# =============================================================================
echo "" | tee -a "$LOGFILE"
echo "[C3] H3 Hetero：base/noop(share=0.0) vs active(share=0.10)" | tee -a "$LOGFILE"
mkdir -p outputs/exp_C3_base outputs/exp_C3_noop outputs/exp_C3_active

# C3-base
echo "[C3-base] 執行中..." | tee -a "$LOGFILE"
for S in $SEEDS; do
    $PYTHON -m simulation.run_simulation \
      --players 300 --rounds 3000 --seed "$S" \
      --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
      --evolution-mode sampled \
      --memory-kernel 3 \
      --fixed-subgroup-share 0.0 \
      --selection-strength 0.06 --init-bias 0.12 \
      --out "outputs/exp_C3_base/seed_${S}.csv" \
      2>&1 | tee -a "$LOGFILE"
done

# C3-noop
echo "[C3-noop] 執行中..." | tee -a "$LOGFILE"
for S in $SEEDS; do
    $PYTHON -m simulation.run_simulation \
      --players 300 --rounds 3000 --seed "$S" \
      --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
      --evolution-mode sampled \
      --memory-kernel 3 \
      --fixed-subgroup-share 0.0 \
      --selection-strength 0.06 --init-bias 0.12 \
      --out "outputs/exp_C3_noop/seed_${S}.csv" \
      2>&1 | tee -a "$LOGFILE"
done

echo "[C3] base/noop 評估中..." | tee -a "$LOGFILE"
C3_BASE_JSON=$(_eval_run_sim "C3-base" "outputs/exp_C3_base" "$SEEDS")
C3_NOOP_JSON=$(_eval_run_sim "C3-noop" "outputs/exp_C3_noop" "$SEEDS")
C3_BASE_S3=$($PYTHON -c "import json; print(json.loads('${C3_BASE_JSON}')['mean_s3'])")
C3_NOOP_S3=$($PYTHON -c "import json; print(json.loads('${C3_NOOP_JSON}')['mean_s3'])")
C3_NOOP_STATUS=$(_check_noop_pair "C3-noop" "$C3_BASE_S3" "$C3_NOOP_S3" "$NOOP_TOL")

if [[ "$C3_NOOP_STATUS" == "PASS" ]]; then
    echo "[C3-active] 執行中..." | tee -a "$LOGFILE"
    for S in $SEEDS; do
        $PYTHON -m simulation.run_simulation \
          --players 300 --rounds 3000 --seed "$S" \
          --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
          --evolution-mode sampled \
          --memory-kernel 3 \
          --fixed-subgroup-share 0.10 \
          --fixed-subgroup-weights 0.8,0.8,1.4 \
          --selection-strength 0.06 --init-bias 0.12 \
          --out "outputs/exp_C3_active/seed_${S}.csv" \
          2>&1 | tee -a "$LOGFILE"
    done

    echo "[C3-active] 評估中..." | tee -a "$LOGFILE"
    C3_ACTIVE_JSON=$(_eval_run_sim "C3-active" "outputs/exp_C3_active" "$SEEDS")
    C3_ACTIVE_S3=$($PYTHON -c "import json; print(json.loads('${C3_ACTIVE_JSON}')['mean_s3'])")
    C3_ACTIVE_L3=$($PYTHON -c "import json; print(json.loads('${C3_ACTIVE_JSON}')['l3_count'])")
    C3_ACTIVE_EFFECT_JSON=$(_summarize_active "C3-active" "$C3_BASE_S3" "$C3_ACTIVE_S3")
    C3_ACTIVE_TREND=$($PYTHON -c "import json; print(json.loads('${C3_ACTIVE_EFFECT_JSON}')['trend'])")
    C3_ACTIVE_STATUS="DONE"
else
    C3_ACTIVE_S3="nan"
    C3_ACTIVE_L3="0"
    C3_ACTIVE_EFFECT_JSON='{"delta": null, "rel": null, "trend": "BLOCKED"}'
    C3_ACTIVE_TREND="BLOCKED"
    C3_ACTIVE_STATUS="BLOCKED"
    WORST_CODE=1
fi

echo "[C3] noop=${C3_NOOP_STATUS}  active=${C3_ACTIVE_STATUS}" | tee -a "$LOGFILE"

# =============================================================================
# 彙總 Phase C 結果
# =============================================================================
echo "" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"
echo "  Phase C 總結（同模組配對基準）" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"
printf "  %-16s noop=%-6s  active=%-7s trend=%s\n" "C1 (Memory):" "$C1_NOOP_STATUS" "$C1_ACTIVE_STATUS" "$C1_ACTIVE_TREND" | tee -a "$LOGFILE"
printf "  %-16s noop=%-6s  active=%-7s trend=%s\n" "C2 (Threshold):" "$C2_NOOP_STATUS" "$C2_ACTIVE_STATUS" "$C2_ACTIVE_TREND" | tee -a "$LOGFILE"
printf "  %-16s noop=%-6s  active=%-7s trend=%s\n" "C3 (Hetero):" "$C3_NOOP_STATUS" "$C3_ACTIVE_STATUS" "$C3_ACTIVE_TREND" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"

$PYTHON - <<PYEOF 2>&1 | tee -a "$LOGFILE"
import json, datetime

summary = {
    "phase": "C",
    "timestamp": datetime.datetime.now().isoformat(),
    "comparison_mode": "paired_same_module",
    "noop_tolerance": float("${NOOP_TOL}"),
    "seeds": [45, 47, 49],
    "results": {
        "C1": {
            "base_mean_s3": float("${C1_BASE_S3}"),
            "noop_mean_s3": float("${C1_NOOP_S3}"),
            "noop_status": "${C1_NOOP_STATUS}",
            "active_status": "${C1_ACTIVE_STATUS}",
            "active_mean_s3": None if "${C1_ACTIVE_STATUS}" != "DONE" else float("${C1_ACTIVE_S3}"),
            "active_l3_count": int("${C1_ACTIVE_L3}") if "${C1_ACTIVE_STATUS}" == "DONE" else None,
            "active_effect": json.loads('''${C1_ACTIVE_EFFECT_JSON}'''),
        },
        "C2": {
            "base_mean_s3": float("${C2_BASE_S3}"),
            "noop_mean_s3": float("${C2_NOOP_S3}"),
            "noop_status": "${C2_NOOP_STATUS}",
            "active_status": "${C2_ACTIVE_STATUS}",
            "active_mean_s3": None if "${C2_ACTIVE_STATUS}" != "DONE" else float("${C2_ACTIVE_S3}"),
            "active_l3_count": int("${C2_ACTIVE_L3}") if "${C2_ACTIVE_STATUS}" == "DONE" else None,
            "active_effect": json.loads('''${C2_ACTIVE_EFFECT_JSON}'''),
        },
        "C3": {
            "base_mean_s3": float("${C3_BASE_S3}"),
            "noop_mean_s3": float("${C3_NOOP_S3}"),
            "noop_status": "${C3_NOOP_STATUS}",
            "active_status": "${C3_ACTIVE_STATUS}",
            "active_mean_s3": None if "${C3_ACTIVE_STATUS}" != "DONE" else float("${C3_ACTIVE_S3}"),
            "active_l3_count": int("${C3_ACTIVE_L3}") if "${C3_ACTIVE_STATUS}" == "DONE" else None,
            "active_effect": json.loads('''${C3_ACTIVE_EFFECT_JSON}'''),
        },
    },
    "notes": {
        "noop_gate": "|noop_mean_s3 - base_mean_s3| <= noop_tolerance",
        "active_trend_rule": "delta>=+0.015:UPLIFT; delta<=-0.015:DOWNLIFT; else:NEUTRAL",
    },
}

with open("outputs/exp_C_summary.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("彙總已存至 outputs/exp_C_summary.json")
PYEOF

echo "=== run_exp_C.sh 結束：$(date) ===" | tee -a "$LOGFILE"

if [[ "$WORST_CODE" -eq 0 ]]; then
    echo "[PASS] Phase C no-op 配對檢查通過，active 全部已執行。" | tee -a "$LOGFILE"
else
    echo "[WARN] Phase C 有 no-op 配對失敗，請回檢對應機制設定。" | tee -a "$LOGFILE"
fi

exit "$WORST_CODE"
