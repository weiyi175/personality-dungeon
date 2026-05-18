#!/usr/bin/env bash
# =============================================================================
# run_exp_A1_A2.sh — 階段 A：A1 基線橋接 + A2 九人格基線
# 規格來源：docs/exp_spec_9persona_v1.md
#
# 用法：bash scripts/run_exp_A1_A2.sh
#
# 執行後自動計算 mean_s3 漂移率並回報 PASS / WARN / HALT 狀態。
# HALT 時以 exit code 1 退出，WARN 以 exit code 2 退出，PASS 以 0 退出。
# =============================================================================
set -euo pipefail

# 無論從哪個目錄呼叫，都切換到專案根目錄
cd "$(dirname "$0")/.."

PYTHON=./venv/bin/python
SEEDS="42,44,45,67,73,90"
N_PLAYERS=300
N_ROUNDS=12000
BURN_IN=4000
TAIL=4000

EVENTS_JSON="docs/personality_dungeon_v1/02_event_templates_smoke_v1.json"

OUT_A1="outputs/exp_A1_baseline_none"
OUT_A2="outputs/exp_A2_baseline_9persona"

LOGFILE="outputs/exp_A1_A2_run.log"
mkdir -p outputs
echo "=== run_exp_A1_A2.sh 開始：$(date) ===" | tee "$LOGFILE"

# ──────────────────────────────────────────────────────────────────────────────
# Step 1：A1 — 既有基線（personality_mode=none）
# ──────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "[A1] 開始執行 personality_mode=none，seeds=${SEEDS}" | tee -a "$LOGFILE"
echo "[A1] 輸出目錄：${OUT_A1}" | tee -a "$LOGFILE"

$PYTHON -m simulation.personality_rl_runtime \
  --seeds "${SEEDS}" \
  --n-players "${N_PLAYERS}" \
  --n-rounds "${N_ROUNDS}" \
  --personality-mode none \
  --event-dispatch-mode async_poisson \
  --event-dispatch-target-rate 0.08 \
  --event-dispatch-fairness-window 2000 \
  --event-dispatch-fairness-tolerance 0.50 \
  --events-json "${EVENTS_JSON}" \
  --world-feedback-mode adaptive_world \
  --lambda-world 0.04 \
  --world-update-interval 200 \
  --out-dir "${OUT_A1}" \
  2>&1 | tee -a "$LOGFILE"

echo "[A1] 執行完成。" | tee -a "$LOGFILE"

# ──────────────────────────────────────────────────────────────────────────────
# Step 2：A1 指標計算
# ──────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "[A1] 計算 cycle metrics..." | tee -a "$LOGFILE"

A1_METRICS_JSON="outputs/exp_A1_metrics.json"

$PYTHON - <<PYEOF 2>&1 | tee -a "$LOGFILE"
import csv, json, pathlib, sys
from analysis.cycle_metrics import classify_cycle_level

OUT_DIR = pathlib.Path("${OUT_A1}")
BURN_IN = ${BURN_IN}
TAIL    = ${TAIL}

results = []
seed_dirs = sorted(OUT_DIR.glob("seed_*"))
if not seed_dirs:
    print("[ERROR] A1: 找不到任何 seed_* 目錄")
    sys.exit(1)

for sdir in seed_dirs:
    csv_path = sdir / "timeseries.csv"
    prov_path = sdir / "provenance.json"
    if not csv_path.exists():
        print(f"[WARN] 缺少 {csv_path}，跳過")
        continue
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    series = {
        "aggressive": [float(r["p_aggressive"]) for r in rows],
        "defensive":  [float(r["p_defensive"])  for r in rows],
        "balanced":   [float(r["p_balanced"])   for r in rows],
    }
    cyc = classify_cycle_level(
        series,
        burn_in=BURN_IN,
        tail=TAIL,
        amplitude_threshold=0.02,
        corr_threshold=0.09,
        eta=0.55,
        stage3_method="turning",
        phase_smoothing=1,
        min_lag=2,
        max_lag=500,
    )
    s3  = float(cyc.stage3.score) if cyc.stage3 else 0.0
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    bhr = prov.get("boundary_hit_rate", 0.0)
    results.append({
        "seed_dir": str(sdir.name),
        "cycle_level": cyc.level,
        "stage3_score": s3,
        "boundary_hit_rate": bhr,
    })
    print(f"  {sdir.name}: level={cyc.level}  s3={s3:.6f}  bhr={bhr:.4f}")

mean_s3 = sum(r["stage3_score"] for r in results) / max(len(results), 1)
mean_bhr = sum(r["boundary_hit_rate"] for r in results) / max(len(results), 1)
level_counts = {}
for r in results:
    lv = str(r["cycle_level"])
    level_counts[lv] = level_counts.get(lv, 0) + 1

summary = {
    "exp_id": "A1",
    "personality_mode": "none",
    "n_seeds": len(results),
    "mean_s3": mean_s3,
    "mean_boundary_hit_rate": mean_bhr,
    "level_counts": level_counts,
    "per_seed": results,
}
pathlib.Path("${A1_METRICS_JSON}").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False)
)
print(f"[A1] mean_s3={mean_s3:.6f}  mean_bhr={mean_bhr:.4f}  level_counts={level_counts}")
print(f"[A1] 指標已存至 ${A1_METRICS_JSON}")
PYEOF

echo "[A1] 指標計算完成。" | tee -a "$LOGFILE"

# ──────────────────────────────────────────────────────────────────────────────
# Step 3：A2 — 九人格基線（personality_mode=random）
# ──────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "[A2] 開始執行 personality_mode=random，seeds=${SEEDS}" | tee -a "$LOGFILE"
echo "[A2] 輸出目錄：${OUT_A2}" | tee -a "$LOGFILE"

$PYTHON -m simulation.personality_rl_runtime \
  --seeds "${SEEDS}" \
  --n-players "${N_PLAYERS}" \
  --n-rounds "${N_ROUNDS}" \
  --personality-mode random \
  --event-dispatch-mode async_poisson \
  --event-dispatch-target-rate 0.08 \
  --event-dispatch-fairness-window 2000 \
  --event-dispatch-fairness-tolerance 0.50 \
  --events-json "${EVENTS_JSON}" \
  --world-feedback-mode adaptive_world \
  --lambda-world 0.04 \
  --world-update-interval 200 \
  --out-dir "${OUT_A2}" \
  2>&1 | tee -a "$LOGFILE"

echo "[A2] 執行完成。" | tee -a "$LOGFILE"

# ──────────────────────────────────────────────────────────────────────────────
# Step 4：A2 指標計算
# ──────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "[A2] 計算 cycle metrics..." | tee -a "$LOGFILE"

A2_METRICS_JSON="outputs/exp_A2_metrics.json"

$PYTHON - <<PYEOF 2>&1 | tee -a "$LOGFILE"
import csv, json, pathlib, sys
from analysis.cycle_metrics import classify_cycle_level

OUT_DIR = pathlib.Path("${OUT_A2}")
BURN_IN = ${BURN_IN}
TAIL    = ${TAIL}

results = []
seed_dirs = sorted(OUT_DIR.glob("seed_*"))
if not seed_dirs:
    print("[ERROR] A2: 找不到任何 seed_* 目錄")
    sys.exit(1)

for sdir in seed_dirs:
    csv_path = sdir / "timeseries.csv"
    prov_path = sdir / "provenance.json"
    if not csv_path.exists():
        print(f"[WARN] 缺少 {csv_path}，跳過")
        continue
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    series = {
        "aggressive": [float(r["p_aggressive"]) for r in rows],
        "defensive":  [float(r["p_defensive"])  for r in rows],
        "balanced":   [float(r["p_balanced"])   for r in rows],
    }
    cyc = classify_cycle_level(
        series,
        burn_in=BURN_IN,
        tail=TAIL,
        amplitude_threshold=0.02,
        corr_threshold=0.09,
        eta=0.55,
        stage3_method="turning",
        phase_smoothing=1,
        min_lag=2,
        max_lag=500,
    )
    s3  = float(cyc.stage3.score) if cyc.stage3 else 0.0
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    bhr = prov.get("boundary_hit_rate", 0.0)
    results.append({
        "seed_dir": str(sdir.name),
        "cycle_level": cyc.level,
        "stage3_score": s3,
        "boundary_hit_rate": bhr,
    })
    print(f"  {sdir.name}: level={cyc.level}  s3={s3:.6f}  bhr={bhr:.4f}")

mean_s3 = sum(r["stage3_score"] for r in results) / max(len(results), 1)
mean_bhr = sum(r["boundary_hit_rate"] for r in results) / max(len(results), 1)
level_counts = {}
for r in results:
    lv = str(r["cycle_level"])
    level_counts[lv] = level_counts.get(lv, 0) + 1

summary = {
    "exp_id": "A2",
    "personality_mode": "random",
    "n_seeds": len(results),
    "mean_s3": mean_s3,
    "mean_boundary_hit_rate": mean_bhr,
    "level_counts": level_counts,
    "per_seed": results,
}
pathlib.Path("${A2_METRICS_JSON}").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False)
)
print(f"[A2] mean_s3={mean_s3:.6f}  mean_bhr={mean_bhr:.4f}  level_counts={level_counts}")
print(f"[A2] 指標已存至 ${A2_METRICS_JSON}")
PYEOF

echo "[A2] 指標計算完成。" | tee -a "$LOGFILE"

# ──────────────────────────────────────────────────────────────────────────────
# Step 5：橋接比較 — 計算漂移率並判定 HALT / WARN / PASS
# ──────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "[橋接比較] 計算 A1/A2 mean_s3 漂移率..." | tee -a "$LOGFILE"

BRIDGE_STATUS=$($PYTHON - <<PYEOF
import json, sys

a1 = json.loads(open("${A1_METRICS_JSON}").read())
a2 = json.loads(open("${A2_METRICS_JSON}").read())

s3_a1 = a1["mean_s3"]
s3_a2 = a2["mean_s3"]
bhr_a1 = a1["mean_boundary_hit_rate"]
bhr_a2 = a2["mean_boundary_hit_rate"]

drift = abs(s3_a2 - s3_a1) / max(abs(s3_a1), 1e-6)
bhr_ratio = bhr_a2 / max(bhr_a1, 1e-9)

# 主要判定
if drift > 0.15:
    status = "HALT"
    reason = f"mean_s3 漂移率 {drift*100:.1f}% > 15% 門檻"
elif drift > 0.10:
    status = "WARN"
    reason = f"mean_s3 漂移率 {drift*100:.1f}% 在 10-15% 警戒區間"
else:
    status = "PASS"
    reason = f"mean_s3 漂移率 {drift*100:.1f}% < 10%"

# 次要判定：boundary_hit_rate 比值
bhr_flag = ""
if bhr_ratio > 2.0:
    bhr_flag = f"  ⚠  boundary_hit_rate 比值 {bhr_ratio:.2f} > 2.0（WARN）"
    if status == "PASS":
        status = "WARN"
        reason += f"；boundary_hit_rate 比值 {bhr_ratio:.2f} > 2.0"

bridge = {
    "A1_mean_s3": s3_a1,
    "A2_mean_s3": s3_a2,
    "drift_pct": round(drift * 100, 2),
    "A1_mean_bhr": bhr_a1,
    "A2_mean_bhr": bhr_a2,
    "bhr_ratio": round(bhr_ratio, 4),
    "bridge_status": status,
    "reason": reason,
}
with open("outputs/exp_A1_A2_bridge.json", "w", encoding="utf-8") as f:
    json.dump(bridge, f, indent=2, ensure_ascii=False)

sep = "=" * 60
print(sep)
print(f"  A1 mean_s3 = {s3_a1:.6f}   bhr = {bhr_a1:.4f}")
print(f"  A2 mean_s3 = {s3_a2:.6f}   bhr = {bhr_a2:.4f}")
print(f"  漂移率     = {drift*100:.1f}%")
if bhr_flag:
    print(bhr_flag)
print(f"  判定       = {status}  （{reason}）")
print(sep)
print(status)
PYEOF
)

# 最後兩行：最後一行是 STATUS，前面是比較表格
FINAL_STATUS=$(echo "$BRIDGE_STATUS" | tail -n 1)
BRIDGE_TABLE=$(echo "$BRIDGE_STATUS" | head -n -1)

echo "$BRIDGE_TABLE" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "=== run_exp_A1_A2.sh 結束：$(date) ===" | tee -a "$LOGFILE"
echo "橋接狀態：${FINAL_STATUS}" | tee -a "$LOGFILE"

# 根據狀態設定 exit code
if [[ "$FINAL_STATUS" == "HALT" ]]; then
    echo "[HALT] 漂移率超標，禁止推進 B/C/D 階段。請回檢 Schema 與人格注入邏輯。" | tee -a "$LOGFILE"
    exit 1
elif [[ "$FINAL_STATUS" == "WARN" ]]; then
    echo "[WARN] 漂移率在警戒區間，可繼續推進但須附加橋接修正條件，並記入研發日誌。" | tee -a "$LOGFILE"
    exit 2
else
    echo "[PASS] 橋接基線正常，可推進 B/C/D 階段。" | tee -a "$LOGFILE"
    exit 0
fi
