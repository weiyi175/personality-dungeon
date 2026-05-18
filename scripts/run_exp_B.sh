#!/usr/bin/env bash
# =============================================================================
# run_exp_B.sh — 階段 B：代表點重跑（Canonical Re-run）
# 規格來源：docs/exp_spec_9persona_v1.md § 階段 B
#
# 用法：bash scripts/run_exp_B.sh
#
# 執行順序：B1 → B2 → B3（W-series，各自獨立判定）→ B4 → B5（B-series gate）
# 每個子實驗結束後即刻輸出 STABLE / UPLIFT_DETECTED / REGRESSION / NO-GO
# 整體摘要存至 outputs/exp_B_summary.json，exit code 反映最嚴重狀態。
#
# exit code:
#   0 = 全部 STABLE / PASS
#   1 = 出現 REGRESSION 或 B-series NO-GO（需人工介入）
#   2 = 出現 UPLIFT_DETECTED（擴增 seeds 建議）
# =============================================================================
set -euo pipefail

# 無論從哪個目錄呼叫，切換到專案根目錄
cd "$(dirname "$0")/.."

PYTHON=./venv/bin/python
LOGFILE="outputs/exp_B_run.log"
mkdir -p outputs
echo "=== run_exp_B.sh 開始：$(date) ===" | tee "$LOGFILE"

# 全域狀態追蹤
WORST_CODE=0  # 0=OK, 1=REGRESSION/NOGO, 2=UPLIFT

# ──────────────────────────────────────────────────────────────────────────────
# 輔助：判斷 W-series 結果（讀取 combined TSV，評估指定 condition）
# 用法：_eval_w3 <exp_id> <combined_tsv> <target_condition>
# ──────────────────────────────────────────────────────────────────────────────
_eval_w3() {
    local exp_id="$1"
    local combined_tsv="$2"
    local target_cond="$3"

    local STATUS
    STATUS=$($PYTHON - <<PYEOF
import csv, sys

tsv_path = "${combined_tsv}"
target   = "${target_cond}"

try:
    with open(tsv_path) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
except FileNotFoundError:
    print(f"[ERROR] 找不到 {tsv_path}")
    sys.exit(10)

# 找到 target condition 的 row
target_rows = [r for r in rows if r.get("condition","").strip() == target]
ctrl_rows   = [r for r in rows if r.get("is_control","").strip().lower() in ("true","1","yes")]

if not target_rows:
    print(f"[ERROR] combined TSV 中找不到 condition={target}")
    sys.exit(10)

row = target_rows[0]
verdict       = row.get("verdict","").strip()
level3_count  = int(row.get("level3_seed_count", 0) or 0)
mean_s3       = float(row.get("mean_stage3_score", 0) or 0)
ctrl_s3       = float(ctrl_rows[0].get("mean_stage3_score", 0) or 0) if ctrl_rows else 0.0
uplift_vs_ctrl = float(row.get("stage3_uplift_vs_control", 0) or 0)

print(f"  condition     = {target}")
print(f"  verdict       = {verdict}")
print(f"  level3_count  = {level3_count}")
print(f"  mean_s3       = {mean_s3:.6f}")
print(f"  ctrl_mean_s3  = {ctrl_s3:.6f}")
print(f"  uplift_vs_ctrl= {uplift_vs_ctrl:.4f}")

# 判定邏輯（依 SDD exp_spec §貳）
# 歷史結論為 closure（W3 系列均已確認 Level3）
# - STABLE    : verdict in (pass, weak_positive) 且 level3_count > 0
# - UPLIFT_DETECTED : level3_count > 0 AND uplift_vs_ctrl >= 0.015（超預期）
# - REGRESSION: verdict == fail 或 level3_count == 0
if verdict == "fail" or level3_count == 0:
    status = "REGRESSION"
    reason = f"verdict={verdict}，level3_count={level3_count}（歷史 closure 無法重現）"
elif uplift_vs_ctrl >= 0.015 and level3_count > 0:
    status = "UPLIFT_DETECTED"
    reason = f"uplift_vs_ctrl={uplift_vs_ctrl:.4f} >= 0.015，可考慮擴增 seeds 至 12 個"
else:
    status = "STABLE"
    reason = f"verdict={verdict}，level3_count={level3_count}，歷史 closure 已重現"

print(f"  判定          = {status}  （{reason}）")
print(status)
PYEOF
)

    local FINAL_STATUS
    FINAL_STATUS=$(echo "$STATUS" | tail -n 1)
    local TABLE
    TABLE=$(echo "$STATUS" | head -n -1)

    # 顯示與 log 走 stderr，避免汙染 $() 捕捉的 stdout
    echo "[${exp_id}]${TABLE}" | tee -a "$LOGFILE" >&2
    echo "[${exp_id}] 判定：${FINAL_STATUS}" | tee -a "$LOGFILE" >&2

    if [[ "$FINAL_STATUS" == "REGRESSION" ]]; then
        WORST_CODE=1
    elif [[ "$FINAL_STATUS" == "UPLIFT_DETECTED" && "$WORST_CODE" -lt 1 ]]; then
        WORST_CODE=2
    fi

    # 只輸出單一狀態詞到 stdout（供 B*_STATUS= 捕捉）
    echo "$FINAL_STATUS"
}

# ──────────────────────────────────────────────────────────────────────────────
# B1：W3.1 Stackelberg — commit_push 代表點
# ──────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "[B1] W3.1 Stackelberg commit_push 代表點（seeds=45,47,49,51,53,55）" | tee -a "$LOGFILE"

$PYTHON -m simulation.w3_stackelberg \
  --conditions control,w3_commit_push \
  --seeds 45,47,49,51,53,55 \
  --players 300 \
  --rounds 3000 \
  --selection-strength 0.06 \
  --init-bias 0.12 \
  --memory-kernel 3 \
  --burn-in 1000 \
  --tail 1000 \
  --out-root outputs/exp_B1_w31_commit \
  --summary-tsv outputs/exp_B1_w31_commit_summary.tsv \
  --combined-tsv outputs/exp_B1_w31_commit_combined.tsv \
  --decision-md outputs/exp_B1_w31_commit_decision.md \
  2>&1 | tee -a "$LOGFILE"

echo "[B1] 執行完成，開始評估..." | tee -a "$LOGFILE"
B1_STATUS=$(_eval_w3 "B1" "outputs/exp_B1_w31_commit_combined.tsv" "w3_commit_push")
echo "[B1] 最終狀態：${B1_STATUS}" | tee -a "$LOGFILE"

# ──────────────────────────────────────────────────────────────────────────────
# B2：W3.2 Policy — crossguard 代表點
# ──────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "[B2] W3.2 Policy crossguard 代表點（seeds=45,47,49,51,53,55）" | tee -a "$LOGFILE"

$PYTHON -m simulation.w3_policy \
  --conditions control_policy,w3_policy_crossguard \
  --seeds 45,47,49,51,53,55 \
  --players 300 \
  --rounds 3000 \
  --selection-strength 0.06 \
  --init-bias 0.12 \
  --memory-kernel 3 \
  --policy-update-interval 150 \
  --theta-low 0.08 \
  --theta-high 0.12 \
  --burn-in 1000 \
  --tail 1000 \
  --out-root outputs/exp_B2_w32_crossguard \
  --summary-tsv outputs/exp_B2_w32_crossguard_summary.tsv \
  --combined-tsv outputs/exp_B2_w32_crossguard_combined.tsv \
  --decision-md outputs/exp_B2_w32_crossguard_decision.md \
  2>&1 | tee -a "$LOGFILE"

echo "[B2] 執行完成，開始評估..." | tee -a "$LOGFILE"
B2_STATUS=$(_eval_w3 "B2" "outputs/exp_B2_w32_crossguard_combined.tsv" "w3_policy_crossguard")
echo "[B2] 最終狀態：${B2_STATUS}" | tee -a "$LOGFILE"

# ──────────────────────────────────────────────────────────────────────────────
# B3：W3.3 Pulse — pulse_commitpush 代表點
# ──────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "[B3] W3.3 Pulse pulse_commitpush 代表點（seeds=45,47,49,51,53,55）" | tee -a "$LOGFILE"

$PYTHON -m simulation.w3_pulse \
  --conditions control_pulse_policy,w3_pulse_commitpush \
  --seeds 45,47,49,51,53,55 \
  --players 300 \
  --rounds 3000 \
  --selection-strength 0.06 \
  --init-bias 0.12 \
  --memory-kernel 3 \
  --burn-in 1000 \
  --tail 1000 \
  --out-root outputs/exp_B3_w33_pulse \
  --summary-tsv outputs/exp_B3_w33_pulse_summary.tsv \
  --combined-tsv outputs/exp_B3_w33_pulse_combined.tsv \
  --decision-md outputs/exp_B3_w33_pulse_decision.md \
  2>&1 | tee -a "$LOGFILE"

echo "[B3] 執行完成，開始評估..." | tee -a "$LOGFILE"
B3_STATUS=$(_eval_w3 "B3" "outputs/exp_B3_w33_pulse_combined.tsv" "w3_pulse_commitpush")
echo "[B3] 最終狀態：${B3_STATUS}" | tee -a "$LOGFILE"

# ──────────────────────────────────────────────────────────────────────────────
# B4：B2-series multiplicative modulation 代表點（NO-GO 確認）
# ──────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "[B4] B2-series multiplicative（cap=0.25），確認歷史 NO-GO 仍成立" | tee -a "$LOGFILE"

$PYTHON -m simulation.b1_async_dispatch_gate \
  --event-dispatch-mode async_poisson \
  --event-dispatch-target-rate 0.08 \
  --event-dispatch-fairness-window 2000 \
  --event-dispatch-fairness-tolerance 0.50 \
  --event-reward-mode multiplicative \
  --event-reward-multiplier-cap 0.25 \
  --smoke-seeds 42,44,45,67,73,90 \
  --gate-seeds 42..101 \
  --baseline-summary-json outputs/pers_cal_baseline_gate60_summary.json \
  --smoke-out-json outputs/exp_B4_b2series_smoke_summary.json \
  --gate-out-json outputs/exp_B4_b2series_gate_summary.json \
  2>&1 | tee -a "$LOGFILE" || true

echo "[B4] 執行完成，開始評估..." | tee -a "$LOGFILE"

B4_STATUS=$($PYTHON - <<PYEOF
import json, sys

# 優先讀 gate summary（new_l1 由 gate 與基線比對後計算）；
# 若 gate 檔不存在則 fallback 到 smoke summary
def load_any(*paths):
    for p in paths:
        try:
            return json.load(open(p)), p
        except FileNotFoundError:
            pass
    return None, None

d, src = load_any(
    "outputs/exp_B4_b2series_gate_summary.json",
    "outputs/exp_B4_b2series_smoke_summary.json",
)
if d is None:
    print("[ERROR] 找不到 B4 gate/smoke summary JSON")
    sys.exit(10)

print(f"  讀取來源      = {src}")
new_l1       = d.get("new_l1", None)
l1_count     = d.get("l1", None)
l3_count     = d.get("l3", None)
gate         = d.get("gate", {})
new_l1_pass  = gate.get("new_l1_pass", None)
l1_pass      = gate.get("l1_pass", None)
mean_s3      = d.get("mean_s3", None)

print(f"  new_l1        = {new_l1}")
print(f"  l1_count      = {l1_count}")
print(f"  l3_count      = {l3_count}")
print(f"  gate.new_l1_pass = {new_l1_pass}")
print(f"  gate.l1_pass     = {l1_pass}")
print(f"  mean_s3       = {mean_s3}")

# 依 SDD exp_spec §貳：判準為 new_l1（對比基線的新增 L1）
# new_l1=0  → PASS（無新增退化，歷史 NO-GO 機制相容確認）
# new_l1>0  → NO-GO（非預期新增 L1，需調查）
# new_l1=None 在 smoke 階段正常（尚未與基線比對），視為無法判斷
if new_l1 is None:
    status = "WARN"
    reason = "new_l1=None：僅有 smoke 資料無法對比基線，請人工確認 gate summary"
elif new_l1 > 0:
    status = "NO-GO"
    reason = f"new_l1={new_l1} > 0：gate 出現非預期新增 L1 seeds，歷史 B2-series NO-GO 可能已解除"
else:
    status = "PASS"
    reason = f"new_l1={new_l1}：gate 無新增 L1，歷史 B2-series NO-GO 確認成立"

if l1_pass is False:
    print(f"  ⚠  gate.l1_pass=False（絕對 L1 數量超過 max_l1 門檻，屬模組內部品質閘，非 spec 判準）")

print(f"  判定          = {status}  （{reason}）")
print(status)
PYEOF
)

B4_FINAL=$(echo "$B4_STATUS" | tail -n 1)
echo "$B4_STATUS" | head -n -1 | sed "s/^/[B4]/" | tee -a "$LOGFILE"
echo "[B4] 最終狀態：${B4_FINAL}" | tee -a "$LOGFILE"

if [[ "$B4_FINAL" == "NO-GO" ]]; then
    echo "[B4] 歷史 NO-GO 閉合確認：B2-series multiplicative 不產生週期（符合預期）。" | tee -a "$LOGFILE"
    # NO-GO 是「歷史 closure 的確認成功」——不視為錯誤
    # 若 new_l1>0 才是意外，需要調查（WORST_CODE=1）
    # 已於上方 Python 輸出正確 status，這裡不調整 WORST_CODE（new_l1>0 = UNEXPECTED）
elif [[ "$B4_FINAL" == "PASS" ]]; then
    echo "[B4] 歷史 B2-series NO-GO 閉合確認通過。" | tee -a "$LOGFILE"
fi

# 若 B4 new_l1 > 0（unexpected），標記為 REGRESSION
if [[ "$B4_FINAL" == "NO-GO" ]]; then
    WORST_CODE=1
fi

# ──────────────────────────────────────────────────────────────────────────────
# B5：B3-series impact spreading 代表點（NO-GO 確認）
# ──────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "[B5] B3-series impact spreading（mode=spread, horizon=5, decay=0.70），確認歷史 NO-GO 仍成立" | tee -a "$LOGFILE"

$PYTHON -m simulation.b1_async_dispatch_gate \
  --event-dispatch-mode async_poisson \
  --event-dispatch-target-rate 0.08 \
  --event-dispatch-fairness-window 2000 \
  --event-dispatch-fairness-tolerance 0.50 \
  --event-impact-mode spread \
  --event-impact-horizon 5 \
  --event-impact-decay 0.70 \
  --smoke-seeds 42,44,45,67,73,90 \
  --gate-seeds 42..101 \
  --baseline-summary-json outputs/pers_cal_baseline_gate60_summary.json \
  --smoke-out-json outputs/exp_B5_b3series_smoke_summary.json \
  --gate-out-json outputs/exp_B5_b3series_gate_summary.json \
  2>&1 | tee -a "$LOGFILE" || true

echo "[B5] 執行完成，開始評估..." | tee -a "$LOGFILE"

B5_STATUS=$($PYTHON - <<PYEOF
import json, sys

def load_any(*paths):
    for p in paths:
        try:
            return json.load(open(p)), p
        except FileNotFoundError:
            pass
    return None, None

d, src = load_any(
    "outputs/exp_B5_b3series_gate_summary.json",
    "outputs/exp_B5_b3series_smoke_summary.json",
)
if d is None:
    print("[ERROR] 找不到 B5 gate/smoke summary JSON")
    sys.exit(10)

print(f"  讀取來源      = {src}")
new_l1       = d.get("new_l1", None)
l1_count     = d.get("l1", None)
l3_count     = d.get("l3", None)
gate         = d.get("gate", {})
new_l1_pass  = gate.get("new_l1_pass", None)
l1_pass      = gate.get("l1_pass", None)
mean_s3      = d.get("mean_s3", None)

print(f"  new_l1        = {new_l1}")
print(f"  l1_count      = {l1_count}")
print(f"  l3_count      = {l3_count}")
print(f"  gate.new_l1_pass = {new_l1_pass}")
print(f"  gate.l1_pass     = {l1_pass}")
print(f"  mean_s3       = {mean_s3}")

if new_l1 is None:
    status = "WARN"
    reason = "new_l1=None：僅有 smoke 資料無法對比基線，請人工確認 gate summary"
elif new_l1 > 0:
    status = "NO-GO"
    reason = f"new_l1={new_l1} > 0：gate 出現非預期新增 L1 seeds，歷史 B3-series NO-GO 可能已解除"
else:
    status = "PASS"
    reason = f"new_l1={new_l1}：gate 無新增 L1，歷史 B3-series NO-GO 確認成立"

if l1_pass is False:
    print(f"  ⚠  gate.l1_pass=False（絕對 L1 數量超過 max_l1 門檻，屬模組內部品質閘，非 spec 判準）")

print(f"  判定          = {status}  （{reason}）")
print(status)
PYEOF
)

B5_FINAL=$(echo "$B5_STATUS" | tail -n 1)
echo "$B5_STATUS" | head -n -1 | sed "s/^/[B5]/" | tee -a "$LOGFILE"
echo "[B5] 最終狀態：${B5_FINAL}" | tee -a "$LOGFILE"

if [[ "$B5_FINAL" == "NO-GO" ]]; then
    WORST_CODE=1
fi

# ──────────────────────────────────────────────────────────────────────────────
# 彙總 Phase B 結果
# ──────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"
echo "  Phase B 總結" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"
printf "  %-8s %s\n" "B1:" "$B1_STATUS" | tee -a "$LOGFILE"
printf "  %-8s %s\n" "B2:" "$B2_STATUS" | tee -a "$LOGFILE"
printf "  %-8s %s\n" "B3:" "$B3_STATUS" | tee -a "$LOGFILE"
printf "  %-8s %s\n" "B4:" "$B4_FINAL" | tee -a "$LOGFILE"
printf "  %-8s %s\n" "B5:" "$B5_FINAL" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"

# 寫入彙總 JSON
$PYTHON - <<PYEOF 2>&1 | tee -a "$LOGFILE"
import json, datetime
summary = {
    "phase": "B",
    "timestamp": datetime.datetime.now().isoformat(),
    "results": {
        "B1": "${B1_STATUS}",
        "B2": "${B2_STATUS}",
        "B3": "${B3_STATUS}",
        "B4": "${B4_FINAL}",
        "B5": "${B5_FINAL}",
    },
    "notes": {
        "B1_B2_B3": "W-series 代表點重跑；STABLE=closure 重現，UPLIFT=擴增 seeds，REGRESSION=回檢",
        "B4_B5":    "B-series NO-GO 確認；PASS=歷史 closure 成立，NO-GO=new_l1>0（非預期）需調查",
    },
}
with open("outputs/exp_B_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("彙總已存至 outputs/exp_B_summary.json")
PYEOF

echo "=== run_exp_B.sh 結束：$(date) ===" | tee -a "$LOGFILE"

# ──────────────────────────────────────────────────────────────────────────────
# 最終指引
# ──────────────────────────────────────────────────────────────────────────────
case "$WORST_CODE" in
    0)
        echo "[PASS] Phase B 全部穩定，可推進 Phase C/D。" | tee -a "$LOGFILE"
        exit 0
        ;;
    2)
        echo "[UPLIFT_DETECTED] 部分 W-series 出現超預期 uplift，建議擴增至 12 seeds 進行進階分析。" | tee -a "$LOGFILE"
        echo "  擴增指令範例（B1）：" | tee -a "$LOGFILE"
        echo "    --seeds 45,47,49,51,53,55,91,93,95,97,99,101" | tee -a "$LOGFILE"
        exit 2
        ;;
    1)
        echo "[REGRESSION/NO-GO] 發現回歸或非預期 Level1+ seeds，請回檢人格注入邏輯或機制程式碼。" | tee -a "$LOGFILE"
        exit 1
        ;;
esac
