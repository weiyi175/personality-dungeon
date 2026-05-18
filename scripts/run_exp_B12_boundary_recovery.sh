#!/usr/bin/env bash
# =============================================================================
# run_exp_B12_boundary_recovery.sh
# 目的：
#   1) P0: A1/A2 bridge sanity 重算（直接讀 timeseries.csv）
#   2) P1: B1/B2 邊界恢復微掃（12 seeds, rounds=6000）
#      - selection-strength: 0.06, 0.08, 0.10
#      - memory-kernel: 預設 3（可用 MK_VALUES 覆寫）
#
# 用法：
#   bash scripts/run_exp_B12_boundary_recovery.sh
#   SS_VALUES="0.06 0.08 0.10" MK_VALUES="3 5" bash scripts/run_exp_B12_boundary_recovery.sh
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=./venv/bin/python
LOGFILE="outputs/exp_B12_boundary_recovery.log"
SUMMARY_JSON="outputs/exp_B12_boundary_recovery_summary.json"
BRIDGE_JSON="outputs/exp_bridge_sanity_recheck.json"
mkdir -p outputs

echo "=== run_exp_B12_boundary_recovery.sh 開始：$(date) ===" | tee "$LOGFILE"

SEEDS="45,47,49,51,53,55,91,93,95,97,99,101"
ROUNDS=6000
PLAYERS=300
BURN_IN=2000
TAIL=2000
SS_VALUES="${SS_VALUES:-0.06 0.08 0.10}"
MK_VALUES="${MK_VALUES:-3}"

# -----------------------------------------------------------------------------
# P0: Bridge sanity re-check（無 pandas，直接從 timeseries.csv 重算）
# -----------------------------------------------------------------------------
$PYTHON - <<'PYEOF' 2>&1 | tee -a "$LOGFILE"
import csv
import json
from pathlib import Path
from datetime import datetime
from analysis.cycle_metrics import classify_cycle_level


def recompute(exp_dir: str):
    levels = []
    s3s = []
    for p in Path(exp_dir).glob('seed_*/timeseries.csv'):
        with p.open() as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        series = {s: [float(r[f'p_{s}']) for r in rows] for s in ['aggressive', 'defensive', 'balanced']}
        cyc = classify_cycle_level(series, burn_in=4000, tail=4000, eta=0.55)
        levels.append(int(cyc.level))
        s3s.append(float(cyc.stage3.score) if cyc.stage3 else 0.0)
    mean_s3 = sum(s3s) / len(s3s) if s3s else 0.0
    level_counts = {}
    for lv in levels:
        key = str(lv)
        level_counts[key] = level_counts.get(key, 0) + 1
    return {
        'n': len(levels),
        'levels': levels,
        'level_counts': level_counts,
        'mean_s3': mean_s3,
    }

res_a1 = recompute('outputs/exp_A1_baseline_none')
res_a2 = recompute('outputs/exp_A2_baseline_9persona')

drift = abs(res_a2['mean_s3'] - res_a1['mean_s3'])

payload = {
    'timestamp': datetime.now().isoformat(),
    'a1': res_a1,
    'a2': res_a2,
    'mean_s3_abs_diff': drift,
    'consistency_note': 'Use level distribution + recomputation consistency, not mean_s3>0 threshold.',
}

Path('outputs/exp_bridge_sanity_recheck.json').write_text(
    json.dumps(payload, indent=2, ensure_ascii=False),
    encoding='utf-8',
)

print('--- [P0] Bridge Sanity Re-check ---')
print(f"A1: mean_s3={res_a1['mean_s3']:.6f}, levels={res_a1['levels']}")
print(f"A2: mean_s3={res_a2['mean_s3']:.6f}, levels={res_a2['levels']}")
print(f"|A2-A1|={drift:.6f}")
print('bridge_json=outputs/exp_bridge_sanity_recheck.json')
PYEOF

# -----------------------------------------------------------------------------
# P1: B1/B2 micro-sweep
# -----------------------------------------------------------------------------
TMP_RESULT="outputs/.tmp_exp_B12_boundary_recovery.tsv"
: > "$TMP_RESULT"
echo -e "module\tcondition\tss\tmk\tcombined_tsv\tverdict\tlevel3_seed_count\tmean_stage3_score\tstatus" >> "$TMP_RESULT"

run_and_eval() {
  local module="$1"
  local condition="$2"
  local tag="$3"
  local ss="$4"
  local mk="$5"

  local out_root="outputs/recovery_${tag}_ss${ss}_mk${mk}_r${ROUNDS}"
  local combined_tsv="${out_root}_combined.tsv"
  local summary_tsv="${out_root}_summary.tsv"
  local decision_md="${out_root}_decision.md"

  echo "" | tee -a "$LOGFILE"
  echo "[P1] module=${module} cond=${condition} ss=${ss} mk=${mk}" | tee -a "$LOGFILE"

  if [[ "$module" == "w3_stackelberg" ]]; then
    $PYTHON -m simulation.w3_stackelberg \
            --conditions control,"$condition" \
      --seeds "$SEEDS" \
      --players "$PLAYERS" \
      --rounds "$ROUNDS" \
      --selection-strength "$ss" \
      --init-bias 0.12 \
      --memory-kernel "$mk" \
      --burn-in "$BURN_IN" \
      --tail "$TAIL" \
      --out-root "$out_root" \
      --summary-tsv "$summary_tsv" \
      --combined-tsv "$combined_tsv" \
      --decision-md "$decision_md" \
      2>&1 | tee -a "$LOGFILE"
  else
    $PYTHON -m simulation.w3_policy \
            --conditions control_policy,"$condition" \
      --seeds "$SEEDS" \
      --players "$PLAYERS" \
      --rounds "$ROUNDS" \
      --selection-strength "$ss" \
      --init-bias 0.12 \
      --memory-kernel "$mk" \
      --policy-update-interval 150 \
      --theta-low 0.08 \
      --theta-high 0.12 \
      --burn-in "$BURN_IN" \
      --tail "$TAIL" \
      --out-root "$out_root" \
      --summary-tsv "$summary_tsv" \
      --combined-tsv "$combined_tsv" \
      --decision-md "$decision_md" \
      2>&1 | tee -a "$LOGFILE"
  fi

  local eval_line
  eval_line=$($PYTHON - <<PYEOF
import csv
from pathlib import Path

combined = Path("$combined_tsv")
condition = "$condition"
if not combined.exists():
    print("\t\t\t\t\t\t\tMISSING_COMBINED")
else:
    with combined.open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    row = None
    for r in rows:
        if r.get('condition', '').strip() == condition:
            row = r
            break
    if row is None:
        print("\t\t\t\t\t\t\tMISSING_ROW")
    else:
        verdict = str(row.get('verdict', '')).strip()
        l3 = str(row.get('level3_seed_count', '')).strip()
        s3 = str(row.get('mean_stage3_score', '')).strip()
        try:
            l3_i = int(l3)
        except Exception:
            l3_i = 0
        status = 'RECOVERED' if l3_i >= 2 else 'STILL_FAIL'
        print(f"{verdict}\t{l3}\t{s3}\t{status}")
PYEOF
)

  local verdict l3 s3 status
  verdict=$(echo "$eval_line" | cut -f1)
  l3=$(echo "$eval_line" | cut -f2)
  s3=$(echo "$eval_line" | cut -f3)
  status=$(echo "$eval_line" | cut -f4)

  if [[ -z "$status" ]]; then
    status="EVAL_ERROR"
  fi

  echo "[P1] verdict=${verdict:-NA} l3=${l3:-NA} s3=${s3:-NA} status=${status}" | tee -a "$LOGFILE"
  echo -e "${module}\t${condition}\t${ss}\t${mk}\t${combined_tsv}\t${verdict}\t${l3}\t${s3}\t${status}" >> "$TMP_RESULT"
}

for mk in $MK_VALUES; do
  for ss in $SS_VALUES; do
    run_and_eval "w3_stackelberg" "w3_commit_push" "B1" "$ss" "$mk"
    run_and_eval "w3_policy" "w3_policy_crossguard" "B2" "$ss" "$mk"
  done
done

# -----------------------------------------------------------------------------
# 彙總 JSON
# -----------------------------------------------------------------------------
$PYTHON - <<'PYEOF' 2>&1 | tee -a "$LOGFILE"
import csv
import json
from pathlib import Path
from datetime import datetime

rows = []
with Path('outputs/.tmp_exp_B12_boundary_recovery.tsv').open() as f:
    reader = csv.DictReader(f, delimiter='\t')
    rows = list(reader)

by_module = {}
for r in rows:
    m = r['module']
    by_module.setdefault(m, []).append(r)

summary = {
    'timestamp': datetime.now().isoformat(),
    'bridge_recheck_json': 'outputs/exp_bridge_sanity_recheck.json',
    'sweep_rows': rows,
    'module_overview': {},
}

for mod, items in by_module.items():
    recovered = [x for x in items if x.get('status') == 'RECOVERED']
    summary['module_overview'][mod] = {
        'n_configs': len(items),
        'n_recovered': len(recovered),
        'best': recovered[0] if recovered else None,
    }

Path('outputs/exp_B12_boundary_recovery_summary.json').write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding='utf-8',
)

print('--- Sweep Summary ---')
for mod, info in summary['module_overview'].items():
    print(f"{mod}: recovered={info['n_recovered']}/{info['n_configs']}")
print('summary_json=outputs/exp_B12_boundary_recovery_summary.json')
PYEOF

rm -f "$TMP_RESULT"

echo "=== run_exp_B12_boundary_recovery.sh 結束：$(date) ===" | tee -a "$LOGFILE"
