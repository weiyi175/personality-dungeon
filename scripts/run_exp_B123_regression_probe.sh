#!/usr/bin/env bash
# =============================================================================
# run_exp_B123_regression_probe.sh
# 目的：B1/B2/B3 REGRESSION 根因補驗（12 seeds + 6000 rounds）
#
# 設計重點：
# 1) 各 cell 擴增到 12 seeds
# 2) rounds 拉長到 6000（burn-in/tail 仍 1000）
# 3) 與既有 6-seed/3000-round 結果並排比較
# 4) 若有 A1/A2 baseline 產物，額外做橋接 drift 再確認
#
# 用法：
#   bash scripts/run_exp_B123_regression_probe.sh
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=./venv/bin/python
LOGFILE="outputs/exp_B123_regression_probe.log"
SUMMARY_JSON="outputs/exp_B123_regression_probe_summary.json"
mkdir -p outputs

echo "=== run_exp_B123_regression_probe.sh 開始：$(date) ===" | tee "$LOGFILE"

SEEDS="45,47,49,51,53,55,91,93,95,97,99,101"
ROUNDS=6000
PLAYERS=300
BURN_IN=1000
TAIL=1000

# 既有 3000 rounds 對照檔（若不存在，評估時會標記為 unavailable）
OLD_B1="outputs/exp_B1_w31_commit_combined.tsv"
OLD_B2="outputs/exp_B2_w32_crossguard_combined.tsv"
OLD_B3="outputs/exp_B3_w33_pulse_combined.tsv"

NEW_B1="outputs/exp_B1_w31_commit_x12_r6000"
NEW_B2="outputs/exp_B2_w32_crossguard_x12_r6000"
NEW_B3="outputs/exp_B3_w33_pulse_x12_r6000"

NEW_B1_COMBINED="outputs/exp_B1_w31_commit_x12_r6000_combined.tsv"
NEW_B2_COMBINED="outputs/exp_B2_w32_crossguard_x12_r6000_combined.tsv"
NEW_B3_COMBINED="outputs/exp_B3_w33_pulse_x12_r6000_combined.tsv"

# B1
# ----------------------------------------------------------------------------
echo "" | tee -a "$LOGFILE"
echo "[B1] 12-seed + rounds=${ROUNDS} 補驗開始" | tee -a "$LOGFILE"
$PYTHON -m simulation.w3_stackelberg \
  --conditions control,w3_commit_push \
  --seeds "$SEEDS" \
  --players "$PLAYERS" \
  --rounds "$ROUNDS" \
  --selection-strength 0.06 \
  --init-bias 0.12 \
  --memory-kernel 3 \
  --burn-in "$BURN_IN" \
  --tail "$TAIL" \
  --out-root "$NEW_B1" \
  --summary-tsv outputs/exp_B1_w31_commit_x12_r6000_summary.tsv \
  --combined-tsv "$NEW_B1_COMBINED" \
  --decision-md outputs/exp_B1_w31_commit_x12_r6000_decision.md \
  2>&1 | tee -a "$LOGFILE"

# B2
# ----------------------------------------------------------------------------
echo "" | tee -a "$LOGFILE"
echo "[B2] 12-seed + rounds=${ROUNDS} 補驗開始" | tee -a "$LOGFILE"
$PYTHON -m simulation.w3_policy \
  --conditions control_policy,w3_policy_crossguard \
  --seeds "$SEEDS" \
  --players "$PLAYERS" \
  --rounds "$ROUNDS" \
  --selection-strength 0.06 \
  --init-bias 0.12 \
  --memory-kernel 3 \
  --policy-update-interval 150 \
  --theta-low 0.08 \
  --theta-high 0.12 \
  --burn-in "$BURN_IN" \
  --tail "$TAIL" \
  --out-root "$NEW_B2" \
  --summary-tsv outputs/exp_B2_w32_crossguard_x12_r6000_summary.tsv \
  --combined-tsv "$NEW_B2_COMBINED" \
  --decision-md outputs/exp_B2_w32_crossguard_x12_r6000_decision.md \
  2>&1 | tee -a "$LOGFILE"

# B3
# ----------------------------------------------------------------------------
echo "" | tee -a "$LOGFILE"
echo "[B3] 12-seed + rounds=${ROUNDS} 補驗開始" | tee -a "$LOGFILE"
$PYTHON -m simulation.w3_pulse \
  --conditions control_pulse_policy,w3_pulse_commitpush \
  --seeds "$SEEDS" \
  --players "$PLAYERS" \
  --rounds "$ROUNDS" \
  --selection-strength 0.06 \
  --init-bias 0.12 \
  --memory-kernel 3 \
  --burn-in "$BURN_IN" \
  --tail "$TAIL" \
  --out-root "$NEW_B3" \
  --summary-tsv outputs/exp_B3_w33_pulse_x12_r6000_summary.tsv \
  --combined-tsv "$NEW_B3_COMBINED" \
  --decision-md outputs/exp_B3_w33_pulse_x12_r6000_decision.md \
  2>&1 | tee -a "$LOGFILE"

# 評估：closure 恢復與根因指標
# ----------------------------------------------------------------------------
$PYTHON - <<'PYEOF' 2>&1 | tee -a "$LOGFILE"
import csv
import json
from pathlib import Path
from datetime import datetime

pairs = [
    {
        "exp_id": "B1",
        "condition": "w3_commit_push",
        "old": "outputs/exp_B1_w31_commit_combined.tsv",
        "new": "outputs/exp_B1_w31_commit_x12_r6000_combined.tsv",
    },
    {
        "exp_id": "B2",
        "condition": "w3_policy_crossguard",
        "old": "outputs/exp_B2_w32_crossguard_combined.tsv",
        "new": "outputs/exp_B2_w32_crossguard_x12_r6000_combined.tsv",
    },
    {
        "exp_id": "B3",
        "condition": "w3_pulse_commitpush",
        "old": "outputs/exp_B3_w33_pulse_combined.tsv",
        "new": "outputs/exp_B3_w33_pulse_x12_r6000_combined.tsv",
    },
]

def load_row(path: Path, condition: str):
    if not path.exists():
        return None
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    for r in rows:
        if r.get('condition', '').strip() == condition:
            return r
    return None

def to_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default

def to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

results = {}
recovered = 0
persistent = 0

for p in pairs:
    old_row = load_row(Path(p['old']), p['condition'])
    new_row = load_row(Path(p['new']), p['condition'])

    if new_row is None:
        results[p['exp_id']] = {
            'status': 'MISSING_NEW',
            'reason': f"new combined row not found: {p['new']} condition={p['condition']}",
        }
        persistent += 1
        continue

    new_verdict = str(new_row.get('verdict', '')).strip()
    new_l3 = to_int(new_row.get('level3_seed_count', 0))
    new_s3 = to_float(new_row.get('mean_stage3_score', 0.0))
    new_uplift = to_float(new_row.get('stage3_uplift_vs_control', 0.0))

    old_verdict = str(old_row.get('verdict', '')).strip() if old_row else 'NA'
    old_l3 = to_int(old_row.get('level3_seed_count', 0)) if old_row else None
    old_s3 = to_float(old_row.get('mean_stage3_score', 0.0)) if old_row else None

    recovered_closure = (new_verdict != 'fail' and new_l3 > 0)
    if recovered_closure:
        status = 'RECOVERED_CLOSURE'
        recovered += 1
    else:
        status = 'PERSISTENT_REGRESSION'
        persistent += 1

    s3_shift_vs_old = None
    if old_s3 is not None:
        s3_shift_vs_old = (new_s3 - old_s3) / max(abs(old_s3), 1e-9)

    results[p['exp_id']] = {
        'status': status,
        'condition': p['condition'],
        'old': {
            'combined_tsv': p['old'],
            'verdict': old_verdict,
            'level3_seed_count': old_l3,
            'mean_stage3_score': old_s3,
        },
        'new': {
            'combined_tsv': p['new'],
            'verdict': new_verdict,
            'level3_seed_count': new_l3,
            'mean_stage3_score': new_s3,
            'stage3_uplift_vs_control': new_uplift,
        },
        'delta': {
            's3_shift_vs_old_ratio': s3_shift_vs_old,
            'level3_seed_count_delta': (new_l3 - old_l3) if old_l3 is not None else None,
        },
    }

# 橋接再確認（若 A1/A2 都存在，計算 drift；否則標示缺件）

def load_stage3_mean_from_prov(root: Path):
    vals = []
    for p in root.glob('seed_*/provenance.json'):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        vals.append(float(d.get('mean_stage3_score', 0.0)))
    if not vals:
        return None
    return sum(vals) / len(vals)

bridge = {
    'status': 'SKIPPED',
    'reason': '',
    'a1_dir': 'outputs/exp_A1_baseline_none',
    'a2_dir': 'outputs/exp_A2_baseline_9persona',
}
a1_dir = Path('outputs/exp_A1_baseline_none')
a2_dir = Path('outputs/exp_A2_baseline_9persona')

if a1_dir.exists() and a2_dir.exists():
    a1 = load_stage3_mean_from_prov(a1_dir)
    a2 = load_stage3_mean_from_prov(a2_dir)
    if a1 is None or a2 is None:
        bridge['status'] = 'INCOMPLETE'
        bridge['reason'] = 'A1/A2 found but provenance means not available'
    else:
        drift = abs(a2 - a1) / max(abs(a1), 1e-6)
        gate = 'HALT' if drift > 0.15 else ('WARN' if drift > 0.10 else 'PASS')
        bridge.update({
            'status': gate,
            'mean_s3_a1': a1,
            'mean_s3_a2': a2,
            'drift_ratio': drift,
        })
else:
    bridge['status'] = 'MISSING_BASELINE'
    bridge['reason'] = 'exp_A1_baseline_none and/or exp_A2_baseline_9persona not found'

if recovered >= 1:
    root_cause_hint = 'PARAMETER_REGION_LIKELY'
    hint_reason = 'At least one W3 cell recovered closure under 12 seeds + 6000 rounds.'
else:
    root_cause_hint = 'MECHANISM_PATH_LIKELY'
    hint_reason = 'All W3 cells remained in regression under expanded seeds and longer rounds.'

summary = {
    'timestamp': datetime.now().isoformat(),
    'probe': {
        'seeds': [45,47,49,51,53,55,91,93,95,97,99,101],
        'rounds': 6000,
        'players': 300,
        'burn_in': 1000,
        'tail': 1000,
    },
    'results': results,
    'bridge_recheck': bridge,
    'root_cause_hint': {
        'classification': root_cause_hint,
        'reason': hint_reason,
        'recovered_count': recovered,
        'persistent_count': persistent,
    },
}

Path('outputs/exp_B123_regression_probe_summary.json').write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding='utf-8',
)

print('============================================================')
print('B1/B2/B3 擴增補驗總結')
print('============================================================')
for exp_id in ('B1', 'B2', 'B3'):
    r = results.get(exp_id, {})
    print(f"{exp_id}: {r.get('status')} | old={r.get('old',{}).get('verdict')} -> new={r.get('new',{}).get('verdict')} | ")
    print(f"    old_l3={r.get('old',{}).get('level3_seed_count')} new_l3={r.get('new',{}).get('level3_seed_count')} ")
print('------------------------------------------------------------')
print(f"bridge_recheck={bridge.get('status')} reason={bridge.get('reason','')}")
print(f"root_cause_hint={root_cause_hint} ({hint_reason})")
print('summary_json=outputs/exp_B123_regression_probe_summary.json')
PYEOF

echo "=== run_exp_B123_regression_probe.sh 結束：$(date) ===" | tee -a "$LOGFILE"
