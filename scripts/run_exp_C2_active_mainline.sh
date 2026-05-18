#!/usr/bin/env bash
# =============================================================================
# run_exp_C2_active_mainline.sh — C2-active 主線（保守區）
#
# 參數區（固定）：
#   a_hi  ∈ {1.2, 1.3}
#   b_hi  = 1.0
#   theta ∈ {0.35, 0.40}
#
# 共同設定：
#   seeds=45,47,49 / players=300 / rounds=3000 / mean_field / payoff_lag=1
#   memory_kernel=3 / selection_strength=0.06 / init_bias=0.12
#
# 產物：
#   outputs/exp_C2_active_mainline/*
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=./venv/bin/python
OUT_ROOT="outputs/exp_C2_active_mainline"
LOGFILE="${OUT_ROOT}/run.log"
mkdir -p "$OUT_ROOT"

echo "=== run_exp_C2_active_mainline.sh 開始：$(date) ===" | tee "$LOGFILE"

$PYTHON - <<'PY'
import csv
import itertools
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path.cwd()
sys.path.insert(0, str(ROOT))

from analysis.cycle_metrics import classify_cycle_level

PYTHON_BIN = str(ROOT / 'venv/bin/python')
OUT_ROOT = ROOT / 'outputs/exp_C2_active_mainline'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

seeds = [45, 47, 49]
players = 300
rounds = 3000

# C2-base (matrix_ab)
base_dir = OUT_ROOT / 'base_matrix_ab'
base_dir.mkdir(parents=True, exist_ok=True)
for s in seeds:
    cmd = [
        PYTHON_BIN, '-m', 'simulation.run_simulation',
        '--players', str(players), '--rounds', str(rounds), '--seed', str(s),
        '--payoff-mode', 'matrix_ab', '--a', '1.0', '--b', '0.9', '--matrix-cross-coupling', '0.20',
        '--evolution-mode', 'mean_field', '--payoff-lag', '1', '--memory-kernel', '3',
        '--selection-strength', '0.06', '--init-bias', '0.12', '--popularity-mode', 'expected',
        '--out', str(base_dir / f'seed_{s}.csv'),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)

def eval_dir(csv_dir: pathlib.Path):
    vals = []
    for s in seeds:
        p = csv_dir / f'seed_{s}.csv'
        with open(p) as f:
            rows = list(csv.DictReader(f))
        series = {st: [float(r[f'p_{st}']) for r in rows] for st in ['aggressive', 'defensive', 'balanced']}
        cyc = classify_cycle_level(
            series,
            burn_in=1000,
            tail=1000,
            amplitude_threshold=0.02,
            corr_threshold=0.09,
            eta=0.55,
            stage3_method='turning',
            phase_smoothing=1,
            min_lag=2,
            max_lag=500,
        )
        s3 = float(cyc.stage3.score) if cyc.stage3 else 0.0
        vals.append((int(cyc.level), s3))
    mean_s3 = sum(v[1] for v in vals) / len(vals)
    l3_count = sum(1 for lv, _ in vals if lv == 3)
    return mean_s3, l3_count

base_mean_s3, base_l3 = eval_dir(base_dir)

# 保守主線候選
theta_values = [0.35, 0.40]
a_hi_values = [1.2, 1.3]
b_hi = 1.0

results = []
for theta, a_hi in itertools.product(theta_values, a_hi_values):
    run_dir = OUT_ROOT / f'theta_{theta:.2f}_ahi_{a_hi:.1f}_bhi_{b_hi:.1f}'
    run_dir.mkdir(parents=True, exist_ok=True)

    for s in seeds:
        cmd = [
            PYTHON_BIN, '-m', 'simulation.run_simulation',
            '--players', str(players), '--rounds', str(rounds), '--seed', str(s),
            '--payoff-mode', 'threshold_ab', '--a', '1.0', '--b', '0.9',
            '--threshold-theta', f'{theta:.2f}', '--threshold-a-hi', f'{a_hi:.1f}', '--threshold-b-hi', f'{b_hi:.1f}',
            '--evolution-mode', 'mean_field', '--payoff-lag', '1', '--memory-kernel', '3',
            '--selection-strength', '0.06', '--init-bias', '0.12', '--popularity-mode', 'expected',
            '--out', str(run_dir / f'seed_{s}.csv'),
        ]
        subprocess.run(cmd, check=True, cwd=ROOT)

    mean_s3, l3_count = eval_dir(run_dir)
    delta = mean_s3 - base_mean_s3
    rel = abs(delta) / max(abs(base_mean_s3), 1e-6)
    if delta >= 0.015:
        trend = 'UPLIFT'
    elif delta <= -0.015:
        trend = 'DOWNLIFT'
    else:
        trend = 'NEUTRAL'

    results.append({
        'theta': theta,
        'a_hi': a_hi,
        'b_hi': b_hi,
        'base_mean_s3': base_mean_s3,
        'base_l3_count': base_l3,
        'mean_s3': mean_s3,
        'l3_count': l3_count,
        'delta': delta,
        'rel': rel,
        'trend': trend,
    })

# 排序：mean_s3 高者優先，再看 |delta| 最小
results = sorted(results, key=lambda r: (-r['mean_s3'], abs(r['delta']), r['theta'], r['a_hi']))

summary = {
    'phase': 'C2-active-mainline',
    'comparison_mode': 'paired_same_module',
    'search_space': {
        'theta': theta_values,
        'a_hi': a_hi_values,
        'b_hi': [b_hi],
    },
    'base': {
        'mean_s3': base_mean_s3,
        'l3_count': base_l3,
    },
    'results': results,
    'recommended': results[0] if results else None,
}

json_path = OUT_ROOT / 'summary.json'
with open(json_path, 'w') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

csv_path = OUT_ROOT / 'summary.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['theta', 'a_hi', 'b_hi', 'base_mean_s3', 'mean_s3', 'delta', 'rel', 'base_l3_count', 'l3_count', 'trend'])
    for r in results:
        w.writerow([r['theta'], r['a_hi'], r['b_hi'], r['base_mean_s3'], r['mean_s3'], r['delta'], r['rel'], r['base_l3_count'], r['l3_count'], r['trend']])

print(f'BASE mean_s3={base_mean_s3:.6f}, l3_count={base_l3}')
for r in results:
    print(
        f"theta={r['theta']:.2f} a_hi={r['a_hi']:.1f} b_hi={r['b_hi']:.1f} "
        f"mean_s3={r['mean_s3']:.6f} delta={r['delta']:+.6f} l3={r['l3_count']} trend={r['trend']}"
    )
print(f'WROTE {json_path}')
print(f'WROTE {csv_path}')
PY

echo "=== run_exp_C2_active_mainline.sh 結束：$(date) ===" | tee -a "$LOGFILE"
