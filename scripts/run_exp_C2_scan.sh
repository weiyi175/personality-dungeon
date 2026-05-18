#!/usr/bin/env bash
# =============================================================================
# run_exp_C2_scan.sh — C2 threshold_ab 參數掃描（theta, a_hi, b_hi）
#
# 目的：找出相對 C2-base（matrix_ab）不 downlift 的參數區間
# 輸出：
#   outputs/c2_scan/c2_threshold_scan_summary.json
#   outputs/c2_scan/c2_threshold_scan_summary.csv
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=./venv/bin/python

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
OUT_ROOT = ROOT / 'outputs/c2_scan'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

seeds = [45, 47, 49]
players = 300
rounds = 3000

base_dir = OUT_ROOT / 'base_matrix_ab'
base_dir.mkdir(parents=True, exist_ok=True)
for s in seeds:
    out_csv = base_dir / f'seed_{s}.csv'
    cmd = [
        PYTHON_BIN, '-m', 'simulation.run_simulation',
        '--players', str(players), '--rounds', str(rounds), '--seed', str(s),
        '--payoff-mode', 'matrix_ab', '--a', '1.0', '--b', '0.9', '--matrix-cross-coupling', '0.20',
        '--evolution-mode', 'mean_field', '--payoff-lag', '1', '--memory-kernel', '3',
        '--selection-strength', '0.06', '--init-bias', '0.12', '--popularity-mode', 'expected',
        '--out', str(out_csv),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def eval_dir(csv_dir: pathlib.Path):
    rows = []
    for s in seeds:
        p = csv_dir / f'seed_{s}.csv'
        with open(p) as f:
            data = list(csv.DictReader(f))
        series = {st: [float(r[f'p_{st}']) for r in data] for st in ['aggressive','defensive','balanced']}
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
        rows.append((int(cyc.level), s3))
    mean_s3 = sum(r[1] for r in rows) / len(rows)
    l3_count = sum(1 for lv, _ in rows if lv == 3)
    return mean_s3, l3_count

base_mean_s3, base_l3 = eval_dir(base_dir)

theta_values = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
a_hi_values = [1.0, 1.1, 1.2, 1.3, 1.4]
b_hi_values = [0.8, 0.9, 1.0, 1.1]

results = []
for theta, a_hi, b_hi in itertools.product(theta_values, a_hi_values, b_hi_values):
    run_dir = OUT_ROOT / f'theta_{theta:.2f}_ahi_{a_hi:.1f}_bhi_{b_hi:.1f}'
    run_dir.mkdir(parents=True, exist_ok=True)

    ok = True
    for s in seeds:
        out_csv = run_dir / f'seed_{s}.csv'
        cmd = [
            PYTHON_BIN, '-m', 'simulation.run_simulation',
            '--players', str(players), '--rounds', str(rounds), '--seed', str(s),
            '--payoff-mode', 'threshold_ab', '--a', '1.0', '--b', '0.9',
            '--threshold-theta', f'{theta:.2f}', '--threshold-a-hi', f'{a_hi:.1f}', '--threshold-b-hi', f'{b_hi:.1f}',
            '--evolution-mode', 'mean_field', '--payoff-lag', '1', '--memory-kernel', '3',
            '--selection-strength', '0.06', '--init-bias', '0.12', '--popularity-mode', 'expected',
            '--out', str(out_csv),
        ]
        try:
            subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            ok = False
            break

    if not ok:
        results.append({'theta': theta, 'a_hi': a_hi, 'b_hi': b_hi, 'status': 'ERROR'})
        continue

    mean_s3, l3_count = eval_dir(run_dir)
    delta = mean_s3 - base_mean_s3
    rel = abs(delta) / max(abs(base_mean_s3), 1e-6)
    if delta >= 0.015:
        trend = 'UPLIFT'
    elif delta <= -0.015:
        trend = 'DOWNLIFT'
    else:
        trend = 'NEUTRAL'
    non_downlift = delta > -0.015

    results.append({
        'theta': theta,
        'a_hi': a_hi,
        'b_hi': b_hi,
        'base_mean_s3': base_mean_s3,
        'mean_s3': mean_s3,
        'delta': delta,
        'rel': rel,
        'l3_count': l3_count,
        'trend': trend,
        'non_downlift': non_downlift,
    })

results_sorted = sorted(results, key=lambda x: (x.get('status') == 'ERROR', -x.get('mean_s3', -1e9), abs(x.get('delta', 999))))

json_path = OUT_ROOT / 'c2_threshold_scan_summary.json'
with open(json_path, 'w') as f:
    json.dump({
        'base_mean_s3': base_mean_s3,
        'base_l3_count': base_l3,
        'scan_space': {'theta': theta_values, 'a_hi': a_hi_values, 'b_hi': b_hi_values},
        'n_combinations': len(theta_values) * len(a_hi_values) * len(b_hi_values),
        'results': results_sorted,
    }, f, ensure_ascii=False, indent=2)

csv_path = OUT_ROOT / 'c2_threshold_scan_summary.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['theta','a_hi','b_hi','base_mean_s3','mean_s3','delta','rel','l3_count','trend','non_downlift'])
    for r in results_sorted:
        if r.get('status') == 'ERROR':
            w.writerow([r['theta'], r['a_hi'], r['b_hi'], '', '', '', '', '', 'ERROR', ''])
        else:
            w.writerow([r['theta'], r['a_hi'], r['b_hi'], r['base_mean_s3'], r['mean_s3'], r['delta'], r['rel'], r['l3_count'], r['trend'], r['non_downlift']])

non_down = [r for r in results_sorted if r.get('non_downlift')]
print(f'BASE mean_s3={base_mean_s3:.6f}, l3_count={base_l3}')
print(f'SCAN done: {len(results_sorted)} combos, non_downlift={len(non_down)}')
print(f'WROTE {json_path}')
print(f'WROTE {csv_path}')
PY
