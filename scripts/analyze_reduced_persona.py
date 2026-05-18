"""Reduced-persona timeseries diagnostic analysis.

Extracts max/min survival seeds, analyzes collapse signatures, and prepares SDD record.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import signal
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Load summary
summary_path = Path('/home/user/personality-dungeon/outputs/actionD_reduced_persona_scan/reduced_persona_scan_summary.json')
with open(summary_path) as f:
    summary = json.load(f)

rows = summary['rows']
reduced3_rows = [r for r in rows if r['condition'] == 'reduced3']

# Find max/min dropout_round
valid_dropouts = [(r['seed'], r['dropout_round']) for r in reduced3_rows if r['dropout_round'] is not None]
valid_dropouts.sort(key=lambda x: x[1], reverse=True)

print('Reduced-3 seeds with valid dropout times:')
for seed, dropout in valid_dropouts:
    print(f'  seed {seed}: dropout_round={dropout}')

max_seed, max_dropout = valid_dropouts[0] if valid_dropouts else (None, None)
min_seed, min_dropout = valid_dropouts[-1] if valid_dropouts else (None, None)

print(f'\nMax survival: seed {max_seed} (dropout={max_dropout})')
print(f'Min survival: seed {min_seed} (dropout={min_dropout})')

# Load CSV data
def load_seed_csv(seed, condition='reduced3'):
    if condition == 'reduced3':
        csv_path = '/home/user/personality-dungeon/outputs/actionD_reduced_persona_scan/reduced3_assertiveness_stability_seeking_curiosity'
    else:
        csv_path = '/home/user/personality-dungeon/outputs/actionD_reduced_persona_scan/control'
    return pd.read_csv(f'{csv_path}/seed{seed}.csv')

df_max = load_seed_csv(max_seed)
df_min = load_seed_csv(min_seed)

print(f'\nLoaded CSV for seed {max_seed}: {len(df_max)} rounds')
print(f'Loaded CSV for seed {min_seed}: {len(df_min)} rounds')

# === PLOT 1: Timeseries Comparison ===
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Max seed: strategy proportions
ax = axes[0, 0]
ax.plot(df_max['round'], df_max['p_aggressive'], label='aggressive', alpha=0.7)
ax.plot(df_max['round'], df_max['p_defensive'], label='defensive', alpha=0.7)
ax.plot(df_max['round'], df_max['p_balanced'], label='balanced', alpha=0.7)
ax.axvline(max_dropout, color='red', linestyle='--', alpha=0.5, label=f'dropout={max_dropout}')
ax.set_title(f'Max Survival: seed {max_seed} (dropout={max_dropout})')
ax.set_ylabel('Strategy Proportion')
ax.legend(loc='best', fontsize=8)
ax.grid(alpha=0.3)

# Min seed: strategy proportions
ax = axes[1, 0]
ax.plot(df_min['round'], df_min['p_aggressive'], label='aggressive', alpha=0.7)
ax.plot(df_min['round'], df_min['p_defensive'], label='defensive', alpha=0.7)
ax.plot(df_min['round'], df_min['p_balanced'], label='balanced', alpha=0.7)
ax.axvline(min_dropout, color='red', linestyle='--', alpha=0.5, label=f'dropout={min_dropout}')
ax.set_title(f'Min Survival: seed {min_seed} (dropout={min_dropout})')
ax.set_ylabel('Strategy Proportion')
ax.set_xlabel('Round')
ax.legend(loc='best', fontsize=8)
ax.grid(alpha=0.3)

# Amplitude comparison
ax = axes[0, 1]
max_amp = np.sqrt(df_max['p_aggressive']**2 + df_max['p_defensive']**2 + df_max['p_balanced']**2)
min_amp = np.sqrt(df_min['p_aggressive']**2 + df_min['p_defensive']**2 + df_min['p_balanced']**2)
ax.plot(df_max['round'], max_amp, label=f'seed {max_seed}', alpha=0.7, linewidth=1)
ax.plot(df_min['round'], min_amp, label=f'seed {min_seed}', alpha=0.7, linewidth=1)
ax.set_title('System Amplitude (L2 norm)')
ax.set_ylabel('Amplitude')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Entropy
ax = axes[1, 1]
def entropy(p_a, p_d, p_b):
    eps = 1e-9
    return -p_a*np.log(p_a+eps) - p_d*np.log(p_d+eps) - p_b*np.log(p_b+eps)

max_ent = entropy(df_max['p_aggressive'], df_max['p_defensive'], df_max['p_balanced'])
min_ent = entropy(df_min['p_aggressive'], df_min['p_defensive'], df_min['p_balanced'])
ax.plot(df_max['round'], max_ent, label=f'seed {max_seed}', alpha=0.7, linewidth=1)
ax.plot(df_min['round'], min_ent, label=f'seed {min_seed}', alpha=0.7, linewidth=1)
ax.set_title('System Entropy (higher=more mixed)')
ax.set_ylabel('Entropy')
ax.set_xlabel('Round')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/personality-dungeon/outputs/actionD_reduced_persona_scan/timeseries_comparison.png', dpi=100)
print('\nSaved: timeseries_comparison.png')
plt.close()

# === PLOT 2: Volatility ===
def rolling_volatility(df, window=100):
    p_a = df['p_aggressive'].values
    volatility = np.array([np.std(p_a[max(0, i-window):i]) for i in range(len(p_a))])
    return volatility

max_vol = rolling_volatility(df_max, window=100)
min_vol = rolling_volatility(df_min, window=100)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

ax = axes[0]
ax.plot(df_max['round'], max_vol, linewidth=1, alpha=0.7, color='blue')
ax.axvline(max_dropout, color='red', linestyle='--', alpha=0.5, label=f'dropout={max_dropout}')
ax.set_title(f'Volatility: Max Survival (seed {max_seed})')
ax.set_xlabel('Round')
ax.set_ylabel('Rolling Std (window=100)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(df_min['round'], min_vol, linewidth=1, alpha=0.7, color='orange')
ax.axvline(min_dropout, color='red', linestyle='--', alpha=0.5, label=f'dropout={min_dropout}')
ax.set_title(f'Volatility: Min Survival (seed {min_seed})')
ax.set_xlabel('Round')
ax.set_ylabel('Rolling Std (window=100)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/personality-dungeon/outputs/actionD_reduced_persona_scan/volatility_comparison.png', dpi=100)
print('Saved: volatility_comparison.png')
plt.close()

print(f'\nMax seed (seed {max_seed}, dropout={max_dropout})')
print(f'  Pre-dropout volatility (rounds {max(0, max_dropout-200)}:{max_dropout}): {np.mean(max_vol[max(0, max_dropout-200):max_dropout]):.6f}')
print(f'\nMin seed (seed {min_seed}, dropout={min_dropout})')
print(f'  Pre-dropout volatility (rounds {max(0, min_dropout-200)}:{min_dropout}): {np.mean(min_vol[max(0, min_dropout-200):min_dropout]):.6f}')

# === SPECTRAL ANALYSIS ===
def fft_analysis(signal_arr, dropout_round=None):
    if dropout_round is not None:
        signal_arr = signal_arr[:dropout_round]
    
    signal_arr = signal.detrend(signal_arr)
    window = signal.windows.hann(len(signal_arr))
    signal_windowed = signal_arr * window
    
    freqs = np.fft.fftfreq(len(signal_windowed))
    fft_vals = np.abs(np.fft.fft(signal_windowed)) ** 2
    
    n = len(freqs)
    low_freq_energy = np.sum(fft_vals[:n//4])
    high_freq_energy = np.sum(fft_vals[n//4:])
    
    return freqs[:n//2], fft_vals[:n//2], low_freq_energy, high_freq_energy

max_freqs, max_fft, max_low, max_high = fft_analysis(df_max['p_aggressive'].values, max_dropout)
min_freqs, min_fft, min_low, min_high = fft_analysis(df_min['p_aggressive'].values, min_dropout)

print(f'\n--- SPECTRAL ANALYSIS ---')
print(f'Max seed (seed {max_seed}): Low-freq energy={max_low:.2e}, High-freq energy={max_high:.2e}')
print(f'  Ratio (low/high)={max_low/(max_high+1e-10):.3f}')
print(f'Min seed (seed {min_seed}): Low-freq energy={min_low:.2e}, High-freq energy={min_high:.2e}')
print(f'  Ratio (low/high)={min_low/(min_high+1e-10):.3f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

ax = axes[0]
ax.semilogy(max_freqs, max_fft, linewidth=1, alpha=0.7, color='blue')
ax.set_title(f'Power Spectrum: Max Survival (seed {max_seed})')
ax.set_xlabel('Frequency')
ax.set_ylabel('Power')
ax.grid(alpha=0.3, which='both')

ax = axes[1]
ax.semilogy(min_freqs, min_fft, linewidth=1, alpha=0.7, color='orange')
ax.set_title(f'Power Spectrum: Min Survival (seed {min_seed})')
ax.set_xlabel('Frequency')
ax.set_ylabel('Power')
ax.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('/home/user/personality-dungeon/outputs/actionD_reduced_persona_scan/fft_comparison.png', dpi=100)
print('Saved: fft_comparison.png')
plt.close()

# === COLLAPSE SIGNATURE ===
max_ratio = max_low / (max_high + 1e-10)
min_ratio = min_low / (min_high + 1e-10)

print('\n' + '='*60)
print('COLLAPSE SIGNATURE ANALYSIS')
print('='*60)

print(f'\nMax Survival (seed {max_seed}, dropout={max_dropout})')
print(f'  Low-freq / High-freq ratio: {max_ratio:.3f}')
if max_ratio > 1.0:
    max_signature = 'ENERGY DEPLETION (low-freq dominated)'
    print(f'  -> {max_signature}')
else:
    max_signature = 'NOISE BURST (high-freq dominated)'
    print(f'  -> {max_signature}')

print(f'\nMin Survival (seed {min_seed}, dropout={min_dropout})')
print(f'  Low-freq / High-freq ratio: {min_ratio:.3f}')
if min_ratio > 1.0:
    min_signature = 'ENERGY DEPLETION (low-freq dominated)'
    print(f'  -> {min_signature}')
else:
    min_signature = 'NOISE BURST (high-freq dominated)'
    print(f'  -> {min_signature}')

print(f'\nDifference in dropout timing: {abs(max_dropout - min_dropout)} rounds')
print(f'Difference in frequency signature ratio: {abs(max_ratio - min_ratio):.3f}')

if max_dropout == min_dropout:
    print('\n*** KEY FINDING: Both seeds collapse at same time despite different frequencies')
    print('    -> Suggests system has hard limit (not individual seed variance)')
else:
    survival_diff_pct = 100 * abs(max_dropout - min_dropout) / max_dropout
    print(f'\n*** Survival variation: {survival_diff_pct:.1f}% difference')
    if survival_diff_pct < 5:
        print('    -> Variation is minimal: suggests deterministic collapse mechanism')
    else:
        print('    -> Moderate variation: suggests seed-dependent randomness')

print('\n' + '='*60)

# === SDD RECORD ===
sdd_finding = f"""## W3.6 - Reduced-Persona Empirical Verdict (actionD)

**Hypothesis**: Reducing 9-persona to 3-core traits (assertiveness, stability_seeking, curiosity) would stabilize B1/B2 by removing noise.

**Scope**: 6 seeds × 6000 rounds, memory_kernel=3, players=300

**Outcome (NEGATIVE)**:
- Control baseline: dropout_median=4820, short_s3_mean=0.5334
- Reduced-3: dropout_median=4820, short_s3_mean=0.5334 (shift=+0)
- No measurable lifespan extension; short_l3_count=0 (relight gate failed)

**Collapse Signature Analysis**:
- Max-survival seed {max_seed}: dropout={max_dropout}, low/high freq ratio={max_ratio:.3f} -> {max_signature}
- Min-survival seed {min_seed}: dropout={min_dropout}, low/high freq ratio={min_ratio:.3f} -> {min_signature}
- Variation: {abs(max_dropout-min_dropout)} rounds ({100*abs(max_dropout-min_dropout)/max_dropout:.1f}%)

**Implication**:
1. Persona dimensionality is NOT the root cause of B1/B2 instability.
2. The collapse mechanism appears {("deterministic (hard limit)" if abs(max_dropout-min_dropout) < 100 else "stochastic")}.
3. Patterns across all interventions (Option2, Option3, Option4) converge:
   - Pulses (Option2): failed to extend lifespan (+120 << +200 gate)
   - Projections (Option3): ephemeral signal; 10k confirm was negative
   - Reduced-persona (Option4): zero effect on dropout timing
4. **Structural hypothesis**: The collapse is rooted in the payoff geometry (a=1.0, b=0.9, c=0.20) and strategy dynamics, not event injection or persona noise.

**Recommendation**: Proceed to payoff matrix sweeps or hybrid topology experiments to identify the true stability boundary."""

print(sdd_finding)
print(f"\n\n📋 SDD Record (ready for SDD.md):")
print("="*60)
print(sdd_finding)
print("="*60)

# Save to file for easy copy
with open('/home/user/personality-dungeon/outputs/actionD_reduced_persona_scan/sdd_record.txt', 'w') as f:
    f.write(sdd_finding)
print('\nSaved SDD record to: outputs/actionD_reduced_persona_scan/sdd_record.txt')
