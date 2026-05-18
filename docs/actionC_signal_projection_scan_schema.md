# Option 3 Output Schema - Signal Projection Matrix Scan

This document defines the output contract for:
- `outputs/actionC_signal_projection_scan/signal_projection_scan_summary.json`
- `outputs/actionC_signal_projection_scan/signal_projection_scan_onepager.md`

## Summary JSON structure

Top-level keys:
- `generated_at` (string, ISO-8601 timestamp)
- `scope` (object)
- `control` (object)
- `grid` (array of objects)
- `rows` (array of objects)

### `scope`
- `seeds` (array[int])
- `rounds` (int)
- `players` (int)
- `selection_strength` (float)
- `init_bias` (float)
- `memory_kernel` (int)
- `projection_matrices` (array[string])
- `projection_gains` (array[float])
- `activation_thetas` (array[float])

### `control`
- `n` (int)
- `short_l3_count` (int)
- `long_l3_count` (int)
- `short_s3_mean` (float)
- `long_s3_mean` (float)
- `dropout_median` (int|null)
- `dropout_count` (int)
- `projection_activation_count_mean` (float)
- `projected_dominant_switch_count_mean` (float)

### `grid[]` (one record per matrix x gain x theta cell)
- `condition` (string)
- `matrix_label` (string)
- `projection_gain` (float)
- `activation_theta` (float)
- `n` (int)
- `short_l3_count` (int)
- `long_l3_count` (int)
- `short_s3_mean` (float)
- `long_s3_mean` (float)
- `dropout_median` (int|null)
- `dropout_count` (int)
- `projection_activation_count_mean` (float)
- `projected_dominant_switch_count_mean` (float)
- `dropout_shift_vs_control` (int|null)
- `short_s3_shift_vs_control` (float)

### `rows[]` (seed-level)
- `condition` (string)
- `seed` (int)
- `matrix_label` (string|null)
- `projection_gain` (float|null)
- `activation_theta` (float|null)
- `window_eval` (object)
- `dropout_round` (int|null)
- `projection_activation_count` (int)
- `first_projection_activation_round` (int|null)
- `projected_dominant_switch_count` (int)
- `csv` (string, relative path)

### `window_eval`
- `burn1000_tail1000` (object)
- `burn2000_tail2000` (object)

Each window object includes:
- `level` (int)
- `stage3_score` (float)

## Onepager markdown table columns

Main table columns:
- `matrix`, `gain`, `theta`
- `short L3`, `long L3`
- `short_s3`, `long_s3`
- `dropout_med`, `dropout_shift`, `s3_shift`
- `act_count_mean`, `switch_mean`

## Decision heuristics used in onepager

- lifespan signal: `dropout_shift_vs_control >= +200`
- relight signal: `short_l3_count >= 2 and short_s3_mean >= 0.55`
- structural move: lifespan signal with `projection_activation_count_mean > 0`
