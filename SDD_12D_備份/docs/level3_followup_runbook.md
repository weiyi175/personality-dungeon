# Level 3 Follow-up Runbook

本文件把目前建議的後續優化方向整理成可直接執行的研究 runbook。
目標不是一次把所有網格炸完，而是用最小成本先確認突破可信，再決定是否往更高分數推進。

> 狀態註記（2026-04-02）：第 1~9 節與第 13~16 節屬於較早期的 cross-term / Batch A~F 執行規劃與進度盤點，現已因 W2.1R ~ W3.3 的正式 closure 而過時，只保留存檔用途，不再作為現行主線執行指引。現行正式結論與 freeze 決策請優先閱讀第 10~12 節。

## 快速導覽

1. 現行正式結論與 freeze 決策：看第 10~12 節
2. 現行後續主線與新機制方向：看第 17~18 節
3. **打破平均化瓶頸的結構性提案**：看第 19 節
4. 歷史存檔與舊版執行計畫：看第 1~9 節與第 13~16 節
5. 通用閱讀前提與診斷原則：看第 0 節

## 0. 原則

1. 先證明 breakthrough 不是 artifact，再追更高 score。
2. 先做 reviewer 最容易質疑的項目：回歸、seed 穩健性、長時間穩定性。
3. 所有 Python 指令一律使用 `./venv/bin/python`。
4. 所有診斷沿用目前鎖定設定：
   - `series=p`
   - `burn_in=2400`
   - `tail=1000`
   - `amplitude_threshold=0.02`
   - `corr_threshold=0.09`
   - `eta=0.55`
   - `stage3_method='turning'`
   - `phase_smoothing=1`

## 1. 舊版 Batch A~F 執行順序（已過時，僅供存檔）

| 優先級 | 批次 | 名稱 | 調整後內容 | 條件數 | 預期目的 | 通過標準 |
|---|---|---|---|---:|---|---|
| P0 | Batch A | 控制組 | `cross=0.0` @ `seed=45` | 1 | 驗證 cross-term 是 causal knob | `score` 回到 `0.546–0.548`，且為 Level 2 |
| P1 | Batch B | 多 seed | `seed ∈ {45,47,49,51,53,55}`，固定 `cross=0.16` | 6 | 判斷是否只是單 seed 神蹟 | `>= 3/6` 達到 Level 3，或平均 `score > 0.552` |
| P2 | Batch C-light | 甜區輕量細修 | `c ∈ {0.14,0.155,0.16,0.165,0.18}` @ `seed=45` | 5 | 看甜區形狀 | 峰值 `>= 0.555` 且非單點突出 |
| P3 | Batch D-light | 10000-round 驗證 | `rounds=10000`，`cross=最佳值`，`seed=45` | 1 | 確認不是短暫暫態 | `tail=1000` 與 `tail=3000` 都維持 Level 3 |
| P4 | Batch E-core | 機制診斷 | phase portrait + `turn_strength` 移動平均 | - | 建立機制故事 | 軌道更封閉、方向更一致 |
| P5 | Batch F-mini | 小型 knob 探索 | 只測 `3–4` 組最有希望條件 | 3–4 | 試探是否還能上推 | 若提升 `> 0.01` 再擴大 |

### 1.1 舊版一句話版策略（已過時）

先用 `1 + 6 + 5 + 1` 的最小可信驗證路徑，把 breakthrough 是否為 artifact 釘死，再決定要不要追 `0.57+`。

### 1.2 舊版更激進的省力方案（三連擊，已過時）

如果運算資源更緊，可先縮成：

1. Batch A：`cross=0.0 @ seed=45`
2. Batch B-light：`cross=0.16 @ seed={45,48,51}`
3. Batch C+D 合併：`cross ∈ {0.155,0.16,0.165}`，`rounds=10000`，`seed=45`

判讀規則：

1. 三連擊都通過，再進入機制分析與小型 knob 探索。
2. 任何一步卡住，就把 paper 敘事降級為「在特定有限條件下可觀察到的暫態強化現象」，主打機制洞察而非極致分數。

## 2. 舊版 Batch 共用環境變數（已過時，僅供存檔）

先在 repo root 執行：

```bash
cd /home/user/personality-dungeon
export PYTHON_BIN=./venv/bin/python
export EVENTS_JSON=docs/personality_dungeon_v1/02_event_templates_v1.json
```

## 3. 舊版 Batch 共用診斷命令（已過時，僅供存檔）

所有批次完成後，建議用下面這段診斷輸出統一比較 `Level / score / turn_strength / gap`。

```bash
$PYTHON_BIN - <<'PY'
import csv
import sys
from pathlib import Path
from analysis.cycle_metrics import classify_cycle_level

for csv_path in sys.argv[1:]:
    path = Path(csv_path)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    series = {
        'aggressive': [float(r['p_aggressive']) for r in rows],
        'defensive': [float(r['p_defensive']) for r in rows],
        'balanced': [float(r['p_balanced']) for r in rows],
    }
    result = classify_cycle_level(
        series,
        burn_in=2400,
        tail=1000,
        amplitude_threshold=0.02,
        corr_threshold=0.09,
        eta=0.55,
        stage3_method='turning',
        phase_smoothing=1,
        min_lag=2,
        max_lag=500,
    )
    print(
        f"{path.name}\tlevel={result.level}\t"
        f"score={result.stage3.score:.4f}\t"
        f"turn={result.stage3.turn_strength:.6f}\t"
        f"gap={0.55 - result.stage3.score:.4f}"
    )
PY
```

用法範例：

```bash
$PYTHON_BIN diag.py outputs/foo.csv outputs/bar.csv
```

若不想另外存檔，可把 `diag.py` 換成 here-doc 版本直接貼上執行。

## 4. 舊版 Batch A：控制組回歸測試（已過時，僅供存檔）

### 4.1 目的

驗證 `matrix_cross_coupling=0.0` 時，結果應回到既有 sampled + adaptive-payoff-v2 基線，
也就是大約 `0.546–0.548`，仍為 Level 2。

### 4.2 執行命令

```bash
$PYTHON_BIN -m simulation.run_simulation \
  --enable-events --events-json "$EVENTS_JSON" \
  --popularity-mode sampled \
  --seed 45 --rounds 8000 --players 300 \
  --payoff-mode matrix_ab --a 0.8 --b 0.9 \
  --matrix-cross-coupling 0.0 \
  --selection-strength 0.06 --init-bias 0.12 \
  --event-failure-threshold 0.72 --event-health-penalty 0.10 \
  --adaptive-payoff-strength 1.5 \
  --payoff-update-interval 400 \
  --adaptive-payoff-target 0.27 \
  --out outputs/control_cross0_seed45_sampled.csv
```

### 4.3 預期輸出檔名

1. `outputs/control_cross0_seed45_sampled.csv`

### 4.4 跑完要看什麼

1. `level` 是否回到 2
2. `score` 是否回到約 `0.5468–0.5479`
3. 與 `outputs/sampled_seed45_apv2_s1p5_i400_crossc0p16.csv` 比較時，cross-term 是否形成明確 uplift

### 4.5 建議診斷命令

```bash
$PYTHON_BIN - <<'PY'
import csv
from pathlib import Path
from analysis.cycle_metrics import classify_cycle_level

for name in [
    'outputs/control_cross0_seed45_sampled.csv',
    'outputs/sampled_seed45_apv2_s1p5_i400_crossc0p16.csv',
]:
    with Path(name).open() as f:
        rows = list(csv.DictReader(f))
    series = {
        'aggressive': [float(r['p_aggressive']) for r in rows],
        'defensive': [float(r['p_defensive']) for r in rows],
        'balanced': [float(r['p_balanced']) for r in rows],
    }
    result = classify_cycle_level(series, burn_in=2400, tail=1000, amplitude_threshold=0.02, corr_threshold=0.09, eta=0.55, stage3_method='turning', phase_smoothing=1, min_lag=2, max_lag=500)
    print(name, result.level, round(result.stage3.score, 4), round(result.stage3.turn_strength, 6))
PY
```

## 5. 舊版 Batch B：Multi-seed Level 3 穩健性（已過時，僅供存檔）

### 5.1 目的

確認 `c=0.16` 的 Level 3 突破是否只屬於 `seed=45`，還是能在較稀疏但仍具代表性的 seeds 上複現。

### 5.2 執行命令

```bash
for seed in 45 47 49 51 53 55; do
  $PYTHON_BIN -m simulation.run_simulation \
    --enable-events --events-json "$EVENTS_JSON" \
    --popularity-mode sampled \
    --seed "$seed" --rounds 8000 --players 300 \
    --payoff-mode matrix_ab --a 0.8 --b 0.9 \
    --matrix-cross-coupling 0.16 \
    --selection-strength 0.06 --init-bias 0.12 \
    --event-failure-threshold 0.72 --event-health-penalty 0.10 \
    --adaptive-payoff-strength 1.5 \
    --payoff-update-interval 400 \
    --adaptive-payoff-target 0.27 \
    --out "outputs/multiseed_crossc0p16_seed${seed}.csv"
done
```

### 5.3 預期輸出檔名

1. `outputs/multiseed_crossc0p16_seed45.csv`
2. `outputs/multiseed_crossc0p16_seed47.csv`
3. `outputs/multiseed_crossc0p16_seed49.csv`
4. `outputs/multiseed_crossc0p16_seed51.csv`
5. `outputs/multiseed_crossc0p16_seed53.csv`
6. `outputs/multiseed_crossc0p16_seed55.csv`

### 5.4 跑完要看什麼

1. `>= 3/6` 是否達到 Level 3
2. 若未達 `3/6`，平均 `score` 是否仍大於 `0.552`
3. 是否存在明顯 seed sensitivity
4. 跑完前三個 seed 就可先看 phase portrait；若方向完全散掉，可提前停止後三個

### 5.5 建議診斷命令

```bash
$PYTHON_BIN - <<'PY'
import csv
from pathlib import Path
from analysis.cycle_metrics import classify_cycle_level

scores = []
hits = 0
for seed in [45, 47, 49, 51, 53, 55]:
    path = Path(f'outputs/multiseed_crossc0p16_seed{seed}.csv')
    with path.open() as f:
        rows = list(csv.DictReader(f))
    series = {
        'aggressive': [float(r['p_aggressive']) for r in rows],
        'defensive': [float(r['p_defensive']) for r in rows],
        'balanced': [float(r['p_balanced']) for r in rows],
    }
    result = classify_cycle_level(series, burn_in=2400, tail=1000, amplitude_threshold=0.02, corr_threshold=0.09, eta=0.55, stage3_method='turning', phase_smoothing=1, min_lag=2, max_lag=500)
    score = float(result.stage3.score)
    scores.append(score)
    hits += int(result.level >= 3)
    print(seed, result.level, round(score, 4), round(result.stage3.turn_strength, 6))
mean_score = sum(scores)/len(scores)
print('Pr(level>=3)=', f'{hits}/{len(scores)}')
print('mean_score=', round(mean_score, 4))
print('pass_by_hits=', hits >= 3)
print('pass_by_mean=', mean_score > 0.552)
PY
```

### 5.6 更激進省力版（Batch B-light）

若只想先做最小 seed 檢查，可先跑：

```bash
for seed in 45 48 51; do
    $PYTHON_BIN -m simulation.run_simulation \
        --enable-events --events-json "$EVENTS_JSON" \
        --popularity-mode sampled \
        --seed "$seed" --rounds 8000 --players 300 \
        --payoff-mode matrix_ab --a 0.8 --b 0.9 \
        --matrix-cross-coupling 0.16 \
        --selection-strength 0.06 --init-bias 0.12 \
        --event-failure-threshold 0.72 --event-health-penalty 0.10 \
        --adaptive-payoff-strength 1.5 \
        --payoff-update-interval 400 \
        --adaptive-payoff-target 0.27 \
        --out "outputs/multiseed_light_crossc0p16_seed${seed}.csv"
done
```

## 6. 舊版 Batch C-light：Cross-coupling 甜區細修（已過時，僅供存檔）

### 6.1 目的

確認 `c=0.16` 是窄峰還是平台，避免把最佳值誤判成單點噪音。

### 6.2 執行命令

```bash
for c in 0.14 0.155 0.16 0.165 0.18; do
  ctag=$(printf '%s' "$c" | tr '.' 'p')
  $PYTHON_BIN -m simulation.run_simulation \
    --enable-events --events-json "$EVENTS_JSON" \
    --popularity-mode sampled \
    --seed 45 --rounds 8000 --players 300 \
    --payoff-mode matrix_ab --a 0.8 --b 0.9 \
    --matrix-cross-coupling "$c" \
    --selection-strength 0.06 --init-bias 0.12 \
    --event-failure-threshold 0.72 --event-health-penalty 0.10 \
    --adaptive-payoff-strength 1.5 \
    --payoff-update-interval 400 \
    --adaptive-payoff-target 0.27 \
    --out "outputs/refine_cross_seed45_c${ctag}.csv"
done
```

### 6.3 預期輸出檔名

1. `outputs/refine_cross_seed45_c0p14.csv`
2. `outputs/refine_cross_seed45_c0p155.csv`
3. `outputs/refine_cross_seed45_c0p16.csv`
4. `outputs/refine_cross_seed45_c0p165.csv`
5. `outputs/refine_cross_seed45_c0p18.csv`

### 6.4 跑完要看什麼

1. 峰值是否 `>= 0.555`
2. `0.155–0.165` 是否形成平台
3. 是否不是只有單點明顯突出

### 6.5 建議診斷命令

```bash
$PYTHON_BIN - <<'PY'
import csv
from pathlib import Path
from analysis.cycle_metrics import classify_cycle_level

results = []
for ctag, c in [
    ('0p14', 0.14), ('0p155', 0.155), ('0p16', 0.16), ('0p165', 0.165), ('0p18', 0.18),
]:
    path = Path(f'outputs/refine_cross_seed45_c{ctag}.csv')
    with path.open() as f:
        rows = list(csv.DictReader(f))
    series = {
        'aggressive': [float(r['p_aggressive']) for r in rows],
        'defensive': [float(r['p_defensive']) for r in rows],
        'balanced': [float(r['p_balanced']) for r in rows],
    }
    result = classify_cycle_level(series, burn_in=2400, tail=1000, amplitude_threshold=0.02, corr_threshold=0.09, eta=0.55, stage3_method='turning', phase_smoothing=1, min_lag=2, max_lag=500)
    score = round(result.stage3.score, 4)
    results.append((c, score, result.level))
    print(c, result.level, score, round(result.stage3.turn_strength, 6))

peak = max(score for _, score, _ in results)
near_peak = [c for c, score, _ in results if peak - score <= 0.003]
print('peak>=0.555 =', peak >= 0.555)
print('near_peak_points =', near_peak)
PY
```

## 7. 舊版 Batch D-light：10000-round 穩定性（已過時，僅供存檔）

### 7.1 目的

先用單一長跑確認最佳條件不是 8000 rounds 尾端的短期暫態。

### 7.2 執行命令

```bash
$PYTHON_BIN -m simulation.run_simulation \
    --enable-events --events-json "$EVENTS_JSON" \
    --popularity-mode sampled \
    --seed 45 --rounds 10000 --players 300 \
    --payoff-mode matrix_ab --a 0.8 --b 0.9 \
    --matrix-cross-coupling 0.16 \
    --selection-strength 0.06 --init-bias 0.12 \
    --event-failure-threshold 0.72 --event-health-penalty 0.10 \
    --adaptive-payoff-strength 1.5 \
    --payoff-update-interval 400 \
    --adaptive-payoff-target 0.27 \
    --out outputs/longrun_crossc0p16_seed45_r10000.csv
```

### 7.3 預期輸出檔名

1. `outputs/longrun_crossc0p16_seed45_r10000.csv`

### 7.4 跑完要看什麼

1. `tail=1000` 是否維持 Level 3
2. `tail=3000` 是否仍維持 Level 3
3. `score` 是否沒有明顯崩回 `0.54x`

### 7.5 建議診斷命令

```bash
$PYTHON_BIN - <<'PY'
import csv
from pathlib import Path
from analysis.cycle_metrics import classify_cycle_level

path = Path('outputs/longrun_crossc0p16_seed45_r10000.csv')
with path.open() as f:
    rows = list(csv.DictReader(f))
series = {
    'aggressive': [float(r['p_aggressive']) for r in rows],
    'defensive': [float(r['p_defensive']) for r in rows],
    'balanced': [float(r['p_balanced']) for r in rows],
}
for tail in [1000, 3000]:
    result = classify_cycle_level(series, burn_in=max(2400, len(rows)//3), tail=tail, amplitude_threshold=0.02, corr_threshold=0.09, eta=0.55, stage3_method='turning', phase_smoothing=1, min_lag=2, max_lag=500)
    print('tail=', tail, 'level=', result.level, 'score=', round(result.stage3.score, 4), 'turn=', round(result.stage3.turn_strength, 6))
PY
```

## 8. 舊版 Batch E-core：機制診斷（已過時，僅供存檔）

### 8.1 目的

理解 cross-term 如何改變 turning consistency，而不是只看分數是否上升。

### 8.2 推薦比較對照

1. `outputs/control_cross0_seed45_sampled.csv`
2. `outputs/sampled_seed45_apv2_s1p5_i400_crossc0p16.csv`

### 8.3 建議分析項目

1. `p_aggressive / p_defensive / p_balanced` tail overlay
2. phase portrait (`p_aggressive` vs `p_defensive`)
3. `turn_strength` 移動平均
4. `final_risk` histogram

### 8.4 跑完要看什麼

1. cross-term 是否讓 phase portrait 更封閉、更方向一致
2. `turn_strength` 移動平均是否更穩
3. `final_risk` 分布是否從雙峰態變成更利於固定旋轉方向的形狀
4. balanced 是否明顯吸收 aggressive-defensive coexistence 壓力

## 9. 舊版 Batch F-mini：組合 knob 探索（已過時，僅供存檔）

### 9.1 目的

在 breakthrough 已確認可信後，再追求更高 score，例如 `0.57+`。

### 9.2 執行命令

```bash
for combo in \
    "1.5 0.265" \
    "1.5 0.275" \
    "1.8 0.265" \
    "1.8 0.275"; do
    set -- $combo
    s="$1"
    target="$2"
    stag=$(printf '%s' "$s" | tr '.' 'p')
    ttag=$(printf '%s' "$target" | tr '.' 'p')
    $PYTHON_BIN -m simulation.run_simulation \
        --enable-events --events-json "$EVENTS_JSON" \
        --popularity-mode sampled \
        --seed 45 --rounds 8000 --players 300 \
        --payoff-mode matrix_ab --a 0.8 --b 0.9 \
        --matrix-cross-coupling 0.16 \
        --selection-strength 0.06 --init-bias 0.12 \
        --event-failure-threshold 0.72 --event-health-penalty 0.10 \
        --adaptive-payoff-strength "$s" \
        --payoff-update-interval 400 \
        --adaptive-payoff-target "$target" \
        --out "outputs/knobscan_mini_crossc0p16_s${stag}_target${ttag}.csv"
done
```

### 9.3 預期輸出檔名

共 4 個：

1. `outputs/knobscan_mini_crossc0p16_s1p5_target0p265.csv`
2. `outputs/knobscan_mini_crossc0p16_s1p5_target0p275.csv`
3. `outputs/knobscan_mini_crossc0p16_s1p8_target0p265.csv`
4. `outputs/knobscan_mini_crossc0p16_s1p8_target0p275.csv`

### 9.4 跑完要看什麼

1. 是否有條件超過 `0.5611`
2. 是否有提升 `> 0.01`
3. 若沒有明顯提升，就不要擴大掃描

## 10. W2.1R Formal Dry-Run（2026-04-02）

### 10.1 執行條件

修正版正式 dry-run 已依下列 protocol 實跑：

1. `players=400`
2. `rounds_per_life=4000`
3. `total_lives=6`
4. `seeds={45,47,49}`
5. tail 判讀視窗固定為 `life 4..6`
6. `events_json=docs/personality_dungeon_v1/02_event_templates_v1.json`
7. cells：`control(single_life, alpha=0)`, `w2_base(alpha=0.12)`, `w2_strong(alpha=0.22)`

實際輸出：

1. `outputs/w2_episode_formal_dryrun_20260402_control_*`
2. `outputs/w2_episode_formal_dryrun_20260402_rev1_*`

### 10.2 正式結果

整體 decision：`close_w2_1`

關鍵 aggregate 結果：

1. `control`：tail `mean_env_gamma=0.048943`，但 `tail_mean_death_rate=1.000000`，`tail_mean_rounds_completed=26.0`
2. `w2_base`：tail `mean_env_gamma=0.000025`，`tail_level3_seed_count=0`，`tail_mean_death_rate=0.474167`，`tail_mean_rounds_completed=4000.0`
3. `w2_strong`：tail `mean_env_gamma=0.000025`，`tail_level3_seed_count=0`，`tail_mean_death_rate=0.448611`，`tail_mean_rounds_completed=4000.0`

### 10.3 解讀

1. W2 機制確實有啟動：兩個 testament cells 在後半段都能跑滿 4000 rounds，且人格漂移持續累積
2. 但 `life 4..6` 仍完全沒有任何 `Level 3 seed`，因此依鎖定 Closure Gate 正式記為 `close_w2_1`
3. `w2_base` / `w2_strong` 的 tail death rate 約 `44.9%~47.4%`，已明顯低於 control 的 `100%`，但仍高於理想區間 `15%~40%`
4. tail personality centroid 明顯往 `caution / stability_seeking / patience` 偏移，說明目前 W2.1 更像是在建立「防禦式存活收斂」，不是在產生 `Level 3` 相位突破
5. control baseline 仍存在嚴重早死偏差；因此這輪 formal dry-run 的主要價值是確認 W2.1 機制方向，而不是提供乾淨的 matched-control gamma 比較

### 10.4 後續限制

1. 依 W2.1R closure 規則，不再微調 `alpha_testament`
2. 若續做 W2，應直接轉向 W2.2（world carryover + Little Dragon）
3. 若放棄 W2 主線，則直接切到 W3（Stackelberg）

## 11. W3.1 Formal Scout（2026-04-02）

### 11.1 執行條件

W3.1 第一輪正式 scout 已依鎖定 protocol 實跑：

1. `players=300`
2. `rounds=3000`
3. `seeds={45,47,49}`
4. `memory_kernel=3`
5. `selection_strength=0.06`
6. `init_bias=0.12`
7. `events_json=docs/personality_dungeon_v1/02_event_templates_v1.json`
8. 固定 4 個 leader commitments：`control`, `w3_cross_strong`, `w3_edge_tilt`, `w3_commit_push`

實際輸出：

1. `outputs/w3_stackelberg_formal_20260402_summary.tsv`
2. `outputs/w3_stackelberg_formal_20260402_combined.tsv`
3. `outputs/w3_stackelberg_formal_20260402_decision.md`

### 11.2 正式結果

整體 decision：`close_w3_1`

關鍵 aggregate 結果：

1. `control`：`mean_stage3_score=0.515010`，`mean_env_gamma=-0.000066`，`level3_seed_count=0`
2. `w3_cross_strong`：`mean_stage3_score=0.516959`，`mean_env_gamma=-0.000065`，`level3_seed_count=0`，`verdict=weak_positive`
3. `w3_edge_tilt`：`mean_stage3_score=0.513999`，`mean_env_gamma=-0.000066`，`level3_seed_count=0`，`verdict=fail`
4. `w3_commit_push`：`mean_stage3_score=0.518930`，`mean_env_gamma=-0.000065`，`level3_seed_count=0`，`verdict=weak_positive`，且為 `best_commitment`

### 11.3 解讀

1. 這輪沒有任何一個 non-control commitment 產生 `Level 3 seed`，因此依 W3.1 Closure Gate，正式記為 `close_w3_1`
2. `w3_cross_strong` 與 `w3_commit_push` 確實略優於 control，但 uplift 非常小：本質上只是把 `mean_stage3_score` 從 `0.5150` 微幅推到 `0.517~0.519`，`mean_env_gamma` 也只從 `-6.6e-05` 改到約 `-6.5e-05`
3. 所有 cells 的 `mean_cycle_level` 都仍是 `2.0`，`level_counts_json` 也完全相同，表示 leader commitment 目前只是在同一個 Level 2 plateau 上做極小位移，沒有改變吸引子結構
4. `best_commitment=w3_commit_push` 的意思只是在固定 lexicographic 排序下，它是目前 3 個 non-control commitments 中最值得保留的單一候選，不代表 W3.1 有通過 promotion gate
5. 因為 4 個固定 cells 已經把「強一點 cross」、「偏一點 edge」、「兩者同時打開」都測過一次，卻仍完全沒有 Level 3 emergence，所以 W3.1 不值得再做 `a/b/cross` 細網格微調

### 11.4 後續限制

1. 依 W3.1 closure 規則，不再做更多 `a/b/matrix_cross_coupling` 微調
2. 若續做 W3，必須升級到更高階 leader policy，而不是繼續留在固定 commitment cell sweep

## 12. W3.2 Formal Scout（2026-04-02）

### 12.1 執行條件

W3.2 第一輪正式 scout 已依鎖定 protocol 實跑：

1. `players=300`
2. `rounds=3000`
3. `seeds={45,47,49}`
4. `memory_kernel=3`
5. `selection_strength=0.06`
6. `init_bias=0.12`
7. `events_json=docs/personality_dungeon_v1/02_event_templates_v1.json`
8. `policy_update_interval=150`
9. `theta_low=0.08`
10. `theta_high=0.12`
11. 固定 4 個 policy cells：`control_policy`, `w3_policy_crossguard`, `w3_policy_edgetilt`, `w3_policy_commitpush`

實際輸出：

1. `outputs/w3_policy_formal_20260402_summary.tsv`
2. `outputs/w3_policy_formal_20260402_combined.tsv`
3. `outputs/w3_policy_formal_20260402_decision.md`

### 12.2 正式結果

整體 decision：`close_w3_2`

關鍵 aggregate 結果：

1. `control_policy`：`mean_stage3_score=0.515010`，`mean_env_gamma=-0.000066`，`level3_seed_count=0`
2. `w3_policy_crossguard`：`mean_stage3_score=0.516959`，`mean_env_gamma=-0.000064`，`level3_seed_count=0`，`policy_activation_rate=1.000000`，`mean_policy_switches=1.000000`，`verdict=weak_positive`
3. `w3_policy_edgetilt`：`mean_stage3_score=0.513999`，`mean_env_gamma=-0.000066`，`level3_seed_count=0`，`policy_activation_rate=1.000000`，`mean_policy_switches=1.000000`，`verdict=fail`
4. `w3_policy_commitpush`：`mean_stage3_score=0.518930`，`mean_env_gamma=-0.000064`，`level3_seed_count=0`，`policy_activation_rate=1.000000`，`mean_policy_switches=1.000000`，`verdict=weak_positive`，且為 `best_policy`

### 12.3 解讀

1. 這輪不是 no-op：3 個 non-control policy cells 的 `policy_activation_rate` 都是 `1.0`，每個 seed 也都剛好發生 `1` 次 regime switch，表示 policy 機制真的有啟動
2. 但 step TSV 顯示，非 control cells 都在第一個 150-round window 就切到 active regime，之後整段 run 都沒有再切回 baseline；因此這個最小 hysteretic policy 在正式 protocol 下，實際上很快退化成「幾乎全程固定 commitment」
3. 在這種情況下，`w3_policy_crossguard` 與 `w3_policy_commitpush` 雖然仍有極小的 aggregate uplift，但所有 cells 的 `level_counts_json` 依舊完全相同：`{0:0,1:0,2:3,3:0}`，沒有任何 `Level 3 seed`
4. 因此 W3.2 的研究結論不是「policy 沒有被觸發」，而是「即使 policy 被觸發且幾乎全程 active，仍然只是在同一個 Level 2 plateau 上做微小位移」
5. 這使 W3.2 成為比 W3.1 更強的 closure signal：固定 commitment 不行，低頻 hysteretic state-feedback 也不行，而且在目前門檻設計下，它還會自然塌縮成近靜態 commitment

### 12.4 後續限制

1. 依 W3.2 closure 規則，不再做 `theta_low/theta_high`、`policy_update_interval`、或既有 3 個 regime 組合的微調
2. 若續做 W3，必須改成真正不會在第一窗就黏住 active 的更高階 leader policy family，例如不同 state signal、連續控制、或多-action policy，而不是繼續細修這個最小 hysteretic 架構

### 12.5 W3.3 鎖定實驗矩陣

W3.3 不再允許「先做一個 pulse 版、再回來調 trigger/interval」這種鬆散流程。第一輪 formal scout 只允許下列 4-cell 矩陣：

| condition | trigger | pulse action | baseline geometry | pulse geometry | 固定 mechanics |
|---|---|---|---|---|---|
| `control_pulse_policy` | none | none | `a=1.00, b=0.90, cross=0.20` | none | `ema_alpha=0.15`, `pulse_horizon=120`, `refractory=240` |
| `w3_pulse_crossguard` | `dominant_transition` | `cross_pulse` | `a=1.00, b=0.90, cross=0.20` | `a=1.00, b=0.90, cross=0.35` | same |
| `w3_pulse_edgetilt` | `dominant_transition` | `edge_pulse` | `a=1.00, b=0.90, cross=0.20` | `a=1.05, b=0.85, cross=0.20` | same |
| `w3_pulse_commitpush` | `dominant_transition` | `commit_pulse` | `a=1.00, b=0.90, cross=0.20` | `a=1.05, b=0.85, cross=0.35` | same |

矩陣設計原則：

1. trigger family 只測 `dominant_transition`，不在第一輪同時混入別的 state signal
2. mechanics 只測一組：`ema_alpha=0.15`, `pulse_horizon=120`, `refractory=240`
3. action family 只重用 W3.1/W3.2 已知的 3 個 payoff geometries，避免把 W3.3 變成「pulse + 新幾何」的混合問題

### 12.6 W3.3 Overall Decision Table

W3.3 的整體判讀固定採下面的優先序，不得在跑完後改口：

| precedence | overall decision | 必要條件 | 操作含義 |
|---:|---|---|---|
| 1 | `pass` | 至少 1 個 non-control cell 通過 promotion gate：`level3_seed_count>=1`、`mean_env_gamma>=0`、`mean_stage3_score>=control`、`pulse_activation_rate>0` | 開啟 W3.3 confirm / follow-up |
| 2 | `close_w3_3` | 所有 non-control cells 都是 `level3_seed_count=0` | W3.3 pulse family 直接結案 |
| 3 | `weak_positive` | 沒有 `pass`，但至少 1 個 non-control cell 有 `level3_seed_count>=1` 或 `leader_prefers_over_control=yes`，且 `pulse_activation_rate>0` | 僅保留為局部訊號，不自動升級 |
| 4 | `fail` | 其餘情形 | 無足夠訊號 |

補充限制：

1. 若某個 non-control cell `pulse_activation_rate=0` 或 `mean_pulse_count=0`，它不能被視為 W3.3 的強證據
2. `close_w3_3` 一旦成立，就不再做 `ema_alpha`、`pulse_horizon`、`refractory` 或 pulse 排列的 local tuning

### 12.7 W3.3 Formal Closure

正式輸出：

1. `outputs/w3_pulse_formal_20260402_summary.tsv`
2. `outputs/w3_pulse_formal_20260402_combined.tsv`
3. `outputs/w3_pulse_formal_20260402_decision.md`

正式結果：

1. overall `decision=close_w3_3`
2. 3 個 non-control pulse cells 全部 `level3_seed_count=0`
3. 3 個 non-control pulse cells 全部 `pulse_activation_rate=1.000000` 且 `mean_pulse_count=1.000000`
4. `best_policy=w3_pulse_edgetilt`，但它仍只到 `weak_positive`，沒有通過 promotion gate

runbook 解讀：

1. W3.3 的負結果是有效負結果，不是因為 trigger 沒啟動；pulse mechanics 確實有實際介入
2. 但 pulse 介入只造成局部擾動，沒有讓任何 seed 脫離 `cycle_level=2` 的 plateau
3. 因為 `pass` 不成立，而且所有 non-control cells 都沒有任何 Level 3 seed，所以必須依固定優先序直接記為 `close_w3_3`
4. 到此為止，W3 pulse family 結案；不再接受 `ema_alpha`、`pulse_horizon`、`refractory`、或 pulse action 排列的後續微調

### 12.8 W2.1R ~ W3.3 綜合負結果總整理

本節把 W2.1R、W3.1、W3.2、W3.3 四條正式線索收斂。重點不在重複 `close_*` decision，而在釐清每一機制「改變了什麼、沒有改變什麼」，並排除哪些原本看似合理的解釋。

四條路線對照總表（摘要）：

| 路線 | 核心假說 | 有發生的事 | 無發生的事 | Decision | 被排除的說法 |
|---|---|---|---|---|---|
| `W2.1R` | `death + testament` 能把 multi-life 系統推向 tail emergence | 存活長度明顯拉長；人格分布往 `caution / stability_seeking` 漂移 | tail `Level 3 seed=0`；gamma uplift 不足以當突破證據 | `close_w2_1` | 「只是死太快所以看不到更高階動態」 |
| `W3.1` | 固定 leader commitment 足以改變吸引子 | 部分 commitment 有小幅 aggregate uplift | seed-level `level_counts` 完全不變 | `close_w3_1` | 「找準 `(a,b,cross)` 就會突破」 |
| `W3.2` | state-feedback policy 應優於靜態 commitment | `policy_activation_rate=1.0`；policy 真實啟動 | 無任何 `Level 3`；policy 很快退化成近靜態 commitment | `close_w3_2` | 「失敗只是因為 leader 還沒有 feedback」 |
| `W3.3` | finite pulse policy 可打破 plateau | `pulse_activation_rate=1.0`；pulse 真實觸發 | aggregate 指標與 control 幾乎重合；無任何 `Level 3` | `close_w3_3` | 「失敗只是因為 regime 太黏」 |

關鍵解讀：

1. `W2.1R` 成功修復了極端早死問題，`life 4..6` 能跑滿 `4000` rounds；但它主要放大的是保守存活策略，而不是創造新的旋轉結構。因此它排除的是「生存修復本身就是 emergence 前提」這個假說。
2. `W3.1 ~ W3.3` 的 leader-side intervention 不論是靜態 commitment、低頻 feedback、還是 finite pulse，都產生了真實局部效果：有小幅 uplift，也有真實 activation evidence。
3. 但這些干預都沒有改變任何 seed 的 attractor basin；所有正式 protocol 最終都停在同一個 `Level 2 plateau`。
4. 因此更合理的總結不是「機制沒有打進去」，而是：目前 formal protocol 下（`memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`），系統的 attractor robustness 很高，局部機制改變不足以推動 `Level 3 emergence`。

正式研究結論：

1. `W2` 與 `W3` 系列已系統性排除兩條最自然的路徑：
    - `W2.1R` 排除了「生存修復」路線。
    - `W3.1 ~ W3.3` 排除了「leader policy 優化」路線，且涵蓋靜態 / feedback / pulse 三種代表性 family。
2. 這批負結果支持的不是 no-op，而是 attractor robustness：干預能改變局部行為、存活型態與 leader activation，但不足以改變 follower population 的 basin topology。
3. 目前最可能的瓶頸，不在控制力道大小，而在於這些控制都沒有直接改寫 sampled population 的相位幾何、旋轉結構或延遲動態。
4. 因此 `W2.1R ~ W3.3` 的正式結果本身已足以構成一段可對外防守的研究敘事：四條最自然的修補路徑都已被正式測過，而且都只能造成 plateau 內局部擾動，無法產生正式可承認的 `Level 3 seed`。

一句話收束：`W2.1R` 到 `W3.3` 的結果，已把「生存修復」、「靜態承諾」、「低頻 feedback」、「event-driven pulse」四條最自然修補路徑完整測過一輪，並證明目前框架下存在一個穩固的 sampled `Level 2 plateau`。這不是單點失敗，而是重要的結構性發現。

### 12.9 現階段研究決策與下一步

1. `W3` 主線正式停止：不再規劃 `W3.4`，也不再接受任何 `W3.1 ~ W3.3` 家族內的局部微調；`a/b/cross`、`theta/interval`、`ema_alpha/pulse_horizon/refractory` 一律凍結。
2. 研究重心轉向整理與敘事：優先完成 `W1 ~ W3.3` 的完整 closure narrative，將這批負結果整理成可對外防守的階段性成果。
3. 若未來重啟 `W` 系列，必須視為全新 family，而不是延續現有 leader-policy 線；新方向應優先考慮能直接操作 basin topology 的機制，例如非同構更新規則、world carryover、或徹底改寫 replicator 核心。
4. 替代方向可先保留為概念候選，而非立即開工：
    - Moran-like 過程或更純 event-driven 的演化機制，減少 replicator 平均化壓制。
    - 更高維度的 population diversity 或更高 `memory_kernel`，但必須搭配更長 tail 與更大玩家數做穩健性檢查。
    - 任何新方向都應先回答：它是否真的改寫相位幾何，而不只是增加一個局部控制鈕。
5. 補充觀察應明確保留：目前前段 `mean_rot_score` 已可達約 `0.606 ~ 0.607`，但多 seed `p_level_3` 仍低，說明真正卡住的不是「完全沒有方向性」，而是方向一致性無法穩定維持到 formal multi-seed protocol。

## 13. 舊版 Batch 決策規則（已過時，僅供存檔）

### 13.1 如果 Batch A 失敗

表示 `c=0.0` 時無法回歸舊 baseline，優先停下來檢查：

1. `matrix_cross_coupling` 預設值是否有污染
2. 診斷設定是否與前次不一致
3. 輸出檔名是否混用舊資料

### 13.2 如果 Batch B 顯示只有 seed=45 達到 Level 3

paper 敘事應改成：

1. 已觀察到 Level 3 breakthrough
2. 但目前具有明顯 seed sensitivity
3. 下一步優先做甜區擴張與更穩定條件搜索，而不是直接宣稱 robust Level 3

### 13.3 如果 Batch D 長跑掉回 Level 2

paper 敘事應改成：

1. 8000-round finite-window Level 3 breakthrough
2. 但長時間吸引子仍待確認

### 13.4 如果 Batch B 與 Batch D 都通過

此時再做 Batch F 才最有價值，因為：

1. 你已經有可信突破
2. 追高分數才不會淪為 tuning artifact

### 13.5 eta 敏感度建議（低成本高價值）

因為很多條件都卡在 `0.54–0.56`，建議在 paper 明確鎖定：

1. 主文一律採用 `eta=0.55`
2. 附錄補 `eta=0.53 / 0.57` 敏感度

這樣可以避免 reviewer 質疑 Stage 3 閾值是事後挑的。

### 13.6 config 指紋建議

建議所有輸出檔名都保留簡短 config 指紋，例如：

1. `crossc0p16_s1p5_t0p27_seed45_v2.csv`
2. `crossc0p0_s1p5_t0p27_seed45_ctrl.csv`

避免後期混淆不同批次產物。

## 14. 舊版今天 / 明天的最小執行集（已過時，僅供存檔）

### 今天

1. Batch A：控制組回歸測試
2. Batch B：multi-seed 穩健性
3. Batch C-light：甜區細修（若時間允許）

### 明天

1. Batch D-light：10000-round 穩定性
2. Batch E-core：機制診斷
3. Batch F-mini：組合 knob 探索（僅在前四批結果站得住時）

## 15. 舊版一句話版執行策略（已過時，僅供存檔）

先驗證 `cross-term` 是真的、可回歸、可跨 seed、可長時間維持，再去追求更高分數。

---

## 16. 2026-03-26 舊版進度盤點（已過時，僅供存檔）

本節把目前已跑完的結果、已確認失敗點、以及接下來要補跑的 simulation 一次整理，避免資訊分散在日誌與暫存輸出。

### 16.1 已完成批次與結論

| 批次 | 狀態 | 輸出 / 結果 | 結論 |
|---|---|---|---|
| Batch A（control vs cross） | ✅ 完成 | `control_cross0_seed45_sampled.csv`: L2, 0.5479；`sampled_seed45_apv2_s1p5_i400_crossc0p16.csv`: L3, 0.5611 | cross-term 是 causal knob（+0.0132） |
| Batch B-light（seed 45/48/51） | ✅ 完成 | seed45: L3, 0.5611；seed48: L2, 0.5137；seed51: L2, 0.5096 | 命中率 1/3，seed sensitivity 明顯 |
| Batch C（c-sweep, 8k） | ✅ 完成 | `c=0.08~0.20` 全部 L3（0.5548~0.5611） | 不是單點噪音，存在甜區平台 |
| Batch D-light（10k） | ✅ 已執行且失敗 | `c=0.155/0.16/0.165` tail=1000 全部 L2（約 0.532~0.534）；c=0.16 tail=3000 -> L1, 0.000 | 8k 的 L3 是暫態峰值，非穩定長期 attractor |

### 16.2 已確認的關鍵判讀

1. c=0.16 的突破在 8000 rounds 可觀測，但延長到 10000 rounds 後無法維持。
2. 當 tail 從 1000 放大到 3000，c=0.16 的 longrun 直接掉到 Level 1，顯示振幅包絡整體在衰減。
3. 因此現階段最精確敘事是：
    - 「在有限窗口（8k, tail=1000）可觀測 Level 3 強化」
    - 「但長期穩定吸引子尚未建立」。

### 16.3 P0 / P1 最終結果（2026-03-26 補記）

#### P0：穩定化主線結果

已完成 4 組 longrun（12000 rounds）：

| 編號 | 條件 | tail=1000 | tail=3000 | 判定 |
|---|---|---|---|---|
| S1 | `aps=2.0, interval=400` | L2, 0.5360 | L1, 0.0000 | 失敗 |
| S2 | `aps=3.0, interval=400` | L2, 0.5345 | L1, 0.0000 | 失敗 |
| S3 | `aps=1.5, interval=800` | L2, 0.5350 | L1, 0.0000 | 失敗 |
| S4 | `aps=2.0, interval=800` | L2, 0.5355 | L1, 0.0000 | 失敗 |

**結論**：提高 `adaptive_payoff_strength` 或延長 `payoff_update_interval` 都沒有把振幅包絡從衰減型推到近中性；4/4 在 `tail=3000` 全部崩到 Level 1。此路線應停止，不再做同類型的 `aps/interval` 小幅延伸。

#### P1：Batch B 完整版結果

正式 runbook seeds：`{45,47,49,51,53,55}`。

| seed | Level | score |
|---:|---:|---:|
| 45 | 3 | 0.5611 |
| 47 | 2 | 0.5046 |
| 49 | 2 | 0.4761 |
| 51 | 2 | 0.5096 |
| 53 | 2 | 0.5437 |
| 55 | 2 | 0.5203 |

彙總：
- `Pr(level>=3) = 1/6`
- `mean_score = 0.5192`
- `pass_by_hits = False`
- `pass_by_mean = False`

**結論**：`c=0.16` 的 breakthrough 高度依賴 `seed=45`，不具跨 seed 穩健性。正式 Batch B 判定失敗。

### 16.4 待執行清單（轉向後）

P0 與 P1 都失敗後，下一輪不再優先做同一 basin 的 tuning，而是直接轉向結構性搜尋。

#### P3-Update：單軸 structural-search 驗證結果（2026-03-26 夜）

已完成兩條單軸候選線的 simulation 驗證：

1. `B-line latest`：固定 `a=0.8`，掃 `b ∈ {0.690, 0.695, 0.699, 0.703, 0.708}`，seeds=`{45,47,49,51,53,55}`，`popularity_mode=sampled`
2. `A-line refine + expected`：固定 `b=0.9`，掃 `a ∈ {1.04, 1.052, 1.064, 1.076, 1.088, 1.10}`，seeds=`{45,47,49,51,53,55}`，`popularity_mode=expected`

結果摘要：

| 批次 | 最佳點 | mean_score | hits | 關鍵 observation |
|---|---|---:|---:|---|
| B-line latest | `(a,b)=(0.8, 0.690)` | 0.5178 | 1/6 | 仍只有 seed45 達 Level 3 |
| A-line refine + expected | `(a,b)=(1.04, 0.9)` | 0.5193 | 1/6 | seed45 降為 L2，seed53 升為 L3 |

**結論**：
1. `expected` 的確改變了「哪個 seed 會突破」，但沒有提高整體 hit rate。
2. 單軸 structural search 只帶來弱改善，尚未形成 robust Level 3 basin。
3. 下一步應從單軸切片升級為 **2D 小網格 expected**，直接在 `(a,b)` 平面局部搜尋。

#### P2：Batch E-core（無需 simulation）

使用既有 CSV 做機制分析：
- `outputs/control_cross0_seed45_sampled.csv`
- `outputs/sampled_seed45_apv2_s1p5_i400_crossc0p16.csv`

重點：
1. phase portrait 是否只在 seed45 呈現短暫封閉軌道
2. `turn_strength` 移動平均是否顯示短窗內短暫對齊
3. `final_risk` 是否存在 seed45 專屬分布偏態

#### P3：Hopf / near-neutral 結構性搜尋（已完成第一輪）

目標：不再固定在 `a=0.8, b=0.9` 這個 basin 上做增益，而是尋找更接近 Hopf 邊界的 `(a,b)`，讓局部線性化更接近：

- ODE：`alpha ≈ 0`
- lagged：`rho - 1 ≈ 0`

第一輪已完成項目：
1. `analysis.hopf_scan --mode lagged --scan b`，固定 `a=0.8`
2. `analysis.hopf_scan --mode lagged --scan a`，固定 `b=0.9`
3. 交叉參考 `--mode ode`，避免只在離散近似上追邊界
4. 對 near-neutral 候選點再回 simulation 做小規模 seed 驗證

狀態判讀：
1. 第一輪 structural search 已成功證明 `B-center expected` 是 plateau，而非 basin。
2. 因此 `P3` 不再作為主要待執行清單；後續只保留「提供遠離 plateau 的新 `(a,b)` 候選點」這個支援角色。

#### P3-next：2D 小網格 expected（已完成，結案）

建議直接跑兩塊局部 2D grid：

1. **A-center expected**：以 `(a,b)=(1.04, 0.9)` 為中心
    - `a ∈ {1.00, 1.04, 1.08}`
    - `b ∈ {0.86, 0.90, 0.94}`
2. **B-center expected**：以 `(a,b)=(0.8, 0.690)` 為中心
    - `a ∈ {0.76, 0.80, 0.84}`
    - `b ∈ {0.67, 0.69, 0.71}`

共通設定：
- `popularity_mode=expected`
- `seeds={45,47,49,51,53,55}`
- `rounds=8000`

原始目的：
1. 檢查 `expected` 是否能在局部 2D 區域把 `hits` 提高到 `>= 2/6` 或 `>= 3/6`
2. 確認最佳點是否來自真正的 2D 組合效應，而非單軸切片偶然點

結案結果：
1. `A-center expected` 沒有打開 robust basin。
2. `B-center expected` 經 2D grid、fixed-a fine-b、skew-band 與 plateau 診斷後已正式封存。
3. 本項從待辦清單移除，不再作為優先方向。

#### P3-final：B-center expected 區域正式封存（2026-03-27 凌晨）

已完成三組互補驗證：

1. **B-center 2D expected**：`a ∈ {0.76,0.80,0.84}`、`b ∈ {0.67,0.69,0.71}`
2. **fixed-a fine-b expected**：固定 `a=0.80`，掃 `b ∈ {0.685,0.690,0.695,0.700,0.705,0.710,0.715}`
3. **skew-band expected**：沿 B-center 斜帶與非對稱側帶做 pair-list 搜尋

三組結果共同結論：

| 區域/批次 | 最佳點 | hits | 關鍵 stage 指標 |
|---|---|---:|---|
| B-center 2D expected | `(0.80, 0.69)` | 1/6 | `amp_max≈0.2122`, `freq_corr≈0.1167`, `turn_strength≈0.000651` |
| fixed-a fine-b expected | `(0.80, 0.685~0.715)` | 1/6 | 整條線指標近乎等價 |
| skew-band expected | `(0.80, 0.69)` 或 `(0.80, 0.70)` | 1/6 | 與 fixed-a/fixed-b plateau 幾乎同構 |

補充：使用 `scripts/diagnose_plateau_manifest.py` 對 manifest 逐點輸出 `amplitude / corr / turning_strength / rotation_consistency` 後，確認不只是最終 `level` 一樣，而是 **stage1 / stage2 / stage3 的核心數值也幾乎平坦**。

**正式判定**：
1. `B-center expected` 區域已無繼續細掃價值。
2. 目前問題不是 classifier 解析度不足，而是模型在此區域本身就呈現近等價 plateau。
3. 後續不再在 `(0.80, 0.69)` 附近追加更細 `a/b` 微調。

#### P4-new：換機制搜尋（新主線）

在 `B-center expected` 被正式封存後，下一輪主問題不再是幾何位置微調，而是 **演化機制是否壓制了 robust Level 3**。

核心假說：

即使 `(a,b)` 已接近 near-neutral，若更新機制本身把相位旋轉持續沖淡，則任何局部幾何 sweet spot 都只會呈現 `1/6` seed-specific 突破，而不會形成 basin。換句話說，當前限制 robust Level 3 的因素，可能不是 payoff 幾何，而是以下機制交互作用：

1. `popularity_mode`
    - `sampled` 與 `expected` 目前只改變成功 seed 身分，未提高 `hits`
    - 下一步應把它當成動態機制因子，而不是單純噪音模型備註
2. `matrix-cross-coupling`
    - `c=0.16` 曾提供 finite-window breakthrough 線索
    - 但尚未驗證更寬的 `c` 區間是否與 `(a,b)` 共同形成新相位結構
3. adaptive payoff 更新規則
    - `adaptive_payoff_strength`
    - `payoff_update_interval`
    - `adaptive_payoff_target`
    - 先前只在固定 basin 內做過小幅 rescue 型調參；下一輪要改成和新 `(a,b)` 候選區域聯合設計，而不是原地補強

本主線的待完成事項改拆成三類主變因；後續所有短期工作都應歸屬於這三類，而不是回到局部 `a/b` 微調。

##### M1：`popularity_mode` 作為主變因

研究問題：
- `sampled` 與 `expected` 是否只是換成功 seed，還是會系統性改變 `turn_strength`、`rotation_consistency`、與 `hits`？

待完成事項：
1. 在同一組新 `(a,b)` 候選點上，固定其他機制參數，並排比較 `sampled` / `expected`。
2. 對每個模式輸出 plateau 診斷，而不是只記最終 `level`。
3. 比較 `per_csv_metrics.tsv` 裡的 `turning_score / turning_strength / freq_best_corr` 是否出現穩定位移。

短期目標：
1. 選 2~3 個遠離 B-center plateau 的 `(a,b)` 候選點。
2. 每個點跑 `sampled` 與 `expected` 各 6 seeds。
3. 生成一份 mode 對照表，至少包含 `hits`、`mean_turn_strength`、`mean_rotation_consistency`。

中期目標：
1. 確認是否存在某一個 mode 在多個 `(a,b)` 點上都能穩定抬升 `turn_strength`。
2. 若存在，將該 mode 固定為後續 cross-coupling / update-rule 搜尋的標準模式。

驗收標準：
1. 至少有一個 mode 在不低於兩個候選點上，使 `mean_turn_strength` 相對另一 mode 提升 `>= 10%`。
2. 或至少有一個 mode 把 `hits` 從 `1/6` 提高到 `>= 2/6`。
3. 若兩者皆未達成，則判定 `popularity_mode` 主要只改 seed 身分，不足以單獨打開 basin。

##### M2：`matrix-cross-coupling` 作為主變因

研究問題：
- 關鍵到底是單點 `c=0.16`，還是 `c` 與 `(a,b)` 的交互作用？

待完成事項：
1. 對少量新 `(a,b)` 候選點做 `c` 的粗網格，而不是固定 `c=0.16` 後只掃幾何。
2. 比較 `c ∈ {低, 中, 高}` 三段對 `turn_strength` 維持、`freq_corr`、與 `hits` 的影響。
3. 特別標記「只提升單 seed 分數」與「真正提升 multi-seed hits」的差異。

短期目標：
1. 先跑粗網格，例如 `c ∈ {0.10, 0.16, 0.22}`。
2. 每個 `c` 搭配 2~3 個遠離 B-center plateau 的新 `(a,b)` 候選。
3. 對每個點都產出 plateau 診斷 summary，而不只留 raw CSV。

中期目標：
1. 找到至少一段 `c` 區間，使多個 `(a,b)` 候選點的 `mean_turn_strength` 共同上升。
2. 判定 `c` 是否應該在後續搜尋中被視為一級主變因，而非固定背景參數。

驗收標準：
1. 至少一個 `c` 設定在不低於兩個 `(a,b)` 候選點上，使 `mean_turn_strength` 與 `mean_freq_best_corr` 同時上升。
2. 或至少一個 `c` 設定把 `hits` 提升到 `>= 2/6`。
3. 若只看到單 seed 加分而 `hits` 不變，則記錄為「有限窗口強化」，不算通過。

##### M2-Result：第一輪 `cross-coupling x (a,b)` 粗網格已完成

已執行腳本：
- `scripts/run_mechanism_cross_coupling_grid.sh`

已完成條件：
- `(a,b) ∈ {(1.00,0.90), (1.04,0.90), (1.076,0.90)}`
- `c ∈ {0.10, 0.16, 0.22}`
- `popularity_mode=expected`
- `adaptive_payoff_strength=1.5`
- `interval=400`
- `target=0.27`
- `selection_strength=0.06`
- `seeds={45,47,49,51,53,55}`

結果摘要：

| `(a,b)` | `c=0.10` | `c=0.16` | `c=0.22` | 判讀 |
|---|---|---|---|---|
| `(1.00,0.90)` | `hits=0/6`, `mean_score=0.5131` | `hits=1/6`, `mean_score=0.5196` | `hits=1/6`, `mean_score=0.5171` | `c` 有門檻效應，但未打開 basin |
| `(1.04,0.90)` | `hits=0/6`, `mean_score=0.5131` | `hits=1/6`, `mean_score=0.5193` | `hits=1/6`, `mean_score=0.5171` | 幾何差異很弱，主訊號來自 `c` |
| `(1.076,0.90)` | `hits=0/6`, `mean_score=0.5130` | `hits=1/6`, `mean_score=0.5193` | `hits=1/6`, `mean_score=0.5171` | near-neutral 幾何本身不足以打開 robust Level 3 |

**目前判定**：
1. `matrix-cross-coupling` 應正式升格為一級主變因。
2. `c=0.16` 不是隨機巧合，而是接近門檻區段的有效機制值。
3. 但 `c` 單獨不足以把 `hits` 從 `1/6` 推到 `>=2/6`；下一步必須與其他機制因子交叉。

##### P4-Exec：下一步採用 S1 screening（唯一優先）

這一步的目標不是全面搜尋，而是用最小成本回答：

- `c × popularity_mode × adaptive_payoff_strength` 是否存在第一個 `hits >= 2/6` 的組合？
- 若沒有，是否至少存在 `mean_turn_strength` 相對目前 baseline 提升 `>= 12%` 的組合？

###### S1-screening 參數（2 點版本，優先）

代表性 `(a,b)` 候選：
- `(1.00, 0.90)`
- `(1.076, 0.90)`

主變因：
- `c ∈ {0.12, 0.16, 0.20}`
- `popularity_mode ∈ {sampled, expected}`
- `adaptive_payoff_strength ∈ {1.2, 1.8}`

固定參數：
- `selection_strength=0.06`
- `payoff_update_interval=400`
- `adaptive_payoff_target=0.27`
- `init_bias=0.12`
- `rounds=8000`
- `seeds={45,47,49,51,53,55}`

總成本：
- `2 (ab) × 3 (c) × 2 (mode) × 2 (strength) × 6 seeds = 144 runs`

###### 為何不用更大格子

1. 目前證據已表明 `a/b` 幾何不是主限制因子。
2. 先保留 2 個代表點，只是為了避免把所有結論壓在單一 `(a,b)` 點上。
3. 若 S1 已失敗，就沒有理由直接擴成更大幾何網格。

###### 可直接執行指令

現有 `scripts/run_mechanism_cross_coupling_grid.sh` 已足夠支撐 S1，只要用外層 loop 交叉 `mode × aps` 即可。

```bash
cd /home/user/personality-dungeon

for mode in sampled expected; do
    for aps in 1.2 1.8; do
        RUN_TAG="mech_s1_${mode}_aps$(printf '%s' "$aps" | tr '.' 'p')" \
        AB_PAIRS="1.00:0.90 1.076:0.90" \
        C_GRID="0.12 0.16 0.20" \
        POPULARITY_MODE="$mode" \
        APS="$aps" \
        INTERVAL=400 \
        TARGET=0.27 \
        SEEDS="45 47 49 51 53 55" \
        J_MAX=12 \
        bash scripts/run_mechanism_cross_coupling_grid.sh
    done
done
```

###### 數據化輸出流程

每個 run 完成後，立刻使用既有 plateau 診斷腳本輸出量化 summary：

```bash
./venv/bin/python scripts/diagnose_plateau_manifest.py \
    outputs/mechanism_search/<RUN_TAG>/manifest_cross_coupling.tsv
```

必看輸出：
- `diagnostics/per_csv_metrics.tsv`
- `diagnostics/per_point_summary.tsv`

主要比較欄位：
- `hits_level3`
- `mean_level`
- `mean_freq_best_corr`
- `mean_turning_score`
- `mean_turning_strength`
- `mean_rotation_consistency`

###### S1 驗收標準

Primary success：
1. 任一組合達到 `hits >= 2/6`

Secondary success：
1. 任一組合達到 `mean_turn_strength >= baseline × 1.12`
2. 且此提升不是由單一 seed 獨自貢獻，而是至少兩個 seed 同向改善

Failure / stop rule：
1. 若所有組合仍 `hits <= 1/6`
2. 且 `mean_turn_strength` 提升未達 `12%`
3. 則判定 `c × mode × strength` 仍不足以打開 robust basin，S1 結束後直接轉入更高層級機制搜尋，不再擴幾何網格

###### S1-Result：已完成，未通過（2026-03-27）

已完成條件：
- `(a,b) ∈ {(1.00,0.90), (1.076,0.90)}`
- `c ∈ {0.12,0.16,0.20}`
- `popularity_mode ∈ {sampled, expected}`
- `adaptive_payoff_strength ∈ {1.2,1.8}`
- `selection_strength=0.06`
- `interval=400`
- `target=0.27`
- `seeds={45,47,49,51,53,55}`

Baseline：
- `outputs/mechanism_search/mechanism_cross_coupling_20260327_170936/manifest_cross_coupling.tsv`
- `baseline_turn_strength = 0.0006506491`

條件總表摘要：

| `c` | mode | aps | hits | `mean_turn_strength` | uplift vs baseline |
|---:|---|---:|---:|---:|---:|
| 0.20 | sampled | 1.8 | 2/12 | 0.00065618 | +0.85% |
| 0.20 | sampled | 1.2 | 2/12 | 0.00065540 | +0.73% |
| 0.16 | sampled | 1.8 | 2/12 | 0.00065455 | +0.60% |
| 0.16 | sampled | 1.2 | 2/12 | 0.00065394 | +0.51% |
| 0.20 | expected | 1.8 | 2/12 | 0.00065360 | +0.45% |
| 0.16 | expected | 1.8 | 2/12 | 0.00065080 | +0.02% |

單點 `(a,b)` 明細顯示：
1. 每個 `(a,b,c,mode,aps)` 單點都仍是 `1/6`。
2. 因此條件總表裡的 `2/12`，只是兩個 `(a,b)` 點各自維持 `1/6`，**不是** 單一條件真正提升到 `2/6`。

正式判定：
1. **Primary success 失敗**：沒有任何單一 `(a,b)` 條件達到真正的 `hits >= 2/6`。
2. **Secondary success 失敗**：最大 `turn_strength uplift` 只有 `+0.85%`，遠低於 `+12%` 門檻。
3. `sampled` 整體略優於 `expected`。
4. `c=0.16~0.20` 可保留為有效機制帶，但不足以單獨形成 basin。
5. `adaptive_payoff_strength` 在 `1.2` 與 `1.8` 之間沒有形成主導性分離。

研究結論：
1. `c × mode × strength` 有弱方向性訊號，但不構成通過。
2. 後續不擴 S1，也不進入 S2 幾何精煉。
3. 下一步直接轉入 **M3 follow-up**，把主交互改成更新時序 / 更新規則。

###### S2 精煉條件（只有在 S1 成功後才做）

若 S1 成功，再進入 S2：

1. 保留最佳 `(a,b)` 點
2. 補跑 `adaptive_payoff_strength=1.5`
3. 必要時把 `c` 補為 `0.14` 與 `0.18`
4. 視情況才引入第二組 seed panel 做外推驗證

若 S1 未成功，則不進入 S2，也不恢復任何局部 `a/b` 細掃。

在目前結果下：
- `S2` 不啟動。
- 任何額外 `c` 細掃或 `aps` 擴點都暫停。

##### M3：adaptive payoff 更新規則作為主變因

研究問題：
- `adaptive_payoff_strength / payoff_update_interval / adaptive_payoff_target` 是否不是在舊 basin 上做 rescue，而是在新候選區域上決定 rotation 能否維持？

待完成事項：
1. 放棄 P0/P1 那種「固定舊 basin、微調小步」策略。
2. 改做代表性機制組合，例如「弱但快 / 中等 / 強但慢」三類更新時序。
3. 在相同 `(a,b)` 候選點上，比較 update rule 對 `turn_strength` 衰減與 `hits` 的影響。

短期目標：
1. 定義 3 組代表性更新規則組合，而不是連續細掃。
2. 將這 3 組與 `sampled/expected` 或 `c` 的最佳候選做小規模交叉。
3. 對每組都輸出 `tail=1000` 的 stage 指標 summary。

中期目標：
1. 確認是否存在某種更新時序，能讓旋轉強度在多 seed 下維持而不只是短窗突破。
2. 若存在，將該更新時序納入新的 baseline，才回頭做下一輪幾何搜尋。

驗收標準：
1. 至少一組更新規則讓 `mean_turn_strength` 相對目前 baseline 提升 `>= 10%`，且 `hits >= 2/6`。
2. 或在 `tail` 加長時，仍能維持 Level 2/3 而不快速塌到 Level 1。
3. 若只在舊 basin 有短窗加分，則視為 rescue 舊路線，不列入新主線通過。

##### M3-follow-up：下一步唯一優先（聚焦版）

S1 未通過後，下一步不再沿 `c × mode × aps` 擴張，而是直接改測 **更新時序是否才是瓶頸**。

固定條件：
- `popularity_mode=sampled`
- `c ∈ {0.16, 0.20}`
- `(a,b) ∈ {(1.00,0.90), (1.076,0.90)}`
- `adaptive_payoff_strength=1.5`
- `selection_strength=0.06`
- `init_bias=0.12`
- `rounds=8000`
- `seeds={45,47,49,51,53,55}`

主變因：
- `payoff_update_interval ∈ {200, 400, 800}`
- `adaptive_payoff_target ∈ {0.25, 0.27, 0.29}`

說明：
1. `sampled` 已在 S1 中略優於 `expected`，因此先固定為較優模式。
2. `c=0.16~0.20` 已被證明是有效機制帶，因此保留作為背景機制區間。
3. 這一步的核心問題不再是 `c` 多大，而是更新時序是否決定 rotation 能否維持。
4. `adaptive_payoff_strength` 固定為 `1.5`，因 S1 顯示 `1.2` 與 `1.8` 沒有形成主導分離，M3 先把自由度保留給 timing / target。

總成本（M3 follow-up v1）：
- `2 (ab) × 2 (c) × 3 (interval) × 3 (target) × 6 seeds = 216 runs`

量化目標：
1. 找出第一個單點條件達到 `hits >= 2/6`
2. 或找到 `mean_turn_strength uplift >= 10%` 的時序組合
3. 若兩者皆失敗，則更有把握地判定目前模型的 robust Level 3 受限於更深層機制，而非現有更新規則可救回

執行後必看輸出：
1. `summary_by_condition.tsv`
2. `summary_by_ab.tsv`
3. 單點是否真正達到 `2/6`，而不是僅在聚合後出現 `2/12`

###### M3 可直接執行指令

```bash
cd /home/user/personality-dungeon

RUN_TAG="mech_m3_timing_v1" \
AB_PAIRS="1.00:0.90 1.076:0.90" \
C_GRID="0.16 0.20" \
POPULARITY_MODE="sampled" \
APS="1.5" \
INTERVAL_GRID="200 400 800" \
TARGET_GRID="0.25 0.27 0.29" \
SEEDS="45 47 49 51 53 55" \
J_MAX=12 \
bash scripts/run_mechanism_m3_timing_grid.sh
```

產物：
- `outputs/mechanism_search/<RUN_TAG>/manifest_m3_timing.tsv`
- `logs/<RUN_TAG>/...`

###### M3 量化摘要指令

```bash
cd /home/user/personality-dungeon

./venv/bin/python scripts/summarize_mechanism_s1.py \
    outputs/mechanism_search/mech_m3_timing_v1/manifest_m3_timing.tsv \
    --out-dir outputs/mechanism_search/mech_m3_timing_v1_summary \
    --baseline-manifest outputs/mechanism_search/mechanism_cross_coupling_20260327_170936/manifest_cross_coupling.tsv
```

判讀重點：
1. 優先看 `summary_by_ab.tsv`，確認是否真的有單點條件達到 `hits >= 2/6`。
2. 再看 `summary_by_condition.tsv`，比較 `interval × target` 是否形成穩定排序。
3. 若只看到聚合後 `2/12`，但單點仍為 `1/6`，則不算通過。

###### M3-Result：已完成，未通過（2026-03-27）

已完成條件：
- `popularity_mode=sampled`
- `c ∈ {0.16, 0.20}`
- `(a,b) ∈ {(1.00,0.90), (1.076,0.90)}`
- `adaptive_payoff_strength=1.5`
- `payoff_update_interval ∈ {200, 400, 800}`
- `adaptive_payoff_target ∈ {0.25, 0.27, 0.29}`
- `selection_strength=0.06`
- `rounds=8000`
- `seeds={45,47,49,51,53,55}`

完成產物：
- `outputs/mechanism_search/mech_m3_timing_v1/manifest_m3_timing.tsv`
- `outputs/mechanism_search/mech_m3_timing_v1_summary/summary_by_condition.tsv`
- `outputs/mechanism_search/mech_m3_timing_v1_summary/summary_by_ab.tsv`

條件總表最佳組合：

| `c` | mode | aps | interval | target | hits | `mean_score` | `mean_turn_strength` | uplift vs baseline |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | sampled | 1.5 | 800 | 0.29 | 2/12 | 0.51762 | 0.00065627 | +0.86% |
| 0.20 | sampled | 1.5 | 400 | 0.27 | 2/12 | 0.51718 | 0.00065611 | +0.84% |
| 0.20 | sampled | 1.5 | 400 | 0.29 | 2/12 | 0.51711 | 0.00065572 | +0.78% |
| 0.20 | sampled | 1.5 | 200 | 0.29 | 2/12 | 0.51817 | 0.00065562 | +0.76% |

單點 `(a,b)` 最佳組合：

| `a` | `b` | `c` | interval | target | hits | `mean_score` | `mean_turn_strength` | uplift vs baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.90 | 0.20 | 400 | 0.27 | 1/6 | 0.51676 | 0.00065648 | +0.90% |
| 1.076 | 0.90 | 0.20 | 800 | 0.29 | 1/6 | 0.51829 | 0.00065629 | +0.87% |

與 S1 並排比較後的結論：
1. M3 的最佳條件 uplift `+0.86%`，只比 S1 最佳條件 `+0.85%` 高極小幅度，可視為持平。
2. M3 的最佳單點仍為 `1/6`，沒有任何單點條件達到真正的 `2/6`。
3. 因此條件表中的 `2/12` 仍然只是兩個 `(a,b)` 候選點各自維持 `1/6` 的聚合效果，而不是新的 basin。

正式判定：
1. **Primary success 失敗**：沒有任何單點 `(a,b,c,interval,target)` 條件達到 `hits >= 2/6`。
2. **Secondary success 失敗**：最大 `turn_strength uplift` 只有 `+0.86%`，遠低於 `+10%` 門檻。
3. `c=0.20` 仍是目前較佳背景機制帶。
4. `target=0.27~0.29` 略優於 `0.25`，但只形成弱訊號。
5. `interval` 沒有出現壓倒性的穩定排序。

研究結論：
1. `c + mode + timing/target` 這條線在目前模型下沒有打開新的 robust Level 3 basin。
2. M3 沒有實質超過 S1，只提供了「`c=0.20`、較高 target 稍優」的弱方向性訊號。
3. 現階段不再擴 M3，也不建議再做更細的 `interval/target` 局部掃描。
4. 若後續仍要追 Level 3，應改換更深層的結構假說，而不是繼續在此機制線上做修補式搜尋。

#### 目前狀態：無待完成實驗，等待新假說

截至目前為止：

1. `B-center expected` 幾何細掃已封存。
2. `S1 (c × mode × aps)` 已完成且未通過。
3. `M3 (interval × target)` 已完成且未通過。
4. 現有 `matrix_ab + cross-coupling + popularity_mode + adaptive update timing` 這條修補式主線，已沒有新的待完成批次。
5. 因此在提出新的結構假說前，**目前無待完成實驗**。

決策規則：

1. 不再延伸任何局部 `a/b` 細掃。
2. 不再延伸任何 `c / aps / interval / target` 的局部擴點。
3. 若下一步要重啟實驗，必須先提出新的結構假說，說明它為何有機會改變 multi-seed basin，而不只是小幅抬升單點分數。
4. 若新假說涉及 payoff 定義、時間索引、更新規則或 CSV 契約，必須先更新 `SDD.md` 再改碼。

#### 下一個可行的新結構假說（候選）

以下不是已排程實驗，而是目前最值得評估的 3 個候選方向。

##### H1：回饋記憶長度才是主限制，而不是單步 cross-coupling

核心想法：

1. 目前模型主要依賴單步 popularity lag。
2. 若 robust rotation 需要更長的回饋記憶，單步 lag 只會產生有限窗口的弱旋轉，無法形成 basin。

可驗證方向：

1. 把 payoff 輸入從單步 popularity 改成多步記憶或加權移動平均。
2. 比較「單步 lag」與「多步 memory kernel」對 `turn_strength`、`freq_corr`、與 `hits` 的影響。

為何值得優先考慮：

1. 現有所有失敗都像是旋轉會被快速沖淡，而不是完全沒有旋轉方向。
2. 這更像回饋記憶不足，而不是靜態幾何位置錯誤。

Spec 影響：

1. 這會改到 payoff 的時間索引與可能的狀態定義。
2. 屬於契約變更，必須先更新 `SDD.md`。

##### H2：需要非線性／分段式 payoff，而不是線性 `matrix_ab`

核心想法：

1. 目前 `matrix_ab` 與 cross-coupling 基本上仍是線性結構。
2. 若 robust Level 3 需要 regime switch 或 threshold effect，線性 payoff 可能天然只會給 near-neutral plateau，而不會給穩定 basin。

可驗證方向：

1. 設計簡單的分段式或 thresholded payoff，例如在某兩策略共存超過門檻時切換懲罰強度。
2. 比較線性版與非線性版是否會把 `1/6` 類突破推成真正的 `2/6` 以上。

為何值得優先考慮：

1. 目前看到的最好結果都是弱 uplift，代表線性參數調整可能只是在平坦面上移動。
2. 若 basin 需要曲率改變，非線性比再細調連續參數更合理。

Spec 影響：

1. 這會直接改 payoff 定義。
2. 必須先更新 `SDD.md`，並明確定義新 payoff 與不變條件。

##### H3：同質 replicator 更新會沖淡旋轉，需引入異質更新速度或固定子族群

核心想法：

1. 現在所有策略都共用同一種更新機制與群體平均 reward 聚合。
2. 若 robust rotation 需要保留較慢反應或固定傾向的子族群，單一同質 replicator 可能會把相位差快速抹平。

可驗證方向：

1. 引入少量固定子族群或 strategy-specific adaptation rate。
2. 檢查異質更新是否能讓 phase lag 在多 seed 下維持更久。

為何值得優先考慮：

1. 現有動態看起來更像「旋轉存在，但太快被平均化」。
2. 這是更新結構層級的改動，與已失敗的 `aps/interval` 微調不同。

Spec 影響：

1. 這會改到演化更新規則與可能的 player/state 定義。
2. 同樣屬於契約變更，需先更新 `SDD.md`。

#### P4：Batch F-mini（暫停）

P0/P1 已失敗，因此暫停 F-mini；在找到新的 near-neutral `(a,b)` 候選點之前，不建議再對 `c=0.16 + aps/interval` 路線追加小型調參。

| 編號 | s | target | out |
|---|---:|---:|---|
| F1 | 1.5 | 0.265 | `outputs/knobscan_mini_crossc0p16_s1p5_target0p265.csv` |
| F2 | 1.5 | 0.275 | `outputs/knobscan_mini_crossc0p16_s1p5_target0p275.csv` |
| F3 | 1.8 | 0.265 | `outputs/knobscan_mini_crossc0p16_s1p8_target0p265.csv` |
| F4 | 1.8 | 0.275 | `outputs/knobscan_mini_crossc0p16_s1p8_target0p275.csv` |

### 16.5 目前執行狀態

1. 現有 runbook 內的 S1 與 M3 都已完成並正式結案。
2. `F-mini` 與所有舊 basin rescue 類調參持續暫停。
3. 目前不排任何新批次，等待新的結構假說先被寫成 spec-level 研究問題。
4. 下一次實驗啟動的前提，是先在本文件或 `SDD.md` 明確寫出新的結構機制、觀測指標、與 stop rule。

### 16.6 論文敘事鎖定（更新版）

在下一輪結果出來前，建議固定以下說法：

1. 已有可重現的 finite-window Level 3 訊號（最佳 score=0.5611）。
2. 但 longrun 驗證與 multi-seed 驗證都失敗，不宜宣稱 robust Level 3 attractor。
3. `c=0.16` 路線提供的是一個機制線索，而不是最終穩定解。
4. 單軸 `(a,b)` structural search 與 `expected` 模式目前只改變了成功 seed 的身分，尚未提升整體 robust hit rate。
5. `B-center expected` 經 2D grid、fixed-a fine-b、skew-band 與 plateau 指標診斷後已正式封存。
6. 目前已完成對 `popularity_mode / cross-coupling / adaptive payoff rule` 這條機制線的第一輪否證；下一步若重啟，核心將是新的結構假說，而不是既有機制線的延伸。

## 17. H-series 結構假說實驗（唯一待完成主線）

本章是 2026-03-28 起唯一保留的主線。舊機制線 `popularity_mode × matrix-cross-coupling × adaptive_payoff_timing/target` 已完成第一輪否證；除非 H-series 再次指出需要回頭比較，否則不再擴 `a/b/c/interval/target` 的局部微調。

### 17.1 問題重述

目前瓶頸不是「完全沒有旋轉」，而是：

1. 方向訊號存在，但會被快速沖淡
2. 線性 payoff 幾何看起來更像平坦 plateau，而不是穩定 basin
3. 同質更新可能過快抹平 phase lag

因此下一輪不再問「哪個 `(a,b,c)` 再高 0.003」，而是直接問「哪個結構機制能把 `1/6` 偶發突破推成可重複 basin」。

### 17.2 優先順序表

| 優先級 | 假說 | 核心理由 | 最小可驗證實驗 | 預估成本 | 通過條件 |
|---|---|---|---|---:|---|
| 1 | H1: 回饋記憶長度 | 單步 lag 只產生衰減螺旋，`env_gamma` 長期偏負 | `memory_kernel ∈ {1,3,5}`；固定 `sampled + c=0.20 + a=1.0 + b=0.9`；6 seeds；8000 rounds | 36 runs | `hits >= 2/6`，或 `mean_turn_strength` 相對 kernel=1 提升 `>= 15%`，且 `tail=3000` 不塌 |
| 2 | H2: 非線性 / 分段式 payoff | 線性 `matrix_ab` 只在 plateau 上移動，缺 regime switch | `payoff_mode=threshold_ab`；`theta ∈ {0.40,0.55}` 粗掃；其餘固定 H1 baseline | 24 runs | 至少一個單點條件達到 `hits >= 2/6`，且 `env_gamma \approx 0` |
| 3 | H3: 異質更新機制 | 同質 replicator 可能快速抹平相位差 | `evolution_mode=hetero`；先測 per-strategy `k` 向量；其餘固定 H1 baseline | 30 runs | `mean_rotation_consistency >= 0.65`，且 long-run tail 仍維持 Level 3 |
| 4 | H1+H2 混合 | 記憶與非線性可能同時處理 lag 與 plateau 問題 | 先取 H1 最佳 kernel，再只接一個 threshold mode | 12 runs | 若混合條件通過，直接升為新 baseline |

### 17.3 立即執行的 3 步

#### Step 1: Spec 先鎖死

這一步已在 [SDD.md](../SDD.md) 完成：新增 `2.5 新 payoff / memory / evolution 擴充規格（H-series）`，明確定義：

1. H1 的 `memory_kernel`
2. H2 的 `threshold_ab`
3. H3 的 `evolution_mode=hetero`
4. 退化回基線所需的不變條件
5. H-series 不改核心 timeseries CSV schema

硬規則：若後續想把 H1 改成加權 kernel、把 H3 改成固定子族群，都必須先回到 spec 再改碼。

#### Step 2: 先做 smoke，再決定要不要進 6-seed confirm

Smoke 的目的不是證明 robust，而是先回答「新機制有沒有比 baseline 更乾淨的方向訊號」。

固定 smoke 基線：

1. `evolution_mode=mean_field`
2. `payoff_lag=1`
3. `seed=45`
4. `rounds=260`
5. `a=1.0`, `b=0.9`, `matrix-cross-coupling=0.20`

H1 smoke 模板：

```bash
cd /home/user/personality-dungeon
export PYTHON_BIN=./venv/bin/python

$PYTHON_BIN -m simulation.run_simulation \
    --payoff-mode matrix_ab \
    --popularity-mode expected \
    --evolution-mode mean_field \
    --payoff-lag 1 \
    --memory-kernel 3 \
    --rounds 260 --seed 45 \
    --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
    --selection-strength 0.06 --init-bias 0.12 \
    --out outputs/h1_smoke_kernel3_seed45.csv
```

H2 smoke 模板：

```bash
$PYTHON_BIN -m simulation.run_simulation \
    --payoff-mode threshold_ab \
    --popularity-mode expected \
    --evolution-mode mean_field \
    --payoff-lag 1 \
    --memory-kernel 3 \
    --threshold-theta 0.40 \
    --threshold-a-hi 1.10 --threshold-b-hi 1.00 \
    --rounds 260 --seed 45 \
    --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
    --selection-strength 0.06 --init-bias 0.12 \
    --out outputs/h2_smoke_theta0p40_seed45.csv
```

H3 smoke 模板：

```bash
$PYTHON_BIN -m simulation.run_simulation \
    --payoff-mode matrix_ab \
    --popularity-mode expected \
    --evolution-mode hetero \
    --strategy-selection-strengths 0.06,0.06,0.04 \
    --payoff-lag 1 \
    --memory-kernel 3 \
    --rounds 260 --seed 45 \
    --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
    --init-bias 0.12 \
    --out outputs/h3_smoke_hetero_seed45.csv
```

Smoke stop rule：

1. 若 `turn_strength` 對基線幾乎無提升，或明顯數值崩塌，該假說直接淘汰
2. 只有在 smoke 顯示方向訊號提升時，才進入 6-seed / 8000-round confirm
3. 不准在 smoke 失敗後直接擴大掃描範圍

註：以上命令是 **H-series CLI 落地後的執行模板**。若參數尚未實作，先完成對應程式碼與回歸測試，再執行本章。

#### Step 3: 通過 smoke 後的最小 confirm 路徑

H1 confirm baseline：

1. `popularity_mode=sampled`
2. `matrix-cross-coupling=0.20`
3. `a=1.0`, `b=0.9`
4. `seeds={45,47,49,51,53,55}`
5. `rounds=8000`

第一批只做 H1：

```bash
for kernel in 1 3 5; do
    for seed in 45 47 49 51 53 55; do
        $PYTHON_BIN -m simulation.run_simulation \
            --payoff-mode matrix_ab \
            --popularity-mode sampled \
            --evolution-mode sampled \
            --payoff-lag 1 \
            --memory-kernel "$kernel" \
            --players 300 --rounds 8000 --seed "$seed" \
            --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
            --selection-strength 0.06 --init-bias 0.12 \
            --out "outputs/h1_confirm_k${kernel}_seed${seed}.csv"
    done
done
```

H1 通過後才做 H2 或 H3；若 H1 完全失敗，優先暫停 H2/H3 的全面展開，先重判 memory 假說本身是否值得保留。

### 17.4 判讀規則

Primary success：

1. 任一 H-series 條件達到單點 `hits >= 2/6`
2. 或任一條件在 `tail=1000` 與 `tail=3000` 都維持 Level 3

Secondary success：

1. `mean_turn_strength` 相對基線提升 `>= 15%`
2. `mean_env_gamma` 接近 0，且不是只由單一 seed 貢獻

Failure / stop rule：

1. smoke 沒訊號就停
2. confirm 仍只有 `1/6`，且 `turn_strength` 提升不顯著，就停
3. 不允許回頭做 `a/b/c/interval` 的局部補洞，除非 H-series 指出明確交互作用值得驗證

### 17.5 章節狀態

截至目前：

1. Spec 已鎖定
2. runbook 已切換到 H-series 主線
3. `memory_kernel` 已落地到 [simulation/run_simulation.py](../simulation/run_simulation.py)、[simulation/seed_stability.py](../simulation/seed_stability.py)、[simulation/rho_curve.py](../simulation/rho_curve.py)、[simulation/control_amp_baseline.py](../simulation/control_amp_baseline.py)、[simulation/grid_refine.py](../simulation/grid_refine.py)
4. `threshold_ab` 已落地到 [simulation/run_simulation.py](../simulation/run_simulation.py) 與 [dungeon/dungeon_ai.py](../dungeon/dungeon_ai.py)；H3 的 `hetero` 已落地到 [simulation/run_simulation.py](../simulation/run_simulation.py)、[simulation/seed_stability.py](../simulation/seed_stability.py) 與 [evolution/replicator_dynamics.py](../evolution/replicator_dynamics.py)，可跑 smoke / short scout
5. H1 已完成第一輪 smoke / confirm / sweep；下一個真正的分岔點，是判定 H1 是否值得加長 tail 或升級 protocol，而不是回頭掃舊 basin
6. 全 repo 文件已重新檢查，沒有再保留「`memory_kernel` 尚未實作」的舊說法

### 17.6 H1 第一輪結果（2026-03-29）

本節記錄 H1 的第一輪完整結果，避免後續重複讀 CSV。

#### 17.6.1 H1 smoke：有訊號，但不是決定性突破

mean-field smoke（`seed=45`, `rounds=260`, `a=1.0`, `b=0.9`, `c=0.20`）顯示：

1. `memory_kernel=3` 相對 `memory_kernel=1` 的 `turn_strength` 約提升 `+31.3%`
2. 振幅約提升 `+15.7%`
3. 因此 H1 不屬於「零訊號假說」，值得進入 6-seed confirm

#### 17.6.2 H1 6-seed confirm：整體仍停在 Level 2

固定條件：

1. `popularity_mode=sampled`
2. `payoff_mode=matrix_ab`
3. `a=1.0`, `b=0.9`, `matrix-cross-coupling=0.20`
4. `players=300`, `rounds=8000`
5. `seeds={45,47,49,51,53,55}`
6. `series=p`, `burn_in=2400`, `tail=1000`
7. `selection_strength=0.06`, `corr_threshold=0.09`, `eta=0.55`

輸出：

1. `outputs/h1_confirm_k1_s6_r8000.csv`
2. `outputs/h1_confirm_k3_s6_r8000.csv`
3. `outputs/h1_confirm_k5_s6_r8000.csv`

摘要：

| kernel | hits(L3) | mean_level | mean_stage3_score | mean_turn_strength | mean_env_gamma |
|---|---:|---:|---:|---:|---:|
| 1 | 0/6 | 2.000 | 0.4969 | 0.0009439 | -1.214e-4 |
| 3 | 0/6 | 2.000 | 0.5083 | 0.0009409 | -1.081e-4 |
| 5 | 0/6 | 2.000 | 0.5070 | 0.0009440 | -1.053e-4 |

判讀：

1. 三個 kernel 都沒有把單點條件推到 `hits >= 2/6`
2. `kernel=3` 的 `mean_stage3_score` 最佳
3. `kernel=5` 的 `mean_env_gamma` 最接近 0，表示衰減略弱
4. 因此 H1 目前應被視為「弱正向訊號」，而不是已打開 robust basin

#### 17.6.3 H1 協定化 rho sweep：曲線有上移，但 `P(Level3)` 仍為 0

固定條件與 confirm 相同，僅把 `selection_strength` 掃成：

1. `k-grid = 0.04:0.10:0.01`
2. 比較 `memory_kernel ∈ {1,3,5}`

輸出：

1. `outputs/h1_rho_curve_k1_s6_r8000.csv`
2. `outputs/h1_rho_curve_k3_s6_r8000.csv`
3. `outputs/h1_rho_curve_k5_s6_r8000.csv`

重點表：

| k | rho-1 | P(L3) k1 | P(L3) k3 | P(L3) k5 | mean_s3 k1 | mean_s3 k3 | mean_s3 k5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.04 | +0.000056 | 0.000 | 0.000 | 0.000 | 0.4984 | 0.5093 | 0.5064 |
| 0.05 | +0.000295 | 0.000 | 0.000 | 0.000 | 0.4981 | 0.5060 | 0.5053 |
| 0.06 | +0.000624 | 0.000 | 0.000 | 0.000 | 0.4969 | 0.5083 | 0.5070 |
| 0.07 | +0.001041 | 0.000 | 0.000 | 0.000 | 0.4886 | 0.5068 | 0.5074 |
| 0.08 | +0.001547 | 0.000 | 0.000 | 0.000 | 0.4920 | 0.5025 | 0.5101 |
| 0.09 | +0.002139 | 0.000 | 0.000 | 0.000 | 0.4928 | 0.5035 | 0.5114 |
| 0.10 | +0.002817 | 0.000 | 0.000 | 0.000 | 0.4966 | 0.5055 | 0.5070 |

判讀：

1. 三條曲線都尚未出現 `P(Level3) > 0`
2. `kernel=3` 幾乎在整條曲線上都高於 `kernel=1`
3. `kernel=5` 在較高 k（約 `0.08~0.09`）時，`mean_stage3_score` 略高於 `kernel=3`
4. 因此 H1 的最佳描述不是「已找到新 basin」，而是「記憶長度讓 Stage3 曲線整體上移，但 uplift 仍不足以跨過 Level3 門檻」

#### 17.6.4 H1 下一步判定

基於目前數據，下一步優先序應調整為：

1. 若要繼續追 H1，優先測較長觀測窗（例如更長 rounds 或 `tail=3000`）

### 17.7 H2 起草版：最小 smoke plan（2026-03-29）

H1 的最新結果已把問題收斂得很清楚：`memory_kernel` 會把 Stage3 曲線上移，但還不足以打開 basin。因此 H2 的角色不是再做一次大 sweep，而是先回答一個更尖銳的問題：

1. 非線性 regime switch 能不能在相同 baseline 上產生比 H1 更乾淨的方向訊號
2. 若不能，H2 就應該在 smoke 階段停止，不進 6-seed confirm

#### 17.7.1 H2 smoke 的固定 baseline

全部 H2 smoke 都先固定在下列 baseline，避免一次引入太多自由度：

1. `payoff_mode=threshold_ab`
2. `popularity_mode=expected`
3. `evolution_mode=mean_field`
4. `payoff_lag=1`
5. `memory_kernel=3`
6. `a=1.0`, `b=0.9`, `matrix-cross-coupling=0.20`
7. `selection_strength=0.06`
8. `init_bias=0.12`
9. `rounds=260`, `seed=45`

選 `memory_kernel=3` 的理由不是它已證明 basin 成立，而是它在 H1 中給出最穩定的整體 uplift，因此適合作為 H2 的低成本基線。

#### 17.7.2 Smoke 0：退化等價性檢查（必做）

第一個 smoke 不是要找訊號，而是先驗證 H2 沒有破壞既有 `matrix_ab` 行為。

```bash
cd /home/user/personality-dungeon
export PYTHON_BIN=./venv/bin/python

$PYTHON_BIN -m simulation.run_simulation \
    --payoff-mode threshold_ab \
    --popularity-mode expected \
    --evolution-mode mean_field \
    --payoff-lag 1 \
    --memory-kernel 3 \
    --threshold-theta 0.40 \
    --threshold-a-hi 1.0 --threshold-b-hi 0.9 \
    --rounds 260 --seed 45 \
    --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
    --selection-strength 0.06 --init-bias 0.12 \
    --out outputs/h2_smoke_noop_theta0p40_seed45.csv
```

Smoke 0 的通過條件：

1. 與 `outputs/h1_smoke_kernel3_seed45.csv` 的 `p_*` / `w_*` 時序在數值上等價，至少不應出現肉眼可見偏移
2. `cycle_metrics` 的 `level`、`max_amp`、`best_corr`、`turn_strength` 不應出現結構性差異

若 Smoke 0 不通過，H2 直接停在實作修正，不進下一步。

#### 17.7.3 Smoke 1：低門檻切換

低門檻版本用來測試「只要 A/D 共存量一升上來，就提早切到 high regime」是否能放大旋轉。

```bash
$PYTHON_BIN -m simulation.run_simulation \
    --payoff-mode threshold_ab \
    --popularity-mode expected \
    --evolution-mode mean_field \
    --payoff-lag 1 \
    --memory-kernel 3 \
    --threshold-theta 0.40 \
    --threshold-a-hi 1.10 --threshold-b-hi 1.00 \
    --rounds 260 --seed 45 \
    --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
    --selection-strength 0.06 --init-bias 0.12 \
    --out outputs/h2_smoke_theta0p40_seed45.csv
```

#### 17.7.4 Smoke 2：高門檻切換

高門檻版本用來測試「只有在 A/D 共存量夠高時才切 regime」能否避免過早把系統推回另一個平坦區。

```bash
$PYTHON_BIN -m simulation.run_simulation \
    --payoff-mode threshold_ab \
    --popularity-mode expected \
    --evolution-mode mean_field \
    --payoff-lag 1 \
    --memory-kernel 3 \
    --threshold-theta 0.55 \
    --threshold-a-hi 1.10 --threshold-b-hi 1.00 \
    --rounds 260 --seed 45 \
    --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
    --selection-strength 0.06 --init-bias 0.12 \
    --out outputs/h2_smoke_theta0p55_seed45.csv
```

#### 17.7.5 H2 smoke 的判讀與 stop rule

三個 smoke 一律只跟 H1 的 `kernel=3` baseline 比，不另開新 baseline。

Primary pass：

1. `turn_strength` 相對 `outputs/h1_smoke_kernel3_seed45.csv` 提升 `>= 15%`
2. `max_amp` 不下降，且沒有數值崩塌或明顯卡死在單點

Secondary pass：

1. `best_corr` 維持高值
2. 視覺上相圖旋轉更乾淨，且 `env_gamma` 在後續 confirm 候選中有望更接近 0

Stop rule：

1. Smoke 0 不等價就停，先修 H2 實作
2. 若 Smoke 1 與 Smoke 2 都沒有超過 H1 `kernel=3` baseline，就停止 H2，不進 6-seed confirm
3. 只有在至少一個 threshold 條件明顯優於 baseline 時，才進 6-seed / 8000-round confirm

#### 17.7.6 若 H2 smoke 通過，最小 confirm 路徑

只有 smoke 通過才啟動 confirm，且第一輪只擴最小組合：

1. `theta ∈ {0.40, 0.55}` 中僅保留 smoke 較優者
2. `memory_kernel` 固定 3
3. `a_hi=1.10`, `b_hi=1.00`
4. 其餘 protocol 完全沿用 H1 confirm：`sampled`, `players=300`, `rounds=8000`, `seeds={45,47,49,51,53,55}`

這樣 H2 的第一輪成本會被鎖在「1 個 no-op 對照 + 2 個 smoke + 最多 1 個 confirm 條件」，避免還沒看出 regime switch 是否有效，就又滑回大範圍調參。

### 17.8 H2 第一輪 smoke 結果（2026-03-29）

本節記錄 H2 在 spec 起草後的第一輪實作與 smoke 結果。

#### 17.8.1 實作範圍

本輪已完成：

1. `simulation.run_simulation` CLI 支援 `threshold_ab`
2. `DungeonAI` sampled payoff 支援 `threshold_ab`
3. mean-field 路徑也支援 `threshold_ab`
4. no-op 等價性回歸測試已加入並通過
5. `simulation/seed_stability.py` 已支援 H2 參數穿透與 per-seed provenance
6. `simulation/rho_curve.py` 已支援 H2 參數穿透與 sweep provenance

尚未完成：

1. 更長視窗的 H2 confirm（例如 `rounds=8000`）
2. H2 的正式 rho sweep / crossing protocol

#### 17.8.2 Smoke 0：no-op 等價性通過

命令：

1. `payoff_mode=threshold_ab`
2. `theta=0.40`
3. `a_hi=1.0`, `b_hi=0.9`
4. 其餘條件完全對齊 H1 `kernel=3` baseline

輸出：

1. `outputs/h2_smoke_noop_theta0p40_seed45.csv`

結果：

1. 與 `outputs/h1_smoke_kernel3_seed45.csv` 的 `p_*` / `w_*` 最大絕對差為 `0`
2. cycle 指標也完全相同：

| run | level | max_amp | best_corr | stage3_score | turn_strength |
|---|---:|---:|---:|---:|---:|
| H1 baseline k=3 | 3 | 0.339605 | 0.998404 | 1.000000 | 4.7846e-07 |
| H2 no-op | 3 | 0.339605 | 0.998404 | 1.000000 | 4.7846e-07 |

判讀：`threshold_ab` 的退化等價性成立，可以進 active smoke。

#### 17.8.3 Smoke 1 / Smoke 2：active threshold 結果

輸出：

1. `outputs/h2_smoke_theta0p40_seed45.csv`
2. `outputs/h2_smoke_theta0p55_seed45.csv`

摘要：

| run | level | max_amp | best_corr | stage3_score | turn_strength | vs H1 turn |
|---|---:|---:|---:|---:|---:|---:|
| H1 baseline k=3 | 3 | 0.339605 | 0.998404 | 1.000000 | 4.7846e-07 | baseline |
| H2 theta=0.40 | 3 | 0.336352 | 0.997613 | 1.000000 | 6.5196e-07 | +36.3% |
| H2 theta=0.55 | 3 | 0.346834 | 0.997928 | 0.996124 | 6.2635e-07 | +30.9% |

判讀：

1. `theta=0.40` 的 turning 訊號明顯高於 baseline，但 `max_amp` 輕微下降，因此不算完全通過原先的 primary pass
2. `theta=0.55` 同時滿足：
    - `turn_strength` 相對 baseline 提升超過 `15%`
    - `max_amp` 沒有下降，反而約提升 `+2.1%`
3. 依 runbook 原先的 stop rule，`theta=0.55` 應被保留為 H2 的最小 confirm 候選

#### 17.8.4 H2 當前結論

1. H2 已不再是紙上 spec，而是已完成單跑與 smoke 路徑的實作
2. no-op 對照已證明 `threshold_ab` 沒有破壞既有 `matrix_ab` baseline
3. 第一輪 active smoke 顯示 H2 不是零訊號；其中 `theta=0.55` 明顯優於 H1 `kernel=3` baseline
4. 因此下一步不該回頭擴大 smoke，而是把 `theta=0.55`, `a_hi=1.10`, `b_hi=1.00`, `memory_kernel=3` 帶進最小 6-seed confirm

#### 17.8.5 H2 最小 6-seed confirm 與 batch/sweep 落地（2026-03-30）

本輪新增：

1. `simulation/seed_stability.py` 已接受：
    - `--payoff-mode threshold_ab`
    - `--threshold-theta`
    - `--threshold-a-hi`
    - `--threshold-b-hi`
2. `simulation/rho_curve.py` 已接受相同 H2 旗標，且 sweep CSV 會保留 threshold provenance 欄位
3. 已補上 batch / sweep 路徑的 no-op 等價性回歸測試

聚焦回歸：

1. `./venv/bin/pytest -q tests/test_seed_stability_parse.py tests/test_rho_curve_csv_provenance_columns.py tests/test_dungeon_ai.py tests/test_simulate_series_window.py`
2. 結果：`19 passed`

最小 6-seed confirm：

1. 條件：`theta=0.55`, `a_hi=1.10`, `b_hi=1.00`, `memory_kernel=3`
2. 命令輸出：`outputs/h2_confirm_theta0p55_k3_seed0_5.csv`
3. 結果：
    - `level_counts = {0: 0, 1: 0, 2: 0, 3: 6}`
    - `P(level>=2) = 1.000`
    - `mean_env_gamma = 0`

最小 rho sweep smoke：

1. 命令輸出：`outputs/h2_rho_curve_smoke_theta0p55.csv`
2. 單點結果：`N=80`, `k=0.06`, `rho-1=+6.238e-04`, `P(L3)=1.000`

判讀：

1. `theta=0.55` 不只在單 seed smoke 有 uplift，最小 6-seed confirm 也維持 `6/6` Level 3
2. H2 已不再卡在單跑驗證，batch confirm 與 rho sweep 管線都可直接使用
3. 因此 H2 的下一步應該是 protocolized rho sweep / longer-horizon confirm，而不是回頭補 plumbing

#### 17.8.6 H2 協定化 rho sweep：長視窗 sampled 協定下仍未打開 Level 3（2026-03-30）

為了避免把 mean-field smoke 的 uplift 誤判成 robust basin，本輪直接沿用 H1 的長視窗 rho protocol 跑 H2 主 sweep：

1. `payoff_mode=threshold_ab`
2. `theta=0.55`, `a_hi=1.10`, `b_hi=1.00`
3. `popularity_mode=sampled`, `evolution_mode=sampled`
4. `players=300`, `rounds=8000`, `seeds={45,47,49,51,53,55}`
5. `series=p`, `burn_in=2400`, `tail=1000`
6. `corr_threshold=0.09`, `eta=0.55`, `stage3_method=turning`

輸出：

1. `outputs/sweeps/rho_curve/h2_rho_curve_theta0p55_ahi1p10_bhi1p00_k3_N300_k0p04_0p10_s0p01_R8000_S6_tail1000.csv`
2. `outputs/sweeps/rho_curve/h2_rho_curve_theta0p55_ahi1p10_bhi1p00_k3_N300_k0p12_0p20_s0p02_R8000_S6_tail1000.csv`

主 sweep（`k=0.04:0.10:0.01`）結果摘要：

| k | rho-1 | P(L3) | P(L>=2) | mean_stage3_score |
|---|---:|---:|---:|---:|
| 0.04 | +5.585e-05 | 0.000 | 1.000 | 0.510 |
| 0.05 | +2.951e-04 | 0.000 | 1.000 | 0.506 |
| 0.06 | +6.238e-04 | 0.000 | 1.000 | 0.505 |
| 0.07 | +1.041e-03 | 0.000 | 1.000 | 0.504 |
| 0.08 | +1.547e-03 | 0.000 | 1.000 | 0.502 |
| 0.09 | +2.139e-03 | 0.000 | 1.000 | 0.503 |
| 0.10 | +2.817e-03 | 0.000 | 1.000 | 0.506 |

高 k scout（`k=0.12:0.20:0.02`）結果摘要：

| k | rho-1 | P(L3) | P(L>=2) | mean_stage3_score |
|---|---:|---:|---:|---:|
| 0.12 | +4.427e-03 | 0.000 | 1.000 | 0.501 |
| 0.14 | +6.363e-03 | 0.000 | 1.000 | 0.502 |
| 0.16 | +8.615e-03 | 0.000 | 1.000 | 0.506 |
| 0.18 | +1.117e-02 | 0.000 | 1.000 | 0.502 |
| 0.20 | +1.400e-02 | 0.000 | 1.000 | 0.506 |

判讀：

1. H2 在 mean-field / expected smoke 中的 uplift，沒有轉化成 sampled / 8000-round 協定下的 `P(Level3)>0`
2. 整段 `k=0.04~0.20` 都維持 `P(Level3)=0`，因此目前沒有 crossing band 可 refine
3. H2 的長視窗表現較接近「穩定 Level 2 plateau」，而不是已打開 robust Level 3 basin
4. 因此 H2 的下一步不應直接做 crossing refine；若要繼續，應先改 long-horizon confirm 假說或重新檢查 sampled 與 mean-field 的機制落差

#### 17.8.7 H2 longer-horizon confirm：`k=0.06` 單點在 sampled 協定下為 6/6 Level 2（2026-03-30）

為了把「正式 rho sweep 的單一 k=0.06 結論」獨立固化成 confirm 證據，本輪另外用 `seed_stability.py` 跑了一份對應的長視窗 confirm：

1. `payoff_mode=threshold_ab`
2. `theta=0.55`, `a_hi=1.10`, `b_hi=1.00`
3. `popularity_mode=sampled`, `evolution_mode=sampled`
4. `players=300`, `rounds=8000`, `selection_strength=0.06`
5. `seeds={45,47,49,51,53,55}`
6. `series=p`, `burn_in=2400`, `tail=1000`, `corr_threshold=0.09`, `eta=0.55`

輸出：

1. `outputs/h2_confirm_theta0p55_k3_s6_r8000_sampled.csv`

摘要：

1. `level_counts = {0: 0, 1: 0, 2: 6, 3: 0}`
2. `P(level>=2) = 1.000`
3. `mean_env_gamma = -1.00494e-04`
4. 六個 seed 全部 `stage3_passed = False`

判讀：

1. H2 在長視窗 sampled 協定下沒有維持先前短窗 confirm 的 `6/6 Level 3`
2. 這與 17.8.6 的 rho sweep 結果一致，表示差異主因是 protocol 本身，而不是某個單點 k 的偶然波動
3. 因此目前對 H2 最準確的描述是：短窗 mean-field 有 uplift，但長窗 sampled 仍穩定停在 Level 2

#### 17.8.8 H2 sampled vs mean-field 機制差異診斷（2026-03-30）

為了避免把 H2 的失敗只描述成「sampled 比較 noisy」，本輪對同一組 H2 參數補做了狀態診斷：

1. `mean_field_short`: `expected + mean_field`, `rounds=260`, `seed=45`
2. `sampled_short`: `sampled + sampled`, `rounds=260`, `seed=45`
3. `sampled_long_seed45`: `sampled + sampled`, `rounds=8000`, `seed=45`

輸出：

1. `outputs/h2_mode_gap_diag_theta0p55.json`
2. `outputs/h2_mode_gap_diag_theta0p55.md`

關鍵結果：

| case | cycle_level | stage3_score | max_amp | q_AD mean | high-regime share | regime switches |
|---|---:|---:|---:|---:|---:|---:|
| mean_field_short | 3 | 0.9961 | 0.3468 | 0.6363 | 0.6846 | 3 |
| sampled_short | 2 | 0.5234 | 0.1667 | 0.6654 | 1.0000 | 0 |
| sampled_long_seed45 | 2 | 0.5340 | 0.1933 | 0.6626 | 1.0000 | 0 |

機制判讀：

1. sampled 路徑不是「很少進 high regime」，而是幾乎從頭到尾都待在 high regime
2. 因此在 sampled 下，`threshold_ab` 實際上退化成「常態化的 high-regime matrix」，不再提供真正的 regime switching
3. 相反地，mean-field 短窗中 `high-regime share ≈ 0.685` 且有 `3` 次切換，表示 uplift 主要來自 regime 來回切換，而不是單純把 `(a,b)` 提高
4. 這也解釋了為什麼 sampled 長視窗不是直接崩塌，而是穩定停在 Level 2 plateau：threshold 結構在該路徑上被鎖死成單一高檔 regime

因此若還要繼續 H2，下一步不該再掃 `k` band，而應改成下面二選一：

1. 直接重寫 H2 的切換變數或 hysteresis，讓 sampled 路徑真的出現 regime switching
2. 停止 H2 crossing 線，轉往 H3 或其他新結構假說

#### 17.8.9 H2.1 可選 hysteresis band：實作已落地，但第一輪 sampled smoke 幾乎無效（2026-03-30）

為了直接測試「sampled 把 H2 鎖死在 high regime」是否可由切換慣性解決，本輪在 `threshold_ab` 上新增可選 hysteresis band：

1. `--threshold-theta-low`
2. `--threshold-theta-high`
3. 若兩者省略，或設成與 `--threshold-theta` 相同，必須完全退化回原始 H2

本輪實作已接到：

1. `simulation/run_simulation.py`
2. `dungeon/dungeon_ai.py`
3. `simulation/seed_stability.py`
4. `simulation/rho_curve.py`

聚焦回歸：

1. `./venv/bin/pytest -q tests/test_dungeon_ai.py tests/test_simulate_series_window.py tests/test_seed_stability_parse.py tests/test_rho_curve_csv_provenance_columns.py`
2. 結果：`22 passed`

第一輪 sampled smoke scout（`seed=45`, `rounds=260`, `sampled + sampled`）輸出：

1. `outputs/h21_hysteresis_sampled_short_scout.json`

測試 band：

1. `(0.58, 0.68)`
2. `(0.60, 0.68)`
3. `(0.62, 0.70)`
4. `(0.64, 0.72)`

摘要：

| theta_low | theta_high | high_share | switches | level | stage3_score | max_amp |
|---|---:|---:|---:|---:|---:|---:|
| 0.58 | 0.68 | 1.000 | 0 | 2 | 0.5234 | 0.1667 |
| 0.60 | 0.68 | 1.000 | 0 | 2 | 0.5234 | 0.1667 |
| 0.62 | 0.70 | 0.773 | 5 | 2 | 0.5234 | 0.1667 |
| 0.64 | 0.72 | 0.000 | 0 | 2 | 0.5214 | 0.1667 |

補充數值對照：

1. 原始 H2 與 `(0.62, 0.70)` hysteresis 版本在同一條 sampled 短窗上的時序最大差只到 `5.20e-04`
2. 終點 `p_*` 與 `w_*` 幾乎完全相同

判讀：

1. hysteresis band 的確能讓 sampled 路徑在某些 band 下重新出現 regime switches
2. 但第一輪 smoke 顯示，這些切換對整體動力學的影響極小，尚未把 sampled 路徑從 Level 2 plateau 推開
3. 因此 H2.1 證明了「缺乏切換慣性」不是唯一瓶頸；下一步更合理的是改切換 trigger，而不是只繼續調 hysteresis band

#### 17.8.10 H2.2 trigger + slow state：主流程已接通，第一輪 ad_product 短 scout 仍是 Level 2 plateau（2026-03-30）

依 17.8.9 的結論，本輪把 H2 往 H2.2 推進，最小版本只做兩件事：

1. 把切換 trigger 從固定 `x_A + x_D` 擴成可選 `ad_share` / `ad_product`
2. 在 trigger 與 regime 之間加入一個可調慢狀態 `z(t)`，由 `--threshold-state-alpha` 控制更新速度

本輪已接通的檔案：

1. `simulation/run_simulation.py`
2. `dungeon/dungeon_ai.py`
3. `simulation/seed_stability.py`
4. `simulation/rho_curve.py`

聚焦回歸：

1. `./venv/bin/pytest -q tests/test_dungeon_ai.py tests/test_simulate_series_window.py tests/test_rho_curve_csv_provenance_columns.py`
2. 結果：`20 passed`

第一輪 sampled short scout：

1. 指令核心：`threshold_trigger=ad_product`, `threshold_state_alpha=0.2`, `theta=0.45`
2. 輸出：`outputs/h22_ad_product_alpha0p2_short_scout.csv`

摘要：

1. `players=300`, `rounds=1500`, `seeds={45,47,49}`
2. `P(Level>=2)=1.000`
3. `P(Level3)=0.000`
4. `level_counts={0:0, 1:0, 2:3, 3:0}`
5. `mean_env_gamma=+1.77946e-04`，短窗下看不出明顯成長包絡

目前判讀：

1. H2.2 的 CLI、mean-field、sampled、batch、CSV provenance 都已接通，且向後相容回歸通過
2. 第一個 `ad_product + alpha=0.2` 短窗 scout 至少證明新機制有被真正執行，不再只是 spec 或 sampled-only patch
3. 但就目前這組參數，sampled 路徑仍停在穩定的 Level 2 plateau，尚未看到 Level 3 uplift
4. 因此下一步不該直接做大規模正式 sweep，而是先做更有方向的 H2.2 小範圍機制掃描，例如固定 `ad_product` 後掃 `theta` 與 `alpha`

#### 17.8.11 H2.2 ad_product 短窗網格掃描：16 組 theta × alpha 全數停在同一個 Level 2 plateau（2026-03-30）

依 17.8.10 的建議，本輪固定 `threshold_trigger=ad_product`，先做最小短窗網格掃描，確認這條 trigger 幾何是否存在任何明顯較佳的 `theta/alpha` 區域。

掃描設定：

1. `theta ∈ {0.35, 0.45, 0.55, 0.65}`
2. `alpha ∈ {0.10, 0.20, 0.35, 0.50}`
3. 其餘協定固定為：`players=300`, `rounds=1500`, `seeds={45,47,49}`, `series=p`, `burn_in=300`, `tail=400`
4. 參數固定：`a=1.0`, `b=0.9`, `a_hi=1.1`, `b_hi=1.0`, `cross=0.2`, `selection_strength=0.06`, `memory_kernel=3`, `init_bias=0.12`

輸出：

1. 摘要表：`outputs/h22_ad_product_short_scan_summary.tsv`
2. 單點 CSV：`outputs/h22_scan_short/`

結果：

1. 16 組全部得到完全相同的 aggregate 結果
2. 每一組都是 `level_counts={0:0, 1:0, 2:3, 3:0}`
3. 因此每一組都是 `P(Level>=2)=1.000`, `P(Level3)=0.000`
4. 在這個短窗協定下，看不到任何一組比其他組更接近 Level 3 crossing

判讀：

1. 這不是單一參數挑錯，而是目前 `ad_product` 線在這個局部區域內，對 sampled 動力學幾乎沒有可辨識的分岔效果
2. 既然 `theta` 與 `alpha` 同時掃過一個合理的小區域，結果卻完全不分家，表示短窗下至少沒有明顯值得進一步長窗 confirm 的候選點
3. 因此 H2.2 的下一步不應優先擴大 `ad_product` 的正式 sweep；比較合理的是改掃另一個 trigger family，或改 high-regime matrix 幾何本身

#### 17.8.12 H2.2 ad_product 幾何掃描：連 high-regime matrix 自身也沒有打破 Level 2 plateau（2026-03-30）

為了把 17.8.11 的負面結果再拆乾淨一層，本輪固定 `threshold_trigger=ad_product`, `theta=0.45`, `alpha=0.20`，只掃 high-regime matrix 幾何，檢查瓶頸是否其實出在切換後的 payoff 結構。

掃描設定：

1. `a_hi ∈ {1.00, 1.10, 1.20, 1.30}`
2. `b_hi ∈ {0.80, 0.90, 1.00, 1.10}`
3. 其餘協定固定為：`players=300`, `rounds=1500`, `seeds={45,47,49}`, `series=p`, `burn_in=300`, `tail=400`

輸出：

1. 摘要表：`outputs/h22_ad_product_geometry_short_scan.tsv`
2. 單點 CSV：`outputs/h22_geometry_short/`

結果：

1. 16 組全部得到完全相同的 aggregate 結果
2. 每一組都是 `level_counts={0:0, 1:0, 2:3, 3:0}`
3. 每一組都是 `P(Level>=2)=1.000`, `P(Level3)=0.000`
4. 摘要表的唯一 outcome 數量為 `1`

判讀：

1. 到這一步為止，`ad_product` 線下可調的三個最直接自由度都已被短窗否定：`theta`、`alpha`、以及 `(a_hi,b_hi)` 幾何
2. 這代表目前 H2.2 的問題已經不是單一 scalar 參數沒調對，也不像只是 high-regime matrix 太弱
3. 較合理的結論是：`ad_product` 這個 trigger family 在現有 H2.2 契約下，對 sampled dynamics 幾乎沒有可利用的結構敏感度
4. 因此若還要繼續救 H2.2，優先順序應該改成：
5. 先停掉 `ad_product` 線的進一步短窗/長窗 sweep
6. 轉去設計新的 trigger family，或直接承認 H2 需要更大幅度的結構改寫

#### 17.9 H3 最小版已落地：hetero smoke 只有弱 uplift，3-seed 短 scout 尚未優於 baseline（2026-03-30）

依 17.8.12 的結論，本輪停止在 H2 上繼續做局部修補，直接切到 H3 的最小版本：per-strategy selection strength。

本輪落地內容：

1. `evolution/replicator_dynamics.py`：`replicator_step()` 新增可選 `strategy_selection_strengths`
2. `simulation/run_simulation.py`：新增 `--evolution-mode hetero` 與 `--strategy-selection-strengths`
3. `simulation/seed_stability.py`：batch confirm 可直接跑 H3 `hetero`
4. 退化契約已用回歸測試鎖住：等強度向量必須完全退化回既有 sampled 更新

聚焦回歸：

1. `./venv/bin/pytest -q tests/test_evolution.py tests/test_simulate_series_window.py tests/test_seed_stability_parse.py`
2. 結果：`18 passed`

第一輪單 seed smoke：

1. baseline：`outputs/h3_baseline_sampled_seed45.csv`
2. H3：`outputs/h3_smoke_hetero_seed45.csv`
3. 候選向量：`k=(0.06,0.06,0.04)`

單 seed 指標對照：

1. baseline：`level=2`, `stage3_score=0.4919`, `turn_strength=0.00546`
2. hetero：`level=2`, `stage3_score=0.4959`, `turn_strength=0.00551`

這代表 H3 不是完全 no-op，但 uplift 很小，所以又做了一輪最小 smoke 向量掃描：

1. 掃描輸出：`outputs/h3_smoke_scan/`
2. 測過的向量：`(0.06,0.06,0.04)`, `(0.06,0.06,0.03)`, `(0.07,0.06,0.04)`, `(0.06,0.07,0.04)`, `(0.05,0.05,0.03)`, `(0.07,0.07,0.04)`
3. 最佳單 seed 候選是 `k=(0.07,0.06,0.04)`，`stage3_score=0.4960`

接著做最小 3-seed 短 scout：

1. baseline：`outputs/h3_short_baseline_sampled.csv`
2. H3 候選：`outputs/h3_short_scout_k070604.csv`
3. 協定：`players=300`, `rounds=1500`, `seeds={45,47,49}`, `series=p`, `burn_in=300`, `tail=400`

3-seed aggregate：

1. baseline 與 hetero 都是 `level_counts={0:0,1:0,2:3,3:0}`
2. 兩者都沒有 Level 3 (`P(Level3)=0.000`)
3. `mean_turn_strength` 只從 `0.0010496` 微升到 `0.0010558`
4. `mean_stage3_score` 反而從 `0.5110` 微降到 `0.5089`

目前判讀：

1. H3 最小版已正式落地，而且 smoke 確認它不是字元級 no-op
2. 但目前只看到非常弱的單 seed uplift，尚未轉成跨 seed 的 aggregate 改善
3. 因此 H3 現階段屬於「可保留、但不能直接升級到長窗 confirm」的弱陽性線索
4. 若要繼續推 H3，下一步更合理的是掃更有結構意義的 `k` 向量家族，或正式升級到 H3.1 固定子族群，而不是直接做大規模 confirm

#### 17.10 H3.1 固定子族群第一個 structured family 失敗：balanced-anchor 家族 18 組全數停在同一個 Level 2 plateau（2026-03-30）

依 17.9 的結論，本輪沒有再做零碎單點 smoke，而是直接把 H3 升級到 H3.1：在異質更新之外，再加入 frozen subgroup。

本輪落地內容：

1. `simulation/run_simulation.py`：新增 `--fixed-subgroup-share` 與 `--fixed-subgroup-weights`
2. `simulation/seed_stability.py`：batch / short scout 路徑可直接傳入 H3.1 參數
3. 回歸鎖定兩個核心契約：`fixed_subgroup_share=0` 必須是 no-op；`share>0` 但缺少 fixed weights 必須報錯
4. sampled 路徑的 `w_*` 語意同步校正為「全體玩家 post-update strategy weights 的平均值」，避免在 mixed population 下誤把單一 adaptive dict 當成全體狀態

聚焦回歸：

1. `./venv/bin/pytest -q tests/test_evolution.py tests/test_simulate_series_window.py tests/test_seed_stability_parse.py`
2. 結果：`21 passed`

第一個 structured family：balanced-anchor frozen subgroup

1. 固定子族群權重：`(0.8, 0.8, 1.4)`
2. adaptive `k` 向量家族：`(k, k, k-δ)`
3. 掃描格點：`k ∈ {0.06, 0.07}`、`δ ∈ {0.01, 0.02, 0.03}`、`share ∈ {0.10, 0.20, 0.30}`
4. 協定：`players=300`, `rounds=1500`, `seeds={45,47,49}`, `series=p`, `burn_in=300`, `tail=400`
5. 輸出 summary：`outputs/h31_balanced_anchor_family_scan.tsv`
6. 單次輸出：`outputs/h31_family_scan/*.csv`

aggregate 結果：

1. 18 組參數全部得到同一個 outcome：`P(Level>=2)=1.000`, `P(Level3)=0.000`
2. 全部都是 `level_counts={0:0,1:0,2:3,3:0}`
3. unique outcome count = 1，代表這條 family 在目前短窗 sampled protocol 下完全平坦

目前判讀：

1. H3.1 已經不是「未實作猜想」，而是有 formal contract、runtime、batch 與 regression 的真機制
2. 但第一個 balanced-anchor frozen-subgroup family 沒有比 H3 最小版更好，至少在 3-seed 短窗協定下完全沒有打開 Level 3 basin
3. 因此不應立刻把這條 family 升級到更長 confirm 或 rho sweep
4. 若還要繼續推 H3.1，下一步應改變 frozen anchor family 本身，而不是在這個 `(0.8,0.8,1.4)` 家族上再做更細局部掃描

#### 17.11 H3.2 非對稱子族群耦合也先撞牆：anchor-gap coupling 短 scout 6 組全數仍是同一個 Level 2 plateau（2026-03-30）

依 17.10 的結論，本輪不再細修 H3.1 的 frozen anchor family，而是直接走更強的 H3.2：讓 fixed subgroup 不只是被動存在，而是直接成為 adaptive subgroup 的 payoff anchor。

本輪落地內容：

1. `SDD.md`：新增 H3.2 `fixed_subgroup_coupling_strength` 契約
2. `evolution/replicator_dynamics.py`：新增 pure helper，計算 `A(a,b) @ (x_fix - x_ad)` 的 subgroup payoff shift
3. `simulation/run_simulation.py`：sampled 路徑在每輪更新前，對 adaptive subgroup reward 套用 H3.2 shift
4. `simulation/seed_stability.py`：batch / short scout 路徑可直接傳入 `--fixed-subgroup-coupling-strength`
5. 回歸新增 H3.2 no-op、非法參數、batch 傳遞與非 no-op 行為測試

聚焦回歸：

1. `./venv/bin/pytest -q tests/test_evolution.py tests/test_simulate_series_window.py tests/test_seed_stability_parse.py`
2. 結果：`27 passed`

第一輪 H3.2 structured scout：anchor-gap coupling

1. 固定子族群權重：`(0.8, 0.8, 1.4)`
2. 掃描 share：`{0.10, 0.20}`
3. 掃描 coupling 強度：`λ ∈ {0.15, 0.30, 0.45}`
4. 協定：`players=300`, `rounds=1500`, `seeds={45,47,49}`, `series=p`, `burn_in=300`, `tail=400`
5. summary：`outputs/h32_anchor_gap_short_scan.tsv`
6. 單次輸出：`outputs/h32_anchor_gap_short/*.csv`

aggregate 結果：

1. 6 組參數全部得到同一個 outcome：`P(Level>=2)=1.000`, `P(Level3)=0.000`
2. 全部都是 `level_counts={0:0,1:0,2:3,3:0}`
3. unique outcome count = 1，代表第一版 H3.2 anchor-gap coupling 在目前短窗 sampled protocol 下也完全平坦

目前判讀：

1. H3.2 已正式落地到 spec / runtime / batch / regression，但第一輪 short scout 沒有提供任何 aggregate uplift
2. 這代表問題可能不是 frozen subgroup 對 adaptive subgroup 的耦合還不夠強，而是這整類單一 frozen-anchor 機制本身缺乏足夠的 phase-shaping 自由度
3. 因此不應在目前這個 `(0.8,0.8,1.4)` + `anchor-gap` 家族上再做更細 λ 掃描
4. 若還要繼續推 subgroup 線，下一步應直接改成更高階的雙向或狀態依賴 coupling，而不是延長這條單向 anchor 線

#### 17.12 H3.3B 第一版先落成最保守的 state-dependent coupling 契約與 runtime（2026-03-31）

在 17.11 已確認 H3.2 的常數型單向 anchor-gap coupling 完全平坦後，H3.3B 先不改「誰推誰」，而是改「何時推、推多強」。

第一版刻意鎖到最保守版本：

1. state signal 只允許 `gap_norm`
2. coupling 只作用在 adaptive subgroup
3. gate 採 `sigmoid(beta * (z - theta))`
4. 其中 `z(t) = ||x_fix - x_ad||_2`
5. 第一版明確與 H3.2 互斥，避免常數耦合與狀態耦合混用

本輪落地內容：

1. `SDD.md`：補上 H3.3B 正式契約
2. `evolution/replicator_dynamics.py`：新增 state-dependent subgroup payoff helper
3. `simulation/run_simulation.py`：新增 H3.3B config / CLI / validation / sampled runtime 掛點
4. `simulation/seed_stability.py`：新增 H3.3B batch 傳遞與 CLI
5. 回歸新增 H3.3B helper、no-op、非法參數、H3.2/H3.3B 互斥、以及 sampled non-no-op 測試

聚焦回歸：

1. `./venv/bin/pytest -q tests/test_evolution.py tests/test_simulate_series_window.py tests/test_seed_stability_parse.py`
2. 結果：`34 passed`

目前只完成落地，尚未跑第一輪 H3.3B scout，因此還沒有新的 Level 3 證據；下一步應直接做最小 gap-norm short scout，而不是再改 signal 家族。

#### 17.13 H3.3B 第一輪 gap-norm gating short scout 也沒有打破 plateau（2026-03-31）

接在 17.12 之後，本輪直接用與 H3.2 相同的短窗 sampled protocol，先驗證 H3.3B 最保守的 `gap_norm` gating 是否至少能打破 H3.2 的完全 flat outcome。

協定：

1. 固定子族群權重：`(0.8, 0.8, 1.4)`
2. 掃描 share：`{0.10, 0.20}`
3. 掃描 base coupling：`lambda0 ∈ {0.30, 0.45}`
4. 掃描 gate threshold：`theta ∈ {0.05, 0.15, 0.30}`
5. 固定 `beta = 8.0`
6. 協定：`players=300`, `rounds=1500`, `seeds={45,47,49}`, `series=p`, `burn_in=300`, `tail=400`
7. summary：`outputs/h33b_gapnorm_short_scan.tsv`
8. 單次輸出：`outputs/h33b_gapnorm_short/*.csv`

aggregate 結果：

1. 12 組參數全部得到同一個 outcome：`P(Level>=2)=1.000`, `P(Level3)=0.000`
2. 全部都是 `level_counts={0:0,1:0,2:3,3:0}`
3. unique outcome count = 1

目前判讀：

1. H3.3B 第一版至少已排除一個合理可能性：問題不是單純因為 H3.2 缺少 `gap_norm` 型 phase gating
2. 在目前 3-seed 短窗 protocol 下，`gap_norm` sigmoid gating 仍然沒有提供任何 aggregate uplift
3. 因此如果 subgroup 線還要往前推，下一步就不應再留在 `gap_norm` 單一訊號 family 裡微調 `theta` 或 `beta`
4. 下一個合理升級應該是：改 signal 家族（例如非 gap-local state）或改 coupling topology（例如雙向 / semi-frozen）

#### 17.14 H3.4 先把 topology 改成 semi-frozen subgroup，而不是再加新的 reward gate（2026-03-31）

在 17.13 已經確認 `gap_norm` state gating 本身不足以打破 plateau 後，這一輪不再往 H3.3B 的 signal family 深挖，而是直接改 subgroup topology。

本輪選擇先做 semi-frozen subgroup，而不是純雙向 payoff coupling，原因只有一個：H3.2 與 H3.3B 都已經證明「只改 adaptive subgroup reward」很可能太弱；若還要繼續 subgroup 線，下一步更有資訊量的變更是讓固定子族群本身重新獲得部分內生更新自由度。

第一版 H3.4 契約：

1. 新增 `fixed_subgroup_anchor_pull_strength = rho_f ∈ [0,1]`
2. `rho_f = 1` 時完全退化回 H3.1 的 frozen subgroup
3. `0 < rho_f < 1` 時，固定子族群每輪先走既有 replicator 更新，再被 anchor 權重做部分回拉
4. `rho_f = 0` 時，整條路徑必須退化回未啟用 subgroup topology effect 的原始 sampled/hetero path
5. 第一版 H3.4 明確禁止與 H3.2 / H3.3B 疊加，避免 reward coupling 與 topology change 混在一起

本輪落地內容：

1. `SDD.md`：新增 H3.4 semi-frozen subgroup 契約
2. `evolution/replicator_dynamics.py`：新增純函式 helper，做 anchor pullback 與 mean-one normalization
3. `simulation/run_simulation.py`：新增 H3.4 config / CLI / validation，並把固定子族群更新從硬覆寫改成可調 pullback
4. `simulation/seed_stability.py`：新增 H3.4 payload 與 CLI 傳遞
5. 回歸新增 H3.4 端點退化、非法參數、與 H3.2/H3.3B 互斥測試

目前只完成落地與回歸，尚未跑第一輪 H3.4 short scout；下一步應直接用 H3.2/H3.3B 同一個 3-seed sampled short protocol 掃一個最小 `rho_f` family，而不是再做概念討論。

#### 17.15 H3.4 semi-frozen short scout 仍然完全 flat，因而直接切到 H3.5 雙向 payoff coupling（2026-03-31）

接在 17.14 之後，本輪直接用與 H3.2 / H3.3B 相同的短窗 sampled protocol 先測試 H3.4 的最小 `rho_f` family。

協定：

1. 固定子族群權重：`(0.8, 0.8, 1.4)`
2. 掃描 share：`{0.10, 0.20}`
3. 掃描 anchor pull：`rho_f ∈ {0.25, 0.50, 0.75}`
4. 協定：`players=300`, `rounds=1500`, `seeds={45,47,49}`, `series=p`, `burn_in=300`, `tail=400`
5. summary：`outputs/h34_semifrozen_short_scan.tsv`
6. 單次輸出：`outputs/h34_semifrozen_short/*.csv`

aggregate 結果：

1. 6 組參數全部得到同一個 outcome：`P(Level>=2)=1.000`, `P(Level3)=0.000`
2. 全部都是 `level_counts={0:0,1:0,2:3,3:0}`
3. unique outcome count = 1

因此本輪不再繼續細調 `rho_f`，而是直接改做 H3.5 真雙向 payoff coupling。

H3.5 第一版契約：

1. 新增 `fixed_subgroup_bidirectional_coupling_strength`
2. 兩個 subgroup 在每輪更新前都收到等量反向 payoff shift
3. H3.5 明確要求 `fixed_subgroup_anchor_pull_strength < 1`，否則 fixed subgroup 仍是 frozen，不能稱為「真正雙向」
4. 第一版仍禁止與 H3.2 / H3.3B 混用，避免多種 coupling 同時打開

本輪落地內容：

1. `SDD.md`：新增 H3.5 正式契約
2. `evolution/replicator_dynamics.py`：新增 pure helper，計算 fixed/adaptive 兩側的 equal-and-opposite payoff shifts
3. `simulation/run_simulation.py`：新增 H3.5 config / CLI / validation / sampled runtime 掛點
4. `simulation/seed_stability.py`：新增 H3.5 batch 傳遞與 CLI
5. 回歸新增 H3.5 helper、no-op、非法參數、與 H3.2/H3.3B 互斥測試

目前只完成落地與回歸，尚未跑第一輪 H3.5 short scout；下一步應直接固定一個 semi-frozen 工作點（例如 `rho_f=0.5`）再掃雙向 coupling 強度，而不是再回去掃 H3.4 的 `rho_f`。

#### 17.16 H3.5 第一輪雙向 coupling short scout 出現第一個非 flat cell（2026-03-31）

接在 17.15 之後，本輪直接固定 H3.4 的工作點 `rho_f=0.50`，只掃 H3.5 的雙向 coupling 強度，目標是判斷「真正雙向」是否比 H3.1/H3.2/H3.3B/H3.4 更有訊號。

協定：

1. 固定子族群權重：`(0.8, 0.8, 1.4)`
2. 固定 semi-frozen 工作點：`rho_f = 0.50`
3. 掃描 share：`{0.10, 0.20}`
4. 掃描雙向 coupling：`lambda_bi ∈ {0.15, 0.30, 0.45}`
5. 協定：`players=300`, `rounds=1500`, `seeds={45,47,49}`, `series=p`, `burn_in=300`, `tail=400`
6. summary：`outputs/h35_bidirectional_short_scan.tsv`
7. 單次輸出：`outputs/h35_bidirectional_short/*.csv`

aggregate 結果：

1. 6 組參數中有 5 組仍是舊 plateau：`P(Level>=2)=1.000`, `P(Level3)=0.000`, `level_counts={0:0,1:0,2:3,3:0}`
2. 但在 `share=0.10`, `rho_f=0.50`, `lambda_bi=0.30` 出現第一個非 flat cell：`P(Level>=2)=1.000`, `P(Level3)=0.333`, `level_counts={0:0,1:0,2:2,3:1}`
3. unique outcome count = 2，代表 H3.5 已經不是前面 H3.1 到 H3.4 那種完全 flat family

目前判讀：

1. H3.5 是目前第一個在短窗 3-seed sampled protocol 下打破「全格同一 plateau」的 subgroup family
2. 訊號目前仍弱，因為 uplift 只出現在單一 cell，還不能說是穩定區域
3. 但這已經足夠支持下一步不再回頭掃 H3.4，而是直接以 `share≈0.10`, `rho_f≈0.50`, `lambda_bi≈0.30` 為中心做 confirm / local refinement

#### 17.17 H3.5 首個非 flat cell 未通過 longer confirm（2026-03-31）

為了快速驗證 17.16 中唯一冒出的 H3.5 uplift 是否穩定，本輪沒有先做細掃，而是直接對該 cell 做更嚴格的 longer confirm。

confirm 工作點：

1. `fixed_subgroup_weights=(0.8,0.8,1.4)`
2. `share=0.10`
3. `rho_f=0.50`
4. `lambda_bi=0.30`
5. `players=300`, `rounds=3000`, `seeds={45,47,...,63}` 共 10 seeds
6. `series=p`, `burn_in=600`, `tail=800`
7. per-seed report：`outputs/h35_bidirectional_confirm_share010_rho050_lambda030.csv`

結果：

1. `level_counts={0:0,1:0,2:10,3:0}`
2. `P(Level>=2)=1.000`
3. longer confirm 下 `P(Level3)=0.000`

目前判讀：

1. 17.16 的單格 uplift 沒有通過更長時間窗與更多 seeds 的 confirm
2. 這代表目前 H3.5 的 `(share=0.10, rho_f=0.50, lambda_bi=0.30)` 至少還不能被視為穩定工作點
3. 下一步不應直接把這格當成已確認結果；若還要繼續 H3.5，較合理的是做小範圍 local refinement 或直接擴增 confirm 候選，而不是圍繞這一格過度敘事

#### 17.18 H3.5 的 short uplift 不是單點，而是 `lambda_bi` 鄰域上的一小段 ridge（2026-03-31）

接在 17.17 之後，本輪沒有再做新的長 confirm，而是先回到 short protocol，檢查 17.16 的 uplift 是否只是 `lambda_bi=0.30` 單點噪音。

local refinement 協定：

1. 固定 `share=0.10`
2. 固定 `rho_f=0.50`
3. 掃描 `lambda_bi ∈ {0.24, 0.27, 0.30, 0.33, 0.36}`
4. 其餘維持 short protocol：`players=300`, `rounds=1500`, `seeds={45,47,49}`, `series=p`, `burn_in=300`, `tail=400`
5. summary：`outputs/h35_bidirectional_lambda_refine_scan.tsv`

aggregate 結果：

1. 5 個鄰近點全部得到同一個 outcome：`P(Level>=2)=1.000`, `P(Level3)=0.333`
2. 全部都是 `level_counts={0:0,1:0,2:2,3:1}`
3. unique outcome count = 1

目前判讀：

1. H3.5 的 short uplift 不是單點偶然；在 `lambda_bi≈0.24..0.36` 至少存在一小段 short-scout ridge
2. 但 17.17 已經表明中心點 `lambda_bi=0.30` 的 longer confirm 失敗，因此這條 ridge 目前仍只能被視為 fragile short-window structure，而非穩定 regime
3. 下一步若要繼續 H3.5，最合理的是挑 ridge 上 1 到 2 個鄰近點做 second longer confirm，而不是再無限制擴大 short scan

#### 17.19 H3.5 ridge 端點的 second longer confirm 也全部塌回 plateau（2026-03-31）

接在 17.18 之後，本輪直接對 ridge 的兩個端點做 second longer confirm，而不是再加密 short scan。

confirm 工作點：

1. 固定 `share=0.10`
2. 固定 `rho_f=0.50`
3. 掃描 `lambda_bi ∈ {0.24, 0.36}`
4. `players=300`, `rounds=3000`, `seeds={45,47,...,63}` 共 10 seeds
5. `series=p`, `burn_in=600`, `tail=800`
6. summary：`outputs/h35_bidirectional_confirm_endpoints_summary.tsv`
7. per-point reports：`outputs/h35_bidirectional_confirm_endpoints/*.csv`

結果：

1. 兩個端點都得到相同結果：`P(Level>=2)=1.000`, `P(Level3)=0.000`
2. 兩個端點都為 `level_counts={0:0,1:0,2:10,3:0}`
3. unique outcome count = 1

目前判讀：

1. 中心點 `lambda_bi=0.30` 的 longer confirm 已經失敗，現在兩個端點 `0.24` 與 `0.36` 的 longer confirm 也同樣失敗
2. 在目前已測的 ridge 區段內，H3.5 的 short uplift 尚未找到任何能撐過 longer protocol 的點
3. 因此這條 `lambda_bi≈0.24..0.36` ridge 目前最合理的標記是：short-window artifact candidate，而不是穩定 regime
4. 若還要繼續 H3.5，下一步就不該再沿這條 ridge 做更多 identical confirms；應該改別的工作點、別的 topology family，或直接暫停 subgroup 線

#### 17.20 最小決定性檢查：payoff-only 線性框架在 deterministic 路徑上尚未到極限（2026-03-31）

為了回答「現有 payoff-only 線性框架是否已到極限」，本輪依最小決定性檢查只做一組 deterministic long-run protocol，不再延伸 subgroup 掃描。

固定工作點：

1. `payoff_mode=matrix_ab`
2. `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`
3. `popularity_mode=expected`
4. `evolution_mode=mean_field`
5. `payoff_lag=1`
6. `selection_strength=0.06`
7. `init_bias=0.12`
8. `players=300`, `rounds=8000`
9. `burn_in=2400`, `tail=2000`
10. 唯一掃描變數：`memory_kernel ∈ {1,3,5}`

輸出：

1. summary：`outputs/deterministic_linear_limit/mean_field_memory_kernel_longrun_summary.tsv`
2. per-kernel reports：`outputs/deterministic_linear_limit/mean_field_kernel*.csv`
3. direct probe reports：`outputs/deterministic_linear_limit/_probe_kernel*.csv`

結果：

1. `memory_kernel=1`：`P(Level3)=1.000`, `mean_env_gamma=+2.24049e-05`
2. `memory_kernel=3`：`P(Level3)=1.000`, `mean_env_gamma=+1.63443e-06`
3. `memory_kernel=5`：`P(Level3)=1.000`, `mean_env_gamma=-4.9675e-05`

依本輪 stop rule，`memory_kernel=1` 與 `3` 已經足以判定 deterministic 路徑穩得住：

1. 至少有一個 kernel 在 long-run tail 下達到 `P(Level3) ≥ 0.8`
2. 且 `mean_env_gamma ≥ 0`

因此目前不能宣告「payoff-only 線性框架已到極限」。更準確的框架級結論是：

1. 現有線性家族在 deterministic / mean-field 路徑上仍可維持 Level 3
2. 真正瓶頸在於 sampled 路徑如何保留這個 deterministic 幾何，而不是再把 payoff 線性做得更強
3. 後續若要往前推，優先序應轉成 sampled-side mechanism：例如 hybrid 更新、inertia/momentum、或更自然的異質性來源，而不是繼續把 subgroup/H3 線當成主戰場

#### 17.21 框架級主軸正式轉向：從「再強化線性 payoff」改為「讓 sampled 保留 deterministic 幾何」（2026-03-31）

在 17.20 完成最小決定性檢查後，這一階段的框架級結論已足夠定稿：

1. `memory_kernel=1`：`P(Level3)=1.000`, `mean_env_gamma=+2.24e-05`
2. `memory_kernel=3`：`P(Level3)=1.000`, `mean_env_gamma=+1.63e-06`
3. `memory_kernel=5`：`P(Level3)=1.000`, 但 `mean_env_gamma` 已轉負

依既定 stop rule，`kernel=1` 與 `3` 已經通過，因此目前不能宣告「deterministic 線性框架已到極限」。比較精確的收斂結論是：

1. payoff-only 線性家族在 deterministic / mean_field 路徑上仍能維持穩定 Level 3 / 近中性環
2. 真正瓶頸不是 payoff 線性不夠強，也不是 deterministic 自身無法穩定
3. 真正瓶頸是 sampled 實現路徑會系統性洗掉 deterministic 已驗證的穩定幾何，來源更像是 replicator 平均化與有限樣本實現路徑的交互作用

因此從這一節開始，後續實驗優先序正式改寫為：

1. 核心問題：如何讓 sampled 路徑保留或重建 deterministic 路徑已驗證的穩定幾何？
2. 停止所有純 payoff 線性強化、純 subgroup rescue、或單純加大 `memory_kernel` 的延伸實驗
3. 任何新機制都必須先在 deterministic / mean_field 下驗證不會破壞既有穩定環，再測 sampled 移植成功率

目前建議的 sampled-side 優先序：

1. hybrid 更新機制：在 sampled population 中保留一部分 deterministic 更新者，先測能否阻止相位被整體平均化抹平
2. inertia / momentum replicator：在權重更新裡加入前一輪更新方向的慣性項，降低相位資訊的衰減速度
3. personality-driven 異質性：讓固定 bias 來自原生 personality 投影，而不是事後硬切 subgroup
4. event-driven 非線性 payoff：把 sampled payoff 的結構異質性直接放進 reward 生成層

如果之後要開新一輪機制探索，應優先從 sampled-side transfer mechanism 開始，而不是再回到 payoff geometry rescue。

#### 17.22 H4 第一版落地：hybrid 更新把 deterministic 更新者直接嵌回 sampled 路徑（2026-03-31）

依 17.21 的新主軸，本輪不再擴充 payoff geometry，而是直接落第一個 sampled-side transfer mechanism：H4 hybrid 更新。

第一版 H4 契約：

1. `evolution_mode=hybrid`
2. `hybrid_update_share` 指定 sampled population 中有多少玩家改用 deterministic expected-payoff 更新
3. 玩家出招與 popularity 更新仍維持 sampled；只有權重更新規則分流
4. hybrid cohort 採 deterministic prefix，作為最小可重現版本
5. 第一版只支援 `payoff_mode=matrix_ab` 且 `popularity_mode=sampled`
6. 第一版不與 H3 subgroup family 混用

本輪落地內容：

1. `SDD.md`：新增 H4 hybrid 契約與退化條件
2. `evolution/replicator_dynamics.py`：新增 deterministic single-vector replicator helper
3. `simulation/run_simulation.py`：新增 `evolution_mode=hybrid`、`hybrid_update_share`、validation 與 sampled runtime 掛點
4. `simulation/seed_stability.py`：新增 H4 batch 傳遞與 CLI
5. 回歸新增 H4 no-op、non-no-op、與非法組合測試

目前只完成落地與回歸，尚未跑第一輪 H4 smoke。下一步應先做最小 short scout，看少量 deterministic 更新者是否能在 sampled 路徑保住更多 Level 3 訊號。

#### 17.23 H4 第一輪 short scout：最小 hybrid share family 仍然 flat（2026-03-31）

依 17.22，本輪先跑 H4 的最小 short scout，直接沿用前面 subgroup family 的短 sampled protocol，避免把結果和 protocol 變更混在一起。

第一輪主掃描：

1. `payoff_mode=matrix_ab`
2. `popularity_mode=sampled`
3. `evolution_mode=hybrid`
4. `a=1.0, b=0.9, cross=0.20`
5. `selection_strength=0.06`
6. `memory_kernel=3`
7. `hybrid_update_share ∈ {0.05, 0.10, 0.20, 0.30, 0.50}`
8. `players=300, rounds=1500, seeds=45:49:2, burn_in=300, tail=400`

結果：5 個 share 全部都回到舊 plateau：

1. `P(Level>=2)=1.000`
2. `P(Level3)=0.000`
3. `level_counts={0:0,1:0,2:3,3:0}`

為了排除只是 `memory_kernel=3` 把 hybrid payoff input 洗掉，本輪再補一個最小 kernel 對照：

1. `hybrid_update_share ∈ {0.20, 0.50}`
2. `memory_kernel ∈ {1, 3}`

補充結果仍完全相同：4 個 cell 全部維持

1. `P(Level>=2)=1.000`
2. `P(Level3)=0.000`
3. `level_counts={0:0,1:0,2:3,3:0}`

因此目前可先下第一個研究判讀：

1. H4 第一版「deterministic prefix + deterministic expected-payoff update」在短 sampled protocol 下沒有打開新區域
2. 這代表 sampled-side transfer 的關鍵可能不只是把 deterministic payoff update 混回來，而更可能需要動到 cohort 選取方式、更新慣性、或更強的 cross-player 結構保留機制
3. 在沒有新結構變因前，不值得繼續對這個最小 share family 細掃

本輪新產物：

1. `outputs/h4_hybrid_short_scan.tsv`
2. `outputs/h4_hybrid_short/*.csv`
3. `outputs/h4_hybrid_kernel_check.tsv`
4. `outputs/h4_hybrid_kernel_check/*.csv`

#### 17.24 H4.1 最小版落地：在 hybrid deterministic 更新上加入一階 inertia（2026-04-01）

依 17.23 的結論，下一步不再細掃 H4 第一版的 share，而是直接改 hybrid 更新結構本身。H4.1 的最小版維持 deterministic prefix cohort 與 deterministic expected-payoff input 不變，只在 hybrid 玩家自己的 deterministic replicator 上加入一階 inertia。

本輪鎖定的 H4.1 契約：

1. 新增 `hybrid_inertia=mu_h`，範圍為 `[0,1)`
2. hybrid 玩家保存 per-player、per-strategy 的 runtime velocity state
3. 更新規則改為 `v(t+1)=mu_h*v(t)+g_det(t)`，再用 `w(t+1) ∝ w(t) exp(v(t+1))`
4. `hybrid_inertia=0` 必須精確退化回 H4 第一版
5. 不改 action sampling、不改 popularity，也不改主 CSV schema

本輪落地內容：

1. `SDD.md`：新增 H4.1 inertia 契約與退化條件
2. `evolution/replicator_dynamics.py`：新增 inertial deterministic replicator helper
3. `simulation/run_simulation.py`：新增 `hybrid_inertia`、hybrid velocity runtime state、與 sampled path 掛點
4. `simulation/seed_stability.py`：新增 H4.1 CLI / payload / provenance 傳遞
5. 回歸新增 H4.1 zero-inertia 等價、positive-inertia non-no-op、與非法組合測試

回歸結果：聚焦 H4.1 相關測試 `63 passed in 0.58s`。

接著直接跑 H4.1 第一輪 short scout，並把新產物獨立寫到新的 `outputs/h41_*` 前綴，避免和既有未提交產物混在一起。

短 sampled scout protocol：

1. 沿用 17.23 的工作點：`matrix_ab`, `popularity_mode=sampled`, `evolution_mode=hybrid`, `a=1.0`, `b=0.9`, `cross=0.2`, `selection_strength=0.06`, `memory_kernel=3`, `players=300`, `rounds=1500`, `seeds=45:49:2`, `series=p`, `burn_in=300`, `tail=400`, `eta=0.55`, `corr_threshold=0.09`, `stage3_method=turning`, `init_bias=0.12`
2. 固定 `hybrid_update_share=0.20`
3. 掃 `hybrid_inertia ∈ {0.15, 0.30, 0.45, 0.60}`

短 scout 結果：

1. 四個 inertia 點全部得到同一個短窗結果：`level_counts={0:0,1:0,2:2,3:1}`，`P(Level3)=0.333`
2. 這是 H4 family 自落地以來第一次出現非 flat 的短窗 uplift
3. 但因為 H3.5 曾出現過相同型態的短窗假陽性，這裡不能直接視為穩定突破

因此立刻對代表點 `hybrid_inertia=0.30`, `hybrid_update_share=0.20` 做 longer confirm：

1. `rounds=3000`
2. `seeds=45:63:2`（10 seeds）
3. `burn_in=600`, `tail=1000`

longer confirm 結果：

1. `level_counts={0:0,1:0,2:10,3:0}`
2. `P(Level3)=0.000`
3. `mean_env_gamma=+7.32428e-05 (r2~0.01, peaks~338.0)`

本輪判讀：

1. H4.1 的最小 inertia family 已經能做出連續 short-window uplift ridge
2. 但代表點在第一個 longer confirm 直接掉回 Level 2 plateau
3. 現階段應把 H4.1 視為「短窗 ridge 候選」，不是已確認穩定 regime

接著補做 ridge 端點 longer confirm：`hybrid_inertia=0.15` 與 `0.60`，其餘條件維持代表點 confirm protocol 不變。

端點 confirm 結果：

1. `mu=0.15`: `level_counts={0:0,1:0,2:10,3:0}`, `P(Level3)=0.000`, `mean_env_gamma=+6.2468e-05 (r2~0.01, peaks~336.0)`
2. `mu=0.60`: `level_counts={0:0,1:0,2:10,3:0}`, `P(Level3)=0.000`, `mean_env_gamma=+7.7029e-05 (r2~0.01, peaks~337.5)`

更新後判讀：

1. 目前已測的 ridge 中心與兩個端點在 longer confirm 全部失敗
2. 因此 H4.1 目前最合理的標記，不再只是「可疑 short-window ridge」，而是「整條已測 ridge 都像短窗 artifact 候選」
3. 在沒有新結構變因前，應停止細修 H4.1，不再做局部參數打磨

本輪新產物：

1. `outputs/h41_inertia_short_scan.tsv`
2. `outputs/h41_inertia_short/mu0.15.csv`
3. `outputs/h41_inertia_short/mu0.30.csv`
4. `outputs/h41_inertia_short/mu0.45.csv`
5. `outputs/h41_inertia_short/mu0.60.csv`
6. `outputs/h41_inertia_confirm/mu0.30_confirm.csv`
7. `outputs/h41_inertia_confirm_mu0.30_summary.tsv`
8. `outputs/h41_inertia_confirm_endpoints/mu0.15_confirm.csv`
9. `outputs/h41_inertia_confirm_endpoints/mu0.60_confirm.csv`
10. `outputs/h41_inertia_confirm_endpoints_summary.tsv`

下一步不再是細修 H4.1，而是停止這條分支的局部 refine，回到機制層重新設計 sampled-side transfer 的結構。

#### 17.25 H5 正式啟動：Sampled 幾何保留機制，先走 H5.1 sampled inertia gate/scout/confirm（2026-04-01）

框架級收斂已經足夠明確：

1. deterministic 線性 `matrix_ab` 可穩，`memory_kernel=1/3` 都已驗證
2. sampled-side 既有轉移線（H1/H2/H3/H4/H4.1）目前都沒有穩定移植這個幾何
3. 瓶頸已從「payoff 強度不夠」收斂到「sampled replicator 如何洗掉相位結構」

因此新階段正式命名為 H5：Sampled 幾何保留機制。第一條線定為 H5.1：把 inertia 直接放到 sampled 更新算子本體，而不是再沿用 hybrid prefix cohort。

本輪採用更硬的三段式 protocol：

##### 第 1 段：Deterministic gate

目的：先驗證 inertia 本身不會破壞已知穩定的 deterministic 線性環。

設定：

1. 沿用目前最佳 deterministic 工作點：`matrix_ab`, `expected + mean_field`, `a=1.0`, `b=0.9`, `cross=0.20`, `selection_strength=0.06`
2. 掃 `sampled_inertia ∈ {0.00, 0.15, 0.30, 0.45, 0.60}`
3. `memory_kernel=1/3` 都要驗證
4. `mu_s=0` 基線必須跑 3 seeds；其他 gate 點先用 1 seed 快篩

判讀標準：

1. 主要指標是 `mean_env_gamma >= 0`
2. `mean_turn_strength` 不得比 `mu_s=0` 基線下降超過 8%
3. Level 3 維持率只作輔助，不作單獨通過條件

硬 stop rule：

1. 若 gate 沒有任何 `mu_s > 0` 通過上述條件，H5.1 直接結案，轉 H5.2

##### 第 2 段：Sampled short scout

目的：只看最小局部鄰域，避免再被短窗假 ridge 誤導。

設定：

1. 先固定 deterministic gate 最佳 interior `mu_s` 作中心點，預設優先考慮 `0.30`
2. short scout 只掃中心加兩側：`{0.15, 0.30, 0.45}`
3. protocol 其餘部分沿用目前鎖定的 sampled short window
4. 對照基線固定使用 `mu_s=0`

成功標準：

1. 至少兩個相鄰 `mu_s` 點同時滿足：
2. `mean_stage3_score` 相對 `mu_s=0` 提升 `>= 0.015`
3. `mean_turn_strength` 相對 `mu_s=0` 提升 `>= 12%`
4. `mean_env_gamma` 相對 `mu_s=0` 改善（更正或更接近 0）

快速 fail 門檻：

1. 若第一個 `mu_s=0.15` 就比 `mu_s=0` 更差，直接停掉剩餘兩點

##### 第 3 段：Longer confirm

目的：只用最小算力回答「是否真的 survive」。

設定：

1. 最多 confirm 2 個點：1 個中心點 + 1 個鄰點
2. 先只跑中心點
3. 只有中心點通過後，才補一個鄰點做封口確認

硬 stop rule：

1. 中心點若 `P(Level3) < 0.5` 或 `env_gamma < 0`，立即判定 H5.1 失敗，不再補鄰點
2. 若中心點通過，再補 1 個鄰點；若鄰點失敗，H5.1 仍視為不穩，不進局部 refine

H5.1 失敗後的下一條線：

1. H5.2：`sampled_inertia × selection_strength` 的 `3×3` 小網格
2. H5.3：phase-preserving global penalty / rotation-preserving weak global constraint

總原則：H5.1 的目標不是做出一條好看的短窗 ridge，而是用最小成本快速判定 sampled inertia 本身是否值得保留。若這三段失敗，就直接結案，不重蹈 H4/H4.1 的短窗假陽性路徑。

#### 17.26 H5.1 最小版已落地；deterministic gate 依 stop rule 直接失敗（2026-04-01）

本輪先把 H5.1 最小版實作進 runtime：

1. 新增 `evolution_mode=sampled_inertial`
2. 新增 `sampled_inertia` 參數，且 `sampled_inertia=0` 精確退化回既有 `sampled` 路徑
3. deterministic gate 不新增新 mode，而是在既有 `expected + mean_field` 路徑上把 `sampled_inertia` 當成控制旋鈕使用
4. `simulation/seed_stability.py` 已同步補上 `sampled_inertia` provenance

Deterministic gate 實際執行：

1. 工作點：`matrix_ab`, `a=1.0`, `b=0.9`, `cross=0.20`, `expected + mean_field`, `selection_strength=0.06`, `init_bias=0.12`
2. 視窗：`rounds=8000`, `burn_in=2400`, `tail=2000`
3. 掃描：`memory_kernel ∈ {1,3}`, `sampled_inertia ∈ {0.00,0.15,0.30,0.45,0.60}`
4. `mu_s=0` 基線各跑 3 seeds；其餘 gate 點各跑 1 seed
5. 產物：`outputs/h51_gate_w/deterministic_gate_summary.tsv` 與各點 per-seed CSV

結果：

1. 兩個 kernel 的所有點都維持 `P(Level3)=1.000`，代表 deterministic 線性環本身沒有崩
2. 但所有 `mu_s > 0` 點的 `mean_stage3_turn_strength` 都低於各自基線的 `92%` 門檻
3. `kernel=1`：`mu_s=0.15/0.30/0.60` 的 `env_gamma` 仍為非負，但 turn-strength gate 全部失敗；`mu_s=0.45` 連 `env_gamma` 也轉負
4. `kernel=3`：`mu_s=0.15/0.60` 的 `env_gamma` 為非負，但 turn-strength gate 同樣全部失敗；`mu_s=0.30/0.45` 則直接轉負

因此依 H5.1 的 hard stop rule：

1. deterministic gate 沒有任何 `mu_s > 0` 通過
2. H5.1 不進 short scout，不進 longer confirm
3. 下一步直接轉 H5.2：`sampled_inertia × selection_strength` 的 `3×3` 小網格

#### 17.27 H5.2 下一步 protocol：`sampled_inertia × selection_strength` 的極小 3x3 交互掃描（2026-04-01）

H5.1 gate 已失敗後，H5.2 不再做大範圍 inertia 掃描，而是只回答一個更窄的問題：sampled inertia 是否只在某些 `selection_strength` 區間內才和 sampled path 相容。

本輪採用優化版 3x3：

固定條件：

1. `payoff_mode=matrix_ab`, `popularity_mode=sampled`, `evolution_mode=sampled_inertial`
2. `a=1.0`, `b=0.9`, `cross=0.20`
3. `memory_kernel=3`
4. `players=300`, `rounds=1500`, `seeds=45:49:2`
5. `series=p`, `burn_in=300`, `tail=400`, `eta=0.55`, `corr_threshold=0.09`, `stage3_method=turning`, `init_bias=0.12`

H5.2 的 9 個 mechanism cell：

1. `sampled_inertia ∈ {0.10, 0.25, 0.40}`
2. `selection_strength ∈ {0.04, 0.06, 0.08}`

但正式判讀前，必須先補 3 個 matched baseline：

1. `mu_s=0` 且 `k=0.04`
2. `mu_s=0` 且 `k=0.06`
3. `mu_s=0` 且 `k=0.08`

理由：既然 H5.2 同時掃 `mu_s` 與 `k`，任何 uplift 都必須相對於相同 `k` 的控制組比較；不能把 `k` 本身造成的差異誤判成 inertia 效應。換句話說，H5.2 的機制格子仍然只有 9 個，但整個 short scout 會有 `9 + 3 = 12` 個實際報告點。

建議輸出：

1. baseline：`outputs/h52_baseline_short/` 與 `outputs/h52_baseline_short_summary.tsv`
2. grid：`outputs/h52_grid_short/` 與 `outputs/h52_grid_short_summary.tsv`
3. 若想減少檔案分散，也可另外寫一份合併 summary：`outputs/h52_grid_short_combined.tsv`

Primary success：

1. 至少 2 個 edge-adjacent cell（共邊，不含對角）同時滿足下列條件
2. `mean_stage3_score` 比 matched baseline 提升 `>= 0.018`
3. `mean_turn_strength` 比 matched baseline 提升 `>= 15%`
4. `mean_env_gamma` 比 matched baseline 改善；實作上直接用「數值更大」判定，因為 sampled 端目前主要風險是 `env_gamma` 轉負或更負

Secondary success：

1. 即使 Primary 失敗，只要有單一 cell 同時滿足 `env_gamma > 0`
2. 且 `turn_strength >= 0.9 ×` matched baseline
3. 就保留該點做 single-point longer confirm

Fail rule：

1. 若 9 個 mechanism cell 全部沒有達到 Primary/Secondary success
2. 且 `mean_env_gamma` 對各自 matched baseline 也沒有形成改善
3. H5.2 直接結案，不再補任何點

完成 H5.2 short scout 後的分流：

1. 若有 Primary success，最多 confirm 2 個點：先取最佳點，再取 1 個與其 edge-adjacent 的支持點
2. 若只有 Secondary success，最多 confirm 1 個點
3. longer confirm 沿用既有 sampled confirm 視窗：`rounds=3000`, `burn_in=600`, `tail=1000`, `seeds=45:63:2`
4. 若 H5.2 失敗，直接轉 H5.3：phase-preserving weak global constraint；或停止 sampled-operator inertia 線，轉回完整 personality/event 模型

#### 17.28 H5.2 short scout 已完成：無 Primary success，但存在單點 Secondary 候選（2026-04-01）

本輪依 17.27 的 protocol 實際跑完 12 個 short-scout 報告：

1. baseline 3 點：`k ∈ {0.04,0.06,0.08}`, 固定 `mu_s=0`
2. mechanism 9 點：`mu_s ∈ {0.10,0.25,0.40}` × `k ∈ {0.04,0.06,0.08}`
3. 產物：`outputs/h52_baseline_short_summary.tsv`, `outputs/h52_grid_short_summary.tsv`, `outputs/h52_grid_short_combined.tsv`

baseline 摘要：

1. 三個 baseline 都維持 `P(Level>=2)=1.000`, `P(Level3)=0.000`
2. baseline score 約 `0.508..0.524`
3. baseline turn strength 約 `1.055e-3`
4. baseline env_gamma 全為正，約 `+1.57e-4 .. +1.82e-4`

H5.2 3x3 結果：

1. 9 個 mechanism cell 全部仍是 `P(Level>=2)=1.000`, `P(Level3)=0.000`
2. 沒有任何 cell 的 `score uplift` 接近 Primary 門檻 `+0.018`；最大只有 `+0.00728`，發生在 `(mu_s=0.40, k=0.08)`
3. 沒有任何 cell 的 `turn_strength uplift` 接近 Primary 門檻 `+15%`；最大只有 `+2.07%`，同樣發生在 `(mu_s=0.40, k=0.08)`
4. `env_gamma` 確實在部分點改善，最大改善量出現在 `(mu_s=0.25, k=0.08)`，`delta_env_gamma ≈ +9.64e-05`

依規則判讀：

1. **Primary success 失敗**：沒有任何單點達成 score / turn / env_gamma 三條件，因此也不存在 edge-adjacent 的 Primary pair
2. **Fail rule 不成立**：因為至少有 5 個 cell 的 `env_gamma` 比 matched baseline 更高，不屬於「全數 flat 或全數更負」
3. **Secondary success 成立**：依目前定義，9 個 mechanism cell 全部滿足 `env_gamma > 0` 且 turn strength 沒有跌破 matched baseline 的 `90%`

這裡有一個重要的 protocol 觀察：

1. H5.2 的 Secondary rule 在目前這組 3x3 上過於寬鬆，因為它對 9 個 cell 全部放行
2. 因此實務上不能把 9 個點都送去 confirm；仍應只保留最強單點

目前最合理的 single-point confirm 候選：

1. `(mu_s=0.25, k=0.08)`
2. 原因：`env_gamma` 絕對值最高且改善最大（`+2.5569e-4`, `delta≈+9.64e-05`），同時 score 也有小幅正向 uplift（`+0.00267`）
3. 雖然 `(mu_s=0.40, k=0.08)` 的 score uplift 較大（`+0.00728`），但 env_gamma 改善不如 `(0.25,0.08)`

因此 H5.2 的現狀應標記為：

1. 不是 flat failure
2. 但也沒有形成足夠強的 short-scout ridge
3. 若要再往前推，只建議補 1 個 longer confirm：`(mu_s=0.25, k=0.08)`

#### 17.29 H5.2 single-point longer confirm：`(mu_s=0.25, k=0.08)` 仍停在 Level 2 plateau（2026-04-01）

依 17.28 的結論，本輪只補 1 個 longer confirm，不再擴張確認範圍。

confirm 設定：

1. 點位：`sampled_inertia=0.25`, `selection_strength=0.08`
2. 其餘固定：`matrix_ab`, `a=1.0`, `b=0.9`, `cross=0.20`, `memory_kernel=3`, `players=300`, `popularity_mode=sampled`, `evolution_mode=sampled_inertial`, `init_bias=0.12`
3. confirm 視窗：`rounds=3000`, `burn_in=600`, `tail=1000`, `seeds=45:63:2`
4. 產物：`outputs/h52_confirm/mu0p25_k0p08_confirm.csv`, `outputs/h52_confirm/mu0p25_k0p08_confirm_summary.tsv`

結果：

1. `level_counts={0:0,1:0,2:10,3:0}`
2. `P(Level>=2)=1.000`, `P(Level3)=0.000`
3. `mean_env_gamma≈+4.27e-05`

與 short scout 對照後的判讀：

1. 這個點在 short scout 只有弱 Secondary 訊號，沒有 Primary uplift
2. 到 longer confirm 後仍完全沒有出現 Level 3 seed
3. 因此 H5.2 目前沒有任何能撐過 confirm 的候選點

結論：

1. H5.2 可以在目前工作點上視為結案
2. 不再補其他 H5.2 confirm 點
3. 下一步若要繼續 H-series，應直接轉 H5.3：phase-preserving weak global constraint；否則可改走完整 personality/event 模型

## 18. 新主線：Personality/Event 完整模型

這一節鎖定的是「最小可執行 smoke」，不是一次把完整世界模型全部接回主線。現況限制要先寫死：

1. 目前 runtime 真正可跑的是 `sampled + events`
2. `mean_field + events` 尚未在主流程接通，因此不能當作第一輪 smoke
3. `03_personality_projection_v1.py` 與 `05_little_dragon_v1.py` 仍屬 design-pack utility；第一輪只允許加薄橋接層，不改 replicator 核心

### 18.1 第一輪 smoke 的最小改動面

只做這 3 件事：

1. 初始化 player personality
2. 把 12 維 personality 投影成初始三策略 weights
3. 用縮減後的固定事件集啟動既有 sampled event path

第一輪不做：

1. 不改 core/evolution 的 replicator operator
2. 不把 Little Dragon 直接塞進 runtime loop
3. 不掃 cohort 比例、不掃大事件庫、不掃 personality 超參數

### 18.2 Gate 0：Projection + Action Sanity

目的：先證明 12D personality 真的會導出可區分的初始策略幾何與事件行為，不要直接跳世界驗證。

固定設定：

1. cohort：Aggressive / Defensive / Balanced，各 100 人
2. personality jitter：每維 `±0.08`
3. 事件模板只用 3 個：`threat_shadow_stalker`、`resource_suspicious_chest`、`uncertainty_altar`
4. projection 一律走 `docs/personality_dungeon_v1/03_personality_projection_v1.py`

Pass：

1. 3 個 cohort 的 projected centroid 必須落在 simplex 的不同三分之一區域
2. 至少 2 個事件模板上，Aggressive 與 Defensive 的最優 action 選擇比例差異 `> 25%`

Fail：

1. 直接停線
2. 先修 projection 工具或 personality-to-action 映射
3. 不進 Gate 1

建議輸出：

1. `outputs/personality_gate0_projection.tsv`
2. `outputs/personality_gate0_actions.tsv`
3. `outputs/personality_gate0_decision.md`

### 18.3 Gate 1：Static-World Sampled Smoke

目的：在現有 sampled + events 路徑上，檢查真實 personality 異質性是否比 zero-personality baseline 更能抵抗平均化。

比較組：

1. baseline：zero-personality（所有 personality 維度設為 0）
2. heterogeneous：3 個 prototype cohort 以 `1:1:1` 混合

固定參數：

1. `players=300`
2. `rounds=3000`
3. `seeds=45,47,49`
4. `memory_kernel=3`
5. `selection_strength=0.06`
6. `init_bias=0.12`
7. `series=p`
8. `enable_events=true`
9. `events_json` 固定為縮減版 3-template event set

Pass：heterogeneous 相對 baseline 必須同時滿足

1. `mean_env_gamma` 改善（數值更大或更接近 0）
2. `mean_stage3_score` 提升至少 `0.018`
3. 至少 `1/3` seeds 達到 Level 3

Weak positive：

1. 只滿足其中一項
2. 可以保留同設定作為 Gate 2 離線 Little Dragon 的輸入
3. 但不得宣稱 personality 靜態異質性已通過

Fail：

1. 若完全 flat，直接結案 personality 靜態異質性路線
2. 下一步改走更強動態機制：`personality + inertia` 或 `event-driven nonlinear payoff`

建議輸出：

1. `outputs/personality_gate1_baseline.csv`
2. `outputs/personality_gate1_heterogeneous.csv`
3. `outputs/personality_gate1_summary.tsv`
4. `outputs/personality_gate1_decision.md`

### 18.4 Gate 2：Adaptive-World Smoke

這一節只有在 Gate 1 至少出現 weak positive 時才開。

第一輪只做 offline Little Dragon：

1. 取 Gate 1 輸出的 global `p` 時序
2. 每 200 rounds 餵給 `docs/personality_dungeon_v1/05_little_dragon_v1.py`
3. 檢查 `event_type`, `a`, `b`, `pressure` 是否朝抑制 dominant strategy 的方向移動

Pass：

1. dominant strategy 改變時，至少 `70%` 的調整步是反壓方向

Fail：

1. 不做 in-loop 版本
2. 保留 Gate 1 輸出，回頭修 Little Dragon 映射或 dominant-detection 規則

建議輸出：

1. `outputs/personality_gate2_offline.tsv`
2. `outputs/personality_gate2_decision.md`

### 18.5 執行順序與 stop rule

1. 今天先做 Gate 0；這是成本最低、最能排除假陽性的步驟
2. Gate 0 通過後，再跑 Gate 1 的 3-seed、3000-round static-world smoke
3. Gate 1 至少 weak positive 才開 Gate 2 offline
4. Gate 1 若完全 flat，直接停 personality 靜態異質性，不再往 in-loop adaptive world 推
5. 整條新主線的第一輪結論，只回答一個問題：真實 personality 異質性是否比 H-series operator patch 更值得繼續投資

### 18.6 Gate 0 / Gate 1 第一輪實測結果（2026-04-01）

Gate 0 已先通過，且不是邊緣通過：

1. 3 個 cohort 的 projected centroid 分別落在 aggressive / defensive / balanced 的不同區域
2. 指定的 3 個事件模板全數出現 Aggressive 與 Defensive 的明顯 action-choice 分化
3. 因此 projection 與 personality-to-action mapping 可視為已接通

Gate 1 本輪依最小 protocol 實跑：

1. baseline：zero-personality
2. experimental：heterogeneous world（3 個 prototype cohort `1:1:1` 混合）
3. 固定：`players=300`, `rounds=3000`, `seeds=45,47,49`, `matrix_ab`, `a=1.0`, `b=0.9`, `cross=0.20`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `enable_events`, `events_json=02_event_templates_smoke_v1.json`
4. 判讀視窗：`series=p`, `burn_in=900`, `tail=1200`

本輪輸出：

1. `outputs/personality_gate1_baseline/seed*.csv`
2. `outputs/personality_gate1_heterogeneous/seed*.csv`
3. `outputs/personality_gate1_summary.tsv`
4. `outputs/personality_gate1_combined.tsv`
5. `outputs/personality_gate1_decision.md`

結果整理：

1. baseline 與 heterogeneous 兩組的 `level_counts` 都是 `{0:0,1:0,2:3,3:0}`
2. `mean_env_gamma` 從 `+3.2e-05` 升到 `+5.2e-05`，因此 env_gamma 條件通過
3. 但 `mean_stage3_score` 反而從 `0.510562` 微降到 `0.502758`
4. heterogeneous 仍是 `P(Level>=2)=1.000`、`P(Level3)=0.000`，沒有出現至少 1 個 Level 3 seed，更沒有達到 `2/3 seeds >= Level 2 且 1/3 seeds = Level 3` 的更硬門檻

因此本輪判讀：

1. Gate 1 不是 flat fail，因為 env_gamma 有改善，action mix 也明顯改變
2. 但它也沒有通過主判準，因為 stage3 score 沒有 uplift，Level 3 也完全沒有出現
3. 正式標記為 `weak_positive`，只能當作「值得保留進 Gate 2 offline Little Dragon 的輸入」，不能宣稱靜態 personality 異質性已經打開 sampled plateau

這一步的研究意義：

1. personality 異質性確實會改變世界內部的事件選擇分布
2. 這個 effect 足以推動 envelope gamma，但目前還不足以把 cycle level 從 Level 2 plateau 拉到 Level 3
3. 若後續不做 Gate 2，最自然的下一條機制線會是 `personality + sampled_inertia` 的輕混合，而不是回頭調 prototype 強度

### 18.7 H5.4 主線 protocol：`personality + sampled_inertia` 最小 smoke（2026-04-01）

這一節覆蓋 18.4 / 18.5 在 Gate 1 得到 `weak_positive` 後「預設先開 Gate 2 offline」的優先序。依 18.6 的第一輪實測，當前主線先回答更直接的機制問題：把 personality 靜態異質性與 sampled inertia 疊加後，是否能比 Gate 1 的 heterogeneous baseline 更有效保留 sampled 幾何，並至少打開 1 個 Level 3 seed。

這一節只做最小 smoke，不做 Gate 2，不動 prototype，不換事件集；目標是用最薄的增量回答「異質性 + 抗平均化」是否值得保留。

前置實作差異：

1. 在現有 `simulation.personality_gate1` 薄 harness 上新增 `evolution_mode` 與 `sampled_inertia` passthrough，讓同一套 personality world 可以直接落到 `sampled_inertial` runtime
2. `sampled_inertia=0` 必須精確退化回 Gate 1 heterogeneous baseline；若同次 rerun 與 18.6 的既有 baseline 有微小數值差，正式報表以同次 rerun 的 `mu_s=0` 控制組為準
3. `events_json`、3 個 prototype cohort、initial projection、provenance 摘要欄位、以及 cycle-metric 視窗都不變
4. H5.4 只比較 heterogeneous world 內部的 `mu_s` 差異；zero-personality baseline 不再重跑，也不作主判準

固定 protocol：

1. world：3 個 prototype cohort 以 `1:1:1` 混合，沿用 Gate 1 的 projected initial weights
2. `players=300`, `rounds=3000`, `seeds=45,47,49`
3. `payoff_mode=matrix_ab`, `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`
4. `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`
5. `enable_events=true`, `events_json=docs/personality_dungeon_v1/02_event_templates_smoke_v1.json`
6. `evolution_mode=sampled_inertial`, `sampled_inertia ∈ {0.15, 0.25, 0.35}`
7. 判讀固定 `series=p`, `burn_in=900`, `tail=1200`

最小執行序列：

1. 先用同一個 H5.4 harness 重跑 `mu_s=0.00` 的 heterogeneous control，鎖定 matched baseline
2. 確認 regression：`mu_s=0` 與 Gate 1 heterogeneous 路徑在行為上等價；`mu_s>0` 則必須是 non-no-op
3. 依序跑 `mu_s=0.15`, `0.25`, `0.35` 三個機制點，不做更大掃描
4. 每個 `mu_s` 都輸出 per-seed CSV、per-seed provenance JSON、以及同一份 aggregate summary
5. 先做一次總結判讀；若只出現弱正向，再補一個 tie-break 點 `mu_s=0.30`

建議產物命名：

1. `outputs/h54_personality_inertia/mu0.00/seed*.csv`
2. `outputs/h54_personality_inertia/mu0.15/seed*.csv`
3. `outputs/h54_personality_inertia/mu0.25/seed*.csv`
4. `outputs/h54_personality_inertia/mu0.35/seed*.csv`
5. `outputs/h54_personality_inertia/mu*/seed*_provenance.json`
6. `outputs/h54_personality_inertia_summary.tsv`
7. `outputs/h54_personality_inertia_combined.tsv`
8. `outputs/h54_personality_inertia_decision.md`

前置實作完成後，命令骨架收斂到下列形式：

```bash
cd /home/user/personality-dungeon
PYTHON_BIN=./venv/bin/python
EVENTS_JSON=docs/personality_dungeon_v1/02_event_templates_smoke_v1.json

$PYTHON_BIN -m simulation.personality_gate1 \
    --protocol h54 \
    --players 300 \
    --rounds 3000 \
    --seeds 45,47,49 \
    --memory-kernel 3 \
    --selection-strength 0.06 \
    --init-bias 0.12 \
    --burn-in 900 \
    --tail 1200 \
    --a 1.0 \
    --b 0.9 \
    --cross 0.20 \
    --events-json "$EVENTS_JSON" \
    --world-mode heterogeneous \
    --evolution-mode sampled_inertial \
    --out-root outputs/h54_personality_inertia \
    --h54-inertias 0.15,0.25,0.35 \
    --h54-summary-tsv outputs/h54_personality_inertia_summary.tsv \
    --h54-combined-tsv outputs/h54_personality_inertia_combined.tsv \
    --h54-decision-md outputs/h54_personality_inertia_decision.md
```

若要讓 harness 在主掃描後，自動依 weak-positive 結果補跑 tie-break 點 `mu_s=0.30`，再加：

```bash
    --h54-auto-tiebreak
```

主判準相對於 `mu_s=0.00` heterogeneous matched baseline：

1. `mean_env_gamma` 改善至少 `25%`
2. `mean_stage3_score` 提升至少 `+0.018`
3. 至少 `1` 個 seed 達到 Level 3

H5.4 判讀表：

| outcome | 條件 | 意義 | 下一步 |
| --- | --- | --- | --- |
| pass | 至少 `2` 個 `mu_s` 點同時滿足 3 個主判準 | `personality + sampled_inertia` 已不只是弱訊號，而是能在最小 smoke 中穩定打開 Level 3 候選 | 直接進 longer confirm：`rounds=5000`, `seeds=45,47,49,51,53,55` |
| weak_positive | 至少 `1` 個 `mu_s` 點滿足 env_gamma 改善，但 stage3 uplift 不足或沒有 Level 3；或僅 `1` 個點完整通過 | 表示抗平均化方向有效，但強度還不夠穩 | 只補 `mu_s=0.30` tie-break；若仍無 Level 3，不再擴掃 |
| fail | 所有 `mu_s` 點都無法把 stage3 從 Gate 1 heterogeneous baseline 往上推，且沒有任何 Level 3 seed | 表示「靜態 personality + inertia」在目前工作點仍不足以突破 sampled plateau | 正式結案 H5.4，轉 `personality + event-driven nonlinear payoff` 或完整 Little Dragon in-loop |

建議填表格式：

| `mu_s` | `mean_env_gamma` | gamma uplift vs `mu_s=0` | `mean_stage3_score` | stage3 uplift vs `mu_s=0` | `level_counts` | `P(Level3)` | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `0.00` | baseline | `1.000x` | baseline | `+0.000` | 待填 | 待填 | control |
| `0.15` | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待判 |
| `0.25` | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待判 |
| `0.35` | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待判 |
| `0.30` | 只在 weak_positive 邊界時補跑 | 只在 weak_positive 邊界時補跑 | 只在 weak_positive 邊界時補跑 | 只在 weak_positive 邊界時補跑 | 只在 weak_positive 邊界時補跑 | 只在 weak_positive 邊界時補跑 | tie-break only |

stop rule：

1. 若前 3 個 `mu_s` 點已有至少 `2` 個點達到 pass，立即停止擴掃，直接進 longer confirm
2. 若前 3 個 `mu_s` 點全部沒有 Level 3，且 stage3 uplift 全數低於 `+0.018`，只允許補 `mu_s=0.30` 一個點；補完仍無 Level 3 就結案
3. H5.4 結案後，不再回頭調 prototype 強度，也不再做更大的 inertia 網格
4. Gate 2 offline Little Dragon 在 H5.4 期間降為 side branch，只用於驗證 world-adaptation engine，不再作為 personality 主線的預設下一步

這條線的判讀重點只有一個：如果 Gate 1 證明「有訊號但不夠」，那 H5.4 就必須回答「抗平均化後是否真的跨過 Level 2 plateau」。若答案仍是否定，就可以非常乾淨地把 `靜態 personality + inertia` 一線結案。

### 18.8 H5.4 第一輪實測結果：`personality + sampled_inertia` 直接失敗（2026-04-01）

H5.4 已依 18.7 的正式 protocol 實跑完成：

1. matched control：heterogeneous world + `sampled_inertia=0.00`
2. 主掃描：`sampled_inertia ∈ {0.15, 0.25, 0.35}`
3. 固定工作點：`players=300`, `rounds=3000`, `seeds=45,47,49`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `matrix_ab`, `a=1.0`, `b=0.9`, `cross=0.20`, `events_json=02_event_templates_smoke_v1.json`
4. `--h54-auto-tiebreak` 已啟用，但只有在主掃描先出現 weak positive 時才會補 `mu_s=0.30`

本輪輸出：

1. `outputs/h54_personality_inertia/mu0.00/seed*.csv`
2. `outputs/h54_personality_inertia/mu0.15/seed*.csv`
3. `outputs/h54_personality_inertia/mu0.25/seed*.csv`
4. `outputs/h54_personality_inertia/mu0.35/seed*.csv`
5. `outputs/h54_personality_inertia_summary.tsv`
6. `outputs/h54_personality_inertia_combined.tsv`
7. `outputs/h54_personality_inertia_decision.md`

aggregate 結果非常一致：

1. matched control 仍是 Gate 1 heterogeneous baseline：`mean_stage3_score=0.502758`, `mean_env_gamma=0.000052`, `level_counts={0:0,1:0,2:3,3:0}`
2. `mu_s=0.15`：`mean_stage3_score=0.496608`, `mean_env_gamma=0.000052`, `P(Level3)=0.000`, verdict=`fail`
3. `mu_s=0.25`：`mean_stage3_score=0.498586`, `mean_env_gamma=0.000052`, `P(Level3)=0.000`, verdict=`fail`
4. `mu_s=0.35`：`mean_stage3_score=0.502426`, `mean_env_gamma=0.000052`, `P(Level3)=0.000`, verdict=`fail`
5. 三個主點的 aggregate `gamma_uplift_ratio_vs_control` 在 six-decimal summary 上都等於 `1.000000`，沒有達到 `>= 1.25`
6. 三個主點的 `stage3_uplift_vs_control` 全為負值，沒有任何一點達到 `+0.018`
7. 三個主點全部維持 `level_counts={0:0,1:0,2:3,3:0}`，沒有任何 Level 3 seed

因此依 18.7 的 stop rule：

1. H5.4 主掃描沒有任何 `pass` 點
2. 也沒有任何 `weak_positive` 點
3. auto tie-break `mu_s=0.30` 不會執行，因為它只在主掃描為 weak positive 時才補跑
4. H5.4 第一輪正式結論為 `fail`

研究含義：

1. Gate 1 已證明靜態 personality heterogeneity 不是 no-op，但把 sampled inertia 疊上去後，並沒有把這個弱訊號放大成可觀測的 Level 3 uplift
2. 在目前工作點上，`personality + sampled_inertia` 甚至沒有保住 Gate 1 的 weak positive；aggregate stage3 還略低於 matched control
3. 因此這條線可以視為已被乾淨否定：至少在目前 `a=1.0`, `b=0.9`, `cross=0.20`, `k=0.06`, `memory_kernel=3` 的 sampled + events 工作點上，`靜態異質性 + 一階 inertia` 不足以打破 Level 2 plateau

主線因此應更新為：

1. 不再擴大 `sampled_inertia` 網格
2. 不再回頭調 prototype 強度
3. 若仍要沿 personality/event 主線前進，下一步應直接轉 `personality + event-driven nonlinear payoff`
4. Gate 2 offline Little Dragon 可保留為 side branch，用來驗證 world-adaptation engine，但不再是突破 Level 3 的主力候選

### 18.9 H5.5 主線 protocol：`personality + event-driven nonlinear payoff` 最小 smoke

H5.4 已乾淨失敗，所以 H5.5 的目標不再是「再給 sampled 一點 inertia」，而是直接測試：把 personality heterogeneity 放進 event-enabled sampled 工作點後，非線性 payoff 是否能重新打開 regime switching，進而把 Gate 1 的弱訊號放大成可觀測 uplift。

這一輪不走 `ad_product`。原因很簡單：H2.2 的 `ad_product` 線已經做過 `theta × alpha` 與 `(a_hi,b_hi)` 兩輪 16 點短掃，結果全平。H5.5 第一輪只保留還有機制理由的 `ad_share` family。

固定工作點：

1. `players=300`
2. `rounds=3000`
3. `seeds=45,47,49`
4. `memory_kernel=3`
5. `selection_strength=0.06`
6. `init_bias=0.12`
7. `matrix_ab` control：`a=1.0`, `b=0.9`, `cross=0.20`
8. `events_json=docs/personality_dungeon_v1/02_event_templates_smoke_v1.json`
9. `world_mode=heterogeneous`

matched control：

1. `matrix_control`
2. `payoff_mode=matrix_ab`
3. 其餘參數與 nonlinear cells 完全一致

第一輪 3 個 nonlinear cells：

| condition | payoff_mode | trigger | theta_low | theta_high | alpha | a_hi | b_hi | 用意 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `threshold_legacy` | `threshold_ab` | `ad_share` | `-` | `-` | `1.00` | `1.10` | `1.00` | 沿用 H2 舊版，檢查 personality/event 下是否已足夠 |
| `threshold_hysteresis` | `threshold_ab` | `ad_share` | `0.62` | `0.70` | `1.00` | `1.10` | `1.00` | 最小 hysteresis rescue，強迫 sampled 路徑重新出現切換 |
| `threshold_slow_state` | `threshold_ab` | `ad_share` | `0.62` | `0.70` | `0.35` | `1.10` | `1.00` | 在 hysteresis 上再加慢狀態，避免切換只剩瞬時噪音 |

正式指令：

```bash
cd /home/user/personality-dungeon
./venv/bin/python -m simulation.personality_gate1 \
    --protocol h55 \
    --players 300 \
    --rounds 3000 \
    --seeds 45,47,49 \
    --memory-kernel 3 \
    --selection-strength 0.06 \
    --init-bias 0.12 \
    --burn-in 900 \
    --tail 1200 \
    --a 1.0 \
    --b 0.9 \
    --cross 0.20 \
    --events-json docs/personality_dungeon_v1/02_event_templates_smoke_v1.json \
    --world-mode heterogeneous \
    --h55-trigger ad_share \
    --h55-base-theta 0.55 \
    --h55-band-low 0.62 \
    --h55-band-high 0.70 \
    --h55-slow-alpha 0.35 \
    --h55-a-hi 1.10 \
    --h55-b-hi 1.00 \
    --h55-out-root outputs/h55_personality_nonlinear \
    --h55-summary-tsv outputs/h55_personality_nonlinear_summary.tsv \
    --h55-combined-tsv outputs/h55_personality_nonlinear_combined.tsv \
    --h55-decision-md outputs/h55_personality_nonlinear_decision.md
```

輸出：

1. `outputs/h55_personality_nonlinear/matrix_control/seed*.csv`
2. `outputs/h55_personality_nonlinear/threshold_legacy/seed*.csv`
3. `outputs/h55_personality_nonlinear/threshold_hysteresis/seed*.csv`
4. `outputs/h55_personality_nonlinear/threshold_slow_state/seed*.csv`
5. `outputs/h55_personality_nonlinear_summary.tsv`
6. `outputs/h55_personality_nonlinear_combined.tsv`
7. `outputs/h55_personality_nonlinear_decision.md`

H5.5 判讀規則：

1. 單一 cell full-pass 需要同時滿足：
     - `mean_env_gamma` 相對 matched control 改善至少 `25%`
     - `mean_stage3_score` 相對 matched control 提升至少 `+0.018`
     - 至少 `1` 個 seed 達到 Level 3
     - 至少 `1` 個 seed 的 `threshold_regime_hi` 出現切換
2. 整體 H5.5 只要有 `1` 個 cell full-pass 就記為 `pass`
3. 若沒有 full-pass，但至少有部分 uplift 或 switch evidence，記為 `weak_positive`
4. 若三個 cells 都沒有 uplift，且 `p_switched_seed=0`，記為 `fail`

建議填表格式：

| condition | `mean_env_gamma` | gamma uplift vs control | `mean_stage3_score` | stage3 uplift vs control | `level_counts` | `P(Level3)` | `p_switched_seed` | `mean_regime_switches` | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `matrix_control` | baseline | `1.000x` | baseline | `+0.000` | 待填 | 待填 | `-` | `-` | control |
| `threshold_legacy` | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待判 |
| `threshold_hysteresis` | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待判 |
| `threshold_slow_state` | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待判 |

stop rule：

1. 若任一 cell 已 full-pass，立即停止再加 cell，直接進 longer confirm
2. 若只有 switch evidence，但 `stage3_uplift < 0.018` 且沒有任何 Level 3 seed，先記為 weak positive，不直接宣稱 nonlinear payoff 已成立
3. 若三個 cells 都沒有 switch evidence，表示 sampled + events 下仍被鎖死在單一 regime，H5.5 第一輪可視為乾淨 fail

### 18.10 H5.5 第一輪實測結果：有 switch evidence，但沒有任何 uplift（2026-04-01）

H5.5 已依 18.9 的正式 protocol 實跑完成：

1. matched control：heterogeneous world + `matrix_ab`
2. nonlinear cells：`threshold_legacy`、`threshold_hysteresis`、`threshold_slow_state`
3. 固定工作點：`players=300`, `rounds=3000`, `seeds=45,47,49`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `a=1.0`, `b=0.9`, `cross=0.20`, `events_json=02_event_templates_smoke_v1.json`

本輪輸出：

1. `outputs/h55_personality_nonlinear/matrix_control/seed*.csv`
2. `outputs/h55_personality_nonlinear/threshold_legacy/seed*.csv`
3. `outputs/h55_personality_nonlinear/threshold_hysteresis/seed*.csv`
4. `outputs/h55_personality_nonlinear/threshold_slow_state/seed*.csv`
5. `outputs/h55_personality_nonlinear_summary.tsv`
6. `outputs/h55_personality_nonlinear_combined.tsv`
7. `outputs/h55_personality_nonlinear_decision.md`

aggregate 結果非常特殊，因為三個 nonlinear cells 在 cycle 指標上與 control 完全重合，但都留下了可觀測的 regime-switch evidence：

1. matched control：`mean_stage3_score=0.502758`, `mean_env_gamma=0.000052`, `level_counts={0:0,1:0,2:3,3:0}`
2. `threshold_legacy`：`mean_stage3_score=0.502758`, `mean_env_gamma=0.000052`, `P(Level3)=0.000`, `p_switched_seed=1.000`, `mean_regime_switches=18.666667`, verdict=`weak_positive`
3. `threshold_hysteresis`：`mean_stage3_score=0.502758`, `mean_env_gamma=0.000052`, `P(Level3)=0.000`, `p_switched_seed=1.000`, `mean_regime_switches=2.000000`, verdict=`weak_positive`
4. `threshold_slow_state`：`mean_stage3_score=0.502758`, `mean_env_gamma=0.000052`, `P(Level3)=0.000`, `p_switched_seed=1.000`, `mean_regime_switches=2.000000`, verdict=`weak_positive`

正式判讀：

1. 三個 nonlinear cells 的 `gamma_uplift_ratio_vs_control` 全都只有 `1.000000`
2. 三個 nonlinear cells 的 `stage3_uplift_vs_control` 全都等於 `0.000000`
3. 三個 nonlinear cells 全部維持 `level_counts={0:0,1:0,2:3,3:0}`，沒有任何 Level 3 seed
4. 但三個 nonlinear cells 全部都有 `p_switched_seed=1.000`，表示 sampled + events 路徑下確實觀察到了 threshold regime 切換，不再像 H2 sampled 那樣完全鎖死在單一 regime

因此依 18.9 的 stop rule：

1. H5.5 第一輪沒有任何 full-pass cell
2. 但三個 cells 都有 switch evidence，所以整體不是 fail，而是 `weak_positive`
3. 這個 `weak_positive` 的含義不是「已經打開 Level 3」，而是「非線性 payoff 已成功改變內部狀態機制，但這個機制尚未轉化成 aggregate uplift」

研究含義：

1. H5.5 比 H5.4 更值得保留，因為它至少真的把 sampled + events 路徑從「單一固定 regime」推到了「有實際切換」
2. 但目前這些切換仍是內部機制訊號，尚未轉成更高的 `mean_env_gamma`、更高的 `mean_stage3_score`，也沒有產生任何 Level 3 seed
3. `threshold_hysteresis` 與 `threshold_slow_state` 在 aggregate 與 per-seed 指標上目前完全一致，表示 `state_alpha=0.35` 這個慢狀態版本在第一輪主掃描中還沒有帶來可分辨的新效果

主線因此更新為：

1. 不把 H5.5 記成 fail，也不宣稱 nonlinear payoff 已成立
2. 下一步若繼續 H5.5，應只做局部 refinement，目標是把已有的 switch evidence 轉成 stage3 或 gamma uplift
3. 在現階段不值得擴大成大網格；優先應該調的是 nonlinear cell 的局部幾何，而不是回去重跑 inertia

### 18.11 H5.5R 最後一次局部 refinement：只允許 6 個 cell + 最多 1 個 longer confirm

H5.5 第一輪已回答了關鍵的機制問題：nonlinear payoff 確實能讓 sampled + events 路徑出現 regime switching。但它還沒回答更重要的結果問題：這些切換能不能轉成 stage3 或 gamma uplift。

因此 H5.5R 的定位很嚴格：不是新 family，也不是大網格，而是給 nonlinear payoff 最後一次救命機會。這一輪如果還是只有 switch evidence 而沒有 uplift，就直接結案 H5.5。

固定工作點：

1. `players=300`
2. `rounds=3000`
3. `seeds=45,47,49`
4. `memory_kernel=3`
5. `selection_strength=0.06`
6. `init_bias=0.12`
7. control：`matrix_ab`, `a=1.0`, `b=0.9`, `cross=0.20`
8. `events_json=docs/personality_dungeon_v1/02_event_templates_smoke_v1.json`
9. `world_mode=heterogeneous`

H5.5R 只允許 6 個 refinement cells：

| condition | family | trigger | theta / band | alpha | a_hi | b_hi | 用意 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `legacy_geo_lo` | legacy | `ad_share` | `theta=0.55` | `1.00` | `1.05` | `0.95` | 降低 legacy 高檔強度 |
| `legacy_geo_mid` | legacy | `ad_share` | `theta=0.55` | `1.00` | `1.10` | `1.00` | H5.5 舊 `threshold_legacy` 中心點 |
| `legacy_geo_hi` | legacy | `ad_share` | `theta=0.55` | `1.00` | `1.15` | `1.05` | 提高 legacy 高檔強度 |
| `slow_alpha_020` | slow-state | `ad_share` | `theta_low=0.62`, `theta_high=0.70` | `0.20` | `1.10` | `1.00` | 更慢的狀態滯後 |
| `slow_alpha_035` | slow-state | `ad_share` | `theta_low=0.62`, `theta_high=0.70` | `0.35` | `1.10` | `1.00` | H5.5 舊 `threshold_slow_state` 中心點 |
| `slow_alpha_050` | slow-state | `ad_share` | `theta_low=0.62`, `theta_high=0.70` | `0.50` | `1.10` | `1.00` | 略快但仍保留慢狀態 |

正式指令：

```bash
cd /home/user/personality-dungeon
./venv/bin/python -m simulation.personality_gate1 \
    --protocol h55r \
    --players 300 \
    --rounds 3000 \
    --seeds 45,47,49 \
    --memory-kernel 3 \
    --selection-strength 0.06 \
    --init-bias 0.12 \
    --burn-in 900 \
    --tail 1200 \
    --a 1.0 \
    --b 0.9 \
    --cross 0.20 \
    --events-json docs/personality_dungeon_v1/02_event_templates_smoke_v1.json \
    --world-mode heterogeneous \
    --h55r-legacy-a-hi-values 1.05,1.10,1.15 \
    --h55r-legacy-b-hi-values 0.95,1.00,1.05 \
    --h55r-band-low 0.62 \
    --h55r-band-high 0.70 \
    --h55r-alpha-values 0.20,0.35,0.50 \
    --h55r-stage3-min-uplift 0.015 \
    --h55r-gamma-min-ratio 1.10 \
    --h55r-out-root outputs/h55r_personality_nonlinear_refine \
    --h55r-summary-tsv outputs/h55r_personality_nonlinear_refine_summary.tsv \
    --h55r-combined-tsv outputs/h55r_personality_nonlinear_refine_combined.tsv \
    --h55r-decision-md outputs/h55r_personality_nonlinear_refine_decision.md
```

H5.5R 判讀規則：

1. 單一 cell 只有在同時滿足以下條件時才算 `confirm_candidate`：
     - `stage3_uplift_vs_control >= 0.015`
     - `gamma_uplift_ratio_vs_control >= 1.10`
     - `p_switched_seed > 0`
2. 若 6 個 cells 全部都不滿足上述條件，整體決策直接記為 `close_h55`
3. 若有多個 cells 通過 gate，只允許取 `1` 個最佳 cell；排序先比 `stage3_uplift_vs_control`，再比 `gamma_uplift_ratio_vs_control`，再比 `mean_regime_switches`
4. longer confirm 只允許對這個最佳 cell 執行一次：`rounds=5000`, `seeds=45,47,49,51,53,55`

這一輪的核心不是再看「有沒有 switch」，而是看「switch 能不能被轉成 uplift」。如果答案仍然是否定，就應乾淨結案所有 replicator 框架內的 nonlinear rescue。

### 18.12 H5.5R 正式結果：6 個 refinement cells 全部只剩 switch_only，直接 `close_h55`（2026-04-01）

H5.5R 已依 18.11 的正式 protocol 實跑完成：

1. matched control：heterogeneous world + `matrix_ab`
2. legacy 幾何主軸：`legacy_geo_lo`, `legacy_geo_mid`, `legacy_geo_hi`
3. slow-state 主軸：`slow_alpha_020`, `slow_alpha_035`, `slow_alpha_050`

本輪輸出：

1. `outputs/h55r_personality_nonlinear_refine/matrix_control/seed*.csv`
2. `outputs/h55r_personality_nonlinear_refine/legacy_geo_*/seed*.csv`
3. `outputs/h55r_personality_nonlinear_refine/slow_alpha_*/seed*.csv`
4. `outputs/h55r_personality_nonlinear_refine_summary.tsv`
5. `outputs/h55r_personality_nonlinear_refine_combined.tsv`
6. `outputs/h55r_personality_nonlinear_refine_decision.md`

aggregate 結果比 H5.5 第一輪更乾淨，因為 6 個 refinement cells 全部完全重演了同一個 pattern：

1. control：`mean_stage3_score=0.502758`, `mean_env_gamma=0.000052`, `level_counts={0:0,1:0,2:3,3:0}`
2. 6 個 refinement cells 的 `mean_stage3_score` 全部仍是 `0.502758`
3. 6 個 refinement cells 的 `mean_env_gamma` 全部仍是 `0.000052`
4. 6 個 refinement cells 的 `gamma_uplift_ratio_vs_control` 全部都是 `1.000000`
5. 6 個 refinement cells 的 `stage3_uplift_vs_control` 全部都是 `0.000000`
6. 6 個 refinement cells 全部都有 `p_switched_seed=1.000`

細節上，只有 switch 強度出現可分辨差異：

1. `legacy_geo_lo`：`mean_regime_switches=19.333333`
2. `legacy_geo_mid`：`mean_regime_switches=18.666667`
3. `legacy_geo_hi`：`mean_regime_switches=18.666667`
4. `slow_alpha_020/035/050`：全部都是 `mean_regime_switches=2.000000`

但依 18.11 的 hard stop：

1. `stage3_uplift_vs_control >= 0.015` 的 cell 數量是 `0`
2. `gamma_uplift_ratio_vs_control >= 1.10` 的 cell 數量也是 `0`
3. 因此 `confirm_gate_pass=yes` 的 cell 數量是 `0`
4. 不存在唯一 best cell，也不存在可合法進 longer confirm 的候選

正式結論：

1. H5.5R decision=`close_h55`
2. hard stop 被完整觸發
3. H5.5 nonlinear rescue line 應在這裡正式結案，不再做任何局部 refinement 或額外 confirm

研究含義：

1. 這條線不是「差一點成功」，而是已經證明 switch evidence 和 aggregate uplift 在目前 replicator + sampled + events 框架下是脫鉤的
2. 即使把 legacy 幾何微調到 `(1.05,0.95)` 或 `(1.15,1.05)`，也只會改變切換次數，不會改變 stage3 或 gamma 的 aggregate 指標
3. slow-state alpha 在 `0.20/0.35/0.50` 間完全不可辨識，表示目前的慢狀態自由度也沒有提供可利用的新結構
4. 因此在 replicator 框架內再加 nonlinear rescue，已不值得繼續投入

### 18.13 H1~H5 rescue lines 正式結案，主線改開 H6：完整 Personality + Event 世界模型（2026-04-01）

H5.5R 的 `close_h55` 不是單一機制失敗，而是把目前框架下的救援空間一起收斂掉了：

1. H1~H4 是 sampled-side transfer / subgroup / hybrid 類補丁，最終都停在短窗假象或 Level 2 plateau
2. H5.1~H5.2 的 sampled inertia 線已證明無法把 deterministic 幾何穩定轉進 sampled + events
3. H5.4 `personality + sampled_inertia` 明確失敗
4. H5.5 雖有真實 switch evidence，但 H5.5R 證明這些 switch 無法轉成 aggregate uplift

因此後續 personality/event 主線不再回頭做任何 H5.x rescue；新主線正式改為 H6：先把靜態 personality 異質性推到比 `1:1:1` 更極端，再用 offline Little Dragon 驗證 world-pressure 映射，兩關都通過後才討論 in-loop adaptive world。

### 18.14 H6 Gate 1：Expanded Static Heterogeneity protocol

固定工作點：

1. `players=300`
2. `rounds=4000`
3. `seeds=45,47,49,51,53,55`
4. `memory_kernel=3`
5. `selection_strength=0.06`
6. `init_bias=0.12`
7. `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`
8. `events_json=docs/personality_dungeon_v1/02_event_templates_smoke_v1.json`

固定 cells：

1. controls：`zero_control`, `ratio_111_reference`
2. `2:1:0` family：`ratio_210_aggressive`, `ratio_120_defensive`, `ratio_102_balanced`
3. `3:0:0` family：`ratio_300_aggressive`, `ratio_030_defensive`, `ratio_003_balanced`

判讀規則：

1. 正式 reference 固定為 `ratio_111_reference`
2. 單一 candidate full-pass：`stage3_uplift_vs_reference >= 0.025` 且 `level3_seed_count >= 2`
3. 整體 Gate 1 `pass`：至少 1 個 candidate full-pass
4. 若沒有 full-pass，但出現任一 `stage3_uplift_vs_reference > 0`、任一 Level 3 seed、或 `mean_env_gamma` 優於 reference，記為 `weak_positive`
5. 若 Gate 1 只有 `weak_positive` 或 `fail`，H6 在此停住，不開 Gate 2

執行指令：

```bash
./venv/bin/python -m simulation.personality_h6 \
    --protocol gate1 \
    --players 300 \
    --rounds 4000 \
    --seeds 45,47,49,51,53,55 \
    --burn-in 800 \
    --tail 1200 \
    --gate1-out-root outputs/h6_gate1_expanded \
    --gate1-summary-tsv outputs/h6_gate1_expanded_summary.tsv \
    --gate1-combined-tsv outputs/h6_gate1_expanded_combined.tsv \
    --gate1-decision-md outputs/h6_gate1_expanded_decision.md
```

### 18.15 H6 Gate 2：Offline Little Dragon protocol

只有在 18.14 的 Gate 1 整體結果為 `pass` 時才允許執行。

固定規則：

1. 只吃 Gate 1 的唯一最佳 cell
2. 每 `300` rounds 擷取一次 global `p`
3. 只統計 dominant strategy 相對前一個 snapshot 發生變化，且 `dominance_bias >= 0.02` 的步
4. pass：可評估變化步中，至少 `70%` 同時滿足 anti-pressure response 與 world-output 實際改變

執行指令：

```bash
./venv/bin/python -m simulation.personality_h6 \
    --protocol h6 \
    --players 300 \
    --rounds 4000 \
    --seeds 45,47,49,51,53,55 \
    --burn-in 800 \
    --tail 1200 \
    --gate1-out-root outputs/h6_gate1_expanded \
    --gate1-summary-tsv outputs/h6_gate1_expanded_summary.tsv \
    --gate1-combined-tsv outputs/h6_gate1_expanded_combined.tsv \
    --gate1-decision-md outputs/h6_gate1_expanded_decision.md \
    --gate2-steps-tsv outputs/h6_gate2_offline_steps.tsv \
    --gate2-summary-tsv outputs/h6_gate2_offline_summary.tsv \
    --gate2-decision-md outputs/h6_gate2_offline_decision.md
```

預期輸出：

1. `outputs/h6_gate1_expanded_summary.tsv`
2. `outputs/h6_gate1_expanded_combined.tsv`
3. `outputs/h6_gate1_expanded_decision.md`
4. `outputs/h6_gate2_offline_steps.tsv`
5. `outputs/h6_gate2_offline_summary.tsv`
6. `outputs/h6_gate2_offline_decision.md`

### 18.16 H6 Gate 1 正式結果：整體仍是 `weak_positive`，不得開 Gate 2（2026-04-01）

H6 Gate 1 已依 18.14 的正式 protocol 實跑完成：

1. `players=300`, `rounds=4000`, `seeds=45,47,49,51,53,55`
2. `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`
3. `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`
4. 8 個固定 conditions：`zero_control`, `ratio_111_reference`, 以及 6 個 extreme-ratio cells

正式輸出：

1. `outputs/h6_gate1_expanded_summary.tsv`
2. `outputs/h6_gate1_expanded_combined.tsv`
3. `outputs/h6_gate1_expanded_decision.md`

整體 decision：

1. `decision=weak_positive`
2. `gate2_allowed=no`
3. `best_condition=none`

reference 與 controls：

1. `ratio_111_reference`：`mean_stage3_score=0.420198`, `mean_env_gamma=0.000031`, `level_counts={0:0,1:1,2:5,3:0}`
2. `zero_control`：`mean_stage3_score=0.509367`, `mean_env_gamma=-0.000003`, `level_counts={0:0,1:0,2:6,3:0}`

6 個 extreme-ratio cells 裡，沒有任何一點通過 Gate 1 的 hard pass：

1. `ratio_210_aggressive`：明確失敗，`stage3_uplift=-0.081042`, `gamma_ratio=0.064516`
2. `ratio_120_defensive`：僅微弱 uplift，`stage3_uplift=0.002366`, `gamma_ratio=0.903226`
3. `ratio_102_balanced`：`gamma` 有改善，但 `stage3_uplift=-0.089187`
4. `ratio_300_aggressive`：`stage3_uplift=0.110570`，但 `mean_env_gamma` 轉負，`gamma_ratio=-0.935484`
5. `ratio_030_defensive`：`stage3_uplift=0.085214`, `gamma_ratio=1.548387`
6. `ratio_003_balanced`：`stage3_uplift=0.083869`, `gamma_ratio=2.096774`

但關鍵限制非常清楚：

1. 6 個 candidates 全部 `level3_seed_count=0`
2. 因此沒有任何一點能滿足 `level3_seed_count >= 2`
3. 也就是說，沒有任何一點能形成 H6 Gate 1 的 formal `pass`

研究判讀：

1. 極端 ratio 的確能把 aggregate stage3 與 gamma 往上推，尤其是 `0:3:0` 與 `0:0:3`
2. 但這些 uplift 仍然只停在 Level 2 plateau，沒有產生任何 Level 3 seed
3. 因此 H6 Gate 1 的最準確結論仍是 `weak_positive`：靜態 personality/event 異質性已接近框架上限，但尚未打開可進 Gate 2 的正式突破
4. 依 18.14 的 stop rule，本輪不得開 H6 Gate 2 offline Little Dragon

### 18.17 H6 正式結案，H7 改為唯一 personality 主線（2026-04-01）

依 18.16 的正式結果，H6 的靜態 personality 異質性線在目前規格下已經足夠清楚：

1. static composition 不是 no-op，因為單一主導世界確實能推高 aggregate stage3 與 gamma
2. 但 static composition 仍然無法產生任何 Level 3 seed
3. 在既有 stop rule 下，H6 已無法合法開 Gate 2

因此 runbook 層面的正式結論是：H1~H6 所有靜態 personality / local rescue / offline-world 驗證線已完成一輪收斂。若仍要留在 replicator 框架內，唯一合理的新主線是 H7：讓 personality 直接進入更新算子，而不再只停留在初始條件。

### 18.18 H7.1 protocol：Personality-modulated inertia / k 最小 4-cell scout

H7.1 的設計原則：

1. 底座固定為 `ratio_111_reference` heterogeneous world
2. 只做單向耦合：`personality -> per-player inertia` 與 `personality -> per-player selection_strength`
3. 明確不做 `event-triggered personality drift`，避免把因果混在一起

固定工作點：

1. `players=300`
2. `rounds=3000`
3. `seeds=45,47,49`
4. `memory_kernel=3`
5. `selection_strength(base)=0.06`
6. `init_bias=0.12`
7. `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`
8. `events_json=docs/personality_dungeon_v1/02_event_templates_smoke_v1.json`

固定 personality mapping：

1. inertia signal：`z_mu = 0.5*(stability_seeking + patience) - 0.5*impulsiveness`
2. `mu_p = clamp(0.00 + lambda_mu * z_mu, 0.0, 0.60)`
3. selection signal：`z_k = 0.5*(ambition + greed) - 0.5*(caution + fearfulness)`
4. `k_p = clamp(0.06 * (1 + lambda_k * z_k), 0.03, 0.09)`

H7.1 短 scout 只允許 4 個 cells：

1. `control`：`lambda_mu=0.00`, `lambda_k=0.00`
2. `inertia_only`：`lambda_mu=0.25`, `lambda_k=0.00`
3. `k_only`：`lambda_mu=0.00`, `lambda_k=0.25`
4. `combined_low`：`lambda_mu=0.15`, `lambda_k=0.15`

判讀規則：

1. Primary success：至少 1 個 non-control cell 同時滿足 `>=1` 個 Level 3 seed、`stage3_uplift_vs_control >= 0.020`、以及 `mean_env_gamma` 相對 control 改善至少 `25%` 或更接近 `0`
2. Secondary success：沒有 Level 3 也可以，但必須同時滿足 `mean_stage3_score` 與 `mean_env_gamma` 相對 control 都改善至少 `15%`；此時最多只保留 1 個最佳 cell 進 longer confirm
3. Fail / close rule：如果 4 個 cells 全部沒有 Level 3 seed，且整體仍只有 `weak_positive`，H7.1 直接結案，不做 refinement

longer confirm（只有合法候選時才開）：

1. `rounds=5000`
2. `seeds=45,47,49,51,53,55`
3. 最多只允許 1 個 candidate

執行層次：

1. 先更新 Spec 與 runbook
2. 再實作新的 personality-coupled sampled runtime mode
3. 最後才跑 H7.1 正式 4-cell scout

### 18.19 H7.1 正式結果：只有 inertia-only 的微弱 aggregate uplift，整體依 stop rule 關閉為 `close_h71`

正式輸出：

1. `outputs/h71_personality_coupled_summary.tsv`
2. `outputs/h71_personality_coupled_combined.tsv`
3. `outputs/h71_personality_coupled_decision.md`

control 自身的固定基準是：

1. `mean_stage3_score=0.518780`
2. `mean_env_gamma=0.000038`
3. `level_counts={0:0,1:0,2:3,3:0}`

四個固定 cells 的正式結論如下：

1. `control`：如預期維持 heterogeneous `1:1:1` baseline，沒有 Level 3 seed
2. `inertia_only`：`mean_stage3_score=0.523643`, `stage3_uplift_vs_control=+0.004863`, `mean_env_gamma=0.000069`, `gamma_uplift_ratio_vs_control=1.815789`，但 `level3_seed_count=0`，未達 primary/secondary gate，只能記為 `weak_positive`
3. `k_only`：`mean_stage3_score=0.492551`, `mean_env_gamma=-0.000137`，stage3 與 gamma 都比 control 更差，正式記為 `fail`
4. `combined_low`：`mean_stage3_score=0.501862`, `mean_env_gamma=-0.000261`，同樣沒有 Level 3 seed，且 aggregate 指標惡化，正式記為 `fail`

因此 H7.1 的正式 decision 鎖定為：

1. `decision=close_h71`
2. `no_confirm_candidate=yes`
3. `single longer confirm` 不得開啟

這一輪把因果切得很乾淨：單純把 personality 接到 per-player inertia / selection strength，最多只能在 `inertia_only` 上帶來很小的 aggregate uplift，但仍完全推不出 Level 3 seed。按照已鎖定的 stop rule，replicator 框架內的 personality 動態耦合主線到此正式結案。

### 18.20 W1 主線：In-loop Adaptive World（post-H7 closure）

H7.1 正式 `close_h71` 之後，runbook 層面的主線要明確轉向：下一步不再是「在 replicator 上再補一層 personality / payoff patch」，而是 W1：讓世界本身擁有 latent state、更新頻率與記憶，並由世界狀態來決定事件分布與事件參數。

W1 第一版的硬邊界鎖定如下：

1. 世界狀態固定為 4 維：`scarcity`, `threat`, `noise`, `intel`
2. 每一維範圍固定為 `[0,1]`
3. 初始值固定為 `(0.5, 0.5, 0.5, 0.5)`
4. 玩家不可直接觀測 world state，只能透過事件比例與事件參數間接感受
5. world update interval 固定為每 `200` rounds 一次 batch update
6. W1.1 只吃 aggregate behavior 與事件統計，不碰 testament / cross-life memory

W1.1 的核心更新式固定為：

1. `s(t+1) = clamp(s(t) + lambda_world * (delta_p + delta_e), 0, 1)`
2. `delta_p = B_p * (p_bar - p_balanced)`
3. `delta_e = (r_bar - 0.27) * (B_e * f_bar)`

其中 W1.1 的固定矩陣為：

1. `B_p`
    - `scarcity <- [0.60, 0.20, -0.40]`
    - `threat <- [0.60, -0.10, -0.30]`
    - `noise <- [-0.25, -0.25, 0.40]`
    - `intel <- [-0.30, -0.10, 0.45]`
2. `B_e`
    - `scarcity <- [-0.10, 0.40, 0.10, -0.10, 0.10]`
    - `threat <- [0.45, -0.15, 0.10, -0.05, 0.05]`
    - `noise <- [0.05, -0.10, 0.35, -0.05, 0.30]`
    - `intel <- [-0.20, 0.15, -0.20, 0.40, -0.05]`

世界輸出只允許經由事件層四類旋鈕生效：

1. event family weights：`Threat / Resource / Uncertainty / Navigation / Internal`
2. risk parameters：各事件家族的 `base_risk / risk_bias`
3. reward multipliers：`utility_delta / risk_delta` 的 scaling
4. trait-delta strength：事件對 personality 的影響大小

明確禁止：

1. 直接修改 `matrix_ab` 的 `a/b`
2. 直接修改 `matrix_cross_coupling`
3. 直接修改 replicator `k/mu` 或任何等價 operator

W1.1 的可還原性硬門檻：

1. 若 `lambda_world=0.0`，世界狀態不得更新
2. 若世界狀態全程固定在 `(0.5,0.5,0.5,0.5)`，事件分布與事件參數必須 100% 退化回既有 `02_event_templates_smoke_v1.json` 的 3-template baseline
3. 若任何 W1 runtime 最後仍只等價於 `time-varying a/b/k/mu` 或單純 event weight 擾動，該版本直接視為不合格

### 18.21 W1.1 protocol：4-state / 4-cell 最小 scout

固定工作點：

1. `players=300`
2. `rounds=3000`
3. `seeds=45,47,49`
4. `memory_kernel=3`
5. `selection_strength=0.06`
6. `init_bias=0.12`
7. `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`
8. `events_json=docs/personality_dungeon_v1/02_event_templates_smoke_v1.json`
9. `world_update_interval=200`

固定 4 個 cells：

1. `control`：`lambda_world=0.00`，世界狀態固定不變
2. `w1_low`：`lambda_world=0.04`
3. `w1_base`：`lambda_world=0.08`
4. `w1_high`：`lambda_world=0.15`

第一版事件映射鎖定為：

1. `w_threat = clamp(1.0 + 0.80*(threat-0.5) + 0.30*(noise-0.5) - 0.20*(intel-0.5), 0.20, 3.00)`
2. `w_resource = clamp(1.0 - 0.80*(scarcity-0.5) + 0.35*(intel-0.5), 0.20, 3.00)`
3. `w_uncertainty = clamp(1.0 + 0.90*(noise-0.5) + 0.20*(threat-0.5), 0.20, 3.00)`
4. `w_navigation = clamp(1.0 + 0.75*(intel-0.5) - 0.30*(noise-0.5), 0.20, 3.00)`
5. `w_internal = clamp(1.0 + 0.55*(noise-0.5) + 0.25*(scarcity-0.5) - 0.20*(intel-0.5), 0.20, 3.00)`
6. risk parameters
7. `threat_risk_multiplier = clamp(1.0 + 0.30*(threat-0.5), 0.85, 1.15)`
8. `resource_risk_multiplier = clamp(1.0 + 0.15*(scarcity-0.5) - 0.10*(intel-0.5), 0.90, 1.10)`
9. `uncertainty_risk_multiplier = clamp(1.0 + 0.35*(noise-0.5), 0.85, 1.15)`
10. `navigation_risk_multiplier = clamp(1.0 - 0.20*(intel-0.5) + 0.10*(noise-0.5), 0.90, 1.10)`
11. `internal_risk_multiplier = clamp(1.0 + 0.20*(noise-0.5) + 0.10*(scarcity-0.5), 0.90, 1.10)`
12. reward multipliers
13. `resource_reward_multiplier = clamp(1.0 - 0.30*(scarcity-0.5), 0.85, 1.15)`
14. `noise_penalty_multiplier = clamp(1.0 + 0.40*(noise-0.5), 0.80, 1.20)`
15. `intel_reward_multiplier = clamp(1.0 + 0.30*(intel-0.5), 0.85, 1.15)`
16. `threat_penalty_multiplier = clamp(1.0 + 0.20*(threat-0.5), 0.85, 1.15)`
17. trait-delta strength
18. `threat_trait_multiplier = clamp(1.0 + 0.20*(threat-0.5), 0.85, 1.15)`
19. `resource_trait_multiplier = clamp(1.0 - 0.15*(scarcity-0.5), 0.85, 1.15)`
20. `uncertainty_trait_multiplier = clamp(1.0 + 0.25*(noise-0.5), 0.80, 1.20)`
21. `navigation_trait_multiplier = clamp(1.0 + 0.20*(intel-0.5), 0.85, 1.15)`
22. `internal_trait_multiplier = clamp(1.0 + 0.20*(noise-0.5) + 0.10*(scarcity-0.5), 0.85, 1.15)`

`a_new` / `b_new` 只作為 step-level 診斷欄位，必須維持等於輸入工作點的 `a` / `b`，不能成為世界作用面。

判讀規則：

1. `pass`：至少 `1` 個 non-control cell 出現 `>=1` 個 Level 3 seed，且 `mean_env_gamma` 相對 control 達到 `>=20%` 改善
2. `weak_positive`：沒有 `pass`，但至少 `1` 個 non-control cell 出現 `stage3_uplift_vs_control > 0`、`mean_env_gamma` 優於 control、或世界狀態真的偏離 `(0.5,0.5,0.5,0.5)`
3. `fail`：所有 non-control cells 都沒有 Level 3 seed，且 `mean_env_gamma` 沒有達到 `>=20%` 改善；這條是為了避免只看到 switch / state drift，卻沒有實際 uplift
4. decision markdown 必須額外標示 `nonisomorphic_pass`；若結果只剩 time-varying event weights，直接記為 `nonisomorphic_fail`，不得進 Gate 2

正式輸出前綴：

1. `outputs/w1_worldstate_summary.tsv`
2. `outputs/w1_worldstate_combined.tsv`
3. `outputs/w1_worldstate_decision.md`
4. 每個 seed 的 step-level 世界狀態檔名固定為 `w1_step_world_{cell_name}_seed{seed}.tsv`
5. step-level TSV 最少要有：`round`, `scarcity`, `threat`, `noise`, `intel`, `dominant_event_type`, `a_new`, `b_new`, `event_distribution`
6. 建議額外保留：`risk_multipliers_json`, `reward_multipliers_json`, `trait_multipliers_json`，供非同構檢查直接讀取

第一輪實作順序固定為：

1. 先補齊 runtime 的 world-state contract 與 CSV schema
2. 再做 control 退化測試，確認 `lambda_world=0.0` 完全還原既有 baseline
3. 最後才跑 4-cell 正式 scout

### 18.22 W1.1 正式 scout 結果：機制存在，但動力學轉移失敗（2026-04-01）

正式 W1 scout 已完成，工作點維持 `players=300`, `rounds=3000`, `seeds=45,47,49`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`，並在正式執行時把事件模板升到 `docs/personality_dungeon_v1/02_event_templates_v1.json`。

這一輪給出的訊息很乾淨：

1. 三個 adaptive cells 全部 `world_state_deviated=yes`
2. 三個 adaptive cells 全部 `nonisomorphic_pass=yes`
3. 所有 non-control cells 都沒有任何 Level 3 seed
4. 所有 non-control cells 的 `mean_stage3_score` 都低於 control
5. `w1_high` 雖然讓 `mean_env_gamma` 明顯更接近 `0`，但仍未形成突破

因此 W1 現在已經通過「機制存在性」檢查，但沒有通過「動力學轉移」檢查。研究判讀應鎖定為：世界狀態確實在動，也確實透過非同構事件旋鈕改變了系統；但這些變化還沒有成功傳導到 sampled replicator，讓玩家策略跳出既有 Level 2 plateau。

更重要的細節是：`w1_high` 已經把平均世界狀態推到接近飽和區，代表下一步不應優先再做更大的 `lambda_world` 或 state 幅度 sweep。現階段更值得優先檢查的瓶頸，是世界更新的時間尺度是否和 sampled replicator 的可觀測旋轉時間尺度錯配。

### 18.23 W1 Timescale Refinement Protocol（最終確認輪）

目的：

1. 只針對 W1 正式 scout 暴露出的「時間尺度錯配」做最後一次極小 refinement
2. 驗證「更快的世界更新」是否能把世界動態有效傳導到 sampled replicator，產生至少單 seed 的 Level 3 與更乾淨的 `env_gamma` uplift
3. 若本輪仍無法突破，則正式結案 W1，不再做任何 `lambda_world`、coupling gain、或 state 幅度微調

固定底座：

1. `players=300`
2. `rounds=3000`
3. `seeds={45,47,49}`
4. `memory_kernel=3`
5. `selection_strength=0.06`
6. `init_bias=0.12`
7. `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`
8. `series=p`
9. `events_json=docs/personality_dungeon_v1/02_event_templates_v1.json`

實作邊界說明：

1. 這一輪沿用的是目前 W1 的全域 adaptive-world runtime，不是 per-cohort `1:1:1` personality protocol
2. 現行 `simulation.world_state_w1` 仍固定 4 個原始 cells，且 `world_update_interval` 是全 scout 共用參數；因此 18.23 要正式執行前，需先做一個最小 harness 擴充，讓 cell set 與 interval 可依 cell 指定

固定 4 個 cells：

| Cell | `lambda_world` | `world_update_interval` | 目的 |
|---|---:|---:|---|
| `control` | `0.00` | `-` | 基準 |
| `w1_fast` | `0.08` | `100` | 直接測試更快世界記憶 |
| `w1_mid_fast` | `0.12` | `150` | 中速平衡點 |
| `w1_high_fast` | `0.15` | `100` | 極端快更新測試 |

本輪硬 Gate：

1. Promotion Gate：至少 `1` 個 non-control cell 同時滿足
    - `>=1` 個 Level 3 seed
    - `mean_env_gamma` 比 control 改善 `>=30%`
    - `mean_stage3_score` 不低於 control
2. Closure Gate：若三個 adaptive cells 全部沒有 Level 3 seed，或 `mean_stage3_score` 持續低於 control，則 W1 主線正式結案，不再做任何 refinement，直接轉向 W2（episode / life-based world model）

判讀表格模板：

| Cell | Level 3 seeds | `mean_stage3_score` | `mean_env_gamma` | `gamma_improve_pct` | verdict |
|---|---:|---:|---:|---:|---|
| `control` |  |  |  |  |  |
| `w1_fast` |  |  |  |  |  |
| `w1_mid_fast` |  |  |  |  |  |
| `w1_high_fast` |  |  |  |  |  |

決策規則：

1. 若 Promotion Gate 通過，W1 主線才算正式成立，之後才有理由考慮開 Gate 2 類型的 world-response validator
2. 若 Closure Gate 觸發，runbook 應把 W1 主線標記為：世界狀態動態雖然非同構，但在目前 sampled replicator working point 下，時間尺度與傳遞形狀仍不足以產生 Level 3
3. 一旦 18.23 結案，不再回頭做 `lambda_world` 強化、coupling gain、或更大 state 幅度 sweep

### 18.24 W1 Timescale Refinement 正式結果：`close_w1`（2026-04-01）

18.23 已完成正式執行，使用 `players=300`, `rounds=3000`, `seeds=45,47,49` 與完整版事件模板 `02_event_templates_v1.json`。結果如下：

1. `control`：`mean_stage3_score=0.515010`, `mean_env_gamma=-0.000066`, `level3_seed_count=0`
2. `w1_fast`：`mean_stage3_score=0.505433`, `mean_env_gamma=-0.000010`, `gamma_improve_pct=84.85%`, `level3_seed_count=0`
3. `w1_mid_fast`：`mean_stage3_score=0.492795`, `mean_env_gamma=-0.000015`, `gamma_improve_pct=77.27%`, `level3_seed_count=0`
4. `w1_high_fast`：`mean_stage3_score=0.515290`, `mean_env_gamma=0.000035`, `gamma_improve_pct=153.03%`, `level3_seed_count=0`

判讀非常直接：

1. 三個 adaptive cells 全部再次通過 `nonisomorphic_pass`
2. `env_gamma` 在 timescale refinement 下確實大幅改善，`w1_high_fast` 甚至轉成正值
3. 但所有 adaptive cells 仍然完全沒有 Level 3 seed
4. 因此 Promotion Gate 仍不成立，整輪依 18.23 硬規則正式記為 `close_w1`

W1 主線至此正式結案。研究敘事應鎖定為：世界 latent state 與事件層非同構調制都是真實存在的，但即使把世界更新頻率拉快到目前版本允許的最小 refinement，仍無法把 sampled replicator 從 Level 2 plateau 拉進 Level 3。下一步不應再做 W1 微調，而應直接轉向 W2（episode / life-based world model）。

### 18.25 W2 主線啟動：Episode / Life-based Model（post-W1 closure）

W1 既然已依 18.24 正式記為 `close_w1`，runbook 層面的下一步不應再停留在「對單一 life 內部 dynamics 補更多結構」的思路。新的主線正式改成 W2：Episode / Life-based Model。

這條線的核心不是再找另一種 in-life patch，而是把下列完整迴圈變成一級動態單位：

1. `birth`
2. `event sequence`
3. `death / life end`
4. `testament`
5. `next life initialization`

這樣做的理由很直接：目前 H1-H7 與 W1 全部失敗的共通瓶頸，都是單一 life 內的 sampled replicator 平均化過強。W2 的價值在於把 personality 更新、risk fate、與下一輪初始條件提升到跨 life 層級，避免再次退化成單純 `time-varying payoff`。

W2 第一版的硬邊界鎖定如下：

1. 每個 life 最多 `3000` rounds，或在死亡條件觸發時提早結束
2. 第一輪 `total_lives=5`
3. 先不開 Little Dragon
4. 第一版先關閉 world carryover；只驗證 death + testament 是否已足夠產生跨 life uplift

### 18.26 W2.1 protocol：最小 3-cell cross-life scout

固定工作點：

1. `players=300`
2. `rounds_per_life=3000`
3. `total_lives=5`
4. `seeds=45,47,49`
5. `events_json=docs/personality_dungeon_v1/02_event_templates_v1.json`
6. 其餘 sampled / `matrix_ab` working point 沿用 W1 formal run 的 baseline

固定 3 個 cells：

1. `control`：單 life baseline，`alpha_testament=0`
2. `w2_base`：跨 life + testament，`alpha_testament=0.12`
3. `w2_strong`：跨 life + 較強 testament，`alpha_testament=0.22`

最小 death/testament 契約：

1. 死亡條件以累積 risk 超過個人 threshold 判定，不是單純 health 歸零
2. threshold 固定為 `1.0 + 0.15 * (caution + stability_seeking - impulsiveness - fearfulness)`
3. personality risk 只保留 4 維 trait：`impulsiveness=+0.22`, `caution=-0.25`, `stability_seeking=-0.20`, `fearfulness=+0.18`
4. testament 更新式的 clip 範圍固定為 `[-0.25, 0.25]`
5. testament 權重固定為 `0.50(util) / 0.35(dom) / 0.15(event)`
6. 每次 death 或 life end 後，才允許執行 testament 更新下一 life 的 personality
7. `alpha_testament=0` 必須完全退化成「personality 不變」

第一輪 pass 標準：

1. 至少 `1` 個 non-control cell 在 `life 3..5` 中出現 `>=1` 個 Level 3 seed
2. 後半段 `mean_env_gamma >= 0`

第一輪 hard stop：

1. 若 3 個 cells 在後半段 life 都沒有任何 Level 3 seed，W2.1 直接記為 `close_w2_1`
2. 若只有單點 uplift 但 death/testament 幾乎未真正發生，結果只記 `weak_positive`，不得直接擴大到 world carryover 或 Little Dragon

輸出建議前綴：

1. `outputs/w2_episode_summary.tsv`
2. `outputs/w2_episode_combined.tsv`
3. `outputs/w2_episode_decision.md`
4. `outputs/w2_episode_life_steps.tsv`


## 19. 打破三層均化瓶頸：B-series 結構性提案（2026-04-02）

### 19.1 總命題

> **目前 Level 3 的瓶頸不是 payoff geometry 缺失，而是三層均化。**
> Deterministic / mean-field 端一直在提示 payoff geometry 本身有 rotation 能力，但 sampled 端的 (1) 公共 growth 聚合、(2) 共享更新回寫、(3) 正規化壓平 會把方向性一路磨平。

這個判斷被以下完整 closure 鏈支撐：

| 已關閉路線 | 做了什麼 | 為什麼失敗 |
|---|---|---|
| W2.1R | 生存修復 + testament 累積 | 只建立「防禦式存活收斂」，不產相位突破 |
| W3.1 | 固定 leader commitment | 只在 Level 2 plateau 做微小位移 |
| W3.2 | 低頻 hysteretic policy | 第一窗就塌縮成靜態 commitment |
| W3.3 | dominant_transition pulse | 脈衝觸發但無法改變吸引子 topology |
| H1 | memory_kernel 擴大 | Stage3 曲線上移，但 uplift 不足跨門檻 |
| H2 | threshold_ab + hysteresis | sampled 鎖死在單一 regime（0 次 switch），非線性退化成線性 |
| H3~H3.5 | 子族群隔離 / coupling | 子族群內仍被 replicator 平均化；短窗 ridge 全部 collapse |
| H4~H4.1 | deterministic prefix + inertia | 短窗假陽性；longer confirm 全掉 Level 2 |
| H5~H5.2 | sampled_inertia × k 聯合 | deterministic gate 掉 turn_strength；最優點 longer confirm 仍 Level 2 |
| H5.4~H7.1 | personality + inertia / nonlinear / coupled k | 所有組合均停 Level 2；H5.5R hard stop |

三層均化的具體程式碼對應：

```
第一層：sampled_growth_vector() → 全族群壓成 3 個共享 growth[s]
  ↓ [replicator_dynamics.py L296-316]
第二層：exp(k * growth[s]) → 全體共用更新方向
  ↓ [replicator_dynamics.py L377-402]
第三層：normalize mean=1 → 壓平權重分布
  ↓ [replicator_dynamics.py L400-402]
下一輪抽樣用近乎均勻的權重 → regime 鎖死 → 循環
```

**關鍵硬數據**：

| 路徑 | regime switches | mean_amp | Stage3 score | Level |
|---|---:|---:|---:|---:|
| Mean-field (deterministic) | 3 次 | 0.347 | 1.000 | **3** |
| Sampled (N=300) | **0 次** | 0.167 | 0.523 | **2** |

### 19.2 已排除方向（不應再嘗試）

- ❌ 純 payoff 參數微調 (a, b, cross)
- ❌ 純 memory_kernel 擴大
- ❌ threshold_ab / hysteresis / state-triggered switching
- ❌ 固定子族群 (frozen anchor, semi-frozen, 單向/雙向 coupling)
- ❌ sampled_inertia (momentum on shared operator)
- ❌ 固定 / 低頻 leader commitment / pulse
- ❌ per-player selection_strength × sampled_inertia 聯合掃描（H5.2/H7.1 已 closure）
- ❌ 純 personality coupling（H5.4/H6/H7.1 已 closure）

### 19.3 分類與優先序

五條線分為三類：

| 類別 | 提案 | 核心理由 |
|---|---|---|
| **主力線**（依序執行） | B4 → B3 → B5 | 與現有骨架最貼近 / Spec 風險最低 / 退化測試最嚴格 |
| **Topology 對照線** | B2 | 回答「保留局部結構本身夠不夠」，不當主線 |
| **高風險理論線** | B1 | 只在主力線全 fail 且局部差異保留不足以打開 basin 時才值得投入 |

執行決策樹：

```
B4 (狀態依賴 k)    ─── 依附現有 personality_coupled 路徑，最低入口成本
  ↓ pass?
  YES → B4 鎖定 + Longer Confirm
    NO  → 2026-04-07 已 closure（linear short scout fail，且不是 clamp 飽和）
             ↓
B3 (growth 分層聚合) ─── 直接切入第二層均化，精確 no-op
  ↓ pass?
  YES → B3 鎖定 + Longer Confirm
    NO  → 2026-04-07 已 closure（index / personality / phase-aware 三版全 fail）
             ↓
B5 (切線漂移)       ─── 幾何來源定義清楚才升格主線，否則只當對照 heuristic
  ↓ pass?
    YES → B5 鎖定 + Longer Confirm
        NO  → 2026-04-07 已 closure（G1 全過，但 G2 `delta<=0.010` 全部 `0/3` Level 3 且 uplift<0.02）
                         ↓
B2 (topology 對照)  ─── 確認「局部結構本身夠不夠」
  ↓ pass?
  YES → B2 + 最佳主力機制混合
  NO  ↓
B1 (切線投影)       ─── 付出數學 + Spec 成本
```

**2026-04-07 當前狀態**：B4、B3、B5 三條主力線都已依 locked formal short scout 正式 closure。B4 的失敗不是 clamp 飽和；B3 在三個升級版下都保留了 strata-level directional heterogeneity 卻仍無 seed-level basin 改變；B5 則更進一步，已證明幾何明確的 tangential drift 雖能通過 deterministic gate，但在 sampled gate 中仍無法打開 `Level 3` basin。因此目前主線不再是「轉入 B5」，而是正式進入 B2 topology 對照，不再回頭微調 B4/B3/B5 family。

**B2 formal closure（2026-04-07）**：B2 第一輪正式 short scout 已完成，`overall_verdict: close_b2`。全 18 active runs（6 conditions × 3 seeds）均為 `0/18` Level 3 seeds；`max(stage3_uplift) = 0.011`，低於 0.02 threshold。關鍵診斷：topology 確實形成了顯著 deme-level phase separation（`mean_inter_deme_phase_spread ≈ 1.93 rad`），但全局採樣仍將差異均化，Level 3 basin 無法觸發。B2 提供正式 negative result（topology 對照），至此 B4/B3/B5/B2 四條線均已正式 closure。

### 19.4 B4: 狀態依賴選擇強度（State-Dependent Selection Strength）

#### 19.4.1 為什麼先做 B4

不是因為概念最炫，而是 **與現有程式骨架最貼近**：

1. `run_simulation.py` L1593-1614 的 `personality_coupled` 分支已經不是把單一 `new_weights` 覆寫給所有玩家，而是先算 shared `growth_vector`，再讓每位玩家各自帶著自己的 `k` 與 `mu` 回寫——這正好是打破均化的最低成本入口
2. `personality_coupling.py` L30-44 的 `resolve_personality_coupling()` 已有 personality → `k` 的映射骨架（`k_value = clamp(k_base * (1 + lambda_k * signal_k), 0.03, 0.09)`）
3. 只需把 `signal_k` 的來源從靜態 personality 改成 **動態系統狀態**（或同時保留兩者的組合），不需新建骨架
4. Spec 風險最低：`k` 值域已鎖在 `[0.03, 0.09]`，退化值（`beta=0` 或 `lambda_k=0`）已有現成 noop 路徑

#### 19.4.2 動機

目前 `selection_strength` 是固定常數。但旋轉動力需要的是：

- **近 simplex 中心**（$x \approx (1/3, 1/3, 1/3)$）：需要 **更強** 選擇壓力推離平衡點
- **近 simplex 邊/角**（一策略佔優）：需要 **更弱** 選擇壓力，避免 overshoot 鎖死

等價於在 simplex 上建構一個 **有效旋轉勢場**，使系統傾向沿中等振幅的環形軌道運動。

#### 19.4.3 實作方式：依附 personality_coupled 路徑

**不新建 evolution mode**，而是在現有 `personality_coupled` 路徑上擴展。

修改 `personality_coupling.py::resolve_personality_coupling()` 使其接受可選的 `state_dominance` 參數（= 當前 `max_s p_s(t)`）：

```python
def resolve_personality_coupling(
    personality, *,
    mu_base, lambda_mu,
    k_base, lambda_k,
    # --- 新增 ---
    state_dominance: float | None = None,
    beta_state_k: float = 0.0,
) -> dict[str, float]:
    signal_k = personality_signal_k(personality)
    # 原始 personality-driven k
    k_personality = k_base * (1.0 + lambda_k * signal_k)
    # 新增：state-dependent 調制
    if state_dominance is not None and beta_state_k > 0.0:
        # dominance ∈ [1/3, 1]；uniform → factor=1；edge → factor<1
        state_factor = 1.0 + beta_state_k * (1.0 / (3.0 * state_dominance) - 1.0)
        k_personality *= state_factor
    k_value = clamp(k_personality, 0.03, 0.09)
    ...
```

在 `run_simulation.py` 的 per-player loop 中，只需傳入當前 `max(p.values())` 即可。

**替代 state proxy（若 `max(p_s)` 區分度不足）**：

| Proxy | 定義 | 優勢 | 何時升格 |
|---|---|---|---|
| `max(p_s)` | 最強策略佔比 | 最簡單，直覺對應 dominance | 預設（第一版） |
| `entropy` | $H = -\sum p_s \ln p_s$ | 連續且平滑，中心 → $\ln 3$，邊角 → 0 | `max(p_s)` 區分度不足時 |
| `dist_to_center` | $\|x(t) - c\|_2$ | 幾何意義直接，與 B5 切線定義一致 | `max(p_s)` 與 entropy 都無效時 |

第一版只用 `max(p_s)`。如果 Short Scout 全 fail 但方向性有微弱改善（uplift 0.01~0.02），則在同一 B4 family 內逐步測試 entropy → dist_to_center，視為 B4 內部變體而非新提案。

**替代 state_factor 函數形式**：

默認 linear：`state_factor = 1.0 + β * (1/(3d) - 1)`

若 linear 在 clamp 邊界上飽和（即有效 k 值大量卡在 0.03 或 0.09），考慮 exponential 形式：

```python
state_factor = exp(beta_state_k * (1.0 / (3.0 * state_dominance) - 1.0))
```

若改用 entropy 或 `dist_to_center`，建議實作也明確寫成可控形式，例如：

```python
entropy_uniform = log(3.0)
state_factor = exp(beta_state_k * (entropy_uniform - current_entropy))
state_factor = clamp(state_factor, 0.5, 2.0)
```

Exponential 的好處是放大 / 衰減更平滑，不需完全依賴 k 的 clamp 做邊界處理。但 exp 可能導致有效 k 範圍偏移，因此 **必須對 state_factor 本身再做一次安全 clamp** 以避免 overflow 或過度放大。**升格條件**：linear 版 Short Scout fail 且多數 player 的 k 值卡在 clamp 邊界上（需在 G2 掃描時記錄 `k_clamped_ratio`）。

#### 19.4.4 退化驗證

- `beta_state_k = 0.0`：`state_factor = 1.0`，完全退化為現行 personality coupling
- `lambda_k = 0.0, beta_state_k = 0.0, lambda_mu = 0.0, mu_base = 0.0`：退化為 `_personality_coupling_is_noop()` → 走 baseline `replicator_step()`

退化測試：max diff < $10^{-12}$。

#### 19.4.5 掃描參數

| 參數 | 掃描值 | 說明 |
|---|---|---|
| `beta_state_k` | 0.0 (control), 0.3, 0.6, 1.0 | state dominance 調制強度 |
| `k_base` | 0.06, 0.08 | baseline selection strength |
| `lambda_k` | 0.0 | 先隔離 state-dependent 效果，不混 personality |
| `lambda_mu` | 0.0 | 先不開 inertia（已知 inertia 獨立無效） |

8 組 × 3 seeds = 24 runs

**擴展掃描（僅在第一輪 Short Scout 顯示 clamp 飽和時啟動）**：

| 參數 | 擴展值 | 說明 |
|---|---|---|
| `k_clamp_range` | [0.02, 0.12] | 放寬有效 k 上下界（需 SDD.md 補契約） |
| `state_factor_fn` | exponential | 見上述替代函數 |
| `state_proxy` | entropy | 替代 `max(p_s)` |

⚠️ 放寬 clamp 範圍需先更新 SDD.md 中 k 值域定義；exponential / entropy 不需 Spec 變更。

#### 19.4.6 Gate 設計

| Gate | 條件 | Fail |
|---|---|---|
| Degrade | `beta_state_k=0` 與 control 完全等價 | bug，不進 smoke |
| Deterministic | mean-field 下 Level 3 維持 + turn_strength ≥ 92% | 不進 sampled scout |
| Short Scout | sampled, 3 seeds, 3000 rounds | 判讀見 19.9 |
| Longer Confirm | sampled, 10 seeds, 3000 rounds, `tail=1000` | 判讀見 19.9 |

#### 19.4.7 建議命令模板

```bash
PYTHON_BIN=./venv/bin/python
EVENTS_JSON=docs/personality_dungeon_v1/02_event_templates_v1.json

# B4 Short Scout: beta_state_k=0.6, k_base=0.06
for seed in 45 47 49; do
  $PYTHON_BIN -m simulation.run_simulation \
    --enable-events --events-json "$EVENTS_JSON" \
    --popularity-mode sampled \
    --seed "$seed" --rounds 3000 --players 300 \
    --payoff-mode matrix_ab --a 1.0 --b 0.9 \
    --matrix-cross-coupling 0.20 \
    --selection-strength 0.06 --init-bias 0.12 \
    --memory-kernel 3 \
    --evolution-mode personality_coupled \
    --personality-coupling-mu-base 0.0 \
    --personality-coupling-lambda-mu 0.0 \
    --personality-coupling-lambda-k 0.0 \
    --beta-state-k 0.6 \
    --out "outputs/b4_scout_beta0p6_k0p06_seed${seed}.csv"
done
```

（注意：`--beta-state-k` 為待新增的 CLI flag）

#### 19.4.8 已知風險：有效 k 調制空間可能不足

目前 clamp 在 `[0.03, 0.09]`，最大放大比 = 0.09/0.03 = 3×。若 `beta_state_k` 需要很大才看到效果，可能只是把整體 k 平均值稍微上移或下移，實質退化成「已排除的純 k 微調」。

**判定方式**（在 G2 Short Scout 結果中檢查）：
1. 記錄每輪每個 player 的有效 `k_value`，計算 `k_clamped_ratio`（被 clamp 截斷的比例）
2. 若 `k_clamped_ratio > 0.3`，即視為「高 clamp 飽和」；若 `k_clamped_ratio > 0.5`，則視為「嚴重飽和」
3. 此時升格到「擴展掃描」（放寬 clamp 或換 exponential），而非直接判 B4 fail

**2026-04-07 實際結論**：第一輪 24-run short scout 已完成，所有 active cells 皆為 `0/3` Level 3，`mean_stage3_score` uplift 最高僅 `0.013733`，且 `k_clamped_ratio` 幾乎為 0（最高 `0.001889`）。因此 B4 linear 的失敗不是 clamp 飽和，而是訊號本身過弱；本 family 依決策樹直接記為 closure，轉入 B3，不再把 exponential / alternative proxy 視為必要前置步驟。

### 19.5 B3: Growth 分層聚合（Stratified Growth Aggregation）

#### 19.5.1 為什麼排第二

B3 直接切入三層均化的第二層：`sampled_growth_vector()` 把全族群壓成 3 個共享 `growth[s]`。

如果改成 **strata 內先聚合、strata 間帶權重再合成**，就能直接測「局部旋轉訊號是否在全域平均前就被殺掉」。而且精確 no-op 極容易實現：只有一個 stratum 時就完全退化回現況。

#### 19.5.2 動機（與舊版 B3 的區別）

舊版構想是「保證每策略最低抽樣人數」（minimum representation）。但反思後，真正的瓶頸不在「某策略人數歸零」（Level 2 就代表振幅足夠），而在 **growth 聚合時把不同相位位置的玩家共同壓成單一方向**。

因此，更精準的介入點是 `sampled_growth_vector()` 本身，而非上游抽樣。

#### 19.5.3 實作方式

在 `replicator_dynamics.py` 新增 `stratified_growth_vector()`：

```python
def stratified_growth_vector(
    players: Iterable[object],
    strategy_space: List[str],
    *,
    strata_key: str = "stratum",    # player attribute 名稱
    n_strata: int = 1,              # 退化值 = 1 → 等於原始
) -> Dict[str, float]:
    """先在 stratum 內算 per-strategy 平均，再跨 strata 做 weighted mean。"""
    # 1. 分桶
    buckets: dict[int, list] = {i: [] for i in range(n_strata)}
    for p in players:
        sid = getattr(p, strata_key, 0) % n_strata
        buckets[sid].append(p)
    # 2. per-stratum growth
    stratum_growths = []
    stratum_counts = []
    for sid in range(n_strata):
        if buckets[sid]:
            g = sampled_growth_vector(buckets[sid], strategy_space)
            stratum_growths.append(g)
            stratum_counts.append(len(buckets[sid]))
    # 3. weighted combination
    total = sum(stratum_counts)
    combined = {s: 0.0 for s in strategy_space}
    for g, c in zip(stratum_growths, stratum_counts):
        w = c / total
        for s in strategy_space:
            combined[s] += w * g[s]
    return combined
```

**分層來源（第一版固定一種，不掃維度）**：

依 player index 等距分割（最簡單，與 personality 正交）。不在第一版引入 personality 作為分層變數，避免維度爆炸。

主要修改點落在 `replicator_dynamics.py`（新函式）與 `run_simulation.py`（在 growth vector 使用點改為 stratified 版本）。

#### 19.5.4 退化驗證

- `n_strata = 1`：所有玩家在同一桶，`stratified_growth_vector()` 精確退化為 `sampled_growth_vector()`
- 數值差異 < $10^{-12}$

#### 19.5.5 掃描參數

| 參數 | 掃描值 | 說明 |
|---|---|---|
| `n_strata` | 1 (control), 3, 5, 10 | 分層數 |
| 分層方式 | index-based（固定） | 不掃分層維度 |

4 組 × 3 seeds × 1 個 k_base = 12 runs（若加第二個 `selection_strength` 共 24 runs）

#### 19.5.6 為什麼可能成功

- 直接切入第二層均化（growth aggregation），不改 payoff、不改 operator、不做子族群
- 若不同 strata 恰好在不同相位位置，stratum-level growth 會保留局部旋轉方向
- 在數學上：stratum-level growth 的「方向訊噪比」可能比 global growth 更高（Simpson's paradox 的反面：分層平均不等於全域平均）

#### 19.5.7 風險與分層變體升級規則

**核心風險**：index-based 分層與動態無關，不保證不同 strata 剛好在不同相位。在數學上，Simpson’s paradox 的反面不一定成立：分層後的加權平均不保證比全域平均更保留方向性——只有當 strata 內部的相位一致性高於 strata 間時才有效。

**結構化升級路徑（在同一 B3 family 內，不算新提案）**：

```
B3.1: index-based strata（預設，最簡單）
  ↓ Short Scout fail 且 strata 間 growth 方差 < 全域方差
  ↓ (代表分層未捕捉到真實結構)
B3.2: personality-based strata（依 personality_signal_k 分組）
    ↓ personality 只作為 latent labels；不改初始權重、不改 payoff、不開 personality_coupled
  ↓ Short Scout fail
B3.3: phase-aware strata（依當前 p 向量投影到環序座標分組）
  ↓ 定義：計算每個 player 的 phase angle θ = atan2(√3(p_B - p_C), 2p_A - p_B - p_C)
    ↓ 依 θ 分層（等分角度）
    ↓ 每 50~100 輪重新計算一次 θ 並重分桶；若重分桶過於擾動，改用 persistent assignment + slow rebalance
  ↓ Short Scout fail → B3 整體 closure，進 B5
```

**診斷指標（每版都要記錄）**：
- per-stratum 平均 growth vector 之間的角度差（cosine similarity）
- per-stratum Stage3 score vs. 全域 Stage3 score
- strata 間 growth 方向一致性（若始終一致，代表分層無效）

#### 19.5.8 2026-04-07 執行決策

由於 B4 linear 已在 locked short scout 下乾淨 fail，且失敗原因不是 clamp 飽和，因此目前主線不再優先消耗時間於 B4 family 升級版；正式進入 B3.1：

1. 先實作 index-based `stratified_growth_vector()` 與 `sampled_growth_n_strata`
2. 先做 G0 退化驗證：`n_strata=1` 必須精確等價於既有 sampled path
3. 再跑第一輪 short scout：`n_strata ∈ {1,3,5,10}` × `seed ∈ {45,47,49}`
4. 若最佳 cell 只有弱 uplift 但仍未過 G2，再視專屬診斷決定是否升級到 B3.2 personality-based strata

**2026-04-07 實際結果**：B3.1 short scout 已完成，`n_strata ∈ {3,5,10}` 全部 `0/3` seeds 達到 Level 3，`mean_stage3_score` uplift 分別為 `0.000000`, `-0.001186`, `-0.001189`，因此三個 active cells 全部 hard stop。

但這輪與 B4 的差異很重要：B3.1 不是完全沒打到機制。`mean_inter_strata_cosine` 從 control 的 `1.000000` 降到 `0.987641 / 0.983675 / 0.964369`，`mean_growth_dispersion` 也從 `0` 升到 `0.036608 / 0.046537 / 0.073441`，表示 index-based strata 確實製造了 strata-level direction difference；只是這個差異沒有對準真正的相位結構，因此沒有轉成 Stage3 uplift。

因此 B3.1 的判讀不是「B3 family 無效」，而是更精確地說：**index-based partition 過於任意，已能製造局部 growth 異質性，但還不足以打開 Level 3 basin**。依 19.5.7 的升級規則，下一步應直接進 B3.2 personality-based strata，而不是回頭重做 B4 family。

B3.2 的第一版實作約束也一併鎖定：

1. 仍沿用 `simulation.b3_stratified_growth` 同一套 harness 與同一套 stop rule
2. `strata_mode` 從 `index` 升級為 `personality`
3. personality 來源固定採用靜態 heterogeneous 1:1:1 prototype sampling，`jitter=0.08`
4. 這些 personality 只用來計算 `personality_signal_k` 與分桶；不得改動初始權重或 sampled update operator

**2026-04-07 B3.2 實際結果**：personality-based short scout 也已完成，`n_strata ∈ {3,5,10}` 仍全部 `0/3` seeds 達到 Level 3，`mean_stage3_score` uplift 分別為 `-0.000659`, `+0.001268`, `-0.000272`，因此三個 active cells 同樣全部 hard stop。

但 B3.2 比 B3.1 更清楚回答了機制問題：`mean_inter_strata_cosine` 進一步降到 `0.897276 / 0.895504 / 0.882767`，`mean_growth_dispersion` 也升到 `0.174903 / 0.196023 / 0.220142`。也就是說，personality-based strata 的確比 index-based strata 更有效地把 growth direction 拆開；然而即使把 static heterogeneity 對齊到 personality signal，這些局部方向差仍不足以打開 Level 3 basin。

因此 B3.2 的正式結論是：**static personality-aligned partition 已經比任意切桶更貼近真實結構，但仍然不夠**。若 B3 family 要繼續，下一步應直接進 B3.3 phase-aware strata，而不是回頭重做 B3.1/B3.2 的微調。

#### 19.5.9 B3 family 正式 negative result（paper / runbook 可直接引用）

綜合 B3.1 與 B3.2 的 locked short scout，B3 family 的第一輪正式負結果可以收斂為一個比「沒效果」更強的結論：**stratified growth aggregation 確實把 sampled path 的局部方向差保留下來了，但這些差異仍不足以把系統從既有 Level 2 plateau 推入穩定的 Level 3 basin**。B3.1 的 index-based strata 已使 `mean_inter_strata_cosine` 從 control 的 `1.000000` 降到 `0.987641 / 0.983675 / 0.964369`，並把 `mean_growth_dispersion` 拉到 `0.036608 / 0.046537 / 0.073441`；B3.2 的 personality-based strata 更進一步把 `mean_inter_strata_cosine` 壓低到 `0.897276 / 0.895504 / 0.882767`，`mean_growth_dispersion` 提高到 `0.174903 / 0.196023 / 0.220142`。也就是說，B3 family 不是沒有打到機制，相反地，它已明確證明 second-layer averaging 可以被部分拆開。

但同一批正式結果也同時劃出目前機制的邊界：不論是 index-based 或 personality-based strata，所有 active cells 在 `seeds={45,47,49}` 的 short scout 中都仍是 `0/3` Level 3 seeds，`mean_stage3_score` uplift 也只落在 `-0.001189` 到 `+0.001268` 的 plateau 內微移範圍。這使得 B3 family 的正式判讀不再是「分層還不夠強、再多調幾個 strata 也許就會過」，而是更精確的研究結論：**在 well-mixed sampled replicator 下，僅靠 static strata 去保留局部 growth direction，已不足以打開 Level 3 basin；若要沿 B3 主線繼續，下一步必須升級為 phase-aware strata，直接讓分層對齊當前相位幾何，而不是回頭做 B3.1/B3.2 的局部調參。**

可直接放入 paper 的精簡版文字：

> Across the first formal B3-family scout, stratified growth aggregation produced clear strata-level directional heterogeneity but still failed to open a robust Level 3 basin. Index-based strata reduced inter-strata cosine similarity from `1.000000` to `0.964369-0.987641`, and personality-based strata reduced it further to `0.882767-0.897276`, with corresponding growth-dispersion increases from `0.036608-0.073441` to `0.174903-0.220142`. However, all active cells remained at `0/3` Level 3 seeds and yielded only marginal Stage-3-score shifts (`-0.001189` to `+0.001268`). The negative result is therefore structural rather than null: partial preservation of local growth directions is not, by itself, sufficient to overcome the averaging pressure of the well-mixed sampled replicator. This directly motivates phase-aware stratification as the next and narrowest justified escalation within the B3 line.

#### 19.5.10 B3.3 phase-aware strata 第一輪草案

B3.3 的目標不是再替 B3.2 做一輪 static-partition 微調，而是直接測試一個更精確的命題：**如果 strata 真正對齊玩家當下所處的 phase 位置，局部 growth direction 是否終於能在 sampled aggregation 前被保留下來，並轉成可觀察的 Stage3 uplift。** 這一版因此只允許做最小升級，不引入新的 payoff、自訂 operator，或額外的人格自由度。

第一輪草案固定如下：

1. 仍沿用 `simulation.b3_stratified_growth` 同一套 harness、同一套 short-scout stop rule、同一組 locked 工作點
2. `strata_mode` 從 `personality` 再升級為 `phase`
3. 每位 player 的 phase label 一律由其當前 mixed strategy 向量 $p=(p_A,p_B,p_C)$ 計算：

$$
θ = \operatorname{atan2}\big(\sqrt{3}(p_B - p_C),\; 2p_A - p_B - p_C\big)
$$

4. phase strata 固定採等角分桶，把區間 $[-\pi, \pi)$ 均分為 `n_strata ∈ {1,3,5,10}`；`n_strata=1` 仍必須精確退化為既有 sampled control
5. 第一輪固定使用 hard rebucket：每 `100` 輪重新計算一次 `θ` 並重分桶，不做 slow rebalance、不掃 rebucket interval、不掃 sector offset
6. 第一輪不得引入 personality sampling、不得改初始權重、不得改 payoff geometry；B3.3 唯一新增自由度只有 `strata_mode=phase` 與固定的 `phase_rebucket_interval=100`

這樣鎖定的好處是，B3.3 只比 B3.2 多出一個真正與相位幾何對齊的分層來源，因此如果它仍 fail，就可以更乾淨地把結論收斂成：**不是 static labels 不夠好，而是 sampled replicator 下，連 phase-aligned local aggregation 都不足以打開 basin。**

B3.3 第一輪的專屬診斷也應同步鎖定：

1. `mean_inter_strata_cosine` / `mean_growth_dispersion`（延續 B3.1/B3.2）
2. `mean_phase_rebucket_churn`：每次重分桶時，有多少 player 換到新 stratum
3. `mean_within_stratum_phase_spread`：同一 stratum 內的 phase 角離散程度
4. `phase_occupancy_entropy`：各 strata 是否有效被占用，而不是塌到少數 bins

第一輪 closure 規則不另外新增門檻，仍完全沿用既有 B3 short-scout stop rule；但若 B3.3 在有明顯 phase 分離診斷的前提下仍 `0/3` Level 3 且 uplift `<0.02`，就應把 B3 family 正式結案並轉入 B5，而不是回頭做 `rebucket_interval`、`sector_offset`、或 `n_strata` 細調。

#### 19.5.11 B3.3 實作清單與測試 gate

這裡的原則是：**最大化重用既有 B3.2 runtime / harness，只新增 phase-aware 分桶與固定週期 rebucket，不另開第二套掃描框架。**

**實作清單**：

1. 在 `simulation/b3_stratified_growth.py` 擴充 `STRATA_MODE_CHOICES`，把 `phase` 納入合法 `strata_mode`
2. 新增 phase helper，至少包含：
    - 由 player `strategy_weights` 正規化得到當前 mixed strategy $p=(p_A,p_B,p_C)$
    - 由 $p$ 計算 phase angle $\theta = \operatorname{atan2}(\sqrt{3}(p_B-p_C),\; 2p_A-p_B-p_C)$
    - 依 `n_strata` 做等角分桶，寫回 player `stratum`
3. 新增 B3.3 專用 setup / round callback adapter：
    - round 0 先做一次初始 phase 分桶
    - 之後每 `phase_rebucket_interval=100` 輪重算一次 phase 並 hard rebucket
    - 若該輪不是 rebucket 輪，只沿用前一輪 `stratum`
4. 既有 `_B3RoundDiagnostics` 要擴充為可記錄 B3.3 專屬診斷：
    - `mean_phase_rebucket_churn`
    - `mean_within_stratum_phase_spread`
    - `phase_occupancy_entropy`
5. `run_b3_scout(...)` 要新增 `phase_rebucket_interval` 參數，並把它寫入 summary TSV、combined TSV、per-seed provenance、decision markdown
6. `_condition_name(...)` 要支援 `phase_strata{n}_k...` 命名，避免和 B3.1/B3.2 產物混名
7. CLI 只新增一個旗標：`--phase-rebucket-interval`；第一輪預設即鎖死為 `100`，只是為了 provenance 與未來 closure 可回溯，不代表本輪允許掃它
8. `simulation/run_simulation.py` 原則上不新增新的演化模式；若 round-level rebucket 能完全靠既有 `round_callback` 完成，就不要再擴大 runtime 介面

**測試 gate**：

**Gate 0: API / 驗證層**

1. `strata_mode=phase` 必須能通過 CLI 與 `run_b3_scout(...)` 參數驗證
2. `phase_rebucket_interval` 必須要求為正整數；`<=0` 直接報錯
3. `n_strata=1` 在 `strata_mode=phase` 下仍必須合法，且語意上退化為 matched control

**Gate 1: 純函式 / 分桶單元測試**

1. 固定幾組手工 player 權重，驗證 phase angle 與分桶順序是可重現的
2. 驗證不同 phase sector 的 player 會被分到不同 stratum，而不是全部塌到同一桶
3. 驗證同一組 player 在不跨 sector 邊界時，重算 phase 不會無故改桶

**Gate 2: sampled 路徑退化與機制測試**

1. `test_phase_strata_n1_matches_sampled_rows`：`strata_mode=phase, n_strata=1` 時，逐列結果必須與既有 sampled control 精確等價
2. `test_phase_strata_n3_changes_sampled_path`：`strata_mode=phase, n_strata=3` 時，至少一個 round 的 `w_*` 序列要和 control 不同，證明不是 no-op
3. `test_phase_rebucket_updates_strata`：在小型短跑中強制 rebucket 至少一次，驗證至少部分 player 的 `stratum` 會被更新

**Gate 3: harness / 輸出 smoke test**

1. 新增 `test_b33_scout_writes_outputs`
2. 最小 smoke protocol 沿用 B3.2 測試風格：`seeds=[45,47]`, `n_strata_values=[1,3]`, `players=30`, `rounds=120`, `burn_in=30`, `tail=60`, `enable_events=False`
3. `combined.tsv` 至少要驗證：
    - condition 名稱為 `phase_strata1_k0p06` / `phase_strata3_k0p06`
    - `strata_mode == phase`
    - 新增欄位 `phase_rebucket_interval`, `mean_phase_rebucket_churn`, `mean_within_stratum_phase_spread`, `phase_occupancy_entropy` 不得缺值
4. `decision.md` 必須明示：若 B3.3 fail，下一步是 B5，而不是再回 B3 做 interval / offset 微調

**Gate 4: 正式 short-scout 前的人為停損檢查**

1. 若 Gate 2 已經顯示 `n_strata=3` 幾乎從不 rebucket，先檢查 phase 計算是否錯用 row-level aggregate 而非 player-level weights
2. 若 Gate 3 顯示 `phase_occupancy_entropy` 長期接近 0，先檢查 sector 分桶或初始偏置是否讓 player 全擠在同一角度區間
3. 若 Gate 2 能改變路徑、Gate 3 也有明顯 phase 分離診斷，才允許進正式 `300 x 3000 x seeds{45,47,49}` short scout

換句話說，B3.3 的最低可接受完成定義不是「程式能跑」，而是：

1. `n_strata=1` 精確退化
2. `n_strata>1` 真能造成 phase-aware 重分桶
3. 診斷輸出能證明分桶不是假動作
4. decision 規則已經把 fail 後路徑鎖到 B5

**2026-04-07 B3.3 實際結果**：正式 short scout 已完成，`n_strata ∈ {3,5,10}` 仍全部 `0/3` seeds 達到 Level 3，`mean_stage3_score` uplift 分別為 `-0.005986`, `-0.008163`, `-0.014437`，因此三個 active cells 全部 hard stop，`longer_confirm_candidate: none`。

這輪的重點在於：B3.3 並不是沒有打到分層機制。`mean_inter_strata_cosine` 已進一步壓低到 `0.554864 / 0.531782 / 0.558069`，`mean_growth_dispersion` 維持在 `0.186589 / 0.208335 / 0.179000` 的明顯異質區間，顯示 phase-aware aggregation 確實比 B3.1/B3.2 更強地把 strata-level growth direction 拆開。但這些差異仍然沒有轉成任何 seed-level attractor 變化；相反地，Stage3 score 對 control 全面轉負。

另一個值得記下的診斷是，`mean_phase_rebucket_churn` 只有 `0.000778 / 0.000869 / 0.001061`，代表在 `phase_rebucket_interval=100` 的 locked protocol 下，player 的 phase bucket 幾乎不重排；同時 `phase_occupancy_entropy` 仍有 `0.934434 / 0.793780 / 0.702750`，說明 bins 並未完全塌縮，只是這個 phase proxy 對實際 basin 切換仍過於溫和。依本輪事先鎖定的 closure rule，這不是回頭微調 `rebucket_interval` 或 `sector_offset` 的理由，而是 B3 family 的正式結案信號。

因此 B3.3 的正式結論是：**即使把 stratified growth 進一步升級為 phase-aware local aggregation，well-mixed sampled replicator 仍然維持同一個 Level 2 plateau；B3 主線到此正式 closure，下一步應轉入 B5 tangential drift，而不是再做 B3.3 內部參數細調。**

### 19.6 B5: 切線漂移項（Tangential Drift）

#### 19.6.1 為什麼排第三（而非更早）

B5 直接碰 operator，能最直接處理「就算有局部差異，更新後還是被重心化」的問題。但如果 drift 只是 heuristic 常數項，而不是幾何上可解釋的切向修正，很容易重演 W3 類型的假陽性：activation 變多、switch 變頻繁、Stage3 漂亮一點，但 basin 沒真的打開。

因此：**drift 的幾何來源必須先定義清楚，才決定是否升格成主線。若做不到可驗證的幾何解釋，就只當對照 heuristic。**

#### 19.6.2 幾何定義

在 3-simplex 上，任何向量 $v$ 可分解為徑向（向/離重心 $c = (1/3,1/3,1/3)$）與切線分量。切線漂移定義為：

設 $r(t) = x(t) - c$（當前狀態相對重心的偏移），則切線方向向量：

$$\tau(t) = \begin{pmatrix} 0 & -1 & 1 \\ 1 & 0 & -1 \\ -1 & 1 & 0 \end{pmatrix} r(t)$$

（即 RPS 環序的 90° 旋轉）。歸一化後：

$$\hat{\tau}(t) = \frac{\tau(t)}{|\tau(t)| + \epsilon}$$

漂移項：

$$\text{drift}(t) = \delta \cdot \hat{\tau}(t)$$

加在 growth vector 上：

$$g_s'(t) = g_s(t) + \text{drift}_s(t)$$

此旋轉算子有三個必須被測試保護的幾何事實：

1. $\tau \cdot r = 0$：純切線，不能偷帶徑向分量
2. $\tau \cdot \mathbf{1} = 0$：保持在 simplex 平面內
3. 對任何不在重心的狀態，原始旋轉向量都滿足 $|\tau| = \sqrt{3}\,|r|$；因此經歸一化後，實際注入的 drift 向量長度應固定為 $\delta$

**零均值投影（必要）**：加上 drift 後 $g'$ 可能不再 zero-mean（原始 `sampled_growth_vector()` 保證 $\sum g_s = 0$）。必須投影回 simplex hyperplane：

$$g''_s = g'_s - \frac{1}{3} \sum_{s'} g'_{s'}$$

確保後續 `exp(k * g''_s)` 仍在正確的 simplex constraint 上操作。

**Drift 注入點**：drift 必須在 `exp(k * g)` **之前**、`sampled_growth_vector()` **之後**加上。不得在 normalize 後才注入（否則失去幾何意義）。

#### 19.6.3 敘事合理性

- 物理類比：Stochastic resonance——有噪音的旋轉系統加小 bias 可穩定繞行
- 演化賽局類比：地下城 AI 有微小的 meta-game 策略偏好旋轉（環境的慢速 regime shift）
- 關鍵約束：$\delta \ll k \cdot \max|g_s|$，只做 symmetry-breaking 催化劑，不做主驅動力

**若 B5 通過但 $\delta$ 必須 > 0.02 才有效 → 不接受**（意味著不是 symmetry breaking 而是在人工注入動力，失去研究價值）。

#### 19.6.4 實作位置

新函式 `evolution/replicator_dynamics.py::tangential_drift_vector()`。在 `run_simulation.py` 的 growth vector 使用點注入。

實作包含兩步：
1. `tangential_drift_vector(x, delta)` → 計算 $\delta \cdot \hat{\tau}(t)$
2. 注入後立即投影回 zero-mean（$g'' = g' - \text{mean}(g')$）

**G1 前置檢查（在 Short Scout 前必做）**：以 deterministic mode 執行，確認加入 drift 後仍維持 Level 3 + turn_strength ≥ 92%。若 deterministic 部分 Level 3 掉到 Level 2 → delta 過大或 zero-mean 投影有 bug，先修再進。

#### 19.6.5 退化驗證

- `delta = 0.0`：drift 為零向量，完全退化
- 數值差異 < $10^{-12}$

#### 19.6.6 掃描參數

| 參數 | 掃描值 |
|---|---|
| `delta` | 0.000 (control), 0.003, 0.006, 0.010, 0.015 |
| `k_base` | 0.06（固定） |

5 組 × 3 seeds = 15 runs

#### 19.6.7 Hard stop rule

若 `delta <= 0.010` 的所有條件都 fail，B5 直接 closure，不掃更大 delta（那代表它不是催化劑而是主力）。

#### 19.6.8 第一輪實作清單

1. `evolution/replicator_dynamics.py` 先新增共用純函式，避免在 sampled / deterministic 各自複製一份 B5 邏輯：
    - `tangential_drift_vector(simplex, strategy_space, delta)`
    - `apply_tangential_drift(growth_vector, simplex, strategy_space, delta)`
   - 規格只鎖向量語意與 invariants，不強制綁定 NumPy；實作可用任意長度 3 的 array / 向量表示
2. `tangential_drift_vector()` 的 invariants 固定為：
    - `delta=0` 時必須精確回傳零向量
    - `x=(1/3,1/3,1/3)` 或 `|r|≈0` 時必須精確回傳零向量，不得製造任意方向
    - 輸出必須 finite，且三個分量和為 `0`
    - 對非退化狀態，輸出 drift 向量必須同時滿足：與 `r=x-c` 正交、與 `ones=(1,1,1)` 正交、且 `|drift|=delta`
3. `apply_tangential_drift()` 的責任固定為：
    - 先保留原始 growth `g`
    - 再加上 tangential drift
    - 最後做 zero-mean projection，回傳 `g_drifted`
    - 同時回傳診斷量，至少包含 `drift_norm`、`effective_delta_growth_ratio`、`tangential_alignment`
4. 第一輪只改兩個 operator 入口：
    - `deterministic_replicator_step()`：供 G1 deterministic gate 使用
    - `replicator_step()`：供 G2 sampled short scout 使用
5. 第一輪**不**修改 `inertial_sampled_replicator_step()`、`personality_coupled`、B3 strata callback 等其他分支；先把 B5 鎖成最小可判讀版本，避免一次跨太多 operator。
6. `simulation/run_simulation.py` 只做最小 wiring：
    - `SimConfig` 新增 `tangential_drift_delta`
    - CLI 新增 `--tangential-drift-delta`
    - 在既有 sampled / mean-field 更新點注入當前 simplex 向量 `x(t)=[p_s(t)]` 作為 drift 幾何狀態
7. B5 第一輪不得新增第二種 timeseries schema；所有新增資訊都留在 summary / combined / decision / PNG 診斷。
8. 新 harness 固定為 `simulation/b5_tangential_drift.py`，責任分成三段：
    - G0：helper-level 與 runtime-level 的 `delta=0` 退化驗證
    - G1：deterministic gate
    - G2：locked short scout 與 decision markdown
9. B5 harness 不另外發明 plotting 規格；沿用既有 simplex 與 phase-amplitude 圖，但必須額外補一張 `drift_vector_rose_plot`，確認 drift 方向是否真的沿 RPS 環序穩定推進。

#### 19.6.9 第一輪測試 gate 與輸出

1. `tests/test_evolution.py` 至少新增四個 helper-level 測試：
    - `tangential_drift_vector()` 在重心回傳零向量
    - `tangential_drift_vector()` 三分量和為零、且與半徑向量正交
    - `tangential_drift_vector()` 同時滿足 `tau·ones=0` 與 `|drift|=delta`
    - `apply_tangential_drift(..., delta=0)` 精確退化為原 growth
    - `replicator_step(..., tangential_drift_delta=0)` 精確退化為既有 sampled control
2. `tests/test_b5_tangential_drift.py` 至少覆蓋：
    - G0 control 與 matched baseline 逐列一致
    - G1 deterministic gate 會輸出 `turn_strength_ratio`、`level_preserved`、`drift_contribution_ratio`
    - 若任一 active cell 的 `drift_contribution_ratio > 0.05`，G1 直接 fail，避免 drift 偷偷變成主力
    - G2 smoke 會產出 `summary.tsv`、`combined.tsv`、`decision.md`、simplex 圖、phase-amplitude 圖、`drift_vector_rose_plot`
3. 第一輪 summary / combined 至少必須新增：`tangential_drift_delta`, `mean_drift_norm`, `mean_effective_delta_growth_ratio`, `mean_tangential_alignment`, `phase_amplitude_stability`, `representative_phase_amplitude_png`, `representative_drift_vector_rose_png`
4. decision markdown 至少必須明示：
    - `delta<=0.010` 是否出現任何 Level 3 seed
    - 最佳 active cell 的 `effective_delta_growth_ratio`
    - 最佳 active cell 的 `tangential_alignment` 與 `phase_amplitude_stability`
    - 若唯一正向候選來自 `delta>0.010`，則明確標記為「不接受，因為 drift 已成主驅動」
    - 若 `delta<=0.010` 全 fail，結尾必須明示：`B5 正式 closure，證明即使幾何明確的切線催化仍不足以對抗三層均化。`
5. 第一輪正式 short scout 命名固定為 `outputs/b5_tangential_drift_short_scout_*`，避免和 B3/B4 產物混淆。

#### 19.6.10 B5 Formal Closure（2026-04-07）

第一輪 formal scout 已依 locked protocol 完成：`simulation.b5_tangential_drift` 在 `delta ∈ {0.000, 0.003, 0.006, 0.010, 0.015}`、`seeds={45,47,49}`、`players=300`、`rounds=3000`、`memory_kernel=3` 的 working point 上產生完整 summary / combined / decision 與 B5 專屬圖表。

**G1 結論**：所有 nonzero deltas 都通過 deterministic gate。`Level 3` 完整保留，`drift_contribution_ratio` 只有 `0.003572 / 0.007110 / 0.011820 / 0.017800`，遠低於 hard cap `0.05`。這表示 B5 的 helper 與 runtime wiring 是乾淨的，drift 在 deterministic 幾何上確實只是催化級擾動，而不是主驅動。

**G2 結論**：matched sampled control 為 `level_counts={0:0,1:0,2:3,3:0}`, `mean_stage3_score=0.515010`, `mean_env_gamma=-0.000066`。三個 closure-relevant active cells `delta=0.003 / 0.006 / 0.010` 全部都是 `0/3` `Level 3` seeds，且 `mean_stage3_score` uplift 分別為 `-0.001174 / -0.000512 / -0.001883`。因此依本節事先鎖定的 hard stop，B5 第一輪正式記為 `close_b5`，不再掃更大 delta。

`delta=0.015` 也沒有救回任何 seed-level breakthrough：`level3_seed_count=0`、`mean_stage3_score` uplift `=-0.002896`，只能記為診斷性的 `weak_positive`，不能當成 B5 尚未 closure 的理由。

值得記下的是，B5 專屬診斷全部顯示「機制有打進去，但沒有轉成 emergence」：`mean_effective_delta_growth_ratio` 大約從 `0.006394` 升到 `0.031968`，`mean_tangential_alignment` 穩定落在約 `-0.011`，而 `phase_amplitude_stability` 幾乎不變（約 `0.8107~0.8109`）。也就是說，切線 drift 並沒有被 sampled noise 完全洗掉；它只是仍不足以把系統推出既有的 `Level 2` plateau。

因此 B5 的正式收束語句可固定為：**幾何明確的 tangential drift 足以通過 deterministic gate，但在 well-mixed sampled replicator 的 locked working point 下，仍無法把任何 seed 從既有 `Level 2` plateau 推入 `Level 3`；B5 到此正式 closure。**

### 19.7 B2: 島嶼模型（Topology-Preserving Control）

#### 19.7.1 定位

**不當主線，當 topology 對照組。** 理由：

1. 本 repo 已有不少 island / subgroup 的前身：`run_simulation.py` 和 `replicator_dynamics.py` 都有 fixed subgroup、anchor pull、coupling 類掛點
2. 若把 B2 誤當主線，風險是又回到 H3/W 類的 family-internal 微調
3. B2 真正的價值是回答：「把全域均勻平均打破成局部結構，這件事本身夠不夠？」

只有在 B4/B3/B5 都 fail 之後，才考慮升格為正式 family。升格前必須先回答：

1. Deme 間互動規則（population exchange vs. payoff coupling？）→ 已決定：物理遷移（pooled random redistribution）
2. 全域指標怎麼由 deme-level 組合（加總 vs. 最佳 deme？）→ 已決定：合併所有 deme 計算 global p_*(t) → Stage3
3. Provenance 怎麼紀錄（CSV schema 影響？）→ 已決定：不改主 CSV schema，B2 專屬指標寫入 summary TSV / decision MD

#### 19.7.2 結構定義（已鎖定）

把 N 個玩家分成 M 個 deme（$N_d = N/M$）。

- **Deme 內**：獨立追蹤 `popularity`，local `sampled_growth_vector()` 只看本 deme
- **Deme 內 payoff**：使用 per-deme strategy distribution 計算 `_matrix_ab_payoff_vec(x=deme_dist)`，不共享全域 payoff
- **Deme 內 evolution**：per-deme `replicator_step(deme_players, ...)` + per-deme weight broadcast
- **Deme 間遷移**：每 $T_{mig}$ 輪，從每個 deme 抽出 $f\%$ 玩家，pool 後隨機重新分配（symmetric pooled redistribution）
- **全域判定**：所有 deme 合併後計算 `p_*(t)` → 走 Stage3

第一輪掃描（已鎖定）：$M = 3$，$T_{mig} \in \{100, 200\}$，$f \in \{0.02, 0.05, 0.10\}$

推薦預設：小 M=3、N_d=100（足夠支撐 deme-local growth 精度）。M=5（N_d=60）保留到第二輪（若第一輪 B2 未 closure）。

**必記診斷指標**：
- `mean_inter_deme_phase_spread`：per-deme phase angle 的 circular range（tail 平均）
- `max_inter_deme_phase_spread`：tail 窗口最大 inter-deme phase spread
- `mean_inter_deme_growth_cosine`：per-deme growth vector 的 pairwise cosine（tail 平均）
- `phase_amplitude_stability`（同 B5）
- per-deme popularity trajectory（每個 deme 的 $p_*(t)$ 獨立記錄在 timeseries per-condition CSV columns）

#### 19.7.3 與 H3 的本質區別

| | H3 子族群 | B2 島嶼 |
|---|---|---|
| **growth 聚合** | 全域共享 growth vector | **deme-local** growth vector |
| **popularity** | 全域共享 | **deme-local** |
| **payoff** | 全域共享 | **deme-local**（per-deme distribution → payoff） |
| **更新規則** | 所有玩家用同一個 weights | 每 deme 用自己的 weights |
| **耦合** | 功能性（frozen anchor pull） | 物理性（玩家遷移） |

若要把 B2 升格為正式 family，以上五項差異必須全部實現，否則本質上只是 H3 的重包裝。

#### 19.7.4 B2 Gate 設計

| Gate | 內容 | Pass 條件 | Fail 動作 |
|---|---|---|---|
| **G0: Degrade** | M=1 control（全體 300 人在一個 deme）| 與 well-mixed sampled 行為一致 | 先修 B2 inner loop |
| **G2: Short Scout** | M=3, 3 seeds, 3000 rounds per condition | 見 §19.9.3 | 該 condition 記 fail，進下一個 |

**為什麼沒有 G1（Deterministic Gate）**：mean-field 是 well-mixed ODE，不存在 deme 結構。B5 需要 G1 是因為 tangential drift 可以在 mean-field 中測試幾何影響；B2 的核心機制（空間 topology）在 mean-field 中無法表達，因此 G1 不適用。

#### 19.7.5 B2 Stop Rule

1. 若所有 M=3 active conditions（6 conditions × 3 seeds = 18 runs）都 `0/3` Level 3 且 `max(stage3_uplift) < 0.02` → B2 直接 closure，不掃 M=5
2. 若所有 M=3 conditions 的 `mean_inter_deme_phase_spread < 0.10` rad → topology 效應未形成，可加註 early closure 標記
3. 若 ≥1/3 seeds 在任何 condition 達 Level 3 且 `mean_env_gamma ∈ [-5e-4, +5e-4]` → `short_scout_pass`，推薦 Longer Confirm

#### 19.7.6 B2 升級路徑（若第一輪 weak positive 或 borderline）

- 增加 M 到 5~8（但注意 N_d 變小，stochasticity 增強）
- 加入 mild deme 間 payoff coupling（小值 0.05~0.1）
- 改用 directed migration 或 size-heterogeneous demes（第一輪禁止）
- 啟用 events（第一輪禁止）

#### 19.7.7 B2 風險分析

- **f 太高**：deme 快速同步 → 退化為 well-mixed（Level 2）→ 應觀察 `mean_inter_deme_phase_spread` 接近 0
- **f 太低 / T_mig 太長**：各 deme 可能獨立收斂到單一策略 → deme 內 rotation 消失 → 全域仍 Level 2 但原因不同
- **N_d 太小**（M=5, N_d=60）：sampling noise 壓過 deme-local signal → 第一輪用 M=3（N_d=100）避免此問題


#### 19.7.8 B2 Formal Result（已完成）

**日期**：2026-04-07  
**overall_verdict**: `close_b2`

| condition | mean_inter_deme_phase_spread | max_stage3_uplift | level3_seeds |
|---|---|---|---|
| g2_M3_f0p02_T100 | 1.934 rad | +0.008 | 0/3 |
| g2_M3_f0p02_T200 | 1.949 rad | +0.011 | 0/3 |
| g2_M3_f0p05_T100 | 1.926 rad | -0.001 | 0/3 |
| g2_M3_f0p05_T200 | 1.929 rad | +0.002 | 0/3 |
| g2_M3_f0p10_T100 | 1.941 rad | -0.002 | 0/3 |
| g2_M3_f0p10_T200 | 1.938 rad | +0.007 | 0/3 |

**數據解讀**：
- 所有 18 active runs（6 conditions × 3 seeds）均為 `0/18` Level 3 seeds，依 stop rule 正式 closure
- `max(stage3_uplift) = 0.011`（f=0.02, T=200, seed47），低於 0.02 threshold；uplift 無系統性上升趨勢
- **Topology 效應確有形成**：所有 conditions 的 `mean_inter_deme_phase_spread ≈ 1.93 rad`，遠高於 early-stop threshold（0.10 rad），`mean_inter_deme_growth_cosine ≈ 0.61`；deme 間確有 phase separation 與 growth diversity
- **均化壓力更強**：即使形成空間相位差異，全局採樣加權仍將 deme-level 局部動態均化，Level 3 basin 始終無法在全局尺度觸發
- 結論：island deme topology 本身不足以對抗三層均化；B2 提供正式 negative result（topology 對照），而非 runtime 失敗

**Artifacts**：
- Decision: `outputs/b2_island_deme_short_scout_decision.md`
- Summary: `outputs/b2_island_deme_short_scout_summary.tsv`
- Plots: `outputs/b2_island_deme_short_scout/*/seed*_deme_simplex.png`

### 19.8 B1: 切線投影 Replicator（最後手段）

#### 19.8.1 定位

B1 在理論上最乾淨：直接承認 simplex 上重要的是切向旋轉不是徑向膨脹。但工程上最難：

1. 投影後需保證所有權重仍為正值（simplex 約束）
2. 與現有 `exp(k * growth) / normalize` 流程的相容性不明確
3. Spec 上需定義「投影後仍是合法演化更新」的語意
4. 無法依附現有 `personality_coupled` 或 `sampled_inertial` 路徑——需要全新 evolution mode

**只有當 B4、B3、B5、B2 都證明「局部差異保留下來仍不足以打開 basin」時，才值得付出 Spec + 工程成本。**

#### 19.8.2 數學定義（保留備用）

設重心 $c = (1/3, 1/3, 1/3)$，growth vector $g$：

$$g_R = \frac{\langle g, \, x(t) - c \rangle}{\|x(t) - c\|^2} (x(t) - c) \quad \text{(radial)}$$

$$g_T = g - g_R \quad \text{(tangential)}$$

修正：

$$g' = \alpha_R \, g_R + \alpha_T \, g_T$$

$\alpha_R \in [0.5, 1.0]$，$\alpha_T \in [1.0, 3.0]$。

正值性保證（open question）：修正後的 `exp(k * g'_s)` 天然為正，但 `g'` 不再保證 zero-mean（因為在 simplex 超平面上的投影未必保持和為零）。需要額外分析或 clamp。

### 19.9 共用 Gate 設計與驗證門檻

#### 19.9.1 與既有 protocol 的一致性

維持目前已鎖定的 protocol，**不重開大 sweep**：

- `players=300, rounds=3000, burn_in=1000, tail=1000`
- `seeds=45,47,49`（Short Scout）/ `seeds=45,47,...,63`（Longer Confirm, 10 seeds）
- `series=p, amplitude_threshold=0.02, corr_threshold=0.09, eta=0.55`
- `stage3_method=turning, phase_smoothing=1`
- `memory_kernel=3, selection_strength=0.06, init_bias=0.12`
- `payoff_mode=matrix_ab, a=1.0, b=0.9, matrix_cross_coupling=0.20`

這些與 W2.1R ~ W3.3 的 formal protocol 以及研發日誌.md 的 closure 語氣保持一致。

#### 19.9.2 階段閘門

| Gate | 內容 | Pass 條件 | Fail 動作 |
|---|---|---|---|
| **G0: Degrade** | 新機制的調控參數 = 0 時，必須完全退化為 control | max diff < $10^{-12}$ | 不進 smoke，先修 code |
| **G1: Deterministic** | mean-field 下新機制不破壞已知 Level 3 | Level 3 維持 + turn_strength ≥ 92% baseline | 不進 sampled |
| **G2: Short Scout** | sampled, 3 seeds, 3000 rounds | 判讀見下 | 該提案記 `fail`，進下一個 |
| **G3: Longer Confirm** | sampled, 10 seeds, 3000 rounds, `tail=1000` | 判讀見下 | 記 `weak_positive`，視 evidence 決定 |

#### 19.9.3 成功判讀標準（不接受短窗 ridge）

**只接受長窗 tail 的 seed-level basin change**。具體：

- **Short Scout Pass**：≥ 1/3 seeds 在 `tail=1000` 達到 Level 3，**且** `mean_env_gamma` 在 $[-5 \times 10^{-4}, +5 \times 10^{-4}]$ 內（不接受系統性發散）
- **Longer Confirm Pass**：$P(\text{Level3}) \ge 0.30$（即 ≥ 3/10 seeds），**且** `mean_env_gamma` 同上約束
- **Hard stop**：若 Short Scout 中 0/3 seeds Level 3 **且** `mean_stage3_score` uplift < 0.02，直接記 `fail`，不做 Longer Confirm

**不接受的假陽性模式**（從 H-series 學到的教訓）：

| 模式 | 為什麼拒絕 | 曾在哪裡出現 |
|---|---|---|
| 短窗 ridge（短 rounds 才 Level 3） | 不代表穩定 basin | H3.5, H4.1 |
| switch 次數增加但 Level 不變 | activation ≠ effectiveness | W3.2, W3.3 |
| Stage3 score uplift 0.003~0.01 | Level 2 plateau 內微移 | W3.1, H5.2 |
| 單一 seed 達 Level 3 而其他全 Level 2 | 不穩健 | 多處 smoke |

#### 19.9.4 共用診斷產出（所有 B-series 提案必備）

不論哪條線通過或 fail，**每次 G2/G3 掃描都必須記錄並產出以下診斷**，以便 decision.md 能寫出可比較的 closure：

| 類別 | 產出 | 用途 |
|---|---|---|
| **核心指標表** | turn_strength, mean_amp, regime_switches, env_gamma, stage3_score（per-seed） | 量化判讀（與 19.9.3 判讀標準對照） |
| **Simplex 軌跡圖** | p(t) 三策略在 2D simplex 上的投影軌跡（tail 窗口；輸出 PNG/SVG） | 視覺確認是否真有穩定旋轉（非零散 switch） |
| **Phase vs. Amplitude 圖** | phase angle θ(t) vs. amplitude ‖x(t)−c‖ 時序圖（輸出 PNG/SVG） | 區分「穩定 limit cycle」vs.「隨機 drift」 |
| **B5 Drift Rose Plot** | drift direction 的極座標圖（tail 窗口；輸出 PNG/SVG） | 檢查 drift 是否真的沿固定 RPS 環序提供對稱性破缺，而非隨機抖動 |
| **提案專屬指標** | B4: k_clamped_ratio; B3: inter-strata cosine similarity; B5: effective δ/g ratio + tangential alignment + phase-amplitude stability; B2: inter-deme phase spread | 判斷機制是否真的在起作用 |

Simplex 軌跡定義：
- $x$-axis: $p_B - p_C$（水平）
- $y$-axis: $p_A - (p_B + p_C)/2$（垂直）
- 只畫 tail 窗口（最後 1000 rounds），用顏色梯度代表時間先後
- decision.md 至少附一張代表性 seed 圖（最佳 seed 與最典型 fail seed 各一，若篇幅允許）

Phase angle 定義（與 B3 phase-aware strata 一致）：
$$\theta(t) = \text{atan2}\big(\sqrt{3}(p_B(t) - p_C(t)),\; 2p_A(t) - p_B(t) - p_C(t)\big)$$

**若 simplex 軌跡顯示穩定繞行但 Stage3 score 未達 Level 3 → 需檢查 Stage3 判讀的 phase smoothing 或 turning 閾值是否需要校準（但不主動調整，先記錄）**。

#### 19.9.5 B-series 全線 fail 後的 meta-結論

若 B4/B3/B5 全線 fail（主力線耗盡），在進入 B2/B1 之前，必須先回答一個 meta-question：

> **「局部差異保留下來仍不足以打開 basin」是否已被證實？**

判定方式：
1. B4 提供了 per-player 異質 k → 是否有任何 seed 顯示 simplex 軌跡的旋轉半徑增加？
2. B3 提供了 strata-level growth → inter-strata 是否真的出現不同方向？
3. B5 提供了外部切線偏移 → 是否有任何 seed 出現完整環繞？

若以上三者的診斷指標全部顯示「異質性/方向性未被保留」 → 結論是「well-mixed sampled replicator 的均化太強，需要 topology (B2) 或 operator (B1) 的根本改變」。這個 meta-結論本身即是重要的研究發現，應記錄入研發日誌。

**2026-04-07 補充判讀**：B3 family 的 closure 尤其重要，因為它不是「機制沒打進去」，而是更強的負結果。B3.1、B3.2、B3.3 依序把 `mean_inter_strata_cosine` 從接近 `1.0` 壓到約 `0.53~0.56`，同時維持非零 `mean_growth_dispersion` 與非塌縮的 phase occupancy；但所有 active cells 仍是 `0/3` Level 3 seeds，且 Stage3 uplift 全面落在 0 以下。這表示第二層 growth aggregation 即使被部分拆開，第三層 normalize 與後續 shared update 仍足以把系統壓回同一個 Level 2 plateau。換句話說，B3 現在已經正式回答了 meta-question 的一半：**局部方向差可以被保留，但仍不足以打開 basin。**

**2026-04-07 B5 補充判讀**：B5 把 meta-question 的另一半也正式補完了。這次已證明幾何明確、量級受控的 tangential drift 在 deterministic gate 下完全可存活，且 `drift_contribution_ratio` 明顯低於主驅動門檻；但一旦回到 sampled gate，`delta<=0.010` 仍全部 `0/3` Level 3 且 `stage3_uplift<0`。因此 B5 現在給出的不是「drift 沒有成功注入」，而是更乾淨的 negative result：**就算把外部切線催化直接打進 operator，well-mixed sampled replicator 仍會把系統壓回同一個 Level 2 plateau。**

建議標準化記錄語句：

> 在 well-mixed sampled replicator 下，即使依序引入 state-dependent k、stratified growth、切線 drift 等局部差異保留機制，三層均化仍強到無法穩定打開 Level 3 basin。

#### 19.9.6 獨立 Discussion：well-mixed sampled framework 的根本限制

這一節的目的不是再把 B4、B3、B5、B2 四條線各自重述一次，而是把它們收斂成一個更強、可對外防守的研究命題：**目前的瓶頸不是「還沒找到夠強的局部異質性」，而是 well-mixed sampled framework 本身會把任何局部差異重新壓回同一個全域更新物件。** 換句話說，B-series 的價值不在於列出四次 fail，而在於用四種不同層級的介入，逐層排除了「局部差異不足」這個較弱解釋。

如果把 sampled 路徑拆成一條因果鏈，可以近似寫成：

1. 先由全域 population share 估計 payoff / growth 訊號
2. 再把這些訊號壓成共享的 growth 或共享的更新方向
3. 最後經過 normalize 與廣播，回寫成整體 population 的下一步混合策略

在這條鏈上，B4、B3、B5、B2 分別對應四個不同層級的反證：

| family | 介入層級 | 已被證明成立的正向診斷 | 最終仍失敗的關鍵結果 | 研究含義 |
|---|---|---|---|---|
| **B4** | per-player selection strength | state-dependent `k` 已成功進入 runtime；`k_clamped_ratio` 幾乎為 0，表示不是 clamp 飽和假 fail | 所有 active cells 皆 `0/3` Level 3；最高 `mean_stage3_score` uplift 僅 `0.013733`；最高 `k_clamped_ratio=0.001889` | 失敗原因不是 `k` 值域太窄，也不是 per-player 異質性根本沒有打進去，而是訊號太弱、且很快被後續共享更新洗平 |
| **B3** | growth aggregation | `mean_inter_strata_cosine` 從接近 `1.0` 壓到約 `0.53~0.56`；`mean_growth_dispersion` 明顯非零；phase bins 未塌縮 | B3.1/B3.2/B3.3 全部 active cells 仍 `0/3` Level 3，且 B3.3 uplift 全面為負 | 第二層 averaging 確實可被部分拆開，但只保留局部 direction difference 仍不足以改變 seed-level basin |
| **B5** | operator-level tangential catalyst | deterministic gate 全通過；`drift_contribution_ratio=0.003572~0.017800`，明顯低於 hard cap；drift 幾何正確且非 no-op | sampled gate 中 `delta<=0.010` 全部 `0/3` Level 3，uplift 全為負值 | 問題不是 operator 完全缺少切向催化；即使明確切線 drift 存活，sampled path 仍會把它壓回 Level 2 plateau |
| **B2** | topology / local phase separation | `mean_inter_deme_phase_spread≈1.93 rad`，`mean_inter_deme_growth_cosine≈0.61`；deme-level phase separation 真實存在 | 全 18 active runs 仍 `0/18` Level 3；`max(stage3_uplift)=0.011 < 0.02` | 問題甚至不只是沒有空間局部化；即使 local phase separation 已經形成，全局 sampled 整合仍無法把它轉成 global Level 3 |

這四條線之所以要被串成**因果鏈**，就在於它們排除的是不同層級的替代解釋，而不是互相重複：

1. **B4 排除「只是每個 player 的有效更新速度太同質」**。如果 sampled plateau 的原因只是全體玩家共用同一個 `k`，那麼 state-dependent `k` 至少應該在一部分條件下產生可持續 uplift，或至少呈現 clamp 飽和跡象，表示方向正確只是力度受限。但實際上，B4 的 `k_clamped_ratio` 幾乎為 0，說明失敗不是因為界限卡死，而是這類 per-player 異質性本身不足以跨過 basin 邊界。
2. **B3 排除「只是 growth aggregation 太早被全域平均」**。這條線比 B4 更強，因為它不是單純注入靜態異質性，而是直接在 second-layer averaging 上動手術。B3.1 到 B3.3 已逐步證明：strata-level direction difference 可以被做出來、可以被加強、甚至可以與當前 phase geometry 對齊；然而這些局部方向差仍然無法轉成任何 robust 的 Level 3 seed。這表示問題不只是缺少 local direction difference，而是這些差異在後續 shared update 中仍被重新平均掉。
3. **B5 排除「只是 operator 本身沒有切向幾何偏壓」**。B5 的重要性在於它不再停留在 growth aggregation，而是把幾何上明確、量級受控的 tangential drift 直接打進 operator。deterministic gate 全通過，意味著 drift 不是 runtime bug，也不是過強主驅動；但 sampled gate 仍完全 flat。這使得 B5 的 negative result 比單純的 "no uplift" 更乾淨：**就算切線催化存在，well-mixed sampled replicator 仍會把它消化成同一個 Level 2 plateau。**
4. **B2 最後排除「只要把 well-mixed 拆成局部結構就夠了」**。B2 是最有力的 topology 對照，因為它真的讓 deme-level payoff、deme-level growth、deme-level weight broadcast 分開跑，而且得到非常強的正向診斷：phase spread 接近 $\pi$，growth cosine 明顯小於 1。也就是說，空間局部化不是沒有形成，而是形成了仍不夠。這個結果把整個問題推到最核心的層級：**真正的瓶頸不是 local structure 缺失，而是 local structure 最終仍被 global sampled aggregation 收斂成單一 population-level 更新物件。**

因此，這四條線合起來所支持的不是「目前還沒找到對的 patch」，而是一個更高階的結論：**well-mixed sampled framework 的根本限制，在於它持續把局部異質性、局部方向差、局部切向催化、甚至局部空間相位分離，都重新投影回同一個全域 sampled 更新路徑。** 一旦這個投影存在，前端再怎麼保留局部訊號，後端仍會把訊號壓回同一個 shared simplex trajectory。

這也解釋了為什麼 B-series 的負結果不是四個平行 patch fail，而是一個逐層收斂的結論：

- B4 說明：不是缺少 per-player heterogeneity。
- B3 說明：不是缺少局部 growth-direction heterogeneity。
- B5 說明：不是缺少 operator-level tangential catalyst。
- B2 說明：不是缺少真實的 local phase separation。

當這四個「不是」都成立後，剩下最合理的主假說就變得非常明確：**若要真正打開 robust Level 3 basin，後續工作不能只是再往現有框架上疊更多局部異質性，而必須直接改變 global sampled aggregation / update 機制本身。**

這個新假說可以寫得更精確：

> 在目前的 well-mixed sampled replicator 中，決定 basin 的不是局部訊號是否存在，而是局部訊號是否被允許在多輪中維持為彼此不同的更新物件。只要系統每輪仍把這些訊號重新壓回單一全域 sampled growth / shared update，局部 heterogeneity 就只能造成 plateau 內微移，無法穩定轉化成 seed-level Level 3 emergence。

依這個假說，下一步最有研究價值的方向不再是「再找一種更強的局部 patch」，而是以下兩類根本改動：

1. **改 aggregation**：把 global sampled growth 改成 local pairwise、mini-batch local replicator、或 network-neighborhood update，使不同區域的局部訊號不必在每輪立即合併成單一方向。
2. **改 synchronization**：把每輪 shared normalize + broadcast 改成非同步、局部同步、或 network-based interaction，避免所有玩家在每一步都被拉回同一個 population-level update。

這也是為什麼目前最合理的後續主線，不是回頭重做 B4/B3/B5/B2 的 family-internal 微調，而是直接測試「**打破 global sampled aggregation**」本身是否就是 Level 3 basin 的必要條件。若未來 local pairwise / network-based interaction 線仍失敗，屆時再回頭做 B1（切線投影 operator）才有真正的 closing-the-framework 意義；否則太早進 B1，會把高成本理論線浪費在尚未釐清的 aggregation 問題上。

給 paper / discussion 可直接使用的核心結論可以固定為：

> B-series does not merely report four negative mechanism patches. It establishes a nested causal exclusion chain. B4 shows that per-player selection heterogeneity is insufficient; B3 shows that partially preserving local growth directions is insufficient; B5 shows that explicit tangential catalytic bias is insufficient; and B2 shows that even genuine local phase separation is insufficient when all local signals are ultimately recombined into a single global sampled update. The strongest remaining hypothesis is therefore structural: robust Level 3 emergence is unlikely to be recovered by adding more local heterogeneity on top of the same well-mixed sampled framework, and instead requires changing the global sampled aggregation rule itself.

### 19.10 執行計畫

| 階段 | 內容 | 產物 | 不碰的東西 |
|---|---|---|---|
| **Phase 0** | 在 SDD.md 補 B-series 研究契約 + stop rule | SDD.md §B | runtime code |
| **Phase 1** | B4 實作 + G0-G3 | CSV + decision.md | 其他 B-series |
| **Phase 2** | B3 實作 + G0-G3（僅在 B4 fail 後） | CSV + decision.md | B5/B2/B1 |
| **Phase 3** | B5 實作 + G0-G3（僅在 B3 fail 後） | CSV + decision.md | B2/B1 |
| **Phase 4** | B2 topology 對照（僅在 B5 fail 後） | CSV + decision.md | B1 |
| **Phase 5** | B1 理論線（僅在全主力 fail 後） | SDD.md 新增 + CSV | — |

### 19.11 為什麼這五個方向與 H-series 不重複

| 層面 | H-series 已做（且已全線 closure） | B-series 新方向 |
|---|---|---|
| **Growth 聚合** | 全族群壓成 shared vector ✗ | B3: strata-level 先聚合再合成 |
| **選擇壓力結構** | 固定 k 或靜態 personality k ✗ | B4: 動態 state-dependent k |
| **Operator 對稱性** | 保持對稱 ✗ | B5: 幾何明確的切線漂移 |
| **Population topology** | well-mixed + 功能子族群 ✗ | B2: 空間 deme 結構（topology 對照） |
| **投影幾何** | 未做 | B1: 切線/徑向分離（最後手段） |

核心差異：H-series 在 well-mixed + 均勻 replicator 的框架內做微調（payoff 強化、子族群隔離、operator inertia），所有方案都無法阻止 sampled growth aggregation 把方向性磨平。B-series 直接攻擊三層均化的不同環節——第二層聚合（B3）、第三層壓平的空間依賴性（B4）、operator 對稱性（B5）、全域結構（B2）、或投影幾何（B1）。
