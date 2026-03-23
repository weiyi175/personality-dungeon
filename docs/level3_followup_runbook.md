# Level 3 Follow-up Runbook

本文件把目前建議的後續優化方向整理成可直接執行的研究 runbook。
目標不是一次把所有網格炸完，而是用最小成本先確認突破可信，再決定是否往更高分數推進。

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

## 1. 推薦執行順序（更省力版）

| 優先級 | 批次 | 名稱 | 調整後內容 | 條件數 | 預期目的 | 通過標準 |
|---|---|---|---|---:|---|---|
| P0 | Batch A | 控制組 | `cross=0.0` @ `seed=45` | 1 | 驗證 cross-term 是 causal knob | `score` 回到 `0.546–0.548`，且為 Level 2 |
| P1 | Batch B | 多 seed | `seed ∈ {45,47,49,51,53,55}`，固定 `cross=0.16` | 6 | 判斷是否只是單 seed 神蹟 | `>= 3/6` 達到 Level 3，或平均 `score > 0.552` |
| P2 | Batch C-light | 甜區輕量細修 | `c ∈ {0.14,0.155,0.16,0.165,0.18}` @ `seed=45` | 5 | 看甜區形狀 | 峰值 `>= 0.555` 且非單點突出 |
| P3 | Batch D-light | 10000-round 驗證 | `rounds=10000`，`cross=最佳值`，`seed=45` | 1 | 確認不是短暫暫態 | `tail=1000` 與 `tail=3000` 都維持 Level 3 |
| P4 | Batch E-core | 機制診斷 | phase portrait + `turn_strength` 移動平均 | - | 建立機制故事 | 軌道更封閉、方向更一致 |
| P5 | Batch F-mini | 小型 knob 探索 | 只測 `3–4` 組最有希望條件 | 3–4 | 試探是否還能上推 | 若提升 `> 0.01` 再擴大 |

### 1.1 一句話版策略

先用 `1 + 6 + 5 + 1` 的最小可信驗證路徑，把 breakthrough 是否為 artifact 釘死，再決定要不要追 `0.57+`。

### 1.2 更激進的省力方案（三連擊）

如果運算資源更緊，可先縮成：

1. Batch A：`cross=0.0 @ seed=45`
2. Batch B-light：`cross=0.16 @ seed={45,48,51}`
3. Batch C+D 合併：`cross ∈ {0.155,0.16,0.165}`，`rounds=10000`，`seed=45`

判讀規則：

1. 三連擊都通過，再進入機制分析與小型 knob 探索。
2. 任何一步卡住，就把 paper 敘事降級為「在特定有限條件下可觀察到的暫態強化現象」，主打機制洞察而非極致分數。

## 2. 共用環境變數

先在 repo root 執行：

```bash
cd /home/user/personality-dungeon
export PYTHON_BIN=./venv/bin/python
export EVENTS_JSON=docs/personality_dungeon_v1/02_event_templates_v1.json
```

## 3. 共用診斷命令

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

## 4. Batch A：控制組回歸測試

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

## 5. Batch B：Multi-seed Level 3 穩健性

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

## 6. Batch C-light：Cross-coupling 甜區細修

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

## 7. Batch D-light：10000-round 穩定性

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

## 8. Batch E-core：機制診斷

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

## 9. Batch F-mini：組合 knob 探索

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

## 10. 決策規則

### 10.1 如果 Batch A 失敗

表示 `c=0.0` 時無法回歸舊 baseline，優先停下來檢查：

1. `matrix_cross_coupling` 預設值是否有污染
2. 診斷設定是否與前次不一致
3. 輸出檔名是否混用舊資料

### 10.2 如果 Batch B 顯示只有 seed=45 達到 Level 3

paper 敘事應改成：

1. 已觀察到 Level 3 breakthrough
2. 但目前具有明顯 seed sensitivity
3. 下一步優先做甜區擴張與更穩定條件搜索，而不是直接宣稱 robust Level 3

### 10.3 如果 Batch D 長跑掉回 Level 2

paper 敘事應改成：

1. 8000-round finite-window Level 3 breakthrough
2. 但長時間吸引子仍待確認

### 10.4 如果 Batch B 與 Batch D 都通過

此時再做 Batch F 才最有價值，因為：

1. 你已經有可信突破
2. 追高分數才不會淪為 tuning artifact

### 10.5 eta 敏感度建議（低成本高價值）

因為很多條件都卡在 `0.54–0.56`，建議在 paper 明確鎖定：

1. 主文一律採用 `eta=0.55`
2. 附錄補 `eta=0.53 / 0.57` 敏感度

這樣可以避免 reviewer 質疑 Stage 3 閾值是事後挑的。

### 10.6 config 指紋建議

建議所有輸出檔名都保留簡短 config 指紋，例如：

1. `crossc0p16_s1p5_t0p27_seed45_v2.csv`
2. `crossc0p0_s1p5_t0p27_seed45_ctrl.csv`

避免後期混淆不同批次產物。

## 11. 建議今天 / 明天的最小執行集

### 今天

1. Batch A：控制組回歸測試
2. Batch B：multi-seed 穩健性
3. Batch C-light：甜區細修（若時間允許）

### 明天

1. Batch D-light：10000-round 穩定性
2. Batch E-core：機制診斷
3. Batch F-mini：組合 knob 探索（僅在前四批結果站得住時）

## 12. 一句話版執行策略

先驗證 `cross-term` 是真的、可回歸、可跨 seed、可長時間維持，再去追求更高分數。