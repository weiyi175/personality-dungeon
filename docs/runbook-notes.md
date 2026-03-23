# Runbook Notes — Personality Dungeon Research

研究級操作備忘，依日期倒序。

---

## Stage 3 天花板確認（2026-03-19）：turning-consistency 收斂於 ≈ 0.535，無法突破

**動機**：前次 k×ft 12-run 網格均 Level 2 後，系統性探索 Stage 3 不通過的根本原因。

**分析方法**：直接計算 turning-consistency score（`classify_cycle_level` 內部 Stage 3 判定值，閾值 η=0.55），跨所有探索軸比較。

### 結果總覽

| 探索方向 | 條件 | Stage 3 score | turn_strength | 結論 |
|---------|------|-------------|--------------|------|
| baseline | a=0.8,σ=0.06,r=8000 | **0.5335** | 0.000632 | 參考 |
| payoff 非對稱 | a=0.7 | 0.5346 | 0.000633 | 微幅上升 |
| payoff 非對稱 | a=0.6 | 0.5356 | 0.000633 | plateau |
| payoff 非對稱 | a=0.5 | 0.5356 | 0.000633 | **plateau 確認** |
| count_cycle | count_cycle mode | 0.5346 | 0.000633 | 無改善 |
| 高 σ | σ=0.30 | 0.5122 | 0.000637 | 反效果 |
| 高 σ | σ=0.40 | 0.5248 | 0.000640 | 反效果 |
| 高 σ | σ=0.50 | 0.5218 | 0.000640 | 反效果 |
| 高 k | k=0.08–0.16 | 0.463–0.481 | 0.000637 | 明顯反效果 |

**`turn_strength` 全程恆定** ≈ 0.000632–0.000642。Score ceiling ≈ **0.536**。

**結論（架構天花板）**：

1. Stage 3 turning-consistency 被 **現有架構限制在 ≈ 0.535**，低於閾值 0.55
2. 任何已知可調參數（σ、a/b、k、ft、payoff mode）均無法突破此上限
3. 需要**質性不同的反饋機制**才能達到 Level 3：
   - Adaptive rule-mutation（dungeon 規則隨策略分佈動態調整）
   - 多輪歷史狀態整合（跨回合 memory）
   - 子族群不對稱耦合（分組 dynamics）

**診斷指令（可重現）**：
```python
# 直接計算 Stage 3 score（需要 venv 啟動）
from analysis.cycle_metrics import classify_cycle_level, assess_stage1_amplitude
rows = read_csv('outputs/longrun_ss0p06_r8000_seed42.csv')
burn_in = max(0, len(rows) // 3)
tail = min(1000, len(rows))
r = classify_cycle_level(series, burn_in=burn_in, tail=tail,
        amplitude_threshold=0.02, corr_threshold=0.09, eta=0.55,
        stage3_method='turning', phase_smoothing=1)
print(r.stage3.score)  # → 0.5335
```

---

## Adaptive payoff matrix 實驗（2026-03-19）：v1+v2 全部失效——gate 拓樸支配 orbit shape

**v1（乘法）**：`a_new = a_base × (1 − 0.15×s×(p_agg−1/3))`，s∈{0.15,0.25,0.35}，interval=500
→ a 變化 <0.3%，3/3 條件 score=0.5335（與 baseline 字元級相同）

**v2（加法+clip）**：`a_new = clamp(a_base + s×(p_agg−0.30), 0.5, 1.2)`，s∈{0.3,0.6,1.2}，interval∈{300,500}
→ s=0.6 實際 a 波動 2–7%（t=299: a→0.746），但 6/6 條件 score=0.5335，ts=0.000632

**根本原因**：orbit shape 由 event gate 機制決定，payoff 矩陣 5–7% 擾動被 attractor 完全吸收。

**五路徑失效彙總（截至 2026-03-19）**：

| 路徑 | 最佳 score | gap | 根本失效原因 |
|-----|-----------|-----|------------|
| EMA risk_ma | 0.5280 | 0.022 | 全局放大 risk |
| Stress-dependent decay | 0.5349 | 0.015 | 只改 gate 頻率 |
| Adaptive ft | 0.5335 | 0.015 | 雙峰空帶 |
| Adaptive payoff v1 | 0.5335 | 0.015 | delta 太小 |
| Adaptive payoff v2 | 0.5335 | 0.015 | gate 拓樸支配 |

**CLI grid（可重現）**：
```bash
EVENTS_JSON=docs/personality_dungeon_v1/02_event_templates_v1.json
for s in 0.3 0.6 1.2; do
  for interval in 300 500; do
    ./venv/bin/python -m simulation.run_simulation \
      --enable-events --events-json "$EVENTS_JSON" \
      --popularity-mode expected --seed 42 --rounds 8000 --players 300 \
      --payoff-mode matrix_ab --a 0.8 --b 0.9 \
      --selection-strength 0.06 --init-bias 0.12 \
      --event-failure-threshold 0.72 --event-health-penalty 0.10 \
      --adaptive-payoff-strength $s --payoff-update-interval $interval \
      --adaptive-payoff-target 0.30 \
      --out outputs/adaptive_v2_s${s}_i${interval}_s42.csv
  done
done
```

---

## Adaptive rule-mutation 實驗（2026-03-19）：ft 動態調節無效——final_risk 雙峰空帶

**機制**：每 500 rounds 根據 `p_aggressive` 動態調整 ft：
`ft_new = ft_base × (1 + s × (p_agg − 1/3))`（`--adaptive-ft-strength` + `--ft-update-interval`）

**條件**：s∈{0.20, 0.30, 0.40, 0.50}，interval=500，seed=42, rounds=8000

### 結果

4/4 條件 s=0.20–0.50，所有條件的 p_agg 軌跡在 8000 輪中**完全相同**（max diff=0.00e+00）。
score=0.5335, ts=0.000632, sr=0.471——與 stress_decay_c=1.0 基線完全一致。

**偵錯發現**：
- ft_new 確有計算（checkpoint 值 0.686–0.720），但 success_count 在任何一輪都 0 差異
- 決定性根本原因：**final_risk ∈ [0.686, 0.720) 的樣本數為零**，adaptive band 是空的

**新發現（§4.5 補充證據）**：
attractor 將 `final_risk` 分布自組織為**雙峰態**——事件的 final_risk 不是遠低於就是遠高於 ft，
中間帶空無一物。Gate 是**硬二元開關**，非平滑梯度；limit cycle 主動迴避轉換帶。
這直接支持並強化了 §4.5 Gate-Stabilised Limit Cycle 的機制描述。

**結論**：state-level 與 gate-level 的所有內部調節路徑（EMA → nonlinear decay → adaptive ft）均告失效。
下一步需在 **payoff 矩陣層面**（自適應 a/b）或**族群結構層面**（子族群耦合）引入質性不同的反饋。

---

## Stress-dependent decay asymmetry 實驗（2026-03-19）：邊際改善，實質失效

**動機**：EMA 失效後，嘗試「非線性延遲強化」。stress 高時 decay 變慢 → 有效延遲增長 → 理論上可誘發 delay-induced Hopf。
**公式**：`effective_rate = 1 - (1 - base_rate) / (1 + c · stress^β)`（β=2.0）
**實作**：`--stress-decay-c` + `--stress-decay-beta` CLI 參數（backward-compatible，default=0.0/2.0）。

**條件**：β=2.0, c∈{1.0, 1.5, 2.0, 2.5}，seed=42, rounds=8000, players=300

### 結果

| c | Stage 3 score | turn_strength | avg_sr（尾部） | vs baseline |
|---|-------------|--------------|-------------|-------------|
| 0（baseline） | 0.5335 | 0.000632 | 0.482 | 參考 |
| 1.0 | 0.5335 | 0.000632 | 0.471 | = |
| **1.5** | **0.5349** | 0.000629 | 0.465 | **▲ +0.0014** |
| 2.0 | 0.5337 | 0.000634 | 0.461 | ≈ |
| 2.5 | 0.5337 | 0.000634 | 0.461 | ≈ |

**stress 確實在累積**（avg_sr 下降驗證）。score 最大改善 +0.0014（噪音幅度），gap=0.0151。
**turn_strength 全程恆定 → 架構天花板未被打破。**

**根本原因**：stress 累積只影響 gate 觸發頻率（final_risk clipping），不直接改變策略間相位關係。
**結論**：非線性 decay 路徑實質失效。→ 下一步：Adaptive rule-mutation（外部動態閘值）。

---

## EMA risk_ma 實驗（2026-03-19）：9 條件全 Level 2，分數低於 baseline

**動機**：「跨回合 memory」假說——延遲反饋可能誘發相位一致的雙穩態振盪（delay-induced bifurcation）。EMA: `risk_ma = α·risk_ma + (1-α)·risk`，再加算至 `final_risk`。

**實作**：`--risk-ma-alpha` × `--risk-ma-multiplier`，backward-compatible（default=0.0/0.0）。

**條件**：baseline + α∈{0.80,0.85,0.90} × mult∈{0.3,0.5,0.7}，seed=42, rounds=8000, players=300

### 結果

| α | mult | Stage 3 score | turn_strength | 相對 baseline |
|---|------|-------------|--------------|-------------|
| 0.80 | 0.3 | 0.5172 | 0.000629 | ▼ -0.0163 |
| 0.80 | 0.5 | 0.5162 | 0.000632 | ▼ -0.0173 |
| 0.80 | 0.7 | 0.5041 | 0.000644 | ▼ -0.0294 |
| 0.85 | 0.3 | 0.5179 | 0.000642 | ▼ -0.0156 |
| 0.85 | 0.5 | 0.5244 | 0.000641 | ▼ -0.0091 |
| 0.85 | 0.7 | 0.5061 | 0.000632 | ▼ -0.0274 |
| 0.90 | 0.3 | 0.5071 | 0.000624 | ▼ -0.0264 |
| **0.90** | **0.5** | **0.5280** | 0.000639 | **▼ -0.0055** |
| 0.90 | 0.7 | 0.5204 | 0.000642 | ▼ -0.0131 |

**最佳**：α=0.90, mult=0.5 → score=0.5280（仍低於 baseline 0.5335，gap=0.0220）

**結論**：EMA risk 記憶**無法**突破架構天花板，反而引入額外系統性偏差（持續風險放大 → 策略分布更不規則）。`turn_strength` 仍恆定於 0.000624–0.000644。Level 3 閾值 0.55 gap 維持 ≈ **0.022**，EMA 路徑宣告失效。

---

## Level 3 網格探索（2026-03-18）：12/12 Level 2，Level 3 未出現


**動機**：確認 Gate-Stabilised Limit Cycle 機制中，Stage 3（Phase Direction Consistency）在更強 stress-risk coupling 或更嚴格 failure threshold 下是否可觸發。

**設定**：seed=42, rounds=4000, players=300, σ=0.06, bias=0.12, hp=0.10  
**網格**：k ∈ {0.08, 0.12, 0.16} × ft ∈ {0.72, 0.74, 0.78, 0.82} = 12 runs

| k \ ft | 0.72 | 0.74 | 0.78 | 0.82 |
|--------|------|------|------|------|
| **0.08** | L2 (sr=0.470) | L2 (sr=0.450) | L2 (sr=0.457) | L2 (sr=0.450) |
| **0.12** | L2 (sr=0.463) | L2 (sr=0.444) | L2 (sr=0.450) | L2 (sr=0.444) |
| **0.16** | L2 (sr=0.444) | L2 (sr=0.424) | L2 (sr=0.427) | L2 (sr=0.424) |

**結果**：全 12 點 Level 2；`level3_hits = 0`，`top_event_for_level3: []`。

**科學意涵**：
- Gate-Stabilised Limit Cycle 在 stress-risk coupling ±60% 擴動範圍內保持 Level 2
- avg_success_rate 在 0.42–0.47 狹窄帶 → gate 統計仍受吸引子控制
- Phase Direction Consistency（Stage 3）需要 **質性不同的機制**，非量化調整
- 後續方向：k ≥ 0.25、不同 payoff 矩陣、或 adaptive rule-mutation feedback

**新增 CLI 參數**（本次實作）：
```bash
--event-stress-risk-coefficient FLOAT   # 控制 stress→risk coupling（預設 0.10）
```

**可重現命令**：
```bash
EVENTS_JSON=docs/personality_dungeon_v1/02_event_templates_v1.json
for k in 0.08 0.12 0.16; do ktag="${k/./p}"
  for ft in 0.72 0.74 0.78 0.82; do fttag="${ft/./p}"
    ./venv/bin/python -m simulation.run_simulation \
      --enable-events --events-json "$EVENTS_JSON" \
      --popularity-mode expected --seed 42 --rounds 4000 --players 300 \
      --payoff-mode matrix_ab --a 0.8 --b 0.9 \
      --selection-strength 0.06 --init-bias 0.12 \
      --event-failure-threshold "$ft" --event-health-penalty 0.10 \
      --event-stress-risk-coefficient "$k" \
      --out "outputs/level3_k${ktag}_ft${fttag}.csv"
  done
done
```

---



**動機**：先前所有邊界掃描（16 點）均僅使用 seed=42，存在 trajectory lock-in 風險。

**設定**：seeds 1–10 × σ ∈ {0.005, 0.05, 0.20}，fixed: bias=0.12, ft=0.72, hp=0.10, rounds=4000

| σ | seeds | Pr(level ≥ 2) | Δγ 範圍 |
|-------|-------|--------------|---------|
| 0.005 | 1–10 | 10/10 = **1.00** | −7.45×10⁻⁵ to +5.23×10⁻⁶ |
| 0.05  | 1–10 | 10/10 = **1.00** | −7.32×10⁻⁵ to +2.30×10⁻⁶ |
| 0.20  | 1–10 | 10/10 = **1.00** | −7.47×10⁻⁵ to −7.47×10⁻⁷ |

**全域結果：30/30 Level 2，Pr = 1.000**

**初始條件敏感性**（σ=0.05, seed=42, bias ∈ [0.00, 0.30]）：

| bias | level | Δγ |
|------|-------|-----|
| 0.00 | 2 | −2.79×10⁻⁵ |
| 0.05 | 2 | −2.79×10⁻⁵ |
| 0.12 | 2 | −2.94×10⁻⁵ |
| 0.20 | 2 | −2.94×10⁻⁵ |
| 0.30 | 2 | −2.94×10⁻⁵ |

→ 5/5 Level 2；Δγ 收斂到兩個穩定值（low/high bias 各一），吸引子與初始條件無關。

**結論：trajectory lock-in 風險已排除。** Level 2 是系統的結構性質，非特定 seed 的軌跡偶發。
系統行為與 **globally attracting limit cycle** 一致。

---

## 機制分析（2026-03-18）：Gate-Stabilised Limit Cycle

**三個關鍵觀測**（來自 seeds 1–10 × σ ∈ {0.005, 0.05, 0.20}）：

1. **Gate pass rate σ 無關**：avg_success_rate = 0.4815 ± 0.022，三個 σ 值完全相同（小數點後第四位）。
   事件閘觸發由 final_risk vs ft 決定，與 σ 解耦。

2. **時間尺度分離**：
   - 狀態衰減 τ：risk=13.78, noise=9.49, stress=7.82, risk_drift=7.18 (rounds)（由 decay rates 固定）
   - Replicator 更新 1/σ：5–200 rounds（跨 40 倍）
   - 無因次延遲比 σ·τ_risk：0.069–2.76（跨 40 倍），但 Δγ 只變化 37%

3. **Δγ 冪律擬合**：log-log 擬合得 Δγ ≈ −4.09×10⁻⁵ × σ^0.08
   - 指數 m ≈ 0.08 ≈ 0 → σ→0 時 Δγ 仍有限（finite）
   - **非 Hopf bifurcation**：classical Hopf 需要 coupling > threshold，但 σ=0.005 遠低於臨界值仍維持 Level 2

**結論：Gate-stabilised limit cycle**（非 Hopf 分叉）
- Payoff rotation (a=0.8, b=0.9) 提供弱振盪傾向（marginally unstable center）
- Event gate 作為非對稱振幅裁剪器：Aggressive 高風險 → 更多 gate failure → 有效 payoff 下壓 → 限制振幅
- Gate 裁剪 σ 無關 → 即使 σ→0 循環仍存在
- Level 3 需修改 gate 結構（ft/hp/stress-risk hook），不能靠增大 σ 觸發

---



- **原估計甜區**：σ ∈ [0.055, 0.065]（± 8.3% 相對寬度）
- **原始掃描**：σ ∈ [0.02, 0.10] 全區間 Level 2（9/9）
- **擴大掃描確認**：σ ∈ [0.005, 0.20] **全 16 點 Level 2**（40× 範圍）
- **Bifurcation diagram**：無任何 bifurcation point，Δγ 隨 σ 增加單調更負但維持 Level 2
- **結論：true boundary 在實用範圍外**，事件閉環對選擇強度具有例外魯棒性

| σ | level | Δγ |
|---|-------|------|
| 0.005 | 2 | −2.74×10⁻⁵ |
| 0.01  | 2 | −2.82×10⁻⁵ |
| 0.015 | 2 | −2.83×10⁻⁵ |
| 0.02–0.10 | 2 (全) | −2.87×10⁻⁵ to −3.15×10⁻⁵ |
| 0.12  | 2 | −3.30×10⁻⁵ |
| 0.15  | 2 | −3.43×10⁻⁵ |
| 0.18  | 2 | −3.63×10⁻⁵ |
| 0.20  | 2 | −3.76×10⁻⁵ |

**決策樹結果**：
- ✅ σ = 0.005 仍 Level 2 → **「近乎無選擇壓力也能維持 Level 2」已成立**
- ❌ σ = 0.20 無 Level 3 → Level 3 探索需從其他方向（ft/stress hook）入手

### 擴大掃描 CLI（可重現）

```bash
for ss in 0.005 0.01 0.015 0.12 0.15 0.18 0.20; do
  stag="${ss/./p}"
  ./venv/bin/python -m simulation.run_simulation \
    --enable-events --events-json docs/personality_dungeon_v1/02_event_templates_v1.json \
    --popularity-mode expected --seed 42 \
    --rounds 4000 --players 300 \
    --payoff-mode matrix_ab --a 0.8 --b 0.9 \
    --selection-strength "$ss" --init-bias 0.12 \
    --event-failure-threshold 0.72 --event-health-penalty 0.10 \
    --out "outputs/boundary_probe_ss${stag}.csv" &
done
wait
```

---

## 基線鎖定（2026-03-18）

以下參數組合已通過 18-run grid 驗證（全 Level 2，gamma_delta 全負），正式鎖定為研究基準：

| 參數 | 鎖定值 | 說明 |
|------|--------|------|
| `payoff-mode` | `matrix_ab` | |
| `a` / `b` | `0.8` / `0.9` | near-neutral rotation 甜區 |
| `popularity-mode` | `expected` | |
| `event-failure-threshold` | `0.72` | ft 甜區（66% success 不過高）|
| `event-health-penalty` | `0.10` | 適中懲罰係數 |
| `selection-strength` | `0.05–0.07`（甜區）| 0.02–0.10 全可用 |
| `init-bias` | `0.12`（可選，影響已小）| |
| `rounds` | `4000`（基礎）/ `8000`（長序列）| |
| `players` | `300` | |
| `seed` | `42`（單次診斷）/ `0–9`（穩健性驗證）| |

### rounds=8000 長序列 CLI 範例

```bash
# 單次長序列診斷（ss=0.06 甜區）
./venv/bin/python -m simulation.run_simulation \
  --enable-events --events-json docs/personality_dungeon_v1/02_event_templates_v1.json \
  --popularity-mode expected --seed 42 \
  --rounds 8000 --players 300 \
  --payoff-mode matrix_ab --a 0.8 --b 0.9 \
  --selection-strength 0.06 --init-bias 0.12 \
  --event-failure-threshold 0.72 --event-health-penalty 0.10 \
  --out outputs/longrun_ss0p06_r8000_seed42.csv

./venv/bin/python -m analysis.event_provenance_summary \
  --csv outputs/longrun_ss0p06_r8000_seed42.csv \
  --baseline-csv outputs/sweep_expected_baseline_seed42.csv \
  --events-json docs/personality_dungeon_v1/02_event_templates_v1.json \
  --compare-envelope-gamma \
  --out outputs/diag_longrun_ss0p06_r8000.md \
  --out-json outputs/diag_longrun_ss0p06_r8000.json
```

### 多 seed 穩健性驗證 CLI 範例（ss=0.06, rounds=4000, seed 0–9）

```bash
for seed in $(seq 0 9); do
  ./venv/bin/python -m simulation.run_simulation \
    --enable-events --events-json docs/personality_dungeon_v1/02_event_templates_v1.json \
    --popularity-mode expected --seed "$seed" \
    --rounds 4000 --players 300 \
    --payoff-mode matrix_ab --a 0.8 --b 0.9 \
    --selection-strength 0.06 --init-bias 0.12 \
    --event-failure-threshold 0.72 --event-health-penalty 0.10 \
    --out "outputs/robust_ss0p06_seed${seed}.csv"
done
```

---

## 邊界掃描驗證（2026-03-18）：ss=0.055 / 0.065 + rounds=12000

### ss 甜區邊界測試（rounds=4000, players=300, seed=42, ft=0.72, hp=0.10）

| ss    | init-bias | level | gamma_delta   | avg_sr |
|-------|-----------|-------|---------------|--------|
| 0.055 | 0.00      | 2     | -2.83e-05     | 0.470  |
| 0.055 | 0.12      | 2     | -2.97e-05     | 0.470  |
| 0.065 | 0.00      | 2     | -2.85e-05     | 0.470  |
| 0.065 | 0.12      | 2     | -3.00e-05     | 0.470  |

**結論**：ss ∈ {0.055, 0.065} 均達 Level 2，gamma_delta 穩定負值（-2.83 ~ -3.00e-05）。  
init-bias 對結果影響極小（<5%）。甜區寬度 **至少 ss ∈ [0.055, 0.065]**，可宣稱「參數魯棒性良好」。

### rounds=12000 長序列驗證（ss=0.06, bias=0.12, seed=42）

- derived_cycle_level：**2**（穩定維持 vs r=8000）
- gamma_delta：**-7.37e-06**（r=8000 為 -7.73e-06，差異 <5%，收斂穩定）
- avg_success_rate：0.471

**結論**：延長至 12000 輪無退化，Level 2 週期結構在更長時間尺度下保持一致。

> 診斷指令：
> ```bash
> ./venv/bin/python -m analysis.event_provenance_summary \
>   --csv outputs/longrun_ss0p06_r12000_seed42.csv \
>   --baseline-csv outputs/sweep_expected_baseline_seed42.csv \
>   --events-json docs/personality_dungeon_v1/02_event_templates_v1.json \
>   --compare-envelope-gamma \
>   --out-json outputs/diag_longrun_ss0p06_r12000_seed42.json
> ```

---

## 多 seed 穩健性驗證（2026-03-18）：Pr(level≥2) = 10/10

**設定**：ss=0.06, bias=0.12, ft=0.72, hp=0.10, rounds=4000, players=300，seed 0–9

| seed | level | gamma_delta     | avg_sr |
|------|-------|-----------------|--------|
| 0    | 2     | -3.90e-05       | 0.486  |
| 1    | 2     | -6.67e-05       | 0.490  |
| 2    | 2     | -4.61e-05       | 0.490  |
| 3    | 2     | +2.18e-06       | 0.447  |
| 4    | 2     | -1.10e-05       | 0.460  |
| 5    | 2     | -3.96e-05       | 0.480  |
| 6    | 2     | -3.49e-05       | 0.477  |
| 7    | 2     | -5.90e-05       | 0.510  |
| 8    | 2     | -5.74e-05       | 0.493  |
| 9    | 2     | -5.73e-06       | 0.454  |

**結論**：**Pr(level ≥ 2) = 10/10 = 1.0 ✓**（門檻 0.7）→ 參數鎖定通過，可進入 paper Results 階段。
- seed 3 gamma_delta 為正（+2.18e-06，接近中性）但 Level 仍為 2，顯示週期結構穩定。
- avg_sr 範圍 0.447–0.510，均在合理懲罰區間。

---

## rounds=8000 長序列驗證（2026-03-18）

**設定**：ss=0.06, bias=0.12, ft=0.72, hp=0.10, seed=42

- derived_cycle_level：**2**（穩定維持）
- gamma_delta：-7.73e-06（較短序列更保守，但仍負值）
- event_gamma：-1.47e-06；baseline_gamma：+6.26e-06
- avg_success_rate：0.471

**Per-Action Success Rate（top 5）**：

| action            | success_rate | count   |
|-------------------|-------------|---------|
| observe           | 0.981       | 231,644 |
| steady_breathing  | 0.977       | 229,321 |
| study             | 0.774       | 286,558 |
| inspect           | 0.676       | 328,605 |
| safe_route        | 0.614       | 372,372 |

→ 低風險 actions（observe/steady_breathing）貢獻高成功率；study/inspect/safe_route 為主力 action，成功率合理（懲罰結構有效）。

---

## 里程碑結案（2026-03-18）

- 事件閉環成功驅動 Level 2 結構性循環（全 seed 出現週期振盪）
- 多 seed 穩健性 100% 通過（Pr(level ≥ 2) = 10/10）
- 基線參數已鎖定，可作為 v1 正式基準，產物：[outputs/final_report.md](../outputs/final_report.md)
- 下一階段選項（按優先序）：
  1. **Level 3 探索**：導入 `final_risk += 0.08 * stress` hook
  2. **economy 接軌**：sample_quality / state_tags 對接 loot / adaptive dungeon
  3. **paper 撰寫**：使用附錄 B skeleton，補圖表後即可送審

> **凍結版本**：Personality Dungeon v1 事件閉環基線（2026-03-18）  
> 所有產物已驗證，參數與程式碼不再變更，除非進入 Level 3 或 economy 階段。

---

## Level 2 全面確認：w_→p_ 序列回退修正

**背景**：`_derive_cycle_level` 過去無條件使用 `w_` 序列；在 `selection_strength` 較小時，
`w_` 尾段振幅僅約 `ss×0.21`（ss=0.10 → 0.021），無法通過 Stage 1（門檻 0.02），
導致整個 ss 掃描結果全為 Level 0，掩蓋真實週期訊號。

**修正**：`analysis/event_provenance_summary.py::_derive_cycle_level()`—若 w_ 未過 Stage 1，
改用 p_ 序列（p_ 振幅穩定約 0.23，與 ss 無關）。

**驗證結果**（ft=0.72, hp=0.10, seed=42, rounds=4000, players=300）：

| ss   | bias | level | gamma_delta     | avg_sr |
|------|------|-------|-----------------|--------|
| 0.02 | 0.00 | 2     | -2.73e-05       | 0.470  |
| 0.02 | 0.12 | 2     | -2.87e-05       | 0.470  |
| 0.05 | 0.00 | 2     | -2.79e-05       | 0.470  |
| 0.05 | 0.12 | 2     | -2.94e-05       | 0.470  |
| 0.10 | 0.00 | 2     | -3.00e-05       | 0.470  |
| 0.10 | 0.12 | 2     | -3.15e-05       | 0.470  |

→ **全 18 個配置均 Level 2**（週期震盪）。

**關鍵發現**：
- p_ 振幅 ≈ 0.23（與 ss 無關）；w_ 尾段振幅 ≈ ss×0.21
- gamma_delta 全為負值（-2.7e-05 ~ -3.1e-05）：引入事件後振幅增長率高於 baseline，符合預期
- avg_success_rate ≈ 0.470（事件成功率約 47%）

---

## [2025-07-xx] ft=0.72 甜區確認：gamma_delta 轉負，成功率 47%

**背景**：failure_threshold 過低（≤0.5）會導致幾乎所有 action 被 hard gate 擋掉（risk ≥ threshold），
造成 flee_randomly 成功率 0.03%，risk 飽和在 0.64。
將所有 failure_threshold 提高至 0.65–0.85 範圍（global default 0.72）後問題解決。

**3×3 grid 結果**（ft × health_penalty；固定 ss=0.05, bias=0.12）：

- ft=0.72, hp=0.10 → **gamma_delta = -2.93e-05**（最佳）
- ft=0.72 為甜區：safe_route 成功率 61%，inspect 66%，study 76%
- avg_final_risk ≈ 0.52，avg_success ≈ 47%

**狀態衰減設定**（`state_policy.decay_rates`）：
- stress×0.88, noise×0.90, risk_drift×0.87, risk×0.93, health+=0.02/round

---

## 診斷標準指令

```bash
# 基本事件模擬（ft=0.72, hp=0.10）
./venv/bin/python -m simulation.run_simulation \
  --enable-events --popularity-mode expected \
  --seed 42 --rounds 4000 --players 300 \
  --payoff-mode matrix_ab --a 0.8 --b 0.9 \
  --init-bias 0.12 \
  --event-failure-threshold 0.72 \
  --event-health-penalty 0.10 \
  --out outputs/diag.csv

# 跑 provenance summary（含 gamma_delta 與 avg_success_rate）
./venv/bin/python -m analysis.event_provenance_summary \
  --csv outputs/diag.csv \
  --baseline-csv outputs/sweep_expected_baseline_seed42.csv \
  --events-json docs/personality_dungeon_v1/02_event_templates_v1.json \
  --compare-envelope-gamma \
  --out-json outputs/diag_summary.json
```

---

## 目前研究基線（最終確認版）

- 控制組 baseline：`outputs/sweep_expected_baseline_seed42.csv`
  - `baseline_gamma = +6.26e-06`, `n_peaks = 1337`
- 事件閉環甜區：ft=0.72, hp=0.10, ss=0.05–0.07, bias=0.00–0.12
- 達到 Level 2 的最低 ss：0.02（p_ series fallback 後）
- **Pr(level ≥ 2) = 10/10 = 1.0**（seed 0–9, ss=0.06, r=4000）✓
- rounds=8000 長序列：Level 2 穩定維持，gamma_delta=-7.73e-06
- 主力 action 成功率：observe 0.981 / steady_breathing 0.977 / study 0.774 / inspect 0.676 / safe_route 0.614
- Level 3（結構性循環）尚未達到，為下一階段目標
- 下一里程碑選項：
  - 邊界掃描 ss=0.04–0.09 step 0.005（bifurcation diagram）
  - 導入 `final_risk += 0.08 * stress` hook（推向 Level 3）
  - 接入 economy_rules（sample_quality / state_tags）
