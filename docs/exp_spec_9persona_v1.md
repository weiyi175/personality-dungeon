# 自動化實驗規劃與執行規格書 v1.0
# Project: Eco-Evolutionary Dynamics（9 人格模型）

**撰寫日期**：2026-05-09  
**版本**：v1.0  
**角色**：計算實驗室首席科學家  
**遵循 SDD 原則**：先 Spec 後程式碼；所有命令使用 `./venv/bin/python`；不破壞既有 CSV schema。

---

## 壹、背景與約束

### 1.1 9 人格特質架構

本規格書的「9 人格模型」指 `players/base_player.py` 與 `players/rl_player.py` 實作的 **9-trait Enneagram 設計**：

| 子群組 | 特質 | Latent Signal |
|--------|------|---------------|
| 擴張組 Drivers | `impulsiveness`, `assertiveness`, `optimism` | `z_expanding` |
| 防禦組 Stabilizers | `risk_aversion`, `suspicion`, `endurance` | `z_contracting` |
| 擾動組 Explorers | `randomness`, `stability_seeking`, `curiosity` | `z_exploring` |

各 Latent Signal 定義：
$$z_{expanding} = \frac{impulsiveness + assertiveness + optimism}{3}$$
$$z_{contracting} = \frac{risk\_aversion + suspicion + endurance}{3}$$
$$z_{exploring} = \frac{randomness + stability\_seeking + curiosity}{3}$$

### 1.2 實驗主線模組

| 系列 | 模組路徑 | 支援 personality_mode |
|------|----------|----------------------|
| 基線 / Phase A | `simulation.personality_rl_runtime` | ✓ (`none` / `random`) |
| W3.1 Stackelberg | `simulation.w3_stackelberg` | ✗（run_simulation 基底） |
| W3.2 Hysteretic | `simulation.w3_policy` | ✗（run_simulation 基底） |
| W3.3 Pulse | `simulation.w3_pulse` | ✗（run_simulation 基底） |
| B2/B3 Gate60 | `simulation.b1_async_dispatch_gate` | 部分（透過 cfg） |
| H-series | `simulation.run_simulation` | ✗ |
| D-series | `simulation.run_simulation` | ✗ |

**架構限制說明**：W3 系列與 H/D 系列採 `run_simulation` 基底，目前尚未整合 `personality_mode`。Phase B（Canonical Re-run）的「9 人格相容性」驗證採**並排比較**策略：先用既有模組跑一次確認 closure 結論，再用 `personality_rl_runtime` + `personality_mode=none/random` 做等效橋接，確認機制開關邏輯未被 personality 注入所改變。

### 1.3 通用固定參數（全實驗共用）

```
n_players        = 300
n_rounds         = 12000   (W3/H/D 系列: 3000)
burn_in          = 4000    (W3/H/D 系列: 1000)
tail             = 4000    (W3/H/D 系列: 1000)
amplitude_threshold = 0.02
corr_threshold   = 0.09
eta              = 0.55
stage3_method    = turning
events_json      = docs/personality_dungeon_v1/02_event_templates_smoke_v1.json
dispatch_target_rate = 0.08
fairness_window  = 2000
fairness_tolerance = 0.50
venv             = ./venv/bin/python
```

---

## 貳、實驗階段與分流規則

### 階段 A：基線橋接（Baseline Bridging）

**目標**：建立新舊人格模式的相容性橋接基準。

#### A1：既有基線（personality_mode=none）

- 理論等價：無人格差異，所有玩家 z-signals 皆為 0
- 輸出目錄：`outputs/exp_A1_baseline_none/`
- 種子：`42,44,45,67,73,90`（6-seed smoke）

#### A2：9 人格基線（personality_mode=random）

- 理論定義：每位玩家獨立從 Uniform(-1,1) 抽取 9 個人格特質
- 輸出目錄：`outputs/exp_A2_baseline_9persona/`
- 種子：`42,44,45,67,73,90`（與 A1 完全對齊）

#### A1/A2 停損判定邏輯

```
偏移率 = |mean_s3(A2) - mean_s3(A1)| / max(mean_s3(A1), 1e-6)

if 偏移率 > 0.15 AND 無法合理解釋:
    STATUS = HALT
    → 回檢 Schema；不得推進 B/C/D 階段
else if 偏移率 > 0.10:
    STATUS = WARN
    → 記錄說明；可進行 B/C/D，但需附加橋接修正條件
else:
    STATUS = PASS
    → 正常推進
```

額外監控：`boundary_hit_rate(A2) / boundary_hit_rate(A1)` 若 > 2.0，亦視為 WARN。

---

### 階段 B：代表點重跑（Canonical Re-run）

**目標**：確認 W-series 與 B-series 的 closure 結論在 9 人格架構下仍然穩健。

**通用策略**：先跑 6 seeds（Smoke）；若出現 uplift（mean_s3 提升 ≥ 0.015）或符號翻轉（env_gamma 正負號改變），擴增至 12 seeds。

| 實驗 ID | 歷史結論 | 代表點 | 模組 |
|---------|----------|--------|------|
| B1 | W3.1 close_w3_1 | `commit_push` cell | `simulation.w3_stackelberg` |
| B2 | W3.2 close_w3_2 | `w3_policy_crossguard` | `simulation.w3_policy` |
| B3 | W3.3 close_w3_3 | `w3_pulse_commitpush` | `simulation.w3_pulse` |
| B4 | B2-series NO-GO | `multiplicative, cap=0.25` | `simulation.b1_async_dispatch_gate` |
| B5 | B3-series NO-GO | `impact_mode=spread, horizon=5, decay=0.70` | `simulation.b1_async_dispatch_gate` |

**B1/B2/B3 特別說明**：由於 W3 模組採 `run_simulation` 基底，9 人格橋接方式為：
1. 先執行原始 W3 模組確認 close 結論可重現
2. 再透過 `personality_rl_runtime` 以等效 payoff 幾何重跑，比較 `mean_s3` / `mean_env_gamma`

**Pass/Fail 標準**：
- `STABLE`：新結論與歷史 closure 一致（Level3 seed count 相同，mean_s3 偏移 < 10%）
- `UPLIFT_DETECTED`：9 人格版本出現 Level3 seed 而舊版沒有 → 擴增至 12 seeds，進入進階分析
- `REGRESSION`：9 人格版本比舊版更差（mean_s3 下降 > 15%）→ 記為 WARN，回檢人格注入邏輯

---

### 階段 C：H-series No-op 驗證

**目標**：驗證 H1/H2/H3 機制開關的退化等價性在 9 人格框架下仍然成立。

**原則**：no-op 結果必須與 A1 基線誤差 < 5%（mean_s3 相對誤差）。若 no-op 退化失敗，禁止推進任何 active 線。

| 實驗 ID | 機制 | No-op 設定 | Active 設定 |
|---------|------|-----------|------------|
| C1 | H1 Memory | `--memory-kernel 1` | `--memory-kernel 3` |
| C2 | H2 Threshold | `--payoff-mode matrix_ab` | `--payoff-mode threshold_ab --threshold-theta 0.40 --threshold-a-hi 1.4 --threshold-b-hi 0.9` |
| C3 | H3 Hetero | `--fixed-subgroup-share 0.0` | `--fixed-subgroup-share 0.10 --fixed-subgroup-weights 0.8,0.8,1.4` |

**C1 不變條件**（依 SDD 3.1）：`memory_kernel=1` 必須與 lag-1 payoff 完全等價。

**C2 不變條件**：`threshold_ab(a_hi=a, b_hi=b)` 在 initial state 低於 theta 時，行為必須與純 `matrix_ab(a,b)` 等價。

**C3 不變條件**：`fixed_subgroup_share=0` 必須與無 subgroup topology 的 sampled 路徑等價。

---

### 階段 D：記憶機制專項（Memory & Lag Analysis）

**目標**：量化 `memory_kernel × payoff_lag` 2×2 交互對 9 人格結構的影響。

| 實驗 ID | memory_kernel | payoff_lag | 備註 |
|---------|--------------|-----------|------|
| D1 | 1 | 0 | 即時 payoff，無記憶窗 |
| D2 | 1 | 1 | **參考基準**（現行 SDD 預設） |
| D3 | 3 | 0 | 3-round 記憶窗 + 即時 payoff |
| D4 | 3 | 1 | 3-round 記憶窗 + 1-lag payoff |

注意：`payoff_lag` 只對 `--evolution-mode mean_field` 有效（SDD 4.1.1）；sampled 路徑不使用此參數，D 系列應以 `mean_field` 模式執行以確保 lag 語意清晰。

**D 系列成功標準**：
- D2 應匹配既有 deterministic gate 結論（P(Level3)=1.0 at `a=1.0, b=0.9, cross=0.20`）
- 其餘 cells 相對 D2 的 `mean_s3` 偏移超過 20% 時，記為顯著差異並記入研發日誌

---

## 參、欄位對齊規格（Data Schema）

所有輸出 CSV / JSON 必須完整包含以下欄位，**嚴禁漏掉任意一項**：

### 時序欄位（Timeseries CSV）

| 欄位 | 類型 | 語意 |
|------|------|------|
| `round` | int | 完成第 t 輪 step 後（1-indexed） |
| `avg_reward` | float | 該輪族群平均即時 reward |
| `avg_utility` | float | 到該輪為止累積效用 |
| `p_aggressive` | float | 第 t 輪策略比例（sampled） |
| `p_defensive` | float | 第 t 輪策略比例（sampled） |
| `p_balanced` | float | 第 t 輪策略比例（sampled） |
| `w_aggressive` | float | 第 t 輪更新後下一輪抽樣權重（族群平均） |
| `w_defensive` | float | 同上 |
| `w_balanced` | float | 同上 |

**w_* 語意**（SDD 3.3）：$w_i^{csv}(t) = \frac{1}{N}\sum_{p=1}^{N} w_i^{(p)}(t+1)$

### 元數據（Provenance JSON）

| 欄位 | 類型 | 階段覆蓋 |
|------|------|---------|
| `seeds` | list[int] | A/B/C/D |
| `n_players` | int | A/B/C/D |
| `personality_mode` | str | A（必填）；B/C/D（標記 "none"） |
| `event_dispatch_mode` | str | A/B4/B5 |
| `fairness_window` | int | A/B4/B5 |
| `fairness_tolerance` | float | A/B4/B5 |
| `world_feedback_mode` | str | A |
| `memory_kernel` | int | C1/D1-D4 |
| `payoff_lag` | int | D1-D4 |
| `threshold_params` | dict | C2/B2 |
| `fixed_subgroup_share` | float | C3 |
| `fixed_subgroup_weights` | list[float] | C3 |

### 結果彙總（Summary JSON）

| 欄位 | 類型 | 來源 |
|------|------|------|
| `fairness_fail_total` | int | B4/B5 gate |
| `overall_pass` | bool | 所有實驗 |
| `mean_s3` | float | `analysis.cycle_metrics` |
| `mean_turn` | float | `analysis.cycle_metrics` |
| `boundary_hit_rate` | float | `provenance.json` |
| `level_counts` | dict | `analysis.cycle_metrics` |
| `mean_env_gamma` | float | `analysis.decay_rate` |
| `p_level3` | float | 計算自 level_counts |

---

## 肆、因果鍊完整性檢查清單

### 4.1 事件鍊（Event Chain）

```
Event Trigger → Player Set Selection → Reward Offset
```

| 檢查點 | 驗證方式 | 判定欄位 |
|--------|---------|---------|
| E.1 Trigger 啟動 | `event_trigger_guard_check_count > 0` when `trigger_mode=entropy_guard` | `provenance.json` |
| E.2 Neutralize 邏輯 | `event_neutrality_max_abs_mean <= event_neutralize_eps` | `provenance.json` |
| E.3 公平性窗口 | `fairness_fail_count == 0` for each seed | `summary.json` |
| E.4 dispatch_mean_affected_ratio | 在 [target_rate × 0.5, target_rate × 1.5] 內 | `provenance.json` |

**自動化檢查腳本**（建議在每批實驗後執行）：
```bash
./venv/bin/python - <<'PY'
import json, glob, sys
errs = []
for path in glob.glob('outputs/exp_*/*/provenance.json'):
    d = json.load(open(path))
    if d.get('event_neutralize_payoff') and not d.get('event_neutrality_pass', True):
        errs.append(f"[FAIL] Neutralize chain broken: {path}")
    if d.get('fairness_fail_count', 0) > 0:
        errs.append(f"[WARN] Fairness fail: {path}")
if errs:
    print('\n'.join(errs)); sys.exit(1)
print("[OK] Event chain integrity check passed.")
PY
```

### 4.2 環境鍊（World Feedback Chain）

```
World State Update → Mode Filter → Payoff Modification
```

| 檢查點 | 模式 | 驗證欄位 |
|--------|------|---------|
| W.1 read_only 不洩漏 | `world_feedback_mode=read_only` | `readonly_leak_score <= 1e-6` |
| W.2 difficulty 隔離 | `world_feedback_mode=difficulty_only` | `payoff_static_pass=true` |
| W.3 off 退化 | `world_feedback_mode=off` | World channels 不影響動力學 |

### 4.3 演化鍊（Evolution Chain）

```
Reward Aggregation → Replicator Normalization → Next Round Sampling Weight
```

| 檢查點 | 不變條件 | 驗證方式 |
|--------|---------|---------|
| R.1 權重正規化 | `mean(w_*) = 1`（誤差 ≤ 1e-9） | 對每輪 CSV 行檢查 |
| R.2 權重有限性 | 所有 `w_*` 非 NaN/Inf | CSV 解析後 assert |
| R.3 p_* 總和 | `p_aggressive + p_defensive + p_balanced = 1`（n>0 時） | 同上 |
| R.4 No-op 退化 | no-op 條件下 w_* 路徑與 A1 基線逐輪誤差 < 1e-6 | 差分比較 |

**R.4 自動比對**：
```bash
./venv/bin/python -m analysis.cycle_metrics \
  --csv outputs/exp_C1_noop/timeseries.csv \
  --ref outputs/exp_A1_baseline_none/timeseries.csv \
  --check-noop --tol 1e-6
```

---

## 伍、執行批次表

| ID | 階段 | 說明 | 模組 | Smoke Seeds | Rounds | personality_mode | 停損條件 |
|----|------|------|------|------------|--------|-----------------|---------|
| **A1** | A | 既有基線（無人格） | personality_rl_runtime | 42,44,45,67,73,90 | 12000 | none | — |
| **A2** | A | 9 人格基線 | personality_rl_runtime | 42,44,45,67,73,90 | 12000 | random | A1偏移>15% → HALT |
| **B1** | B | W3.1 commit_push 代表點 | w3_stackelberg | 45,47,49,51,53,55 | 3000 | n/a | — |
| **B2** | B | W3.2 crossguard 代表點 | w3_policy | 45,47,49,51,53,55 | 3000 | n/a | — |
| **B3** | B | W3.3 pulse_commitpush 代表點 | w3_pulse | 45,47,49,51,53,55 | 3000 | n/a | — |
| **B4** | B | B2-series multiplicative 代表點 | b1_async_dispatch_gate | 42,44,45,67,73,90 | 12000 | none | new_L1>0 → NO-GO |
| **B5** | B | B3-series impact_spread 代表點 | b1_async_dispatch_gate | 42,44,45,67,73,90 | 12000 | none | new_L1>0 → NO-GO |
| **C1-noop** | C | H1 Memory no-op (k=1) | run_simulation | 45,47,49 | 3000 | n/a | 退化失敗 → 阻斷 C1-active |
| **C1-active** | C | H1 Memory active (k=3) | run_simulation | 45,47,49 | 3000 | n/a | — |
| **C2-noop** | C | H2 Threshold no-op (matrix_ab) | run_simulation | 45,47,49 | 3000 | n/a | 退化失敗 → 阻斷 C2-active |
| **C2-active** | C | H2 Threshold active (theta=0.40) | run_simulation | 45,47,49 | 3000 | n/a | — |
| **C3-noop** | C | H3 Hetero no-op (share=0) | run_simulation | 45,47,49 | 3000 | n/a | 退化失敗 → 阻斷 C3-active |
| **C3-active** | C | H3 Hetero active (share=0.10) | run_simulation | 45,47,49 | 3000 | n/a | — |
| **D1** | D | m=1, lag=0 | run_simulation | 45,47,49,51,53,55 | 3000 | n/a | — |
| **D2** | D | m=1, lag=1（參考基準） | run_simulation | 45,47,49,51,53,55 | 3000 | n/a | — |
| **D3** | D | m=3, lag=0 | run_simulation | 45,47,49,51,53,55 | 3000 | n/a | — |
| **D4** | D | m=3, lag=1 | run_simulation | 45,47,49,51,53,55 | 3000 | n/a | — |

**擴增規則**（適用 B1-B5）：  
若 Smoke 結果出現 `level3_seed_count > 0`（B1-B3）或 uplift ≥ 0.015（B4-B5），自動擴增 seeds 至 12 個（加 `51,53,55,91,93,95`）。

---

## 陸、終端機指令集

> 所有指令均以 `./venv/bin/python` 執行，確保環境一致性（SDD Hard Rule §3）。  
> 輸出目錄若不存在，Python 模組會自動建立。

### 階段 A：基線橋接

```bash
# A1：既有基線（無人格）
./venv/bin/python -m simulation.personality_rl_runtime \
  --seeds 42,44,45,67,73,90 \
  --n-players 300 \
  --n-rounds 12000 \
  --personality-mode none \
  --event-dispatch-mode async_poisson \
  --event-dispatch-target-rate 0.08 \
  --event-dispatch-fairness-window 2000 \
  --event-dispatch-fairness-tolerance 0.50 \
  --events-json docs/personality_dungeon_v1/02_event_templates_smoke_v1.json \
  --world-feedback-mode adaptive_world \
  --lambda-world 0.04 \
  --world-update-interval 200 \
  --out-dir outputs/exp_A1_baseline_none

# A2：9 人格基線
./venv/bin/python -m simulation.personality_rl_runtime \
  --seeds 42,44,45,67,73,90 \
  --n-players 300 \
  --n-rounds 12000 \
  --personality-mode random \
  --event-dispatch-mode async_poisson \
  --event-dispatch-target-rate 0.08 \
  --event-dispatch-fairness-window 2000 \
  --event-dispatch-fairness-tolerance 0.50 \
  --events-json docs/personality_dungeon_v1/02_event_templates_smoke_v1.json \
  --world-feedback-mode adaptive_world \
  --lambda-world 0.04 \
  --world-update-interval 200 \
  --out-dir outputs/exp_A2_baseline_9persona
```

**A1/A2 橋接比較（自動化）**：
```bash
./venv/bin/python - <<'PY'
import json, pathlib, sys
def load(out_dir):
    rows = []
    for d in pathlib.Path(out_dir).glob('seed_*'):
        prov = json.loads((d/'provenance.json').read_text())
        rows.append(prov)
    return rows

a1 = load('outputs/exp_A1_baseline_none')
a2 = load('outputs/exp_A2_baseline_9persona')
s3_a1 = sum(r.get('mean_stage3_score',0) for r in a1) / max(len(a1),1)
s3_a2 = sum(r.get('mean_stage3_score',0) for r in a2) / max(len(a2),1)
drift = abs(s3_a2 - s3_a1) / max(s3_a1, 1e-6)
status = 'HALT' if drift > 0.15 else ('WARN' if drift > 0.10 else 'PASS')
print(f"A1 mean_s3={s3_a1:.6f}  A2 mean_s3={s3_a2:.6f}")
print(f"Drift={drift*100:.1f}%  STATUS={status}")
if status == 'HALT':
    sys.exit(1)
PY
```

---

### 階段 B：代表點重跑

```bash
# B1：W3.1 commit_push 代表點
./venv/bin/python -m simulation.w3_stackelberg \
  --conditions control_stackelberg,w3_commit_push \
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
  --decision-md outputs/exp_B1_w31_commit_decision.md

# B2：W3.2 crossguard 代表點
./venv/bin/python -m simulation.w3_policy \
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
  --decision-md outputs/exp_B2_w32_crossguard_decision.md

# B3：W3.3 pulse_commitpush 代表點
./venv/bin/python -m simulation.w3_pulse \
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
  --decision-md outputs/exp_B3_w33_pulse_decision.md

# B4：B2-series multiplicative modulation 代表點（gate smoke only）
./venv/bin/python -m simulation.b1_async_dispatch_gate \
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
  --gate-out-json outputs/exp_B4_b2series_gate_summary.json

# B5：B3-series impact spreading 代表點（gate smoke only）
./venv/bin/python -m simulation.b1_async_dispatch_gate \
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
  --gate-out-json outputs/exp_B5_b3series_gate_summary.json
```

**注意**：B4/B5 執行前須先確認 `outputs/pers_cal_baseline_gate60_summary.json` 存在。如果不存在：
```bash
./venv/bin/python -m simulation.pers_cal_baseline_gate60 \
  --seeds 42..101 \
  --out-json outputs/pers_cal_baseline_gate60_summary.json
```

---

### 階段 C：H-series No-op 驗證

```bash
# C1-noop：H1 Memory no-op（memory_kernel=1）
./venv/bin/python -m simulation.run_simulation \
  --players 300 --rounds 3000 --seed 45 \
  --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
  --evolution-mode mean_field --payoff-lag 1 \
  --memory-kernel 1 \
  --selection-strength 0.06 --init-bias 0.12 \
  --out outputs/exp_C1_noop/seed_45.csv
# （重複 seed 47, 49）

# C1-active：H1 Memory active（memory_kernel=3）
./venv/bin/python -m simulation.run_simulation \
  --players 300 --rounds 3000 --seed 45 \
  --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
  --evolution-mode mean_field --payoff-lag 1 \
  --memory-kernel 3 \
  --selection-strength 0.06 --init-bias 0.12 \
  --out outputs/exp_C1_active/seed_45.csv

# C2-noop：H2 Threshold no-op（pure matrix_ab）
./venv/bin/python -m simulation.run_simulation \
  --players 300 --rounds 3000 --seed 45 \
  --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
  --evolution-mode mean_field --payoff-lag 1 \
  --memory-kernel 3 \
  --selection-strength 0.06 --init-bias 0.12 \
  --out outputs/exp_C2_noop/seed_45.csv

# C2-active：H2 Threshold active（threshold_ab）
./venv/bin/python -m simulation.run_simulation \
  --players 300 --rounds 3000 --seed 45 \
  --payoff-mode threshold_ab --a 1.0 --b 0.9 \
  --threshold-theta 0.40 --threshold-a-hi 1.4 --threshold-b-hi 0.9 \
  --evolution-mode mean_field --payoff-lag 1 \
  --memory-kernel 3 \
  --selection-strength 0.06 --init-bias 0.12 \
  --out outputs/exp_C2_active/seed_45.csv

# C3-noop：H3 Hetero no-op（fixed_subgroup_share=0）
./venv/bin/python -m simulation.run_simulation \
  --players 300 --rounds 3000 --seed 45 \
  --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
  --evolution-mode sampled --memory-kernel 3 \
  --fixed-subgroup-share 0.0 \
  --selection-strength 0.06 --init-bias 0.12 \
  --out outputs/exp_C3_noop/seed_45.csv

# C3-active：H3 Hetero active（fixed_subgroup_share=0.10）
./venv/bin/python -m simulation.run_simulation \
  --players 300 --rounds 3000 --seed 45 \
  --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
  --evolution-mode sampled --memory-kernel 3 \
  --fixed-subgroup-share 0.10 --fixed-subgroup-weights 0.8,0.8,1.4 \
  --selection-strength 0.06 --init-bias 0.12 \
  --out outputs/exp_C3_active/seed_45.csv
```

**C-series No-op 自動驗證**：
```bash
./venv/bin/python - <<'PY'
import csv, sys
def mean_col(path, col):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return sum(float(r[col]) for r in rows[-1000:]) / 1000

for exp_id, noop_path, ref_path in [
    ('C1', 'outputs/exp_C1_noop/seed_45.csv', 'outputs/exp_A1_baseline_none/seed_42/timeseries.csv'),
]:
    try:
        noop_val = mean_col(noop_path, 'p_aggressive')
        ref_val  = mean_col(ref_path,  'p_aggressive')
        err = abs(noop_val - ref_val)
        status = 'PASS' if err < 0.05 else 'FAIL'
        print(f"[{exp_id}] noop={noop_val:.4f} ref={ref_val:.4f} err={err:.4f} {status}")
    except Exception as e:
        print(f"[{exp_id}] ERROR: {e}")
PY
```

---

### 階段 D：記憶機制專項

```bash
# D1：m=1, lag=0
./venv/bin/python -m simulation.run_simulation \
  --players 300 --rounds 3000 \
  --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
  --evolution-mode mean_field --payoff-lag 0 --memory-kernel 1 \
  --selection-strength 0.06 --init-bias 0.12 \
  --popularity-mode expected \
  --out outputs/exp_D1_m1_lag0/timeseries.csv

# D2：m=1, lag=1（參考基準）
./venv/bin/python -m simulation.run_simulation \
  --players 300 --rounds 3000 \
  --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
  --evolution-mode mean_field --payoff-lag 1 --memory-kernel 1 \
  --selection-strength 0.06 --init-bias 0.12 \
  --popularity-mode expected \
  --out outputs/exp_D2_m1_lag1/timeseries.csv

# D3：m=3, lag=0
./venv/bin/python -m simulation.run_simulation \
  --players 300 --rounds 3000 \
  --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
  --evolution-mode mean_field --payoff-lag 0 --memory-kernel 3 \
  --selection-strength 0.06 --init-bias 0.12 \
  --popularity-mode expected \
  --out outputs/exp_D3_m3_lag0/timeseries.csv

# D4：m=3, lag=1
./venv/bin/python -m simulation.run_simulation \
  --players 300 --rounds 3000 \
  --payoff-mode matrix_ab --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
  --evolution-mode mean_field --payoff-lag 1 --memory-kernel 3 \
  --selection-strength 0.06 --init-bias 0.12 \
  --popularity-mode expected \
  --out outputs/exp_D4_m3_lag1/timeseries.csv
```

**D-series 批次評估**：
```bash
./venv/bin/python - <<'PY'
import csv
from analysis.cycle_metrics import classify_cycle_level

cells = {
    'D1': 'outputs/exp_D1_m1_lag0/timeseries.csv',
    'D2': 'outputs/exp_D2_m1_lag1/timeseries.csv',
    'D3': 'outputs/exp_D3_m3_lag0/timeseries.csv',
    'D4': 'outputs/exp_D4_m3_lag1/timeseries.csv',
}
print("ID\tlevel\ts3_score\tturn_strength")
for cid, path in cells.items():
    with open(path) as f:
        rows = list(csv.DictReader(f))
    series = {s: [float(r[f'p_{s}']) for r in rows]
              for s in ['aggressive','defensive','balanced']}
    cyc = classify_cycle_level(series, burn_in=1000, tail=1000,
          amplitude_threshold=0.02, corr_threshold=0.09,
          eta=0.55, stage3_method='turning', min_lag=2, max_lag=500)
    s3 = round(float(cyc.stage3.score), 6) if cyc.stage3 else 0.0
    turn = round(float(cyc.stage3.turn_strength), 4) if cyc.stage3 else 0.0
    print(f"{cid}\t{cyc.level}\t{s3}\t{turn}")
PY
```

---

## 柒、監控看板定義（3 張關鍵比較圖）

### 圖表 1：Level 分布橫向比較熱圖（Heatmap）

**目的**：一眼看出各實驗 ID 的 Level 0-3 分布，識別「有 Level3 seed 出現」的關鍵點。

**資料來源**：各實驗 summary JSON 的 `level_counts`

**圖表規格**：
- 橫軸：Level 等級（0 / 1 / 2 / 3）
- 縱軸：實驗 ID（A1, A2, B1...D4）
- 顏色：seed count（白=0，藍漸層=多）
- 標注：Level3 count ≥ 1 的格子用紅框高亮

**生成命令**：
```bash
./venv/bin/python - <<'PY'
import json, pathlib, matplotlib.pyplot as plt, numpy as np

EXP_IDS = ['A1','A2','B1','B2','B3','B4','B5',
           'C1-noop','C1-active','C2-noop','C2-active',
           'C3-noop','C3-active','D1','D2','D3','D4']

def load_level_counts(exp_id):
    for p in pathlib.Path(f'outputs/exp_{exp_id}').rglob('provenance.json'):
        d = json.load(open(p))
        if 'cycle_level' in d:
            return {int(lv): cnt for lv,cnt in d.get('level_counts',{}).items()}
    return {0:0,1:0,2:0,3:0}

mat = np.zeros((len(EXP_IDS),4), dtype=int)
for i,eid in enumerate(EXP_IDS):
    lc = load_level_counts(eid)
    for lv in range(4): mat[i,lv] = lc.get(lv,0)

fig,ax = plt.subplots(figsize=(7,10))
im = ax.imshow(mat, aspect='auto', cmap='Blues', vmin=0)
ax.set_xticks([0,1,2,3]); ax.set_xticklabels(['L0','L1','L2','L3'])
ax.set_yticks(range(len(EXP_IDS))); ax.set_yticklabels(EXP_IDS)
for i in range(len(EXP_IDS)):
    for j in range(4):
        ax.text(j,i,str(mat[i,j]),ha='center',va='center',fontsize=9)
        if j==3 and mat[i,j]>0:
            ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,fill=False,edgecolor='red',lw=2))
plt.colorbar(im,ax=ax,label='seed count')
ax.set_title('Level Distribution Heatmap — 9-Persona Experiment Suite')
plt.tight_layout()
plt.savefig('outputs/dashboard_fig1_level_heatmap.png', dpi=150)
print("Saved: outputs/dashboard_fig1_level_heatmap.png")
PY
```

---

### 圖表 2：mean_s3 × mean_env_gamma 散點圖（Convergence Plot）

**目的**：評估各實驗在「週期強度」與「包絡穩定性」兩個維度上的分布，識別有望進入 Level3 的候選點。

**資料來源**：各 summary JSON 的 `mean_s3` 與 `mean_env_gamma`

**圖表規格**：
- 橫軸：`mean_stage3_score`（Stage3 判定分數）
- 縱軸：`mean_env_gamma`（包絡衰減率；正值=穩定）
- 每點標注實驗 ID
- 參考線：橫向 `y=0`（穩定邊界）；縱向 `x=0.55`（eta 門檻）
- 顏色：A 系列=藍，B 系列=橙，C 系列=綠，D 系列=紫

**生成命令**：
```bash
./venv/bin/python - <<'PY'
import json, pathlib, matplotlib.pyplot as plt, matplotlib.patches as mpatches

COLOR = {'A':'steelblue','B':'darkorange','C':'forestgreen','D':'purple'}

def load_metrics(exp_id):
    for p in pathlib.Path(f'outputs/exp_{exp_id}').rglob('provenance.json'):
        d = json.load(open(p))
        s3 = d.get('mean_stage3_score', None)
        g = d.get('mean_env_gamma', None)
        if s3 is not None and g is not None:
            return s3, g
    return None, None

EXP_IDS = ['A1','A2','B1','B2','B3','B4','B5',
           'C1-noop','C1-active','C2-noop','C2-active',
           'C3-noop','C3-active','D1','D2','D3','D4']

fig,ax = plt.subplots(figsize=(10,7))
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.axvline(0.55, color='gray', lw=0.8, ls='--', label='eta=0.55')

for eid in EXP_IDS:
    s3, g = load_metrics(eid)
    if s3 is None: continue
    series = eid[0]
    c = COLOR.get(series, 'black')
    ax.scatter(s3, g, color=c, s=80, zorder=3)
    ax.annotate(eid, (s3, g), fontsize=8, ha='left', va='bottom')

patches = [mpatches.Patch(color=v,label=f'{k}-series') for k,v in COLOR.items()]
ax.legend(handles=patches, loc='lower right')
ax.set_xlabel('mean_stage3_score'); ax.set_ylabel('mean_env_gamma')
ax.set_title('Stage3 Score vs. Envelope Gamma — 9-Persona Experiment Suite')
plt.tight_layout()
plt.savefig('outputs/dashboard_fig2_s3_gamma_scatter.png', dpi=150)
print("Saved: outputs/dashboard_fig2_s3_gamma_scatter.png")
PY
```

---

### 圖表 3：H-series No-op 等價性誤差條形圖（Regression Guard Plot）

**目的**：確認 C1/C2/C3 的 no-op 條件與 A1 基線的逐指標偏移量，任何 > 5% 的偏移必須立即標紅。

**資料來源**：各 C-series no-op 實驗的 `mean_s3` / `level_counts` 對比 A1

**圖表規格**：
- 橫軸：驗證指標（mean_s3, p_level3, mean_env_gamma, boundary_hit_rate）
- 縱軸：相對偏移率（|noop - A1| / A1 × 100%）
- 顏色：< 1%=綠，1-5%=黃，> 5%=紅（硬性告警）
- 參考線：`y=5%`（退化等價性邊界）

**生成命令**：
```bash
./venv/bin/python - <<'PY'
import json, pathlib, matplotlib.pyplot as plt, numpy as np

METRICS = ['mean_stage3_score','p_level3','mean_env_gamma','boundary_hit_rate']
NOOPS = ['C1-noop','C2-noop','C3-noop']

def load_summary(exp_id, key):
    for p in pathlib.Path(f'outputs/exp_{exp_id}').rglob('provenance.json'):
        d = json.load(open(p))
        return d.get(key, 0.0)
    return 0.0

ref_vals = {m: load_summary('A1', m) for m in METRICS}

fig, axes = plt.subplots(1, len(NOOPS), figsize=(12,5), sharey=True)
for ax, noop in zip(axes, NOOPS):
    drifts = []
    colors = []
    for m in METRICS:
        noop_val = load_summary(noop, m)
        ref = ref_vals[m]
        drift = abs(noop_val - ref) / max(abs(ref), 1e-6) * 100
        drifts.append(drift)
        colors.append('green' if drift < 1 else ('gold' if drift < 5 else 'red'))
    bars = ax.bar(METRICS, drifts, color=colors)
    ax.axhline(5, color='red', ls='--', lw=1, label='5% boundary')
    ax.set_title(noop); ax.set_ylabel('Relative Drift (%)')
    ax.set_xticklabels(METRICS, rotation=30, ha='right')

axes[0].legend()
plt.suptitle('No-op Degeneracy Regression Guard — C-series vs A1 Baseline', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/dashboard_fig3_noop_regression.png', dpi=150)
print("Saved: outputs/dashboard_fig3_noop_regression.png")
PY
```

---

## 捌、分流決策樹總覽

```
START
│
├─ [Phase A] A1 + A2 橋接
│    ├─ HALT (drift>15%) ──────────────────────────────→ 停止，回檢 Schema
│    ├─ WARN (10-15%)    ──→ 記錄說明後繼續，附橋接修正條件
│    └─ PASS (<10%)      ──→ 正常推進
│
├─ [Phase B] B1/B2/B3 W-series 代表點
│    ├─ STABLE           ──→ 歷史 closure 確認，不需額外分析
│    ├─ UPLIFT_DETECTED  ──→ 擴增至 12 seeds，進入進階分析
│    └─ REGRESSION       ──→ WARN，回檢人格注入邏輯
│
├─ [Phase B] B4/B5 B-series 代表點
│    ├─ new_L1=0         ──→ 機制相容，記為 PASS
│    └─ new_L1>0         ──→ NO-GO，停止該系列，不擴增 seeds
│
├─ [Phase C] C1/C2/C3 No-op 驗證
│    ├─ no-op drift<5%   ──→ 執行 active 對照
│    └─ no-op drift≥5%   ──→ BLOCK：禁止執行 active，回檢機制程式碼
│
└─ [Phase D] D1-D4 2×2 網格
     └─ D2 需匹配歷史 deterministic gate（P(Level3)≥0.9）
          ├─ 通過 ──→ 正常記錄 D1/D3/D4 結果
          └─ 失敗 ──→ D2 退化錯誤，回檢 run_simulation payoff-lag 路徑
```

---

## 玖、SDD 合規聲明

1. **CSV schema 不變**：本規格書所有實驗均沿用現有欄位，不新增主 timeseries 欄位。
2. **架構分層**：`analysis/` 模組只讀 CSV/JSON，不 import `simulation/`。
3. **venv 隔離**：所有指令使用 `./venv/bin/python`。
4. **no-op 退化**：C 系列任意 no-op 條件必須精確退化回 A1 基線（< 5% 誤差）；若失敗視為 bug，不允許推進 active 線。
5. **CSV 語意鎖定**：`w_<strategy>` = 全體玩家平均更新後策略權重；`p_<strategy>` = 該輪實際策略比例。

---

*本規格書由 GitHub Copilot（Claude Sonnet 4.6）依照研究 SDD 框架自動生成。執行前請核對各模組 CLI 參數是否與當前程式碼版本一致（`python -m <module> --help`）。*
