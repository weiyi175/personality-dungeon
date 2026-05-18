# H3 下一步一頁式執行規格（Smoke Only）

日期：2026-05-09  
狀態：**CLOSED — H3-option3-negative（2026-05-09）**

> Option 3 已執行完畢（2 dispatch × 3 cells × 6 seeds = 36 runs）。所有 6 cells 指標完全等值，H3 分支全閉。詳見 `研發日誌.md` 第十一節。

---

## 0. 背景與前提

已知結果：H3 第一輪三格（h3_ctrl / h3_d1 / h3_d1s2）在 `async_poisson` 下為 `H3-first-scout negative`，三格指標等值。

本文件提供兩個「下一步」的一頁式 protocol：

- Option 2：最小擴格（同 dispatch 模式）
- Option 3：跨 dispatch 對照（保持三格結構）

共同硬限制（沿用 B1 smoke lock）：

- Python：`./venv/bin/python`（3.10.12）
- seeds：`42,44,45,67,73,90`
- `n_players=300`, `n_rounds=12000`
- `events_json=docs/personality_dungeon_v1/02_event_templates_smoke_v1.json`
- `event_dispatch_target_rate=0.08`
- `event_dispatch_fairness_window=2000`
- `event_dispatch_fairness_tolerance=0.50`
- 只做 smoke，不開 gate60
- 主要 gate 僅看 `fairness_fail_count=0`

共用分析口徑（固定）：

- `analysis.cycle_metrics.classify_cycle_level`
- `burn_in=2400`, `tail=1000`
- `amplitude_threshold=0.02`, `corr_threshold=0.09`
- `eta=0.55`, `stage3_method=turning`, `phase_smoothing=1`
- `min_lag=2`, `max_lag=500`

---

## 1. Option 2：最小擴格（同 dispatch）

### 1.1 目的

在不改 dispatch 模式（固定 `async_poisson`）下，只做一次最小擴格，檢查第一輪 negative 是否為過窄取樣造成。

### 1.2 變更軸

只允許以下其中一種方案，二選一，不可同時展開：

- O2-A：`delay=2, smooth=1`
- O2-B：`delay=1, smooth=3`

其餘條件固定為 H3 第一輪控制條件。

### 1.3 Cells 設計（固定三格）

- `h3_ctrl`: `delay=0`, `smooth=1`
- `h3_anchor`: `delay=1`, `smooth=1`
- `h3_expand`: 採 O2-A 或 O2-B 的擴格點

### 1.4 輸出命名

- O2-A：`outputs/personality_rl_async_poisson_h3_delay_d2_s1`
- O2-B：`outputs/personality_rl_async_poisson_h3_delay_d1_s3`
- 控制點沿用：`..._h3_delay_ctrl`, `..._h3_delay_d1_s1`

### 1.5 驗收與停損

主 gate：

- 三格都必須 `fairness_fail_count=0`

secondary 判讀（僅監控）：

- `mean_stage3_score`
- `mean_turn_strength`
- `boundary_hit_rate`
- `instability_warning_count`

停損：

- 若 `h3_expand` 未同時優於 `h3_ctrl` 與 `h3_anchor`（至少在 `mean_stage3_score` 或 `mean_turn_strength` 其一有可辨識 uplift），記為 `H3-option2-negative`，立即關閉 Option 2，不再加密網格。

### 1.6 可直接執行命令模板（僅模板）

```bash
./venv/bin/python -m simulation.personality_rl_runtime \
  --seeds 42,44,45,67,73,90 \
  --n-players 300 --n-rounds 12000 \
  --event-dispatch-mode async_poisson \
  --event-dispatch-target-rate 0.08 \
  --event-dispatch-fairness-window 2000 \
  --event-dispatch-fairness-tolerance 0.50 \
  --events-json docs/personality_dungeon_v1/02_event_templates_smoke_v1.json \
  --world-feedback-mode adaptive_world \
  --lambda-world 0.04 \
  --world-update-interval 200 \
  --world-feedback-delay-windows 2 \
  --world-feedback-smooth-windows 1 \
  --out-dir outputs/personality_rl_async_poisson_h3_delay_d2_s1
```

---

## 2. Option 3：跨 dispatch 對照（保持三格結構）

### 2.1 目的

檢查 H3 delayed-feedback 是否受 dispatch topology 影響。保留同一組三格，僅切換 `event_dispatch_mode`。

### 2.2 變更軸

- `event_dispatch_mode ∈ {async_poisson, async_round_robin}`
- H3 三格固定不變：
  - `h3_ctrl`: `delay=0, smooth=1`
  - `h3_d1`: `delay=1, smooth=1`
  - `h3_d1s2`: `delay=1, smooth=2`

### 2.3 Cells 設計（2 x 3）

- `poisson × {ctrl, d1, d1s2}`
- `round_robin × {ctrl, d1, d1s2}`

共 6 cells，全部 smoke-only。

### 2.4 輸出命名

- `outputs/personality_rl_async_poisson_h3_delay_ctrl`
- `outputs/personality_rl_async_poisson_h3_delay_d1_s1`
- `outputs/personality_rl_async_poisson_h3_delay_d1_s2`
- `outputs/personality_rl_async_rr_h3_delay_ctrl`
- `outputs/personality_rl_async_rr_h3_delay_d1_s1`
- `outputs/personality_rl_async_rr_h3_delay_d1_s2`

### 2.5 驗收與停損

主 gate：

- 六格都必須 `fairness_fail_count=0`

secondary 判讀：

- 同一 delay/smooth 下比較 `poisson` 與 `round_robin` 的 `mean_stage3_score` 與 `mean_turn_strength`
- 檢查是否僅在單一 dispatch 出現 delayed uplift

停損：

- 若兩個 dispatch 下三格都呈現等值或同向無 uplift，記為 `H3-option3-negative`，關閉 H3 分支，不再做 dispatch 維度擴展。

### 2.6 可直接執行命令模板（僅模板）

```bash
./venv/bin/python -m simulation.personality_rl_runtime \
  --seeds 42,44,45,67,73,90 \
  --n-players 300 --n-rounds 12000 \
  --event-dispatch-mode async_round_robin \
  --event-dispatch-target-rate 0.08 \
  --event-dispatch-fairness-window 2000 \
  --event-dispatch-fairness-tolerance 0.50 \
  --events-json docs/personality_dungeon_v1/02_event_templates_smoke_v1.json \
  --world-feedback-mode adaptive_world \
  --lambda-world 0.04 \
  --world-update-interval 200 \
  --world-feedback-delay-windows 1 \
  --world-feedback-smooth-windows 2 \
  --out-dir outputs/personality_rl_async_rr_h3_delay_d1_s2
```

---

## 3. 決策規則（執行前先選）

- 若要最低成本確認「只差一格會不會翻案」：選 Option 2
- 若要確認「機制是否依賴 dispatch topology」：選 Option 3
- 二者不可同輪同時啟動，避免解釋耦合

建議預設：先 Option 3（資訊增益較高，能直接回答 H3 是否為 dispatch-specific）。
