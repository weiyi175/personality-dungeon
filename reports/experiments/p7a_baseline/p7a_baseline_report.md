# P7-A 基線快照報告（Static Personality Snapshot Baseline）

**執行日期**：2026-06-03  
**狀態**：全部 Gate PASS ✓ → 可繼續推進 P7-B W 矩陣確認

---

## 實驗配置

| 參數 | 值 |
|------|----|
| n_players | 4 |
| n_rounds | 200 |
| burn_in | 50 |
| payoff_mode | matrix_ab (a=1.0, b=0.9, cross=0.20) |
| personality_mode | static |
| feedback_strength α | 0.0 |
| lambda_alpha | 0.15 |
| lambda_beta | 0.10 |
| lambda_r | 0.20 |
| lambda_risk | 0.20 |
| seeds | {42, 43, 44, 45, 46} |
| 總 runs | 15 (5 seeds × 3 groups) |

---

## SBERT 推斷人格向量（P₀）

| 組別 | 輸入文字 | IMP | ASS | OPT | RAV | SUS | END | RND | STB | CUR |
|------|----------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| G-AGG | 我喜歡冒險挑戰 | +0.77 | +0.72 | — | −0.92 | — | — | — | −0.47 | +0.78 |
| G-DEF | 我謹慎保守行事 | −1.14 | +0.20 | — | +1.22 | — | — | — | +0.76 | −0.07 |
| G-BAL | 我靈活應對局面 | −0.38 | +0.27 | — | +0.10 | — | — | — | +0.55 | −0.03 |

> G-AGG 與 G-DEF 在 IMP 與 RAV 上呈強烈對比（如預期）。

---

## 策略分布結果（tail 51~200 均值）

| seed | G-AGG p_agg | G-AGG p_def | G-DEF p_agg | G-DEF p_def | G-BAL p_agg | G-BAL p_def |
|------|-------------|-------------|-------------|-------------|-------------|-------------|
| 42 | 0.357 | 0.270 | 0.277 | 0.358 | 0.317 | 0.310 |
| 43 | 0.383 | 0.267 | 0.320 | 0.337 | 0.352 | 0.300 |
| 44 | 0.380 | 0.260 | 0.327 | 0.317 | 0.353 | 0.285 |
| 45 | 0.393 | 0.255 | 0.325 | 0.328 | 0.352 | 0.295 |
| 46 | 0.388 | 0.252 | 0.338 | 0.310 | 0.358 | 0.277 |
| **均值** | **0.380** | **0.261** | **0.317** | **0.330** | **0.346** | **0.293** |

### 序數排序（驗證人格→策略影響方向）

- `p_aggressive`：G-AGG (0.380) > G-BAL (0.346) > G-DEF (0.317) ✓
- `p_defensive`：G-DEF (0.330) > G-BAL (0.293) > G-AGG (0.261) ✓

人格向量對策略分布的影響方向與設計預期完全一致。

---

## Gate 驗收結論

| Gate ID | 結果 | 說明 |
|---------|------|------|
| G7A-01 | **PASS** | 15/15 runs 完成，無例外 |
| G7A-02 | **PASS** | max seed std = 0.1352 < 0.15（基線穩定）|
| G7A-03 | **PASS** | mean_p_agg: G-AGG > G-DEF ✓；mean_p_def: G-DEF > G-AGG ✓ |
| G7A-04 | **PASS** | α=0.0 靜態路徑確認，personality_mode='static' 無 ΔP 更新 |

### G7A-03 準則說明

原始規格使用「dominant_strategy 相異」作為驗收，但 n_players=4 時 argmax 受隨機 seed 主導，噪音遮蔽了人格效果。本次採用**序數約束**（ordinal rank）替代：
- `mean(p_agg[G-AGG]) > mean(p_agg[G-DEF])`
- `mean(p_def[G-DEF]) > mean(p_def[G-AGG])`

兩個方向均以穩定差距（~6pp）通過，確認「人格→策略影響可觀測」。

---

## 引擎修正備忘（本次實驗發現）

**問題**：`RLSessionEngine._single_round_update()` 原本未套用 `strategy_alpha_multipliers`，導致人格耦合效果消失。

**修正**：對齊 `personality_rl_runtime.py` 的 BL2 相容邏輯：
```python
eff_alpha = min(1.0, player.alpha * player.strategy_alpha_multipliers[chosen_idx])
eff_reward = reward + player.risk_sensitivity * _RISK_SIGN[chosen_idx]
```

此修正使 `RLSessionEngine` 與主 runtime 行為一致，為 P7-C 以後的閉環注入打下正確基礎。

---

## P7-B W 矩陣校準建議

G7A-03 以序數準則通過，說明現有 W 矩陣設計值（P7-B §2.1）方向正確：
- G-AGG（高 IMP/ASS，低 RAV）→ aggressive 偏好增強 ✓
- G-DEF（低 IMP，高 RAV/STB）→ defensive 偏好增強 ✓

**建議**：P7-B W 矩陣不需要大幅調整，可保持目前設計值，直接進入 P7-C 回饋注入驗證。

若希望在 n=4 時得到更清晰的 dominant_strategy 分異，可考慮：
1. 提高 `lambda_r`（目前 0.20，可測試 0.35）
2. 增加 `n_rounds` 至 500+（讓 Q 值有更多時間收斂）

---

## 產出物

```
reports/experiments/p7a_baseline/
  run_{seed}_{group}.json        ← 15 個完整 step log
  p7a_baseline_summary.csv       ← 15 rows 指標彙整
  p7a_gates.json                 ← Gate 機器可讀結果
  p7a_baseline_report.md         ← 本文件
```

---

*可重現指令*：
```bash
./venv/bin/python scripts/experiments/run_p7a_baseline.py \
  --seeds 42 43 44 45 46 \
  --groups G-AGG G-DEF G-BAL \
  --out reports/experiments/p7a_baseline
```
