# W2.1 Implementation Blueprint

本文件把 W2.1 最小 3-cell cross-life protocol 拆成可直接實作的工程藍圖。
目標不是一次做完完整世界模型，而是用最小侵入方式把 `death + testament + next life reset` 接到既有 sampled + events 管線上。

## 1. Runtime 拆分

建議新增一個獨立 harness：

1. `simulation/w2_episode.py`

責任切分：

1. 組裝 `control / w2_base / w2_strong` 三個固定 cells
2. 逐 seed 跑 `total_lives=5`
3. 在每個 life 內重用既有 sampled + events simulation 主幹
4. 產出 seed-level、life-level、combined summary 與 decision markdown

建議新增的最小資料結構：

```python
@dataclass(frozen=True)
class W2CellConfig:
    condition: str
    testament_alpha: float
    total_lives: int
    rounds_per_life: int
    players: int
    events_json: Path
    a: float
    b: float
    cross: float
    selection_strength: float
    init_bias: float
    memory_kernel: int
    out_dir: Path


@dataclass
class LifeSnapshot:
    life_index: int
    ended_by_death: bool
    n_deaths: int
    rounds_completed: int
    mean_stage3_score: float
    mean_env_gamma: float
    level3_seed_count: int
    testament_applied: bool
```

建議新增的最小函式：

1. `run_w2_cell(config, seed) -> dict`
2. `run_life(config, seed, life_index, players, inherited_personalities) -> dict`
3. `apply_testament(players, alpha) -> list[dict[str, float]]`
4. `compute_death_threshold(player) -> float`
5. `compute_personality_risk_delta(player) -> float`
6. `reset_player_for_next_life(player, next_personality) -> None`

最少侵入的掛點：

1. 若要最大化重用 metrics / summary / decision 管線，優先仿照 `simulation/world_state_w1.py` 的 cell-runner 介面，而不是直接複製整個 `run_simulation.py`
2. death check 最適合掛在 event 結算之後；現有 `player.state["risk"]` 已由 [dungeon/event_loader.py](dungeon/event_loader.py#L625) 累加
3. personality 更新與 state reset 最適合重用 [players/base_player.py](players/base_player.py#L67) 與 [players/base_player.py](players/base_player.py#L75) 的現成入口

## 2. Life Summary Schema

建議拆成兩層 TSV。

### 2.1 Seed x Life TSV

建議檔名：

1. `outputs/w2_episode_life_steps.tsv`

建議欄位：

1. `protocol`
2. `condition`
3. `seed`
4. `life_index`
5. `ended_by_death`
6. `n_deaths`
7. `mean_life_rounds`
8. `rounds_completed`
9. `mean_stage3_score`
10. `mean_turn_strength`
11. `mean_env_gamma`
12. `level3_seed_count`
13. `testament_alpha`
14. `testament_applied`
15. `dominant_strategy_last500`
16. `mean_utility`
17. `std_utility`
18. `mean_success_rate`
19. `mean_risk_final`
20. `mean_threshold`
21. `out_csv`
22. `provenance_json`

欄位意義：

1. `life_index` 是跨 life 的主要時間軸
2. `ended_by_death` / `n_deaths` 用來區分真正有沒有啟動 W2 機制
3. `dominant_strategy_last500`、`mean_utility`、`std_utility` 是 testament 可追溯來源

### 2.2 Cell Combined TSV

建議檔名：

1. `outputs/w2_episode_combined.tsv`

建議欄位：

1. `protocol`
2. `condition`
3. `is_control`
4. `n_seeds`
5. `total_lives`
6. `testament_alpha`
7. `mean_stage3_score_all_lives`
8. `mean_env_gamma_all_lives`
9. `mean_stage3_score_tail_lives`
10. `mean_env_gamma_tail_lives`
11. `tail_level3_seed_count`
12. `tail_p_level_3`
13. `mean_n_deaths`
14. `mean_life_rounds`
15. `testament_activation_rate`
16. `control_tail_mean_env_gamma`
17. `control_tail_mean_stage3_score`
18. `tail_gamma_uplift_vs_control`
19. `tail_stage3_uplift_vs_control`
20. `tail_gamma_nonnegative`
21. `tail_level3_pass`
22. `full_pass`
23. `verdict`
24. `events_json`
25. `selection_strength`
26. `memory_kernel`
27. `out_dir`

其中 tail 固定指 `life 3..5`。

這樣設計的理由是：W2 不應該主要看全期平均，而要明確看後半段 lives 是否第一次出現 uplift。

## 3. Decision Markdown

建議檔名：

1. `outputs/w2_episode_decision.md`

建議結構：

```md
# W2.1 Decision

- protocol: w2_episode
- decision: pass | weak_positive | close_w2_1
- control_tail_mean_stage3_score: ...
- control_tail_mean_env_gamma: ...
- control_tail_level3_seed_count: ...

## Tail-Life Verdicts

| Cell | tail Level 3 seeds | tail mean_stage3_score | tail mean_env_gamma | testament_activation_rate | verdict |
|---|---:|---:|---:|---:|---|
| control | ... | ... | ... | ... | control |
| w2_base | ... | ... | ... | ... | ... |
| w2_strong | ... | ... | ... | ... | ... |

## Gate

- pass: 至少 1 個 non-control cell 在 life 3..5 出現 >=1 個 Level 3 seed，且 tail mean_env_gamma >= 0。
- close_w2_1: 若後半段 life 完全沒有 Level 3 seed，則不再微調 alpha，直接結案 W2.1。

## Interpretation

- 是否真的發生 death / testament
- uplift 是否只在前半段 life 出現
- 是否值得進 W2.2 world carryover
```

判決規則建議：

1. `pass`：tail `level3_seed_count >= 1` 且 tail `mean_env_gamma >= 0`
2. `weak_positive`：沒有 pass，但 tail `mean_env_gamma` 明顯優於 control，且 death / testament 確實有發生
3. `close_w2_1`：tail 完全沒有 Level 3 seed，或根本沒有真正啟動 death / testament

## 4. 最小測試集

建議新增：

1. `tests/test_w2_episode.py`

最小測試分成四組。

### 4.1 契約測試

1. `alpha_testament=0` 時，下一 life personality 完全不變
2. `DeltaP` 在乘上 alpha 前一定先 clip 到 `[-0.25, 0.25]`
3. control cell 不得進入 cross-life testament path

### 4.2 死亡條件測試

1. 只有 4 個 trait 影響 `Delta_risk_personality`
2. `threshold_i` 只由 `caution`, `stability_seeking`, `impulsiveness`, `fearfulness` 決定
3. personality risk weights 全為 `0` 時，death 仍可能由 event `risk_delta` 觸發；這個測試用來防止把 event-only death 誤寫成 no-death

### 4.3 Runtime / I/O 測試

1. `run_w2_scout(...)` 會寫出 `life_steps.tsv`, `combined.tsv`, `decision.md`
2. `life_steps.tsv` 至少包含 `life_index`, `ended_by_death`, `n_deaths`, `testament_applied`
3. `combined.tsv` 至少包含 tail-life 欄位與 `verdict`

### 4.4 Decision 邏輯測試

1. tail 出現 Level 3 seed 且 tail gamma 非負時，decision 必須是 `pass`
2. 只有 gamma 改善、沒有 tail Level 3 seed 時，decision 最多只能是 `weak_positive`
3. 後半段 life 完全沒有 Level 3 seed 時，decision 必須是 `close_w2_1`

## 5. 推薦實作順序

1. 先做純函式：`compute_personality_risk_delta`, `compute_death_threshold`, `apply_testament`
2. 再做 `run_life(...)`，先讓單 seed / 單 cell / 多個 life 可跑
3. 再接 `run_w2_scout(...)` 的 summary / combined / decision pipeline
4. 最後才跑正式 `control / w2_base / w2_strong` 三格 scout