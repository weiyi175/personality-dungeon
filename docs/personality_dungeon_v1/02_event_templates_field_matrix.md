# 02 Event Templates Runtime Field Matrix

本文件整理 [02_event_templates_v1.json](docs/personality_dungeon_v1/02_event_templates_v1.json) 的 schema 欄位，標示目前 repo 中哪些欄位已進入 runtime，哪些仍停留在 spec / provenance，並給出建議落地位置。

狀態定義：

- 已落地：欄位已被 runtime 讀取，且直接參與驗證、決策、更新或輸出。
- 部分落地：欄位已被讀取或寫入，但尚未形成完整閉環。
- 未落地：欄位目前只存在於 spec / JSON，未被 runtime 消化。

補充評級：

- 影響 Level 2/3：評估該欄位若完整落地，對形成 delayed feedback / 結構性 cycle 的潛在槓桿。以 `★` 表示，滿分 5。
- 風險等級：
   - 高：未落地會直接卡住事件閉環。
   - 中：已部分落地，但不完整會扭曲分析或限制可解釋性。
   - 低：可延後，不影響當前核心動態判讀。

## 當前判斷摘要

- 強項：`weights -> risk -> success -> reward / trait / state -> replicator` 主鏈已通，且 provenance 足夠細，可精準定位問題來源。
- 核心缺口：`state_effects` 與 `risk_delta` 已開始回流下一輪 `risk / utility / success`，但目前仍是最小閉環，尚未涵蓋 `health`、全域 state decay 與 `failure_threshold`。
- 立即 bottleneck：`stress`、`risk_drift`、`noise`、`intel` 已進入 runtime hooks，但 `health` 仍未影響失敗機率或死亡 gate，`failure_threshold` 也仍未具正式語意。
- dead field：`risk_model.failure_threshold` 與 `risk_policy.default_failure_threshold` 現在幾乎沒有 runtime 語意。

## Root-Level Schema

| 欄位路徑 | 現況 | 目前用途 | 影響 Level 2/3 | 風險等級 | 建議落地位置 |
| --- | --- | --- | --- | --- | --- |
| `version` | 未落地 | 純 metadata，runtime 不讀取版本分支邏輯。 | ★☆☆☆☆ | 低 | `dungeon/event_loader.py` 啟動時做 schema version gate。 |
| `schema_status` | 未落地 | 純 metadata，未參與驗證。 | ★☆☆☆☆ | 低 | `dungeon/event_loader.py` 或 CLI warning。 |
| `event_types` | 部分落地 | event result 會輸出 `event_type`，但 loader 不驗證 type 是否屬於此清單。 | ★★☆☆☆ | 低 | `EventLoader.validate_schema()` 增加 type membership 檢查。 |
| `dimensions_order` | 已落地 | 決定人格向量順序、權重長度驗證、utility 計算。 | ★★★☆☆ | 低 | 已在 `EventLoader.__init__()` / `validate_schema()` / `compute_action_utility()`。 |
| `templates` | 已落地 | 事件資料主體，供抽樣、查詢、執行。 | ★★★★☆ | 低 | 已在 `EventLoader` 全流程。 |

## Global Policy Layer

| 欄位路徑 | 現況 | 目前用途 | 影響 Level 2/3 | 風險等級 | 建議落地位置 |
| --- | --- | --- | --- | --- | --- |
| `utility_weight_policy.raw_range` | 部分落地 | `normalize_weights()` 實際會 clamp 到 `[-1, 1]`，但未明確讀此欄位數值。 | ★★☆☆☆ | 低 | `normalize_weights()` 改為讀 schema 設定值。 |
| `utility_weight_policy.normalization` | 部分落地 | runtime 實作為 L1 normalization，但未檢查值是否為 `l1`。 | ★★☆☆☆ | 低 | `validate_schema()` 驗證枚舉值。 |
| `utility_weight_policy.eps` | 已落地 | 權重正規化防除零。 | ★★☆☆☆ | 低 | 已在 `normalize_weights()`。 |
| `utility_weight_policy.loader_rule` | 未落地 | 純說明文字。 | ★☆☆☆☆ | 低 | 保持 spec 用途即可，或改成 comment/doc。 |
| `utility_weight_policy.fallback` | 部分落地 | runtime 的確在全 0 / non-finite 時回傳零向量，但未逐字讀取此欄。 | ★☆☆☆☆ | 低 | 若要 machine-readable，改成 enum。 |
| `risk_policy.formula` | 未落地 | 純規格文字，實際公式寫死在 `compute_final_risk()`。 | ★★☆☆☆ | 低 | 保持 spec，或改成 structured config。 |
| `risk_policy.fallback` | 部分落地 | runtime 發生非有限值時 fallback 到 `base_risk`，但未逐字讀取此欄。 | ★★☆☆☆ | 低 | 若要 machine-readable，改成 enum。 |
| `risk_policy.clamp_range` | 已落地 | action 未提供 clamp 時作為全域預設。 | ★★★☆☆ | 低 | 已在 `compute_final_risk()`。 |
| `risk_policy.default_failure_threshold` | 未落地 | schema 有定義，但 runtime 未使用。 | ★★★☆☆ | 中 | `compute_success_prob()` 或 failure gate 層。 |
| `success_policy.default_model` | 已落地 | 作為 action 未指定 success model 時的預設。 | ★★★☆☆ | 低 | 已在 `_default_success_model()`。 |
| `success_policy.default_model_kwargs` | 已落地 | 併入預設 success model kwargs。 | ★★★☆☆ | 低 | 已在 `_default_success_model()`。 |
| `success_policy.registry_models` | 部分落地 | runtime 有 registry，但未驗證 JSON 清單與實際 registry 一致。 | ★★☆☆☆ | 低 | `validate_schema()` 比對 `SUCCESS_MODEL_REGISTRY`。 |
| `success_policy.default_probability_formula` | 部分落地 | 只用於 legacy inference，不是正式執行路徑。 | ★☆☆☆☆ | 低 | 長期可退役，保留 backward compatibility。 |
| `success_policy.alternative_formula` | 未落地 | 純 spec 說明。 | ★☆☆☆☆ | 低 | 保持文件用途，或移到 schema docs。 |
| `success_policy.fallback` | 部分落地 | return payload 內會攜帶 fallback 字串，但真正 fallback 邏輯仍寫死。 | ★★☆☆☆ | 低 | `compute_success_prob()` 轉為讀 structured fallback config。 |
| `success_policy.clamp_range` | 已落地 | success probability clamp 範圍。 | ★★★☆☆ | 低 | 已在 `compute_success_prob()`。 |
| `success_policy.default_success_bias` | 已落地 | logistic 預設 bias。 | ★★☆☆☆ | 低 | 已在 `_default_success_model()`。 |
| `success_policy.default_success_steepness` | 已落地 | logistic 預設 steepness。 | ★★☆☆☆ | 低 | 已在 `_default_success_model()`。 |
| `state_policy.state_variables` | 已落地 | schema 驗證允許的 state keys。 | ★★★★☆ | 中 | 已在 `validate_schema()`。 |
| `state_policy.default_state_effects.on_success.*` | 已落地 | action 未宣告時補預設 state deltas。 | ★★★★☆ | 中 | 已在 `resolve_state_effects()`。 |
| `state_policy.default_state_effects.on_failure.*` | 已落地 | action 未宣告時補預設 state deltas。 | ★★★★☆ | 中 | 已在 `resolve_state_effects()`。 |
| `reward_policy.success_field` | 未落地 | schema naming metadata，runtime 未動態讀欄位名。 | ★☆☆☆☆ | 低 | 若要 generic payload engine，可在 `_apply_reward_payload()` 前做欄位解析。 |
| `reward_policy.failure_field` | 未落地 | schema naming metadata，runtime 未動態讀欄位名。 | ★☆☆☆☆ | 低 | 同上。 |
| `reward_policy.state_field` | 未落地 | schema naming metadata，runtime 未動態讀欄位名。 | ★☆☆☆☆ | 低 | 同上。 |
| `reward_policy.required_reward_keys` | 已落地 | 驗證 `reward_effects` 必備欄位。 | ★★☆☆☆ | 低 | 已在 `validate_schema()`。 |

## Template-Level Fields

| 欄位路徑 | 現況 | 目前用途 | 影響 Level 2/3 | 風險等級 | 建議落地位置 |
| --- | --- | --- | --- | --- | --- |
| `templates[].event_id` | 已落地 | 建立索引、外部查詢、provenance 輸出。 | ★★★☆☆ | 低 | 已在 `template_by_id` / `process_turn()` / CSV。 |
| `templates[].type` | 部分落地 | 只在 event result / 分析輸出使用。 | ★★☆☆☆ | 低 | `validate_schema()` 增加 enum 驗證。 |
| `templates[].description` | 未落地 | 純敘事文字，runtime 不使用。 | ★☆☆☆☆ | 低 | 未來 UI / log / prompt 層。 |
| `templates[].actions` | 已落地 | action 集合，供選擇與執行。 | ★★★★☆ | 低 | 已在 `choose_action()` / `process_turn()`。 |

## Action-Level Fields

| 欄位路徑 | 現況 | 目前用途 | 影響 Level 2/3 | 風險等級 | 建議落地位置 |
| --- | --- | --- | --- | --- | --- |
| `templates[].actions[].name` | 已落地 | action 查找、provenance 輸出。 | ★★☆☆☆ | 低 | 已在 `process_turn()` / CSV。 |
| `templates[].actions[].weights` | 已落地 | 計算人格對 action 的 utility。 | ★★★★☆ | 中 | 已在 `compute_action_utility()`。 |
| `templates[].actions[].base_risk` | 已落地 | final risk 基底。 | ★★★★☆ | 中 | 已在 `compute_final_risk()`。 |
| `templates[].actions[].risk_model` | 已落地 | action 的風險模型容器。 | ★★★★☆ | 中 | 已在 `compute_final_risk()`。 |
| `templates[].actions[].risk_model.risk_bias` | 已落地 | 加到 final risk。 | ★★★★☆ | 中 | 已在 `compute_final_risk()`。 |
| `templates[].actions[].risk_model.trait_weights` | 已落地 | 依人格 trait 修正 risk。 | ★★★★☆ | 中 | 已在 `compute_final_risk()`。 |
| `templates[].actions[].risk_model.clamp` | 已落地 | action-specific 風險 clamp 範圍。 | ★★☆☆☆ | 低 | 已在 `compute_final_risk()`。 |
| `templates[].actions[].risk_model.failure_threshold` | 未落地 | schema 有值，但成功失敗不看它。 | ★★★☆☆ | 高 | `compute_success_prob()` 前增加 hard gate 或 piecewise risk regime。 |
| `templates[].actions[].success_model` | 已落地 | 支援 registry name 與 legacy mapping。 | ★★★☆☆ | 中 | 已在 `resolve_success_model()`。 |
| `templates[].actions[].success_model.name` | 部分落地 | 僅 legacy object 寫法可讀。 | ★☆☆☆☆ | 低 | 長期建議統一為 `success_model: string`。 |
| `templates[].actions[].success_model.probability_formula` | 部分落地 | 只用於 legacy model inference。 | ★☆☆☆☆ | 低 | 長期退役或保留 backward compatibility。 |
| `templates[].actions[].success_model.success_bias` | 部分落地 | legacy object 可被轉成 kwargs。 | ★★☆☆☆ | 低 | 長期統一到 `success_model_kwargs`。 |
| `templates[].actions[].success_model.success_steepness` | 部分落地 | legacy object 可被轉成 kwargs。 | ★★☆☆☆ | 低 | 長期統一到 `success_model_kwargs`。 |
| `templates[].actions[].success_model.fallback` | 未落地 | object 內 fallback 目前不驅動實際執行。 | ★★☆☆☆ | 低 | `compute_success_prob()`。 |
| `templates[].actions[].success_model.kwargs` | 部分落地 | legacy object 路徑可讀。 | ★☆☆☆☆ | 低 | 與 `success_model_kwargs` 擇一保留。 |
| `templates[].actions[].success_model_kwargs` | 已落地 | 主要 registry kwargs 輸入。 | ★★★☆☆ | 中 | 已在 `resolve_success_model()`。 |

## Success / Failure Payload Fields

| 欄位路徑 | 現況 | 目前用途 | 影響 Level 2/3 | 風險等級 | 建議落地位置 |
| --- | --- | --- | --- | --- | --- |
| `templates[].actions[].reward_effects` | 已落地 | success payload 主體。 | ★★★☆☆ | 低 | 已在 `process_turn()`。 |
| `templates[].actions[].reward_effects.utility_delta` | 已落地 | 加到該 round 總 reward。 | ★★★☆☆ | 中 | 已在 `_apply_reward_payload()` / `DungeonAI.resolve_player_outcome()`。 |
| `templates[].actions[].reward_effects.risk_delta` | 已落地 | 累加到 `player.state["risk"]`，並回流下一輪 `final_risk`。 | ★★★★★ | 中 | 已在 `compute_final_risk()` 讀 `player.state`。 |
| `templates[].actions[].reward_effects.popularity_shift` | 已落地 | 改玩家 strategy biases，影響後續抽樣。 | ★★★★☆ | 中 | 已在 `BasePlayer.apply_popularity_shift()` / `choose_strategy()`。 |
| `templates[].actions[].reward_effects.trait_deltas` | 已落地 | 改人格 trait，影響後續 action utility 與 risk。 | ★★★★☆ | 中 | 已在 `BasePlayer.apply_trait_deltas()`。 |
| `templates[].actions[].reward_effects.sample_quality` | 部分落地 | 會寫入 `last_sample_quality` 與 provenance，但不影響模擬。 | ★★☆☆☆ | 低 | `simulation/` 或 `04_economy_rules_v1.json` 接軌。 |
| `templates[].actions[].reward_effects.state_tags` | 部分落地 | 會寫入 `last_event_tags` 與 provenance，但不影響模擬。 | ★★☆☆☆ | 低 | `simulation/` orchestration、event chain、economy、meta engine。 |
| `templates[].actions[].state_effects` | 部分落地 | `stress`、`risk_drift`、`noise`、`intel` 已回流；`health` 與 decay / gate 尚未回流。 | ★★★★★ | 中 | `compute_action_utility()` / `compute_final_risk()` / `compute_success_prob()` / global state decay。 |
| `templates[].actions[].state_effects.on_success.stress_delta` | 已落地 | 寫入 `player.state["stress"]`，並影響下一輪 risk。 | ★★★★☆ | 中 | 已在 `compute_final_risk()`。 |
| `templates[].actions[].state_effects.on_success.noise_delta` | 已落地 | 寫入 `player.state["noise"]`，並放大 utility 抖動。 | ★★★★☆ | 中 | 已在 `compute_action_utility()`。 |
| `templates[].actions[].state_effects.on_success.risk_drift_delta` | 已落地 | 寫入 `player.state["risk_drift"]`，並影響下一輪 risk。 | ★★★★★ | 中 | 已在 `compute_final_risk()`。 |
| `templates[].actions[].state_effects.on_success.health_delta` | 部分落地 | 寫入 `player.state["health"]`，但尚未影響成功率或死亡 gate。 | ★★★☆☆ | 中 | 影響死亡 / 脆弱度 / failure sensitivity。 |
| `templates[].actions[].state_effects.on_success.intel_delta` | 已落地 | 寫入 `player.state["intel"]`，並影響 success probability。 | ★★★★☆ | 中 | 已在 `compute_success_prob()`。 |
| `templates[].actions[].state_effects.on_failure.stress_delta` | 已落地 | 同上。 | ★★★★☆ | 中 | 同上。 |
| `templates[].actions[].state_effects.on_failure.noise_delta` | 已落地 | 同上。 | ★★★★☆ | 中 | 同上。 |
| `templates[].actions[].state_effects.on_failure.risk_drift_delta` | 已落地 | 同上。 | ★★★★★ | 中 | 同上。 |
| `templates[].actions[].state_effects.on_failure.health_delta` | 部分落地 | 同上。 | ★★★☆☆ | 中 | 同上。 |
| `templates[].actions[].state_effects.on_failure.intel_delta` | 已落地 | 同上。 | ★★★★☆ | 中 | 同上。 |
| `templates[].actions[].failure_outcomes` | 已落地 | failure payload 候選集合。 | ★★★☆☆ | 中 | 已在 `choose_failure_outcome()`。 |
| `templates[].actions[].failure_outcomes[].kind` | 已落地 | failure 類型名稱，輸出到 `result_kind`。 | ★★☆☆☆ | 低 | 已在 `process_turn()` / provenance。 |
| `templates[].actions[].failure_outcomes[].probability` | 已落地 | 多個 failure outcome 時作抽樣權重。 | ★★☆☆☆ | 低 | 已在 `choose_failure_outcome()`。 |
| `templates[].actions[].failure_outcomes[].utility_delta` | 已落地 | failure reward 變動。 | ★★★☆☆ | 中 | 已在 `_apply_reward_payload()`。 |
| `templates[].actions[].failure_outcomes[].risk_delta` | 已落地 | 進 `player.state["risk"]`，並回流下一輪 `final_risk`。 | ★★★★★ | 中 | 已在 `compute_final_risk()` 讀 state。 |
| `templates[].actions[].failure_outcomes[].popularity_shift` | 已落地 | 影響 strategy bias。 | ★★★★☆ | 中 | 已在 `BasePlayer.apply_popularity_shift()`。 |
| `templates[].actions[].failure_outcomes[].trait_deltas` | 已落地 | 影響人格。 | ★★★★☆ | 中 | 已在 `BasePlayer.apply_trait_deltas()`。 |
| `templates[].actions[].failure_outcomes[].state_tags` | 部分落地 | 只保存在 state / provenance。 | ★★☆☆☆ | 低 | event chain / environment / economy hook。 |
| `templates[].actions[].failure_outcomes[].sample_quality` | 部分落地 | payload 若提供可被寫出，但現有模板幾乎沒用它。 | ★★☆☆☆ | 低 | economy / loot / sample pipeline。 |

## Provenance / Analysis Coverage

| 欄位來源 | 現況 | 目前輸出 | 影響 Level 2/3 | 風險等級 | 建議落地位置 |
| --- | --- | --- | --- | --- | --- |
| `event_id`, `event_type`, `action_name` | 已落地 | timeseries CSV 與 provenance summary。 | ★★☆☆☆ | 低 | 已在 `simulation/run_simulation.py`。 |
| `success`, `result_kind`, `final_risk`, `success_prob` | 已落地 | timeseries CSV 與 analysis summary。 | ★★★☆☆ | 中 | 已在 `process_turn()` / `_aggregate_event_records()`。 |
| `trait_deltas`, `popularity_shift`, `state_effects` | 已落地 | JSON aggregation 進 CSV。 | ★★★☆☆ | 中 | 已在 `_aggregate_event_records()`。 |
| `sample_quality`, `state_tags` | 部分落地 | 存於 player.state，但目前未寫進 CSV。 | ★★☆☆☆ | 低 | `simulation/run_simulation.py` 增加 provenance 欄。 |
| row-level cycle attribution | 部分落地 | `analysis/event_provenance_summary.py` 用 derived cycle level 補判。 | ★★★☆☆ | 中 | 若要嚴格因果分析，需在 simulation 寫 row-level cycle metrics。 |

## 現階段最重要的未落地項目

1. `risk_model.failure_threshold`
   目前 schema 已宣告，但 success / failure 判定完全不看它。

2. `health_delta`
   已寫入 `player.state["health"]`，但尚未回流 `compute_success_prob()`、死亡 gate 或 failure sensitivity。

3. `state_effects.*`
   `stress`、`noise`、`risk_drift`、`intel` 已開始影響下一輪，但目前仍缺 `health`、state decay 與更強的全域 delayed coupling。

4. `sample_quality` 與 `state_tags`
   已有觀測價值，但目前尚未接到 `04_economy_rules_v1.json`、事件鏈或 meta world-pressure。

## 快速診斷表

目前最可能卡住 Level 2 的欄位：

1. `health_delta` 尚未回流：脆弱度仍未進入 success / death gate，限制事件鏈分叉。
2. `risk_model.failure_threshold` dead：風險門檻沒有實際語意，schema 空間被浪費。
3. 缺少 state decay / global coupling：目前是最小閉環，但 delayed structure 強度仍可能不足以推升到 Level 2。

## 三步落地順序建議

1. 已完成：state 回流 risk / utility / success
   目前 `compute_final_risk()` 已讀 `stress`、`risk_drift`、`state["risk"]`；
   `compute_action_utility()` 已讀 `noise`；
   `compute_success_prob()` 已讀 `intel`。

2. 接著做：補 `health` 與 `failure_threshold`
   讓 `health` 進入 failure sensitivity 或 death gate；
   明確決定 `failure_threshold` 是 hard gate 還是 success-model regime switch。

3. 最後做：補 provenance 與更強 delayed coupling
   在 `simulation/run_simulation.py` 補 `sample_quality` / `state_tags` 欄位；
	   補 state decay 或更明確的 global coupling；
	   經濟與事件鏈 integration 再延後到 `04_economy_rules_v1.json` / orchestration 層。

## 實驗建議

完成第 1 步後，優先重跑既有的 `expected + init_bias=0.12` 診斷組合。這是目前最乾淨的對照條件，最容易看出 `env_gamma` 是否下降、`derived_cycle_level` 是否從 0 升到 2。