# Track B v2 完整實驗藍圖（Blueprint v2.1）

版本：2.1  
日期：2026-04-18  
適用專案：Personality Dungeon（SDD-driven）

## 執行摘要（Executive Summary）

本藍圖以 [SDD.md](../../SDD.md)、[研發日誌.md](../../研發日誌.md)、[high_coupling_failure_diagnosis_spec.md](../high_coupling_failure_diagnosis_spec.md)、[phase_based_coupling_blueprint_v2.md](../phase_based_coupling_blueprint_v2.md) 為唯一契約，並將以下結果視為鎖定事實：

- Phase 0 通過：[outputs/phase_coupling_v2/p0_lock_status.json](../../../outputs/phase_coupling_v2/p0_lock_status.json)
- Phase 1 通過，gate_overall_pass=true、gate_new_l1=0、pre_a3_overall_pass=true、post_a3_overall_pass=true：[outputs/phase_coupling_v2/p1_lock_status.json](../../../outputs/phase_coupling_v2/p1_lock_status.json)
- Phase 2 通過，gate_overall_pass=true、gate_new_l1=0、pre_a3_overall_pass=true、post_a3_overall_pass=true：[outputs/phase_coupling_v2/p2_lock_status.json](../../../outputs/phase_coupling_v2/p2_lock_status.json)
- Phase 3 失敗，gate_overall_pass=false、gate_new_l1=1、gate_l1=3、gate_healthy=51、gate_fairness_fail_count=0、gate_invariant_overall_pass=true、pre_a3_overall_pass=true、post_a3_overall_pass=true：[outputs/phase_coupling_v2/p3_lock_status.json](../../../outputs/phase_coupling_v2/p3_lock_status.json)
- Phase 3 first-break 已定位：minimal_new_l1_seed=73、new_l1_seeds=[73]、broke_seeds=[55,59,67,69,73,85]：[outputs/phase_coupling_v2/p3_first_break_localization.json](../../../outputs/phase_coupling_v2/p3_first_break_localization.json)
- Bridge-3 fallback 決議成立：第一破壞點落在 high_coupling_bundle_v1，停止同族局部微調：[bridge3_first_break_decision_draft_20260418.md](../bridge3_first_break_decision_draft_20260418.md)

Track B v2.1 的主目標是直接改寫三層均化瓶頸的架構，不再沿用 Bridge-3 同族微調。  
Track B v2.1 的硬約束如下：

- 保留 Phase 2 PASS 基線能力：read-only world state 與 difficulty modulation 不破壞
- 主攻 Layer 3：asynchronous 更新、per-player event queue、multiplicative reward modulation、impact spreading kernel
- 同步強化 Layer 1/2：async dispatch、local/hierarchical averaging、phase lag / memory kernel
- 僅允許六個固定子任務：B1、B2、B3、B1+B2、B1+B3、B1+B2+B3
- 每個子任務都必跑 pre-A3、主實驗（smoke+gate60）、post-A3
- 60-seed Gate 硬規則固定：L1<=3、Healthy>=42、new_L1=0、fairness_fail_count=0、gate.invariant_overall_pass=true
- 全部 Python 指令固定使用 ./venv/bin/python -m
- SDD 先 Spec 後程式碼；analysis 不得 import simulation；evolution 不做 I/O；simulation 維護 CSV 契約

---

## Track B v2 與既有 Blueprint 的對應關係

| 既有鎖定結果 | 結論 | Track B v2.1 對應動作 |
|---|---|---|
| [outputs/phase_coupling_v2/p0_lock_status.json](../../../outputs/phase_coupling_v2/p0_lock_status.json) | P0 baseline 穩定 | P0 baseline 固定為 new_L1 唯一母體 |
| [outputs/phase_coupling_v2/p1_lock_status.json](../../../outputs/phase_coupling_v2/p1_lock_status.json) | read-only world state 已驗證可保留 | 所有 Track B 任務維持 world_readonly 不破壞約束 |
| [outputs/phase_coupling_v2/p2_lock_status.json](../../../outputs/phase_coupling_v2/p2_lock_status.json) | difficulty-only 已驗證可保留 | Track B 所有主實驗固定在 phase2_pass_locked_v1 錨點上擴充 |
| [outputs/phase_coupling_v2/p3_lock_status.json](../../../outputs/phase_coupling_v2/p3_lock_status.json) | full bundle 在 new_L1=1 失敗 | 禁止同族 Bridge-3 局部微調，改走 Track B 新架構 |
| [outputs/phase_coupling_v2/p3_first_break_localization.json](../../../outputs/phase_coupling_v2/p3_first_break_localization.json) | first-break seed 已定位到 73 | Track B 每個子任務都輸出 broke/new_L1 seeds 與 provenance |
| [bridge3_first_break_decision_draft_20260418.md](../bridge3_first_break_decision_draft_20260418.md) | fallback 已簽核草案成立 | Track B 僅允許新分支架構修改，不回頭微調 Bridge-3 |

Track B v2.1 對既有 Blueprint v2 的承接規則：

- Phase 0/1/2 視為封存完成，不重跑、不重定義門檻
- Track B 作為新主分支，phase_id 命名改為 tb2_*
- 每次 smoke 僅單一機制或單一組合，禁止多組混跑
- Gate 一律 42..101 共 60 seeds
- 任務前進條件固定為：pre-A3 PASS 且 gate PASS 且 post-A3 PASS

---

## Track B v2 詳細實驗設計（每子任務獨立一節）

### B1 Async Dispatch v2

目標：  
以非同步 per-player 更新與 per-player event queue 破壞 synchronized replicator 的同相耗散，並保持 Phase 2 的 difficulty-only 能力不退化。

核心程式變更清單（檔案路徑 + 最小改動）：

1. [simulation/personality_rl_runtime.py](../../../simulation/personality_rl_runtime.py)：新增 replicator_update_mode=async_per_player、event_queue_mode=per_player、queue cap/drain、async 更新 diagnostics。
2. [evolution/independent_rl.py](../../../evolution/independent_rl.py)：新增 async q-update operator（按 player queue 與 minibatch 更新）。
3. [simulation/b1_async_dispatch_gate.py](../../../simulation/b1_async_dispatch_gate.py)：新增 B1 v2 CLI 參數與 gate 欄位彙總（async_update_applied、queue_overflow_count）。
4. [analysis/event_provenance_summary.py](../../../analysis/event_provenance_summary.py)：新增 summarize_async_update_skew、compute_queue_saturation。
5. [tests/test_personality_rl_runtime.py](../../../tests/test_personality_rl_runtime.py)、[tests/test_b1_async_dispatch_gate.py](../../../tests/test_b1_async_dispatch_gate.py)、[tests/test_event_provenance_summary.py](../../../tests/test_event_provenance_summary.py)：新增 B1 v2 契約測試。

完整可執行 CLI（pre-A3、主實驗、post-A3）：

~~~bash
./venv/bin/python -m simulation.track_a_protocol_regression --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json --recheck-out-json outputs/track_b_v2/b1_async_dispatch_v2/tb2_b1_pre_a3_recheck_42_101_20260420_summary.json --pytest-target tests/test_personality_rl_runtime.py --phase-id tb2_b1 --bridge-id b1_async_dispatch_v2 --bridge-count 1 --anchor-profile-id phase2_pass_locked_v1 --python-bin ./venv/bin/python --summary-json outputs/track_b_v2/b1_async_dispatch_v2/tb2_b1_pre_a3_20260420_summary.json --summary-md outputs/track_b_v2/b1_async_dispatch_v2/tb2_b1_pre_a3_20260420_summary.md

./venv/bin/python -m simulation.b1_async_dispatch_gate --phase-id tb2_b1 --bridge-id b1_async_dispatch_v2 --bridge-count 1 --anchor-profile-id phase2_pass_locked_v1 --smoke-seeds 45,47,49,51,53,55 --gate-seeds 42..101 --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json --world-feedback-mode difficulty_only --event-dispatch-mode async_poisson --event-dispatch-target-rate 0.08 --event-dispatch-fairness-window 2000 --event-dispatch-fairness-tolerance 0.50 --event-trigger-mode entropy_guard --event-trigger-entropy-threshold 0.85 --event-neutralize-payoff --event-reward-mode additive --event-impact-mode instant --event-warmup-rounds 1000 --replicator-update-mode async_per_player --replicator-async-minibatch 32 --replicator-async-jitter 0.15 --event-queue-mode per_player --event-queue-cap 8 --event-queue-drain-rate 1.00 --smoke-out-json outputs/track_b_v2/b1_async_dispatch_v2/tb2_b1_smoke_s6_20260420_summary.json --gate-out-json outputs/track_b_v2/b1_async_dispatch_v2/tb2_b1_gate60_20260420_summary.json

./venv/bin/python -m simulation.track_a_protocol_regression --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json --recheck-out-json outputs/track_b_v2/b1_async_dispatch_v2/tb2_b1_post_a3_recheck_42_101_20260420_summary.json --pytest-target tests/test_b1_async_dispatch_gate.py --phase-id tb2_b1 --bridge-id b1_async_dispatch_v2 --bridge-count 1 --anchor-profile-id phase2_pass_locked_v1 --python-bin ./venv/bin/python --summary-json outputs/track_b_v2/b1_async_dispatch_v2/tb2_b1_post_a3_20260420_summary.json --summary-md outputs/track_b_v2/b1_async_dispatch_v2/tb2_b1_post_a3_20260420_summary.md
~~~

Gate 契約與 new_L1 判定方式：

- smoke 目標：L3_rate >= 4/6 且 fairness_fail_count=0
- gate60 硬規則：L1<=3、Healthy>=42、new_L1=0、fairness_fail_count=0、gate.invariant_overall_pass=true
- 定義：
$$
new\_L1^{(tb2\_b1)} = |\{s \mid healthy_{P0}(s)=1 \land level_{tb2\_b1}(s)=1\}|
$$
- 硬規則：
$$
new\_L1^{(tb2\_b1)}=0
$$
- baseline 母體固定：outputs/phase_coupling_v2/p0_baseline_gate60_summary.json

診斷輸出要求（新增 CSV 欄位與 analysis 函式）：

- 新增 CSV 欄位：replicator_update_mode、async_update_applied、player_event_queue_depth_mean、player_event_queue_depth_p95、queue_drop_count、update_skew_index、phase_lag_index
- 新增 analysis 函式：summarize_async_update_skew、compute_queue_saturation、verify_async_dispatch_contract（放在 [analysis/event_provenance_summary.py](../../../analysis/event_provenance_summary.py)）

風險與 fallback（明確 stop condition）：

- stop condition：gate_new_l1>0
- stop condition：dispatch_fairness_fail_count>0
- stop condition：async_update_applied=false（任一 gate seed）
- stop condition：queue_drop_count>0（任一 gate seed）
- stop condition：pre_a3_overall_pass=false 或 post_a3_overall_pass=false
- fallback：立即停用 B1 v2，回退至 Phase 2 鎖定配置（bridge_count=0），輸出 tb2_b1_fail_lock.json 與 broke/new_L1 seeds 清單

---

### B2 Multiplicative Modulation v2

目標：  
在不改 payoff 幾何（a,b,cross）前提下，導入 log-space multiplicative reward modulation，提升旋轉訊號傳遞強度並維持 new_L1=0。

核心程式變更清單（檔案路徑 + 最小改動）：

1. [simulation/personality_rl_runtime.py](../../../simulation/personality_rl_runtime.py)：新增 event_modulation_mode=multiplicative_v2、zero-mean log modulation、floor/ceiling clamp。
2. [dungeon/event_loader.py](../../../dungeon/event_loader.py)：新增 multiplicative_v2 modulation adapter（只作用於 event reward modulation，不改 base payoff matrix）。
3. [simulation/b1_async_dispatch_gate.py](../../../simulation/b1_async_dispatch_gate.py)：新增 modulation diagnostics 匯總欄位（multiplicative_static_pass、modulation_zero_mean_residual）。
4. [analysis/event_provenance_summary.py](../../../analysis/event_provenance_summary.py)：新增 verify_multiplicative_static_v2、summarize_modulation_gain_distribution。
5. [tests/test_personality_rl_runtime.py](../../../tests/test_personality_rl_runtime.py)、[tests/test_event_loader.py](../../../tests/test_event_loader.py)、[tests/test_b1_async_dispatch_gate.py](../../../tests/test_b1_async_dispatch_gate.py)：新增 B2 v2 不變性測試。

完整可執行 CLI（pre-A3、主實驗、post-A3）：

~~~bash
./venv/bin/python -m simulation.track_a_protocol_regression --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json --recheck-out-json outputs/track_b_v2/b2_multiplicative_modulation_v2/tb2_b2_pre_a3_recheck_42_101_20260422_summary.json --pytest-target tests/test_event_loader.py --phase-id tb2_b2 --bridge-id b2_multiplicative_modulation_v2 --bridge-count 1 --anchor-profile-id phase2_pass_locked_v1 --python-bin ./venv/bin/python --summary-json outputs/track_b_v2/b2_multiplicative_modulation_v2/tb2_b2_pre_a3_20260422_summary.json --summary-md outputs/track_b_v2/b2_multiplicative_modulation_v2/tb2_b2_pre_a3_20260422_summary.md

./venv/bin/python -m simulation.b1_async_dispatch_gate --phase-id tb2_b2 --bridge-id b2_multiplicative_modulation_v2 --bridge-count 1 --anchor-profile-id phase2_pass_locked_v1 --smoke-seeds 45,47,49,51,53,55 --gate-seeds 42..101 --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json --world-feedback-mode difficulty_only --event-dispatch-mode sync --event-dispatch-target-rate 0.08 --event-dispatch-fairness-window 2000 --event-dispatch-fairness-tolerance 0.50 --event-trigger-mode entropy_guard --event-trigger-entropy-threshold 0.85 --event-neutralize-payoff --event-reward-mode multiplicative --event-reward-multiplier-cap 0.20 --event-modulation-mode multiplicative_v2 --event-modulation-gain 0.12 --event-modulation-log-center 0.00 --event-modulation-zero-mean --event-modulation-floor 0.85 --event-modulation-ceiling 1.15 --event-impact-mode instant --smoke-out-json outputs/track_b_v2/b2_multiplicative_modulation_v2/tb2_b2_smoke_s6_20260422_summary.json --gate-out-json outputs/track_b_v2/b2_multiplicative_modulation_v2/tb2_b2_gate60_20260422_summary.json

./venv/bin/python -m simulation.track_a_protocol_regression --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json --recheck-out-json outputs/track_b_v2/b2_multiplicative_modulation_v2/tb2_b2_post_a3_recheck_42_101_20260422_summary.json --pytest-target tests/test_personality_rl_runtime.py --phase-id tb2_b2 --bridge-id b2_multiplicative_modulation_v2 --bridge-count 1 --anchor-profile-id phase2_pass_locked_v1 --python-bin ./venv/bin/python --summary-json outputs/track_b_v2/b2_multiplicative_modulation_v2/tb2_b2_post_a3_20260422_summary.json --summary-md outputs/track_b_v2/b2_multiplicative_modulation_v2/tb2_b2_post_a3_20260422_summary.md
~~~

Gate 契約與 new_L1 判定方式：

- smoke 目標：L3_rate >= 4/6 且 multiplicative_static_pass=true（6/6）
- gate60 硬規則：L1<=3、Healthy>=42、new_L1=0、fairness_fail_count=0、gate.invariant_overall_pass=true
- 定義：
$$
new\_L1^{(tb2\_b2)} = |\{s \mid healthy_{P0}(s)=1 \land level_{tb2\_b2}(s)=1\}|
$$
- 硬規則：
$$
new\_L1^{(tb2\_b2)}=0
$$
- baseline 母體固定：outputs/phase_coupling_v2/p0_baseline_gate60_summary.json

診斷輸出要求（新增 CSV 欄位與 analysis 函式）：

- 新增 CSV 欄位：reward_multiplier_raw、reward_multiplier_clamped、log_reward_multiplier、modulation_gain_effective、modulation_zero_mean_residual、multiplicative_static_pass
- 新增 analysis 函式：verify_multiplicative_static_v2、summarize_modulation_gain_distribution、check_log_multiplier_zero_mean（放在 [analysis/event_provenance_summary.py](../../../analysis/event_provenance_summary.py)）

風險與 fallback（明確 stop condition）：

- stop condition：gate_new_l1>0
- stop condition：multiplicative_static_pass=false（任一 gate seed）
- stop condition：modulation_zero_mean_residual>1e-6（任一 gate seed）
- stop condition：pre_a3_overall_pass=false 或 post_a3_overall_pass=false
- fallback：立即關閉 multiplicative_v2，回退 Phase 2 鎖定配置，輸出 tb2_b2_fail_lock.json 與 broke/new_L1 seeds 清單

---

### B3 Impact Spreading v2

目標：  
以 hierarchical impact spreading kernel 把事件擾動由單點衝擊改為可控時空擴散，降低瞬時同步耗散並提升有效 phase lag。

核心程式變更清單（檔案路徑 + 最小改動）：

1. [simulation/personality_rl_runtime.py](../../../simulation/personality_rl_runtime.py)：新增 impact_spread_kernel_id、local/neighbor mass、neighbor hop、memory kernel。
2. [simulation/w2_episode.py](../../../simulation/w2_episode.py)：新增 carryover 與 spreading 相容欄位，不更動 memory cap 契約。
3. [simulation/b1_async_dispatch_gate.py](../../../simulation/b1_async_dispatch_gate.py)：新增 kernel mass conservation 與 spread diagnostics gate 欄位。
4. [analysis/event_provenance_summary.py](../../../analysis/event_provenance_summary.py)：新增 summarize_impact_spreading_kernel、compute_spread_phase_lag、verify_kernel_mass_conservation。
5. [tests/test_personality_rl_runtime.py](../../../tests/test_personality_rl_runtime.py)、[tests/test_w2_episode.py](../../../tests/test_w2_episode.py)、[tests/test_b1_async_dispatch_gate.py](../../../tests/test_b1_async_dispatch_gate.py)：新增 B3 v2 測試。

完整可執行 CLI（pre-A3、主實驗、post-A3）：

~~~bash
./venv/bin/python -m simulation.track_a_protocol_regression --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json --recheck-out-json outputs/track_b_v2/b3_impact_spreading_v2/tb2_b3_pre_a3_recheck_42_101_20260424_summary.json --pytest-target tests/test_w2_episode.py --phase-id tb2_b3 --bridge-id b3_impact_spreading_v2 --bridge-count 1 --anchor-profile-id phase2_pass_locked_v1 --python-bin ./venv/bin/python --summary-json outputs/track_b_v2/b3_impact_spreading_v2/tb2_b3_pre_a3_20260424_summary.json --summary-md outputs/track_b_v2/b3_impact_spreading_v2/tb2_b3_pre_a3_20260424_summary.md

./venv/bin/python -m simulation.b1_async_dispatch_gate --phase-id tb2_b3 --bridge-id b3_impact_spreading_v2 --bridge-count 1 --anchor-profile-id phase2_pass_locked_v1 --smoke-seeds 45,47,49,51,53,55 --gate-seeds 42..101 --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json --world-feedback-mode difficulty_only --event-dispatch-mode sync --event-dispatch-target-rate 0.08 --event-dispatch-fairness-window 2000 --event-dispatch-fairness-tolerance 0.50 --event-trigger-mode entropy_guard --event-trigger-entropy-threshold 0.85 --event-neutralize-payoff --event-reward-mode additive --event-impact-mode spread --event-impact-horizon 7 --event-impact-decay 0.82 --impact-spread-kernel-id hierarchical_v2 --impact-spread-local-mass 0.65 --impact-spread-neighbor-mass 0.35 --impact-spread-neighbor-hop 2 --impact-spread-memory-kernel 3 --smoke-out-json outputs/track_b_v2/b3_impact_spreading_v2/tb2_b3_smoke_s6_20260424_summary.json --gate-out-json outputs/track_b_v2/b3_impact_spreading_v2/tb2_b3_gate60_20260424_summary.json

./venv/bin/python -m simulation.track_a_protocol_regression --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json --recheck-out-json outputs/track_b_v2/b3_impact_spreading_v2/tb2_b3_post_a3_recheck_42_101_20260424_summary.json --pytest-target tests/test_personality_rl_runtime.py --phase-id tb2_b3 --bridge-id b3_impact_spreading_v2 --bridge-count 1 --anchor-profile-id phase2_pass_locked_v1 --python-bin ./venv/bin/python --summary-json outputs/track_b_v2/b3_impact_spreading_v2/tb2_b3_post_a3_20260424_summary.json --summary-md outputs/track_b_v2/b3_impact_spreading_v2/tb2_b3_post_a3_20260424_summary.md
~~~

Gate 契約與 new_L1 判定方式：

- smoke 目標：L3_rate >= 4/6 且 impact_spread_applied=true（6/6）
- gate60 硬規則：L1<=3、Healthy>=42、new_L1=0、fairness_fail_count=0、gate.invariant_overall_pass=true
- 定義：
$$
new\_L1^{(tb2\_b3)} = |\{s \mid healthy_{P0}(s)=1 \land level_{tb2\_b3}(s)=1\}|
$$
- 硬規則：
$$
new\_L1^{(tb2\_b3)}=0
$$
- baseline 母體固定：outputs/phase_coupling_v2/p0_baseline_gate60_summary.json

診斷輸出要求（新增 CSV 欄位與 analysis 函式）：

- 新增 CSV 欄位：impact_kernel_id、impact_kernel_mass_local、impact_kernel_mass_neighbor、impact_spread_radius、impact_spread_delay_mean、impact_spread_alignment、impact_kernel_mass_error
- 新增 analysis 函式：summarize_impact_spreading_kernel、compute_spread_phase_lag、verify_kernel_mass_conservation（放在 [analysis/event_provenance_summary.py](../../../analysis/event_provenance_summary.py)）

風險與 fallback（明確 stop condition）：

- stop condition：gate_new_l1>0
- stop condition：impact_kernel_mass_error>1e-9（任一 gate seed）
- stop condition：impact_spread_applied=false（任一 gate seed）
- stop condition：pre_a3_overall_pass=false 或 post_a3_overall_pass=false
- fallback：關閉 B3 v2 spread kernel，回退 Phase 2 鎖定配置，輸出 tb2_b3_fail_lock.json 與 broke/new_L1 seeds 清單

---

### 組合 sweep：B1+B2

目標：  
在 async dispatch + per-player async update 上疊加 multiplicative_v2 modulation，測試 Layer 1/3 協同增益。

核心程式變更清單（檔案路徑 + 最小改動）：

1. [simulation/personality_rl_runtime.py](../../../simulation/personality_rl_runtime.py)：新增組合 profile b12_combo_v2 與互擾診斷欄位。
2. [simulation/b1_async_dispatch_gate.py](../../../simulation/b1_async_dispatch_gate.py)：新增 b12 組合 gate 診斷輸出。
3. [analysis/event_provenance_summary.py](../../../analysis/event_provenance_summary.py)：新增 evaluate_b1_b2_coupling、verify_combo_invariant_b1_b2。
4. [tests/test_personality_rl_runtime.py](../../../tests/test_personality_rl_runtime.py)、[tests/test_b1_async_dispatch_gate.py](../../../tests/test_b1_async_dispatch_gate.py)：新增 B1+B2 組合測試。

完整可執行 CLI（pre-A3、主實驗、post-A3）：

~~~bash
./venv/bin/python -m simulation.track_a_protocol_regression --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json --recheck-out-json outputs/track_b_v2/b12_combo_v2/tb2_b12_pre_a3_recheck_42_101_20260427_summary.json --pytest-target tests/test_personality_rl_runtime.py --phase-id tb2_b12 --bridge-id b12_combo_v2 --bridge-count 1 --anchor-profile-id phase2_pass_locked_v1 --python-bin ./venv/bin/python --summary-json outputs/track_b_v2/b12_combo_v2/tb2_b12_pre_a3_20260427_summary.json --summary-md outputs/track_b_v2/b12_combo_v2/tb2_b12_pre_a3_20260427_summary.md

./venv/bin/python -m simulation.b1_async_dispatch_gate --phase-id tb2_b12 --bridge-id b12_combo_v2 --bridge-count 1 --anchor-profile-id phase2_pass_locked_v1 --smoke-seeds 45,47,49,51,53,55 --gate-seeds 42..101 --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json --world-feedback-mode difficulty_only --event-dispatch-mode async_poisson --event-dispatch-target-rate 0.08 --event-dispatch-fairness-window 2000 --event-dispatch-fairness-tolerance 0.50 --event-trigger-mode entropy_guard --event-trigger-entropy-threshold 0.85 --event-neutralize-payoff --replicator-update-mode async_per_player --replicator-async-minibatch 32 --replicator-async-jitter 0.15 --event-queue-mode per_player --event-queue-cap 8 --event-queue-drain-rate 1.00 --event-reward-mode multiplicative --event-reward-multiplier-cap 0.20 --event-modulation-mode multiplicative_v2 --event-modulation-gain 0.12 --event-modulation-log-center 0.00 --event-modulation-zero-mean --event-modulation-floor 0.85 --event-modulation-ceiling 1.15 --event-impact-mode instant --smoke-out-json outputs/track_b_v2/b12_combo_v2/tb2_b12_smoke_s6_20260427_summary.json --gate-out-json outputs/track_b_v2/b12_combo_v2/tb2_b12_gate60_20260427_summary.json

./venv/bin/python -m simulation.track_a_protocol_regression --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json --recheck-out-json outputs/track_b_v2/b12_combo_v2/tb2_b12_post_a3_recheck_42_101_20260427_summary.json --pytest-target tests/test_b1_async_dispatch_gate.py --phase-id tb2_b12 --bridge-id b12_combo_v2 --bridge-count 1 --anchor-profile-id phase2_pass_locked_v1 --python-bin ./venv/bin/python --summary-json outputs/track_b_v2/b12_combo_v2/tb2_b12_post_a3_20260427_summary.json --summary-md outputs/track_b_v2/b12_combo_v2/tb2_b12_post_a3_20260427_summary.md
~~~

Gate 契約與 new_L1 判定方式：

- smoke 目標：L3_rate >= 5/6 且組合不變量全通過
- gate60 硬規則：L1<=3、Healthy>=42、new_L1=0、fairness_fail_count=0、gate.invariant_overall_pass=true
- 定義：
$$
new\_L1^{(tb2\_b12)} = |\{s \mid healthy_{P0}(s)=1 \land level_{tb2\_b12}(s)=1\}|
$$
- 硬規則：
$$
new\_L1^{(tb2\_b12)}=0
$$
- baseline 母體固定：outputs/phase_coupling_v2/p0_baseline_gate60_summary.json

診斷輸出要求（新增 CSV 欄位與 analysis 函式）：

- 新增 CSV 欄位：combo_profile_id、b1_async_applied、b2_modulation_applied、async_modulation_coupling_gain、queue_modulation_interference_score
- 新增 analysis 函式：evaluate_b1_b2_coupling、verify_combo_invariant_b1_b2（放在 [analysis/event_provenance_summary.py](../../../analysis/event_provenance_summary.py)）

風險與 fallback（明確 stop condition）：

- stop condition：gate_new_l1>0
- stop condition：queue_modulation_interference_score>0.15（任一 gate seed）
- stop condition：dispatch_fairness_fail_count>0
- stop condition：pre_a3_overall_pass=false 或 post_a3_overall_pass=false
- fallback：關閉 b12 組合，回退到單項 PASS 分支，輸出 tb2_b12_fail_lock.json 與 broke/new_L1 seeds 清單

---

### 組合 sweep：B1+B3

目標：  
在 async per-player 更新上疊加 impact spreading v2，建立跨層 phase lag 傳遞鏈。

核心程式變更清單（檔案路徑 + 最小改動）：

1. [simulation/personality_rl_runtime.py](../../../simulation/personality_rl_runtime.py)：新增 b13 組合 profile 與 async-spread transfer 診斷欄位。
2. [simulation/b1_async_dispatch_gate.py](../../../simulation/b1_async_dispatch_gate.py)：新增 b13 組合 gate 診斷輸出。
3. [analysis/event_provenance_summary.py](../../../analysis/event_provenance_summary.py)：新增 evaluate_b1_b3_phase_transfer、verify_combo_invariant_b1_b3。
4. [tests/test_personality_rl_runtime.py](../../../tests/test_personality_rl_runtime.py)、[tests/test_b1_async_dispatch_gate.py](../../../tests/test_b1_async_dispatch_gate.py)、[tests/test_w2_episode.py](../../../tests/test_w2_episode.py)：新增 B1+B3 測試。

完整可執行 CLI（pre-A3、主實驗、post-A3）：

~~~bash
./venv/bin/python -m simulation.track_a_protocol_regression --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json --recheck-out-json outputs/track_b_v2/b13_combo_v2/tb2_b13_pre_a3_recheck_42_101_20260430_summary.json --pytest-target tests/test_w2_episode.py --phase-id tb2_b13 --bridge-id b13_combo_v2 --bridge-count 1 --anchor-profile-id phase2_pass_locked_v1 --python-bin ./venv/bin/python --summary-json outputs/track_b_v2/b13_combo_v2/tb2_b13_pre_a3_20260430_summary.json --summary-md outputs/track_b_v2/b13_combo_v2/tb2_b13_pre_a3_20260430_summary.md

./venv/bin/python -m simulation.b1_async_dispatch_gate --phase-id tb2_b13 --bridge-id b13_combo_v2 --bridge-count 1 --anchor-profile-id phase2_pass_locked_v1 --smoke-seeds 45,47,49,51,53,55 --gate-seeds 42..101 --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json --world-feedback-mode difficulty_only --event-dispatch-mode async_poisson --event-dispatch-target-rate 0.08 --event-dispatch-fairness-window 2000 --event-dispatch-fairness-tolerance 0.50 --event-trigger-mode entropy_guard --event-trigger-entropy-threshold 0.85 --event-neutralize-payoff --replicator-update-mode async_per_player --replicator-async-minibatch 32 --replicator-async-jitter 0.15 --event-queue-mode per_player --event-queue-cap 8 --event-queue-drain-rate 1.00 --event-reward-mode additive --event-impact-mode spread --event-impact-horizon 7 --event-impact-decay 0.82 --impact-spread-kernel-id hierarchical_v2 --impact-spread-local-mass 0.65 --impact-spread-neighbor-mass 0.35 --impact-spread-neighbor-hop 2 --impact-spread-memory-kernel 3 --smoke-out-json outputs/track_b_v2/b13_combo_v2/tb2_b13_smoke_s6_20260430_summary.json --gate-out-json outputs/track_b_v2/b13_combo_v2/tb2_b13_gate60_20260430_summary.json

./venv/bin/python -m simulation.track_a_protocol_regression --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json --recheck-out-json outputs/track_b_v2/b13_combo_v2/tb2_b13_post_a3_recheck_42_101_20260430_summary.json --pytest-target tests/test_b1_async_dispatch_gate.py --phase-id tb2_b13 --bridge-id b13_combo_v2 --bridge-count 1 --anchor-profile-id phase2_pass_locked_v1 --python-bin ./venv/bin/python --summary-json outputs/track_b_v2/b13_combo_v2/tb2_b13_post_a3_20260430_summary.json --summary-md outputs/track_b_v2/b13_combo_v2/tb2_b13_post_a3_20260430_summary.md
~~~

Gate 契約與 new_L1 判定方式：

- smoke 目標：L3_rate >= 5/6 且 async_spread_phase_transfer>0
- gate60 硬規則：L1<=3、Healthy>=42、new_L1=0、fairness_fail_count=0、gate.invariant_overall_pass=true
- 定義：
$$
new\_L1^{(tb2\_b13)} = |\{s \mid healthy_{P0}(s)=1 \land level_{tb2\_b13}(s)=1\}|
$$
- 硬規則：
$$
new\_L1^{(tb2\_b13)}=0
$$
- baseline 母體固定：outputs/phase_coupling_v2/p0_baseline_gate60_summary.json

診斷輸出要求（新增 CSV 欄位與 analysis 函式）：

- 新增 CSV 欄位：combo_profile_id、b1_async_applied、b3_spread_applied、async_spread_phase_transfer、queue_spread_backlog_ratio
- 新增 analysis 函式：evaluate_b1_b3_phase_transfer、verify_combo_invariant_b1_b3（放在 [analysis/event_provenance_summary.py](../../../analysis/event_provenance_summary.py)）

風險與 fallback（明確 stop condition）：

- stop condition：gate_new_l1>0
- stop condition：async_spread_phase_transfer<=0（任一 gate seed）
- stop condition：queue_spread_backlog_ratio>0.20（任一 gate seed）
- stop condition：pre_a3_overall_pass=false 或 post_a3_overall_pass=false
- fallback：關閉 b13 組合，回退到單項 PASS 分支，輸出 tb2_b13_fail_lock.json 與 broke/new_L1 seeds 清單

---

### 組合 sweep：B1+B2+B3

目標：  
在單一組合橋接下整合 async update、multiplicative modulation、impact spreading 三機制，直接對三層均化瓶頸做架構性對抗並恢復高耦合 Level 3 能力。

核心程式變更清單（檔案路徑 + 最小改動）：

1. [simulation/personality_rl_runtime.py](../../../simulation/personality_rl_runtime.py)：新增 b123_combo_v2 profile、tri-layer diagnostics、bridge_activation_round。
2. [simulation/w3_policy.py](../../../simulation/w3_policy.py)、[simulation/w3_pulse.py](../../../simulation/w3_pulse.py)、[simulation/w3_stackelberg.py](../../../simulation/w3_stackelberg.py)：新增 bridge metadata 輸出，不更改既有判決規則。
3. [simulation/b1_async_dispatch_gate.py](../../../simulation/b1_async_dispatch_gate.py)：新增 tri-layer gate 彙總欄位。
4. [analysis/event_provenance_summary.py](../../../analysis/event_provenance_summary.py)：新增 evaluate_tri_layer_rotation_gain、verify_combo_invariant_b123、localize_track_b_first_break。
5. [tests/test_personality_rl_runtime.py](../../../tests/test_personality_rl_runtime.py)、[tests/test_w3_policy.py](../../../tests/test_w3_policy.py)、[tests/test_w3_pulse.py](../../../tests/test_w3_pulse.py)、[tests/test_w3_stackelberg.py](../../../tests/test_w3_stackelberg.py)、[tests/test_b1_async_dispatch_gate.py](../../../tests/test_b1_async_dispatch_gate.py)：新增 B1+B2+B3 測試。

完整可執行 CLI（pre-A3、主實驗、post-A3）：

~~~bash
./venv/bin/python -m simulation.track_a_protocol_regression --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json --recheck-out-json outputs/track_b_v2/b123_combo_v2/tb2_b123_pre_a3_recheck_42_101_20260503_summary.json --pytest-target tests/test_w3_policy.py --phase-id tb2_b123 --bridge-id b123_combo_v2 --bridge-count 1 --anchor-profile-id phase2_pass_locked_v1 --python-bin ./venv/bin/python --summary-json outputs/track_b_v2/b123_combo_v2/tb2_b123_pre_a3_20260503_summary.json --summary-md outputs/track_b_v2/b123_combo_v2/tb2_b123_pre_a3_20260503_summary.md

./venv/bin/python -m simulation.b1_async_dispatch_gate --phase-id tb2_b123 --bridge-id b123_combo_v2 --bridge-count 1 --anchor-profile-id phase2_pass_locked_v1 --smoke-seeds 45,47,49,51,53,55 --gate-seeds 42..101 --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json --world-feedback-mode difficulty_only --event-dispatch-mode async_poisson --event-dispatch-target-rate 0.08 --event-dispatch-fairness-window 2000 --event-dispatch-fairness-tolerance 0.50 --event-trigger-mode entropy_guard --event-trigger-entropy-threshold 0.85 --event-neutralize-payoff --replicator-update-mode async_per_player --replicator-async-minibatch 32 --replicator-async-jitter 0.15 --event-queue-mode per_player --event-queue-cap 8 --event-queue-drain-rate 1.00 --event-reward-mode multiplicative --event-reward-multiplier-cap 0.20 --event-modulation-mode multiplicative_v2 --event-modulation-gain 0.12 --event-modulation-log-center 0.00 --event-modulation-zero-mean --event-modulation-floor 0.85 --event-modulation-ceiling 1.15 --event-impact-mode spread --event-impact-horizon 7 --event-impact-decay 0.82 --impact-spread-kernel-id hierarchical_v2 --impact-spread-local-mass 0.65 --impact-spread-neighbor-mass 0.35 --impact-spread-neighbor-hop 2 --impact-spread-memory-kernel 3 --smoke-out-json outputs/track_b_v2/b123_combo_v2/tb2_b123_smoke_s6_20260503_summary.json --gate-out-json outputs/track_b_v2/b123_combo_v2/tb2_b123_gate60_20260503_summary.json

./venv/bin/python -m simulation.track_a_protocol_regression --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json --recheck-out-json outputs/track_b_v2/b123_combo_v2/tb2_b123_post_a3_recheck_42_101_20260503_summary.json --pytest-target tests/test_b1_async_dispatch_gate.py --phase-id tb2_b123 --bridge-id b123_combo_v2 --bridge-count 1 --anchor-profile-id phase2_pass_locked_v1 --python-bin ./venv/bin/python --summary-json outputs/track_b_v2/b123_combo_v2/tb2_b123_post_a3_20260503_summary.json --summary-md outputs/track_b_v2/b123_combo_v2/tb2_b123_post_a3_20260503_summary.md
~~~

Gate 契約與 new_L1 判定方式：

- smoke 目標：L3_rate >= 5/6 且 tri_layer_rotation_gain>0
- gate60 硬規則：L1<=3、Healthy>=42、new_L1=0、fairness_fail_count=0、gate.invariant_overall_pass=true
- 定義：
$$
new\_L1^{(tb2\_b123)} = |\{s \mid healthy_{P0}(s)=1 \land level_{tb2\_b123}(s)=1\}|
$$
- 硬規則：
$$
new\_L1^{(tb2\_b123)}=0
$$
- baseline 母體固定：outputs/phase_coupling_v2/p0_baseline_gate60_summary.json

診斷輸出要求（新增 CSV 欄位與 analysis 函式）：

- 新增 CSV 欄位：combo_profile_id、b1_async_applied、b2_modulation_applied、b3_spread_applied、tri_layer_rotation_gain、tri_layer_stability_margin、bridge_activation_round
- 新增 analysis 函式：evaluate_tri_layer_rotation_gain、verify_combo_invariant_b123、localize_track_b_first_break（放在 [analysis/event_provenance_summary.py](../../../analysis/event_provenance_summary.py)）

風險與 fallback（明確 stop condition）：

- stop condition：gate_new_l1>0
- stop condition：tri_layer_rotation_gain<=0（任一 gate seed）
- stop condition：tri_layer_stability_margin<0.02（任一 gate seed）
- stop condition：pre_a3_overall_pass=false 或 post_a3_overall_pass=false
- fallback：立即判定 b123_combo_v2 為 Track B first-break bridge，輸出 tb2_b123_first_break_localization.json 與簽核草案 tb2_b123_first_break_decision_draft.md，停止同族微調

---

## 工程實施檢查清單（SDD 契約）

1. 規格先行：所有行為契約變更先更新 [SDD.md](../../SDD.md) 與 [high_coupling_failure_diagnosis_spec.md](../high_coupling_failure_diagnosis_spec.md)。
2. 分層不變條件：analysis 不 import simulation；evolution 不做 I/O；CSV schema 僅由 simulation 維護。
3. protocol lock：沿用 [phase_based_coupling_blueprint_v2.md](../phase_based_coupling_blueprint_v2.md) 的 gate 門檻，不調整 amplitude/corr/eta/stage3_method。
4. baseline 鎖定：new_L1 比對母體固定為 outputs/phase_coupling_v2/p0_baseline_gate60_summary.json。
5. memory cap 鎖定：memory_cap_sentences=20，全部子任務不得變更。
6. smoke/gate 執行鎖定：一次 smoke 僅單一機制或單一組合；gate 固定 42..101。
7. 必跑回歸：
~~~bash
./venv/bin/python -m pytest -q tests/test_personality_rl_runtime.py tests/test_b1_async_dispatch_gate.py tests/test_event_provenance_summary.py tests/test_world_state_w1.py tests/test_w2_episode.py tests/test_w3_stackelberg.py tests/test_w3_policy.py tests/test_w3_pulse.py
~~~
8. outputs 命名規則：
   - outputs/track_b_v2/{task_id}/{phase_id}_{bridge_id}_{pre_a3|smoke_s6|gate60|post_a3}_{yyyymmdd}_summary.json
   - outputs/track_b_v2/{task_id}/{phase_id}_{bridge_id}_{yyyymmdd}_decision.md
   - outputs/track_b_v2/{task_id}/{phase_id}_{bridge_id}_{yyyymmdd}_lock_status.json
9. provenance 必填欄位：
   - phase_id、bridge_id、bridge_count、anchor_profile_id、memory_cap_sentences、cycle_gate_profile、git_commit、config_hash
   - world_feedback_mode、dispatch_mode、dispatch_target_rate、event_reward_mode、event_impact_mode
   - new_l1_seeds、broke_seeds、gate_overall_pass、pre_a3_overall_pass、post_a3_overall_pass
10. Stage-1 Tier 治理鎖定（R14 addendum）：
  - 同一輪 Stage-1 必須 run-level freeze：全候選共用同一組 Tier（不得中途更新）。
  - failure set 來源鎖定為「上一輪完整 gate60 / Stage-3」；不得使用本輪 Stage-1 中間評估（禁止 look-ahead bias）。
  - Tier-2a hard cap：<=4，且當輪 NewL1 至少保留 1 席（取當輪 NewL1 中 I_t 最高者）。
  - cooldown：seed 連續 3 輪在 Tier-2a，下一輪強制 Tier-2b 1 輪。
  - Promotion：每輪最多 1 顆，Tier-1 cap=5，若滿員先 demote 非核心 seed；當前主線核心 seed（86,61）不可 demote。
  - Tier-2b 為 memory reservoir，必須保留於 ranking log 與跨輪頻率計算。

---

## 60-seed Gate 與 A3 整合協議

### 固定流程（每子任務皆相同）

1. pre-A3：執行 simulation.track_a_protocol_regression，必須 overall_pass=true。
2. 主實驗：執行 simulation.b1_async_dispatch_gate，一次產出 smoke_s6 與 gate60。
3. post-A3：再次執行 simulation.track_a_protocol_regression，必須 overall_pass=true。

### Gate 判定

- 全局硬規則：
  - gate_l1 <= 3
  - gate_healthy >= 42
  - gate_new_l1 = 0
  - gate_fairness_fail_count = 0
  - gate.invariant_overall_pass = true
  - pre_a3_overall_pass = true
  - post_a3_overall_pass = true

- new_L1 統一定義：
$$
new\_L1^{(p)} = |\{s \mid healthy_{P0}(s)=1 \land level_{Pp}(s)=1\}|
$$

- new_L1 硬規則：
$$
new\_L1^{(p)} = 0
$$

- 任一任務若 new_L1>0：
  - 立即 FAIL
  - 輸出 new_l1_seeds、broke_seeds、minimal_new_l1_seed 與 provenance
  - 停止前進到下一子任務

### Stage-1 Tier 更新協議（R14 addendum）

固定順序（每輪）：

1. 更新 F_acc（最近 3 輪視窗）。
2. 計算 I_t(s)。
3. 先做 Promotion 判定。
4. Tier-1 若已滿，先 demote 再升級。
5. 組 Tier-2a（含當輪 NewL1 保留位）。
6. 套用 cooldown。
7. 剩餘集合進 Tier-2b。

最小分數規則：

$$
I_t(s)=4\cdot\mathbf{1}[s\in NewL1_t]+2\cdot freq_{L1}^{(t)}(s)+w_b\cdot freq_{broke}^{(t)}(s),\quad w_b\in[0.5,1.0]
$$

備註：$w_b$ 預設固定；僅可在回合邊界依簽核規則調整，且必須寫入 provenance。

### 子任務 phase_id / bridge_id 對照

| 子任務 | phase_id | bridge_id |
|---|---|---|
| B1 Async Dispatch v2 | tb2_b1 | b1_async_dispatch_v2 |
| B2 Multiplicative Modulation v2 | tb2_b2 | b2_multiplicative_modulation_v2 |
| B3 Impact Spreading v2 | tb2_b3 | b3_impact_spreading_v2 |
| 組合 sweep：B1+B2 | tb2_b12 | b12_combo_v2 |
| 組合 sweep：B1+B3 | tb2_b13 | b13_combo_v2 |
| 組合 sweep：B1+B2+B3 | tb2_b123 | b123_combo_v2 |

---

## 里程碑時程與簽核

### 里程碑時程（自 2026-04-20 起）

| 里程碑 | 日期區間 | 任務 | 輸出要求 |
|---|---|---|---|
| M0 | 2026-04-20 | B1 Async Dispatch v2 | pre-A3、smoke、gate60、post-A3、decision、lock |
| M1 | 2026-04-22 | B2 Multiplicative Modulation v2 | pre-A3、smoke、gate60、post-A3、decision、lock |
| M2 | 2026-04-24 | B3 Impact Spreading v2 | pre-A3、smoke、gate60、post-A3、decision、lock |
| M3 | 2026-04-27 | 組合 sweep：B1+B2 | pre-A3、smoke、gate60、post-A3、decision、lock |
| M4 | 2026-04-30 | 組合 sweep：B1+B3 | pre-A3、smoke、gate60、post-A3、decision、lock |
| M5 | 2026-05-03 | 組合 sweep：B1+B2+B3 | pre-A3、smoke、gate60、post-A3、decision、lock、first-break（若 FAIL） |
| M6 | 2026-05-05 | Track B v2.1 結案判定 | 綜合 gate、new_L1、A3、provenance 完整性簽核 |

### 簽核欄位

- Blueprint 版本：v2.1  
- 文件日期：2026-04-18  
- 啟動日期：2026-04-20  
- Research Lead：＿＿＿＿＿＿＿＿  
- Runtime Owner：＿＿＿＿＿＿＿＿  
- Analysis Owner：＿＿＿＿＿＿＿＿  
- QA Owner：＿＿＿＿＿＿＿＿  
- 簽核狀態：待簽核（符合啟動條件）
