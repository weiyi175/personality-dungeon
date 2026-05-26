# Phase-based Coupling 未來實驗規劃大綱（Blueprint v2）

版本：2.0  
日期：2026-04-18  
適用專案：Personality Dungeon（SDD-driven）

---

## 1. 執行摘要（Executive Summary）

高耦合世界的核心失敗不是單一機制錯誤，而是三層均化（Sampling noise → Popularity averaging → Synchronized replicator）對旋轉訊號的乘法耗散；在 sampled discrete synchronous 路徑下，可將有效旋轉振幅近似為

$$
A_{eff}(t)=A_0\cdot\alpha_1\cdot\alpha_2\cdot e^{-\lambda t},\quad \lambda>0
$$

因此只做局部 patch 幾乎必然失敗；必須採用「每 phase 只增加 1 條 bridge」的 Phase-based Coupling，才能在可回歸條件下精準定位第一個破壞 L3 的橋接點。

### 里程碑時程（Phase 0–3）

| Phase | 起迄日期 | seeds（pilot + gate） | 目標 | 預期 L3_rate |
|---|---|---|---|---|
| Phase 0 | 2026-04-20 至 2026-04-24 | pilot 6 + gate 60 | 鎖定低耦合基線 | pilot 6/6 |
| Phase 1 | 2026-04-25 至 2026-04-27 | pilot 6 + gate 60 | read-only world state | pilot ≥5/6 |
| Phase 2 | 2026-04-28 至 2026-05-02 | pilot 6 + gate 60 | difficulty modulation（不改 payoff） | pilot ≥4/6 |
| Phase 3 | 2026-05-03 至 2026-05-07 | pilot 6 + gate 60 | 完整高耦合 | 預期 FAIL（0/6~3/6） |

---

## 2. 虛擬錨點保留與破壞風險矩陣（5x4 表格）

說明：每列同時標註「若破壞先導致哪一層均化失效」。

| 虛擬錨點（破壞先失效層） | Phase 0 | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|---|
| 固定事件模板（Layer 2） | ✅ | ✅ | ⚠ | ❌ |
| 固定 risk/reward formula（Layer 3） | ✅ | ✅ | ✅（明確不改 payoff） | ❌ |
| bounded memory cap 20-sentence（Layer 2） | ✅ | ✅ | ✅ | ✅ |
| control baseline（Layer 1 比對） | ✅ | ✅ | ✅ | ⚠ |
| 單一 cycle_level gate（三層共同診斷） | ✅ | ✅ | ✅ | ✅ |
| 單一人格表示 12D（Layer 1） | ✅ | ✅ | ✅ | ⚠ |

---

## 3. Phase 0–3 詳細實驗設計（每 Phase 獨立一節）

### Phase 0：Low-Coupling Baseline Lock

Phase X 目標：鎖定可重現的低耦合 L3 基線，作為後續 new_L1 的唯一對照母體。

保留的虛擬錨點：
- 固定事件模板
- 固定 risk/reward formula
- bounded memory cap（20-sentence）
- control baseline
- 單一 cycle_level gate
- 單一人格表示（12D）

新增/修改的 bridge（不得超過 1 條）：
- 無（bridge_count=0）

核心程式變更清單（檔案路徑 + 最小改動描述）：
- simulation/b1_async_dispatch_gate.py：新增 phase_id、bridge_id、bridge_count 輸出欄位（不改 gate 判定邏輯）
- simulation/track_a_protocol_regression.py：新增 phase-run metadata 寫入 summary
- tests/test_b1_async_dispatch_gate.py：新增 metadata 輸出契約測試

CLI 指令範例（完整可執行）：

~~~bash
./venv/bin/python -m simulation.p21_asymmetric_alpha \
  --configs mild_cw \
  --seeds 45,47,49,51,53,55 \
  --rounds 12000 --burn-in 4000 --tail 4000 \
  --out-root outputs/phase_coupling_v2/p0_lowcoupling \
  --summary-tsv outputs/phase_coupling_v2/p0_lowcoupling_summary.tsv \
  --combined-tsv outputs/phase_coupling_v2/p0_lowcoupling_combined.tsv \
  --decision-md outputs/phase_coupling_v2/p0_lowcoupling_decision.md
~~~

~~~bash
./venv/bin/python -m simulation.pers_cal_baseline_gate60 \
  --seeds 42..101 \
  --out-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json
~~~

~~~bash
./venv/bin/python -m simulation.track_a_protocol_regression \
  --baseline-summary-json outputs/pers_cal_baseline_gate60_summary.json \
  --recheck-out-json outputs/phase_coupling_v2/p0_a3_recheck_42_101_summary.json \
  --summary-json outputs/phase_coupling_v2/p0_a3_protocol_regression_summary.json \
  --summary-md outputs/phase_coupling_v2/p0_a3_protocol_regression_summary.md
~~~

驗證契約（L3_rate PASS/FAIL、60-seed Gate、new_L1）：
- PASS：pilot L3_rate = 6/6，且 gate60 同時滿足 L1<=3、Healthy>=42、new_L1=0
- FAIL：pilot < 6/6，或 gate60 任一硬規則失敗

診斷輸出要求（CSV 欄位 / analysis 函數）：
- 新增 seed summary 欄位：phase_id、bridge_id、bridge_count、anchor_profile_id
- analysis/event_provenance_summary.py：新增 phase-level gate 聚合輸出（僅讀 CSV/JSON）

風險與 fallback：
- 風險：基線不穩導致 Phase 1 無效
- fallback：停止後續 phase，重建 p0 基線（含 bit-exact 對照）

---

### Phase 1：Read-only World State（最小破壞）

Phase X 目標：只引入 world_state 可讀取與可記錄，不允許它改變策略抽樣、reward、risk、policy。

保留的虛擬錨點：
- 固定事件模板
- 固定 risk/reward formula
- bounded memory cap（20-sentence）
- control baseline
- 單一 cycle_level gate
- 單一人格表示（12D）

新增/修改的 bridge（不得超過 1 條）：
- Bridge-1：world_state_read_only_v1

核心程式變更清單（檔案路徑 + 最小改動描述）：
- simulation/personality_rl_runtime.py：新增 world_feedback_mode=read_only（只觀測）
- simulation/world_runtime_w1.py：新增 read-only adapter，不調整 event multipliers
- analysis/event_provenance_summary.py：新增 readonly_leak_score 計算
- tests/test_personality_rl_runtime.py：新增 read-only 不改動力學回歸測試
- tests/test_world_state_w1.py：新增 read-only 退化路徑測試

CLI 指令範例（完整可執行）：

~~~bash
./venv/bin/python -m simulation.b1_async_dispatch_gate \
  --smoke-seeds 45,47,49,51,53,55 \
  --gate-seeds 42..101 \
  --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json \
  --event-dispatch-mode sync \
  --event-dispatch-target-rate 0.00 \
  --event-dispatch-fairness-window 2000 \
  --event-dispatch-fairness-tolerance 0.50 \
  --event-neutralize-payoff \
  --event-trigger-mode entropy_guard \
  --event-trigger-entropy-threshold 0.85 \
  --smoke-out-json outputs/phase_coupling_v2/p1_readonly_smoke_s6_summary.json \
  --gate-out-json outputs/phase_coupling_v2/p1_readonly_gate60_summary.json
~~~

備註：Phase 1 需要先在 simulation/personality_rl_runtime.py 落地 world_feedback_mode=read_only，否則無法做 read-only 漏訊診斷。

驗證契約（L3_rate PASS/FAIL、60-seed Gate、new_L1）：
- PASS：pilot L3_rate >= 5/6，且 gate60 滿足 L1<=3、Healthy>=42、new_L1=0
- FAIL：pilot L3_rate <= 4/6，或 gate60 任一硬規則失敗
- 額外硬規則：readonly_leak_score <= 1e-6

診斷輸出要求（CSV 欄位 / analysis 函數）：
- 新增 CSV 欄位：world_state_scarcity、world_state_threat、world_state_noise、world_state_intel、world_readonly_applied、readonly_leak_score
- analysis 函數：compute_world_readonly_leak

風險與 fallback：
- 風險：只讀介面仍導致隱性耦合（污染）
- fallback：立即停在 Phase 1，Bridge-1 標記為 first_break

---

### Phase 2：Difficulty Modulation（不改 payoff）

Phase X 目標：僅允許 world_state 調整事件難度權重，不改 payoff 幾何（a,b,cross 固定）。

保留的虛擬錨點：
- 固定 risk/reward formula（嚴格不改 payoff）
- bounded memory cap（20-sentence）
- control baseline
- 單一 cycle_level gate
- 單一人格表示（12D）

新增/修改的 bridge（不得超過 1 條）：
- Bridge-2：world_to_event_difficulty_v1

核心程式變更清單（檔案路徑 + 最小改動描述）：
- simulation/personality_rl_runtime.py：新增 world_feedback_mode=difficulty_only
- dungeon/event_loader.py：新增 bounded difficulty multiplier hook（只改 event sampling 權重）
- analysis/event_provenance_summary.py：新增 payoff_static_pass 驗證
- tests/test_personality_rl_runtime.py：新增 payoff 不變性測試

CLI 指令範例（完整可執行）：

~~~bash
./venv/bin/python -m simulation.b1_async_dispatch_gate \
  --smoke-seeds 45,47,49,51,53,55 \
  --gate-seeds 42..101 \
  --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json \
  --event-dispatch-mode sync \
  --event-dispatch-target-rate 0.00 \
  --event-dispatch-fairness-window 2000 \
  --event-dispatch-fairness-tolerance 0.50 \
  --event-neutralize-payoff \
  --event-trigger-mode entropy_guard \
  --event-trigger-entropy-threshold 0.85 \
  --smoke-out-json outputs/phase_coupling_v2/p2_difficulty_smoke_s6_summary.json \
  --gate-out-json outputs/phase_coupling_v2/p2_difficulty_gate60_summary.json
~~~

備註：Phase 2 需要先落地 world_feedback_mode=difficulty_only 與 payoff_static_pass 診斷欄位。

驗證契約（L3_rate PASS/FAIL、60-seed Gate、new_L1）：
- PASS：pilot L3_rate >= 4/6，且 gate60 滿足 L1<=3、Healthy>=42、new_L1=0
- FAIL：pilot L3_rate <= 3/6，或 gate60 任一硬規則失敗
- 額外硬規則：payoff_static_pass = true（全 seed）

診斷輸出要求（CSV 欄位 / analysis 函數）：
- 新增 CSV 欄位：difficulty_index、event_difficulty_multiplier、difficulty_modulation_applied、payoff_static_pass
- analysis 函數：verify_payoff_static、summarize_difficulty_modulation

風險與 fallback：
- 風險：difficulty 調制意外改動 payoff 或引入 new_L1
- fallback：停在 Phase 2，關閉 Bridge-2，回退至 Phase 1 PASS 狀態

---

### Phase 3：完整高耦合（Full High-Coupling）

Phase X 目標：在前兩個 phase 皆 PASS 後，開啟完整高耦合 bundle，驗證是否仍能保留 L3。

保留的虛擬錨點：
- bounded memory cap（20-sentence，硬性）
- 單一 cycle_level gate（硬性）

新增/修改的 bridge（不得超過 1 條）：
- Bridge-3：high_coupling_bundle_v1（單一開關）

核心程式變更清單（檔案路徑 + 最小改動描述）：
- simulation/personality_rl_runtime.py：新增 high_coupling_bundle_v1（統一接入）
- simulation/w2_episode.py：輸出 carryover 指標並保留 memory cap
- simulation/w3_stackelberg.py：加入 bridge metadata（不改既有判決）
- simulation/w3_policy.py：加入 bridge metadata（不改既有判決）
- simulation/w3_pulse.py：加入 bridge metadata（不改既有判決）
- tests/test_w2_episode.py、tests/test_w3_stackelberg.py、tests/test_w3_policy.py、tests/test_w3_pulse.py：新增 bundle smoke 測試

CLI 指令範例（完整可執行）：

~~~bash
./venv/bin/python -m simulation.b1_async_dispatch_gate \
  --smoke-seeds 45,47,49,51,53,55 \
  --gate-seeds 42..101 \
  --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json \
  --event-dispatch-mode async_poisson \
  --event-dispatch-target-rate 0.08 \
  --event-dispatch-fairness-window 2000 \
  --event-dispatch-fairness-tolerance 0.50 \
  --event-neutralize-payoff \
  --event-trigger-mode entropy_guard \
  --event-trigger-entropy-threshold 0.85 \
  --smoke-out-json outputs/phase_coupling_v2/p3_full_smoke_s6_summary.json \
  --gate-out-json outputs/phase_coupling_v2/p3_full_gate60_summary.json
~~~

驗證契約（L3_rate PASS/FAIL、60-seed Gate、new_L1）：
- PASS：pilot L3_rate >= 3/6 且 gate60 全通過
- FAIL：pilot < 3/6，或 gate60 任一硬規則失敗
- Gate60 硬規則：L1<=3、Healthy>=42、new_L1=0、fairness_fail_count=0、gate.invariant_overall_pass=true

診斷輸出要求（CSV 欄位 / analysis 函數）：
- 新增 CSV 欄位：policy_activation_share、policy_switches、pulse_count、first_pulse_round、carryover_health_mean、bridge_activation_round
- analysis 函數：localize_first_break_bridge

風險與 fallback：
- 風險：橋接一次開滿導致無法定位破壞點
- fallback：若 FAIL，直接判定第一破壞點落在 Bridge-3，不再做同族局部微調

---

## 4. 工程實施檢查清單（SDD 契約）

### 4.1 關鍵不變條件（5 條）

- 一次只增加 1 條 bridge（bridge_count 必須可由 provenance 驗證）
- bounded memory cap 固定 20-sentence（全 phase 不得放寬）
- cycle_level gate 固定（amplitude_threshold=0.02、corr_threshold=0.09、eta=0.55、stage3_method=turning）
- analysis 不得 import simulation；evolution 不做 I/O；CSV schema 由 simulation 層維護
- 不得違反既有 H-series、W-series、B-series protocol lock 與 closure 決策

### 4.2 回歸測試清單（bit-exact 要求）

~~~bash
./venv/bin/python -m simulation.track_a_protocol_regression \
  --summary-json outputs/phase_coupling_v2/a3_protocol_regression_summary.json \
  --summary-md outputs/phase_coupling_v2/a3_protocol_regression_summary.md
~~~

~~~bash
./venv/bin/pytest -q \
  tests/test_personality_rl_runtime.py \
  tests/test_b1_async_dispatch_gate.py \
  tests/test_world_state_w1.py \
  tests/test_w2_episode.py \
  tests/test_w3_stackelberg.py \
  tests/test_w3_policy.py \
  tests/test_w3_pulse.py
~~~

### 4.3 outputs 命名規則與 provenance

- 命名規則：outputs/phase_coupling_v2/p{phase}_{bridge}_{smoke|gate60}_{yyyymmdd}_summary.json
- 同批次必有：smoke summary + gate summary + decision md + a3 summary
- provenance 必填欄位：phase_id、bridge_id、bridge_count、anchor_profile_id、memory_cap_sentences、cycle_gate_profile、git_commit、config_hash

---

## 5. 60-seed Gate 與 A3 整合協議

### 5.1 每個 Phase 結束後必跑的 protocol_regression

| Phase | A3 protocol_regression | Gate 指令 |
|---|---|---|
| Phase 0 | simulation.track_a_protocol_regression | simulation.pers_cal_baseline_gate60 + simulation.b1_async_dispatch_gate |
| Phase 1 | simulation.track_a_protocol_regression | simulation.b1_async_dispatch_gate（read-only profile） |
| Phase 2 | simulation.track_a_protocol_regression | simulation.b1_async_dispatch_gate（difficulty-only profile） |
| Phase 3 | simulation.track_a_protocol_regression | simulation.b1_async_dispatch_gate（full profile） |

### 5.2 new_L1 = 0 硬規則落實

定義：

$$
new\_L1^{(p)} = |\{s \mid healthy_{P0}(s)=1 \land level_{Pp}(s)=1\}|
$$

硬規則：

$$
new\_L1^{(p)} = 0
$$

執行約束：
- Phase 1–3 一律以同一份 p0 baseline summary 作 new_L1 比對母體
- 任一 phase 若 new_L1 > 0，立即 FAIL，不得前進下一 phase
- FAIL 必須輸出 broke seeds 與對應 provenance，供 A3 與論文回溯

---

## 6. 簽核與下一里程碑

### 6.1 藍圖資訊與簽核欄位

- 藍圖版本：Blueprint v2.0
- 文件日期：2026-04-18
- 啟動日期：2026-04-25（Phase 1）
- 簽核欄位：Research Lead、Runtime Owner、Analysis Owner、QA Owner

### 6.2 2026-04-25 Phase 1 啟動完整執行指令

~~~bash
./venv/bin/python -m simulation.track_a_protocol_regression \
  --summary-json outputs/phase_coupling_v2/p1_pre_a3_summary.json \
  --summary-md outputs/phase_coupling_v2/p1_pre_a3_summary.md

./venv/bin/python -m simulation.b1_async_dispatch_gate \
  --smoke-seeds 45,47,49,51,53,55 \
  --gate-seeds 42..101 \
  --baseline-summary-json outputs/phase_coupling_v2/p0_baseline_gate60_summary.json \
  --event-dispatch-mode sync \
  --event-dispatch-target-rate 0.00 \
  --event-dispatch-fairness-window 2000 \
  --event-dispatch-fairness-tolerance 0.50 \
  --event-neutralize-payoff \
  --event-trigger-mode entropy_guard \
  --event-trigger-entropy-threshold 0.85 \
  --smoke-out-json outputs/phase_coupling_v2/p1_readonly_smoke_s6_summary.json \
  --gate-out-json outputs/phase_coupling_v2/p1_readonly_gate60_summary.json

./venv/bin/python -m simulation.track_a_protocol_regression \
  --summary-json outputs/phase_coupling_v2/p1_post_a3_summary.json \
  --summary-md outputs/phase_coupling_v2/p1_post_a3_summary.md
~~~

---

簽核狀態：待簽核（可啟動 Phase 1 實作）
