# Personality Dungeon World Framework v1

> **9D Enneagram 版本（2026-05-26 重建）**
> 舊版本（12D）已封存至 `SDD_12D_備份/docs/personality_dungeon_v1/00_world_framework_v1.md`。
> 本文件以 `03_personality_projection_v1.py::PRIMARY_GROUPS` 為正典定義。

## Core Premise

The player does not directly control a hero. The player iteratively shapes a personality-bearing agent through one-line testament prompts left between lives.

Each life is one sampled trajectory through a stochastic dungeon pressure environment:

`birth -> event sequence -> autonomous decisions -> risk accumulation -> death -> testament -> personality update -> next life`

## System Layers

### 1. Personality Layer

- State: `P in [-1, 1]^9`（9D Enneagram，三組各 3 維）
- Meaning: stable behavioral tendencies, not combat stats
- Update source: one natural-language sentence after each death

**正典維度定義**（`DIMENSIONS` in `03_personality_projection_v1.py`）：

| 組別 | 英文名 | 維度 |
|------|--------|------|
| 擴張組 The Drivers | aggressive | `impulsiveness`, `assertiveness`, `optimism` |
| 防禦組 The Stabilizers | defensive | `risk_aversion`, `suspicion`, `endurance` |
| 擾動組 The Explorers | balanced | `randomness`, `stability_seeking`, `curiosity` |

**FORBIDDEN 12D keys**（已廢棄，任何程式碼不得使用）：
`greed`, `ambition`, `caution`, `fearfulness`, `patience`, `persistence`

### 2. Strategy Layer

- The 9D personality vector is projected into three macro-strategies via `project_to_simplex()`:
  - **Aggressive**（Drivers：impulsiveness + assertiveness + optimism 平均分）
  - **Defensive**（Stabilizers：risk_aversion + suspicion + endurance 平均分）
  - **Balanced**（Explorers：randomness + stability_seeking + curiosity 平均分）
- Softmax over group scores → `(p_aggressive, p_defensive, p_balanced)` ∈ 2-simplex
- This keeps the world interpretable and connects directly to the repo's existing 3-strategy evolutionary backbone.

### 3. Event Layer

- Dungeon content is modeled as a sequence of events, not a tile map.
- Event families（`02_event_templates_smoke_v1.json`）：
  - Threat
  - Resource
  - Uncertainty
  - Navigation
  - Internal
- Event personality weights：9-element array，對應 `_PERSONALITY_KEYS_ORDERED` 順序
  （`impulsiveness, assertiveness, optimism, risk_aversion, suspicion, endurance, randomness, stability_seeking, curiosity`）

### 4. Evolution Layer

- Agent behavior produces a time series of strategy proportions.
- Evolution pressure follows the repo's lagged cyclic framing.
- The existing paper result gives a usable anchor: non-monotone finite-size transition structure is expected rather than pathological.

### 5. Meta Layer

- Little Dragon is the world-pressure engine.
- It observes global distributions and emits adaptive dungeons that counter dominant strategies.

## Life and Death

Death is not HP depletion. Death is fate collapse driven by accumulated risk.

State update:

`risk_{t+1} = risk_t + base_risk(event, action) + personality_risk(P, action, event) + world_drift`

Death occurs when `risk > threshold`.

## Natural Language Intervention

The player can only leave one sentence between runs.

Update pipeline:

1. Parse sentence into a sparse personality delta（9D key 空間）。
2. Clip delta magnitude。
3. Apply bounded update to `P`。
4. Store sentence in memory bank with a maximum of 20 entries。

## Memory Constraint

- Maximum active testament count: 20
- If a new sentence is added beyond the limit, one existing sentence must be removed.
- This turns prompt history into an evolutionary bottleneck rather than free accumulation.

## Research Value

This framework supports:

- Human-in-the-loop policy shaping
- language-conditioned agent design
- Enneagram-grounded personality dynamics under evolutionary pressure

---

*重建依據：`03_personality_projection_v1.py` PRIMARY_GROUPS（正典），`SDD.md §5`（架構不變條件），`SDD_12D_備份/` 封存版本。*
