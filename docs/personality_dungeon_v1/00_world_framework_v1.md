# Personality Dungeon World Framework v1

## Core Premise

The player does not directly control a hero. The player iteratively shapes a personality-bearing agent through one-line testament prompts left between lives.

Each life is one sampled trajectory through a stochastic dungeon pressure environment:

`birth -> event sequence -> autonomous decisions -> risk accumulation -> death -> testament -> personality update -> next life`

## System Layers

### 1. Personality Layer

- State: `P in [-1, 1]^12`
- Meaning: stable behavioral tendencies, not combat stats
- Update source: one natural-language sentence after each death

### 2. Strategy Layer

- The 12D personality vector is projected into three macro-strategies:
  - Aggressive
  - Defensive
  - Balanced
- This keeps the world interpretable and connects directly to the repo's existing 3-strategy evolutionary backbone.

### 3. Event Layer

- Dungeon content is modeled as a sequence of events, not a tile map.
- Event families:
  - Threat
  - Resource
  - Uncertainty
  - Navigation
  - Internal

### 4. Evolution Layer

- Agent behavior produces a time series of strategy proportions.
- Evolution pressure follows the repo's lagged cyclic framing.
- The existing paper result gives a usable anchor: non-monotone finite-size transition structure is expected rather than pathological.

### 5. Meta Layer

- Little Dragon is the world-pressure engine.
- It observes global distributions and emits adaptive dungeons that counter dominant strategies.

## Life and Death

Death is not HP depletion. Death is fate collapse driven by accumulated risk.

Suggested state update:

`risk_{t+1} = risk_t + base_risk(event, action) + personality_risk(P, action, event) + world_drift`

Death occurs when `risk > threshold`.

## Natural Language Intervention

The player can only leave one sentence between runs.

Suggested update pipeline:

1. Parse sentence into a sparse personality delta.
2. Clip delta magnitude.
3. Apply bounded update to `P`.
4. Store sentence in memory bank with a maximum of 20 entries.

## Memory Constraint

- Maximum active testament count: 20
- If a new sentence is added beyond the limit, one existing sentence must be removed.
- This turns prompt history into an evolutionary bottleneck rather than free accumulation.

## Research Value

This framework supports:

- Human-in-the-loop policy shaping
- language-conditioned agent design
- bounded-memory adaptation
- open-ended evolutionary pressure
- finite-size cycle analysis under delayed feedback

## Connection to Current Paper

The current paper establishes that under the iter8-locked protocol, the cyclic system exhibits:

- non-monotone `k50(N)`
- BMA `k_inf = 0.9407` 
- conditional collapse basin around `beta ~= 0.52`

For Personality Dungeon, this means the target world should preserve structured oscillation and avoid collapsing into a single frozen policy.