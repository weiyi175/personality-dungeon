# Personality Dungeon v1 Design Pack

This folder stores research-grade design artifacts for Personality Dungeon as independent files that can be consumed by code, JSON loaders, or future config pipelines.

Files:

- `00_world_framework_v1.md`: full world, system, and loop overview.
- `01_personality_dimensions_v1.json`: formal 12D personality vector specification.
- `02_event_templates_v1.json`: event templates with actions, normalized utility-weight policy, clamped risk models, registry-based success models, quantized reward effects, failure outcomes, and cross-round state effects.
- `03_personality_projection_v1.py`: 12D personality to 3-strategy projection utilities.
- `04_economy_rules_v1.json`: economy formulas and balancing constraints.
- `05_little_dragon_v1.py`: adaptive dungeon generation logic for Meta GAI.

Design constraints:

1. `simulation/` remains the only layer that performs I/O.
2. `evolution/` remains pure compute.
3. These files are planning and implementation specs, not runtime source of truth yet.
4. Numerical anchors align with the iter8 paper draft where applicable:
   - BMA `k_inf` median: `0.9407`
   - 95% CI: `[0.8183, 0.9558]`
   - representative `k50(N)` non-monotone pattern over `N = 50, 100, 200, 300, 500, 1000`

Recommended implementation order:

1. Load `01_personality_dimensions_v1.json` into a personality module.
2. Use `03_personality_projection_v1.py` to map the 12D vector into the 3-strategy simplex.
3. Load `02_event_templates_v1.json` in a future dungeon event engine.
4. Apply `04_economy_rules_v1.json` at the service or simulation orchestration layer.
5. Use `05_little_dragon_v1.py` as the first adaptive world-pressure engine.