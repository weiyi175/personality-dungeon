#!/usr/bin/env python3
"""驗證「人格 → 選擇」對應關係（被動版）。

把幾個對比鮮明的人格原型（archetype）丟進全部 10 個事件，
印出每個原型在每個事件選了哪個選項，確認不同人格確實導向不同選擇。

用法：
    python scripts/validate_passive_choices.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dungeon.event_loader import EventLoader  # noqa: E402
from simulation.passive_choice import (  # noqa: E402
    resolve_passive_choice,
    vector_to_personality,
)

EVENTS_JSON = ROOT / "docs" / "personality_dungeon_v1" / "02_event_templates_v1.json"

# 維度順序：IMP AST OPT RSK SUS END RND STB CUR
ARCHETYPES: dict[str, list[float]] = {
    #              IMP   AST   OPT   RSK   SUS   END   RND   STB   CUR
    "aggressive": [0.9,  0.8,  0.6, -0.8, -0.5,  0.2,  0.0, -0.6,  0.1],
    "cautious":   [-0.7, -0.3,  0.0,  0.9,  0.7,  0.6, -0.4,  0.7, -0.2],
    "explorer":   [0.2,  0.1,  0.4, -0.2,  0.0,  0.1,  0.8, -0.3,  0.9],
    "balanced":   [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
}


def main() -> int:
    loader = EventLoader(EVENTS_JSON)
    archetypes = {
        name: vector_to_personality(loader, vec) for name, vec in ARCHETYPES.items()
    }

    names = list(archetypes.keys())
    col_w = 16

    differ = 0
    total = 0
    for template in loader.templates:
        event_id = str(template["event_id"])
        options = [str(a["name"]) for a in template["actions"]]
        print(f"\n=== {event_id}  [{template.get('type')}]  options={options}")
        header = "  ".join(f"{n:<{col_w}}" for n in names)
        print(f"    {header}")

        choices: list[str] = []
        row = []
        for name in names:
            res = resolve_passive_choice(
                loader,
                archetypes[name],
                event_id=event_id,
                deterministic_outcome=True,
            )
            chosen = res["chosen_action"]
            # 找到該選項的 lean_prob 顯示信心
            lean = next(o["lean_prob"] for o in res["options"] if o["chosen"])
            choices.append(chosen)
            row.append(f"{chosen}({lean:.0%})")
        print("    " + "  ".join(f"{c:<{col_w}}" for c in row))

        total += 1
        if len(set(choices)) > 1:
            differ += 1

    print(
        f"\n結論：{differ}/{total} 個事件在不同人格間產生了不同選擇"
        f"（差異率 {differ / total:.0%}）。"
    )
    print("若差異率偏低，代表 weights 的人格區辨度不足，需要回去調事件模板的 weights。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
