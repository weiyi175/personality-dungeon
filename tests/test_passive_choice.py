from __future__ import annotations

import random
from pathlib import Path

from dungeon.event_loader import EventLoader
from simulation.passive_choice import resolve_passive_choice, vector_to_personality


def _event_json_path() -> Path:
	return Path(__file__).resolve().parents[1] / "docs" / "personality_dungeon_v1" / "02_event_templates_v1.json"


def _loader() -> EventLoader:
	return EventLoader(_event_json_path())


# 維度順序：IMP AST OPT RSK SUS END RND STB CUR
_AGGRESSIVE = [0.9, 0.8, 0.6, -0.8, -0.5, 0.2, 0.0, -0.6, 0.1]
_CAUTIOUS = [-0.7, -0.3, 0.0, 0.9, 0.7, 0.6, -0.4, 0.7, -0.2]


def test_chosen_action_matches_loader_choose_action() -> None:
	loader = _loader()
	personality = vector_to_personality(loader, _AGGRESSIVE)
	event = loader.get_event_template("threat_shadow_stalker")
	expected = loader.choose_action(event, personality)
	res = resolve_passive_choice(
		loader, personality, event_id="threat_shadow_stalker", deterministic_outcome=True
	)
	assert res["chosen_action"] == expected["name"]
	# 恰有一個選項被標記為 chosen，且其 utility 最大。
	chosen = [o for o in res["options"] if o["chosen"]]
	assert len(chosen) == 1
	assert chosen[0]["utility"] == max(o["utility"] for o in res["options"])


def test_lean_probs_form_distribution() -> None:
	loader = _loader()
	personality = vector_to_personality(loader, _CAUTIOUS)
	res = resolve_passive_choice(
		loader, personality, event_id="threat_ambush", deterministic_outcome=True
	)
	total = sum(o["lean_prob"] for o in res["options"])
	assert abs(total - 1.0) < 1e-6
	assert all(0.0 <= o["lean_prob"] <= 1.0 for o in res["options"])


def test_distinct_personalities_can_choose_differently() -> None:
	loader = _loader()
	agg = vector_to_personality(loader, _AGGRESSIVE)
	cau = vector_to_personality(loader, _CAUTIOUS)
	differ = 0
	for template in loader.templates:
		eid = str(template["event_id"])
		a = resolve_passive_choice(loader, agg, event_id=eid, deterministic_outcome=True)
		c = resolve_passive_choice(loader, cau, event_id=eid, deterministic_outcome=True)
		if a["chosen_action"] != c["chosen_action"]:
			differ += 1
	# 對比鮮明的兩種人格，應在多數事件產生不同選擇。
	assert differ >= len(loader.templates) // 2


def test_seed_makes_outcome_reproducible() -> None:
	loader = _loader()
	personality = vector_to_personality(loader, _AGGRESSIVE)
	r1 = resolve_passive_choice(loader, personality, rng=random.Random(123))
	r2 = resolve_passive_choice(loader, personality, rng=random.Random(123))
	assert r1 == r2


def test_readonly_does_not_mutate_personality() -> None:
	loader = _loader()
	personality = vector_to_personality(loader, _AGGRESSIVE)
	snapshot = dict(personality)
	resolve_passive_choice(loader, personality, event_id="navigation_fork")
	assert personality == snapshot
