from __future__ import annotations

from pathlib import Path

from scripts.finalize_registry import build_registry_manifest


def test_finalize_registry_manifest_is_flat_and_normalized() -> None:
	frontier_path = Path("/home/user/personality-dungeon/outputs/personality_frontier_v1.json")
	manifest = build_registry_manifest(frontier_path)

	assert manifest["schema_version"] == "registry.v1"
	provenance = manifest["provenance"]
	assert provenance["python_version"] == "3.10.12"
	assert provenance["sample_count"] == 50
	assert provenance["selection"]["selected_candidates"] == 50
	assert provenance["signal_basis"]["exploring"] == {
		"curiosity": 0.5,
		"randomness": 0.3,
		"stability_seeking": 0.2,
	}
	assert provenance["source_scan"]["peak_fraction"] == 0.95
	assert provenance["source_scan"]["width_rule"] == "peak_relative_ma3"
	assert provenance["width_stats"]["sw_width_zero_count"] == 0
	assert provenance["width_stats"]["sw_width_positive_count"] == 50
	assert provenance["width_stats"]["sw_width_hit_rate"] == 1.0

	entities = manifest["entities"]
	assert len(entities) == 50

	expected_keys = {
		"entity_id",
		"original_seed",
		"trial_id",
		"rank",
		"w_exp",
		"w_con",
		"w_ext",
		"re_peak",
		"sw_width",
		"z_drift",
		"slr",
		"n_energy",
		"n_stability",
		"n_drift",
		"tag_alpha",
		"tag_survivor",
		"tag_pulsar",
	}
	for entity in entities:
		assert set(entity) == expected_keys
		assert entity["entity_id"].startswith("ent_s")
		assert 0.0 <= entity["n_energy"] <= 1.0
		assert 0.0 <= entity["n_stability"] <= 1.0
		assert 0.0 <= entity["n_drift"] <= 1.0
		assert entity["tag_alpha"] in (0, 1)
		assert entity["tag_survivor"] in (0, 1)
		assert entity["tag_pulsar"] in (0, 1)