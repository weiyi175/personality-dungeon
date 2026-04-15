from __future__ import annotations

import csv
from pathlib import Path

from players.base_player import BasePlayer
from simulation.w2_episode import (
	W2CellConfig,
	apply_testament,
	compute_death_threshold,
	compute_personality_risk_delta,
	run_w2_cell,
	run_w2_scout,
)


def _player(*, utility: float, dominant: str, trait_scale: float = 0.0) -> BasePlayer:
	player = BasePlayer(["aggressive", "defensive", "balanced"])
	player.utility = float(utility)
	player.personality["impulsiveness"] = 0.10
	player.personality["caution"] = 0.20
	setattr(player, "w2_strategy_history", [dominant] * 600)
	setattr(
		player,
		"w2_trait_delta_sum",
		{
			"impulsiveness": float(trait_scale),
			"caution": -float(trait_scale),
			"greed": 0.0,
			"optimism": 0.0,
			"suspicion": 0.0,
			"persistence": 0.0,
			"randomness": 0.0,
			"stability_seeking": 0.0,
			"ambition": 0.0,
			"patience": 0.0,
			"curiosity": 0.0,
			"fearfulness": 0.0,
		},
	)
	setattr(player, "w2_event_count", 4)
	setattr(player, "w2_event_success_count", 2)
	return player


def test_apply_testament_alpha_zero_and_clip() -> None:
	players = [
		_player(utility=10.0, dominant="aggressive", trait_scale=10.0),
		_player(utility=-10.0, dominant="defensive", trait_scale=10.0),
	]
	unchanged = apply_testament(players, 0.0)
	assert unchanged[0] == players[0].personality
	shifted = apply_testament(players, 1.0)
	for before, after in zip(players, shifted, strict=True):
		for key, value in after.items():
			assert -1.0 <= float(value) <= 1.0
			assert abs(float(value) - float(before.personality.get(key, 0.0))) <= 0.25 + 1e-9


def test_death_contract_uses_only_four_traits() -> None:
	baseline = {
		"impulsiveness": 0.4,
		"caution": 0.2,
		"stability_seeking": -0.1,
		"fearfulness": 0.3,
		"greed": 1.0,
	}
	variant = dict(baseline)
	variant["greed"] = -1.0
	assert compute_personality_risk_delta(baseline) == compute_personality_risk_delta(variant)
	assert compute_death_threshold(baseline) == compute_death_threshold(variant)


def test_run_w2_scout_writes_outputs_and_close_decision(tmp_path: Path) -> None:
	seen_total_lives: dict[str, int] = {}

	def _fake_runner(config: W2CellConfig, seed: int) -> dict[str, object]:
		seen_total_lives[config.condition] = int(config.total_lives)
		life_rows = []
		for life_index in range(1, int(config.total_lives) + 1):
			gamma = -0.020
			stage3 = 0.20
			if config.condition == "w2_base":
				gamma = -0.005
				stage3 = 0.25
			elif config.condition == "w2_strong":
				gamma = 0.010
				stage3 = 0.28
			life_rows.append(
				{
					"protocol": "w2_episode",
					"condition": config.condition,
					"seed": int(seed),
					"life_index": life_index,
					"mean_personality_abs_shift": "0.000000" if config.is_control else "0.080000",
					"mean_personality_l2_shift": "0.000000" if config.is_control else "0.200000",
					"mean_personality_centroid_json": "{}",
					"ended_by_death": "yes" if config.condition != "control" else "no",
					"n_deaths": 2 if config.condition != "control" else 0,
					"mean_life_rounds": "40.000000",
					"rounds_completed": 40,
					"cycle_level": 2,
					"mean_stage3_score": f"{stage3:.6f}",
					"mean_turn_strength": "0.050000",
					"mean_env_gamma": f"{gamma:.6f}",
					"level3_seed_count": 0,
					"testament_alpha": f"{config.testament_alpha:.6f}",
					"testament_applied": "yes" if (not config.is_control and life_index < 5) else "no",
					"dominant_strategy_last500": "aggressive",
					"mean_utility": "1.000000",
					"std_utility": "0.100000",
					"mean_success_rate": "0.500000",
					"mean_risk_final": "0.400000",
					"mean_threshold": "1.000000",
					"verdict": "control" if config.is_control else "non_level3",
					"out_csv": str(config.life_out_csv(seed, life_index)),
					"provenance_json": str(config.provenance_path(seed, life_index)),
				}
			)
		return {"seed": seed, "condition": config.condition, "life_rows": life_rows}

	result = run_w2_scout(
		seeds=[45, 47],
		out_root=tmp_path / "w2",
		life_steps_tsv=tmp_path / "w2_life.tsv",
		combined_tsv=tmp_path / "w2_combined.tsv",
		decision_md=tmp_path / "w2_decision.md",
		cell_runner=_fake_runner,
	)

	assert result["decision"] == "close_w2_1"
	assert seen_total_lives == {"control": 1, "w2_base": 5, "w2_strong": 5}
	with (tmp_path / "w2_life.tsv").open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 22
		assert {row["condition"] for row in rows} == {"control", "w2_base", "w2_strong"}
		control_rows = [row for row in rows if row["condition"] == "control"]
		assert len(control_rows) == 2
		assert all(row["verdict"] == "control" for row in control_rows)
	with (tmp_path / "w2_combined.tsv").open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 3
		assert next(row for row in rows if row["condition"] == "control")["verdict"] == "control"
		strong_row = next(row for row in rows if row["condition"] == "w2_strong")
		assert strong_row["tail_level3_seed_count"] == "0"
		assert strong_row["verdict"] == "weak_positive"
	assert (tmp_path / "w2_decision.md").exists()


def test_run_w2_cell_smoke(tmp_path: Path) -> None:
	config = W2CellConfig(
		condition="w2_base",
		testament_alpha=0.12,
		total_lives=2,
		rounds_per_life=20,
		players=9,
		events_json=Path("docs/personality_dungeon_v1/02_event_templates_smoke_v1.json"),
		selection_strength=0.06,
		init_bias=0.12,
		memory_kernel=3,
		burn_in=0,
		tail=20,
		a=1.0,
		b=0.9,
		cross=0.20,
		out_dir=tmp_path / "w2_base",
	)

	result = run_w2_cell(config, 45)
	assert len(result["life_rows"]) == 2
	for row in result["life_rows"]:
		assert Path(row["out_csv"]).exists()
		assert Path(row["provenance_json"]).exists()