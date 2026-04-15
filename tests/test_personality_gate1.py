from __future__ import annotations

import csv
from pathlib import Path

from simulation.personality_gate1 import decide_gate1, gamma_improvement_flags, main as gate1_main
from simulation.run_simulation import SimConfig, simulate


def test_simulate_player_setup_callback_is_invoked() -> None:
	called = {"value": False}

	def _setup(players, _strategy_space, _cfg) -> None:
		called["value"] = True
		for player in players:
			player.update_weights({"aggressive": 3.0, "defensive": 0.5, "balanced": 0.5})

	cfg = SimConfig(
		n_players=6,
		n_rounds=5,
		seed=7,
		payoff_mode="count_cycle",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=0.0,
		b=0.0,
		matrix_cross_coupling=0.0,
		init_bias=0.0,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.05,
	)

	_sim_strategy_space, rows = simulate(cfg, player_setup_callback=_setup)

	assert called["value"] is True
	assert rows


def test_gamma_improvement_flags_accepts_closer_to_zero() -> None:
	flags = gamma_improvement_flags(baseline_gamma=-0.10, hetero_gamma=-0.07)

	assert flags["gamma_pass"] is True
	assert flags["gamma_closer_to_zero"] is True


def test_decide_gate1_requires_gamma_pass() -> None:
	decision = decide_gate1(
		baseline_summary={"mean_env_gamma": "-0.010000", "mean_stage3_score": "0.100000", "p_level_ge_2": "0.333333", "p_level_3": "0.000000"},
		hetero_summary={"mean_env_gamma": "-0.020000", "mean_stage3_score": "0.140000", "p_level_ge_2": "1.000000", "p_level_3": "0.333333"},
	)

	assert decision["decision"] == "fail"


def test_gate1_main_writes_outputs(tmp_path: Path, monkeypatch) -> None:
	baseline_dir = tmp_path / "baseline"
	hetero_dir = tmp_path / "hetero"
	summary_tsv = tmp_path / "summary.tsv"
	combined_tsv = tmp_path / "combined.tsv"
	decision_md = tmp_path / "decision.md"

	monkeypatch.setattr(
		"sys.argv",
		[
			"personality_gate1",
			"--players",
			"30",
			"--rounds",
			"120",
			"--seeds",
			"45,47",
			"--burn-in",
			"30",
			"--tail",
			"60",
			"--baseline-dir",
			str(baseline_dir),
			"--hetero-dir",
			str(hetero_dir),
			"--summary-tsv",
			str(summary_tsv),
			"--combined-tsv",
			str(combined_tsv),
			"--decision-md",
			str(decision_md),
		],
	)

	assert gate1_main() == 0
	with summary_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 4
	with combined_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 2
	assert decision_md.exists()


def test_h54_main_writes_outputs(tmp_path: Path, monkeypatch) -> None:
	out_root = tmp_path / "h54"
	summary_tsv = tmp_path / "h54_summary.tsv"
	combined_tsv = tmp_path / "h54_combined.tsv"
	decision_md = tmp_path / "h54_decision.md"

	monkeypatch.setattr(
		"sys.argv",
		[
			"personality_gate1",
			"--protocol",
			"h54",
			"--players",
			"30",
			"--rounds",
			"120",
			"--seeds",
			"45,47",
			"--burn-in",
			"30",
			"--tail",
			"60",
			"--h54-inertias",
			"0.15,0.25",
			"--out-root",
			str(out_root),
			"--h54-summary-tsv",
			str(summary_tsv),
			"--h54-combined-tsv",
			str(combined_tsv),
			"--h54-decision-md",
			str(decision_md),
		],
	)

	assert gate1_main() == 0
	with summary_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 6
	with combined_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 3
		assert rows[0]["is_control"] == "yes"
	assert decision_md.exists()


def test_h55_main_writes_outputs(tmp_path: Path, monkeypatch) -> None:
	out_root = tmp_path / "h55"
	summary_tsv = tmp_path / "h55_summary.tsv"
	combined_tsv = tmp_path / "h55_combined.tsv"
	decision_md = tmp_path / "h55_decision.md"

	monkeypatch.setattr(
		"sys.argv",
		[
			"personality_gate1",
			"--protocol",
			"h55",
			"--players",
			"30",
			"--rounds",
			"120",
			"--seeds",
			"45,47",
			"--burn-in",
			"30",
			"--tail",
			"60",
			"--h55-out-root",
			str(out_root),
			"--h55-summary-tsv",
			str(summary_tsv),
			"--h55-combined-tsv",
			str(combined_tsv),
			"--h55-decision-md",
			str(decision_md),
		],
	)

	assert gate1_main() == 0
	with summary_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 8
	with combined_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 4
		assert rows[0]["is_control"] == "yes"
		assert rows[1]["payoff_mode"] == "threshold_ab"
	assert decision_md.exists()


def test_h55r_main_writes_outputs(tmp_path: Path, monkeypatch) -> None:
	out_root = tmp_path / "h55r"
	summary_tsv = tmp_path / "h55r_summary.tsv"
	combined_tsv = tmp_path / "h55r_combined.tsv"
	decision_md = tmp_path / "h55r_decision.md"

	monkeypatch.setattr(
		"sys.argv",
		[
			"personality_gate1",
			"--protocol",
			"h55r",
			"--players",
			"30",
			"--rounds",
			"120",
			"--seeds",
			"45,47",
			"--burn-in",
			"30",
			"--tail",
			"60",
			"--h55r-out-root",
			str(out_root),
			"--h55r-summary-tsv",
			str(summary_tsv),
			"--h55r-combined-tsv",
			str(combined_tsv),
			"--h55r-decision-md",
			str(decision_md),
		],
	)

	assert gate1_main() == 0
	with summary_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 14
	with combined_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 7
		assert rows[0]["is_control"] == "yes"
		assert rows[1]["condition"].startswith("legacy_geo_")
		assert "confirm_gate_pass" in rows[1]
	assert decision_md.exists()