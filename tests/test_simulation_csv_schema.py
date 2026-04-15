import csv
import sys
from pathlib import Path


def test_timeseries_csv_schema_contains_required_columns(tmp_path: Path):
	out_csv = tmp_path / "ts.csv"

	argv_backup = sys.argv[:]
	try:
		sys.argv = [
			"simulation.run_simulation",
			"--players",
			"10",
			"--rounds",
			"3",
			"--payoff-mode",
			"matrix_ab",
			"--a",
			"1.0",
			"--b",
			"1.2",
			"--selection-strength",
			"0.02",
			"--out",
			str(out_csv),
		]
		import simulation.run_simulation as sim

		sim.main()
	finally:
		sys.argv = argv_backup

	with out_csv.open(newline="") as f:
		reader = csv.DictReader(f)
		fieldnames = set(reader.fieldnames or [])

	required = {
		"round",
		"avg_reward",
		"avg_utility",
		"p_aggressive",
		"p_defensive",
		"p_balanced",
		"w_aggressive",
		"w_defensive",
		"w_balanced",
		"threshold_regime_hi",
		"threshold_state_value",
	}
	assert required.issubset(fieldnames)


def test_timeseries_csv_schema_contains_event_provenance_when_enabled(tmp_path: Path):
	out_csv = tmp_path / "ts_events.csv"
	events_json = Path(__file__).resolve().parents[1] / "docs" / "personality_dungeon_v1" / "02_event_templates_v1.json"

	argv_backup = sys.argv[:]
	try:
		sys.argv = [
			"simulation.run_simulation",
			"--players",
			"6",
			"--rounds",
			"2",
			"--seed",
			"123",
			"--payoff-mode",
			"matrix_ab",
			"--a",
			"1.0",
			"--b",
			"1.2",
			"--selection-strength",
			"0.02",
			"--enable-events",
			"--events-json",
			str(events_json),
			"--out",
			str(out_csv),
		]
		import simulation.run_simulation as sim

		sim.main()
	finally:
		sys.argv = argv_backup

	with out_csv.open(newline="") as f:
		reader = csv.DictReader(f)
		fieldnames = set(reader.fieldnames or [])

	required = {
		"threshold_regime_hi",
		"threshold_state_value",
		"event_count",
		"success_count",
		"event_types_json",
		"event_ids_json",
		"action_names_json",
		"result_kinds_json",
		"successes_json",
		"final_risks_json",
		"success_probs_json",
		"trait_deltas_json",
		"trait_deltas_per_event_json",
		"popularity_shift_json",
		"state_effects_json",
	}
	assert required.issubset(fieldnames)


def test_threshold_ab_timeseries_writes_threshold_diagnostics(tmp_path: Path):
	out_csv = tmp_path / "ts_threshold.csv"

	argv_backup = sys.argv[:]
	try:
		sys.argv = [
			"simulation.run_simulation",
			"--players",
			"12",
			"--rounds",
			"6",
			"--seed",
			"123",
			"--payoff-mode",
			"threshold_ab",
			"--a",
			"1.0",
			"--b",
			"0.9",
			"--threshold-theta",
			"0.55",
			"--threshold-a-hi",
			"1.1",
			"--threshold-b-hi",
			"1.0",
			"--selection-strength",
			"0.02",
			"--out",
			str(out_csv),
		]
		import simulation.run_simulation as sim

		sim.main()
	finally:
		sys.argv = argv_backup

	with out_csv.open(newline="") as f:
		rows = list(csv.DictReader(f))

	assert rows
	assert any(str(row.get("threshold_regime_hi", "")).strip() != "" for row in rows)
	assert any(str(row.get("threshold_state_value", "")).strip() != "" for row in rows)