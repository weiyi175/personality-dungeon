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
	}
	assert required.issubset(fieldnames)