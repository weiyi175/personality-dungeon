from __future__ import annotations

import csv
from pathlib import Path
import sys


def _read_csv_text(path: Path) -> str:
	# Normalize line endings and avoid platform-specific differences.
	return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_run_simulation_reproducible_with_seed(tmp_path: Path, monkeypatch) -> None:
	# Import lazily so the test remains close to actual CLI wiring.
	from simulation import run_simulation

	out1 = tmp_path / "out1.csv"
	out2 = tmp_path / "out2.csv"

	args1 = [
		"prog",
		"--players",
		"25",
		"--rounds",
		"30",
		"--seed",
		"123",
		"--payoff-mode",
		"matrix_ab",
		"--a",
		"0.12",
		"--b",
		"0.08",
		"--selection-strength",
		"0.9",
		"--out",
		str(out1),
	]
	args2 = args1.copy()
	args2[args2.index(str(out1))] = str(out2)

	monkeypatch.setattr(sys, "argv", args1)
	run_simulation.main()

	monkeypatch.setattr(sys, "argv", args2)
	run_simulation.main()

	assert out1.exists() and out2.exists()
	assert _read_csv_text(out1) == _read_csv_text(out2)

	# Basic sanity: file is valid CSV and has at least a header + 1 row.
	with out1.open("r", encoding="utf-8", newline="") as f:
		rows = list(csv.reader(f))
	assert len(rows) >= 2
	assert len(rows[0]) >= 1


def test_run_simulation_with_events_is_reproducible_with_seed(tmp_path: Path, monkeypatch) -> None:
	from simulation import run_simulation

	out1 = tmp_path / "events1.csv"
	out2 = tmp_path / "events2.csv"
	events_json = Path(__file__).resolve().parents[1] / "docs" / "personality_dungeon_v1" / "02_event_templates_v1.json"

	args1 = [
		"prog",
		"--players",
		"12",
		"--rounds",
		"10",
		"--seed",
		"321",
		"--payoff-mode",
		"matrix_ab",
		"--a",
		"0.12",
		"--b",
		"0.08",
		"--selection-strength",
		"0.9",
		"--enable-events",
		"--events-json",
		str(events_json),
		"--out",
		str(out1),
	]
	args2 = args1.copy()
	args2[args2.index(str(out1))] = str(out2)

	monkeypatch.setattr(sys, "argv", args1)
	run_simulation.main()

	monkeypatch.setattr(sys, "argv", args2)
	run_simulation.main()

	assert out1.exists() and out2.exists()
	assert _read_csv_text(out1) == _read_csv_text(out2)
