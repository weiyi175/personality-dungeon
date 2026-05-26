from __future__ import annotations

import csv
from pathlib import Path

from simulation.personality_h6 import main as h6_main


def test_h6_gate1_main_writes_outputs(tmp_path: Path, monkeypatch) -> None:
	out_root = tmp_path / "h6_gate1"
	summary_tsv = tmp_path / "h6_gate1_summary.tsv"
	combined_tsv = tmp_path / "h6_gate1_combined.tsv"
	decision_md = tmp_path / "h6_gate1_decision.md"

	monkeypatch.setattr(
		"sys.argv",
		[
			"personality_h6",
			"--protocol",
			"gate1",
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
			"--gate1-out-root",
			str(out_root),
			"--gate1-summary-tsv",
			str(summary_tsv),
			"--gate1-combined-tsv",
			str(combined_tsv),
			"--gate1-decision-md",
			str(decision_md),
		],
	)

	assert h6_main() == 0
	with summary_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 16
	with combined_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 8
		assert rows[0]["condition"] == "zero_control"
		assert any(row["condition"] == "ratio_111_reference" for row in rows)
	assert decision_md.exists()


def test_h6_gate2_main_writes_outputs(tmp_path: Path, monkeypatch) -> None:
	out_root = tmp_path / "h6_gate1"
	gate1_summary_tsv = tmp_path / "h6_gate1_summary.tsv"
	gate1_combined_tsv = tmp_path / "h6_gate1_combined.tsv"
	gate1_decision_md = tmp_path / "h6_gate1_decision.md"
	gate2_steps_tsv = tmp_path / "h6_gate2_steps.tsv"
	gate2_summary_tsv = tmp_path / "h6_gate2_summary.tsv"
	gate2_decision_md = tmp_path / "h6_gate2_decision.md"

	monkeypatch.setattr(
		"sys.argv",
		[
			"personality_h6",
			"--protocol",
			"gate1",
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
			"--gate1-out-root",
			str(out_root),
			"--gate1-summary-tsv",
			str(gate1_summary_tsv),
			"--gate1-combined-tsv",
			str(gate1_combined_tsv),
			"--gate1-decision-md",
			str(gate1_decision_md),
		],
	)

	assert h6_main() == 0

	monkeypatch.setattr(
		"sys.argv",
		[
			"personality_h6",
			"--protocol",
			"gate2",
			"--players",
			"30",
			"--rounds",
			"120",
			"--seeds",
			"45,47",
			"--gate2-input-dir",
			str(out_root / "ratio_111_reference"),
			"--gate2-source-condition",
			"ratio_111_reference",
			"--gate2-sampling-interval",
			"30",
			"--gate2-steps-tsv",
			str(gate2_steps_tsv),
			"--gate2-summary-tsv",
			str(gate2_summary_tsv),
			"--gate2-decision-md",
			str(gate2_decision_md),
		],
	)

	assert h6_main() == 0
	with gate2_steps_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert rows
		assert rows[0]["source_condition"] == "ratio_111_reference"
	with gate2_summary_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 2
	assert gate2_decision_md.exists()