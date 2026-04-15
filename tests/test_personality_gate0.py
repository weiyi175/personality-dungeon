from __future__ import annotations

import csv
import json
from pathlib import Path

from simulation.personality_gate0 import (
	ACTION_FIELDNAMES,
	DEFAULT_FULL_EVENTS_JSON,
	GATE0_EVENT_IDS,
	PROJECTION_FIELDNAMES,
	evaluate_gate0,
	write_gate0_outputs,
	write_reduced_events_json,
)


def test_write_reduced_events_json_keeps_only_gate0_templates(tmp_path: Path) -> None:
	destination = tmp_path / "events_gate0.json"
	write_reduced_events_json(destination, source_json=DEFAULT_FULL_EVENTS_JSON)
	data = json.loads(destination.read_text())

	assert [template["event_id"] for template in data["templates"]] == list(GATE0_EVENT_IDS)


def test_gate0_projection_and_action_sanity_passes(tmp_path: Path) -> None:
	events_json = tmp_path / "events_gate0.json"
	write_reduced_events_json(events_json, source_json=DEFAULT_FULL_EVENTS_JSON)
	result = evaluate_gate0(cohort_size=100, jitter=0.08, seed=45, events_json=events_json)
	projection_rows = result["projection_rows"]

	assert result["projection_pass"] is True
	assert result["action_pass"] is True
	assert result["action_pass_events"] >= 2
	assert {row["dominant_sector"] for row in projection_rows} == {"aggressive", "defensive", "balanced"}


def test_gate0_output_files_use_expected_schemas(tmp_path: Path) -> None:
	events_json = tmp_path / "events_gate0.json"
	write_reduced_events_json(events_json, source_json=DEFAULT_FULL_EVENTS_JSON)
	result = evaluate_gate0(cohort_size=100, jitter=0.08, seed=45, events_json=events_json)
	out_prefix = tmp_path / "personality_gate0"
	write_gate0_outputs(
		result,
		out_prefix=out_prefix,
		events_json=events_json,
		jitter=0.08,
		seed=45,
		cohort_size=100,
	)

	projection_path = Path(f"{out_prefix}_projection.tsv")
	action_path = Path(f"{out_prefix}_actions.tsv")
	decision_path = Path(f"{out_prefix}_decision.md")

	with projection_path.open(newline="") as handle:
		reader = csv.DictReader(handle, delimiter="\t")
		assert reader.fieldnames == PROJECTION_FIELDNAMES
		rows = list(reader)
		assert len(rows) == 3

	with action_path.open(newline="") as handle:
		reader = csv.DictReader(handle, delimiter="\t")
		assert reader.fieldnames == ACTION_FIELDNAMES
		rows = list(reader)
		assert rows

	decision_text = decision_path.read_text()
	assert "gate0_pass: yes" in decision_text
	assert str(events_json) in decision_text