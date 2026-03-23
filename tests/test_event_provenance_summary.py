from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from analysis.event_provenance_summary import summarize_event_provenance


def test_event_provenance_summary_reports_event_type_and_top_actions(tmp_path: Path) -> None:
	input_csv = tmp_path / "timeseries.csv"
	baseline_csv = tmp_path / "baseline.csv"
	events_json = Path(__file__).resolve().parents[1] / "docs" / "personality_dungeon_v1" / "02_event_templates_v1.json"

	fieldnames = [
		"round",
		"cycle_level",
		"event_count",
		"success_count",
		"event_types_json",
		"event_ids_json",
		"action_names_json",
		"successes_json",
		"final_risks_json",
		"success_probs_json",
		"trait_deltas_json",
		"trait_deltas_per_event_json",
		"p_aggressive",
		"p_defensive",
		"p_balanced",
		"w_aggressive",
		"w_defensive",
		"w_balanced",
	]
	rows = [
		{
			"round": 0,
			"cycle_level": 0,
			"event_count": 0,
			"success_count": 0,
			"event_types_json": json.dumps([]),
			"event_ids_json": json.dumps([]),
			"action_names_json": json.dumps([]),
			"successes_json": json.dumps([]),
			"final_risks_json": json.dumps([]),
			"success_probs_json": json.dumps([]),
			"trait_deltas_json": json.dumps({}),
			"trait_deltas_per_event_json": json.dumps({}),
			"p_aggressive": 0.34,
			"p_defensive": 0.33,
			"p_balanced": 0.33,
			"w_aggressive": 1.01,
			"w_defensive": 1.0,
			"w_balanced": 0.99,
		},
		{
			"round": 1,
			"cycle_level": 3,
			"event_count": 2,
			"success_count": 1,
			"event_types_json": json.dumps(["Threat", "Resource"]),
			"event_ids_json": json.dumps(["threat_shadow_stalker", "resource_suspicious_chest"]),
			"action_names_json": json.dumps(["attack", "open_now"]),
			"successes_json": json.dumps([True, False]),
			"final_risks_json": json.dumps([0.7, 0.4]),
			"success_probs_json": json.dumps([0.3, 0.6]),
			"trait_deltas_json": json.dumps({"ambition": 0.03, "fearfulness": -0.02}),
			"trait_deltas_per_event_json": json.dumps({
				"threat_shadow_stalker": {"ambition": 0.03},
				"resource_suspicious_chest": {"fearfulness": -0.02},
			}),
			"p_aggressive": 0.55,
			"p_defensive": 0.20,
			"p_balanced": 0.25,
			"w_aggressive": 1.45,
			"w_defensive": 0.70,
			"w_balanced": 0.85,
		},
		{
			"round": 2,
			"cycle_level": 2,
			"event_count": 1,
			"success_count": 1,
			"event_types_json": json.dumps(["Threat"]),
			"event_ids_json": json.dumps(["threat_shadow_stalker"]),
			"action_names_json": json.dumps(["attack"]),
			"successes_json": json.dumps([True]),
			"final_risks_json": json.dumps([0.5]),
			"success_probs_json": json.dumps([0.5]),
			"trait_deltas_json": json.dumps({"ambition": 0.01}),
			"trait_deltas_per_event_json": json.dumps({"threat_shadow_stalker": {"ambition": 0.01}}),
			"p_aggressive": 0.48,
			"p_defensive": 0.22,
			"p_balanced": 0.30,
			"w_aggressive": 1.30,
			"w_defensive": 0.78,
			"w_balanced": 0.92,
		},
	]
	with input_csv.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	with baseline_csv.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerow({
			"round": 0,
			"cycle_level": 0,
			"event_count": 0,
			"success_count": 0,
			"event_types_json": json.dumps([]),
			"event_ids_json": json.dumps([]),
			"action_names_json": json.dumps([]),
			"successes_json": json.dumps([]),
			"final_risks_json": json.dumps([]),
			"success_probs_json": json.dumps([]),
			"trait_deltas_json": json.dumps({}),
			"trait_deltas_per_event_json": json.dumps({}),
			"p_aggressive": 0.36,
			"p_defensive": 0.32,
			"p_balanced": 0.32,
			"w_aggressive": 1.02,
			"w_defensive": 0.99,
			"w_balanced": 0.99,
		})

	summary = summarize_event_provenance(input_csv, events_json=events_json, baseline_csv=baseline_csv)

	assert summary["rounds"] == 3
	assert summary["event_rounds"] == 2
	assert summary["comparison"]["mode"] == "baseline_csv"
	assert summary["amplitude_comparison"]["mode"] == "baseline_csv"
	assert summary["envelope_gamma_comparison"] is None
	assert summary["top_event_action_pairs"][0]["event_id"] == "threat_shadow_stalker"
	assert summary["event_id_summary"][0]["event_id"] == "threat_shadow_stalker"
	assert summary["event_id_summary"][0]["avg_trait_deltas"]["ambition"] == 0.02

	types = {row["event_type"]: row for row in summary["event_type_summary"]}
	assert types["Threat"]["count"] == 2
	assert types["Threat"]["success_rate"] == 1.0
	assert types["Resource"]["count"] == 1


def test_event_provenance_summary_filters_by_cycle_level(tmp_path: Path) -> None:
	input_csv = tmp_path / "timeseries.csv"
	fieldnames = [
		"round",
		"cycle_level",
		"event_count",
		"success_count",
		"event_types_json",
		"event_ids_json",
		"action_names_json",
		"successes_json",
		"final_risks_json",
		"success_probs_json",
		"trait_deltas_json",
		"trait_deltas_per_event_json",
		"p_aggressive",
		"p_defensive",
		"p_balanced",
		"w_aggressive",
		"w_defensive",
		"w_balanced",
	]
	with input_csv.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows([
			{
				"round": 0,
				"cycle_level": 2,
				"event_count": 1,
				"success_count": 1,
				"event_types_json": json.dumps(["Threat"]),
				"event_ids_json": json.dumps(["threat_shadow_stalker"]),
				"action_names_json": json.dumps(["attack"]),
				"successes_json": json.dumps([True]),
				"final_risks_json": json.dumps([0.5]),
				"success_probs_json": json.dumps([0.5]),
				"trait_deltas_json": json.dumps({"ambition": 0.01}),
				"trait_deltas_per_event_json": json.dumps({"threat_shadow_stalker": {"ambition": 0.01}}),
				"p_aggressive": 0.45,
				"p_defensive": 0.25,
				"p_balanced": 0.30,
				"w_aggressive": 1.20,
				"w_defensive": 0.80,
				"w_balanced": 1.00,
			},
			{
				"round": 1,
				"cycle_level": 3,
				"event_count": 1,
				"success_count": 1,
				"event_types_json": json.dumps(["Threat"]),
				"event_ids_json": json.dumps(["threat_shadow_stalker"]),
				"action_names_json": json.dumps(["attack"]),
				"successes_json": json.dumps([True]),
				"final_risks_json": json.dumps([0.4]),
				"success_probs_json": json.dumps([0.6]),
				"trait_deltas_json": json.dumps({"ambition": 0.04}),
				"trait_deltas_per_event_json": json.dumps({"threat_shadow_stalker": {"ambition": 0.04}}),
				"p_aggressive": 0.56,
				"p_defensive": 0.18,
				"p_balanced": 0.26,
				"w_aggressive": 1.42,
				"w_defensive": 0.70,
				"w_balanced": 0.88,
			},
		])

	summary = summarize_event_provenance(input_csv, filter_cycle_level=3)

	assert summary["rounds"] == 1
	assert summary["source_rounds"] == 2
	assert summary["event_id_summary"][0]["avg_trait_deltas"]["ambition"] == 0.04


def test_event_provenance_summary_can_compare_envelope_gamma(tmp_path: Path) -> None:
	input_csv = tmp_path / "event.csv"
	baseline_csv = tmp_path / "baseline.csv"
	fieldnames = [
		"round",
		"event_count",
		"success_count",
		"event_types_json",
		"event_ids_json",
		"action_names_json",
		"successes_json",
		"final_risks_json",
		"success_probs_json",
		"trait_deltas_json",
		"trait_deltas_per_event_json",
		"p_aggressive",
		"p_defensive",
		"p_balanced",
		"w_aggressive",
		"w_defensive",
		"w_balanced",
	]

	with input_csv.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for t in range(60):
			amp = 0.01 + 0.0006 * t
			wave = amp * math.sin((2.0 * math.pi * t) / 6.0)
			writer.writerow({
				"round": t,
				"event_count": 1,
				"success_count": 1,
				"event_types_json": json.dumps(["Threat"]),
				"event_ids_json": json.dumps(["threat_shadow_stalker"]),
				"action_names_json": json.dumps(["attack"]),
				"successes_json": json.dumps([True]),
				"final_risks_json": json.dumps([0.4]),
				"success_probs_json": json.dumps([0.6]),
				"trait_deltas_json": json.dumps({"ambition": 0.02}),
				"trait_deltas_per_event_json": json.dumps({"threat_shadow_stalker": {"ambition": 0.02}}),
				"p_aggressive": 0.3333333333 + wave,
				"p_defensive": 0.3333333333 - 0.5 * wave,
				"p_balanced": 0.3333333334 - 0.5 * wave,
				"w_aggressive": 1.0 + 1.2 * wave,
				"w_defensive": 1.0 - 0.6 * wave,
				"w_balanced": 1.0 - 0.6 * wave,
			})

	with baseline_csv.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for t in range(60):
			amp = 0.008 + 0.0002 * t
			wave = amp * math.sin((2.0 * math.pi * t) / 6.0)
			writer.writerow({
				"round": t,
				"event_count": 0,
				"success_count": 0,
				"event_types_json": json.dumps([]),
				"event_ids_json": json.dumps([]),
				"action_names_json": json.dumps([]),
				"successes_json": json.dumps([]),
				"final_risks_json": json.dumps([]),
				"success_probs_json": json.dumps([]),
				"trait_deltas_json": json.dumps({}),
				"trait_deltas_per_event_json": json.dumps({}),
				"p_aggressive": 0.3333333333 + wave,
				"p_defensive": 0.3333333333 - 0.5 * wave,
				"p_balanced": 0.3333333334 - 0.5 * wave,
				"w_aggressive": 1.0 + 1.2 * wave,
				"w_defensive": 1.0 - 0.6 * wave,
				"w_balanced": 1.0 - 0.6 * wave,
			})

	summary = summarize_event_provenance(input_csv, baseline_csv=baseline_csv, compare_envelope_gamma=True)

	assert summary["envelope_gamma_comparison"] is not None
	assert summary["envelope_gamma_comparison"]["series"] == "p"