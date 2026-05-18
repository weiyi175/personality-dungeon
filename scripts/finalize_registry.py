#!/usr/bin/env python3
from __future__ import annotations

"""finalize_registry.py — compress the frontier report into a registry manifest.

This script keeps only the final entity snapshot for SQLite-friendly ingestion.
It does not rescan, rerun simulation, or retain trajectory arrays.
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from evolution.personality_mapping import (  # noqa: E402
	WEIGHTS_EXPLORING_ASYMMETRIC,
	WEIGHTS_SYMMETRIC,
)


SCHEMA_VERSION = "registry.v1"
FORMULA_VERSION = "v2_log_e_k"
DEFAULT_INPUT_JSON = REPO_ROOT / "outputs" / "personality_frontier_v1.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "outputs" / "registry_manifest.json"

SIGNAL_BASIS = {
	"expanding": dict(WEIGHTS_SYMMETRIC["expanding"]),
	"contracting": dict(WEIGHTS_SYMMETRIC["contracting"]),
	"exploring": dict(WEIGHTS_EXPLORING_ASYMMETRIC["exploring"]),
}


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Finalize the frontier report into registry_manifest.json")
	parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON, help="Path to personality_frontier_v1.json")
	parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT_JSON, help="Output registry manifest path")
	return parser.parse_args()


def _load_json(path: Path) -> dict:
	if not path.exists():
		raise FileNotFoundError(f"Input JSON not found: {path}")
	data = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(data, dict):
		raise ValueError(f"Expected top-level object in {path}")
	return data


def _sha256_file(path: Path) -> str:
	digest = hashlib.sha256(path.read_bytes()).hexdigest()
	return f"sha256_{digest}"


def _normalize(value: float, minimum: float, maximum: float) -> float:
	if maximum <= minimum:
		return 0.0
	normalized = (float(value) - float(minimum)) / (float(maximum) - float(minimum))
	if normalized < 0.0:
		return 0.0
	if normalized > 1.0:
		return 1.0
	return float(normalized)


def _tag_from_labels(labels: set[str], *, survivor: bool, pulsar: bool, alpha: bool) -> tuple[int, int, int]:
	return (
		1 if alpha or "[Alpha Seed]" in labels else 0,
		1 if survivor or "[The Survivor]" in labels else 0,
		1 if pulsar or "[The Pulsar]" in labels else 0,
	)


def _frontier_threshold_tags(metrics: dict, thresholds: dict) -> tuple[bool, bool, bool]:
	survivor = (
		float(metrics["sweetspot_width_axis"]) >= float(thresholds.get("width_p90", float("inf")))
		and float(metrics["drift_p95"]) <= float(thresholds.get("drift_p50", float("-inf")))
	)
	pulsar = (
		float(metrics["peak_re"]) >= float(thresholds.get("peak_p90", float("inf")))
		and float(metrics["resonance_mean"]) >= float(thresholds.get("resonance_p50", float("inf")))
	)
	alpha = survivor and pulsar
	return alpha, survivor, pulsar


def build_registry_manifest(frontier_path: Path) -> dict:
	frontier = _load_json(frontier_path)
	entities = frontier.get("frontier")
	if not isinstance(entities, list) or not entities:
		raise ValueError(f"Expected non-empty frontier list in {frontier_path}")

	selection = frontier.get("selection")
	if not isinstance(selection, dict):
		raise ValueError(f"Expected selection block in {frontier_path}")

	thresholds = frontier.get("thresholds")
	if not isinstance(thresholds, dict):
		raise ValueError(f"Expected thresholds block in {frontier_path}")

	raw_rows: list[dict] = []
	raw_values = {
		"re_peak": [],
		"sw_width": [],
		"z_drift": [],
	}
	entity_ids: set[str] = set()

	for index, candidate in enumerate(entities):
		if not isinstance(candidate, dict):
			raise ValueError(f"Invalid frontier entity at index {index}")

		seed = int(candidate.get("seed"))
		trial_id = int(candidate.get("trial_id"))
		rank = int(candidate.get("frontier_rank", candidate.get("pareto_rank", index)))
		snapshot = candidate.get("signal_snapshot")
		if not isinstance(snapshot, dict):
			raise ValueError(f"Missing signal_snapshot for entity {index}")
		asy = candidate.get("asy")
		if not isinstance(asy, dict):
			raise ValueError(f"Missing asy block for entity {index}")
		asy_metrics = asy.get("metrics")
		if not isinstance(asy_metrics, dict):
			raise ValueError(f"Missing asy.metrics block for entity {index}")

		entity_id = f"ent_s{seed}_t{trial_id}"
		if entity_id in entity_ids:
			raise ValueError(f"Duplicate entity_id generated: {entity_id}")
		entity_ids.add(entity_id)

		labels = {str(label) for label in candidate.get("labels", []) if isinstance(label, str)}
		alpha_tag, survivor_tag, pulsar_tag = _frontier_threshold_tags(asy_metrics, thresholds)
		alpha_tag, survivor_tag, pulsar_tag = _tag_from_labels(
			labels,
			survivor=survivor_tag,
			pulsar=pulsar_tag,
			alpha=alpha_tag,
		)

		w_exp = float(snapshot.get("z_expanding", 0.0))
		w_con = float(snapshot.get("z_contracting", 0.0))
		w_ext = float(snapshot.get("z_exploring", 0.0))
		re_peak = float(asy_metrics["peak_re"])
		sw_width = float(asy_metrics["sweetspot_width_axis"])
		z_drift = float(asy_metrics["drift_median"])
		slr = float(candidate.get("slr", 0.0))

		raw_rows.append(
			{
				"entity_id": entity_id,
				"original_seed": seed,
				"trial_id": trial_id,
				"rank": rank,
				"w_exp": w_exp,
				"w_con": w_con,
				"w_ext": w_ext,
				"re_peak": re_peak,
				"sw_width": sw_width,
				"z_drift": z_drift,
				"slr": slr,
				"tag_alpha": alpha_tag,
				"tag_survivor": survivor_tag,
				"tag_pulsar": pulsar_tag,
			}
		)
		raw_values["re_peak"].append(re_peak)
		raw_values["sw_width"].append(sw_width)
		raw_values["z_drift"].append(z_drift)

	norm_basis = {
		key: {"min": float(min(values)), "max": float(max(values))}
		for key, values in raw_values.items()
	}

	entities_out: list[dict] = []
	for row in raw_rows:
		entities_out.append(
			{
				"entity_id": row["entity_id"],
				"original_seed": row["original_seed"],
				"trial_id": row["trial_id"],
				"rank": row["rank"],
				"w_exp": row["w_exp"],
				"w_con": row["w_con"],
				"w_ext": row["w_ext"],
				"re_peak": row["re_peak"],
				"sw_width": row["sw_width"],
				"z_drift": row["z_drift"],
				"slr": row["slr"],
				"n_energy": _normalize(row["re_peak"], norm_basis["re_peak"]["min"], norm_basis["re_peak"]["max"]),
				"n_stability": _normalize(row["sw_width"], norm_basis["sw_width"]["min"], norm_basis["sw_width"]["max"]),
				"n_drift": _normalize(row["z_drift"], norm_basis["z_drift"]["min"], norm_basis["z_drift"]["max"]),
				"tag_alpha": row["tag_alpha"],
				"tag_survivor": row["tag_survivor"],
				"tag_pulsar": row["tag_pulsar"],
			}
		)

	provenance = {
		"formula_version": FORMULA_VERSION,
		"python_version": platform.python_version(),
		"source_frontier_path": str(frontier_path),
		"source_summary_path": str(frontier.get("source_summary_path", "")),
		"source_hash": _sha256_file(frontier_path),
		"sample_count": len(entities_out),
		"selection": {
			"method": str(selection.get("method", "non_dominated_sort + crowding_distance")),
			"trait_precision": int(selection.get("trait_precision", 6)),
			"raw_candidates": int(selection.get("raw_candidates", len(entities_out))),
			"unique_candidates": int(selection.get("unique_candidates", len(entities_out))),
			"selected_candidates": int(selection.get("selected_candidates", len(entities_out))),
		},
		"signal_basis": SIGNAL_BASIS,
		"source_scan": {
			"scan_min": float(frontier.get("scan", {}).get("scan_min", 0.0)),
			"scan_max": float(frontier.get("scan", {}).get("scan_max", 0.0)),
			"scan_step": float(frontier.get("scan", {}).get("scan_step", 0.0)),
			"t_steps": int(frontier.get("scan", {}).get("t_steps", 0)),
			"tail_start": int(frontier.get("scan", {}).get("tail_start", 0)),
			"peak_fraction": float(frontier.get("scan", {}).get("peak_fraction", 0.95)),
			"width_rule": str(frontier.get("scan", {}).get("width_rule", "peak_relative_ma3")),
			"gap_tolerance": int(frontier.get("scan", {}).get("gap_tolerance", 0)),
			"resonance_max_lag": int(frontier.get("scan", {}).get("resonance_max_lag", 0)),
			"fitness_k": float(frontier.get("scan", {}).get("fitness_k", 0.0)),
		},
		"tag_thresholds": thresholds,
		"norm_basis": norm_basis,
		"width_stats": {
			"sw_width_zero_count": int(sum(1 for row in raw_rows if row["sw_width"] == 0.0)),
			"sw_width_positive_count": int(sum(1 for row in raw_rows if row["sw_width"] > 0.0)),
			"sw_width_hit_rate": float(sum(1 for row in raw_rows if row["sw_width"] > 0.0) / len(raw_rows)) if raw_rows else 0.0,
		},
	}

	return {
		"schema_version": SCHEMA_VERSION,
		"provenance": provenance,
		"entities": entities_out,
	}


def main() -> None:
	args = _parse_args()
	manifest = build_registry_manifest(args.input_json)
	args.out_json.parent.mkdir(parents=True, exist_ok=True)
	args.out_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"Wrote registry manifest to {args.out_json}")


if __name__ == "__main__":
	main()