from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from dungeon.event_loader import EventLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FULL_EVENTS_JSON = REPO_ROOT / "docs" / "personality_dungeon_v1" / "02_event_templates_v1.json"
DEFAULT_GATE0_EVENTS_JSON = REPO_ROOT / "docs" / "personality_dungeon_v1" / "02_event_templates_smoke_v1.json"
DEFAULT_OUT_PREFIX = REPO_ROOT / "outputs" / "personality_gate0"
PROJECTION_MODULE_PATH = REPO_ROOT / "docs" / "personality_dungeon_v1" / "03_personality_projection_v1.py"
STRATEGY_KEYS = ("aggressive", "defensive", "balanced")
GATE0_EVENT_IDS = (
	"threat_shadow_stalker",
	"resource_suspicious_chest",
	"uncertainty_altar",
)
PROJECTION_FIELDNAMES = [
	"cohort",
	"count",
	"centroid_aggressive",
	"centroid_defensive",
	"centroid_balanced",
	"dominant_sector",
]
ACTION_FIELDNAMES = [
	"event_id",
	"cohort",
	"action_name",
	"choice_count",
	"choice_share",
	"dominant_action",
	"dominant_share",
	"aggressive_defensive_gap",
]


PROTOTYPES: dict[str, dict[str, float]] = {
	"aggressive": {
		"impulsiveness": 0.85,
		"caution": -0.70,
		"greed": 0.80,
		"optimism": 0.45,
		"suspicion": -0.55,
		"persistence": 0.10,
		"randomness": 0.10,
		"stability_seeking": -0.60,
		"ambition": 0.85,
		"patience": -0.45,
		"curiosity": 0.25,
		"fearfulness": -0.70,
	},
	"defensive": {
		"impulsiveness": -0.65,
		"caution": 0.85,
		"greed": -0.10,
		"optimism": -0.10,
		"suspicion": 0.80,
		"persistence": 0.35,
		"randomness": -0.20,
		"stability_seeking": 0.65,
		"ambition": -0.20,
		"patience": 0.70,
		"curiosity": 0.15,
		"fearfulness": 0.85,
	},
	"balanced": {
		"impulsiveness": -0.10,
		"caution": 0.30,
		"greed": 0.05,
		"optimism": 0.20,
		"suspicion": 0.10,
		"persistence": 0.85,
		"randomness": -0.25,
		"stability_seeking": 0.80,
		"ambition": 0.20,
		"patience": 0.80,
		"curiosity": 0.35,
		"fearfulness": 0.10,
	},
}


def _clamp(value: float, lower: float, upper: float) -> float:
	return max(lower, min(upper, value))


@lru_cache(maxsize=1)
def _projection_module() -> Any:
	spec = importlib.util.spec_from_file_location("personality_projection_v1", PROJECTION_MODULE_PATH)
	if spec is None or spec.loader is None:
		raise RuntimeError(f"failed to load projection module from {PROJECTION_MODULE_PATH}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def projection_dimensions() -> list[str]:
	return list(_projection_module().DIMENSIONS)


def project_to_simplex(personality: Mapping[str, float]) -> dict[str, float]:
	return dict(_projection_module().project_to_simplex(personality))


def zero_personality() -> dict[str, float]:
	return {key: 0.0 for key in projection_dimensions()}


def projected_initial_weights(
	personality: Mapping[str, float],
	*,
	init_bias: float,
) -> dict[str, float]:
	simplex = project_to_simplex(personality)
	bias = float(init_bias)
	weights = {
		"aggressive": float(simplex["aggressive"]) * 3.0 * (1.0 + bias),
		"defensive": float(simplex["defensive"]) * 3.0 * (1.0 - bias),
		"balanced": float(simplex["balanced"]) * 3.0,
	}
	mean_weight = sum(float(value) for value in weights.values()) / float(len(weights))
	if mean_weight <= 0.0:
		return {key: 1.0 for key in STRATEGY_KEYS}
	return {key: float(value) / mean_weight for key, value in weights.items()}


def filter_event_schema(
	source_json: Path,
	*,
	event_ids: tuple[str, ...] = GATE0_EVENT_IDS,
) -> dict[str, Any]:
	data = json.loads(source_json.read_text())
	templates = list(data.get("templates", []))
	selected = [template for template in templates if str(template.get("event_id")) in event_ids]
	if len(selected) != len(event_ids):
		found = {str(template.get("event_id")) for template in selected}
		missing = [event_id for event_id in event_ids if event_id not in found]
		raise ValueError(f"missing event templates: {missing}")
	data["templates"] = selected
	return data


def write_reduced_events_json(
	destination: Path,
	*,
	source_json: Path = DEFAULT_FULL_EVENTS_JSON,
	event_ids: tuple[str, ...] = GATE0_EVENT_IDS,
) -> Path:
	data = filter_event_schema(source_json, event_ids=event_ids)
	destination.parent.mkdir(parents=True, exist_ok=True)
	destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
	return destination


def sample_personality(
	prototype: Mapping[str, float],
	*,
	jitter: float,
	rng: random.Random,
) -> dict[str, float]:
	personality: dict[str, float] = {}
	for key in projection_dimensions():
		base = float(prototype.get(key, 0.0))
		personality[key] = _clamp(base + rng.uniform(-jitter, jitter), -1.0, 1.0)
	return personality


def build_cohorts(
	*,
	cohort_size: int,
	jitter: float,
	seed: int,
	prototypes: Mapping[str, Mapping[str, float]] = PROTOTYPES,
) -> dict[str, list[dict[str, float]]]:
	rng = random.Random(seed)
	cohorts: dict[str, list[dict[str, float]]] = {}
	for cohort, prototype in prototypes.items():
		cohorts[cohort] = [
			sample_personality(prototype, jitter=jitter, rng=rng)
			for _ in range(int(cohort_size))
		]
	return cohorts


def centroid_weights(personalities: list[Mapping[str, float]]) -> dict[str, float]:
	if not personalities:
		return {key: 0.0 for key in STRATEGY_KEYS}
	totals = {key: 0.0 for key in STRATEGY_KEYS}
	for personality in personalities:
		weights = project_to_simplex(personality)
		for key in STRATEGY_KEYS:
			totals[key] += float(weights[key])
	count = float(len(personalities))
	return {key: totals[key] / count for key in STRATEGY_KEYS}


def dominant_sector(weights: Mapping[str, float]) -> str:
	return max(STRATEGY_KEYS, key=lambda key: float(weights[key]))


def action_distribution(
	loader: EventLoader,
	*,
	event_id: str,
	personalities: list[Mapping[str, float]],
) -> dict[str, float]:
	event = loader.get_event_template(event_id)
	action_names = [str(action["name"]) for action in event["actions"]]
	counts: Counter[str] = Counter()
	for personality in personalities:
		action = loader.choose_action(event, personality)
		counts[str(action["name"])] += 1
	count = float(len(personalities)) or 1.0
	return {name: counts.get(name, 0) / count for name in action_names}


def aggressive_defensive_gap(
	aggressive_shares: Mapping[str, float],
	defensive_shares: Mapping[str, float],
) -> float:
	all_actions = set(aggressive_shares) | set(defensive_shares)
	return max(abs(float(aggressive_shares.get(name, 0.0)) - float(defensive_shares.get(name, 0.0))) for name in all_actions)


def evaluate_gate0(
	*,
	cohort_size: int,
	jitter: float,
	seed: int,
	events_json: Path,
) -> dict[str, Any]:
	cohorts = build_cohorts(cohort_size=cohort_size, jitter=jitter, seed=seed)
	loader = EventLoader(events_json)

	projection_rows: list[dict[str, Any]] = []
	dominant_sectors: list[str] = []
	for cohort, personalities in cohorts.items():
		centroid = centroid_weights(personalities)
		sector = dominant_sector(centroid)
		dominant_sectors.append(sector)
		projection_rows.append(
			{
				"cohort": cohort,
				"count": len(personalities),
				"centroid_aggressive": f"{centroid['aggressive']:.6f}",
				"centroid_defensive": f"{centroid['defensive']:.6f}",
				"centroid_balanced": f"{centroid['balanced']:.6f}",
				"dominant_sector": sector,
			}
		)

	action_rows: list[dict[str, Any]] = []
	action_pass_events = 0
	for event_id in GATE0_EVENT_IDS:
		shares_by_cohort = {
			cohort: action_distribution(loader, event_id=event_id, personalities=personalities)
			for cohort, personalities in cohorts.items()
		}
		gap = aggressive_defensive_gap(shares_by_cohort["aggressive"], shares_by_cohort["defensive"])
		if gap > 0.25:
			action_pass_events += 1
		for cohort, shares in shares_by_cohort.items():
			dominant_action = max(shares, key=shares.get)
			dominant_share = float(shares[dominant_action])
			for action_name, share in shares.items():
				action_rows.append(
					{
						"event_id": event_id,
						"cohort": cohort,
						"action_name": action_name,
						"choice_count": int(round(float(share) * cohort_size)),
						"choice_share": f"{float(share):.6f}",
						"dominant_action": dominant_action,
						"dominant_share": f"{dominant_share:.6f}",
						"aggressive_defensive_gap": f"{gap:.6f}",
					}
				)

	projection_pass = len(set(dominant_sectors)) == len(STRATEGY_KEYS)
	action_pass = action_pass_events >= 2
	return {
		"projection_rows": projection_rows,
		"action_rows": action_rows,
		"projection_pass": projection_pass,
		"action_pass": action_pass,
		"action_pass_events": action_pass_events,
		"gate0_pass": projection_pass and action_pass,
	}


def _write_tsv(path: Path, *, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
		writer.writeheader()
		writer.writerows(rows)


def write_gate0_outputs(result: Mapping[str, Any], *, out_prefix: Path, events_json: Path, jitter: float, seed: int, cohort_size: int) -> None:
	projection_path = Path(f"{out_prefix}_projection.tsv")
	action_path = Path(f"{out_prefix}_actions.tsv")
	decision_path = Path(f"{out_prefix}_decision.md")
	_write_tsv(projection_path, fieldnames=PROJECTION_FIELDNAMES, rows=list(result["projection_rows"]))
	_write_tsv(action_path, fieldnames=ACTION_FIELDNAMES, rows=list(result["action_rows"]))
	decision_lines = [
		"# Personality Gate 0 Decision",
		"",
		f"- gate0_pass: {'yes' if result['gate0_pass'] else 'no'}",
		f"- projection_pass: {'yes' if result['projection_pass'] else 'no'}",
		f"- action_pass: {'yes' if result['action_pass'] else 'no'}",
		f"- action_pass_events: {result['action_pass_events']}/3",
		f"- events_json: {events_json}",
		f"- personality_jitter: {jitter:.2f}",
		f"- cohort_size: {cohort_size}",
		f"- seed: {seed}",
		"",
		"## Stop Rule",
		"",
		"- Gate 0 只有在 projection 與 action-choice 兩者都通過時才算成立。",
		"- 若未通過，先修 projection 或 personality-to-action 映射，不進 Gate 1。",
	]
	decision_path.parent.mkdir(parents=True, exist_ok=True)
	decision_path.write_text("\n".join(decision_lines) + "\n")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Gate 0 smoke harness for personality/event mainline")
	parser.add_argument("--source-events-json", type=Path, default=DEFAULT_FULL_EVENTS_JSON)
	parser.add_argument("--events-json", type=Path, default=DEFAULT_GATE0_EVENTS_JSON)
	parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
	parser.add_argument("--cohort-size", type=int, default=100)
	parser.add_argument("--jitter", type=float, default=0.08)
	parser.add_argument("--seed", type=int, default=45)
	parser.add_argument(
		"--materialize-events-json-only",
		action="store_true",
		help="Only write the reduced 3-template event schema and exit.",
	)
	return parser.parse_args()


def main() -> int:
	args = _parse_args()
	write_reduced_events_json(args.events_json, source_json=args.source_events_json)
	if args.materialize_events_json_only:
		print(f"WROTE_EVENTS_JSON {args.events_json}")
		return 0
	result = evaluate_gate0(
		cohort_size=args.cohort_size,
		jitter=args.jitter,
		seed=args.seed,
		events_json=args.events_json,
	)
	write_gate0_outputs(
		result,
		out_prefix=args.out_prefix,
		events_json=args.events_json,
		jitter=args.jitter,
		seed=args.seed,
		cohort_size=args.cohort_size,
	)
	print(f"projection_pass={result['projection_pass']}")
	print(f"action_pass={result['action_pass']}")
	print(f"action_pass_events={result['action_pass_events']}")
	print(f"gate0_pass={result['gate0_pass']}")
	print(f"projection_tsv={args.out_prefix}_projection.tsv")
	print(f"actions_tsv={args.out_prefix}_actions.tsv")
	print(f"decision_md={args.out_prefix}_decision.md")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())