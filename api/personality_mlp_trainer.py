from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from api.schemas import PERSONALITY_BASIS


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "outputs" / "personality_text_pairs.tsv"
DEFAULT_SPLIT_MANIFEST = REPO_ROOT / "outputs" / "mlp_split_manifest.json"
DEFAULT_REPORT = REPO_ROOT / "outputs" / "mlp_pilot_report.json"
DEFAULT_PIPELINE = REPO_ROOT / "outputs" / "mlp_pilot_pipeline.joblib"
DEFAULT_INCLUDED_SOURCES = ("seed_batch", "augmented")
DEFAULT_PILOT_SIZE = 300
DEFAULT_VAL_RATIO = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_FEATURES = 3000
DEFAULT_HIDDEN_LAYERS = (256, 128)
DEFAULT_MAX_ITER = 500


@dataclass(frozen=True, slots=True)
class PersonalityRecord:
    text: str
    source: str
    vector: dict[str, float]
    length: int
    text_hash: str


def _normalize_text(text: str) -> str:
    return text.replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()


def _text_hash(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


def _safe_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _normalize_source(value: str | None) -> str:
    source = _normalize_text(value or "")
    return source or "unknown"


def _length_bucket(length: int) -> str:
    if 1 <= length <= 5:
        return "1-5"
    if 6 <= length <= 12:
        return "6-12"
    if 13 <= length <= 20:
        return "13-20"
    if length <= 0:
        return "empty"
    return ">20"


def _parse_sources(raw_value: str | None) -> set[str] | None:
    if raw_value is None:
        return set(DEFAULT_INCLUDED_SOURCES)
    normalized = _normalize_text(raw_value)
    if not normalized or normalized.lower() == "all":
        return None
    sources = {item.strip() for item in normalized.split(",") if item.strip()}
    return sources or None


def _load_tsv_records(path: Path) -> list[PersonalityRecord]:
    records: list[PersonalityRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_index, row in enumerate(reader, start=2):
            text = _normalize_text(row.get("text", ""))
            if not text:
                continue
            vector = {trait: _safe_float(row.get(trait)) for trait in PERSONALITY_BASIS}
            records.append(
                PersonalityRecord(
                    text=text,
                    source=_normalize_source(row.get("source")),
                    vector=vector,
                    length=int(float(row.get("length", len(text)))),
                    text_hash=_text_hash(text),
                )
            )
    return records


def _load_jsonl_records(path: Path) -> list[PersonalityRecord]:
    records: list[PersonalityRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = _normalize_text(str(row.get("text", "")))
            if not text:
                continue
            vector_obj = row.get("vector", {})
            vector = {trait: float(vector_obj.get(trait, 0.0)) for trait in PERSONALITY_BASIS}
            records.append(
                PersonalityRecord(
                    text=text,
                    source=_normalize_source(str(row.get("source", ""))),
                    vector=vector,
                    length=int(row.get("length", len(text))),
                    text_hash=_text_hash(text),
                )
            )
    return records


def _load_records(path: Path) -> list[PersonalityRecord]:
    if not path.exists():
        raise FileNotFoundError(f"input dataset not found: {path}")
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl_records(path)
    return _load_tsv_records(path)


def _filter_records_by_source(records: list[PersonalityRecord], included_sources: set[str] | None) -> list[PersonalityRecord]:
    if included_sources is None:
        return records
    return [record for record in records if record.source in included_sources]


def _summarize_records(records: list[PersonalityRecord]) -> dict[str, Any]:
    source_counts = Counter(record.source for record in records)
    length_buckets = Counter(_length_bucket(record.length) for record in records)
    unique_texts = {record.text_hash for record in records}
    return {
        "row_count": len(records),
        "unique_text_count": len(unique_texts),
        "duplicate_row_count": len(records) - len(unique_texts),
        "source_counts": dict(source_counts),
        "length_buckets": dict(length_buckets),
    }


def _group_records_by_hash(records: list[PersonalityRecord]) -> dict[str, list[PersonalityRecord]]:
    groups: dict[str, list[PersonalityRecord]] = defaultdict(list)
    for record in records:
        groups[record.text_hash].append(record)
    return groups


def _take_group_sample(
    records: list[PersonalityRecord],
    *,
    target_size: int,
    random_state: int,
) -> list[PersonalityRecord]:
    if target_size <= 0 or target_size >= len(records):
        return list(records)

    groups = list(_group_records_by_hash(records).values())
    rng = random.Random(random_state)
    rng.shuffle(groups)

    sampled: list[PersonalityRecord] = []
    for group in groups:
        sampled.extend(group)
        if len(sampled) >= target_size:
            break
    return sampled


def _split_group_records(
    records: list[PersonalityRecord],
    *,
    val_ratio: float,
    random_state: int,
) -> tuple[list[PersonalityRecord], list[PersonalityRecord], dict[str, Any]]:
    if not records:
        return [], [], {
            "train_count": 0,
            "val_count": 0,
            "train_group_count": 0,
            "val_group_count": 0,
            "train_hashes": [],
            "val_hashes": [],
        }

    groups = list(_group_records_by_hash(records).items())
    rng = random.Random(random_state)
    rng.shuffle(groups)

    total_count = len(records)
    target_val_count = int(round(total_count * val_ratio))
    if total_count > 1:
        target_val_count = max(1, min(target_val_count, total_count - 1))
    else:
        target_val_count = 0

    val_hashes: set[str] = set()
    val_count = 0
    for text_hash, group in groups:
        if val_count >= target_val_count:
            break
        val_hashes.add(text_hash)
        val_count += len(group)

    train_records: list[PersonalityRecord] = []
    val_records: list[PersonalityRecord] = []
    for text_hash, group in groups:
        if text_hash in val_hashes:
            val_records.extend(group)
        else:
            train_records.extend(group)

    if not train_records and val_records:
        moved_hash, moved_group = next(iter(_group_records_by_hash(val_records).items()))
        val_hashes.remove(moved_hash)
        train_records.extend(moved_group)
        val_records = [record for record in val_records if record.text_hash != moved_hash]

    manifest = {
        "train_count": len(train_records),
        "val_count": len(val_records),
        "train_group_count": len(_group_records_by_hash(train_records)),
        "val_group_count": len(_group_records_by_hash(val_records)),
        "train_hashes": sorted({record.text_hash for record in train_records}),
        "val_hashes": sorted({record.text_hash for record in val_records}),
    }
    return train_records, val_records, manifest


def _records_to_xy(records: list[PersonalityRecord]) -> tuple[list[str], np.ndarray]:
    texts = [record.text for record in records]
    targets = np.array([[record.vector[trait] for trait in PERSONALITY_BASIS] for record in records], dtype=np.float64)
    return texts, targets


def _oversample_by_source(
    records: list[PersonalityRecord],
    *,
    source: str,
    factor: int,
    random_state: int,
) -> list[PersonalityRecord]:
    """Return records with `source` rows duplicated so they appear `factor` times as often."""
    if factor <= 1:
        return list(records)
    target_rows = [r for r in records if r.source == source]
    if not target_rows:
        return list(records)
    extra = target_rows * (factor - 1)
    result = list(records) + extra
    random.Random(random_state).shuffle(result)
    return result


def _build_pipeline(*, max_features: int, hidden_layers: tuple[int, ...], random_state: int, max_iter: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(1, 3),
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "mlp",
                MultiOutputRegressor(
                    MLPRegressor(
                        hidden_layer_sizes=hidden_layers,
                        activation="tanh",
                        max_iter=max_iter,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=random_state,
                    )
                ),
            ),
        ]
    )


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if y_true.size == 0:
        return {
            "per_trait": {},
            "average_r2": 0.0,
            "average_mae": 0.0,
        }

    per_trait: dict[str, dict[str, float]] = {}
    r2_values: list[float] = []
    mae_values: list[float] = []
    for index, trait in enumerate(PERSONALITY_BASIS):
        trait_true = y_true[:, index]
        trait_pred = y_pred[:, index]
        trait_r2 = float(r2_score(trait_true, trait_pred))
        trait_mae = float(mean_absolute_error(trait_true, trait_pred))
        per_trait[trait] = {
            "r2": trait_r2,
            "mae": trait_mae,
            "mean_true": float(np.mean(trait_true)),
            "mean_pred": float(np.mean(trait_pred)),
        }
        r2_values.append(trait_r2)
        mae_values.append(trait_mae)

    return {
        "per_trait": per_trait,
        "average_r2": float(np.mean(r2_values)),
        "average_mae": float(np.mean(mae_values)),
    }


def _collect_view_metrics(records: list[PersonalityRecord], y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "metrics": _evaluate_predictions(np.zeros((0, len(PERSONALITY_BASIS))), np.zeros((0, len(PERSONALITY_BASIS)))),
        }

    return {
        "count": len(records),
        "metrics": _evaluate_predictions(y_true, y_pred),
    }


def _group_view_metrics(
    records: list[PersonalityRecord],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    key_fn,
) -> dict[str, Any]:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped_indices[key_fn(record)].append(index)

    grouped_report: dict[str, Any] = {}
    for group_name, indices in sorted(grouped_indices.items()):
        group_y_true = y_true[indices]
        group_y_pred = y_pred[indices]
        grouped_report[group_name] = {
            "count": len(indices),
            "metrics": _evaluate_predictions(group_y_true, group_y_pred),
        }
    return grouped_report


def _build_split_manifest(
    *,
    input_path: Path,
    input_snapshot_hash: str,
    all_records: list[PersonalityRecord],
    included_records: list[PersonalityRecord],
    pilot_records: list[PersonalityRecord],
    pilot_train_records: list[PersonalityRecord],
    pilot_val_records: list[PersonalityRecord],
    included_sources: set[str] | None,
    random_state: int,
    val_ratio: float,
    pilot_size: int,
    train_manifest: dict[str, Any],
) -> dict[str, Any]:
    excluded_records = [record for record in all_records if record not in included_records]
    return {
        "input_path": str(input_path),
        "input_snapshot_sha256": input_snapshot_hash,
        "random_state": random_state,
        "val_ratio": val_ratio,
        "pilot_size": pilot_size,
        "included_sources": sorted(included_sources) if included_sources is not None else None,
        "excluded_sources": sorted({record.source for record in excluded_records}),
        "all_records": _summarize_records(all_records),
        "included_records": _summarize_records(included_records),
        "pilot_records": _summarize_records(pilot_records),
        "pilot_split": train_manifest,
        "pilot_train_records": _summarize_records(pilot_train_records),
        "pilot_val_records": _summarize_records(pilot_val_records),
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_pilot_training(
    *,
    input_path: Path,
    included_sources: set[str] | None,
    pilot_size: int,
    val_ratio: float,
    random_state: int,
    max_features: int,
    hidden_layers: tuple[int, ...],
    max_iter: int,
    pipeline_output: Path,
    split_manifest_path: Path,
    report_path: Path,
    seed_oversample: int = 1,
) -> dict[str, Any]:
    all_records = _load_records(input_path)
    included_records = _filter_records_by_source(all_records, included_sources)
    if not included_records:
        raise ValueError("no records remain after source filtering")

    input_snapshot_hash = _hash_file(input_path)
    pilot_records = _take_group_sample(included_records, target_size=pilot_size, random_state=random_state)
    pilot_train_records, pilot_val_records, pilot_split_manifest = _split_group_records(
        pilot_records,
        val_ratio=val_ratio,
        random_state=random_state,
    )

    pipeline = _build_pipeline(
        max_features=max_features,
        hidden_layers=hidden_layers,
        random_state=random_state,
        max_iter=max_iter,
    )

    effective_train_records = _oversample_by_source(
        pilot_train_records,
        source="seed_batch",
        factor=seed_oversample,
        random_state=random_state,
    )
    train_texts, train_targets = _records_to_xy(effective_train_records)
    val_texts, val_targets = _records_to_xy(pilot_val_records)
    pipeline.fit(train_texts, train_targets)

    val_predictions = pipeline.predict(val_texts) if val_texts else np.zeros((0, len(PERSONALITY_BASIS)))
    val_metrics = _evaluate_predictions(val_targets, val_predictions)

    report: dict[str, Any] = {
        "input_path": str(input_path),
        "input_snapshot_sha256": input_snapshot_hash,
        "random_state": random_state,
        "val_ratio": val_ratio,
        "pilot_size": pilot_size,
        "seed_oversample": seed_oversample,
        "included_sources": sorted(included_sources) if included_sources is not None else None,
        "all_records": _summarize_records(all_records),
        "included_records": _summarize_records(included_records),
        "pilot_records": _summarize_records(pilot_records),
        "pilot_split": pilot_split_manifest,
        "train_records": _summarize_records(effective_train_records),
        "val_records": _summarize_records(pilot_val_records),
        "validation": {
            "overall": _collect_view_metrics(pilot_val_records, val_targets, val_predictions),
            "by_source": _group_view_metrics(
                pilot_val_records,
                val_targets,
                val_predictions,
                key_fn=lambda record: record.source,
            ),
            "by_length_bucket": _group_view_metrics(
                pilot_val_records,
                val_targets,
                val_predictions,
                key_fn=lambda record: _length_bucket(record.length),
            ),
        },
        "metrics": val_metrics,
        "pipeline": {
            "tfidf": {
                "analyzer": "char_wb",
                "ngram_range": [1, 3],
                "max_features": max_features,
                "sublinear_tf": True,
            },
            "mlp": {
                "hidden_layer_sizes": list(hidden_layers),
                "activation": "tanh",
                "max_iter": max_iter,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "random_state": random_state,
            },
        },
    }

    split_manifest = _build_split_manifest(
        input_path=input_path,
        input_snapshot_hash=input_snapshot_hash,
        all_records=all_records,
        included_records=included_records,
        pilot_records=pilot_records,
        pilot_train_records=pilot_train_records,
        pilot_val_records=pilot_val_records,
        included_sources=included_sources,
        random_state=random_state,
        val_ratio=val_ratio,
        pilot_size=pilot_size,
        train_manifest=pilot_split_manifest,
    )

    split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_output.parent.mkdir(parents=True, exist_ok=True)

    split_manifest_path.write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    dump(pipeline, pipeline_output)

    return {
        "split_manifest": split_manifest,
        "report": report,
        "pipeline_output": str(pipeline_output),
        "split_manifest_path": str(split_manifest_path),
        "report_path": str(report_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B-stage personality MLP training skeleton")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input TSV or JSONL path")
    parser.add_argument(
        "--include-sources",
        default=",".join(DEFAULT_INCLUDED_SOURCES),
        help="Comma-separated source names to include; use 'all' for every row",
    )
    parser.add_argument("--pilot-size", type=int, default=DEFAULT_PILOT_SIZE, help="Pilot subset size")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation ratio for group split")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Deterministic split seed")
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES, help="TF-IDF max features")
    parser.add_argument(
        "--hidden-layers",
        default=",".join(str(value) for value in DEFAULT_HIDDEN_LAYERS),
        help="Comma-separated hidden layer sizes for the MLP",
    )
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help="Maximum MLP iterations")
    parser.add_argument(
        "--oversample-seed",
        type=int,
        default=1,
        help="Duplicate seed_batch training rows by this factor to counteract augmented-domain over-representation (default: 1 = disabled)",
    )
    parser.add_argument("--pipeline-output", type=Path, default=DEFAULT_PIPELINE, help="Pilot pipeline joblib path")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST, help="Split manifest JSON path")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Pilot report JSON path")
    parser.add_argument(
        "--mode",
        choices=("pilot", "split-only"),
        default="pilot",
        help="Run pilot training or only emit the split manifest",
    )
    args = parser.parse_args(argv)

    included_sources = _parse_sources(args.include_sources)
    hidden_layers = tuple(int(value.strip()) for value in args.hidden_layers.split(",") if value.strip())
    if not hidden_layers:
        raise ValueError("--hidden-layers must contain at least one integer")

    all_records = _load_records(args.input)
    included_records = _filter_records_by_source(all_records, included_sources)
    if not included_records:
        raise ValueError("no records remain after source filtering")

    input_snapshot_hash = _hash_file(args.input)
    pilot_records = _take_group_sample(included_records, target_size=args.pilot_size, random_state=args.random_state)
    pilot_train_records, pilot_val_records, pilot_split_manifest = _split_group_records(
        pilot_records,
        val_ratio=args.val_ratio,
        random_state=args.random_state,
    )
    split_manifest = _build_split_manifest(
        input_path=args.input,
        input_snapshot_hash=input_snapshot_hash,
        all_records=all_records,
        included_records=included_records,
        pilot_records=pilot_records,
        pilot_train_records=pilot_train_records,
        pilot_val_records=pilot_val_records,
        included_sources=included_sources,
        random_state=args.random_state,
        val_ratio=args.val_ratio,
        pilot_size=args.pilot_size,
        train_manifest=pilot_split_manifest,
    )

    args.split_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.split_manifest.write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.mode == "split-only":
        print(json.dumps(split_manifest, ensure_ascii=False, indent=2))
        return 0

    result = run_pilot_training(
        input_path=args.input,
        included_sources=included_sources,
        pilot_size=args.pilot_size,
        val_ratio=args.val_ratio,
        random_state=args.random_state,
        max_features=args.max_features,
        hidden_layers=hidden_layers,
        max_iter=args.max_iter,
        pipeline_output=args.pipeline_output,
        split_manifest_path=args.split_manifest,
        report_path=args.report,
        seed_oversample=args.oversample_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())