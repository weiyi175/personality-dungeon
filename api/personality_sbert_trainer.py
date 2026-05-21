"""B-stage SBERT trainer: replaces TF-IDF char-ngram with dense multilingual embeddings.

Architecture
------------
TF-IDF (v1-v5) → char_wb ngram(1,3) features → MLPRegressor
SBERT   (v6+)  → paraphrase-multilingual-MiniLM-L12-v2 (384-dim) → MLPRegressor

The SBERT encoder bridges the augmented/seed_batch domain gap because both text
styles share semantic meaning, even if their surface character n-grams differ.

Saved artefacts
---------------
- outputs/mlp_v6_mlp.joblib     : trained MultiOutputRegressor (sklearn, no TF-IDF)
- outputs/mlp_v6_embeddings.npz : cached train+val embeddings (for debugging)
- outputs/mlp_v6_training_report.json
"""
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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

from api.schemas import PERSONALITY_BASIS


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "outputs" / "personality_text_pairs.tsv"
DEFAULT_REPORT = REPO_ROOT / "outputs" / "mlp_v6_training_report.json"
DEFAULT_MLP_OUTPUT = REPO_ROOT / "outputs" / "mlp_v6_mlp.joblib"
DEFAULT_EMBEDDINGS_CACHE = REPO_ROOT / "outputs" / "mlp_v6_embeddings.npz"
DEFAULT_SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_INCLUDED_SOURCES = ("seed_batch", "augmented")
DEFAULT_VAL_RATIO = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_HIDDEN_LAYERS = (256, 128)
DEFAULT_MAX_ITER = 500
DEFAULT_PILOT_SIZE = 300
DEFAULT_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Data types (mirrors personality_mlp_trainer.py)
# ---------------------------------------------------------------------------

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


def _parse_sources(raw_value: str | None) -> set[str] | None:
    if raw_value is None:
        return set(DEFAULT_INCLUDED_SOURCES)
    normalized = _normalize_text(raw_value)
    if not normalized or normalized.lower() == "all":
        return None
    sources = {item.strip() for item in normalized.split(",") if item.strip()}
    return sources or None


# ---------------------------------------------------------------------------
# I/O helpers (shared with existing trainer)
# ---------------------------------------------------------------------------

def _load_tsv_records(path: Path) -> list[PersonalityRecord]:
    records: list[PersonalityRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            text = _normalize_text(row.get("text", ""))
            if not text:
                continue
            vector = {trait: _safe_float(row.get(trait)) for trait in PERSONALITY_BASIS}
            records.append(PersonalityRecord(
                text=text,
                source=_normalize_source(row.get("source")),
                vector=vector,
                length=int(float(row.get("length", len(text)))),
                text_hash=_text_hash(text),
            ))
    return records


def _load_records(path: Path) -> list[PersonalityRecord]:
    if not path.exists():
        raise FileNotFoundError(f"input dataset not found: {path}")
    return _load_tsv_records(path)


def _filter_by_source(records: list[PersonalityRecord], sources: set[str] | None) -> list[PersonalityRecord]:
    if sources is None:
        return records
    return [r for r in records if r.source in sources]


def _group_by_hash(records: list[PersonalityRecord]) -> dict[str, list[PersonalityRecord]]:
    groups: dict[str, list[PersonalityRecord]] = defaultdict(list)
    for r in records:
        groups[r.text_hash].append(r)
    return groups


def _take_sample(records: list[PersonalityRecord], *, target_size: int, random_state: int) -> list[PersonalityRecord]:
    if target_size <= 0 or target_size >= len(records):
        return list(records)
    groups = list(_group_by_hash(records).values())
    random.Random(random_state).shuffle(groups)
    sampled: list[PersonalityRecord] = []
    for g in groups:
        sampled.extend(g)
        if len(sampled) >= target_size:
            break
    return sampled


def _split_records(
    records: list[PersonalityRecord],
    *,
    val_ratio: float,
    random_state: int,
) -> tuple[list[PersonalityRecord], list[PersonalityRecord]]:
    groups = list(_group_by_hash(records).items())
    random.Random(random_state).shuffle(groups)
    total = len(records)
    target_val = max(1, min(int(round(total * val_ratio)), total - 1))

    val_hashes: set[str] = set()
    val_count = 0
    for text_hash, group in groups:
        if val_count >= target_val:
            break
        val_hashes.add(text_hash)
        val_count += len(group)

    train: list[PersonalityRecord] = []
    val: list[PersonalityRecord] = []
    for text_hash, group in groups:
        (val if text_hash in val_hashes else train).extend(group)
    return train, val


def _records_to_targets(records: list[PersonalityRecord]) -> np.ndarray:
    return np.array(
        [[r.vector[trait] for trait in PERSONALITY_BASIS] for r in records],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# SBERT encoding
# ---------------------------------------------------------------------------

def _load_sbert(model_name: str):
    from sentence_transformers import SentenceTransformer  # type: ignore
    return SentenceTransformer(model_name)


def _encode_texts(
    model,
    texts: list[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if y_true.size == 0:
        return {"per_trait": {}, "average_r2": 0.0, "average_mae": 0.0}
    per_trait: dict[str, Any] = {}
    r2s, maes = [], []
    for i, trait in enumerate(PERSONALITY_BASIS):
        r2 = float(r2_score(y_true[:, i], y_pred[:, i]))
        mae = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        per_trait[trait] = {"r2": r2, "mae": mae}
        r2s.append(r2)
        maes.append(mae)
    return {
        "per_trait": per_trait,
        "average_r2": float(np.mean(r2s)),
        "average_mae": float(np.mean(maes)),
    }


def _source_breakdown(records: list[PersonalityRecord], y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        grouped[r.source].append(i)
    result: dict[str, Any] = {}
    for source, indices in sorted(grouped.items()):
        result[source] = {
            "count": len(indices),
            "metrics": _evaluate(y_true[indices], y_pred[indices]),
        }
    return result


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_sbert_training(
    *,
    input_path: Path,
    included_sources: set[str] | None,
    pilot_size: int,
    val_ratio: float,
    random_state: int,
    sbert_model_name: str,
    hidden_layers: tuple[int, ...],
    max_iter: int,
    batch_size: int,
    mlp_output: Path,
    embeddings_cache: Path,
    report_path: Path,
) -> dict[str, Any]:
    print(f"[sbert] Loading records from {input_path}")
    all_records = _load_records(input_path)
    included = _filter_by_source(all_records, included_sources)
    pilot = _take_sample(included, target_size=pilot_size, random_state=random_state)
    train_records, val_records = _split_records(pilot, val_ratio=val_ratio, random_state=random_state)

    source_counts = Counter(r.source for r in included)
    print(f"[sbert] Source counts: {dict(source_counts)}")
    print(f"[sbert] Train={len(train_records)}, Val={len(val_records)}")

    # Load SBERT model
    print(f"[sbert] Loading model: {sbert_model_name}")
    sbert = _load_sbert(sbert_model_name)

    # Encode (train then val separately so sizes match targets)
    print("[sbert] Encoding train texts …")
    train_emb = _encode_texts(sbert, [r.text for r in train_records], batch_size=batch_size)
    print("[sbert] Encoding val texts …")
    val_emb = _encode_texts(sbert, [r.text for r in val_records], batch_size=batch_size)

    train_targets = _records_to_targets(train_records)
    val_targets = _records_to_targets(val_records)

    # Cache embeddings for debugging
    embeddings_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(embeddings_cache),
        train_emb=train_emb,
        val_emb=val_emb,
        train_targets=train_targets,
        val_targets=val_targets,
    )
    print(f"[sbert] Embeddings cached → {embeddings_cache}")

    # Train MLP on dense embeddings
    print(f"[sbert] Training MLPRegressor {hidden_layers}, max_iter={max_iter} …")
    mlp = MultiOutputRegressor(
        MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="tanh",
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
        )
    )
    mlp.fit(train_emb, train_targets)

    # Evaluate
    train_pred = mlp.predict(train_emb)
    val_pred = mlp.predict(val_emb)
    overall_val = _evaluate(val_targets, val_pred)
    source_val = _source_breakdown(val_records, val_targets, val_pred)

    # Extract augmented-only metrics
    aug_indices = [i for i, r in enumerate(val_records) if r.source == "augmented"]
    seed_indices = [i for i, r in enumerate(val_records) if r.source == "seed_batch"]
    aug_metrics = _evaluate(val_targets[aug_indices], val_pred[aug_indices]) if aug_indices else {}
    seed_metrics = _evaluate(val_targets[seed_indices], val_pred[seed_indices]) if seed_indices else {}

    # Save MLP
    mlp_output.parent.mkdir(parents=True, exist_ok=True)
    dump(mlp, mlp_output)
    print(f"[sbert] MLP saved → {mlp_output}")

    report: dict[str, Any] = {
        "version": "v6_sbert",
        "sbert_model": sbert_model_name,
        "embedding_dim": int(train_emb.shape[1]),
        "hidden_layers": list(hidden_layers),
        "max_iter": max_iter,
        "pilot_size": pilot_size,
        "val_ratio": val_ratio,
        "random_state": random_state,
        "train_count": len(train_records),
        "val_count": len(val_records),
        "source_counts": dict(source_counts),
        "overall_val": overall_val,
        "augmented_only_val": aug_metrics,
        "seed_batch_val": seed_metrics,
        "source_breakdown_val": source_val,
        "report_path": str(report_path),
        "mlp_path": str(mlp_output),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[sbert] Report saved → {report_path}")
    print(f"[sbert] Overall val avg R²={overall_val['average_r2']:.4f}")
    if aug_metrics:
        print(f"[sbert] Augmented-only val avg R²={aug_metrics['average_r2']:.4f}")
    if seed_metrics:
        print(f"[sbert] Seed_batch val avg R²={seed_metrics['average_r2']:.4f}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train personality MLP using SBERT dense embeddings (v6+)"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--mlp-output", type=Path, default=DEFAULT_MLP_OUTPUT)
    parser.add_argument("--embeddings-cache", type=Path, default=DEFAULT_EMBEDDINGS_CACHE)
    parser.add_argument("--sbert-model", type=str, default=DEFAULT_SBERT_MODEL)
    parser.add_argument(
        "--pilot-size",
        type=int,
        default=DEFAULT_PILOT_SIZE,
        help="Max training+val rows. Set > total rows to use all data.",
    )
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument(
        "--hidden-layers",
        type=str,
        default=",".join(str(x) for x in DEFAULT_HIDDEN_LAYERS),
    )
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--include-sources",
        type=str,
        default=None,
        help="Comma-separated source names, or 'all'.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    hidden_layers = tuple(int(v.strip()) for v in args.hidden_layers.split(",") if v.strip())
    if not hidden_layers:
        raise ValueError("--hidden-layers must contain at least one integer")

    included_sources = _parse_sources(args.include_sources)

    result = run_sbert_training(
        input_path=args.input,
        included_sources=included_sources,
        pilot_size=args.pilot_size,
        val_ratio=args.val_ratio,
        random_state=args.random_state,
        sbert_model_name=args.sbert_model,
        hidden_layers=hidden_layers,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        mlp_output=args.mlp_output,
        embeddings_cache=args.embeddings_cache,
        report_path=args.report,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
