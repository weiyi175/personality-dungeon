from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence
import sys

import numpy as np
import onnxruntime as ort
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.personality_mlp_trainer import _load_records
from api.schemas import PERSONALITY_BASIS


DEFAULT_INPUT = REPO_ROOT / "outputs" / "personality_text_pairs.tsv"
DEFAULT_PIPELINE = REPO_ROOT / "outputs" / "mlp_full_pipeline.joblib"
DEFAULT_VECTORIZER = REPO_ROOT / "outputs" / "mlp_full_vectorizer.joblib"
DEFAULT_SPLIT_MANIFEST = REPO_ROOT / "outputs" / "mlp_full_split_manifest.json"
DEFAULT_ONNX = REPO_ROOT / "api" / "personality_model.onnx"
DEFAULT_REPORT = REPO_ROOT / "outputs" / "onnx_validation_report.json"
DEFAULT_THRESHOLD = 1e-4


def _select_records(records, hashes: set[str]):
    return [record for record in records if record.text_hash in hashes]


def _predict(pipeline: Any, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, len(PERSONALITY_BASIS)), dtype=np.float64)
    return np.asarray(pipeline.predict(texts), dtype=np.float64)


def _predict_onnx(session: ort.InferenceSession, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, len(PERSONALITY_BASIS)), dtype=np.float64)

    input_name = session.get_inputs()[0].name
    input_array = np.array([[text] for text in texts], dtype=object)
    outputs = session.run(None, {input_name: input_array})
    if not outputs:
        raise RuntimeError("ONNX model returned no outputs")
    prediction = np.asarray(outputs[0], dtype=np.float64)
    if prediction.ndim == 1:
        prediction = prediction.reshape(-1, len(PERSONALITY_BASIS))
    return prediction


def _predict_onnx_features(session: ort.InferenceSession, features: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return np.zeros((0, len(PERSONALITY_BASIS)), dtype=np.float64)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: features.astype(np.float32)})
    if not outputs:
        raise RuntimeError("ONNX model returned no outputs")
    prediction = np.asarray(outputs[0], dtype=np.float64)
    if prediction.ndim == 1:
        prediction = prediction.reshape(-1, len(PERSONALITY_BASIS))
    return prediction


def _summarize_diff(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = np.abs(a - b)
    return {
        "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(diff)) if diff.size else 0.0,
        "per_trait_max_abs_diff": {
            trait: float(np.max(diff[:, index])) if diff.size else 0.0
            for index, trait in enumerate(PERSONALITY_BASIS)
        },
        "per_trait_mean_abs_diff": {
            trait: float(np.mean(diff[:, index])) if diff.size else 0.0
            for index, trait in enumerate(PERSONALITY_BASIS)
        },
    }


def validate_onnx_pipeline(
    *,
    input_path: Path,
    pipeline_path: Path,
    vectorizer_path: Path,
    split_manifest_path: Path,
    onnx_path: Path,
    report_path: Path,
    threshold: float,
) -> dict[str, Any]:
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    val_hashes = set(split_manifest["pilot_split"]["val_hashes"])
    included_sources = split_manifest.get("included_sources")
    input_snapshot_hash = split_manifest.get("input_snapshot_sha256")

    original_pipeline = load(pipeline_path)
    vectorizer = load(vectorizer_path)
    if not isinstance(vectorizer, TfidfVectorizer):
        raise TypeError(f"Unsupported vectorizer: {type(vectorizer)!r}")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    all_records = _load_records(input_path)
    if included_sources is not None:
        included_source_set = set(included_sources)
        all_records = [record for record in all_records if record.source in included_source_set]

    val_records = _select_records(all_records, val_hashes)
    original_texts = [record.text for record in val_records]
    features = vectorizer.transform(original_texts).astype(np.float32).toarray()

    original_pred = _predict(original_pipeline, original_texts)
    onnx_pred = _predict_onnx_features(session, features)

    original_vs_onnx = _summarize_diff(original_pred, onnx_pred)

    report: dict[str, Any] = {
        "input_path": str(input_path),
        "input_snapshot_sha256": input_snapshot_hash,
        "pipeline_path": str(pipeline_path),
        "vectorizer_path": str(vectorizer_path),
        "onnx_path": str(onnx_path),
        "split_manifest_path": str(split_manifest_path),
        "feature_count": int(features.shape[1]),
        "validation_count": len(val_records),
        "threshold": threshold,
        "original_vs_onnx": original_vs_onnx,
        "passed": original_vs_onnx["max_abs_diff"] <= threshold,
        "sample_texts": original_texts[:5],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate ONNX parity against the sklearn personality pipeline")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input TSV or JSONL path")
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE, help="Original sklearn pipeline joblib path")
    parser.add_argument("--vectorizer", type=Path, default=DEFAULT_VECTORIZER, help="Exported TF-IDF vectorizer joblib path")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST, help="Split manifest JSON path")
    parser.add_argument("--onnx-model", type=Path, default=DEFAULT_ONNX, help="ONNX model path")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Validation report JSON path")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Maximum allowed absolute diff")
    args = parser.parse_args(argv)

    report = validate_onnx_pipeline(
        input_path=args.input,
        pipeline_path=args.pipeline,
        vectorizer_path=args.vectorizer,
        split_manifest_path=args.split_manifest,
        onnx_path=args.onnx_model,
        report_path=args.report,
        threshold=args.threshold,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())