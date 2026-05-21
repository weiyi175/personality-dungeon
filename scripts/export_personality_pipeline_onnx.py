from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence
import sys

import numpy as np
from joblib import dump, load
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.personality_mlp_trainer import _load_records
from api.schemas import PERSONALITY_BASIS


DEFAULT_INPUT = REPO_ROOT / "outputs" / "personality_text_pairs.tsv"
DEFAULT_PIPELINE = REPO_ROOT / "outputs" / "mlp_full_pipeline.joblib"
DEFAULT_SPLIT_MANIFEST = REPO_ROOT / "outputs" / "mlp_full_split_manifest.json"
DEFAULT_ONNX = REPO_ROOT / "api" / "personality_model.onnx"
DEFAULT_VECTORIZER = REPO_ROOT / "outputs" / "mlp_full_vectorizer.joblib"
DEFAULT_REPORT = REPO_ROOT / "outputs" / "onnx_export_report.json"
DEFAULT_THRESHOLD = 1e-4


def _select_records_by_hashes(records, hashes: set[str]):
    return [record for record in records if record.text_hash in hashes]


def _predict_texts(pipeline: Any, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, len(PERSONALITY_BASIS)), dtype=np.float64)
    return np.asarray(pipeline.predict(texts), dtype=np.float64)


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


def export_pipeline_to_onnx(
    *,
    pipeline_path: Path,
    input_path: Path,
    split_manifest_path: Path,
    onnx_path: Path,
    vectorizer_path: Path,
    report_path: Path,
    threshold: float,
) -> dict[str, Any]:
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    train_hashes = set(split_manifest["pilot_split"]["train_hashes"])
    val_hashes = set(split_manifest["pilot_split"]["val_hashes"])
    included_sources = split_manifest.get("included_sources")
    input_snapshot_hash = split_manifest.get("input_snapshot_sha256")

    original_pipeline = load(pipeline_path)
    tfidf_step = original_pipeline.named_steps["tfidf"]
    if not isinstance(tfidf_step, TfidfVectorizer):
        raise TypeError(f"Unsupported vectorizer: {type(tfidf_step)!r}")

    mlp_step = original_pipeline.named_steps["mlp"]
    all_records = _load_records(input_path)
    if included_sources is not None:
        included_source_set = set(included_sources)
        all_records = [record for record in all_records if record.source in included_source_set]

    train_records = _select_records_by_hashes(all_records, train_hashes)
    val_records = _select_records_by_hashes(all_records, val_hashes)
    train_texts = [record.text for record in train_records]
    val_texts = [record.text for record in val_records]
    val_features = tfidf_step.transform(val_texts).astype(np.float32).toarray()
    if not hasattr(mlp_step, "predict"):
        raise TypeError(f"Unsupported MLP step: {type(mlp_step)!r}")

    original_val_pred = _predict_texts(original_pipeline, val_texts)
    mlp_val_pred = np.asarray(mlp_step.predict(val_features), dtype=np.float64)
    mirror_vs_original = _summarize_diff(original_val_pred, mlp_val_pred)

    if mirror_vs_original["max_abs_diff"] > threshold:
        raise RuntimeError(
            f"MLP diverges from original pipeline: max_abs_diff={mirror_vs_original['max_abs_diff']:.6g}"
        )

    onnx_model = convert_sklearn(mlp_step, initial_types=[("features", FloatTensorType([None, val_features.shape[1]]))])
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(onnx_model.SerializeToString())
    dump(tfidf_step, vectorizer_path)

    onnx_session = __import__("onnxruntime").InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = onnx_session.get_inputs()[0].name
    onnx_val_pred = np.asarray(onnx_session.run(None, {input_name: val_features})[0], dtype=np.float64)
    onnx_vs_original = _summarize_diff(original_val_pred, onnx_val_pred)

    report: dict[str, Any] = {
        "input_path": str(input_path),
        "input_snapshot_sha256": input_snapshot_hash,
        "pipeline_path": str(pipeline_path),
        "vectorizer_path": str(vectorizer_path),
        "onnx_path": str(onnx_path),
        "split_manifest_path": str(split_manifest_path),
        "feature_count": int(val_features.shape[1]),
        "train_count": len(train_records),
        "validation_count": len(val_records),
        "threshold": threshold,
        "mirror_vs_original": mirror_vs_original,
        "onnx_vs_original": onnx_vs_original,
        "passed": True,
        "sample_texts": val_texts[:5],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the trained personality MLP to ONNX")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input TSV or JSONL path")
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE, help="Sklearn pipeline joblib path")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST, help="Split manifest JSON path")
    parser.add_argument("--output", type=Path, default=DEFAULT_ONNX, help="Output ONNX path")
    parser.add_argument("--vectorizer-output", type=Path, default=DEFAULT_VECTORIZER, help="Exported TF-IDF vectorizer joblib path")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Export report JSON path")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Maximum allowed absolute diff")
    args = parser.parse_args(argv)

    result = export_pipeline_to_onnx(
        pipeline_path=args.pipeline,
        input_path=args.input,
        split_manifest_path=args.split_manifest,
        onnx_path=args.output,
        vectorizer_path=args.vectorizer_output,
        report_path=args.report,
        threshold=args.threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())