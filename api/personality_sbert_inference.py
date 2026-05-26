"""Offline SBERT + v7 MLP personality inference (no LLM / no network).

Loads paraphrase-multilingual-MiniLM-L12-v2 and outputs/mlp_v7_mlp.joblib
on first call (lazy, thread-safe).  Returns the same (vector, meta) tuple
shape as personality_text_inference.infer_personality_vector so the API
layer can reuse the same response model.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

import joblib  # type: ignore
import numpy as np

from api.schemas import PERSONALITY_BASIS

REPO_ROOT = Path(__file__).resolve().parents[1]

_SBERT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_MLP_PATH = REPO_ROOT / "outputs" / "mlp_v7_mlp.joblib"
_MODEL_TAG = "sbert-mlp-v7"
_MAX_CHARS = 20

_lock = threading.Lock()
_sbert = None
_mlp = None


def _ensure_loaded() -> None:
    global _sbert, _mlp
    if _sbert is not None:
        return
    with _lock:
        if _sbert is not None:
            return
        from sentence_transformers import SentenceTransformer  # type: ignore
        _sbert = SentenceTransformer(_SBERT_MODEL_NAME)
        _mlp = joblib.load(_MLP_PATH)


def infer_personality_vector_sbert(
    text: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Return (vector, meta) for *text* using SBERT embeddings + v7 MLP.

    Compatible with the (vector, meta) interface of
    personality_text_inference.infer_personality_vector so the caller only
    needs to swap the function reference.

    Raises
    ------
    ValueError
        If *text* is empty or exceeds _MAX_CHARS.
    RuntimeError
        If the MLP model file cannot be loaded.
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("text must not be empty")
    if len(cleaned) > _MAX_CHARS:
        raise ValueError(f"text exceeds max length {_MAX_CHARS}")

    try:
        _ensure_loaded()
    except Exception as exc:
        raise RuntimeError(f"Failed to load SBERT/MLP models: {exc}") from exc

    emb: np.ndarray = _sbert.encode(  # type: ignore[union-attr]
        [cleaned],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    pred: np.ndarray = _mlp.predict(emb)[0]  # type: ignore[union-attr]  # shape (9,)

    vector = {
        trait: float(pred[i])
        for i, trait in enumerate(PERSONALITY_BASIS)
    }
    meta: dict[str, Any] = {
        "request_id": uuid.uuid4().hex,
        "model": _MODEL_TAG,
        "temperature": 0.0,
        "length": len(cleaned),
    }
    return vector, meta
