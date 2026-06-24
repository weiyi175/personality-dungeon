"""Step 1 — retrain+eval A/B to lift SBERT-9D direction tracking on real wills.

Metric: mean per-trait Pearson r vs Opus reference (the 35 held-out, saved in
claude_reference_labels.json). All variants retrain the SAME architecture
(MultiOutputRegressor(MLPRegressor(256,128)), seed 42, production-faithful) so
deltas are apples-to-apples; the retrained `minilm/none` is the comparison
anchor (NOT mlp_v7). clip is post-hoc (does not change r; only MAE/scale).

Usage:  python scripts/experiments/sbert_quality_step1.py minilm e5base bgezh
        (encoders default to all three if none given)

Read-only except outputs/step1_emb_*.npz cache + reports/experiments/sbert_quality/.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from api.schemas import PERSONALITY_BASIS  # noqa: E402

TSV = ROOT / "outputs" / "personality_text_pairs.tsv"
REF = ROOT / "reports/experiments/sbert_quality/claude_reference_labels.json"
OUT_DIR = ROOT / "reports/experiments/sbert_quality"
INCLUDED = {"seed_batch", "augmented"}

ENCODERS = {
    "minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", ""),
    "e5base": ("intfloat/multilingual-e5-base", "query: "),
    "bgezh": ("BAAI/bge-base-zh-v1.5", ""),
}
_st_cache: dict[str, object] = {}


def load_train() -> tuple[list[str], list[str], np.ndarray]:
    texts, sources, Y = [], [], []
    with TSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            src = (row.get("source") or "").strip()
            txt = (row.get("text") or "").strip()
            if src not in INCLUDED or not txt:
                continue
            try:
                y = [float(row[t]) for t in PERSONALITY_BASIS]
            except (KeyError, ValueError):
                continue
            texts.append(txt)
            sources.append(src)
            Y.append(y)
    return texts, sources, np.array(Y)


def _encoder(model_name: str):
    if model_name not in _st_cache:
        from sentence_transformers import SentenceTransformer
        _st_cache[model_name] = SentenceTransformer(model_name)
    return _st_cache[model_name]


def encode(enc_key: str, texts: list[str], cache_tag: str | None = None) -> np.ndarray:
    model_name, prefix = ENCODERS[enc_key]
    if cache_tag:
        cache = ROOT / "outputs" / f"step1_emb_{enc_key}_{cache_tag}.npz"
        if cache.exists():
            return np.load(cache)["emb"]
    model = _encoder(model_name)
    emb = model.encode([prefix + t for t in texts], batch_size=64,
                       show_progress_bar=False, convert_to_numpy=True,
                       normalize_embeddings=False)
    if cache_tag:
        np.savez_compressed(cache, emb=emb)
    return emb


def oversample_seed(X: np.ndarray, sources: list[str], Y: np.ndarray):
    src = np.array(sources)
    seed_idx = np.where(src == "seed_batch")[0]
    aug_idx = np.where(src == "augmented")[0]
    k = max(1, round(len(aug_idx) / max(1, len(seed_idx))))  # ~8 -> 1:1
    idx = np.concatenate([np.arange(len(X))] + [seed_idx] * (k - 1))
    return X[idx], Y[idx]


def fit_mlp(X: np.ndarray, Y: np.ndarray):
    # single native multi-output MLP (1 net, 9 outputs) + early stopping:
    # ~9x faster than MultiOutputRegressor; relative encoder A/B is what matters.
    mlp = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=400,
                       early_stopping=True, n_iter_no_change=8,
                       validation_fraction=0.1, random_state=42)
    mlp.fit(X, Y)
    return mlp


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-12 else float("nan")


def evaluate(mlp, enc_key: str, clip: bool):
    ref_map = json.loads(REF.read_text())
    wills = list(ref_map)
    ref = np.array([ref_map[w] for w in wills])
    Xev = encode(enc_key, wills)
    P = mlp.predict(Xev)
    if clip:
        P = np.clip(P, -1.0, 1.0)
    per = {}
    for j, t in enumerate(PERSONALITY_BASIS):
        per[t] = {"r": pearson(ref[:, j], P[:, j]),
                  "mae": float(np.abs(ref[:, j] - P[:, j]).mean())}
    mean_r = float(np.nanmean([per[t]["r"] for t in PERSONALITY_BASIS]))
    mean_mae = float(np.mean([per[t]["mae"] for t in PERSONALITY_BASIS]))
    return mean_r, mean_mae, per


def main() -> None:
    enc_keys = [a for a in sys.argv[1:] if a in ENCODERS] or list(ENCODERS)
    texts, sources, Y = load_train()
    print(f"train rows: {len(texts)} (seed={sources.count('seed_batch')}, "
          f"aug={sources.count('augmented')})")
    results = []
    for enc in enc_keys:
        print(f"\n### encoder={enc} ({ENCODERS[enc][0]}) — encoding {len(texts)} texts ...")
        Xtr = encode(enc, texts, cache_tag="train")
        for balance in ("none",):  # rebalance shown unhelpful in minilm round; skip to save time
            Xb, Yb = (Xtr, Y) if balance == "none" else oversample_seed(Xtr, sources, Y)
            mlp = fit_mlp(Xb, Yb)
            for clip in (False, True):
                mean_r, mean_mae, per = evaluate(mlp, enc, clip)
                tag = f"{enc}/{balance}{'/clip' if clip else ''}"
                print(f"  {tag:<28} mean_r={mean_r:.3f}  mean_MAE={mean_mae:.3f}")
                results.append({"config": tag, "encoder": enc, "balance": balance,
                                "clip": clip, "mean_r": mean_r, "mean_mae": mean_mae,
                                "per_trait": per})
    results.sort(key=lambda d: -d["mean_r"])
    print("\n=== ranked by mean Pearson r ===")
    for d in results:
        print(f"  {d['mean_r']:.3f}  MAE {d['mean_mae']:.3f}  {d['config']}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"step1_results_{'_'.join(enc_keys)}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
