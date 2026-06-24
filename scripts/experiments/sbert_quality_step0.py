"""Step 0 — quantify SBERT-9D quality on REAL human wills (the radar readout).

User verdict: the radar readout (= raw `will_sbert_vector` = output of
api.personality_sbert_inference.infer_personality_vector_sbert) does not match
the will's meaning. This harness turns that gut feeling into per-trait numbers.

What runs OFFLINE (no LLM, works now)
- load real human wills (p7h_player_test_sessions.json, is_human + non-empty)
- recompute SBERT-9D offline; assert it reproduces stored will_sbert_vector
- per-trait distribution stats (range utilisation / collapse check)
- face-validity review sheet (will_text + 9 traits) for human eyeballing
- a FROZEN seeded held-out split over unique wills, so future fixes
  (rebalance / encoder-swap / SetFit) are evaluated without leakage

What needs an LLM endpoint (per-trait R^2/MAE vs teacher)
- set PERSONALITY_LLM_BASE_URL + PERSONALITY_LLM_MODEL, then this script also
  runs the teacher on each unique will and reports per-trait R^2/MAE + the
  worst SBERT-vs-teacher disagreements. Without it, that block is skipped.

WRITE SAFETY: only reads p7h json + models; writes ONLY under
reports/experiments/sbert_quality/. Does not import any singleton-persistence
module (no ecology_tracker), so nothing can clobber ecology_state.json.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from api.personality_sbert_inference import infer_personality_vector_sbert  # noqa: E402
from api.schemas import PERSONALITY_BASIS  # noqa: E402

SESSIONS = ROOT / "reports/experiments/p7h_real_study/p7h_player_test_sessions.json"
OUT_DIR = ROOT / "reports/experiments/sbert_quality"
SPLIT_SEED = 42
HELD_OUT_RATIO = 0.2

SHORT = ("imp", "ast", "opt", "rsk", "sus", "end", "rnd", "stb", "cur")


def load_real_wills() -> list[dict]:
    """Unique human wills with offline SBERT-9D, dedup by will_text."""
    sessions = json.loads(SESSIONS.read_text())["sessions"]
    by_text: dict[str, dict] = {}
    for rec in sessions.values():
        if not rec.get("is_human"):
            continue
        wt = (rec.get("will_text") or "").strip()
        sb = rec.get("will_sbert_vector")
        if not wt or not (isinstance(sb, list) and len(sb) == 9):
            continue
        entry = by_text.setdefault(wt, {"will_text": wt, "count": 0, "stored": sb})
        entry["count"] += 1
    return list(by_text.values())


def predict(entries: list[dict]) -> np.ndarray:
    """Offline SBERT-MLP 9D per will; assert it reproduces stored vector."""
    preds = np.zeros((len(entries), 9))
    max_drift = 0.0
    for i, e in enumerate(entries):
        vec, _ = infer_personality_vector_sbert(e["will_text"])
        preds[i] = [vec[t] for t in PERSONALITY_BASIS]
        max_drift = max(max_drift, float(np.abs(preds[i] - np.array(e["stored"])).max()))
    print(f"[reproduce] max |offline - stored| over {len(entries)} wills = {max_drift:.4f}")
    if max_drift > 1e-3:
        print("  WARNING: offline recompute differs from stored will_sbert_vector")
    return preds


def distribution_report(preds: np.ndarray) -> dict:
    rep = {}
    print("\n=== per-trait distribution on REAL wills (N=%d unique) ===" % len(preds))
    print(f"{'trait':<18} {'mean':>7} {'std':>7} {'min':>7} {'max':>7} {'range':>7}")
    for j, t in enumerate(PERSONALITY_BASIS):
        col = preds[:, j]
        lo, hi = float(col.min()), float(col.max())
        rep[t] = {"mean": float(col.mean()), "std": float(col.std()),
                  "min": lo, "max": hi, "range": hi - lo}
        print(f"{t:<18} {col.mean():>7.3f} {col.std():>7.3f} {lo:>7.3f} {hi:>7.3f} {hi-lo:>7.3f}")
    return rep


def face_validity_sheet(entries: list[dict], preds: np.ndarray) -> str:
    """Markdown: will_text + 9 traits + compact top+/top- summary."""
    order = sorted(range(len(entries)), key=lambda i: -preds[i, 1])  # by assertiveness
    lines = ["# SBERT-9D face-validity sheet (real wills)\n",
             "Mark any row where the traits don't match the will. Codes: "
             + " ".join(f"{s}={t}" for s, t in zip(SHORT, PERSONALITY_BASIS)) + "\n",
             "| n | will_text | " + " | ".join(SHORT) + " | reads as |",
             "|--:|---|" + "|".join(["--:"] * 9) + "|---|"]
    for i in order:
        v = preds[i]
        top_pos = PERSONALITY_BASIS[int(np.argmax(v))]
        top_neg = PERSONALITY_BASIS[int(np.argmin(v))]
        cells = " | ".join(f"{x:+.2f}" for x in v)
        lines.append(f"| {entries[i]['count']} | {entries[i]['will_text']} | {cells} "
                     f"| +{top_pos} / -{top_neg} |")
    return "\n".join(lines)


def frozen_split(entries: list[dict]) -> dict:
    texts = sorted(e["will_text"] for e in entries)
    rng = np.random.RandomState(SPLIT_SEED)
    idx = rng.permutation(len(texts))
    n_test = max(1, round(len(texts) * HELD_OUT_RATIO))
    test = sorted(texts[i] for i in idx[:n_test])
    train = sorted(texts[i] for i in idx[n_test:])
    print(f"\n[split] frozen seed={SPLIT_SEED}: train_eligible={len(train)} held_out_test={len(test)}")
    return {"seed": SPLIT_SEED, "ratio": HELD_OUT_RATIO,
            "held_out_test": test, "train_eligible": train}


def teacher_block(entries: list[dict], preds: np.ndarray) -> dict | None:
    if not (os.getenv("PERSONALITY_LLM_BASE_URL") and os.getenv("PERSONALITY_LLM_MODEL")):
        print("\n[teacher] SKIPPED — set PERSONALITY_LLM_BASE_URL + PERSONALITY_LLM_MODEL "
              "to compute per-trait R^2/MAE vs teacher.")
        return None
    from api.personality_text_inference import infer_personality_vector  # noqa: E402
    print(f"\n[teacher] running on {len(entries)} unique wills ...")
    ref = np.zeros_like(preds)
    for i, e in enumerate(entries):
        vec, _ = infer_personality_vector(e["will_text"])
        ref[i] = [vec[t] for t in PERSONALITY_BASIS]
    out = {"per_trait": {}}
    print(f"{'trait':<18} {'R2':>7} {'MAE':>7}")
    r2s = []
    for j, t in enumerate(PERSONALITY_BASIS):
        y, yh = ref[:, j], preds[:, j]
        ss_res = float(((y - yh) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-12
        r2 = 1 - ss_res / ss_tot
        mae = float(np.abs(y - yh).mean())
        out["per_trait"][t] = {"r2": r2, "mae": mae}
        r2s.append(r2)
        print(f"{t:<18} {r2:>7.3f} {mae:>7.3f}")
    out["average_r2"] = float(np.mean(r2s))
    print(f"{'AVERAGE':<18} {out['average_r2']:>7.3f}")
    # worst disagreements
    err = np.abs(ref - preds).mean(axis=1)
    worst = np.argsort(-err)[:15]
    out["worst_disagreements"] = [
        {"will": entries[i]["will_text"], "mae": float(err[i])} for i in worst]
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = load_real_wills()
    print(f"loaded {len(entries)} unique real human wills "
          f"({sum(e['count'] for e in entries)} records)")
    preds = predict(entries)
    dist = distribution_report(preds)
    split = frozen_split(entries)
    sheet = face_validity_sheet(entries, preds)
    teacher = teacher_block(entries, preds)

    (OUT_DIR / "face_validity_sheet.md").write_text(sheet)
    (OUT_DIR / "held_out_split.json").write_text(json.dumps(split, ensure_ascii=False, indent=1))
    summary = {"n_unique_wills": len(entries),
               "n_records": sum(e["count"] for e in entries),
               "trait_distribution": dist,
               "teacher_eval": teacher}
    (OUT_DIR / "step0_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\nwrote: {OUT_DIR}/face_validity_sheet.md, held_out_split.json, step0_summary.json")


if __name__ == "__main__":
    main()
