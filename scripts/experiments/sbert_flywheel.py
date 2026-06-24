"""Flywheel: new real wills -> Opus auto-label -> retrain bge-zh head -> ship.

As multiplayer testing collects new real player wills, run this to:
  1. gather wills NOT yet labelled (scan all p7h sources, dedupe, subtract known)
  2. label each with Claude Opus 4.8 using the exact 9-trait rubric
  3. merge into reports/experiments/sbert_quality/opus_train_labels.json
  4. retrain the frozen-bge-zh + MLP head on all labels EXCEPT the frozen 35-will
     held-out test, report held-out mean Pearson r, and (unless --no-save) write
     a new outputs/mlp_opus_bgezh_v*.joblib

The 35 held-out wills (claude_reference_labels.json) are NEVER trained on, so the
reported r stays an honest generalisation estimate as the training set grows.

Usage:
  python scripts/experiments/sbert_flywheel.py --dry-run     # list new wills only (no API, no SDK needed)
  python scripts/experiments/sbert_flywheel.py --label-only  # label + merge, skip retrain
  python scripts/experiments/sbert_flywheel.py               # label + merge + retrain + save

Requires for labelling: pip install anthropic ; ANTHROPIC_API_KEY set.
Read-only except the two json files + outputs/mlp_opus_bgezh_v*.joblib.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SB = ROOT / "reports/experiments/sbert_quality"
TRAIN = SB / "opus_train_labels.json"
TEST = SB / "claude_reference_labels.json"
ENCODER = "BAAI/bge-base-zh-v1.5"
MODEL = "claude-opus-4-8"
TRAITS = ("impulsiveness", "assertiveness", "optimism", "risk_aversion", "suspicion",
          "endurance", "randomness", "stability_seeking", "curiosity")

RUBRIC = (
    "You label a dungeon-game 'will' — a short Chinese intention (<=20 chars) a player "
    "writes for their character — on 9 personality traits, each a float in [-1, 1]. "
    "Judge from the will's meaning; use the full range; be decisive.\n"
    "- impulsiveness: rash/frenzied/no-planning (+) vs deliberate/planned/patient (-)\n"
    "- assertiveness: aggressive/dominant/attacks/leads (+) vs passive/avoidant/follows (-)\n"
    "- optimism: hopeful/positive/life-affirming (+) vs pessimistic/dark/death/cynical (-)\n"
    "- risk_aversion: cautious/safety-first/defensive/retreats (+) vs reckless/high-risk (-)\n"
    "- suspicion: distrustful/sneaky/exploits/betrays/ambushes (+) vs trusting/cooperative/friendly (-)\n"
    "- endurance: persistence/grind/training/preparation/resilience (+) vs gives-up/quits/fragile (-)\n"
    "- randomness: chaotic/whimsical/aimless/let-fate-decide (+) vs systematic/orderly/structured (-)\n"
    "- stability_seeking: security/steady/settle/protect/comfort (+) vs change/upheaval/novelty (-)\n"
    "- curiosity: explores-unknown/investigates/learns/asks-why (+) vs incurious/repetitive (-)"
)
SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {t: {"type": "number"} for t in TRAITS},
          "required": list(TRAITS)}


def gather_new_wills() -> list[str]:
    known = set(json.loads(TRAIN.read_text())) | set(json.loads(TEST.read_text()))
    found: set[str] = set()
    for f in glob.glob(str(ROOT / "reports/**/*.json"), recursive=True):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue

        def walk(o):
            if isinstance(o, dict):
                if isinstance(o.get("will_text"), str) and o.get("is_human", True) \
                        and not str(o.get("session_id", "")).startswith("wsim"):
                    found.add(o["will_text"].strip())
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(d)
    return sorted(w for w in found if w and w not in known and len(w) <= 20)


def opus_label(client, will: str) -> list[float]:
    resp = client.messages.create(
        model=MODEL, max_tokens=300, system=RUBRIC,
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": f"will: {will}"}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    obj = json.loads(text)
    return [max(-1.0, min(1.0, float(obj[t]))) for t in TRAITS]


def retrain_and_eval(save: bool) -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.neural_network import MLPRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from joblib import dump

    tr = json.loads(TRAIN.read_text()); te = json.loads(TEST.read_text())
    train_only = {w: v for w, v in tr.items() if w not in te}  # never train on the 35 test
    enc = SentenceTransformer(ENCODER)
    emb = lambda ws: enc.encode(ws, convert_to_numpy=True, normalize_embeddings=False)

    trw = list(train_only); Xtr = emb(trw); Ytr = np.array([train_only[w] for w in trw])
    tew = list(te); Xte = emb(tew); Yte = np.array([te[w] for w in tew])
    mk = lambda: MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(256, 128),
                                                   max_iter=800, random_state=42))

    def pear(a, b):
        a, b = a - a.mean(), b - b.mean(); d = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / d) if d > 1e-12 else float("nan")

    P = mk().fit(Xtr, Ytr).predict(Xte)
    per = {t: round(pear(Yte[:, j], P[:, j]), 2) for j, t in enumerate(TRAITS)}
    print(f"held-out (train {len(trw)} / eval {len(tew)}): mean_r = "
          f"{round(float(np.nanmean(list(per.values()))), 3)}  per-trait {per}")
    if save:
        Xall = np.vstack([Xtr, Xte]); Yall = np.vstack([Ytr, Yte])
        n = 1 + len(glob.glob(str(ROOT / "outputs/mlp_opus_bgezh_v*.joblib")))
        out = ROOT / f"outputs/mlp_opus_bgezh_v{n}.joblib"
        dump(mk().fit(Xall, Yall), out)
        print(f"saved {out} (trained on all {len(trw) + len(tew)}) — point "
              f"api/personality_sbert_inference.py:_MLP_PATH at it + restart server")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--label-only", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    new = gather_new_wills()
    print(f"new unlabelled real wills: {len(new)}")
    for w in new:
        print(f"  + {w}")
    if args.dry_run:
        return
    if new:
        import anthropic  # lazy; only needed when actually labelling
        client = anthropic.Anthropic()
        labels = json.loads(TRAIN.read_text())
        for i, w in enumerate(new, 1):
            labels[w] = opus_label(client, w)
            print(f"  labelled {i}/{len(new)}: {w}")
            time.sleep(0.2)
        TRAIN.write_text(json.dumps(labels, ensure_ascii=False, indent=1))
        print(f"merged {len(new)} new labels into {TRAIN.name}")
    if not args.label_only:
        retrain_and_eval(save=not args.no_save)


if __name__ == "__main__":
    main()
