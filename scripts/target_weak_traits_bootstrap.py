#!/usr/bin/env python3
"""
Target weak traits: copy seed_batch rows specific to weak traits multiple times.
"""

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.schemas import PERSONALITY_BASIS

DEFAULT_TSV = REPO_ROOT / "outputs" / "personality_text_pairs.tsv"
DEFAULT_JSONL = REPO_ROOT / "outputs" / "personality_text_pairs.jsonl"

WEAK_TRAITS = {"assertiveness", "optimism", "suspicion", "endurance", "curiosity"}


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _simple_variants(text: str, n: int = 5) -> list[str]:
    """Very simple: just add/remove punctuation and slight rewording."""
    variants = []
    
    # Base
    variants.append(text)
    
    # Remove punctuation
    for c in "。！，、；：":
        if c in text:
            variants.append(text.replace(c, ""))
    
    # Add punctuation
    if not text.endswith(("。", "!")):
        variants.append(text + "。")
        variants.append(text + "!")
    
    # Prefix variations (just for short texts)
    if len(text) <= 15:
        for prefix in ["", "我", "我會", "我覺得", "據我", "一般來說"]:
            if prefix:
                v = (prefix + text).strip()
                if 1 <= len(v) <= 30:
                    variants.append(v)
    
    # Deduplicate
    unique = []
    seen = set()
    for v in variants:
        if v not in seen and 1 <= len(v) <= 30:
            seen.add(v)
            unique.append(v)
    
    return unique[:n]


def main(argv: list[str] | None = None) -> int:
    import argparse
    
    parser = argparse.ArgumentParser(description="Target weak traits")
    parser.add_argument("--ratio", type=int, default=5, help="Variants per row")
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    
    session_id = f"bootstrap_weak_traits_{_now_utc().strftime('%Y%m%d_%H%M%S')}"
    
    # Read existing
    existing_rows = []
    seed_batch_weak = []
    seen = {}
    
    if args.tsv.exists():
        with open(args.tsv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                existing_rows.append(row)
                h = _text_hash(row.get("text", ""))
                seen[h] = row.get("text", "")
                
                # Find seed_batch rows with ANY signal in weak traits
                if row.get("source") == "seed_batch":
                    for trait in WEAK_TRAITS:
                        val = float(row.get(trait, 0))
                        if abs(val) >= 0.6:
                            seed_batch_weak.append(row)
                            break
    
    print(f"[*] Read {len(existing_rows)} rows", file=sys.stderr)
    print(f"[*] Found {len(seed_batch_weak)} seed_batch rows for weak traits", file=sys.stderr)
    
    # Generate variants
    generated = []
    base_timestamp = _now_utc()
    counter = 0
    
    for seed_row in seed_batch_weak:
        text = seed_row.get("text", "")
        variants = _simple_variants(text, n=args.ratio)
        
        for variant in variants:
            h = _text_hash(variant)
            if h in seen:
                continue
            
            seen[h] = variant
            
            new_row = {
                "timestamp": (base_timestamp.replace(microsecond=0) + __import__("datetime").timedelta(seconds=counter)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "request_id": uuid4().hex,
                "text": variant,
                "length": len(variant),
                "model": "bootstrap-weak-traits",
                "temperature": "0.35",
                "source": "augmented",
                "session_id": session_id,
                "user_id": None,
            }
            
            # Copy personality vector
            for trait in PERSONALITY_BASIS:
                new_row[trait] = seed_row.get(trait, "0.0")
            
            generated.append(new_row)
            counter += 1
    
    print(f"[*] Generated {len(generated)} variants", file=sys.stderr)
    
    # Write
    if not args.dry_run:
        with open(args.tsv, "a", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            for row in generated:
                values = []
                for key in ["timestamp", "request_id", "text", "length", "model", "temperature", "source", "session_id", "user_id"]:
                    values.append(row.get(key, ""))
                for trait in PERSONALITY_BASIS:
                    values.append(row.get(trait, "0.0"))
                writer.writerow(values)
        
        with open(args.jsonl, "a", encoding="utf-8") as f:
            for row in generated:
                obj = {
                    "timestamp": row["timestamp"],
                    "request_id": row["request_id"],
                    "text": row["text"],
                    "length": row["length"],
                    "model": row["model"],
                    "temperature": float(row["temperature"]),
                    "source": row["source"],
                    "session_id": row["session_id"],
                    "user_id": row["user_id"],
                    "vector": {trait: float(row.get(trait, 0.0)) for trait in PERSONALITY_BASIS},
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        
        print(f"[+] Wrote {len(generated)} rows", file=sys.stderr)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
