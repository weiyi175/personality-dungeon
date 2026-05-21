#!/usr/bin/env python3
"""
Direct duplication: copy high-signal seed rows with minor textual variations.
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


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _get_variants(text: str, num: int = 5) -> list[str]:
    """Generate simple textual variants of the same meaning."""
    variants = [text]
    
    # Remove trailing punctuation
    if text.endswith(("。", "!", "，")):
        variants.append(text.rstrip("。!，"))
    
    # Remove leading/trailing spaces
    if text != text.strip():
        variants.append(text.strip())
    
    # Add soft emphasis markers (common in Chinese)
    if not text.endswith(("啊", "呢")):
        variants.append(text + "啊")
        variants.append(text + "呢")
    
    # Rephrase slightly
    if text.startswith("我"):
        variants.append(text[1:].lstrip())  # remove "我"
        variants.append("別人覺得我" + text)  # wrap with "別人覺得我"
    else:
        variants.append("我" + text)
    
    # Deduplicate and limit
    unique_variants = []
    seen = set()
    for v in variants:
        if 1 <= len(v) <= 30 and v not in seen:
            seen.add(v)
            unique_variants.append(v)
            if len(unique_variants) >= num:
                break
    
    return unique_variants


def main(argv: list[str] | None = None) -> int:
    import argparse
    
    parser = argparse.ArgumentParser(description="Duplicate high-signal seeds")
    parser.add_argument("--ratio", type=int, default=10, help="Variants per seed")
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    
    session_id = f"bootstrap_direct_dup_{_now_utc().strftime('%Y%m%d_%H%M%S')}"
    
    # Read existing
    existing_rows = []
    seed_batch_strong = []
    seen = {}
    
    if args.tsv.exists():
        with open(args.tsv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                existing_rows.append(row)
                h = _text_hash(row.get("text", ""))
                seen[h] = row.get("text", "")
                
                if row.get("source") == "seed_batch":
                    max_sig = max(abs(float(row.get(trait, 0))) for trait in PERSONALITY_BASIS)
                    if max_sig >= 0.75:
                        seed_batch_strong.append(row)
    
    print(f"[*] Read {len(existing_rows)} rows, found {len(seed_batch_strong)} strong seed_batch", file=sys.stderr)
    
    # Generate variants
    generated = []
    base_timestamp = _now_utc()
    counter = 0
    
    for seed_row in seed_batch_strong:
        text = seed_row.get("text", "")
        variants = _get_variants(text, num=args.ratio)
        
        for variant in variants[1:]:  # Skip first (original)
            h = _text_hash(variant)
            if h in seen:
                continue
            
            seen[h] = variant
            
            new_row = {
                "timestamp": (base_timestamp.replace(microsecond=0) + __import__("datetime").timedelta(seconds=counter)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "request_id": uuid4().hex,
                "text": variant,
                "length": len(variant),
                "model": "bootstrap-direct-dup",
                "temperature": "0.35",
                "source": "augmented",
                "session_id": session_id,
                "user_id": None,
            }
            
            # Copy personality vector (preserve strong signals)
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
