#!/usr/bin/env python3
"""
High-signal bootstrap: amplify seed_batch strong signals into augmented layer.
"""

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.schemas import PERSONALITY_BASIS

DEFAULT_TSV = REPO_ROOT / "outputs" / "personality_text_pairs.tsv"
DEFAULT_JSONL = REPO_ROOT / "outputs" / "personality_text_pairs.jsonl"

# Simple synonym replacements to create variants from seed texts
SYNONYM_PAIRS = [
    ("喜歡", "愛", "最愛", "熱愛"),
    ("討厭", "反感", "不喜歡", "厭惡"),
    ("主導", "掌控", "領導", "帶頭"),
    ("聆聽", "聽取", "接納", "側耳"),
    ("決策", "決定", "拿主意", "定奪"),
    ("被動", "被牽著", "不主動", "配合"),
    ("堅持", "堅定", "固守", "不放"),
    ("放棄", "中止", "停止", "放手"),
    ("冒險", "冒進", "大膽", "激進"),
    ("害怕", "恐懼", "擔心", "提防"),
    ("樂觀", "正面", "積極", "看好"),
    ("悲觀", "消極", "失望", "看衰"),
    ("快速", "迅速", "馬上", "立即"),
    ("緩慢", "慢速", "逐步", "逐漸"),
    ("直接", "率直", "坦白", "明確"),
    ("試試", "嘗試", "體驗", "實驗"),
    ("動", "行動", "出手", "著手"),
    ("想", "思考", "考慮", "琢磨"),
    ("相信", "信任", "依賴", "託付"),
    ("懷疑", "質疑", "不信", "存疑"),
    ("完美", "完整", "全面", "周密"),
    ("簡單", "直接", "易", "輕鬆"),
]


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _apply_synonym_transform(text: str, num_transforms: int = 1) -> list[str]:
    """Apply synonym replacements to create variants."""
    results = [text]
    
    for orig, *syns in SYNONYM_PAIRS:
        if orig not in text:
            continue
        
        for syn in syns:
            variant = text.replace(orig, syn, 1)
            if variant not in results and 1 <= len(variant) <= 30:
                results.append(variant)
                if len(results) >= num_transforms + 1:
                    break
        
        if len(results) >= num_transforms + 1:
            break
    
    return results


def main(argv: list[str] | None = None) -> int:
    import argparse
    
    parser = argparse.ArgumentParser(description="Amplify seed_batch signals")
    parser.add_argument("--target", type=int, default=2000, help="Target new rows")
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    
    session_id = f"bootstrap_signal_amplify_{_now_utc().strftime('%Y%m%d_%H%M%S')}"
    
    # Read existing
    existing_rows = []
    seed_batch_rows = []
    seen = {}
    
    if args.tsv.exists():
        with open(args.tsv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                existing_rows.append(row)
                h = _text_hash(row.get("text", ""))
                seen[h] = row.get("text", "")
                
                if row.get("source") == "seed_batch":
                    seed_batch_rows.append(row)
    
    print(f"[*] Read {len(existing_rows)} existing rows ({len(seed_batch_rows)} seed_batch)", file=sys.stderr)
    
    # Amplify high-signal seed rows
    generated = []
    base_timestamp = _now_utc()
    counter = 0
    
    for seed_row in seed_batch_rows:
        if counter >= args.target:
            break
        
        text = seed_row.get("text", "")
        
        # Check if this is a strong signal
        strong_signals = {}
        for trait in PERSONALITY_BASIS:
            val = float(seed_row.get(trait, 0))
            if abs(val) >= 0.7:
                strong_signals[trait] = val
        
        if not strong_signals:
            continue  # Skip weak signals
        
        # Generate variants via synonym replacement
        variants = _apply_synonym_transform(text, num_transforms=4)
        
        for variant in variants:
            if counter >= args.target:
                break
            
            h = _text_hash(variant)
            if h in seen:
                continue
            
            seen[h] = variant
            
            new_row = {
                "timestamp": (base_timestamp.replace(microsecond=0) + __import__("datetime").timedelta(seconds=counter)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "request_id": uuid4().hex,
                "text": variant,
                "length": len(variant),
                "model": "bootstrap-signal-amplify",
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
    
    print(f"[*] Generated {len(generated)} new rows from {len(seed_batch_rows)} seed rows", file=sys.stderr)
    
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
