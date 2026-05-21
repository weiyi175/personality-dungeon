#!/usr/bin/env python3
"""
Rebalancing bootstrap: prioritize weak traits and short texts.
"""

import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.schemas import PERSONALITY_BASIS

DEFAULT_TSV = REPO_ROOT / "outputs" / "personality_text_pairs.tsv"
DEFAULT_JSONL = REPO_ROOT / "outputs" / "personality_text_pairs.jsonl"

# Weak traits that need more representation
WEAK_TRAITS = {
    "assertiveness": ["neg", "pos"],
    "optimism": ["neg", "pos"],
    "endurance": ["neg", "pos"],
    "curiosity": ["neg", "pos"],
    "suspicion": ["neg", "pos"],
}

# Additional short-text focused transforms
SHORT_TEXT_TEMPLATES = {
    ("assertiveness", "pos"): [
        "主導", "決定", "帶頭", "領導", "掌控",
        "說話", "發聲", "表達", "大聲", "直言",
        "影響", "推動", "引導", "指揮", "帶隊",
    ],
    ("assertiveness", "neg"): [
        "退讓", "讓步", "聽從", "安靜", "配合",
        "聽取", "接納", "服從", "沉默", "側耳",
        "尊重", "遵循", "適應", "調整", "順應",
    ],
    ("optimism", "pos"): [
        "樂觀", "希望", "積極", "正面", "相信",
        "可能", "成功", "會好", "開放", "歡迎",
        "進步", "改善", "向上", "安心", "放心",
    ],
    ("optimism", "neg"): [
        "悲觀", "失望", "消極", "負面", "懷疑",
        "不行", "困難", "麻煩", "害怕", "擔心",
        "謹慎", "警惕", "預期差", "想壞", "防範",
    ],
    ("endurance", "pos"): [
        "堅持", "堅定", "固守", "不放", "維持",
        "耐力", "決心", "韌性", "毅力", "持久",
        "長期", "延續", "推進", "不退", "堅守",
    ],
    ("endurance", "neg"): [
        "放棄", "放鬆", "停止", "放手", "中止",
        "易敗", "短期", "臨時", "一時", "鬆懈",
        "彈性", "變通", "靈活", "調整", "妥協",
    ],
    ("curiosity", "pos"): [
        "好奇", "探索", "研究", "學習", "發現",
        "未知", "新鮮", "興趣", "投入", "沉浸",
        "開放", "接納", "歡迎", "尋求", "求知",
    ],
    ("curiosity", "neg"): [
        "保守", "傳統", "守舊", "熟悉", "安心",
        "依賴", "相信", "仰仗", "既定", "安全",
        "簡單", "直接", "基本", "不深究", "表面",
    ],
    ("suspicion", "pos"): [
        "多疑", "警惕", "懷疑", "不信", "防範",
        "察言", "觀察", "留意", "注視", "監視",
        "隱藏", "保留", "防守", "疏遠", "距離",
    ],
    ("suspicion", "neg"): [
        "信任", "相信", "依賴", "託付", "開放",
        "坦白", "透露", "分享", "親近", "靠近",
        "樂觀", "正面", "積極", "希望", "歡迎",
    ],
}


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _read_existing(tsv_path: Path) -> dict[str, dict[str, Any]]:
    """Read existing rows. Returns dict[text_hash] -> row."""
    existing = {}
    if not tsv_path.exists():
        return existing
    
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = row.get("text", "")
            h = _text_hash(text)
            existing[h] = row
    
    return existing


def _generate_rebalanced(existing_rows: list[dict[str, Any]], session_id: str, target: int = 3000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate rebalanced rows focusing on weak traits and short texts."""
    generated = []
    seen = {_text_hash(row["text"]): row["text"] for row in existing_rows}
    
    # Group existing by trait+endpoint
    grouped = defaultdict(list)
    for row in existing_rows:
        for trait in PERSONALITY_BASIS:
            val = float(row.get(trait, 0.0))
            if trait in [t for t, _ in WEAK_TRAITS.items()]:
                if val > 0.3:
                    endpoint = "pos"
                elif val < -0.3:
                    endpoint = "neg"
                else:
                    endpoint = "neutral"
                grouped[(trait, endpoint)].append(row)
    
    base_timestamp = _now_utc()
    counter = 0
    
    # Priority: weak traits first
    for trait in WEAK_TRAITS:
        if counter >= target:
            break
        
        for endpoint in WEAK_TRAITS[trait]:
            if counter >= target:
                break
            
            if (trait, endpoint) not in grouped:
                continue
            
            base_rows = grouped[(trait, endpoint)]
            
            # Generate short-text variants for this trait+endpoint
            for template_idx, template in enumerate(SHORT_TEXT_TEMPLATES.get((trait, endpoint), [])):
                if counter >= target:
                    break
                
                # Try different lengths
                for prefix in ["", "我", "我會", "我常", "我也", "我總"]:
                    if counter >= target:
                        break
                    
                    variant = (prefix + template).strip()
                    
                    # Only accept if within length bounds
                    if not (1 <= len(variant) <= 25):
                        continue
                    
                    h = _text_hash(variant)
                    if h in seen:
                        continue
                    
                    seen[h] = variant
                    
                    # Use first matching base row for labels
                    if not base_rows:
                        continue
                    
                    base_row = base_rows[0]
                    new_row = {
                        "timestamp": (base_timestamp.replace(microsecond=0) + __import__("datetime").timedelta(seconds=counter)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "request_id": uuid4().hex,
                        "text": variant,
                        "length": len(variant),
                        "model": "bootstrap-rebalance",
                        "temperature": "0.35",
                        "source": "augmented",
                        "session_id": session_id,
                        "user_id": None,
                    }
                    
                    # Copy personality vector
                    for trait_name in PERSONALITY_BASIS:
                        new_row[trait_name] = base_row.get(trait_name, "0.0")
                    
                    generated.append(new_row)
                    counter += 1
    
    return generated, {
        "session_id": session_id,
        "target": target,
        "generated": counter,
        "focus_traits": list(WEAK_TRAITS.keys()),
        "unique_count": len(seen),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    
    parser = argparse.ArgumentParser(description="Rebalance bootstrap for weak traits")
    parser.add_argument("--target", type=int, default=3000, help="Target new rows")
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    
    session_id = f"bootstrap_rebalance_{_now_utc().strftime('%Y%m%d_%H%M%S')}"
    
    # Read existing
    existing_rows = []
    if args.tsv.exists():
        with open(args.tsv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                existing_rows.append(row)
    
    print(f"[*] Read {len(existing_rows)} existing rows", file=sys.stderr)
    
    # Generate rebalanced
    generated, stats = _generate_rebalanced(existing_rows, session_id, args.target)
    print(f"[*] Generated {len(generated)} new rows (target was {args.target})", file=sys.stderr)
    
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
    
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
