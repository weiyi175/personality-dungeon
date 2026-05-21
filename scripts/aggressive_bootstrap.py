#!/usr/bin/env python3
"""
Aggressive bootstrap augmentation using semantic transformations and text variations.
Target: 10k+ dataset from ~1200 baseline
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
DEFAULT_MANIFEST = REPO_ROOT / "outputs" / "personality_bootstrap_aggressive_manifest.json"

# Semantic transformation rules per trait+endpoint
TRANSFORMS = {
    ("impulsiveness", "pos"): [
        ("先衝", ["直衝", "直接行動", "馬上動"]),
        ("不多想", ["不想太多", "不考慮太多", "少想"]),
        ("衝到底", ["一直衝", "持續衝", "衝下去"]),
        ("直覺", ["本能", "感覺", "預感"]),
        ("立即", ["當下", "馬上", "立刻"]),
    ],
    ("impulsiveness", "neg"): [
        ("先想", ["先思考", "想清楚", "想好"]),
        ("三思", ["多思考", "審視", "謹慎"]),
        ("計畫", ["規劃", "籌畫", "預備"]),
        ("緩", ["慢", "暫停", "停頓"]),
        ("謹慎", ["小心", "仔細", "警惕"]),
    ],
    ("assertiveness", "pos"): [
        ("主導", ["領導", "掌控", "引導"]),
        ("決定", ["做決策", "拿主意", "選擇"]),
        ("帶頭", ["起頭", "率先", "領先"]),
        ("表達", ["發聲", "說出", "講述"]),
        ("影響", ["改變", "驅動", "推動"]),
    ],
    ("assertiveness", "neg"): [
        ("讓步", ["退讓", "退縮", "妥協"]),
        ("聽從", ["接納", "跟隨", "配合"]),
        ("安靜", ["沉默", "被動", "側言"]),
        ("尊重", ["聽取", "考慮", "重視"]),
        ("配合", ["適應", "調整", "遵循"]),
    ],
    ("optimism", "pos"): [
        ("樂觀", ["正面", "積極", "希望"]),
        ("會好", ["變好", "改善", "進步"]),
        ("可能", ["能夠", "有機會", "成功"]),
        ("相信", ["確信", "篤定", "肯定"]),
        ("開放", ["接納", "歡迎", "迎接"]),
    ],
    ("optimism", "neg"): [
        ("悲觀", ["負面", "消極", "失望"]),
        ("不行", ["會失敗", "困難", "麻煩"]),
        ("懷疑", ["不確定", "不信", "存疑"]),
        ("預期壞", ["預想糟", "設想差", "想像差"]),
        ("謹慎", ["警惕", "提防", "防範"]),
    ],
    ("risk_aversion", "pos"): [
        ("謹慎", ["小心", "保守", "穩妥"]),
        ("先看風險", ["先想風險", "預想問題", "提前防範"]),
        ("規避", ["迴避", "躲避", "遠離"]),
        ("確認", ["驗證", "檢查", "確保"]),
        ("保護", ["防守", "守護", "維護"]),
    ],
    ("risk_aversion", "neg"): [
        ("冒進", ["大膽", "激進", "勇敢"]),
        ("先做", ["直接做", "立即行", "馬上試"]),
        ("挑戰", ["挑戰", "嘗試", "實驗"]),
        ("膽量", ["勇氣", "決心", "魄力"]),
        ("不怕", ["無懼", "無畏", "不在乎"]),
    ],
    ("suspicion", "pos"): [
        ("多疑", ["警惕", "懷疑", "不信"]),
        ("先懷疑", ["先提防", "先不信", "先警惕"]),
        ("察言觀色", ["觀察", "注視", "監視"]),
        ("隱藏", ["保留", "防守", "不洩露"]),
        ("距離", ["冷淡", "疏遠", "隔離"]),
    ],
    ("suspicion", "neg"): [
        ("信任", ["相信", "依賴", "託付"]),
        ("先相信", ["先信任", "先接納", "先認可"]),
        ("開放", ["坦白", "透露", "分享"]),
        ("親近", ["靠近", "接近", "貼近"]),
        ("樂觀", ["正面", "積極", "希望"]),
    ],
    ("endurance", "pos"): [
        ("堅持", ["堅定", "固守", "維持"]),
        ("一直堅持", ["長期堅持", "不放棄", "持續進行"]),
        ("耐力", ["毅力", "決心", "韌性"]),
        ("持久", ["長期", "延續", "持續"]),
        ("不退", ["前進", "繼續", "推進"]),
    ],
    ("endurance", "neg"): [
        ("放棄", ["停止", "終止", "結束"]),
        ("易放棄", ["容易停", "常中止", "易中止"]),
        ("彈性", ["變通", "靈活", "調整"]),
        ("短期", ["短暫", "臨時", "一時"]),
        ("放鬆", ["鬆懈", "怠惰", "鬆開"]),
    ],
    ("randomness", "pos"): [
        ("隨機", ["隨機應變", "彈性", "不規律"]),
        ("隨機應變", ["臨機應變", "因應", "隨時變"]),
        ("不固定", ["不定型", "浮動", "不穩定"]),
        ("變化", ["轉變", "改變", "切換"]),
        ("即興", ["臨機", "當機立斷", "隨興"]),
    ],
    ("randomness", "neg"): [
        ("規律", ["有規則", "按步", "規劃"]),
        ("按規劃", ["依計畫", "循序", "按部就班"]),
        ("固定", ["不變", "穩定", "恆定"]),
        ("秩序", ["順序", "組織", "結構"]),
        ("重複", ["習慣", "模式", "常規"]),
    ],
    ("stability_seeking", "pos"): [
        ("穩定", ["平穩", "靜止", "恆定"]),
        ("穩定節奏", ["平穩進度", "恆定速度", "不變步調"]),
        ("安全", ["放心", "保險", "無憂"]),
        ("習慣", ["慣例", "常規", "老辦法"]),
        ("不變", ["不改", "堅守", "維持"]),
    ],
    ("stability_seeking", "neg"): [
        ("變動", ["變化", "改變", "轉變"]),
        ("換節奏", ["改步調", "調速度", "轉進度"]),
        ("追求新", ["好新", "尋新鮮", "求刺激"]),
        ("不定", ["無常", "易變", "莫測"]),
        ("冒險", ["嘗試", "實驗", "挑戰"]),
    ],
    ("curiosity", "pos"): [
        ("好奇", ["感興趣", "想探索", "求知"]),
        ("探索未知", ["發現新", "認識陌生", "深入研究"]),
        ("開放", ["接納新", "歡迎異", "包容"]),
        ("學習", ["研究", "探究", "鑽研"]),
        ("興趣", ["關注", "投入", "沉浸"]),
    ],
    ("curiosity", "neg"): [
        ("保守", ["傳統", "守舊", "經典"]),
        ("熟悉環境", ["已知範圍", "安全地帶", "熟知領域"]),
        ("依賴", ["靠信", "相信", "仰仗"]),
        ("不求變", ["安於現狀", "不尋求", "按既定"]),
        ("簡單", ["直接", "基本", "不深究"]),
    ],
}


@dataclass
class Row:
    timestamp: str
    request_id: str
    text: str
    length: int
    model: str
    temperature: str
    source: str
    session_id: str
    user_id: str | None
    vector: dict[str, float]

    def to_tsv(self) -> list[str]:
        values = [
            self.timestamp,
            self.request_id,
            self.text,
            str(self.length),
            self.model,
            self.temperature,
            self.source,
            self.session_id,
            str(self.user_id) if self.user_id else "",
        ]
        for trait in PERSONALITY_BASIS:
            values.append(str(self.vector.get(trait, 0.0)))
        return values


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _normalize_text(text: str) -> str:
    """Basic normalization."""
    return text.strip()


def _read_existing(tsv_path: Path) -> dict[str, dict[str, Any]]:
    """Read existing rows from TSV. Returns dict[text_hash] -> row_dict."""
    existing = {}
    if not tsv_path.exists():
        return existing
    
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            return existing
        for row in reader:
            text = row.get("text", "")
            h = _text_hash(text)
            existing[h] = row
    
    return existing


def _apply_transforms(text: str, trait: str, endpoint: str) -> list[str]:
    """Apply semantic transformations to text."""
    results = [text]
    
    key = (trait, endpoint)
    if key not in TRANSFORMS:
        return results
    
    for original, replacements in TRANSFORMS[key]:
        if original in text:
            for replacement in replacements:
                new_text = text.replace(original, replacement, 1)
                new_text = _normalize_text(new_text)
                if 1 <= len(new_text) <= 25 and new_text not in results:
                    results.append(new_text)
    
    return results


def _apply_structural_transforms(text: str) -> list[str]:
    """Apply structural transformations (prefix/suffix variations)."""
    results = []
    
    # Add prefixes
    prefixes = ["我", "我會", "我常", "我也", "我總是", "我傾向", "我喜歡", "我習慣"]
    for prefix in prefixes:
        if not text.startswith(prefix):
            new_text = f"{prefix}{text}"
            if 1 <= len(new_text) <= 25 and new_text not in results:
                results.append(new_text)
    
    # Add suffixes
    suffixes = ["", "!", "，我認為", "，據我", "，我覺得", "，一直都是"]
    for suffix in suffixes:
        if suffix and not text.endswith(suffix):
            new_text = f"{text}{suffix}"
            if 1 <= len(new_text) <= 25 and new_text not in results:
                results.append(new_text)
    
    # Remove prefixes
    for prefix in ["我", "我會", "我常", "我也", "我總是"]:
        if text.startswith(prefix):
            new_text = text[len(prefix):]
            if 1 <= len(new_text) <= 25 and new_text not in results:
                results.append(new_text)
    
    # Remove suffixes
    for suffix in ["!", "。", "啊", "呢", "吧"]:
        if text.endswith(suffix):
            new_text = text[:-len(suffix)]
            if 1 <= len(new_text) <= 25 and new_text not in results:
                results.append(new_text)
    
    return results


def _generate_augmented(existing_rows: list[dict[str, Any]], session_id: str, target: int = 8800) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate augmented rows using semantic transforms."""
    generated = []
    seen = {_text_hash(row["text"]): row["text"] for row in existing_rows}
    
    traits_with_endpoint = [
        ("impulsiveness", "pos"), ("impulsiveness", "neg"),
        ("assertiveness", "pos"), ("assertiveness", "neg"),
        ("optimism", "pos"), ("optimism", "neg"),
        ("risk_aversion", "pos"), ("risk_aversion", "neg"),
        ("suspicion", "pos"), ("suspicion", "neg"),
        ("endurance", "pos"), ("endurance", "neg"),
        ("randomness", "pos"), ("randomness", "neg"),
        ("stability_seeking", "pos"), ("stability_seeking", "neg"),
        ("curiosity", "pos"), ("curiosity", "neg"),
    ]
    
    # Group existing rows by trait+endpoint
    grouped = defaultdict(list)
    for row in existing_rows:
        for trait in PERSONALITY_BASIS:
            val = float(row.get(trait, 0.0))
            if trait in [t for t, _ in traits_with_endpoint]:
                if val > 0.3:
                    endpoint = "pos"
                elif val < -0.3:
                    endpoint = "neg"
                else:
                    endpoint = "neutral"
                grouped[(trait, endpoint)].append(row)
    
    # Generate from each group
    base_timestamp = _now_utc()
    counter = 0
    
    for trait, endpoint in traits_with_endpoint:
        if (trait, endpoint) not in grouped:
            continue
        
        rows = grouped[(trait, endpoint)]
        for idx, base_row in enumerate(rows):
            if counter >= target:
                break
            
            text = base_row["text"]
            
            # Apply semantic transforms
            variants = _apply_transforms(text, trait, endpoint)
            # Also apply structural transforms
            variants.extend(_apply_structural_transforms(text))
            
            for var_idx, variant in enumerate(variants):
                if counter >= target:
                    break
                
                h = _text_hash(variant)
                if h in seen:
                    continue
                
                seen[h] = variant
                
                # Create new row based on variant
                new_row = {
                    "timestamp": (base_timestamp.replace(microsecond=0) + __import__("datetime").timedelta(seconds=counter)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "request_id": uuid4().hex,
                    "text": variant,
                    "length": len(variant),
                    "model": "bootstrap-semantic-transform",
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
        
        if counter >= target:
            break
    
    return generated, {
        "session_id": session_id,
        "target": target,
        "generated": counter,
        "groups_processed": len(traits_with_endpoint),
        "unique_count": len(seen),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggressive bootstrap to 10k+")
    parser.add_argument("--target", type=int, default=8800, help="Target new rows to add")
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    
    session_id = f"bootstrap_aggressive_{_now_utc().strftime('%Y%m%d_%H%M%S')}"
    
    # Read existing data
    existing_rows = []
    if args.tsv.exists():
        with open(args.tsv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                existing_rows.append(row)
    
    print(f"[*] Read {len(existing_rows)} existing rows", file=sys.stderr)
    
    # Generate augmented rows
    generated, stats = _generate_augmented(existing_rows, session_id, args.target)
    print(f"[*] Generated {len(generated)} new rows (target was {args.target})", file=sys.stderr)
    
    # Write outputs
    if not args.dry_run:
        # Append to TSV
        with open(args.tsv, "a", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            for row in generated:
                values = []
                for key in ["timestamp", "request_id", "text", "length", "model", "temperature", "source", "session_id", "user_id"]:
                    values.append(row.get(key, ""))
                for trait in PERSONALITY_BASIS:
                    values.append(row.get(trait, "0.0"))
                writer.writerow(values)
        
        # Append to JSONL
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
        
        # Write manifest
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"[+] Wrote {len(generated)} rows to TSV/JSONL", file=sys.stderr)
        print(f"[+] Manifest: {args.manifest}", file=sys.stderr)
    
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
