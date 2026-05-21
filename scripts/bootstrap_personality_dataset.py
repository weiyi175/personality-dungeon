from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.schemas import PERSONALITY_BASIS


DEFAULT_TSV = REPO_ROOT / "outputs" / "personality_text_pairs.tsv"
DEFAULT_JSONL = REPO_ROOT / "outputs" / "personality_text_pairs.jsonl"
DEFAULT_MANIFEST = REPO_ROOT / "outputs" / "personality_bootstrap_manifest.json"

SHORT_SUBJECTS = ["我", "我會", "我常", "我也", "我總"]
SHORT_SUFFIXES = ["", "!", "啊", "呢"]
MEDIUM_CONTEXTS = ["面對壓力", "在團隊中", "遇到變化", "需要決策", "處理衝突", "踏入陌生", "面臨挑戰", "觀察細節", "做長期事", "看見機會"]
LONG_CONTEXTS = ["當壓力逼近時", "在陌生情境裡", "面對關鍵抉擇時", "做長期計畫時", "當意見分歧時", "需要快速反應時", "碰到反覆失敗時", "探索新地圖時", "走進不熟場域時", "遇到複雜問題時"]


@dataclass(frozen=True)
class GroupSpec:
    key: str
    source: str
    short_core: str
    medium_core: str
    long_core: str
    primary_trait: str
    primary_sign: float
    secondary: dict[str, float]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_text(text: str) -> str:
    return text.replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()


def _clamp(value: float, *, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _load_existing_texts(tsv_path: Path) -> set[str]:
    if not tsv_path.exists():
        return set()

    texts: set[str] = set()
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            text = _normalize_text(row.get("text", ""))
            if text:
                texts.add(text)
    return texts


def _length_bucket(length: int) -> str | None:
    if 1 <= length <= 5:
        return "short"
    if 6 <= length <= 12:
        return "medium"
    if 13 <= length <= 20:
        return "long"
    return None


def _make_vector(group: GroupSpec, *, offset: int, bucket: str) -> dict[str, float]:
    vector = {trait: 0.0 for trait in PERSONALITY_BASIS}
    jitter_cycle = [-0.06, 0.0, 0.06, 0.03, -0.03]
    jitter = jitter_cycle[offset % len(jitter_cycle)]

    if group.key == "neutral":
        for trait in PERSONALITY_BASIS:
            vector[trait] = _clamp(jitter if trait in {"optimism", "curiosity"} else 0.0)
        return vector

    primary = _clamp(group.primary_sign + jitter)
    vector[group.primary_trait] = primary
    for trait, value in group.secondary.items():
        vector[trait] = _clamp(value + (jitter / 2.0))

    if bucket == "short":
        vector[group.primary_trait] = _clamp(group.primary_sign + jitter * 0.5)
    elif bucket == "medium":
        vector[group.primary_trait] = _clamp(group.primary_sign + jitter)
    else:
        vector[group.primary_trait] = _clamp(group.primary_sign - jitter * 0.25)

    return vector


def _group_specs() -> list[GroupSpec]:
    return [
        GroupSpec("impulsiveness_pos", "augmented", "先衝", "先衝再說", "先衝到底再說", "impulsiveness", 0.90, {"risk_aversion": -0.65, "randomness": 0.30, "stability_seeking": -0.20}),
        GroupSpec("impulsiveness_neg", "augmented", "先想", "先想再說", "先想清楚再說", "impulsiveness", -0.90, {"risk_aversion": 0.65, "randomness": -0.20, "stability_seeking": 0.20}),
        GroupSpec("assertiveness_pos", "augmented", "主導", "我主導", "我會主導局面", "assertiveness", 0.90, {"endurance": 0.20, "suspicion": -0.10, "curiosity": 0.10}),
        GroupSpec("assertiveness_neg", "augmented", "讓步", "我讓步", "我常讓別人決定", "assertiveness", -0.90, {"endurance": -0.10, "suspicion": 0.10, "stability_seeking": 0.10}),
        GroupSpec("optimism_pos", "augmented", "樂觀", "我很樂觀", "我總覺得會更好", "optimism", 0.90, {"endurance": 0.20, "suspicion": -0.10, "risk_aversion": -0.20}),
        GroupSpec("optimism_neg", "augmented", "悲觀", "我偏悲觀", "我常先想到壞結果", "optimism", -0.90, {"suspicion": 0.20, "risk_aversion": 0.20, "stability_seeking": 0.10}),
        GroupSpec("risk_aversion_pos", "augmented", "謹慎", "我很謹慎", "我會先看風險", "risk_aversion", 0.90, {"impulsiveness": -0.60, "randomness": -0.20, "stability_seeking": 0.20}),
        GroupSpec("risk_aversion_neg", "augmented", "冒進", "我敢冒進", "我常先做再說", "risk_aversion", -0.90, {"impulsiveness": 0.60, "randomness": 0.20, "curiosity": 0.10}),
        GroupSpec("suspicion_pos", "augmented", "多疑", "我很多疑", "我總先懷疑別人", "suspicion", 0.90, {"risk_aversion": 0.20, "assertiveness": -0.10, "curiosity": -0.10}),
        GroupSpec("suspicion_neg", "augmented", "信任", "我信任", "我願先相信別人", "suspicion", -0.90, {"optimism": 0.20, "curiosity": 0.10, "risk_aversion": -0.10}),
        GroupSpec("endurance_pos", "augmented", "堅持", "我很堅持", "我會一直堅持下去", "endurance", 0.90, {"stability_seeking": 0.20, "impulsiveness": -0.10, "randomness": -0.10}),
        GroupSpec("endurance_neg", "augmented", "放棄", "我易放棄", "我常半途而廢", "endurance", -0.90, {"impulsiveness": 0.20, "randomness": 0.20, "stability_seeking": -0.10}),
        GroupSpec("randomness_pos", "augmented", "隨機", "我愛隨機", "我常隨機應變", "randomness", 0.90, {"stability_seeking": -0.60, "curiosity": 0.20, "endurance": -0.10}),
        GroupSpec("randomness_neg", "augmented", "規律", "我愛規律", "我常按規劃走", "randomness", -0.90, {"stability_seeking": 0.60, "curiosity": -0.10, "endurance": 0.10}),
        GroupSpec("stability_pos", "augmented", "穩定", "我很穩定", "我偏好穩定節奏", "stability_seeking", 0.90, {"randomness": -0.60, "impulsiveness": -0.10, "endurance": 0.10}),
        GroupSpec("stability_neg", "augmented", "變動", "我愛變動", "我常想換節奏", "stability_seeking", -0.90, {"randomness": 0.60, "curiosity": 0.10, "endurance": -0.10}),
        GroupSpec("curiosity_pos", "augmented", "好奇", "我很好奇", "我常探索未知", "curiosity", 0.90, {"stability_seeking": -0.30, "randomness": 0.20, "optimism": 0.10}),
        GroupSpec("curiosity_neg", "augmented", "保守", "我偏保守", "我傾向熟悉環境", "curiosity", -0.90, {"stability_seeking": 0.30, "randomness": -0.10, "suspicion": 0.10}),
        GroupSpec("neutral", "augmented", "中性", "我不偏不倚", "我通常不偏不倚", "optimism", 0.00, {"stability_seeking": 0.00, "curiosity": 0.00}),
    ]


def _generate_texts_for_group(group: GroupSpec, bucket: str, limit: int = 20) -> list[str]:
    texts: list[str] = []
    if bucket == "short":
        for subject in SHORT_SUBJECTS:
            for suffix in SHORT_SUFFIXES:
                if group.key == "neutral":
                    candidate = f"{subject}{group.short_core}{suffix}"
                else:
                    candidate = f"{subject}{group.short_core}{suffix}"
                candidate = _normalize_text(candidate)
                if 1 <= len(candidate) <= 5 and candidate not in texts:
                    texts.append(candidate)
                if len(texts) >= limit:
                    return texts
    elif bucket == "medium":
        for context in MEDIUM_CONTEXTS:
            for subject in SHORT_SUBJECTS:
                candidate = _normalize_text(f"{context}{subject}{group.medium_core}")
                if 6 <= len(candidate) <= 12 and candidate not in texts:
                    texts.append(candidate)
                if len(texts) >= limit:
                    return texts
    else:
        for context in LONG_CONTEXTS:
            for subject in SHORT_SUBJECTS:
                for suffix in ["", "吧", "啊", "呢"]:
                    candidate = _normalize_text(f"{context}{subject}{group.long_core}{suffix}")
                    if 13 <= len(candidate) <= 20 and candidate not in texts:
                        texts.append(candidate)
                    if len(texts) >= limit:
                        return texts
    return texts


def _existing_hash_set(tsv_path: Path) -> set[str]:
    if not tsv_path.exists():
        return set()
    hashes: set[str] = set()
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            text = _normalize_text(row.get("text", ""))
            if text:
                hashes.add(text)
    return hashes


def _append_tsv(tsv_path: Path, rows: list[dict[str, Any]]) -> None:
    needs_header = not tsv_path.exists() or tsv_path.stat().st_size == 0
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "timestamp",
            "request_id",
            "text",
            "length",
            "model",
            "temperature",
            "source",
            "session_id",
            "user_id",
            *PERSONALITY_BASIS,
        ], delimiter="\t")
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_jsonl(jsonl_path: Path, rows: list[dict[str, Any]]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "timestamp": row["timestamp"],
                "request_id": row["request_id"],
                "text": row["text"],
                "length": row["length"],
                "model": row["model"],
                "temperature": float(row["temperature"]),
                "source": row["source"],
                "session_id": row["session_id"],
                "user_id": row["user_id"],
                "vector": {trait: float(row[trait]) for trait in PERSONALITY_BASIS},
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap 1000+ augmented personality rows")
    parser.add_argument("--target-additions", type=int, default=1140, help="How many new rows to add")
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV, help="Target TSV path")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL, help="Target JSONL path")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Manifest JSON path")
    parser.add_argument("--session-id", default=None, help="Session id for generated rows")
    parser.add_argument("--dry-run", action="store_true", help="Do not write outputs")
    args = parser.parse_args(argv)

    session_id = args.session_id or f"bootstrap_a_stage_{_now_utc().strftime('%Y%m%d_%H%M%S')}"
    base_timestamp = _now_utc().replace(microsecond=0)
    existing_texts = _existing_hash_set(args.tsv)
    generated_rows: list[dict[str, Any]] = []
    seen_texts = set(existing_texts)

    groups = _group_specs()
    bucket_targets = {"short": 0, "medium": 0, "long": 0}
    target_per_bucket = max(1, args.target_additions // 3)
    bucket_order = ["short", "medium", "long"]
    total_target = args.target_additions

    for bucket in bucket_order:
        bucket_targets[bucket] = target_per_bucket

    # Use any remainder on the long bucket so the length distribution stays balanced.
    bucket_targets["long"] += total_target - sum(bucket_targets.values())

    generated_count = 0
    per_group_stats: dict[str, dict[str, int]] = {}

    for group in groups:
        per_group_stats[group.key] = {bucket: 0 for bucket in bucket_order}
        for bucket in bucket_order:
            texts = _generate_texts_for_group(group, bucket, limit=20)
            for text_index, text in enumerate(texts):
                if generated_count >= total_target:
                    break
                if text in seen_texts:
                    continue
                seen_texts.add(text)

                vector = _make_vector(group, offset=text_index + generated_count, bucket=bucket)
                timestamp = (base_timestamp + timedelta(seconds=generated_count)).strftime("%Y-%m-%dT%H:%M:%SZ")
                row: dict[str, Any] = {
                    "timestamp": timestamp,
                    "request_id": uuid4().hex,
                    "text": text,
                    "length": len(text),
                    "model": "bootstrap-manual-label",
                    "temperature": 0.35,
                    "source": group.source,
                    "session_id": session_id,
                    "user_id": None,
                }
                row.update(vector)
                generated_rows.append(row)
                generated_count += 1
                per_group_stats[group.key][bucket] += 1

            if generated_count >= total_target:
                break
        if generated_count >= total_target:
            break

    manifest = {
        "session_id": session_id,
        "started_at": base_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_additions": total_target,
        "generated_additions": generated_count,
        "source": "augmented",
        "bucket_targets": bucket_targets,
        "group_stats": per_group_stats,
        "output_tsv": str(args.tsv),
        "output_jsonl": str(args.jsonl),
    }

    if not args.dry_run:
        _append_tsv(args.tsv, generated_rows)
        _append_jsonl(args.jsonl, generated_rows)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if generated_count >= total_target else 1


if __name__ == "__main__":
    raise SystemExit(main())