from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SEED_CSV = REPO_ROOT / "data" / "personality_seed_texts.csv"
DEFAULT_OUTPUTS_TSV = REPO_ROOT / "outputs" / "personality_text_pairs.tsv"
DEFAULT_OUTPUTS_JSONL = REPO_ROOT / "outputs" / "personality_text_pairs.jsonl"
DEFAULT_REPORT = REPO_ROOT / "outputs" / "personality_a_stage_collect_report.json"
DEFAULT_ENDPOINT = os.getenv("PERSONALITY_INFER_ENDPOINT", "http://127.0.0.1:8000/personality/infer")
DEFAULT_TIMEOUT_SEC = 120.0
DEFAULT_SLEEP_SEC = 1.0
DEFAULT_TEMPERATURE = 0.35


@dataclass(frozen=True)
class SeedRow:
    text: str
    source: str = "seed_batch"
    trait_tags: str = ""
    notes: str = ""


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_session_id() -> str:
    return datetime.now(timezone.utc).strftime("seed_batch_%Y%m%d_%H%M%S")


def _normalize_text(text: str) -> str:
    return text.replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()


def _text_hash(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


def _load_seed_rows(seed_csv: Path) -> list[SeedRow]:
    rows: list[SeedRow] = []
    with seed_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            text = _normalize_text(raw_row.get("text", ""))
            if not text:
                continue
            rows.append(
                SeedRow(
                    text=text,
                    source=_normalize_text(raw_row.get("source", "seed_batch")) or "seed_batch",
                    trait_tags=_normalize_text(raw_row.get("trait_tags", "")),
                    notes=_normalize_text(raw_row.get("notes", "")),
                )
            )
    return rows


def _read_existing_text_hashes(tsv_path: Path) -> set[str]:
    if not tsv_path.exists():
        return set()

    hashes: set[str] = set()
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            text = _normalize_text(row.get("text", ""))
            if text:
                hashes.add(_text_hash(text))
    return hashes


def _prune_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _call_api(endpoint: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    request = urlrequest.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8")
    except urlerror.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"endpoint unreachable: {exc.reason}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON response: {body[:200]}") from exc


def _call_direct(
    text: str,
    *,
    temperature: float,
    source: str | None,
    session_id: str | None,
    user_id: str | None,
) -> dict[str, Any]:
    from api.personality_text_inference import infer_personality_vector, load_inference_config, log_personality_pair

    config = load_inference_config()
    vector, meta = infer_personality_vector(text, config=config, temperature_override=temperature)
    logged, log_error = log_personality_pair(
        text,
        vector,
        meta,
        config=config,
        source=source,
        session_id=session_id,
        user_id=user_id,
    )
    return {
        "request_id": str(meta.get("request_id", "")),
        "text": text,
        "vector": vector,
        "model": str(meta.get("model", "")),
        "temperature": float(meta.get("temperature", temperature)),
        "logged": bool(logged),
        "log_error": log_error,
    }


def _load_seeds(seed_csv: Path) -> list[SeedRow]:
    if seed_csv.exists():
        return _load_seed_rows(seed_csv)

    return [
        SeedRow("我做事憑直覺，先衝再說", "seed_batch", "impulsiveness;risk_aversion", "high impulsiveness"),
        SeedRow("我謹慎評估後再行動", "seed_batch", "impulsiveness;risk_aversion", "high risk aversion"),
        SeedRow("喜歡主導對話和決策", "seed_batch", "assertiveness", "short assertive"),
        SeedRow("我偏好聆聽，不喜歡出風頭", "seed_batch", "assertiveness", "low assertiveness"),
        SeedRow("未來一定更好，我非常樂觀", "seed_batch", "optimism", "high optimism"),
        SeedRow("事情很少如預期發展", "seed_batch", "optimism", "low optimism"),
        SeedRow("我能長期堅持一件困難的事", "seed_batch", "endurance", "high endurance"),
        SeedRow("做事缺乏耐心，遇到困難就放棄", "seed_batch", "endurance;impulsiveness", "low endurance"),
        SeedRow("我不信任陌生人的好意", "seed_batch", "suspicion", "high suspicion"),
        SeedRow("人性本善，我願意相信大多數人", "seed_batch", "suspicion", "low suspicion"),
        SeedRow("我喜歡隨機應變，不按計劃行事", "seed_batch", "randomness;stability_seeking", "high randomness"),
        SeedRow("穩定規律的生活讓我感到安心", "seed_batch", "stability_seeking;randomness", "high stability seeking"),
        SeedRow("探索未知事物是我最大樂趣", "seed_batch", "curiosity", "high curiosity"),
        SeedRow("熟悉的環境才讓我有安全感", "seed_batch", "curiosity;stability_seeking", "low curiosity"),
        SeedRow("我平時不特別冒進也不保守", "seed_batch", "neutral", "neutral baseline"),
        SeedRow("冒險", "seed_batch", "impulsiveness;curiosity", "short form"),
        SeedRow("謹慎", "seed_batch", "risk_aversion;impulsiveness", "short form"),
        SeedRow("我充滿好奇心，什麼都想嘗試", "seed_batch", "curiosity;randomness", "exploratory"),
        SeedRow("凡事先想清楚再動手", "seed_batch", "risk_aversion;impulsiveness", "careful"),
        SeedRow("我喜歡掌控局面，不喜歡被動", "seed_batch", "assertiveness;endurance", "control"),
    ]


def _build_payloads(
    seed_rows: list[SeedRow],
    *,
    existing_hashes: set[str],
    temperature: float,
    session_id: str,
    user_id: str | None,
    source_override: str | None,
) -> tuple[list[dict[str, Any]], int]:
    payloads: list[dict[str, Any]] = []
    skipped = 0
    seen_hashes = set(existing_hashes)
    for row in seed_rows:
        normalized_text = _normalize_text(row.text)
        if not normalized_text:
            continue
        text_hash = _text_hash(normalized_text)
        if text_hash in seen_hashes:
            skipped += 1
            continue
        seen_hashes.add(text_hash)
        payloads.append(
            {
                "text": normalized_text,
                "source": source_override or row.source or "seed_batch",
                "session_id": session_id,
                "user_id": user_id,
                "temperature": temperature,
                "trait_tags": row.trait_tags,
                "notes": row.notes,
                "text_hash": text_hash,
            }
        )
    return payloads, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect A-stage weak labels via /personality/infer")
    parser.add_argument("--seed-csv", type=Path, default=DEFAULT_SEED_CSV, help="Seed CSV path")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="POST endpoint for /personality/infer")
    parser.add_argument("--outputs-tsv", type=Path, default=DEFAULT_OUTPUTS_TSV, help="Collected TSV path")
    parser.add_argument("--outputs-jsonl", type=Path, default=DEFAULT_OUTPUTS_JSONL, help="Collected JSONL path")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Collection report path")
    parser.add_argument("--limit", type=int, default=0, help="Optional seed limit")
    parser.add_argument("--sleep-sec", type=float, default=DEFAULT_SLEEP_SEC, help="Delay between requests")
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC, help="HTTP timeout in seconds")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Override temperature")
    parser.add_argument("--source", default=None, help="Override source field")
    parser.add_argument("--session-id", default=None, help="Override session_id field")
    parser.add_argument("--user-id", default=None, help="Override user_id field")
    parser.add_argument("--transport", choices=("api", "direct"), default="api", help="How to perform inference")
    parser.add_argument("--dry-run", action="store_true", help="Print planned requests without calling inference")
    args = parser.parse_args(argv)

    session_id = args.session_id or _default_session_id()

    seed_rows = _load_seeds(args.seed_csv)
    if args.limit and args.limit > 0:
        seed_rows = seed_rows[: args.limit]

    args.outputs_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.outputs_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    existing_hashes = _read_existing_text_hashes(args.outputs_tsv)
    payloads, skipped = _build_payloads(
        seed_rows,
        existing_hashes=existing_hashes,
        temperature=args.temperature,
        session_id=session_id,
        user_id=args.user_id,
        source_override=args.source,
    )

    summary: dict[str, Any] = {
        "started_at": _now_utc(),
        "ended_at": None,
        "endpoint": args.endpoint,
        "seed_csv": str(args.seed_csv),
        "transport": args.transport,
        "dry_run": bool(args.dry_run),
        "session_id": session_id,
        "attempted": len(payloads),
        "collected": 0,
        "skipped_existing": skipped,
        "failed": 0,
        "results": [],
    }

    if args.dry_run:
        for item in payloads:
            summary["results"].append(
                {
                    "text_hash": item["text_hash"],
                    "text": item["text"],
                    "status": "dry_run",
                    "source": item["source"],
                }
            )
        summary["ended_at"] = _now_utc()
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    for index, item in enumerate(payloads, start=1):
        text = item["text"]
        print(f"[{index}/{len(payloads)}] collecting: {text}")
        try:
            if args.transport == "api":
                payload = _prune_none(
                    {
                        "text": text,
                        "source": item["source"],
                        "session_id": item["session_id"],
                        "user_id": item["user_id"],
                        "temperature": item["temperature"],
                    }
                )
                response = _call_api(args.endpoint, payload, args.timeout_sec)
            else:
                response = _call_direct(
                    text,
                    temperature=item["temperature"],
                    source=item["source"],
                    session_id=item["session_id"],
                    user_id=item["user_id"],
                )
        except Exception as exc:
            summary["failed"] += 1
            summary["results"].append(
                {
                    "text_hash": item["text_hash"],
                    "text": text,
                    "status": "failed",
                    "error": str(exc),
                    "source": item["source"],
                }
            )
            print(f"  failed: {exc}", file=sys.stderr)
            continue

        summary["collected"] += 1
        summary["results"].append(
            {
                "text_hash": item["text_hash"],
                "text": text,
                "status": "ok",
                "request_id": response.get("request_id"),
                "model": response.get("model"),
                "logged": response.get("logged"),
                "source": item["source"],
            }
        )
        if args.sleep_sec > 0.0 and index < len(payloads):
            time.sleep(args.sleep_sec)

    summary["ended_at"] = _now_utc()
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
