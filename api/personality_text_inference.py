"""LLM-backed text -> 9-trait personality inference (short prompts).

OpenAI-compatible interface: POST /v1/chat/completions.
Logging: append successful pairs to outputs/ as TSV + JSONL.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from api.schemas import PERSONALITY_BASIS


REPO_ROOT = Path(__file__).resolve().parents[1]

_DEFAULT_MAX_CHARS = 20
_DEFAULT_TEMPERATURE = 0.35
_DEFAULT_TIMEOUT_SEC = 30.0
_DEFAULT_MAX_TOKENS = 3000

_LOG_TSV = "personality_text_pairs.tsv"
_LOG_JSONL = "personality_text_pairs.jsonl"

_LOG_LOCK = threading.Lock()


@dataclass(frozen=True)
class PersonalityInferenceConfig:
    base_url: str
    model: str
    temperature: float
    max_chars: int
    timeout_sec: float
    max_tokens: int
    log_strict: bool
    tsv_path: Path
    jsonl_path: Path
    ollama_native: bool = False


def load_inference_config() -> PersonalityInferenceConfig:
    base_url = os.getenv("PERSONALITY_LLM_BASE_URL", "").strip()
    model = os.getenv("PERSONALITY_LLM_MODEL", "").strip()
    if not base_url:
        raise ValueError("PERSONALITY_LLM_BASE_URL is required")
    if not model:
        raise ValueError("PERSONALITY_LLM_MODEL is required")

    temperature = _read_float_env("PERSONALITY_LLM_TEMPERATURE", _DEFAULT_TEMPERATURE)
    max_chars = _read_int_env("PERSONALITY_TEXT_MAX_LEN", _DEFAULT_MAX_CHARS)
    timeout_sec = _read_float_env("PERSONALITY_LLM_TIMEOUT_SEC", _DEFAULT_TIMEOUT_SEC)
    max_tokens = _read_int_env("PERSONALITY_LLM_MAX_TOKENS", _DEFAULT_MAX_TOKENS)
    log_strict = _read_bool_env("PERSONALITY_LOG_STRICT", True)
    ollama_native = _read_bool_env("PERSONALITY_LLM_OLLAMA_NATIVE", False)

    outputs_dir = REPO_ROOT / "outputs"
    return PersonalityInferenceConfig(
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_chars=max_chars,
        timeout_sec=timeout_sec,
        max_tokens=max_tokens,
        log_strict=log_strict,
        tsv_path=outputs_dir / _LOG_TSV,
        jsonl_path=outputs_dir / _LOG_JSONL,
        ollama_native=ollama_native,
    )


def infer_personality_vector(
    text: str,
    *,
    config: PersonalityInferenceConfig,
    temperature_override: float | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("text must not be empty")
    if len(cleaned) > config.max_chars:
        raise ValueError(f"text exceeds max length {config.max_chars}")

    temperature = config.temperature if temperature_override is None else float(temperature_override)
    request_id = uuid.uuid4().hex

    messages = _build_messages(cleaned)
    response_text = _call_chat_completion(
        config.base_url,
        model=config.model,
        messages=messages,
        temperature=temperature,
        max_tokens=config.max_tokens,
        timeout_sec=config.timeout_sec,
        use_ollama_native=config.ollama_native,
    )
    vector = _parse_personality_vector(response_text)

    meta = {
        "request_id": request_id,
        "model": config.model,
        "temperature": temperature,
        "length": len(cleaned),
    }
    return vector, meta


def log_personality_pair(
    text: str,
    vector: dict[str, float],
    meta: dict[str, Any],
    *,
    config: PersonalityInferenceConfig,
    source: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> tuple[bool, str | None]:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record = {
        "timestamp": timestamp,
        "request_id": meta.get("request_id"),
        "text": text,
        "length": meta.get("length"),
        "model": meta.get("model"),
        "temperature": meta.get("temperature"),
        "source": source,
        "session_id": session_id,
        "user_id": user_id,
        "vector": vector,
    }

    try:
        _append_logs(config, record)
        return True, None
    except Exception as exc:  # pragma: no cover - log failure is surfaced upstream
        if config.log_strict:
            raise
        return False, str(exc)


def _append_logs(config: PersonalityInferenceConfig, record: dict[str, Any]) -> None:
    config.tsv_path.parent.mkdir(parents=True, exist_ok=True)
    config.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    header = _tsv_header()
    row = _tsv_row(record)
    json_line = json.dumps(record, ensure_ascii=True)

    with _LOG_LOCK:
        if not config.tsv_path.exists():
            config.tsv_path.write_text("\t".join(header) + "\n", encoding="utf-8")
        with config.tsv_path.open("a", encoding="utf-8") as f:
            f.write("\t".join(row) + "\n")
        with config.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json_line + "\n")


def _tsv_header() -> list[str]:
    return [
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
    ]


def _tsv_row(record: dict[str, Any]) -> list[str]:
    text = _safe_text(record.get("text"))
    vector = record.get("vector", {})
    ordered = [_format_float(vector.get(key, 0.0)) for key in PERSONALITY_BASIS]
    return [
        str(record.get("timestamp", "")),
        str(record.get("request_id", "")),
        text,
        str(record.get("length", "")),
        str(record.get("model", "")),
        _format_float(record.get("temperature", 0.0)),
        str(record.get("source") or ""),
        str(record.get("session_id") or ""),
        str(record.get("user_id") or ""),
        *ordered,
    ]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def _format_float(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "0.0"
    if not isfinite(num):
        return "0.0"
    return f"{num:.6f}"


def _build_messages(text: str) -> list[dict[str, str]]:
    traits = ", ".join(PERSONALITY_BASIS)
    system = (
        "You rate a short text into a 9-trait personality vector. "
        "Return ONLY a JSON object with exactly these keys: "
        f"{traits}. "
        "Each value must be a number in [-1, 1]."
    )
    # /no_think suppresses Qwen3's chain-of-thought when using the OpenAI-compat
    # endpoint. When ollama_native=True, `think: false` is sent in the API payload
    # instead, so /no_think is a no-op but harmless to keep.
    user = f"Text: {text} /no_think"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _call_chat_completion(
    base_url: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_sec: float,
    use_ollama_native: bool = False,
) -> str:
    if use_ollama_native:
        return _call_ollama_api_chat(
            base_url,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )
    endpoint = _chat_endpoint(base_url)
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("PERSONALITY_LLM_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def _post(payload: dict[str, Any]) -> str:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return response.read().decode("utf-8")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
        "think": False,
    }
    try:
        body = _post(payload)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        if _is_response_format_error(exc.code, detail):
            payload.pop("response_format", None)
            try:
                body = _post(payload)
            except urllib.error.HTTPError as exc2:
                detail2 = exc2.read().decode("utf-8", errors="ignore")
                raise ValueError(f"LLM request failed: {exc2.code} {detail2}") from exc2
        else:
            raise ValueError(f"LLM request failed: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"LLM request failed: {exc}") from exc

    parsed = json.loads(body)
    choices = parsed.get("choices") or []
    if not choices:
        raise ValueError("LLM response missing choices")
    message = choices[0].get("message") or {}
    content = _extract_message_content(message)
    if content is None:
        raise ValueError("LLM response missing content")
    return content


def _call_ollama_api_chat(
    base_url: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_sec: float,
) -> str:
    """Call Ollama's native /api/chat endpoint with think:false to skip CoT."""
    ollama_base = base_url.rstrip("/")
    if ollama_base.endswith("/v1/chat/completions"):
        ollama_base = ollama_base[: -len("/v1/chat/completions")]
    elif ollama_base.endswith("/v1"):
        ollama_base = ollama_base[:-3]
    endpoint = f"{ollama_base}/api/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise ValueError(f"Ollama request failed: {exc}") from exc
    parsed = json.loads(body)
    message = parsed.get("message") or {}
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama response missing content")
    return content


def _chat_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _parse_personality_vector(text: str) -> dict[str, float]:
    payload = _extract_json(text)
    if not isinstance(payload, dict):
        raise ValueError("LLM response is not a JSON object")

    missing = [k for k in PERSONALITY_BASIS if k not in payload]
    if missing:
        raise ValueError(f"LLM response missing keys: {', '.join(missing)}")

    result: dict[str, float] = {}
    for key in PERSONALITY_BASIS:
        value = _require_float(payload.get(key), key)
        result[key] = _clamp(value, -1.0, 1.0)

    return result


def _extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM response is not valid JSON")
        snippet = text[start : end + 1]
        return json.loads(snippet)


def _require_float(value: Any, field: str) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"LLM response field {field} is not a number")
    if not isfinite(num):
        raise ValueError(f"LLM response field {field} is not finite")
    return num


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _is_response_format_error(status: int, detail: str) -> bool:
    if status not in {400, 422}:
        return False
    lowered = detail.lower()
    return "response_format" in lowered or "json_object" in lowered


def _extract_message_content(message: dict[str, Any]) -> str | None:
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    return content if isinstance(content, str) else None
