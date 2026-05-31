#!/usr/bin/env python3
"""Minimal RL Session API connectivity check.

This script verifies that the PlayableLoopScene backend is reachable and that
POST /rl_sessions/initialize returns a 200 response with a session_id.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got: {value!r}") from exc


def main() -> int:
    base_url = os.environ.get("RL_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    endpoint = f"{base_url}/rl_sessions/initialize"

    payload = {
        "n_players": _env_int("N_PLAYERS", 4),
        "n_rounds": _env_int("N_ROUNDS", 200),
        "burn_in": _env_int("BURN_IN", 50),
        "seed": _env_int("SEED", 42),
    }

    request_body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"POST {endpoint}")
    print(f"Payload: {json.dumps(payload, ensure_ascii=False)}")

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response_text = response.read().decode("utf-8", errors="replace")
            print(f"HTTP {response.status}")
            print(response_text)

            data = json.loads(response_text)
            session_id = data.get("session_id")
            if not session_id:
                print("FAIL: response did not include session_id", file=sys.stderr)
                return 1

            print(f"OK: session_id={session_id}")
            return 0
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        print(f"HTTP {error.code}", file=sys.stderr)
        if error_body:
            print(error_body, file=sys.stderr)
        return 1
    except urllib.error.URLError as error:
        print(f"FAIL: cannot reach RL Session API at {endpoint}", file=sys.stderr)
        print(f"Reason: {error.reason}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as error:
        print(f"FAIL: response was not valid JSON: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())