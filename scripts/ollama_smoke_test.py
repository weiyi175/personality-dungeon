#!/usr/bin/env python3
"""Start Ollama if needed and verify server health via /api/tags.

Usage:
    ./venv/bin/python scripts/ollama_smoke_test.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_HOST = "http://localhost:11434"


def _normalize_host(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    if text.startswith("http://") or text.startswith("https://"):
        return text.rstrip("/")
    return f"http://{text.rstrip('/')}"


def _wsl_gateway_host() -> str:
    try:
        with open("/proc/sys/kernel/osrelease", "r", encoding="utf-8") as f:
            release = f.read().lower()
        if "microsoft" not in release:
            return ""
    except OSError:
        return ""

    try:
        output = subprocess.check_output(
            ["bash", "-lc", "ip route show default | awk '{print $3}'"],
            text=True,
        ).strip()
    except Exception:
        return ""

    if not output:
        return ""
    return f"http://{output}:11434"


def _candidate_hosts(explicit_host: str) -> list[str]:
    hosts: list[str] = []

    for raw in [explicit_host, os.getenv("OLLAMA_HOST", ""), DEFAULT_HOST, _wsl_gateway_host()]:
        host = _normalize_host(raw)
        if host and host not in hosts:
            hosts.append(host)

    return hosts


def _fetch_tags(host: str, timeout_sec: float) -> dict[str, Any]:
    url = f"{host}/api/tags"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def _check_host(host: str, timeout_sec: float) -> tuple[bool, str]:
    try:
        payload = _fetch_tags(host, timeout_sec)
    except urllib.error.URLError as exc:
        return False, f"{host} unreachable: {exc.reason}"
    except Exception as exc:
        return False, f"{host} check failed: {exc}"

    models = payload.get("models", [])
    if not isinstance(models, list):
        return False, f"{host} responded but payload format is unexpected"
    return True, f"{host} healthy (models={len(models)})"


def _find_healthy_host(hosts: list[str], timeout_sec: float) -> tuple[str, str]:
    last_msg = "No hosts checked"
    for host in hosts:
        ok, msg = _check_host(host, timeout_sec)
        print(f"[probe] {msg}")
        if ok:
            return host, msg
        last_msg = msg
    return "", last_msg


def _start_ollama_serve() -> tuple[bool, str]:
    exe = shutil.which("ollama")
    if not exe:
        return False, "cannot find 'ollama' in PATH, skip auto-start"

    try:
        kwargs: dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True

        subprocess.Popen([exe, "serve"], **kwargs)
        return True, f"started: {exe} serve"
    except Exception as exc:
        return False, f"failed to start ollama serve: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ollama start + health smoke test")
    parser.add_argument(
        "--host",
        default="",
        help="Preferred base URL (fallback to OLLAMA_HOST, localhost, WSL gateway)",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=5.0,
        help="Timeout for each health request",
    )
    parser.add_argument(
        "--wait-sec",
        type=float,
        default=20.0,
        help="Max wait after auto-start before giving up",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=1.0,
        help="Polling interval while waiting for service",
    )
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="Only probe health; do not auto-start ollama serve",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hosts = _candidate_hosts(args.host)
    if not hosts:
        print("[error] no candidate host available")
        return 1

    print(f"[info] host candidates: {', '.join(hosts)}")

    healthy_host, _ = _find_healthy_host(hosts, args.timeout_sec)
    if healthy_host:
        payload = _fetch_tags(healthy_host, args.timeout_sec)
        models = [m.get("name", "") for m in payload.get("models", []) if isinstance(m, dict)]
        print(f"[ok] Ollama server is running at {healthy_host}")
        print(f"[ok] models: {', '.join(models[:10]) if models else '(no models found)'}")
        return 0

    if args.no_start:
        print("[error] server not healthy and --no-start is set")
        return 1

    started, start_msg = _start_ollama_serve()
    print(f"[start] {start_msg}")
    if not started:
        return 1

    deadline = time.time() + max(args.wait_sec, 0.0)
    while time.time() <= deadline:
        healthy_host, _ = _find_healthy_host(hosts, args.timeout_sec)
        if healthy_host:
            payload = _fetch_tags(healthy_host, args.timeout_sec)
            models = [m.get("name", "") for m in payload.get("models", []) if isinstance(m, dict)]
            print(f"[ok] Ollama server became healthy at {healthy_host}")
            print(f"[ok] models: {', '.join(models[:10]) if models else '(no models found)'}")
            return 0
        time.sleep(max(args.interval_sec, 0.1))

    print("[error] Ollama did not become healthy before timeout")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
