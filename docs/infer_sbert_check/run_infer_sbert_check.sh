#!/usr/bin/env bash
set -euo pipefail

API_URL=${API_URL:-"http://127.0.0.1:8000/personality/infer_sbert"}
TEXT=${TEXT:-"quick test"}

curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$TEXT\"}"
