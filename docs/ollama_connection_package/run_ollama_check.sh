#!/usr/bin/env bash
set -euo pipefail

cd /home/user/personality-dungeon
source venv/bin/activate
python scripts/ollama_smoke_test.py --wait-sec 15 --timeout-sec 3
