#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MANIFEST=${1:-"$REPO_ROOT/docs/output_move_restore_manifest_20260329.tsv"}

if [[ ! -f "$MANIFEST" ]]; then
  echo "manifest not found: $MANIFEST" >&2
  exit 1
fi

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r original_path cloud_path; do
  if [[ -z "$original_path" || -z "$cloud_path" ]]; then
    continue
  fi

  target="$REPO_ROOT/$original_path"
  source_path="$cloud_path"

  if [[ ! -f "$source_path" ]]; then
    echo "skip missing: $source_path" >&2
    continue
  fi

  mkdir -p "$(dirname -- "$target")"
  mv -- "$source_path" "$target"
  echo "restored $original_path"
done
