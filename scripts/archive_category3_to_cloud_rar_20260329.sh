#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PLAN=${1:-"$REPO_ROOT/docs/category3_archive_plan_20260329.tsv"}
DEST_ROOT=${2:-"/mnt/e/User/personality-dungeon_cloud/outputs_csv_move_20260329/category3_rar"}

if [[ ! -f "$PLAN" ]]; then
  echo "plan not found: $PLAN" >&2
  exit 1
fi

if ! command -v rar >/dev/null 2>&1; then
  echo "rar not found. Install package 'rar' first, then rerun this script." >&2
  exit 2
fi

mkdir -p "$DEST_ROOT"

tail -n +2 "$PLAN" | while IFS=$'\t' read -r category original_path archive_target csv_count size_gib; do
  if [[ -z "$original_path" || -z "$archive_target" ]]; then
    continue
  fi

  source_dir="$REPO_ROOT/$original_path"
  archive_name=$(basename -- "$archive_target")
  archive_path="$DEST_ROOT/$archive_name"

  if [[ ! -d "$source_dir" ]]; then
    echo "skip missing dir: $source_dir" >&2
    continue
  fi

  echo "archiving $original_path -> $archive_path"
  rar a -r -m5 -ep1 -- "$archive_path" "$source_dir"
done
