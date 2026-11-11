#!/bin/sh
# Sequential worker for an explicit config list; parallel launch stays outside.
# Usage: sh run_study.sh <logdir> <config> [<config> ...]
set -e
logdir="$1"; shift
mkdir -p "$logdir"
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
for cfg in "$@"; do
  name=$(basename "$cfg" .json)
  echo "start $name $(date -u +%H:%M:%S)"
  uv run --locked python -m experiments.inverse_cascade.run --config "$cfg" \
    > "$logdir/$name.log" 2>&1
  echo "done $name $(date -u +%H:%M:%S) $(tail -1 "$logdir/$name.log")"
done
