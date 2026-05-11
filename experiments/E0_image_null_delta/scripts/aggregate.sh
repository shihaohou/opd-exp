#!/usr/bin/env bash
# Aggregate all teacher/student shard jsonls into the E0 summary CSV + verdict.
# CPU-only; no model load. Safe to run repeatedly during a sweep.
#
# Usage:
#   bash experiments/E0_image_null_delta/scripts/aggregate.sh

set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-experiments/E0_image_null_delta/results}"

python -m experiments.E0_image_null_delta.src.metrics \
    --results-dir "$RESULTS_DIR" \
    --teacher-glob "e0_teacher32b_*.jsonl" \
    --student-glob "e0_student7b_*.jsonl"

echo "[aggregate] outputs:"
ls -la "$RESULTS_DIR"/e0_summary.csv "$RESULTS_DIR"/e0_verdict.md "$RESULTS_DIR"/top_delta_tokens.json
