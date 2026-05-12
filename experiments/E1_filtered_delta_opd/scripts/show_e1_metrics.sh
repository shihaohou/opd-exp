#!/usr/bin/env bash
# Extract the E1 monitoring keys from a smoke run log file.
#
# Usage:
#     # Save the run output, then inspect:
#     bash scripts/run_e1_recipe_smoke.sh C ... 2>&1 | tee /tmp/c.log
#     bash experiments/E1_filtered_delta_opd/scripts/show_e1_metrics.sh /tmp/c.log
#
# By default it shows the LAST step's metrics (the most useful for a smoke
# health check). Pass --all to dump every step.

set -euo pipefail

LOG=${1:?usage: $0 <path/to/log> [--all]}
MODE=${2:-last}

if [[ "$MODE" == "--all" ]]; then
    # Every step, one metric per line, grouped per-step.
    awk '/training\/global_step:/{print "--- step ---"; for(i=1;i<=NF;i++) print $i}' "$LOG" \
        | grep -E "^(--- step ---|actor/e1_v1/|actor/distillation/|actor/loss|actor/grad_norm|response_length/)"
else
    # Just the most recent step.
    grep "training/global_step:" "$LOG" | tail -1 \
        | tr ' ' '\n' \
        | grep -E "^(actor/e1_v1/|actor/distillation/loss|actor/loss|actor/grad_norm|response_length/mean|response_length/min|response_length/max|training/global_step|training/epoch)"
fi
