#!/usr/bin/env bash
# 72B teacher sanity check — VLMBias only, first 200 samples.
# Uses device_map=auto to spread the 72B across 2 H800s (set via CUDA_VISIBLE_DEVICES).
# Do NOT scale this to all datasets / all samples until the 32B run shows signal.
#
# Usage:
#   bash experiments/E0_image_null_delta/scripts/run_e0_teacher72b_sanity.sh

set -euo pipefail

CONFIG="${CONFIG:-experiments/E0_image_null_delta/configs/e0_default.yaml}"
RESULTS_DIR="${RESULTS_DIR:-experiments/E0_image_null_delta/results}"
GPUS="${GPUS:-0,1}"
N_LIMIT="${N_LIMIT:-200}"
MODEL_KEY="teacher72b"
DATASET="vlmbias_main"

mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/logs"

OUT="${RESULTS_DIR}/e0_${MODEL_KEY}_${DATASET}.sanity.jsonl"
LOG="${RESULTS_DIR}/logs/${MODEL_KEY}_${DATASET}.sanity.log"

echo "[run_e0_teacher72b_sanity] running ${N_LIMIT} samples of ${DATASET} on GPUs ${GPUS}"
CUDA_VISIBLE_DEVICES="$GPUS" \
    python -m experiments.E0_image_null_delta.src.dual_forward \
    --config "$CONFIG" \
    --model "$MODEL_KEY" \
    --dataset "$DATASET" \
    --shard-index 0 --num-shards 1 \
    --limit "$N_LIMIT" \
    --output "$OUT" \
    2>&1 | tee "$LOG"

echo "[run_e0_teacher72b_sanity] done: $OUT"
