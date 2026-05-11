#!/usr/bin/env bash
# Run the E0 dual-forward diagnostic with Qwen2.5-VL-7B as student.
# Same 8-way data-parallel layout as the 32B teacher run, but on the 7B model.
# Student outputs are used only for metric 5a (student-teacher wrong overlap)
# on VLMBias — POPE and MathVista runs are optional but cheap.
#
# Usage:
#   bash experiments/E0_image_null_delta/scripts/run_e0_student7b.sh

set -euo pipefail

NUM_SHARDS="${NUM_SHARDS:-8}"
CONFIG="${CONFIG:-experiments/E0_image_null_delta/configs/e0_default.yaml}"
RESULTS_DIR="${RESULTS_DIR:-experiments/E0_image_null_delta/results}"
DATASETS="${DATASETS:-vlmbias_main pope_adversarial mathvista_mini}"
MODEL_KEY="student7b"

mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/logs"

for DATASET in $DATASETS; do
    echo "[run_e0_student7b] dataset=$DATASET, fanning out $NUM_SHARDS shards"
    pids=()
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        OUT="${RESULTS_DIR}/e0_${MODEL_KEY}_${DATASET}.shard${i}.jsonl"
        LOG="${RESULTS_DIR}/logs/${MODEL_KEY}_${DATASET}.shard${i}.log"
        CUDA_VISIBLE_DEVICES="$i" \
            python -m experiments.E0_image_null_delta.src.dual_forward \
            --config "$CONFIG" \
            --model "$MODEL_KEY" \
            --dataset "$DATASET" \
            --shard-index "$i" --num-shards "$NUM_SHARDS" \
            --output "$OUT" \
            >"$LOG" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    echo "[run_e0_student7b] dataset=$DATASET done"
done

echo "[run_e0_student7b] all datasets complete."
