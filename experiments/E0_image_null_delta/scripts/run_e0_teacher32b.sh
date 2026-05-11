#!/usr/bin/env bash
# Run the E0 dual-forward diagnostic with Qwen2.5-VL-32B as teacher.
# 8-way data parallel across the 8 H800s — each shard is a fully independent
# Python process pinned to one GPU. All three datasets are run per shard.
#
# Usage (from repo root, with `source activate.sh` already done):
#   bash experiments/E0_image_null_delta/scripts/run_e0_teacher32b.sh
#
# Tunables via env vars:
#   NUM_SHARDS  (default 8)  — how many parallel processes / GPUs to use
#   CONFIG      (default e0_default.yaml)
#   RESULTS_DIR (default experiments/E0_image_null_delta/results)
#   DATASETS    (default "vlmbias_main pope_adversarial mathvista_mini")
#
# Output:
#   $RESULTS_DIR/e0_teacher32b_${DATASET}.shard${i}.jsonl

set -euo pipefail

NUM_SHARDS="${NUM_SHARDS:-8}"
CONFIG="${CONFIG:-experiments/E0_image_null_delta/configs/e0_default.yaml}"
RESULTS_DIR="${RESULTS_DIR:-experiments/E0_image_null_delta/results}"
DATASETS="${DATASETS:-vlmbias_main pope_adversarial mathvista_mini}"
MODEL_KEY="teacher32b"

mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/logs"

# Run datasets sequentially (one full sweep at a time), with N parallel shards
# inside each sweep. This keeps GPU memory ownership simple: each shard process
# loads the 32B model once on its assigned GPU, then iterates samples.
for DATASET in $DATASETS; do
    echo "[run_e0_teacher32b] dataset=$DATASET, fanning out $NUM_SHARDS shards"
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
    # Wait for all shards in this dataset to finish before moving on.
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    echo "[run_e0_teacher32b] dataset=$DATASET done"
done

echo "[run_e0_teacher32b] all datasets complete. Run aggregate.sh next."
