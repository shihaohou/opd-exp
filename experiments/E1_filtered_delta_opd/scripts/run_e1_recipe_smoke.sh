#!/usr/bin/env bash
# Smoke launcher for the four E1 on-policy recipes.
#
# Usage:
#     E1_TRAIN_PARQUET=... E1_VAL_PARQUET=... bash scripts/run_e1_recipe_smoke.sh A
#
# Argument: A | B | C | D  → recipe_A_vanilla_kd / _B_raw_delta_kd / etc.
#
# Run from the repo root so the in-yaml `verl/trainer/config` search path
# and `experiments/E1_filtered_delta_opd/configs/agent_loop.yaml` resolve.
# This script just composes env defaults + Hydra overrides; it does NOT
# touch the model weights itself.

set -euo pipefail

RECIPE_LETTER=${1:?usage: $0 <A|B|C|D> [extra hydra overrides]}
case "$RECIPE_LETTER" in
    A) RECIPE_NAME=recipe_A_vanilla_kd ;;
    B) RECIPE_NAME=recipe_B_raw_delta_kd ;;
    C) RECIPE_NAME=recipe_C_filtered_kd ;;
    D) RECIPE_NAME=recipe_D_filtered_delta_kd ;;
    *) echo "Unknown recipe $RECIPE_LETTER (expected A|B|C|D)" >&2; exit 1 ;;
esac
shift  # consume the recipe letter so "$@" only carries Hydra-style key=value overrides

# ---- Required: dataset parquets (output of precompute_teacher.py) ----
: "${E1_TRAIN_PARQUET:?must point to the precomputed E1 train parquet}"
: "${E1_VAL_PARQUET:?must point to the precomputed E1 val parquet}"

# ---- Defaults from CLAUDE.md § Models root ----
MODELS_ROOT=${MODELS_ROOT:-/home/web_server/antispam/project/houshihao/models}
E1_STUDENT_MODEL=${E1_STUDENT_MODEL:-$MODELS_ROOT/Qwen2.5-VL-7B-Instruct}
E1_TEACHER_MODEL=${E1_TEACHER_MODEL:-$MODELS_ROOT/Qwen2.5-VL-32B-Instruct}

# ---- GPU layout (single 8-GPU node) ----
# Actor/rollout pool and teacher pool are DISJOINT — they don't share GPUs.
# Default split for an 8-GPU box: 4 actor + 4 teacher = 8 total.
# Override NGPUS_PER_NODE / TEACHER_WORLD_SIZE to retune; both must sum to
# <= the number of GPUs that are actually free on the box.
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
NNODES=${NNODES:-1}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-4}
TEACHER_TP=${TEACHER_TP:-4}      # single teacher replica, tp=4 inside the teacher pool
ROLLOUT_TP=${ROLLOUT_TP:-2}      # 2 vLLM rollout replicas inside the actor pool (4 / 2 = 2)

# ---- Batch / length ----
# Qwen2.5-VL image processor expands each image into ~1000-1500 image_pad
# tokens at default resolution, so the prompt cap MUST leave room for them
# (the small geo3k example default of 1024 drops every multimodal sample).
# 4096 prompt + 2048 response covers single-image ViRL39K with headroom.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
MAX_NUM_TOKENS=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 ))

# ---- Pin PYTHONPATH so Ray workers can resolve the worker_process_setup_hook ----
# The runtime_env.env_vars route is too late: Ray imports the hook FQDN before
# applying the worker's env_vars. By exporting PYTHONPATH in the launcher shell,
# the entire process tree (driver → Ray subprocesses → workers) inherits it,
# and `experiments.E1_filtered_delta_opd...` resolves at hook-load time.
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ---- Hydra overrides on top of the recipe yaml ----
python -m experiments.E1_filtered_delta_opd.src.on_policy.entrypoint \
    --config-path="$PWD/experiments/E1_filtered_delta_opd/configs" \
    --config-name="$RECIPE_NAME" \
    data.train_files="['$E1_TRAIN_PARQUET']" \
    data.val_files="['$E1_VAL_PARQUET']" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path="$E1_STUDENT_MODEL" \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    distillation.teacher_models.teacher_model.model_path="$E1_TEACHER_MODEL" \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=$TEACHER_TP \
    distillation.teacher_models.teacher_model.inference.max_model_len=$MAX_NUM_TOKENS \
    distillation.n_gpus_per_node=$TEACHER_WORLD_SIZE \
    distillation.nnodes=$NNODES \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    "$@"
