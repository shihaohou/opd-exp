#!/usr/bin/env bash
# Phase 3 (1K mini-sweep, Plan C) runner — idempotent + structured output.
#
# Usage:
#   export DATASETS=/path/to/datasets
#   export MODELS=/path/to/models
#   export RESULTS=/path/to/repo/results
#   bash experiments/E1_filtered_delta_opd/scripts/run_phase3.sh
#
# Behaviour:
#   * Every step writes a marker on success. Re-running this script
#     SKIPS completed steps. Dev-box crashes are recoverable — just
#     re-run and it picks up where it left off.
#   * Outputs land under $PHASE3_DIR (default $RESULTS/02_phase3_1k),
#     organized by sub-step + per-config sub-dirs. No more flat dump.
#   * On first run, migrates the previous flat-layout files
#     (e1_mini_1k_train.parquet, e1_1k_precompute_shard_*.jsonl, etc.)
#     into the new layout so step 3a/3b/3c don't redo work that already
#     succeeded.
#   * If a training step produces zero `training/global_step:` lines,
#     the script bails immediately with the log tail printed — no more
#     "ran the whole loop for nothing" failure mode.

set -uo pipefail   # NOT -e: we want to keep going on per-config failures

# ---------------------------------------------------------------------------
# Env vars (required).
# ---------------------------------------------------------------------------
: "${DATASETS:?DATASETS must be set, e.g. export DATASETS=/.../datasets}"
: "${MODELS:?MODELS must be set, e.g. export MODELS=/.../models}"
: "${RESULTS:?RESULTS must be set, e.g. export RESULTS=/.../repo/results}"

PHASE3_DIR=${PHASE3_DIR:-$RESULTS/02_phase3_1k}
mkdir -p "$PHASE3_DIR"/{mixture,precompute/shards,parquet,configs}

# Walk to repo root so relative imports / launcher paths work.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO_ROOT"

LAUNCHER="experiments/E1_filtered_delta_opd/scripts/run_e1_recipe_smoke.sh"
TEACHER_VLMBIAS_GLOB="experiments/E0_image_null_delta/results/e0_teacher32b_vlmbias_main.shard*.jsonl"
BASELINE_VLMBIAS="$RESULTS/baseline_7b_vlmbias.jsonl"

# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
log()       { echo "[$(date '+%H:%M:%S')] $*"; }
is_done()   { [ -s "$1" ]; }
mark_done() { touch "$1"; }
require_file() { [ -s "$1" ] || { log "[FAIL] required file missing: $1"; exit 1; }; }

# ---------------------------------------------------------------------------
# One-time migration: pull existing flat-layout files into PHASE3_DIR.
# Safe to call repeatedly — only moves files when destination doesn't exist.
# ---------------------------------------------------------------------------
migrate_flat_layout() {
  local moved=0

  # Mixture: $DATASETS/e1_mini_v1/mixture_1k.jsonl → PHASE3_DIR/mixture/
  if [ ! -s "$PHASE3_DIR/mixture/mixture_1k.jsonl" ] \
     && [ -s "$DATASETS/e1_mini_v1/mixture_1k.jsonl" ]; then
    cp "$DATASETS/e1_mini_v1/mixture_1k.jsonl" "$PHASE3_DIR/mixture/"
    moved=1
  fi

  # Precompute shards: $RESULTS/e1_1k_precompute_shard_N.jsonl
  if [ ! -s "$PHASE3_DIR/precompute/all.jsonl" ]; then
    local n_shards=0
    for f in "$RESULTS"/e1_1k_precompute_shard_*.jsonl; do
      [ -e "$f" ] || continue
      mv "$f" "$PHASE3_DIR/precompute/shards/$(basename "$f")"
      n_shards=$((n_shards + 1))
    done
    for f in "$RESULTS"/log_1k_precompute_shard_*.log; do
      [ -e "$f" ] || continue
      mv "$f" "$PHASE3_DIR/precompute/"
    done
    if [ $n_shards -gt 0 ]; then
      cat "$PHASE3_DIR"/precompute/shards/*.jsonl > "$PHASE3_DIR/precompute/all.jsonl"
      moved=1
    fi
  fi

  # Parquet: $RESULTS/e1_mini_1k_train.parquet
  if [ ! -s "$PHASE3_DIR/parquet/train.parquet" ] \
     && [ -s "$RESULTS/e1_mini_1k_train.parquet" ]; then
    mv "$RESULTS/e1_mini_1k_train.parquet" "$PHASE3_DIR/parquet/train.parquet"
    moved=1
  fi

  # Failed Config A train log (from the first bad run with wrong launcher path).
  # Move it aside so a retry produces a fresh log.
  if [ -s "$RESULTS/log_1k_A.log" ] && [ ! -d "$PHASE3_DIR/configs/A" ]; then
    mkdir -p "$PHASE3_DIR/configs/A/logs"
    mv "$RESULTS/log_1k_A.log" "$PHASE3_DIR/configs/A/logs/train_FAILED_wrong_launcher_path.log"
    mv "$RESULTS"/train_1k_A.{csv,json,md} "$PHASE3_DIR/configs/A/logs/" 2>/dev/null || true
  fi

  [ $moved -eq 1 ] && log "[migrate] moved flat-layout files into $PHASE3_DIR/"
  return 0
}

migrate_flat_layout

# ---------------------------------------------------------------------------
# Step 3a: 1K mixture (700 ViRL39K + 300 synth).
# ---------------------------------------------------------------------------
MIXTURE_OUT="$PHASE3_DIR/mixture/mixture_1k.jsonl"
if is_done "$MIXTURE_OUT"; then
  log "[skip] 3a mixture ($(wc -l < "$MIXTURE_OUT") rows)"
else
  log "[run]  3a mixture"
  python -m experiments.E1_filtered_delta_opd.data.mixture \
    --output "$MIXTURE_OUT" \
    --virl39k-root "$DATASETS/ViRL39K" \
    --synth-dir "$DATASETS/e1_synth_v1" \
    --n-virl39k 700 --n-synthetic 300 \
    --skip pope_style tallyqa \
    2>&1 | tee "$PHASE3_DIR/mixture/build.log"
  require_file "$MIXTURE_OUT"
  log "[done] 3a mixture ($(wc -l < "$MIXTURE_OUT") rows)"
fi

# ---------------------------------------------------------------------------
# Step 3b: 32B teacher precompute on 1K (8 shards parallel, ~10 min).
# ---------------------------------------------------------------------------
PRECOMPUTE_OUT="$PHASE3_DIR/precompute/all.jsonl"
if is_done "$PRECOMPUTE_OUT" && [ "$(wc -l < "$PRECOMPUTE_OUT")" -ge 1000 ]; then
  log "[skip] 3b precompute ($(wc -l < "$PRECOMPUTE_OUT") rows)"
else
  log "[run]  3b precompute (32B teacher, 8 shards)"
  for SHARD in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$SHARD \
    python -m experiments.E1_filtered_delta_opd.src.precompute_teacher \
      --bucket mixture \
      --loader-kwargs "{\"manifest_path\":\"$MIXTURE_OUT\"}" \
      --model-path "$MODELS/Qwen2.5-VL-32B-Instruct" \
      --output "$PHASE3_DIR/precompute/shards/shard${SHARD}.jsonl" \
      --shard-index "$SHARD" --num-shards 8 \
      --device-map auto \
      > "$PHASE3_DIR/precompute/shards/shard${SHARD}.log" 2>&1 &
  done
  wait
  cat "$PHASE3_DIR"/precompute/shards/shard*.jsonl > "$PRECOMPUTE_OUT"
  require_file "$PRECOMPUTE_OUT"
  log "[done] 3b precompute ($(wc -l < "$PRECOMPUTE_OUT") rows)"
fi

# ---------------------------------------------------------------------------
# Step 3c: build training parquet.
# ---------------------------------------------------------------------------
PARQUET_OUT="$PHASE3_DIR/parquet/train.parquet"
if is_done "$PARQUET_OUT"; then
  log "[skip] 3c parquet ($(du -h "$PARQUET_OUT" | cut -f1))"
else
  log "[run]  3c parquet"
  python -m experiments.E1_filtered_delta_opd.data.make_train_parquet \
    --jsonl "$PHASE3_DIR"/precompute/shards/shard*.jsonl \
    --student-tokenizer "$MODELS/Qwen2.5-VL-7B-Instruct" \
    --output "$PARQUET_OUT" \
    2>&1 | tee "$PHASE3_DIR/parquet/build.log"
  require_file "$PARQUET_OUT"
  log "[done] 3c parquet"
fi

# ---------------------------------------------------------------------------
# Step 3d: 4-config sweep (train → merge → eval → metrics).
# ---------------------------------------------------------------------------

run_one_config() {
  local L="$1"
  local CDIR="$PHASE3_DIR/configs/$L"
  mkdir -p "$CDIR/eval/shards" "$CDIR/logs"

  # === Train ===
  if [ -f "$CDIR/.train_done" ]; then
    log "[skip] $L train"
  else
    log "[run]  $L train"
    E1_TRAIN_PARQUET="$PARQUET_OUT" \
    E1_VAL_PARQUET="$PARQUET_OUT" \
    bash "$LAUNCHER" "$L" 2>&1 | tee "$CDIR/logs/train.log"
    local LAUNCHER_EXIT=${PIPESTATUS[0]}

    # Check 1: launcher exit code (pipefail preserved via PIPESTATUS).
    if [ "$LAUNCHER_EXIT" -ne 0 ]; then
      log "[FAIL] $L train: launcher exited $LAUNCHER_EXIT. Tail of log:"
      tail -50 "$CDIR/logs/train.log"
      return 1
    fi

    # Check 2: at least some training steps logged.
    if ! grep -q "training/global_step:" "$CDIR/logs/train.log"; then
      log "[FAIL] $L train produced no steps. Tail of log:"
      tail -50 "$CDIR/logs/train.log"
      return 1
    fi

    # Check 3: no uncaught Traceback / RuntimeError / RayTaskError
    # (catches mid-train crashes where steps 1..N-1 logged successfully but
    # step N hit an exception. Old version of the script wrongly marked
    # these as "train done" and proceeded with bad checkpoints — see the
    # 4096 vs 4849 prompt-length crash on 2026-05-13.)
    if grep -qE "RuntimeError|RayTaskError|raise self\._exception" "$CDIR/logs/train.log"; then
      log "[FAIL] $L train: exception found mid-run. Excerpt:"
      grep -B 2 -A 8 -E "RuntimeError|RayTaskError" "$CDIR/logs/train.log" | head -40
      return 1
    fi

    mark_done "$CDIR/.train_done"
    log "[done] $L train"
  fi

  # Extract train metrics — always re-run, it's cheap and idempotent.
  python -m experiments.E1_filtered_delta_opd.scripts.extract_train_metrics \
    "$CDIR/logs/train.log" \
    --out "$CDIR/train_metrics" \
    --config "$L" \
    > "$CDIR/logs/extract_metrics.log" 2>&1 || log "  (extract_train_metrics warned, see logs/extract_metrics.log)"

  # === Merge FSDP → HF ===
  if [ -f "$CDIR/.merge_done" ] && [ -d "$CDIR/checkpoint_hf" ]; then
    log "[skip] $L merge"
  else
    log "[run]  $L merge (FSDP → HF)"
    # Find the actor checkpoint dir written by THIS config's training.
    # Prefer dirs newer than this train.log to avoid grabbing a stale one.
    local CKPT_SRC
    CKPT_SRC=$(find . -path "*/global_step_*/actor" -newer "$CDIR/logs/train.log" -type d 2>/dev/null | head -1 || true)
    if [ -z "$CKPT_SRC" ]; then
      CKPT_SRC=$(find . -path "*/global_step_*/actor" -type d 2>/dev/null | head -1 || true)
    fi
    if [ -z "$CKPT_SRC" ]; then
      log "[FAIL] no global_step_*/actor dir found for $L"
      return 1
    fi
    log "       source checkpoint: $CKPT_SRC"

    python -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "$CKPT_SRC" \
      --target_dir "$CDIR/checkpoint_hf" \
      2>&1 | tee "$CDIR/logs/merge.log"

    if [ ! -d "$CDIR/checkpoint_hf" ] || [ -z "$(ls -A "$CDIR/checkpoint_hf" 2>/dev/null)" ]; then
      log "[FAIL] $L merge produced empty checkpoint_hf"
      return 1
    fi

    # Archive source FSDP ckpt under the config dir so the next config
    # doesn't pick it up via `find`.
    mv "$CKPT_SRC" "$CDIR/source_fsdp_ckpt" || true
    mark_done "$CDIR/.merge_done"
    log "[done] $L merge"
  fi

  # === Eval on 3 datasets (8-shard parallel each) ===
  if [ -f "$CDIR/metrics.json" ]; then
    log "[skip] $L eval"
  else
    log "[run]  $L eval (3 datasets × 8 shards)"
    for DS in vlmbias pope mathvista; do
      local DSROOT
      case $DS in
        vlmbias)   DSROOT="$DATASETS/VLMBias" ;;
        pope)      DSROOT="$DATASETS/POPE-adversarial" ;;
        mathvista) DSROOT="$DATASETS/MathVista-mini" ;;
      esac
      for SHARD in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=$SHARD \
        python -m experiments.E1_filtered_delta_opd.src.eval_tei infer \
          --checkpoint "$CDIR/checkpoint_hf" \
          --dataset "$DS" --dataset-root "$DSROOT" \
          --output "$CDIR/eval/shards/${DS}_shard${SHARD}.jsonl" \
          --shard-index "$SHARD" --num-shards 8 \
          > "$CDIR/logs/eval_${DS}_shard${SHARD}.log" 2>&1 &
      done
      wait
      cat "$CDIR"/eval/shards/${DS}_shard*.jsonl > "$CDIR/eval/${DS}.jsonl"
    done

    python -m experiments.E1_filtered_delta_opd.src.eval_tei metrics \
      --student-vlmbias-jsonl "$CDIR/eval/vlmbias.jsonl" \
      --teacher-vlmbias-jsonl $TEACHER_VLMBIAS_GLOB \
      --student-pope-jsonl "$CDIR/eval/pope.jsonl" \
      --student-mathvista-jsonl "$CDIR/eval/mathvista.jsonl" \
      --student-base-vlmbias-jsonl "$BASELINE_VLMBIAS" \
      --tokenizer-path "$MODELS/Qwen2.5-VL-7B-Instruct" \
      --output "$CDIR/metrics.json" \
      > "$CDIR/logs/metrics.log" 2>&1
    log "[done] $L eval"
  fi
}

for L in A B C D; do
  if ! run_one_config "$L"; then
    log "[ABORT] Config $L failed; subsequent configs not attempted."
    log "         Inspect $PHASE3_DIR/configs/$L/logs/ and re-run this script."
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Final A/B/C/D + baseline comparison table.
# ---------------------------------------------------------------------------
log "[run]  comparison"
python3 - <<PYEOF
import json, os
PD = "$PHASE3_DIR"
RES = "$RESULTS"

rows = []
# Baseline first
try:
    with open(f"{RES}/eval_baseline_7b.json") as f:
        rows.append(("base", json.load(f)))
except FileNotFoundError:
    pass
for L in "ABCD":
    p = f"{PD}/configs/{L}/metrics.json"
    if os.path.exists(p):
        with open(p) as f:
            rows.append((f"cfg_{L}", json.load(f)))

def fmt_val(v, w, prec=3):
    if v is None: return f"{'—':>{w}}"
    if isinstance(v, float): return f"{v:>{w}.{prec}f}"
    return f"{v:>{w}}"

header = f"{'cfg':>7}  {'recog':>6}  {'gainLN':>7}  {'TEI':>6}  {'Acc|Tw':>7}  {'Esc':>5}  {'popF1':>6}  {'popY':>6}  {'halluc':>6}  {'mvAcc':>6}  {'mvLen':>5}"
sep    = "-" * len(header)
lines = [header, sep]

for label, r in rows:
    va = r['vlmbias']['recognition_aggregate']
    lines.append(
        f"{label:>7}  "
        f"{fmt_val(va['accuracy'], 6)}  "
        f"{fmt_val(va.get('gain_margin_lengthnorm'), 7)}  "
        f"{fmt_val(r['tei']['tei_rate'], 6)}  "
        f"{fmt_val(r['tei']['acc_s_on_t_wrong'], 7)}  "
        f"{fmt_val(r['tei']['escape_rate'], 5)}  "
        f"{fmt_val(r['pope']['f1_yes'], 6)}  "
        f"{fmt_val(r['pope']['yes_rate'], 6)}  "
        f"{fmt_val(r['pope']['hallucinated_yes'], 6, prec=0)}  "
        f"{fmt_val(r['mathvista']['accuracy'], 6)}  "
        f"{fmt_val(r['mathvista']['response_length_p50'], 5, prec=0)}"
    )

out = "\n".join(lines)
print(out)
with open(f"{PD}/comparison.txt", "w") as f:
    f.write(out + "\n")
print(f"\n→ saved to {PD}/comparison.txt")
PYEOF

log "Phase 3 complete."
