#!/usr/bin/env bash
# Quick read-only status of Phase 3 progress.
#
# Usage:
#   export RESULTS=/path/to/repo/results
#   bash experiments/E1_filtered_delta_opd/scripts/phase3_status.sh

: "${RESULTS:?RESULTS must be set}"
PHASE3_DIR=${PHASE3_DIR:-$RESULTS/02_phase3_1k}

if [ ! -d "$PHASE3_DIR" ]; then
  echo "Phase 3 dir does not exist yet: $PHASE3_DIR"
  echo "Run scripts/run_phase3.sh to start."
  exit 0
fi

echo "=== Phase 3 status @ $PHASE3_DIR ==="
echo

check() {
  local label="$1"
  local file="$2"
  if [ -s "$file" ]; then
    local size
    size=$(du -h "$file" 2>/dev/null | cut -f1)
    local extra=""
    if [[ "$file" == *.jsonl ]]; then
      extra=" ($(wc -l < "$file") lines)"
    fi
    echo "  ✓ $label: $size$extra"
  else
    echo "  ✗ $label: missing"
  fi
}

# Step 3a/3b/3c
check "3a mixture"   "$PHASE3_DIR/mixture/mixture_1k.jsonl"
check "3b precompute" "$PHASE3_DIR/precompute/all.jsonl"
check "3c parquet"    "$PHASE3_DIR/parquet/train.parquet"

echo
echo "  Per-config:"
for L in A B C D; do
  CDIR="$PHASE3_DIR/configs/$L"
  if [ ! -d "$CDIR" ]; then
    echo "    Config $L: not started"
    continue
  fi
  s_train="✗"; [ -f "$CDIR/.train_done" ] && s_train="✓"
  s_merge="✗"; [ -f "$CDIR/.merge_done" ] && s_merge="✓"
  s_eval="✗";  [ -f "$CDIR/metrics.json" ] && s_eval="✓"

  extra=""
  if [ "$s_train" = "✓" ] && [ -f "$CDIR/logs/train.log" ]; then
    n_steps=$(grep -c "training/global_step:" "$CDIR/logs/train.log" 2>/dev/null || echo 0)
    extra="  ($n_steps train steps logged)"
  fi
  echo "    Config $L: train=$s_train  merge=$s_merge  eval=$s_eval$extra"
done

echo
if [ -s "$PHASE3_DIR/comparison.txt" ]; then
  echo "  ✓ comparison.txt exists. Showing:"
  echo
  sed 's/^/    /' "$PHASE3_DIR/comparison.txt"
fi
