# NEXT — what to do next
*Last updated: 2026-05-13 (after Day 2 data pipeline). Pair this file with `PROGRESS.md`.*

> **For new Claude sessions:** if you're starting fresh, first read `CLAUDE.md`, then `PROGRESS.md` (what we did), then this file. Recommended next concrete action is at **§ Right now, Day 3** below. If anything in the on-policy trainer breaks at smoke time, also read `docs/e1_smoke_runbook.md`.

---

## Status overview

```
E0 (forward-only diagnostic)     : DONE — Conditional GO
E0.3-A (per-topic m5a)           : DONE
E0.3-B (length-norm gain)        : DONE — per-topic direction stable
E0.3-C (PPL_S smoke)             : deferred (does not block E1)
E1 design                        : DONE — locked on-policy v1 with 4 configs A/B/C/D
E1 Day 1   (loader + spikes)     : DONE
E1 Day 1.5 (Stage 2 trainer code): DONE — agent_loop + 4 losses + 4 configs + parquet
                                   builder shipped; all 4 configs (A/B/C/D) verified
                                   end-to-end on 50 ViRL39K samples
E1 Day 2   (data buckets 2/3 + dedup + mixture): DONE locally — 6 deliverables shipped,
                                   35 local unit tests pass; awaiting commit + server
                                   data download
E1 Day 3   (precompute 8K + B/C/D smoke on new parquet): NEXT
E1 Day 4+  (full 4-config sweep + eval): pending
E2 (mask sensitivity)            : not started
```

---

## Right now, Day 3 (eval-first 4-step plan, per 2026-05-13 GPT realignment)

Day 2 data pipeline shipped locally. **Plan changed**: eval before train.
Without `src/eval_tei.py`, an 8K sweep produces only ordinary accuracy
numbers and the E1 causal questions (does OPD inherit teacher errors,
does delta amplify wrong-direction influence, does filtered Delta-OPD
reduce inheritance) stay unanswered. **Eval is the gate; precompute is downstream.**

Read in order before starting:
1. `experiments/E1_filtered_delta_opd/README.md` § "E1 Protocol" (the boxed block at the top) — the only source of truth for what E1 tests.
2. `PROGRESS.md` § "E1 — Day 2 / Data pipeline" + § "External review (2026-05-13 evening)"
3. `experiments/E1_filtered_delta_opd/on_policy_v1_design.md` § 8 (engineering staged plan)
4. `docs/e1_smoke_runbook.md` (R1-R5 if anything in the trainer breaks again)

### The 4 steps (strict order)

**Step 1 — Implement `src/eval_tei.py` (~250 LoC, CPU-friendly metric layer)**

One summary JSON per checkpoint with the Primary + Safety block from README § "E1 Protocol":

| Section | Metrics |
|---|---|
| VLMBias | global acc, per-topic acc, Recognition Aggregate acc, length-norm `gain_margin` per topic |
| TEI family | `Acc_S \| T_wrong`, TEI rate, Escape rate (T_wrong sourced from E0 32B-teacher VLMBias jsonls) |
| POPE-adv | acc, F1, yes-rate, grounded-yes count, hallucinated-yes count |
| MathVista-mini | acc, response length p50/p95, parse success rate |

Reuse from E0 `src/metrics.py`: length-normalized `gain_margin` (boundary-trick), VLMBias topic taxonomy (Optical Illusion vs Recognition Aggregate), `\boxed{}` parser. New for E1: HF-format student checkpoint loading (verify the verl FSDP → HF consolidation path), TEI / Escape definitions on E0 T_wrong subset, POPE F1 / yes-rate, MathVista choice-parse success.

Metric layer testable on Mac with fixture jsonls (no GPU). Model loading + dataset-side scoring lives in a separate function that only the server invokes.

**Step 2 — Bucket-3 teacher sanity (8 GPUs on server, ~1h)**

Two commands:

```bash
# 2a. Run 32B teacher on all 1500 synth images (~1h on 8 H800 shard-parallel,
#     or ~5-7h single GPU). The mixture bucket name is "synthetic"; the
#     per-record `extras.topic` carries the sub-topic.
for SHARD in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$SHARD \
  python -m experiments.E1_filtered_delta_opd.src.precompute_teacher \
    --bucket synthetic \
    --loader-kwargs '{"out_dir":"'$DATASETS'/e1_synth_v1"}' \
    --model-path $MODELS/Qwen2.5-VL-32B-Instruct \
    --output $RESULTS/e1_synth_teacher_shard_${SHARD}.jsonl \
    --shard-index $SHARD --num-shards 8 &
done; wait
cat $RESULTS/e1_synth_teacher_shard_*.jsonl > $RESULTS/e1_synth_teacher.jsonl

# 2b. Apply the gate + emit MD / JSON / CSV reports.
python -m experiments.E1_filtered_delta_opd.scripts.synth_sanity \
    $RESULTS/e1_synth_teacher.jsonl \
    --out $RESULTS/e1_synth_sanity
```

`synth_sanity.py` exits 0 (PASS) or 1 (FAIL) and writes:

- `<out>.md`  — verdict + per-topic table, copy-paste into GPT
- `<out>.json` — programmatic aggregate
- `<out>.csv` — per-topic rows for plotting

**Decision gate** (encoded in the script):

| condition | verdict |
|---|---|
| `animal` `accuracy_on_counterfactual >= 0.80` | **FAIL** — teacher is correctly counting; silhouette doesn't trigger the canonical prior |
| `accuracy_on_counterfactual < 0.80` AND `prior_trigger_rate < 0.30` | **FAIL** — teacher errors not from the prior |
| `accuracy_on_counterfactual < 0.80` AND `prior_trigger_rate >= 0.30` | **PASS** — bucket-3 is ready for training |

If FAIL: do NOT proceed to Step 3. Either fix the synth pipeline (CLIP-guided / diffusion-based animal generation in a v2) or drop bucket 3 from the mixture for v1 and accept that Animals failure mode won't be covered. The other 4 topics (grid / flag / board / chess) report accuracy only; they're not the suspect.

**Step 3 — 1K mini-sweep (8 GPUs, ~1.5h total for A/B/C/D)**

Subsample the 8K mixture to 1K (same per-bucket ratio), run A/B/C/D × ~50 steps each, extract metrics, merge checkpoint, run eval_tei. Look at **direction**, not magnitude.

```bash
# 3a. Build a 1K mini parquet (sample 1/8 from the 8K). The simplest path
#     is to re-run `mixture.py` with the per-bucket counts scaled down:
python -m experiments.E1_filtered_delta_opd.data.mixture \
    --output $DATASETS/e1_mini_v1/mixture_1k.jsonl \
    --virl39k-root $DATASETS/ViRL39K --coco-train-root $DATASETS/coco \
    --pope-adv-root $DATASETS/POPE-adversarial \
    --tallyqa-json $DATASETS/tallyqa/train.json \
    --tallyqa-images-root $DATASETS/tallyqa_images \
    --synth-dir $DATASETS/e1_synth_v1 \
    --n-virl39k 500 --n-pope-style 200 --n-tallyqa 112 --n-synthetic 188

# 3b. Precompute teacher on the 1K (~10 min, 8 shards)
for SHARD in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$SHARD \
  python -m experiments.E1_filtered_delta_opd.src.precompute_teacher \
    --bucket mixture \
    --loader-kwargs '{"manifest_path":"'$DATASETS'/e1_mini_v1/mixture_1k.jsonl"}' \
    --model-path $MODELS/Qwen2.5-VL-32B-Instruct \
    --output $RESULTS/e1_1k_precompute_shard_${SHARD}.jsonl \
    --shard-index $SHARD --num-shards 8 &
done; wait

# 3c. Build train parquet (cap a small val set off the same precompute)
python -m experiments.E1_filtered_delta_opd.data.make_train_parquet \
    --jsonl $RESULTS/e1_1k_precompute_shard_*.jsonl \
    --student-tokenizer $MODELS/Qwen2.5-VL-7B-Instruct \
    --output $RESULTS/e1_mini_1k_train.parquet

# 3d. For each config: train, extract metrics, merge checkpoint, eval.
for L in A B C D; do
  E1_TRAIN_PARQUET=$RESULTS/e1_mini_1k_train.parquet \
  E1_VAL_PARQUET=$RESULTS/e1_mini_1k_train.parquet \
  bash scripts/run_e1_recipe_smoke.sh $L 2>&1 | tee $RESULTS/log_1k_$L.log

  # Extract structured train metrics → CSV + MD + JSON; exits 1 on health-check fail
  python -m experiments.E1_filtered_delta_opd.scripts.extract_train_metrics \
      $RESULTS/log_1k_$L.log \
      --out $RESULTS/train_1k_$L --config $L

  # Merge FSDP shards → HF format. CKPT_DIR is whatever
  # `trainer.default_local_dir/global_step_N/actor` the smoke run wrote.
  CKPT_DIR=$RESULTS/e1_1k_$L_ckpt   # adjust to your verl output path
  python -m verl.model_merger merge \
      --backend fsdp \
      --local_dir $CKPT_DIR \
      --target_dir $RESULTS/e1_1k_${L}_hf

  # Run student inference on the 3 eval datasets (~15-30 min each on 1 GPU)
  for DS in vlmbias pope mathvista; do
    python -m experiments.E1_filtered_delta_opd.src.eval_tei infer \
        --checkpoint $RESULTS/e1_1k_${L}_hf \
        --dataset $DS \
        --dataset-root $DATASETS/$(case $DS in vlmbias) echo VLMBias;; \
                                                pope) echo POPE-adversarial;; \
                                                mathvista) echo MathVista-mini;; esac) \
        --output $RESULTS/eval_1k_${L}_${DS}.jsonl
  done

  # Compute metrics summary
  python -m experiments.E1_filtered_delta_opd.src.eval_tei metrics \
      --student-vlmbias-jsonl $RESULTS/eval_1k_${L}_vlmbias.jsonl \
      --teacher-vlmbias-jsonl experiments/E0_image_null_delta/results/e0_teacher32b_vlmbias_main.shard*.jsonl \
      --student-pope-jsonl $RESULTS/eval_1k_${L}_pope.jsonl \
      --student-mathvista-jsonl $RESULTS/eval_1k_${L}_mathvista.jsonl \
      --student-base-vlmbias-jsonl experiments/E0_image_null_delta/results/e0_student7b_vlmbias_main.shard*.jsonl \
      --tokenizer-path $MODELS/Qwen2.5-VL-7B-Instruct \
      --output $RESULTS/eval_1k_${L}.json
done
```

After all 4 configs finish, compare them by directly diffing the 4 `eval_1k_{A,B,C,D}.json` files (or jq into a table). Look at:

| Signal | Interpretation |
|---|---|
| `eval_1k_B.json.vlmbias.recognition_aggregate.accuracy` < A's | Raw delta amplifies wrong-direction influence (H2 ✓) |
| `eval_1k_C.json.tei.tei_rate` < A's (and < E0 baseline 0.620) | Filtering + CE reduces inheritance (H1 ✓) |
| `eval_1k_D.json.tei.tei_rate` lower than C's, OR `vlmbias.recognition_aggregate.accuracy` higher | Delta adds value beyond filtering (H3 ✓) |
| All four `mathvista.accuracy` collapse | Filtering too aggressive — reduce β |
| All four ≈ E0 baseline (acc=0.226, TEI=0.620) | 1K too small / method has no signal / eval not sensitive |

If 1K shows zero direction, 8K probably won't either — diagnose before scaling up.

**Step 4 — 8K full sweep (only after 1-3 pass)**

Then run the originally-planned full pipeline: COCO/TallyQA download → synth → mixture → dedup → 32B precompute on 8K → parquet → 4-config sweep → full eval. Commands at the bottom of this section.

### Day-3 prerequisites (any order)

- **Commit Day-2 code** (10 files; see § "How to resume").
- **Server**: COCO train2017 + annotations (~18 GB) + TallyQA train.json + image dir.
- **Server**: `pip install --no-deps imagehash open_clip_torch` (dedup deps; lazy-imported but needed at run time).
- **Local**: copy E0 32B-teacher VLMBias jsonls from server → Mac (`experiments/E0_image_null_delta/results/e0_teacher32b_vlmbias_shard_*.jsonl`) so `eval_tei.py` unit tests have a real T_wrong fixture.

### Step-4 command reference (run only after Step 3 passes)

```bash
# Synth build (~5 min, no GPU)
python -m experiments.E1_filtered_delta_opd.data.synthesize_counterfactuals \
    --out-dir $DATASETS/e1_synth_v1 --n-samples 1500 --seed 42

# Mixture manifest (~1-2 min, no GPU)
python -m experiments.E1_filtered_delta_opd.data.mixture \
    --output $DATASETS/e1_mini_v1/mixture.jsonl \
    --virl39k-root $DATASETS/ViRL39K --coco-train-root $DATASETS/coco \
    --pope-adv-root $DATASETS/POPE-adversarial \
    --tallyqa-json $DATASETS/tallyqa/train.json \
    --tallyqa-images-root $DATASETS/tallyqa_images \
    --synth-dir $DATASETS/e1_synth_v1

# Dedup (MUST PASS, ~10-30 min, 1 GPU)
python -m experiments.E1_filtered_delta_opd.data.dedup_check \
    --pope-style-manifest '{...}' --tallyqa-manifest '{...}' \
    --synth-dir $DATASETS/e1_synth_v1 \
    --pope-adv-root $DATASETS/POPE-adversarial \
    --vlmbias-root $DATASETS/VLMBias \
    --output /tmp/dedup_findings.jsonl

# 32B teacher precompute on 8K mixture (~1h on 8 H800)
for SHARD in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$SHARD \
  python -m experiments.E1_filtered_delta_opd.src.precompute_teacher \
    --bucket mixture \
    --loader-kwargs '{"manifest_path":"'$DATASETS'/e1_mini_v1/mixture.jsonl"}' \
    --model-path $MODELS/Qwen2.5-VL-32B-Instruct \
    --output $RESULTS/e1_precompute_shard_${SHARD}.jsonl \
    --shard-index $SHARD --num-shards 8 &
done; wait

# Build parquet (~min)
python -m experiments.E1_filtered_delta_opd.data.make_train_parquet \
    --jsonl $RESULTS/e1_precompute_shard_*.jsonl \
    --student-tokenizer $MODELS/Qwen2.5-VL-7B-Instruct \
    --output $RESULTS/e1_mini_v1_train.parquet

# 4-config full sweep
for L in A B C D; do
  E1_TRAIN_PARQUET=$RESULTS/e1_mini_v1_train.parquet \
  E1_VAL_PARQUET=$RESULTS/e1_mini_v1_val.parquet \
  bash scripts/run_e1_recipe_smoke.sh $L
done
```

### Optional cleanups (don't block Day 3, pay later)

- **Long-term R3 fix**: chase the `LD_LIBRARY_PATH` leak from NGC system torch (`/usr/local/lib/python3.12/dist-packages/torch/lib`). Filter in `activate.sh`. Until then, `TORCH_CUDNN_V8_API_DISABLED=1` is baked into the launcher.
- **Long-term R4 fix**: patch `verl/utils/dataset/rl_dataset.py:_build_messages` to not mutate the input image dict in place. Currently sidestepped via `filter_overlong_prompts: false`.
- **Teardown noise** at end of smoke run (vLLM EngineCore / resource_tracker errors after checkpoint write). Exit-0, not blocking.
- **Animal silhouette upgrade**: if Step 2 confirms PIL silhouettes don't trigger canonical prior, replace with CLIP-guided / diffusion synth in v2.

---

## E0.x — pending diagnostic cleanup

| Task | Status | Cost | Why |
|---|---|---|---|
| **E0.3-A: per-topic m5a + chance-normalized overlap** | ✅ **DONE** (`c9615363`; table in `e0_verdict.md` § Student/teacher overlap) | — | Global 72.9% same-wrong was a mix of mechanical binary-task ceiling and real shared prior. Broken out per topic with `excess_over_chance`. |
| **E0.3-B: Length-normalized `gain_margin`** | ✅ **DONE** (`70aac08a`) | ~30 LoC + `--tokenizer-path` Qwen2.5-VL-7B-Instruct. | Eliminated length bias. Result: per-topic direction stable, no sign flips. Multi-token-option topics' magnitudes shrunk 20-31%; same-length-option topics unchanged. E1 motivation intact. |
| **E0.3-C: PPL_S(teacher_wrong_response \| x, I)** | Deferred, **does not block E1** | New ~15 min server run; new script `e0_ppl_student.py`. | Distribution-level overlap proxy. More OPD-faithful than answer-overlap. Useful as an E0 baseline before E1 trained students. Schedule when GPU is free. |
| **72B teacher sanity** | Deferred | `bash experiments/E0_image_null_delta/scripts/run_e0_teacher72b_sanity.sh`; ~20 min on 2 H800. | Tells us if the failure modes scale away with larger teacher (probably no) or are property of architecture (probably yes). |
| **Per-topic top tokens validation** | Deferred | Hand-inspect `top_delta_tokens.json["vlmbias_by_topic"]`; Logos has anomalous `" on"` token, decide if signal or noise. | Confirms metric 4 quality across topics, not just globally. |

---

## E1 — Filtered Delta-OPD training (the main work going forward)

**Read `experiments/E1_filtered_delta_opd/README.md` for the design.** That's the source of truth for the 4-config ablation, training-data composition, loss form, evaluation matrix, and engineering punch list. This section is the *workflow* on top of it.

### 4-config matrix (recap from on_policy_v1_design.md)

On-policy. Student rolls out at student's prefix `s_t`; teacher dual-scores
(image + null) on the rollout to provide `top-K logp` + `delta_t`.

| Config | What | Role |
|---|---|---|
| A. `VanillaKD` | `Σ_t KL_topK(P_T^I(.|s_t) \|\| P_S(.|s_t))`, no filter, no delta | Existing-OPD baseline |
| B. `RawDeltaKD` | `Σ_t w_t · KL_topK`, `w_t = clip(delta_t, p95)/mean`, no filter | **Negative control** — tests "image influence alone is insufficient" |
| C. `FilteredKD` | KL branch on `T_correct=1`; CE-on-gold on `T_correct=0` | **Critical control** — isolates filtering+CE contribution |
| D. `FilteredDeltaKD` | KL × delta on `T_correct=1`; CE-on-gold on `T_correct=0` | **Primary candidate** |

Central comparisons: **B vs A** (does raw delta hurt?), **D vs C** (does
delta help given filtering?), **C vs A** (how much is just filtering+CE?).

Compute-budget fallback: drop B first, keep A+C+D. **Do NOT drop C** —
without it D's gains are unattributable.

SFT (`L = −log p_S(y_T | x, I)`) deferred to optional E1.5.

### Day-by-day (updated 2026-05-13 — post Day 2)

| Day | Status | Goal |
|---|---|---|
| **Day 1 (2026-05-12)** | ✅ done | E0.3-B; on-policy pivot; ViRL39K verified + loader; precompute_teacher.py (smoke); losses.py (smoke-baseline only); 3 spikes (verl FSDP, verl on-policy, vLLM dual-forward); env Q5 runbook; on_policy_v1_design.md. |
| **Day 1.5 (2026-05-12 → 13)** | ✅ done | Stage 2 trainer code: agent_loop + 4 losses + parquet builder + launcher. All 4 configs (A/B/C/D) verified end-to-end on 50 ViRL39K. 5 integration traps surfaced + fixed (`docs/e1_smoke_runbook.md`). |
| **Day 2 (2026-05-13)** | ✅ code done | Data pipeline: POPE-style / TallyQA / synth / dedup / mixture / multi-bucket parquet builder + multi-bucket precompute. 35 local unit tests pass. Awaiting commit + server data download. |
| **Day 3** | NEXT | Commit Day 2 + download COCO/TallyQA → synth build → mixture → dedup gate → 32B teacher precompute on 8K (~1h shard-parallel) → parquet → B/C/D smoke on real mixture. |
| Day 4 | pending | Full 4-config sweep on 8K E1-mini, on-policy. |
| Day 5 | pending | First eval (TEI / Escape / per-topic gain_margin); decide v2 hyperparameters. |

### Evaluation harness (must exist before launching training)

These all run on the trained student against the **frozen** evaluation sets:

- VLMBias (full 2780 main, but eval per-topic separately — split Optical Illusion vs Recognition Aggregate)
- POPE-adversarial (full 1000)
- MathVista testmini (full 500) — retention check, must not drop > 1pp
- (optional) MMMU-mini — general retention

Plus the **TEI metrics on a frozen E0 teacher-wrong subset**:
- `Acc_S | T_wrong` — higher is better
- TEI rate = `P(S_after = T_wrong_answer | T_wrong)` — **lower is better** (THE key safety metric)
- Escape rate = `P(S_after = GT | T_wrong AND S_base = T_wrong_answer)` — higher is better

Length-normalized student gain_margin should also be reported per VLMBias topic.

### POPE-specific evaluation gotcha

Delta-weighting might amplify object-token attention, which could raise `yes`-rate and inflate hallucination. Report:
- F1, accuracy
- yes-rate
- grounded yes rate (gold=yes when answered yes)
- hallucinated yes rate (gold=no when answered yes)
- mean_delta on grounded vs hallucinated (continuity with E0 metric 5b)

---

## Open questions — all resolved through Day 1.5

| # | Question | Status |
|---|---|---|
| 1 | verl recipe entry point | ✅ Resolved. **verl `trainer/distillation/` FSDP** + `utils/dataset/rl_dataset.py` for multimodal + subclass `verl/experimental/agent_loop/SingleTurnAgentLoop` for dual-forward. NOT `recipe/gkd/` (Megatron-only). |
| 2 | Online teacher forward vs precomputed | ✅ Resolved. **Online** for per-token `teacher_logp` + `delta_t` (on student rollout prefix, can't precompute). **Precomputed** for sample-level `trajectory_pass` + `gold_token_ids` (precompute_teacher.py output). |
| 3 | ViRL39K starter subsample size | ✅ 8K E1-mini. Stratified by category from the 11,847 PassRate-filtered single-image rows. |
| 4 | Adversarial-recognition counterfactual source | ✅ Resolved post-GPT-review: synthesis-primary (~1500 VLMBias-like) + TallyQA complex (~900). NOT VLMBias `withtitle`/`remove_background` (image leakage with `main` eval). |
| 5 | `β` (CE weight on gold for filtered configs) | ✅ Resolved (GPT review): start at 0.1, sweep 0.05–0.3 in v2. CE acts as "answer anchor", not main signal. Baked into `e1.beta=0.1` in `e1_base.yaml`. |
| 6 | Top-K for KL in training | ✅ 50 (matches E0). vLLM teacher serving needs `max_logprobs=50` in engine config (spike learning). Baked into `e1_base.yaml`. |
| 7 | CE-on-gold dispatch implementation | ✅ Resolved per `on_policy_v1_design.md § 6` + GPT review: per-sample branch at the agent-loop level (`DeltaOPDAgentLoop.run()` skips vLLM rollout on CE samples, sets `response_ids=gold_token_ids` directly). Loss layer dispatches via `data["loss_branch"]`. Smoke run only exercised A; B/C/D paths are code-tested but not yet GPU-tested. |

---

## Hard rules — **DO NOT**

1. **Don't fall back to off-policy weighted CE as the E1 main experiment.**
   It's mathematically defensible as a smoke baseline only. The actual E1
   experiment uses on-policy student rollouts + dual teacher scoring — that
   is what "OPD" means. The `e1_offline_weighted_sft_*` losses are smoke;
   never report them as E1 results.
2. **Don't drop `FilteredKD` (Config C) from the 4-config sweep.** Without
   the filtering-only control, any gain in D is unattributable to delta.
   If compute-constrained, drop `RawDeltaKD` (B) instead.
3. **Don't generate teacher data on the VLMBias eval set.** That's
   evaluation, not training. Use held-out adversarial-recognition data only.
4. **Don't use VLMBias `withtitle` / `remove_background_*` subsets for
   training** — they share base images with `main` (the eval set). Image
   leakage.
5. **Don't change the null mode** (still only `black`) until E2 ablation.
6. **Don't `pip install -e` anything without `--no-deps`** on the server.
   Rebuilding TransformerEngine costs 30–40 minutes.
7. **Don't touch `.venv` symlink** on the server.
8. **Don't share the server** with other tenants during E1 training.
9. **Don't trust same-wrong overlap rate as a primary signal** — use TEI
   metric family.
10. **Don't `pip install -U triton` without re-applying the Q5 sed patch
    afterward** on HPC-X machines (see `docs/migrate-env.md § Q5`).
    `activate.sh` warns at startup if the patch was overwritten.

---

## Pending coordination items

- Confirm with operations team / `gemma-4-31B vllm serve` owner: when does that long-running serve come down? (Currently 19+ days running on the same box.)
- Confirm `Qwen3-VL-8B-Instruct` download (PID 164459 in earlier ps output) — is someone going to use it on the same box?
- If E1 training needs dedicated GPUs, schedule a window.

---

## How to resume a fresh Claude session

1. Open this repo.
2. Read in order: `CLAUDE.md` → `PROGRESS.md` → `NEXT.md`.
3. Skim `experiments/E0_image_null_delta/results/e0_verdict.md` for the canonical E0 numbers.
4. Open `experiments/E1_filtered_delta_opd/README.md` (E1 design) and `experiments/E1_filtered_delta_opd/on_policy_v1_design.md` (Stage 2 trainer).
5. If touching anything in `src/on_policy/`, scan `docs/e1_smoke_runbook.md` first — R1-R5 are the non-obvious gotchas that the Config A smoke uncovered.
6. Check `git log --oneline -10` to confirm where we left off (current HEAD: `5d4b5826` as of this writing; Day 2 code is in the worktree but not yet committed — see `git status` for the 8 changed files).
7. Pick up at the next un-checked item in **§ Right now, Day 3** above.
