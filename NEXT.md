# NEXT — what to do next
*Last updated: 2026-05-12 (evening, end of Day 1). Pair this file with `PROGRESS.md`.*

> **For new Claude sessions:** if you're starting fresh, first read `CLAUDE.md`, then `PROGRESS.md` (what we did), then this file. Recommended next concrete action is at **§ Right now, Stage 2** below.

---

## Status overview

```
E0 (forward-only diagnostic)   : DONE — Conditional GO
E0.3-A (per-topic m5a)         : DONE
E0.3-B (length-norm gain)      : DONE — per-topic direction stable
E0.3-C (PPL_S smoke)           : deferred (does not block E1)
E1 design                      : DONE — locked on-policy v1 with 4 configs A/B/C/D
E1 Day 1                       : DONE — bucket-1 loader, smoke precompute, verl spike,
                                  vLLM dual-forward verified, runbook for env Q5
E1 Stage 2 (on-policy code)    : NEXT — agent_loop subclass + 4 losses + 4 yaml configs
E2 (mask sensitivity)          : not started
```

---

## Right now, Stage 2 (on-policy v1 trainer code)

Day 1 verified all preconditions. Stage 2 is **writing the on-policy training
loop**. Estimated 1 full day. **Best done in a fresh Claude session** (this
session has covered many topic shifts — fresh context for code work).

Read in order before writing code:
1. `CLAUDE.md` (top 6 reading list — includes on_policy_v1_design.md)
2. `experiments/E1_filtered_delta_opd/on_policy_v1_design.md` (the **canonical
   design** for what Stage 2 must build; see § "Engineering path")
3. `experiments/E1_filtered_delta_opd/README.md` § Configs / Loss specs

Concrete deliverables (in suggested order):

1. **`src/on_policy/agent_loop.py`** — subclass `verl.experimental.agent_loop.SingleTurnAgentLoop`,
   override `_compute_teacher_logprobs` to make TWO teacher calls (image and
   null, with `multi_modal_data["images"]` swapped) and compute per-token
   `delta_t = KL_topK(P_T^I || P_T^null)` locally. Reuse E0's
   `kl_topk_union` for the math. ~120 LoC.

2. **`src/on_policy/losses.py`** — register 4 on-policy losses with verl's
   `@register_distillation_loss`:
   - `e1_onpolicy_vanilla_kd` — `KL_topK(P_T^I || P_S)` (wraps verl's existing
     `compute_forward_kl_topk`)
   - `e1_onpolicy_raw_delta_kd` — same × `data["delta_t_normalized"]`
   - `e1_onpolicy_filtered_kd` — KL branch on `trajectory_pass=1`; CE-on-gold branch on `trajectory_pass=0`
   - `e1_onpolicy_filtered_delta_kd` — KL × delta on `trajectory_pass=1`; CE-on-gold on `trajectory_pass=0`

   delta normalization: `clip(delta_t, p95) / mean(clip)` per batch.
   CE-on-gold dispatch: per-sample branch based on a precompute-time-set
   `loss_branch` field in the data row ("kl" or "ce"). See
   on_policy_v1_design.md § 3 and § 6 for the full math.

3. **`src/on_policy/trainer.py`** — thin wrapper that registers our agent loop
   + losses, then delegates to verl's FSDP distillation trainer. Loads the
   precompute parquet + multimodal dataset via `verl.utils.dataset.rl_dataset.RLHFDataset`.

4. **`configs/recipe_*.yaml`** — 4 YAMLs (one per config). Each overrides:
   - `distillation.distillation_loss.loss_mode` → one of the 4 registered names
   - `distillation.distillation_loss.topk` → 50
   - `distillation.teacher_models.<key>.inference.max_logprobs` → 50 (see
     spike Q below)
   - `actor_rollout_ref.rollout.agent_loop_class` → our DeltaOPDAgentLoop

5. **`src/on_policy/smoke.sh`** — 1K-subset, 100-step smoke run; confirms
   rollout → dual teacher → loss → backward → checkpoint cycle.

### Stage 2 prerequisites already satisfied (don't re-spike)

- ✅ vLLM multimodal + `prompt_logprobs=K` on Qwen2.5-VL-32B verified
  (`spike_vllm_dual_forward.py` passed; position-50 in K12-1000-0 showed
  clean delta signal: image="geometric/triangle" vs null="completely/solid/plain")
- ✅ verl FSDP distillation has 80% of what we need; subclass agent_loop, no
  verl source edits required (Spike B)
- ✅ Triton ldconfig env bug fixed and documented (`docs/migrate-env.md § Q5`);
  activate.sh detects on future machines
- ✅ ViRL39K bucket-1 loader DONE and smoke-tested (11,847 eligible rows)
- ✅ E1 design pivoted to on-policy after external review; 4-config matrix
  locked as A/B/C/D (VanillaKD / RawDeltaKD / FilteredKD / FilteredDeltaKD)

### Stage 2 known-issue heads-up

- **`max_logprobs=K` engine arg**: the spike learned that vLLM caps
  `prompt_logprobs` at 20 by default. Teacher serving config in Stage 2 must
  also set `max_logprobs=50` (or whatever K we use). Bake into `recipe_*.yaml`.
- **Triton patch is venv-local** and gets overwritten by triton reinstalls.
  If `pip install -U triton` runs in Stage 2, re-apply the Q5 sed.

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

### Day-by-day (updated 2026-05-12 evening)

| Day | Status | Goal |
|---|---|---|
| **Day 1 (2026-05-12)** | ✅ done | E0.3-B; on-policy pivot; ViRL39K verified + loader; precompute_teacher.py (smoke); losses.py (smoke-baseline only); 3 spikes (verl FSDP, verl on-policy, vLLM dual-forward); env Q5 runbook; on_policy_v1_design.md. |
| **Day 1.5 (next)** | pending | Stage 2 of on_policy_v1_design.md — agent_loop subclass + losses + 4 yaml configs. Smoke-test on 1K. |
| Day 2 | pending | Build bucket 2 (POPE-style on COCO train) + bucket 3 (synthesis + TallyQA). Dedup pipeline. Freeze 8K E1-mini mixture. |
| Day 3 | pending | `make_train_parquet.py`; run precompute on the 8K mixture (sample-level trajectory_pass + gold tokenize only — drop the per-token forced-score path; v1 doesn't use it). |
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

## Open questions — most resolved 2026-05-12; one remaining for Stage 2

| # | Question | Status |
|---|---|---|
| 1 | verl recipe entry point | ✅ Resolved. **verl `trainer/distillation/` FSDP** + `utils/dataset/rl_dataset.py` for multimodal + subclass `verl/experimental/agent_loop/SingleTurnAgentLoop` for dual-forward. NOT `recipe/gkd/` (Megatron-only). |
| 2 | Online teacher forward vs precomputed | ✅ Resolved. **Online** for per-token `teacher_logp` + `delta_t` (on student rollout prefix, can't precompute). **Precomputed** for sample-level `trajectory_pass` + `gold_token_ids` (precompute_teacher.py output). |
| 3 | ViRL39K starter subsample size | ✅ 8K E1-mini. Stratified by category from the 11,847 PassRate-filtered single-image rows. |
| 4 | Adversarial-recognition counterfactual source | ✅ Resolved post-GPT-review: synthesis-primary (~1500 VLMBias-like) + TallyQA complex (~900). NOT VLMBias `withtitle`/`remove_background` (image leakage with `main` eval). |
| 5 | `β` (CE weight on gold for filtered configs) | Start at 1.0. Sweep in v2. |
| 6 | Top-K for KL in training | 50 (matches E0). vLLM teacher serving needs `max_logprobs=50` in engine config (spike learning). |
| **7** | **CE-on-gold dispatch implementation** | **Open for Stage 2.** Trainer-level sub-batch split (route by `loss_branch`) vs in-loss per-sample branching. Per `on_policy_v1_design.md § 6`, precompute pre-bakes `response_token_ids` per sample (teacher's response for KL samples, gold tokens for CE samples) so the loss is just `if data["loss_branch"] == "ce": NLL; else: KL_topK * weights`. |

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
3. Skim `experiments/E0_image_null_delta/results/e0_verdict.md` for the canonical numbers.
4. Open `experiments/E1_filtered_delta_opd/README.md` for the E1 design.
5. Check `git log --oneline -10` to confirm where we left off (current HEAD: `c9615363` as of this writing).
6. Pick up at the next un-checked item in **§ Right now, today** above.
