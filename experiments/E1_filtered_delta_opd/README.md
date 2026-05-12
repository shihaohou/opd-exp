# E1 — Filtered Delta-OPD (design doc, not yet executable)

**Status**: design doc / starting point for tomorrow.
**Goal**: Train a 7B Qwen2.5-VL student from a 32B Qwen2.5-VL teacher using
Filtered Delta-OPD, with a 4-config ablation that establishes whether the
delta signal — when conditioned on teacher correctness — actually reduces
teacher-error inheritance vs vanilla OPD.

E1 is gated on E0's Conditional GO verdict (see `../E0_image_null_delta/results/e0_verdict.md`).

## Hypothesis

> Per-token image-vs-null KL on the teacher (`delta_t`) tracks image
> influence, but **direction of that influence is task-dependent** — it can
> be wrong-direction on adversarial recognition tasks. Filtering the
> distillation signal to teacher-correct trajectories should let the
> student learn the *useful* part of the visual reasoning signal without
> inheriting the prior-locked errors that E0 quantified.

## Configs (4 ablations)

| Key | Recipe | Purpose |
|---|---|---|
| A. `vanilla_opd` | Student rollout + teacher reverse-KL on full prefix, no filtering, no delta | Reproduces existing multimodal OPD recipe. Establishes the bias-propagation rate we're trying to beat. |
| B. `raw_delta_opd` | `L = Σ_t delta_t · KL(p_T(.\|.) \|\| p_S(.\|.))`, no filtering | **Negative control.** Tests "image influence alone is insufficient." Expected to amplify wrong-direction influence on VLMBias recognition topics. |
| C. `correct_filtered_opd` | `L = 1[T_correct] · Σ_t KL(p_T \|\| p_S)`, no delta weighting; CE on gold for teacher-wrong | **Critical control.** Isolates the contribution of teacher-correct filtering alone. Without this, any gain in D could be attributed to filtering rather than to delta weighting. |
| D. `correct_filtered_delta_opd` | `L = 1[T_correct] · Σ_t delta_t · KL(p_T \|\| p_S)`; CE on gold for teacher-wrong | **Primary candidate.** Filtering + delta. |

Compute-budget fallback: if 4 runs are too much, run A + C + D only (drop Raw Delta-OPD). Note: dropping C is **not** acceptable — without the filtering-only control, D's gains are unattributable.

**SFT (`L = -log p_S(y_T | x, I)` on teacher outputs)** is deferred to an optional E1.5 ablation. It answers a different question (does pure teacher imitation amplify pattern-matching errors?) and is not needed for the core method validation. Drop it from the v1 matrix unless compute is abundant.

## Training data composition (starting point, not protocol)

| Bucket | Share | Source | Why |
|---|---|---|---|
| Verifier-friendly general VL reasoning | ~50% | ViRL39K (TIGER-Lab/ViRL39K) subsample + LLaVA-CoT-100K (Xkev/LLaVA-CoT-100k) subsample | Where delta-signal-correctness correlation is healthy (E0 MathVista Spearman=+0.41). Most training mass here. |
| Object presence / hallucination | ~20% | POPE-style yes-no, balanced positive/negative | Keeps the model honest about object presence; E0 showed POPE is positive for the method. |
| Adversarial recognition counterfactuals | ~30% | Held-out VLMBias-style examples (NOT the eval set), TallyQA, custom counting/OCR | Where the failure mode lives. Forces exposure at train time. |

Total target: ~8K-20K samples for E1-mini. Full scale comes later.

## Loss specs (all 4 configs)

For a teacher-generated trajectory `y_T` on prompt `(x, I)`, let
`trajectory_pass = 1` if `parse_correctness(y_T, gold) == True` (or `verifier_pass(y_T) == True`), else `0`.

```
# Config A — Vanilla OPD
L_vanilla = Σ_t KL_topK(p_T(.|x,I,y_<t) || p_S(.|x,I,y_<t))

# Config B — Raw Delta-OPD
L_raw = Σ_t delta_t · KL_topK(p_T || p_S)

# Config C — Correct-filtered OPD  (no delta)
L_cf = trajectory_pass · Σ_t KL_topK(p_T || p_S)
     + (1 - trajectory_pass) · λ_ans · CE(answer_tokens, gold)

# Config D — Correct-filtered Delta-OPD  (primary candidate)
L_cf_delta = trajectory_pass · Σ_t delta_t · KL_topK(p_T || p_S)
           + (1 - trajectory_pass) · λ_ans · CE(answer_tokens, gold)
```

Open knobs (shared):
- `λ_ans` (CE weight on gold answer when teacher wrong): start at 1.0 — same scale as KL
- `delta_t` normalization: per-trajectory mean=1, or raw, or top-percentile clipping
- `K` for top-K KL: 50 (consistent with E0)
- Whether `trajectory_pass` is sample-level (current spec) or token-level (verifier on span granularity — future work)

The C-vs-D contrast is the experiment's central comparison: same filtering, same gold-CE on teacher-wrong, the **only** difference is whether KL is delta-weighted. That isolates the contribution of delta.

## Evaluation matrix

Primary metrics (must report all):

1. **VLMBias per-topic**: Optical Illusion vs Recognition Aggregate
   - acc_S
   - student-side gain_margin (length-normalized)
2. **Teacher-Error Inheritance (TEI) on E0's teacher-wrong subset (frozen!)**:
   - `Acc_S | T_wrong` (higher = better)
   - TEI rate = `P(S = T_wrong_answer | T_wrong)` (lower = better)
   - Escape rate = `P(S = GT | T_wrong AND S_base = T_wrong_answer)` (higher = better)
3. **POPE-adv**: F1, accuracy, yes-rate, grounded yes mean_delta, hallucinated yes mean_delta
4. **MathVista-mini**: accuracy (retention check, must not drop > 1pp)
5. **MMMU-mini** (optional but useful): general retention

Secondary (analysis):
- Loss / KL / response_length / token_entropy curves during training
- Top-delta token category distribution before vs after training

## Engineering punch list (not started)

1. **Decide verl recipe entry point.** Options:
   - `verl/recipe/gkd/` (general knowledge distillation) — closest match
   - `verl/trainer/distillation/` — has fsdp + megatron losses
   - Roll our own thin trainer wrapping `verl.trainer.sft_trainer`
2. **Implement Filtered Delta-OPD loss.** Requires:
   - per-batch teacher forward (image + null) to compute `delta_t`
   - per-batch trajectory correctness flag (cached from a precomputed pass)
   - reverse-KL computation on student student logits vs teacher logits
3. **Precompute teacher outputs + correctness flags** on training data
   (similar code path to E0's `dual_forward.py`).
4. **Multimodal data loader** for ViRL39K / LLaVA-CoT (HF arrow), with
   image preprocessing matching teacher / student processor.
5. **Eval hooks**: at every N steps, run a small eval pass on VLMBias
   subsample + POPE subsample, report TEI metrics.
6. **Length-normalized option scoring** (deferred from E0.x) for student-
   side gain_margin during eval.

## Risks / open questions

- Will the student converge fast enough on Correct-filtered Delta-OPD when
  most of the Animals topic has no teacher-correct trajectories to weight?
  (E0 finding: teacher acc on Animals = 0/546.)
  Probable mitigation: gold-CE on answer tokens for the non-filtered samples
  (the `λ_ans · CE` term is exactly for this).
- Are 32B-teacher generations on ViRL39K actually correct often enough to
  give Correct-filtered Delta-OPD useful training signal? Need to verify
  before committing to ViRL39K as the main bucket. Sample ~500 first.
- Memory: training 7B + holding 32B teacher in memory for online delta
  computation may not fit on a single H800. Options:
  - Precompute teacher logits / delta offline (cheap if we cache top-K only)
  - Run teacher on separate GPUs via vLLM serve, fetch logits over IPC
- The C-vs-D contrast may be small if delta_t is nearly uniform on filtered
  (teacher-correct) trajectories. Track `delta_t` variance on the filtered
  subset during training — if it collapses, the delta signal is doing little
  work and Raw Delta-OPD's "image-influence-alone" story loses force.

## Day-by-day plan (tomorrow start)

| Day | Goal |
|---|---|
| **Day 1** | Download ViRL39K + LLaVA-CoT; spike verl GKD recipe; pick entry point; write multimodal loader; implement teacher pre-generation. |
| **Day 2** | Implement Vanilla OPD trainer; smoke-test on 1K subset, 100 steps; verify checkpointing + eval hooks. |
| **Day 3** | Implement Raw Delta-OPD + Correct-filtered OPD + Correct-filtered Delta-OPD losses (the latter three share filtering/CE plumbing, build them together). |
| **Day 4** | Launch 4-config full run on 8K-sample E1-mini. |
| **Day 5** | First eval, decide v2 hyperparameters. |

## Files / structure (planned, not yet created)

```
experiments/E1_filtered_delta_opd/
├── README.md                   # this file
├── configs/
│   ├── e1_default.yaml         # base config (mirrors E0 spirit)
│   ├── recipe_vanilla_opd.yaml
│   ├── recipe_raw_delta_opd.yaml
│   ├── recipe_correct_filtered_opd.yaml
│   └── recipe_correct_filtered_delta_opd.yaml
├── data/
│   ├── virl39k_loader.py
│   ├── llava_cot_loader.py
│   └── mixture.py              # samples across buckets per the table above
├── src/
│   ├── precompute_teacher.py   # generates greedy responses + delta_t + correctness flags
│   ├── trainer.py              # the actual training loop (or thin wrapper over verl)
│   ├── losses.py               # delta-weighted reverse-KL + CE + filtering mask
│   └── eval_tei.py             # TEI / Escape / gain_margin on frozen teacher-wrong set
├── scripts/
│   ├── precompute_teacher_on_virl39k.sh
│   ├── run_e1_<recipe>.sh
│   └── eval_e1_all.sh
└── results/                    # gitignored
```

---

*This doc is a planning artifact. None of the files above exist yet. Touch nothing on the cluster tonight.*
