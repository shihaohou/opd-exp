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

Three-bucket mixture, ~8K samples for E1-mini, scaling later.

| Bucket | Share | Source decision | Status |
|---|---|---|---|
| Verifier-friendly general VL reasoning | ~50% (~4K) | **ViRL39K** subsample, filtered by `PassRate_32BTrained ∈ [0.3, 0.9]` and single-image | ✅ **Locked.** Schema verified 2026-05-12 (see `data/virl39k_loader.py`); 13.4K eligible rows after pre-filter, plenty for 4K. |
| Object presence / hallucination | ~20% (~1.6K) | Open — see § below | ❌ Open |
| Adversarial recognition counterfactuals | ~30% (~2.4K) | Open — see § below | ❌ Open |

### Why ViRL39K is the right choice for bucket 1

Verified properties (2026-05-12 schema probe on the downloaded copy):

- **38,870 verifiable QA pairs**, `\boxed{...}` answer format — trivial verifier (`re.search(r"\\boxed\{(.+?)\}")`)
- **`PassRate_32BTrained` is populated for 100% of rows** (no `-1` sentinels in this snapshot). Distribution:
  - `= 1.0` (always correct): 20,085 (51.7%) — too easy, Filtered ≡ Vanilla on these, no E1 signal
  - `[0.7, 1.0)`: 7,007 (18%)
  - `[0.3, 0.7)`: 6,428 (16.5%) — **sweet spot for Filtered Delta-OPD**
  - `(0, 0.3)`: 5,350 (13.8%) — too hard, our own 32B likely also fails; needs CE-on-gold fallback
- **8 categories**, all math / science / chart / spatial / commonsense reasoning. No hallucination data, no adversarial recognition data.
- **94.1% single-image** — v1 loader filters multi-image for simplicity
- The `PassRate_32BTrained` column is a *free* pre-filter for trajectory-correctness: we can drop the always-correct rows (no filtering benefit) and the always-wrong rows (no positive trajectories to delta-weight) without running our own 32B teacher generation across the full 38K. Saves a generation pass.

**Caveat**: `32BTrained` is VL-Rethinker's RL-trained 32B, not vanilla `Qwen2.5-VL-32B-Instruct`. The pass rates are *correlated* with our teacher's behavior but not identical. Use as a coarse pre-filter, not as the final trajectory_pass label — that still has to come from our own teacher generation.

### Locked data-source decisions (after external review 2026-05-12)

#### Bucket 2: Object presence / hallucination (~20%, ~1.6K samples)

**Purpose**: prevent regression on POPE-style honest-about-objects behavior. E0 metric 5b ✅ showed `grounded yes` mean_delta > `hallucinated yes` mean_delta on POPE-adv — this signal must survive training.

**Decision**: **Self-build POPE-style train set on COCO `train2017` (or `train2014`)**.
Do NOT use official POPE `random` / `popular` / `adversarial` HF splits — they are not separate image splits. All three are constructed on the same ~500 COCO `val` images per `img_num=500`, so even "random" and "popular" leak images into POPE-adv eval.

Builder spec (to live in `data/pope_style_builder.py`):
- Base images: COCO `train2017` (NOT `val`)
- Object annotations: COCO instance annotations
- Negative sampling: balanced mix of random / popular / co-occurring hard negatives (the same scheme POPE-adversarial uses, just on training images)
- Question template: `"Is there a {object} in the image?"` with `\boxed{Yes}` / `\boxed{No}` answer for verifier compatibility
- Yes/No ratio: 1:1
- **Mandatory image-level disjoint check** against POPE-adversarial eval set IDs (must be 0 overlap before commit)

Supplementary (only if 1.6K is hard to reach from COCO alone):
- **AMBER discriminative** (`Junfei2019/AMBER`) — covers existence / attribute / relation, image-level disjoint from MS-COCO POPE eval

Explicitly **NOT used** in v1:
- Official POPE `random` / `popular` splits — image overlap with POPE-adv
- RLHF-V — preference pair format, not Yes/No; format pipeline overhead

#### Bucket 3: Adversarial recognition counterfactuals (~30%, ~2.4K samples)

**Purpose**: force training-time exposure to the canonical-prior failure mode. E0 per-topic table: Animals / Chess / Flags / Logos / Game Boards / Patterned Grid all negative `gain_margin` — image triggers recognition, language prior routes to canonical-wrong.

**Decision**: **Synthesis-primary mix**.

```
~1500 synthetic VLMBias-like held-out counterfactuals  (PRIMARY)
~900  TallyQA complex/counting subset                  (counting backbone)
```

Builder spec (synthetic, to live in `data/synthesize_counterfactuals.py`):
- Targets: animals leg-count modified, flag stripe/star count modified, game board row/column modified, patterned-grid count, chess piece count
- Image generation: parametric / template-based (PIL composition for grids/flags/board; segmentation+edit for animals)
- **All base assets must be NEW** — generated with our own scripts, not pulled from VLMBias `main` images
- Answer format: `\boxed{N}` integer for counting; `\boxed{X}` for identification

TallyQA usage:
- Source: `manoja328/TallyQA_dataset` — 287K questions, 165K images from COCO + Visual Genome
- Subset: `complex` only (multi-step counting, closer to VLMBias Animals failure mode)
- **Mandatory COCO image_id filter**: exclude any image_id present in POPE-adversarial eval set
- Reformat answer to `\boxed{N}` for verifier compatibility

Explicitly **NOT used** in v1:
- **VLMBias `withtitle` / `remove_background_*` subsets** — README on `anvo25/vlms-are-biased` confirms these are `main`-image-with-modifications, NOT separate image splits. Using them for training while `main` is eval is image leakage. They can still be used as *analysis sets* during eval reporting.
- **SEED-Bench-2** — benchmark not training data, multiple-choice format adds verifier complexity, no clear counterfactual subset
- **HowManyQA / TDIUC** — defer; only add if TallyQA complex doesn't yield 900

Fallback if synthesis pipeline isn't ready by Day 1.5:
- 1500 TallyQA complex + 900 simple parametric synth (flags/grids only) — but flag this as a known weakness in E1 results: Animals failure mode underrepresented in training.

### Mandatory dedup pipeline (before training launch)

Per GPT review, three layers of overlap check are required:

1. **Filename / numeric image_id intersection check**
   - For all bucket 2 + bucket 3 training images: extract COCO numeric `image_id` (or VG `image_id`, or our synth `qid`)
   - Assert `train_image_ids ∩ pope_adv_eval_ids == ∅`
   - Assert `train_image_ids ∩ vlmbias_eval_image_ids == ∅`
   - (ViRL39K bucket 1 has its own image namespace, likely no overlap, but check anyway)

2. **Perceptual hash (pHash / dHash) near-duplicate detection**
   - Catches "same image, different filename" — relevant for VG images that may have been re-cropped, and for synth/COCO overlap edge cases
   - Threshold: Hamming distance < 5 on 64-bit pHash → flag as near-duplicate

3. **CLIP embedding nearest-neighbor (belt-and-suspenders)**
   - For each training image, find top-1 nearest eval image by CLIP cosine
   - Threshold: cos > 0.95 → flag for manual review

This goes in `data/dedup_check.py`. Must run + pass before any training launches.

### Per-bucket training monitoring (mandatory)

GPT raised a critical attribution risk: **if a bucket has high teacher-wrong rate, the CE-on-gold loss term will dominate, and any D-vs-C improvement could come from CE, not from delta weighting.**

For all 4 configs we must log per-step + per-bucket:

| Metric | Why |
|---|---|
| `bucket/{1,2,3}/teacher_correct_rate` | Sanity-check whether `trajectory_pass=1` rate matches expected (bucket 1 ~50-70% after PassRate filter; bucket 2 high; bucket 3 likely lower) |
| `bucket/{1,2,3}/n_effective_kl_tokens` | Number of tokens where KL loss is non-zero (teacher-correct AND delta_t > threshold) |
| `bucket/{1,2,3}/n_effective_ce_samples` | Number of samples where CE-on-gold is active (teacher-wrong, configs C+D only) |
| `bucket/{1,2,3}/kl_loss_contribution` | KL loss summed over bucket / total loss |
| `bucket/{1,2,3}/ce_loss_contribution` | CE loss summed over bucket / total loss |

The diagnostic question: in config D, does the gain over C come from changed KL gradients (delta-reweighted) or from a different CE balance? If `kl_loss_contribution` for D ≈ C and `ce_loss_contribution` for D ≈ C, then improvements are attributable to delta.

### Locked 8K E1-mini recipe

```
Bucket 1: 4000 ViRL39K subset
    filter: PassRate_32BTrained ∈ [0.3, 0.9], single_image_only=True, \boxed{}-parseable
    (~13.4K eligible rows; subsample to 4K stratified by category)

Bucket 2: 1600 self-built POPE-style on COCO train
    template: "Is there a {object} in the image?" → \boxed{Yes/No}
    yes:no = 1:1; negatives = mix of random / popular / co-occurring
    image-level disjoint with POPE-adv eval (asserted)

Bucket 3: 2400
    ~1500 synthetic VLMBias-like counterfactuals (Animals / Flags / Grids / Boards / Chess)
    ~900  TallyQA complex/counting (with COCO image_id filter against POPE eval)
```

Total: 8000 samples. Treat ratios as v1, sweep later.

### Remaining decision questions

1. Whether to use `PassRate_32BTrained` as the trajectory_pass label directly, or generate our own 32B greedy and parse. Probably **both**: use PassRate as pre-filter (cheap), use our own gen as the canonical training-time label (correct). Open: do we need our own gen for bucket 2 / 3 since their answers are simple yes/no or integers — the 32B teacher pass rate there is plausibly close to 1.
2. Synthesis pipeline complexity vs day budget. Building VLMBias-like generation is non-trivial — estimate 1 full day for a usable v1 with ~500 samples per failure mode.

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

## Engineering punch list

1. **Verl backbone — decided 2026-05-12.**
   - Use `verl/trainer/distillation/` FSDP path (NOT `verl/recipe/gkd/`).
     The recipe is Megatron-only, forward-KL-only, single-teacher-forward,
     text-only. Bad fit.
   - Use `verl/utils/dataset/rl_dataset.py` for multimodal data plumbing
     (already supports `image_key`, `<image>` placeholder, HF ProcessorMixin).
   - Register our 4 losses via `@register_distillation_loss(...)`.
   - Reverse-KL: use the existing `compute_distillation_loss_reverse_kl_estimator`
     (modes `k1` / `k2` / `k3`) as the per-token KL primitive.
   - Per-token weight (delta_t): extend the existing `response_mask * loss`
     multiplication to `response_mask * delta_t * trajectory_pass * loss`.
2. **Multimodal data loaders** (one per bucket):
   - ✅ `data/virl39k_loader.py` — DONE (2026-05-12). Includes PassRate filter,
     `<image>` strip, `\boxed{}` extraction, single-image filter.
   - ❌ `data/pope_style_builder.py` — TODO. Builds POPE-style Yes/No yes/no
     from COCO train2017 + instance annotations, with image-level disjoint
     against POPE-adv eval.
   - ❌ `data/synthesize_counterfactuals.py` — TODO. Parametric/template
     synthesis of VLMBias-like counterfactuals (animals leg count, flags
     stripes, grids, game boards, chess pieces).
   - ❌ `data/tallyqa_loader.py` — TODO. Filters TallyQA `complex` subset by
     COCO image_id against POPE-adv.
   - ❌ `data/mixture.py` — TODO. Samples across buckets per the locked recipe.
3. **`data/dedup_check.py`** — TODO. Three-layer dedup (image_id intersection
   + pHash near-duplicate + CLIP embedding NN). MUST PASS before training
   launches.
4. **`src/precompute_teacher.py`** — TODO. Reuses E0 `dual_forward.py` logic
   on the locked 8K mixture; dumps per-sample `(response, correctness, delta_t[],
   teacher_topk_logp[])` to disk. Output is what training loads.
5. **`src/losses.py`** — TODO. Registers `vanilla_opd` / `raw_delta_opd` /
   `correct_filtered_opd` / `correct_filtered_delta_opd` in verl's loss
   registry. Each is a combination of: base reverse-KL × delta_t weight ×
   trajectory_pass mask, plus optional CE-on-gold for teacher-wrong samples.
6. **`src/trainer.py`** — TODO. Thin wrapper assembling verl FSDP trainer +
   rl_dataset + our loss. Loads precomputed teacher cache.
7. **`src/eval_tei.py`** — TODO. TEI / Escape rate / length-normalized
   gain_margin on the frozen E0 teacher-wrong subset, plus per-topic VLMBias /
   POPE-adv / MathVista evals.
8. **Per-bucket training-time monitoring hooks** (in trainer):
   teacher_correct_rate, n_effective_kl_tokens, n_effective_ce_samples,
   kl_loss_contribution, ce_loss_contribution — all logged per step per bucket.
9. **Length-normalized student-side option scoring** for the gain_margin eval
   metric — port from E0 `metrics.py:make_option_len_fn`. Already merged in E0;
   just needs wiring into `eval_tei.py`.

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
- **CE-on-gold attribution risk** (flagged by external review 2026-05-12):
  in configs C and D, the `λ_ans · CE(answer_tokens, gold)` term activates on
  every teacher-wrong sample. If bucket 3 (adversarial recognition) has high
  teacher-wrong rate, CE contribution dominates and any D-over-C improvement
  could come from CE, not from delta weighting. **Mitigation**: log
  per-bucket KL vs CE loss contributions (see Engineering punch list #8). If
  in config D the KL contribution to total loss is near 0 (because most
  teacher-wrong samples take the CE branch), the delta result is uninformative
  and we need either a higher `λ_ans` knob study or a different filtering
  strategy.

## Day-by-day plan (updated 2026-05-12 after data-source decisions locked)

| Day | Goal |
|---|---|
| **Day 1 (in progress)** | ✅ ViRL39K downloaded + schema verified + `virl39k_loader.py` written. ✅ verl entry point decided (FSDP `trainer/distillation/` + `rl_dataset.py`, NOT `recipe/gkd/`). Next: smoke-test loader on server. |
| **Day 1.5 (new)** | Build POPE-style train set on COCO train2017 (`pope_style_builder.py`); set up TallyQA complex loader with COCO image_id filter (`tallyqa_loader.py`); start synthesis pipeline for VLMBias-like counterfactuals (`synthesize_counterfactuals.py`) — first ~500 samples. |
| **Day 2** | Finish synthesis pipeline (full ~1500); run mandatory dedup (`dedup_check.py`); freeze 8K E1-mini mixture (`mixture.py`); implement `precompute_teacher.py` (mirrors E0 `dual_forward.py` on the locked mixture). |
| **Day 3** | Run precompute teacher pass on the full 8K mixture (~3-5 hrs on 8×H800 expected). Implement `losses.py` (4 configs registered in verl loss registry). Implement `trainer.py` thin wrapper. Implement per-bucket monitoring hooks. |
| **Day 4** | Smoke-test Vanilla OPD on 1K subset, 100 steps; verify checkpointing + eval hooks + bucket monitoring. |
| **Day 5** | Launch 4-config full run on 8K E1-mini. |
| **Day 6** | First eval (TEI / Escape / gain_margin per topic), decide v2 hyperparameters. |

## Files / structure (planned, not yet created)

```
experiments/E1_filtered_delta_opd/
├── README.md                          # this file
├── configs/
│   ├── e1_default.yaml                # base config (mirrors E0 spirit)
│   ├── recipe_vanilla_opd.yaml
│   ├── recipe_raw_delta_opd.yaml
│   ├── recipe_correct_filtered_opd.yaml
│   └── recipe_correct_filtered_delta_opd.yaml
├── data/
│   ├── virl39k_loader.py              # ✅ DONE
│   ├── pope_style_builder.py          # TODO: COCO train2017 → POPE-style yes/no
│   ├── tallyqa_loader.py              # TODO: TallyQA complex + COCO image_id filter
│   ├── synthesize_counterfactuals.py  # TODO: parametric VLMBias-like generation
│   ├── mixture.py                     # TODO: samples across buckets per locked recipe
│   └── dedup_check.py                 # TODO: image_id ∩ pHash ∩ CLIP NN
├── src/
│   ├── precompute_teacher.py          # TODO: greedy gen + delta_t + correctness on mixture
│   ├── trainer.py                     # TODO: FSDP wrapper over verl/trainer/distillation
│   ├── losses.py                      # TODO: 4 losses via verl @register_distillation_loss
│   └── eval_tei.py                    # TODO: TEI / Escape / gain_margin
├── scripts/
│   ├── build_mixture.sh
│   ├── precompute_teacher.sh
│   ├── run_e1_<recipe>.sh
│   └── eval_e1_all.sh
└── results/                           # gitignored
```

---

*This doc is a planning artifact. None of the files above exist yet. Touch nothing on the cluster tonight.*
