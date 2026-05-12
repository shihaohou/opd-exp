# E1 — Filtered Delta-OPD

> **E1 Protocol — read this first.**
> *Locked after external review on 2026-05-13. This box is the source of truth for what E1 is for; if any concrete plan elsewhere in this doc contradicts the box, the box wins.*
>
> **Goal.** E1 is the first **training-side causal experiment** for Delta-OPD. E0 showed `delta_t` tracks image influence, but image influence can be wrong-direction on VLMBias adversarial-recognition topics. E1 tests three things:
> 1. Does vanilla on-policy KD inherit teacher wrong patterns?
> 2. Does raw delta weighting *amplify* wrong-direction image influence?
> 3. Does correctness-filtered Delta-OPD *reduce* teacher-error inheritance without hurting POPE / MathVista?
>
> **E1 is not a score-chasing experiment.** It is a causal test of the inheritance mechanism. SOTA on any single benchmark is out of scope for E1-mini.
>
> **Models (fixed for E1-mini)**: Student = Qwen2.5-VL-7B-Instruct, Teacher = Qwen2.5-VL-32B-Instruct. Do **not** add 72B in E1-mini — variable explosion.
>
> **Data (8K E1-mini; ratios fixed for v1)**:
> - 4K ViRL39K (PassRate∈[0.3, 0.9], single-image, stratified by category)
> - 1.6K POPE-style on COCO train2017 (yes:no=1:1, POPE-adv eval-id disjoint)
> - 2.4K adversarial recognition (1.5K synthetic counterfactuals + 0.9K TallyQA complex)
>
> **Configs (4 on-policy ablations)**:
> - **A** `VanillaKD` — all samples, w_t = 1
> - **B** `RawDeltaKD` — all samples, w_t = normalized delta_t (negative control)
> - **C** `FilteredKD` — KL on T_correct; β·CE-on-gold otherwise (filtering control)
> - **D** `FilteredDeltaKD` — KL × delta_t on T_correct; β·CE-on-gold otherwise (primary candidate)
>
> **Primary comparisons (the only ones that matter for the causal story)**:
> - **B vs A** — does raw delta amplify wrong-direction image influence?
> - **C vs A** — does filtering + CE reduce teacher-error inheritance?
> - **D vs C** — does delta weighting add value *beyond* filtering + CE?
> - **D vs B** — does filtering block wrong-direction delta?
>
> Compute fallback: drop B first. **Do NOT drop C** — without C, D's gain is not attributable to delta.
>
> **Primary metrics (ordered, all output by `src/eval_tei.py`)**:
> 1. VLMBias Recognition Aggregate accuracy (Animals + Chess Pieces + Flags + Logos + Game Boards + Patterned Grid)
> 2. TEI rate = `P(S_after = T_wrong_answer | T_wrong)` — **lower is better**
> 3. Escape rate = `P(S_after = GT | T_wrong AND S_base = T_wrong_answer)` — **higher is better**
> 4. Student-side length-normalized `gain_margin` on VLMBias recognition topics
>
> **Safety metrics**:
> - POPE-adv: accuracy, F1, yes-rate, hallucinated-yes rate (Delta-OPD must not amplify object hallucination)
> - MathVista-mini: accuracy, response length (retention; must not drop > 1pp)
>
> **Outcome interpretation tree** (so we know what each result *means* before we look at it):
> - **Ideal** — `B < A` on recognition; `C > A` on TEI; `D > C` on recognition or gain_margin; POPE / MathVista unchanged. → Full Delta-OPD story holds.
> - **Acceptable** — `B ≈ A`; `C > A`; `D ≈ C`; POPE / MathVista unchanged. → Method gain comes from filtering + CE; rename to "Correctness-filtered on-policy distillation".
> - **Danger (`D > A` but `D ≈ C`)** — CE-on-gold is doing the work, not delta. Re-examine `kl_ce_ratio` per bucket; if CE dominates, increase β sweep or reduce CE weight in v2.
> - **Uninformative** — A/B/C/D all similar, or all hurt POPE/MathVista. → Method does not move the needle at 8K; either scale up or pivot to claim-gated OPD.
>
> **Day-3 ordering (eval-first)** — see `NEXT.md` § "Right now, Day 3": (1) implement `src/eval_tei.py`, (2) Bucket-3 teacher sanity, (3) 1K mini-sweep + eval, (4) 8K full sweep. Do NOT precompute the 8K before `eval_tei.py` exists — otherwise the run produces only ordinary accuracy and the causal questions stay unanswered.
>
> **Hard rule**: `e1_offline_weighted_sft_*` (off-policy weighted SFT in `src/losses.py`) is a pipeline smoke baseline only. **Never report it as an E1 scientific result.**

---

**Status**: code complete through Day 1.5 (on-policy trainer + 4 configs verified on 50 ViRL39K) and Day 2 (data pipeline, locally unit-tested). Day 3 (eval_tei + Bucket-3 sanity + 1K mini-sweep) is the next milestone.

**Goal**: Train a 7B Qwen2.5-VL student from a 32B Qwen2.5-VL teacher using
Filtered Delta-OPD, with a 4-config ablation that establishes whether the
delta signal — when conditioned on teacher correctness — actually reduces
teacher-error inheritance vs vanilla OPD.

E1 is gated on E0's Conditional GO verdict (see `../E0_image_null_delta/results/e0_verdict.md`).

## Method: **on-policy** distillation

Delta-OPD = Delta **On-Policy Distillation**. The student rolls out its own
responses; the teacher provides per-token logprobs (and top-K distributions
under both image and null conditions) **on the student's rollout tokens**.
This is the same data-distribution discipline as the
[Thinking Machines On-Policy Distillation blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
and verl's
[Async On-Policy KD recipe](https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html).

What this means concretely for E1:

- **Trajectories at training time come from the student**, not from teacher
  greedy. Teacher's pre-generated response is used only for the sample-level
  `trajectory_pass` flag and for the CE-on-gold branch of filtered configs.
- **`delta_t` is computed at training time on the student's rollout prefix**,
  via a dual teacher forced-score (image-conditioned + null-conditioned).
  Per-completion cost: 1 student rollout + 2 teacher forced-scores (≈ a few
  GPU-seconds on Qwen2.5-VL-32B). NOT per-token.
- **Loss form chosen**: top-K sparse KL `KL_topK(P_T^I(.|s_t) || P_S(.|s_t))`
  on the teacher's top-K support at each student-visited prefix s_t. This
  directly back-props (no PPO / importance-sampling) and reuses the top-K
  data we already need for delta_t.

A separate off-policy "weighted-SFT" baseline (`src/losses.py`) exists as a
100-300 step **engineering smoke test** to validate the data / monitoring /
eval pipeline before the on-policy v1 trainer is wired up. **Smoke results
are NOT E1's scientific results** — see § "Off-policy smoke baseline" below.

## Hypothesis

> Per-token image-vs-null KL on the teacher (`delta_t`) tracks image
> influence, but **direction of that influence is task-dependent** — it can
> be wrong-direction on adversarial recognition tasks. Filtering the
> distillation signal to teacher-correct trajectories should let the
> student learn the *useful* part of the visual reasoning signal without
> inheriting the prior-locked errors that E0 quantified.

## Configs (4 on-policy ablations)

All four use student rollouts as the trajectory source. The loss is top-K
sparse KL on teacher's top-K support at each student-visited position.
`w_t` is the per-token weight (1 or normalized delta_t). `1[T_correct]`
is the per-sample mask (1 if the precomputed teacher greedy got the gold,
else 0). For configs C and D, samples with `1[T_correct] = 0` go through
a CE-on-gold branch instead of the KL branch.

| Key | Recipe | Role |
|---|---|---|
| A. `VanillaKD` | All samples; `w_t = 1`; no filter. `L = Σ_t KL_topK(P_T^I(.|s_t) \|\| P_S(.|s_t))` | Existing-OPD baseline. Reproduces vanilla on-policy KD, what we want to beat. |
| B. `RawDeltaKD` | All samples; `w_t = delta_t`; no filter. `L = Σ_t delta_t · KL_topK(P_T^I \|\| P_S)` | **Negative control.** Tests "image influence alone is insufficient." Expected to amplify wrong-direction influence on VLMBias recognition topics. |
| C. `FilteredKD` | If `T_correct`: `L = Σ_t KL_topK(P_T^I \|\| P_S)`. Else: `L = β · CE(gold)`. No delta weighting. | **Critical control.** Isolates the contribution of teacher-correct filtering + CE-on-gold from delta weighting. Without C in the table, any gain in D could be attributed to filtering alone. |
| D. `FilteredDeltaKD` | If `T_correct`: `L = Σ_t delta_t · KL_topK(P_T^I \|\| P_S)`. Else: `L = β · CE(gold)`. | **Primary candidate.** Filtering + delta. |

The central comparisons are:

- **B vs A** — does raw delta hurt? (Expected: yes, on VLMBias recognition.)
- **D vs C** — does delta help *given* filtering + CE? (The actual hypothesis.)
- **C vs A** — does filtering + CE alone help? (Confound to isolate.)

Compute-budget fallback: drop B if necessary. **Do not drop C** — without it
any D > A gain is unattributable to delta.

SFT (`L = -log p_S(y_T | x, I)` on teacher outputs) is deferred to optional
E1.5. It's a different question (does teacher imitation amplify pattern-
matching?) and not part of the core method validation.

### delta_t normalization (required)

E0 smoke test on ViRL39K confirmed `delta_t` is long-tailed:
`mean ≈ 0.26, median ≈ 0.001` over response tokens. Raw delta as
multiplicative weight would let ~1% of tokens dominate the gradient. v1
applies a stable clip-then-rescale:

```python
w_t = clip(delta_t, p95(delta_t over valid tokens))
w_t = w_t / mean(w_t over valid tokens)
```

So `mean(w_t) ≈ 1` per batch (vanilla setting becomes the natural reference).
An alternative form `w_t = 1 + α · normalize(delta_t)` may be swept in v2.

### Per-batch monitoring (mandatory, GPT-flagged)

For configs C and D, log each step:
- `e1/effective_kd_tokens` — # tokens contributing to KL branch (= `1[T_correct]` · response_len)
- `e1/effective_ce_samples` — # samples in CE-on-gold branch
- `e1/kd_loss` and `e1/ce_loss` separately
- `e1/kd_ce_ratio` = `kd_loss / (kd_loss + ce_loss)`

If `kd_ce_ratio` ≈ 0 (CE dominates) we cannot attribute D > C to delta —
this would be a recovery diagnostic, not a method result.

## Off-policy smoke baseline (NOT E1 results)

`src/losses.py` currently implements an **off-policy weighted SFT** form:
student is force-scored on the teacher's precomputed greedy response, with
per-token NLL weighted by delta_t and/or masked by trajectory_pass. The
registry names are `e1_offline_weighted_sft_{vanilla, raw_delta,
correct_filtered, correct_filtered_delta}` to avoid confusion with the
on-policy v1 losses (which will live in a separate file once written).

This is **for engineering smoke testing only** — 100-300 steps on a 1K
sample subset, to validate:
- ViRL39K loader + mixture sampling + dedup
- per-bucket monitoring hooks
- delta_t clip / normalize doesn't explode
- TEI eval pipeline reads checkpoints cleanly
- CE-on-gold vs KL loss magnitudes stay sane

It does NOT answer E1's central question (whether teacher-error inheritance
happens on student-visited states; whether filtered Delta-OPD blocks it on
student-visited states). Smoke results stay in `results/e1_smoke/`; the
on-policy v1 results go in `results/e1_v1/`.

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

## Loss specs (all 4 on-policy configs)

For each training sample, let:

- `s_t = (x, I, y_<t)` where `y_<t` are tokens of the **student's current rollout**
- `P_T^I(.|s_t)` = teacher's top-K distribution conditioned on the real image, at prefix `s_t`
- `P_T^null(.|s_t)` = teacher's top-K distribution conditioned on null image, at prefix `s_t`
- `P_S(.|s_t)` = student's distribution
- `delta_t = KL_topK(P_T^I(.|s_t) || P_T^null(.|s_t))` — image influence at student's prefix
- `w_t` = normalized weight (see § "delta_t normalization" above); = 1 for non-delta configs
- `1[T_correct]` = sample-level mask from precomputed teacher greedy correctness
- `β` = CE-on-gold weight (default 1.0)

```
# Config A — VanillaKD
L_A = Σ_t KL_topK(P_T^I(.|s_t) || P_S(.|s_t))

# Config B — RawDeltaKD  (negative control)
L_B = Σ_t w_t · KL_topK(P_T^I(.|s_t) || P_S(.|s_t))
      where w_t = normalize(clip(delta_t, p95))

# Config C — FilteredKD + CE-on-gold  (filtering control)
if 1[T_correct]:
    L_C = Σ_t KL_topK(P_T^I(.|s_t) || P_S(.|s_t))
else:
    L_C = β · CE(student | gold_tokens)
        (student is force-scored on gold; no rollout this sample)

# Config D — FilteredDeltaKD + CE-on-gold  (primary candidate)
if 1[T_correct]:
    L_D = Σ_t w_t · KL_topK(P_T^I(.|s_t) || P_S(.|s_t))
else:
    L_D = β · CE(student | gold_tokens)
```

Open knobs:
- `β` (CE weight on gold answer when teacher wrong): start at 1.0
- `K` for top-K KL: 50 (consistent with E0)
- `w_t` form: default `clip(delta_t, p95) / mean(clip)`; v2 may sweep `1 + α·normalize(delta_t)`

Central comparisons:
- **B vs A** — does raw delta hurt? (E0 predicts yes on VLMBias recognition)
- **D vs C** — does delta help *given* the same filtering + CE branch? (The actual hypothesis test)
- **C vs A** — how much of D > A is just filtering + CE? (Attribution control)

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

1. **Verl backbone — decided 2026-05-12, refined after GPT review.**
   - Use `verl/trainer/distillation/` FSDP path. NOT `verl/recipe/gkd/`
     (Megatron-only; Qwen2.5-VL Megatron conversion is multi-day for the
     ViT + projector + LLM composite).
   - Borrow GKD recipe's three-stage shape: **rollout → teacher scoring → update**.
   - Use `verl/utils/dataset/rl_dataset.py` for multimodal data plumbing.
   - Distillation losses registered via `@register_distillation_loss(...)`.
   - For on-policy v1: top-K sparse KL `KL_topK(P_T^I || P_S)` at student-
     visited prefixes (back-props directly, no PPO/IS).
2. **Data loaders** (one per bucket):
   - ✅ `data/virl39k_loader.py` — DONE. PassRate filter, `<image>` strip,
     `\boxed{}` extraction, single-image filter. 11,847 eligible rows.
   - ❌ `data/pope_style_builder.py` — TODO. Builds POPE-style yes/no from
     COCO train2017 + instance annotations; image-level disjoint with POPE-adv.
   - ❌ `data/synthesize_counterfactuals.py` — TODO. Parametric synthesis
     (animals leg count, flags stripes, grids, game boards, chess pieces).
   - ❌ `data/tallyqa_loader.py` — TODO. TallyQA `complex` subset with
     COCO image_id filter against POPE-adv.
   - ❌ `data/mixture.py` — TODO. 8K E1-mini sampler per locked recipe.
3. **`data/dedup_check.py`** — TODO. Three-layer dedup (image_id ∩ pHash ∩
   CLIP NN). MUST PASS before training launches.
4. **`src/precompute_teacher.py`** — ✅ DONE (off-policy form).
   - Currently outputs per-token `teacher_logp_I/null` + `delta_t` on
     teacher's response. These per-token fields are NOT used by on-policy v1.
   - **Useful for on-policy v1**: only the sample-level fields
     (`trajectory_pass`, `gold`, `ans_T_I`, `pass_rate_32b`, `category`).
   - **Useful for v0 smoke**: all fields.
   - When on-policy v1 is wired in, this script can be trimmed to skip the
     dual forced-score (5× faster) — see `on_policy_v1_design.md`.
5. **`src/losses.py`** — ✅ DONE (off-policy smoke baseline).
   - Registers `e1_offline_weighted_sft_{vanilla,raw_delta,correct_filtered,
     correct_filtered_delta}` for the 4 smoke configs.
   - SMOKE TEST ONLY — see § "Off-policy smoke baseline" above.
6. **`src/on_policy_trainer.py`** — TODO (the actual E1 v1 trainer).
   - Student rollout (vLLM)
   - Teacher dual forced-score (image + null) on student's rollout → top-K
     log-probs + indices + `delta_t` per position
   - Top-K sparse KL loss on teacher's top-K support
   - Per-sample branch: `1[T_correct]` → KL; else CE-on-gold (no rollout, force-score gold)
   - Per-bucket monitoring hooks (effective_kd_tokens, kd_ce_ratio, etc.)
   - **Prerequisite spike**: read `verl/workers/rollout/` + `verl/experimental/
     teacher_loop/` for the dual-forward teacher serving wiring.
7. **`src/on_policy_losses.py`** — TODO. The 4 on-policy KD losses:
   `e1_onpolicy_vanilla_kd`, `e1_onpolicy_raw_delta_kd`,
   `e1_onpolicy_filtered_kd`, `e1_onpolicy_filtered_delta_kd`. Top-K KL
   primitive; CE-on-gold branch dispatch via `data["loss_branch"]`.
8. **`src/eval_tei.py`** — TODO. TEI / Escape / length-normalized gain_margin
   on the frozen E0 teacher-wrong subset; per-topic VLMBias / POPE-adv /
   MathVista evals.
9. **Per-bucket training-time monitoring hooks** (in trainer):
   teacher_correct_rate, effective_kd_tokens, effective_ce_samples,
   kd_loss vs ce_loss, kd_ce_ratio, delta_t mean/var clipped — all logged
   per step per bucket.
10. **Length-normalized student-side option scoring** — port from E0
    `metrics.py:make_option_len_fn` into `eval_tei.py`.

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
