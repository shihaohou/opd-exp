# E1 on-policy v1 — design

*Last updated: 2026-05-12 (afternoon, post external review).*

This document specifies the **on-policy** training loop for E1. It supersedes
the off-policy interpretation that was implicit in the early code drafts
(`src/precompute_teacher.py`, `src/losses.py`). Those files remain as a
**smoke-test baseline only**; v1 results require the loop specified here.

This is the source of truth for the on-policy v1 trainer to be built. The
E1 README references this document for the math; the implementation lives
in `src/on_policy_trainer.py` (TODO) and `src/on_policy_losses.py` (TODO).

---

## 1. Core training loop (per prompt)

```
INPUTS:
  x         — prompt text (question + choices, image placeholders stripped)
  I         — image (PIL)
  gold      — gold answer string
  T_correct — sample-level bool (from precompute: did teacher greedy match gold?)

LOOP:
  if T_correct:
      # ----- on-policy KL branch -----
      y_S ~ pi_S(. | x, I)                                       # student rollout via vLLM
      forced_score(teacher, x, I,    y_S)  -> logp_T_I[t], topK_I[t]    # teacher dual forward
      forced_score(teacher, x, null, y_S)  -> logp_T_null[t], topK_null[t]
      delta_t = KL_topK(P_T^I(.|s_t) || P_T^null(.|s_t))                # per position t
      w_t     = clip(delta_t, p95) / mean(clip)         # = 1 in vanilla configs
      L_per_token[t] = w_t * KL_topK(P_T^I(.|s_t) || P_S(.|s_t))        # see § 3
      L = sum_t L_per_token[t]
  else:
      # ----- CE-on-gold branch -----
      forced_score(student, x, I, gold_tokens) -> student_logp[gold_t]  # student forward only
      L = beta * sum_t (-student_logp[gold_t])                          # standard NLL on gold
```

Per-step gradient accumulation over a batch of prompts. Mixed-config
batches are allowed; each sample takes its own branch.

## 2. Forward / backward cost per sample

- **KL branch** (T_correct=1): 1 student rollout (vLLM, ~10K tok/sec on 7B)
  + 2 teacher forced-scores (image and null, ~1 sec each on 32B for ~200 tok
  responses). Backward through student. **Cost: ~2 sec teacher + 0.05 sec student rollout.**
- **CE branch** (T_correct=0): 1 student forced-score on gold (~0.1 sec on 7B
  for short gold answers). No teacher. Backward through student. **Cost: ~0.1 sec.**

For an 8K E1-mini run with ~50% trajectory_pass rate:
- 4K KL samples × 2 sec ≈ **2.2 hours teacher GPU time**
- 4K CE samples × 0.1 sec ≈ negligible
- Plus student backward (~30 min on 8 H800)
- **Total full epoch ≈ 3 hours**. 4 configs × few epochs ≈ 1-2 days for E1-mini sweep.

## 3. Loss form: top-K sparse KL `KL_topK(P_T^I || P_S)` on student prefix

The top-K KL formula at student-visited prefix `s_t`:

```
S       = top-K support from P_T^I (union with P_T's top-K is optional)
P_T(k)  = teacher I-conditioned prob on token k ∈ S (renormalized over S)
P_S(k)  = student prob on token k ∈ S (renormalized over S)

KL_topK(P_T || P_S) = Σ_{k ∈ S} P_T(k) * (log P_T(k) - log P_S(k))
```

This is directly differentiable w.r.t. student parameters (back-prop through
`log P_S(k)` for k in the top-K support). Teacher's top-K is detached.

Choosing forward KL over reverse KL (`KL(P_S || P_T)`) for v1:
- Reverse KL is mode-seeking — risks collapsing student to a single teacher
  mode.
- Recent 2026 work (Entropy-Aware OPD, cited in external review) flags
  this as a real failure mode when teacher is high-entropy.
- For an ablation experiment focused on the **delta_t weighting**, forward
  KL gives a more interpretable gradient (cross-entropy-like).
- v2 may sweep reverse-KL / mixed / entropy-gated forms.

K = 50 to match E0 (where the delta_t numbers we report are computed).

## 4. Delta normalization (mandatory)

E0 ViRL39K smoke confirmed `delta_t` is heavy-tailed:
`mean ≈ 0.259, median ≈ 0.001`. ~1% of tokens carry most of the weight.

```
w_t_raw = delta_t
w_t_clipped = clip(w_t_raw, p95(w_t_raw over valid tokens in batch))
w_t = w_t_clipped / mean(w_t_clipped over valid tokens in batch)
```

After this, `mean(w_t) ≈ 1`, so vanilla configs (w_t = 1) and delta
configs train at comparable gradient magnitudes.

v2 alternatives to sweep:
- `w_t = 1 + α * normalize(delta_t)` — additive form, less skewed
- `w_t = softmax(delta_t / τ) * len(valid)` — temperature-controlled
- per-trajectory mean=1 (E1 README's earlier idea)

## 5. Per-batch monitoring (mandatory)

Logged each train step:

| Metric | Configs | Why |
|---|---|---|
| `e1_v1/effective_kl_tokens` | A, B, C, D | Σ over KL-branch samples of valid response tokens. Drops if T_correct rate drops. |
| `e1_v1/effective_ce_samples` | C, D | # samples in CE-on-gold branch. |
| `e1_v1/kl_loss` | all | Pre-aggregated KL loss summed over batch. |
| `e1_v1/ce_loss` | C, D | Pre-aggregated CE loss summed over batch. |
| `e1_v1/kl_ce_ratio` | C, D | `kl_loss / (kl_loss + ce_loss)`. If ≈ 0, CE dominates → cannot attribute D>C to delta. |
| `e1_v1/delta_t_mean_pre_norm` | B, D | Sanity check on the clipping. |
| `e1_v1/delta_t_mean_post_norm` | B, D | Should hover near 1.0 if normalization is correct. |
| `e1_v1/delta_t_p99_pre_norm` | B, D | Tail size monitor. |

Per-bucket breakdowns (bucket = ViRL39K / POPE-style / synth+TallyQA)
of each of the above, so we can see if e.g. bucket 3's CE branch is
dominating the gradient.

## 6. CE-on-gold details

For samples where teacher greedy got the answer wrong (`T_correct = 0`):

- **No student rollout** — we don't want to imitate student's likely-wrong
  response on a task the teacher itself fails. Instead force-score student
  on gold answer tokens.
- Gold tokens come from precompute (tokenize the gold string with the same
  Qwen tokenizer used during student training).
- Loss = `β · sum_t (-student_logp[gold_t])` — standard NLL.
- `β` scales relative to KL branch. Initial value 1.0; expect to sweep.

This is a **hybrid policy**: KL branch is on-policy, CE branch is
off-policy (gold-supervised). Thinking Machines paper notes off-policy
SFT as a typical cold-start initialization before on-policy KD; here we
use it inline per-sample whenever teacher signal is unavailable.

The risk (per GPT review):

> If `kl_ce_ratio → 0` (CE dominates loss), then D > C is not evidence
> for delta — it could be entirely CE behavior. The
> `e1_v1/kl_ce_ratio` monitor catches this.

## 7. Trajectory-pass labeling

`T_correct` per sample is derived offline by `precompute_teacher.py`:

```
T_correct = verifier_pass(teacher_greedy_response, gold)
```

This is computed once per sample at precompute time. The `precompute_teacher.py`
script remains useful for this purpose; the per-token `teacher_logp` /
`delta_t` fields it also writes are NOT used by the on-policy trainer
(those need to be recomputed online on the student's rollout).

Eventually `precompute_teacher.py` should grow a `--lite` mode that skips
the dual forced-score step (~5× speedup) since only the greedy response
+ correctness flag are needed for on-policy v1.

## 8. Engineering path

```
Stage 1 (verl rollout spike, 0.5 day):
  Read verl/workers/rollout/ to understand how vLLM rollouts get
  attached to DataProto. Find the integration point for our teacher
  scoring callback.

Stage 2 (dual teacher scorer, 0.5 day):
  Write a teacher worker that, given (x, I, y_S), returns:
    - logp_T_I[t], topK_I[t]   (image-conditioned)
    - logp_T_null[t], topK_null[t]   (null-image-conditioned)
  Two HF forwards per call, batched across samples for GPU efficiency.
  Lives in src/teacher_scorer.py.

Stage 3 (on-policy losses, 0.5 day):
  src/on_policy_losses.py — register 4 losses with verl. Each reads
  data["teacher_topk_logp_I"], data["teacher_topk_ids"], data["delta_t"],
  data["loss_branch"] ("kl" or "ce") and dispatches.

Stage 4 (trainer + smoke run, 0.5-1 day):
  src/on_policy_trainer.py — orchestrates rollout → teacher scoring →
  loss. Plug into verl FSDP actor. 1K-subset smoke at 100 steps to
  verify forwards and metric outputs.

Stage 5 (full 4-config E1-mini, ~1 day):
  Run A, B, C, D on locked 8K mixture. Eval per § 6 of E1 README.
```

Total estimate: **3-5 days** for first real E1 v1 results. The smoke
test on the off-policy baseline (Stage 0) can run in parallel during
Stage 1-2.

## 9. Risks / open questions specific to on-policy v1

- **Student rollout quality at step 0**: a from-scratch student won't
  generate coherent responses. May need a 1-epoch off-policy weighted-SFT
  warmup before switching to on-policy. This is what Thinking Machines
  recommends (off-policy cold start → on-policy fine-tune).
- **Teacher dual-forward cost compounding**: at scale this is the
  bottleneck. Async / one-step-stale teacher scoring (verl scheduler
  supports it) is a known mitigation.
- **delta_t variance on student rollouts**: if student's rollouts are
  far from teacher's distribution (especially early in training), the
  teacher's `KL(P_T^I || P_T^null)` on those prefixes might be very
  different from E0's delta_t numbers measured on teacher's own response.
  Monitor and re-tune normalization if needed.
- **CE branch domination as student improves**: if student gets better,
  more samples will be in KL branch (T_correct mainly depends on teacher
  not student), but student rollouts may diverge more from teacher
  patterns → larger KL gradients. Track `kl_ce_ratio` evolution over
  training.
- **`P_T^null(.|s_t)` is conditioned on a black null image with student's
  rollout prefix**. This is a different distribution from E0's
  `P_T^null(.|s_t = teacher's prefix)`. The delta_t we compute online
  is the *correct* one for on-policy training, but means E0's per-topic
  gain_margin direction conclusions are NOT directly transferable —
  they were measured on teacher's prefix. v1 should report per-topic
  delta_t distribution on student's rollouts and compare to E0.
