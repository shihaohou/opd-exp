# NEXT — what to do next
*Last updated: 2026-05-13 (after Day 1.5 / Stage 2 v1 smoke). Pair this file with `PROGRESS.md`.*

> **For new Claude sessions:** if you're starting fresh, first read `CLAUDE.md`, then `PROGRESS.md` (what we did), then this file. Recommended next concrete action is at **§ Right now, Day 2** below. If anything in the on-policy trainer breaks at smoke time, also read `docs/e1_smoke_runbook.md`.

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
                                   builder shipped; Config A smoke passes (12 steps
                                   + validation + checkpoint on 50 ViRL39K samples)
E1 Day 2   (buckets 2/3 + dedup) : NEXT
E1 Day 3+  (mixture + full sweep): pending
E2 (mask sensitivity)            : not started
```

---

## Right now, Day 2 (data — buckets 2 and 3 + dedup + mixture)

Stage 2 trainer code is shipped and the Config A smoke is green. Bucket 1
(ViRL39K) parquet build also works (`make_train_parquet.py`). The blocker
for a real E1 sweep is the rest of the 8K mixture.

Read in order before starting:
1. `CLAUDE.md` (top reading list)
2. `experiments/E1_filtered_delta_opd/README.md` § Training data composition + dedup
3. `docs/e1_smoke_runbook.md` (if Stage 2 wiring needs touching again — should be stable now)

Concrete deliverables (in suggested order):

1. **`data/pope_style_builder.py`** — self-build POPE-style yes/no on COCO
   `train2017` (NOT the official POPE split — image leakage with POPE-adv eval).
   ~1.6K samples, balanced yes/no, mixed random/popular/co-occurring negatives.
   Output: append to a shared jsonl in the same precompute schema so
   `make_train_parquet.py` can ingest it once it grows beyond ViRL39K.

2. **`data/tallyqa_loader.py`** — TallyQA `complex` subset with mandatory
   COCO image_id filter against POPE-adv eval. Target ~900 samples.

3. **`data/synthesize_counterfactuals.py`** — parametric VLMBias-like
   synthesis (animals leg-count, flags stripe/star, game-board grids, chess
   counts, patterned-grid count). Target ~1500 samples. All base assets must
   be NEW (NOT pulled from VLMBias `main` — eval leakage).

4. **`data/dedup_check.py`** — three-layer image_id ∩ pHash ∩ CLIP NN check
   against POPE-adv + VLMBias `main` eval sets. **MUST PASS before any
   non-smoke launch.**

5. **`data/mixture.py`** — sample the 8K E1-mini from the 3 buckets per the
   locked recipe (4K ViRL39K + 1.6K POPE-style + 2.4K bucket 3).

6. **Extend `make_train_parquet.py`** to handle the new buckets (currently
   ViRL39K-only). Each bucket's loader yields the same `Sample` shape; the
   parquet builder just iterates over all of them.

After Day 2 closes, Day 3 runs the precompute pass on the full 8K mixture
and launches the 4-config sweep.

### Day 2 dependencies / things you can re-use

- **The bucket-loader pattern is already established** by `virl39k_loader.py`
  — copy its shape (a `Sample` dataclass + an `iter_*()` generator).
- **`precompute_teacher.py`** already handles arbitrary buckets via the
  `BUCKET_ITERS` registry — just add new entries when each loader lands.
- **The parquet schema is locked** by `make_train_parquet.py` (data_source,
  prompt, images, gold_text, gold_response_text, gold_token_ids,
  trajectory_pass, bucket). Just plug new buckets in.
- **Reward function**: `dummy_reward.py` already handles arbitrary
  `e1_*` data_sources via `data_source.startswith("e1_")` semantics — but
  currently it returns 0 for everything; if you need a real reward for
  monitoring (e.g., POPE F1 in training logs) extend it here.

### Stage 2 verification heads-up (before launching full sweep)

The Config A smoke run only exercised the **vanilla KD** code path. Before
launching a real run, sanity-check B / C / D each on the 50-sample parquet:

- **B (raw_delta_kd)**: dual teacher forward should run on every sample;
  expect `e1_v1/delta_t_mean_post_norm ≈ 1.0` in logs (clip-then-rescale
  normalization is supposed to land there).
- **C (filtered_kd)**: `e1_v1/effective_ce_samples > 0` (~half the batch
  on ViRL39K PassRate filter). `e1_v1/kl_ce_ratio` should sit in [0.3, 0.7]
  — if it drops to ~0, CE dominates and any D > C result is unattributable.
- **D (filtered_delta_kd)**: both delta_t mean ≈ 1.0 AND `kl_ce_ratio` in
  a healthy band.

The launcher syntax stays the same; just pass `B`, `C`, `D` instead of `A`.

### Optional cleanups (won't block Day 2/3 but pay dividends later)

- **Long-term R3 fix**: chase the `LD_LIBRARY_PATH` leak from NGC system
  torch (`/usr/local/lib/python3.12/dist-packages/torch/lib`). Probably
  filter in `activate.sh`. Until then, `TORCH_CUDNN_V8_API_DISABLED=1` is
  baked into the launcher.
- **Long-term R4 fix**: patch `verl/utils/dataset/rl_dataset.py:_build_messages`
  to not mutate the input image dict in place. Currently sidestepped via
  `filter_overlong_prompts: false`.
- **Teardown noise** at end of smoke run (vLLM EngineCore /
  resource_tracker errors after checkpoint write). Exit-0, not blocking;
  fix when it interferes with automation.

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
| **Day 1.5 (2026-05-12 → 13)** | ✅ done | Stage 2 trainer code: agent_loop + 4 losses + parquet builder + launcher. Config A smoke passes (12 steps + validation + checkpoint on 50 ViRL39K samples). 5 integration traps surfaced + fixed (see `docs/e1_smoke_runbook.md`). `make_train_parquet.py` already written (ViRL39K only). |
| Day 2 | pending | Build bucket 2 (POPE-style on COCO train) + bucket 3 (synthesis + TallyQA). Dedup pipeline. Freeze 8K E1-mini mixture. Extend `make_train_parquet.py` to multi-bucket. |
| Day 3 | pending | Run precompute on the 8K mixture (sample-level trajectory_pass + gold tokenize only — drop the per-token forced-score path; v1 doesn't use it). Sanity-check B / C / D smoke runs on the new parquet. |
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
6. Check `git log --oneline -10` to confirm where we left off (current HEAD: `b58932b7` as of this writing).
7. Pick up at the next un-checked item in **§ Right now, Day 2** above.
