# PROGRESS — what has been built and what we found
*Last updated: 2026-05-13 (after Day 2 data pipeline). Last commit: `5d4b5826` (B/C/D smoke verified). Day 2 code shipped locally, awaiting commit.*

> **For new Claude sessions:** read this file first, then `NEXT.md`, then `CLAUDE.md`. Read the latest `experiments/E0_image_null_delta/results/e0_verdict.md` for the canonical E0 findings; read `experiments/E1_filtered_delta_opd/on_policy_v1_design.md` for the canonical E1 training design. New: read `docs/e1_smoke_runbook.md` if anything in the on-policy trainer breaks at smoke time.

---

## TL;DR

- **Project**: Delta-OPD — **on-policy** distillation for VLMs reweighted by per-token image-vs-null teacher KL.
- **Phase**: E0 **complete** (Conditional GO + E0.3-A/B done). E1 **Day 1.5 / Stage 2 v1 complete** — on-policy trainer code + all 4 configs (A/B/C/D) verified end-to-end on 50 ViRL39K samples; B/D `delta_t_mean_post_norm ≈ 1.0`, C/D `kl_ce_ratio = 0.5` in the [0.3, 0.7] attribution-guard band. E1 **Day 2 data pipeline complete locally** (POPE-style builder / TallyQA loader / synthetic counterfactuals / 3-layer dedup / 8K mixture sampler / multi-bucket parquet builder — all unit-tested, awaiting commit and server-side data download for Day 3 precompute).
- **E0 verdict**: **Conditional GO**. 3 of 5 primary criteria pass; failures (delta-correctness correlation, gain_margin on VLMBias) are *informative* and constrain E1 method choice.
- **E0.3-B finding**: Length-normalized `gain_margin` confirms VLMBias per-topic direction is **stable** (no sign flips). Motivation is not a length-bias artifact.
- **E1 design (locked)**: On-policy v1. 4 configs A/B/C/D = `VanillaKD` / `RawDeltaKD` / `FilteredKD` / `FilteredDeltaKD`. **C is the critical control** — without it any D > A gain is unattributable to delta. Loss = top-K sparse forward KL on student-rollout prefixes, optionally × normalized `delta_t` × `1[T_correct]` mask, with CE-on-gold branch for teacher-wrong samples.
- **E1 mixture (locked, GPT-reviewed)**: 8K E1-mini = 4K ViRL39K (PassRate∈[0.3,0.9], single-image, stratified by category) + 1600 self-built POPE-style on COCO train2017 (NOT official POPE — image leakage; POPE-adv eval-id disjoint filter mandatory) + 2400 (1500 synth VLMBias-like + 900 TallyQA complex with COCO-id leakage filter). Three-layer dedup mandatory pre-launch.
- **Stage 2 v1 status (Day 1.5)**: all 4 losses, agent-loop manager + worker subclass, parquet builder, launcher → committed. All 4 configs A/B/C/D pass smoke on 50 ViRL39K. 5 non-obvious integration traps hit + resolved in `docs/e1_smoke_runbook.md` (NCCL duplicate-GPU from early CUDA init, Hydra `<locals>` qualname, cuDNN v9 sublib load on Conv3d, multimodal prompt-cap + filter mutation bug, missing reward registry).
- **Day 2 status (today)**: 6 deliverables shipped + locally unit-tested (35 unit tests across the data layer). `BUCKET_ITERS` in `precompute_teacher.py` now exposes 5 buckets (virl39k / pope_style / tallyqa / synthetic / mixture); `make_train_parquet.py` accepts all 4 individual buckets with `image_paths` resolution from precompute records (virl39k lookup retained as backward-compat fallback). No verl source touched; no submodule bumps needed. **Blocker for Day 3: server-side download of COCO train2017 + TallyQA (~20 GB).**
- **Headline**: `delta_t` tracks image influence, but the *direction* can be wrong on VLMBias adversarial recognition. **The right E1 recipe is `FilteredDeltaKD` (D)**, with `RawDeltaKD` (B) as negative-control and `FilteredKD` (C) as filtering-only control.

---

## What this project is

A VLM distillation experiment: train Qwen2.5-VL-7B from Qwen2.5-VL-32B, but reweight the per-token reverse-KL objective by `delta_t = KL(p_T(.|x,I) || p_T(.|x,null))`. Hypothesis: high-delta tokens are visually-grounded, so weighting by delta should encourage the student to imitate the *visual reasoning* part of the teacher specifically — not the language-prior coast-through.

E0's job is to validate that delta is a useful signal **before any training**. If delta doesn't track image influence, no point training Delta-OPD.

---

## Architecture / repo decisions made

- `shihaohou/opd-exp` is the **experiment parent repo** (this repo). It was originally a fork of `verl-project/verl`, but we **renamed the old fork to `shihaohou/verl`** and created `opd-exp` as a fresh non-fork repo with verl as a git submodule.
- `verl/` is a git submodule pointing to `shihaohou/verl` (which is our fork of upstream `verl-project/verl`). Pinned to commit `5a506cc5`. **Modifying verl source = commit inside `verl/` first, then bump submodule pointer in parent.** Three-layer workflow if recipe (`verl/recipe/`) is touched — see CLAUDE.md.
- All training scripts (when they exist) will be **launched from the parent repo root**, importing verl as a library and writing outputs under `experiments/<EXP>/results/` (gitignored).

---

## E0 — what was built

### Code (under `experiments/E0_image_null_delta/`, committed)

| File | Purpose |
|---|---|
| `configs/e0_default.yaml` | Single source of truth for paths, sample budgets, K=50, greedy gen, seed=42. |
| `data/loaders.py` | Sample dataclass + 3 loaders (VLMBias main / POPE-adv / MathVista testmini) with unified `Sample(dataset, sample_id, question, image, gold, extras)` shape. |
| `src/null_image.py` | All-black PIL image generator; image_drop is a prompt-level mode (not implemented yet). |
| `src/prompting.py` | Qwen2.5-VL chat-template wrappers for generation + forced scoring. |
| `src/dual_forward.py` | Per-sample driver: 2 greedy generations + 2 forced-score forwards on response + 4 option-score forwards. Writes one jsonl line per sample. Single GPU; bash launchers fan out 8 shards. |
| `src/metrics.py` | CPU-only aggregator. Reads shard jsonls → CSV + verdict.md + top_delta_tokens.json. |
| `scripts/run_e0_teacher32b.sh` | 8-way data-parallel launcher across 8 CUDA devices. |
| `scripts/run_e0_student7b.sh` | Same shape for 7B. |
| `scripts/run_e0_teacher72b_sanity.sh` | Single proc, `device_map=auto` across 2 GPUs, first 200 VLMBias samples. **NOT YET RUN.** |
| `scripts/aggregate.sh` | Wrapper invoking metrics.py. |
| `analysis/e0_report.py` | Matplotlib script producing 3 figures from jsonls; runs on Mac. |

### Runs executed on remote (`arc-wlf1-ge103-4`)

| Model | Dataset coverage | Wall clock | Output |
|---|---|---|---|
| Qwen2.5-VL-32B (teacher) | VLMBias main (2780) + POPE-adv (1000) + MathVista (500) | ~80 min on 8 H800 (with GPU contention from other tenants) | 24 shard jsonls |
| Qwen2.5-VL-7B (student) | Same coverage | ~35 min | 24 shard jsonls |
| Qwen2.5-VL-72B (sanity) | VLMBias 200 samples | **NOT RUN** | — |

Aggregated outputs (gitignored, but generated locally on Mac too):
- `experiments/E0_image_null_delta/results/e0_summary.csv`
- `experiments/E0_image_null_delta/results/e0_verdict.md`
- `experiments/E0_image_null_delta/results/top_delta_tokens.json`

---

## E0 — what we found

### Primary criteria results (5 criteria, 2-of-N required for GO)

| # | Metric | Where | Result | Pass? |
|---|---|---|---|---|
| 1 | Acc(T,I) > Acc(T,null) | VLMBias `main` | 0.216 vs 0.145 (+0.071) | ✅ |
| 2 | delta_gap > 0 AND Spearman > 0 | VLMBias | gap=-0.41, Spearman=-0.26 | ❌ |
| 3 | gain(ground_truth) > gain(expected_bias) | VLMBias | margin=-1.82 | ❌ |
| 4 | Top-delta tokens are vision-bearing | All datasets, manual | VLMBias ~92%, POPE ~98%, MathVista ~52–60% (strict) | ✅ (manual) |
| 5b | POPE mean_delta(grounded) > mean_delta(hallucinated) | POPE-adv | gap=+0.108 (g=0.827, h=0.720) | ✅ |

→ **3 / 5 pass → Conditional GO.**

### Sanity-dataset highlights (not in primary criteria)

- **POPE delta_acc = +0.363** (acc 50% null → 86.3% image). Image *really* matters on object presence.
- **MathVista Spearman(mean_delta, correctness) = +0.41**. Cleanest healthy-VQA signal in the run.

### VLMBias per-topic gain_margin (sorted)

| Topic | n | acc_I | gain_margin |
|---|---:|---:|---:|
| Optical Illusion | 792 | 0.528 | **+0.71** (only positive) |
| Patterned Grid | 336 | 0.167 | -0.44 |
| Logos | 414 | 0.169 | -1.50 |
| Flags | 240 | 0.175 | -2.60 |
| Chess Pieces | 288 | 0.017 | -2.80 |
| Game Boards | 168 | 0.065 | -3.90 |
| Animals | 546 | **0.000** | **-5.10** |

Recognition topics (everything except Optical Illusion) have negative `gain_margin`: image-conditional logP pushes the *biased wrong* answer up more than the right answer. This is exactly the VLMBias paper's failure-mode claim and it shows up *cleanly* in our delta metric.

### Student-teacher same-wrong overlap (metric 5a) — **HEAVILY caveated**

- Global same-wrong rate (after fixing the answer-extraction bug) = **72.9%** (was 39.1% with naïve full-string compare).
- Two same-family pretrained models share a substantial prior **before any distillation happens** — this is the floor, not evidence of inheritance.
- Per-topic excess-over-chance (`rate − 1/(answer_space−1)`):
  - Flags **+0.558** / Game Boards **+0.516** / Chess Pieces **+0.239** / Logos **+0.229** ← real shared-bias signal
  - Animals and Optical Illusion sit near 100% mechanical ceiling (binary answer space).

---

## External review

Multiple GPT review passes converged on the same verdicts as Claude. Genuine value-adds:

**During E0 (initial)**:
- **Teacher-Error Inheritance (TEI) metric family** for E1: `Acc_S | T_wrong`, TEI rate, Escape rate. Codified in `experiments/E1_filtered_delta_opd/README.md`.
- **Chance baseline** framing for 5a. Drove the per-topic table.
- **Filtered Delta-OPD cannot fix Animals alone** (teacher acc = 0/546 → no positives to weight) — must combine with ground-truth CE.

**During E1 design (today)**:
- **Don't use official POPE random/popular as bucket-2 training** — they share the COCO val 500 image pool with POPE-adversarial eval. Self-build POPE-style on COCO train instead.
- **VLMBias `withtitle` / `remove_background_*` siblings are NOT image-disjoint with `main`** — they're title-injected variants of the same base counterfactuals. Synthesis-primary mix for bucket 3.
- **Mandatory three-layer dedup** before training launch: image_id intersection + pHash near-duplicate + CLIP embedding NN.
- **Per-bucket monitoring** with `kl_ce_ratio` attribution guard — if CE-on-gold dominates loss, D > C is unattributable to delta.
- **OPD = On-Policy Distillation** (not off-policy). The first off-policy code drafts (precompute_teacher.py, losses.py) had to be reframed as smoke baselines; the real E1 trainer uses on-policy student rollouts. Design now in `on_policy_v1_design.md`.

---

## E1 — Day 1 progress (2026-05-12)

### Locked design decisions

- **On-policy, top-K sparse forward KL `KL_topK(P_T^I || P_S)` at student rollout prefixes.** Reverse KL deferred (mode-collapse risk per Thinking Machines, Entropy-Aware OPD 2026).
- **4 configs renamed A/B/C/D**: `VanillaKD` / `RawDeltaKD` / `FilteredKD` (with CE-on-gold) / `FilteredDeltaKD` (with CE-on-gold). Central comparison is **D vs C** (does delta help given filtering?), with **B vs A** isolating raw delta and **C vs A** isolating filtering+CE.
- **Verl backbone**: FSDP `verl/trainer/distillation/` + `verl/utils/dataset/rl_dataset.py` for multimodal + subclass `verl/experimental/agent_loop/SingleTurnAgentLoop` for dual-forward. NOT `recipe/gkd/` (Megatron-only). No verl source edits required.
- **delta_t normalization**: `clip(delta_t, p95) / mean(clip)` per batch.
- **8K E1-mini mixture**: 4K ViRL39K (PassRate∈[0.3,0.9], single-image) + 1600 self-built POPE-style on COCO train + 2400 (1500 synth VLMBias-like + 900 TallyQA complex).

### Code built today (committed)

| File | Purpose | Status |
|---|---|---|
| `experiments/E0_image_null_delta/src/metrics.py` | E0.3-B: length-normalized `gain_margin`. Loads Qwen tokenizer; boundary-trick `len(tok("\n"+opt)) - len(tok("\n"))`. Reports raw + lengthnorm. | ✅ Run on existing jsonls; no GPU re-run. **No sign flips.** |
| `experiments/E1_filtered_delta_opd/data/virl39k_loader.py` | ViRL39K parquet streamer with PassRate filter, `<image>` strip, `\boxed{}` extraction, single-image filter. | ✅ 11,847 eligible rows; server-side smoke test 5/5 image loads OK. |
| `experiments/E1_filtered_delta_opd/src/precompute_teacher.py` | Per-sample teacher dual forward (image + null) + delta_t. Reuses E0 helpers via sys.path injection. **v1 use: only sample-level `trajectory_pass` + `gold`. Per-token data is wrong-trajectory (will be recomputed online).** | ✅ Smoke 10/10 records, 5/10 `trajectory_pass`, delta_t mean=0.259 / median=0.001 — sparse signal reproduced from E0. |
| `experiments/E1_filtered_delta_opd/src/losses.py` | 4 verl-registered losses named `e1_offline_weighted_sft_*`. Math: `-student_logp × delta × pass_mask`. **Smoke baseline only — NOT E1 results.** | ✅ Local unit-tests 5/5 passed. |
| `experiments/E1_filtered_delta_opd/src/spike_vllm_dual_forward.py` | Verifies vLLM multimodal + `prompt_logprobs=K` on Qwen2.5-VL-32B. | ✅ Server-side: K=50 verified after `max_logprobs=K` engine fix; **position-50 in K12-1000-0 showed killer delta signal** (image: `geometric/triangle/geometry/几何`; null: `completely/solid/plain/blank`). |
| `experiments/E1_filtered_delta_opd/on_policy_v1_design.md` | **Canonical design for Stage 2.** Per-prompt loop with KL branch + CE-on-gold branch; cost model; loss math; delta normalization; per-batch monitoring; engineering staged plan; risks. | ✅ Written; referenced from CLAUDE.md new-session reading list. |
| `experiments/E1_filtered_delta_opd/README.md` | Major rewrite: on-policy explicit; 4-config A/B/C/D renamed; delta normalization; off-policy smoke baseline section; updated engineering punch list. | ✅ |
| `docs/migrate-env.md` | New env-troubleshooting runbook. Q1-Q4 are existing NGC traps; **Q5 is new**: triton ldconfig UnicodeDecodeError on HPC-X machines (dev box 1). | ✅ |
| `activate.sh.template` | Added detect-only check for Q5 (warns, never auto-patches). | ✅ |
| `CLAUDE.md` | Pointer to `docs/migrate-env.md`; on_policy_v1_design.md added to new-session reading list. | ✅ |

### Spikes done

- **Spike A — verl recipe**: `verl/recipe/gkd/` is Megatron-only, single-condition teacher, no multimodal. Not a fit. `verl/trainer/distillation/` (FSDP) + `verl/utils/dataset/rl_dataset.py` (multimodal) is the right backbone.
- **Spike B — verl on-policy infra**: `verl/experimental/agent_loop/agent_loop.py` already does rollout → teacher logp → DataProto. `AsyncTeacherLLMServerManager` already accepts `multi_modal_data["images"]`. **We just subclass the agent loop and call the teacher twice (image + null) per sample. No verl source edits.**
- **Spike C — vLLM dual-forward (the precondition for Spike B)**: Triton + HPC-X env bug fixed with one-line sed on triton driver.py (`.decode("utf-8", errors="ignore")`). vLLM `LLM(..., max_logprobs=50)` then accepts `SamplingParams(prompt_logprobs=50)`. Multimodal works. Position-50 demo shows clean image-vs-null divergence — exactly the delta signal we want to upweight.

### What is NOT in Day 1 (deferred to Stage 2+)

- `src/on_policy/agent_loop.py` (the dual-forward subclass) — 1 day code work
- `src/on_policy/losses.py` (4 on-policy KD wrappers) — same
- `configs/recipe_*.yaml` — 4 YAML configs
- `data/pope_style_builder.py` — Day 2
- `data/tallyqa_loader.py` — Day 2
- `data/synthesize_counterfactuals.py` — Day 2
- `data/dedup_check.py` — Day 2
- `data/mixture.py` + `make_train_parquet.py` — Day 3
- `src/eval_tei.py` — Day 3-4

---

## E1 — Day 1.5 / Stage 2 v1 (2026-05-12 late → 2026-05-13)

Stage 2 of `on_policy_v1_design.md` (on-policy trainer code) landed and the
**Config A smoke run passed end-to-end** — Qwen2.5-VL-32B teacher + 7B
student, 50 ViRL39K samples, 12 train steps + validation + checkpoint
write. The 4 monitored configs are wired (A/B/C/D); only A has been actually
exercised on the GPU so far.

### Code built (committed)

| File | Purpose | Status |
|---|---|---|
| `experiments/E1_filtered_delta_opd/src/on_policy/agent_loop.py` | `DeltaOPDAgentLoop` (subclass of `SingleTurnAgentLoop`, per-sample KL/CE dispatch) + `DeltaOPDAgentLoopWorker` (overrides `_compute_teacher_logprobs` for dual image+null teacher forward + `delta_t = KL_topK_union(P_T^I, P_T^null)`) + `DeltaOPDAgentLoopManager` (Hydra-instantiable manager that picks our worker subclass). | ✅ |
| `experiments/E1_filtered_delta_opd/src/on_policy/losses.py` | 4 registered losses: `e1_onpolicy_{vanilla,raw_delta,filtered,filtered_delta}_kd`. Per-sample dispatch via `is_kl`/`is_ce` masks. CE branch = `β · NLL(student | gold)`. Mandatory metrics (`kl_loss_sum` / `ce_loss_sum` / `kl_ce_ratio` / `effective_kl_tokens` / `effective_ce_samples` / `delta_t_*` / per-bucket breakdown). | ✅ Local 6/6 unit tests pass. |
| `experiments/E1_filtered_delta_opd/src/on_policy/__init__.py` | `enable()` (driver-side) + `install_late_hook()` (compatibility no-op). The Ray worker_process_setup_hook approach was abandoned — see runbook R1. | ✅ |
| `experiments/E1_filtered_delta_opd/src/on_policy/entrypoint.py` | Driver-side launcher: calls `enable()`, then dispatches to `verl.trainer.main_ppo.main()`. Imports verl deeply only in the driver, where it's safe to init CUDA. | ✅ |
| `experiments/E1_filtered_delta_opd/src/on_policy/dummy_reward.py` | Always returns `0.0`. verl's agent loop calls `_compute_score` unconditionally even with `use_task_rewards=false`, so we need a stub matching the `compute_score(data_source, solution_str, ground_truth, ...)` signature. | ✅ |
| `experiments/E1_filtered_delta_opd/data/make_train_parquet.py` | jsonl from `precompute_teacher.py` → parquet. Tokenizes gold via student tokenizer; formats gold response as `"Final answer: \boxed{X}."` (GPT-locked anchor form, not bare `\boxed{}`). Bundles image bytes + bucket + `trajectory_pass` for the agent-loop dispatch. v1 covers ViRL39K bucket only. | ✅ |
| `experiments/E1_filtered_delta_opd/configs/{e1_base, recipe_A/B/C/D, agent_loop}.yaml` | Hydra config bundle. `e1_base` extends verl's `ppo_trainer` via `hydra.searchpath: [pkg://verl.trainer.config]`. Recipe yamls bake `loss_mode` + `experiment_name`. `agent_loop.yaml` registers `DeltaOPDAgentLoop` as `delta_opd_single_turn`. | ✅ |
| `experiments/E1_filtered_delta_opd/scripts/run_e1_recipe_smoke.sh` | `bash scripts/run_e1_recipe_smoke.sh A` style launcher. Bakes the non-obvious env defaults (`TORCH_CUDNN_V8_API_DISABLED=1`, `PYTHONPATH=$PWD`, 4+4 GPU split, `num_workers=NGPUS`, multimodal prompt cap). | ✅ |
| `verl/workers/config/distillation.py` (submodule edit at `2c118243`) | `DistillationLossConfig.__post_init__` now lazy-registers `e1_onpolicy_*` losses. Submodule pointer bumped in `022c465f`. | ✅ |
| `docs/e1_smoke_runbook.md` | R1-R5: each of the 5 non-obvious integration traps + verification + fix + long-term cleanup. | ✅ |

### Smoke run results (all 4 configs verified)

Run command per config: `bash scripts/run_e1_recipe_smoke.sh {A,B,C,D}` with
`TRAIN_BATCH_SIZE=4`, `MAX_PROMPT_LENGTH=4096`, etc. (launcher defaults).
50 ViRL39K samples → 12 train steps per epoch, 1 epoch, then validation
and checkpoint. Each run takes ~3 min wall time.

Per-stage pipeline (same for A/B/C/D):

| Stage | Status |
|---|---|
| Parquet build (50 ViRL39K samples → `/tmp/e1_smoke.parquet`) | ✅ |
| Ray cluster init + 4 actor + 4 teacher placement | ✅ |
| Student FSDP init (was the NCCL Duplicate-GPU failure point) | ✅ after R1 fix |
| Teacher vLLM (Qwen2.5-VL-32B tp=4) + CUDA graphs capture | ✅ |
| Agent loop chunking (was failing at batch=4 / num_workers=8) | ✅ after `num_workers=NGPUS` |
| Rollout + `compute_log_prob` (was the cuDNN sublib fail point) | ✅ after R3 fix |
| 12 train steps | ✅ |
| Validation pass | ✅ |
| Checkpoint write | ✅ |
| Graceful exit | ⚠️ vLLM/Ray teardown emits noisy errors but exit-0; not blocking |

Per-config monitoring metrics (step 12):

| Config | `kl_ce_ratio` | `effective_ce_samples` (per micro) | `delta_t_mean_post_norm` | `delta_t_mean_pre_norm` | Verdict |
|---|---|---|---|---|---|
| A vanilla_kd | — | — | — | — | ✅ all KL, no delta, no filter |
| B raw_delta_kd | — | 0 (no filter) | **0.99999998** | 0.203 (matches E0 ViRL39K) | ✅ delta normalization mathematically tight |
| C filtered_kd | **0.50** | 0.5 (batch=4, ~50% T_wrong) | — (no delta) | — | ✅ KL/CE balance dead-center of healthy band |
| D filtered_delta_kd | **0.50** | 0.5 | **0.99999997** | 0.057 | ✅ both delta + CE active |

Attribution guard from GPT review (`kl_ce_ratio` in `[0.3, 0.7]`) holds for
C and D → β=0.1 is fine, no tuning needed before Day 2.

Interesting (not a blocker): D's `delta_t_mean_pre_norm` is ~3.5× lower than
B's. D only computes delta over teacher-correct samples; that subset shows
weaker image dependence than the full ViRL39K population. Worth a closer
look in the eval pass (D-vs-C gain might be small if `delta_t` itself is
small on the filtered subset), but doesn't block running the full 8K sweep.

How to re-extract these metrics from a future smoke run:
```bash
bash scripts/run_e1_recipe_smoke.sh C ... 2>&1 | tee /tmp/c.log
bash experiments/E1_filtered_delta_opd/scripts/show_e1_metrics.sh /tmp/c.log
```

### Stage 2 design deviations from `on_policy_v1_design.md`

None of substance. Implementation diverged from the design doc only at the
mechanical level (e.g., late-register vs Ray hook for loss registration —
the design doc didn't specify a registration mechanism). The math and the
4-config matrix are exactly as designed.

### What is NOT in Day 1.5 (closed by Day 2 below, except long-term cleanups)

- `data/pope_style_builder.py` — **Day 2 ✅**
- `data/tallyqa_loader.py` — **Day 2 ✅**
- `data/synthesize_counterfactuals.py` — **Day 2 ✅**
- `data/dedup_check.py` — **Day 2 ✅** (must pass before non-smoke launch)
- `data/mixture.py` — **Day 2 ✅**
- `src/eval_tei.py` — Day 3-4 (TEI / Escape / per-topic gain_margin + MathVista retention)
- B / C / D smoke runs on the new 8K mixture — Day 3 (Day 1.5's B/C/D runs were on 50 ViRL39K, not on multi-bucket)
- Long-term fix for the cuDNN v9 sublib load (LD_LIBRARY_PATH cleanup; runbook R3 "Long-term")
- Long-term fix for `filter_overlong_prompts` (verl `_build_messages` mutation; runbook R4 "Long-term")

---

## E1 — Day 2 / Data pipeline (2026-05-13)

Day 2 ships the rest of the data layer needed for the 8K E1-mini run.
**All 6 deliverables landed locally with unit tests; nothing committed
yet and nothing exercised on the server (COCO train2017 + TallyQA still
need to be downloaded — see § "Day 3 prerequisites" in NEXT.md).** No
verl source was touched; no submodule bumps needed.

### Code built (locally, awaiting commit)

| File | Purpose | Local unit tests |
|---|---|---|
| `experiments/E1_filtered_delta_opd/data/pope_style_builder.py` | Bucket 2: POPE-style yes/no on COCO train2017. random/popular/cooccur negatives, yes:no=1:1, mandatory POPE-adv eval-id disjoint filter. `COCOInstanceIndex` reusable across builders. | 4/4 (index, quota, disjoint, question format) |
| `experiments/E1_filtered_delta_opd/data/tallyqa_loader.py` | Bucket 3a: TallyQA `complex` subset with COCO-id × POPE-adv leakage filter; answer rewrapped as `\boxed{N}`. Accepts both JSON-list and JSONL ingests. | 7/7 (filters, JSONL/wrapped-list/answer-range/seed determinism) |
| `experiments/E1_filtered_delta_opd/data/synthesize_counterfactuals.py` | Bucket 3b: parametric synth of 5 topics (patterned_grid / flag / game_board / chess_position / animal). PIL-only generators + `build_synthetic_counterfactuals` writes images and `manifest.jsonl`. | 7/7 (per-topic, determinism, build/iter round-trip, gold range) |
| `experiments/E1_filtered_delta_opd/data/dedup_check.py` | 3-layer dedup against POPE-adv + VLMBias `main` evals: (1) numeric image_id ∩, (2) pHash Hamming<5, (3) CLIP cos>0.95. Exit-1 on any finding (intended for CI gate). | Layer-1 + edges (imagehash/open_clip not on Mac); Layers 2-3 run on server |
| `experiments/E1_filtered_delta_opd/data/mixture.py` | 8K E1-mini sampler. Stratified ViRL39K by category with slack-redistribution; POPE-style 1600; TallyQA 900; synth 1500. Writes uniform `mixture.jsonl`. Graceful skip on missing-bucket inputs. | 6/6 (proportional, capped strata, empty strata, e2e synth-only, missing-bucket skip) |
| `experiments/E1_filtered_delta_opd/data/make_train_parquet.py` (updated) | Multi-bucket: `ACCEPTED_BUCKETS = {virl39k, pope_style, tallyqa, synthetic}`. Reads image bytes from `rec["image_paths"]` (Day-2 schema); virl39k lookup retained as fallback for Day-1.5 smoke jsonls. | e2e simulate-without-GPU ✓ |
| `experiments/E1_filtered_delta_opd/src/precompute_teacher.py` (updated) | `BUCKET_ITERS` now has all 5 (`virl39k` / `pope_style` / `tallyqa` / `synthetic` / `mixture`). Output record carries `image_paths` + lets `extras["bucket"]` override the function-arg bucket (so mixture rows tag themselves with the original sub-bucket). | CLI choices verified; lazy imports preserved |
| `experiments/E1_filtered_delta_opd/data/__init__.py` (updated) | Exports 15 new APIs. | import sanity ✓ |

### Day-2 design notes (not derivable from code alone)

- **POPE-adv disjointness is not automatic from the COCO 2014 vs 2017 split**. train2017 = train2014 ∪ (val2014 \\ minival5k), so val2014 image_ids (POPE's sampling pool) sit inside train2017. We extract POPE-adv's actual image-id set at runtime via `load_pope_adv_image_ids` and filter explicitly. Same id-set is reused by the TallyQA loader (any COCO-sourced TallyQA row in the set is dropped).
- **Stratified sampling with slack redistribution**: ViRL39K categories have very uneven sizes (Geometric ~9k vs other categories smaller); pure proportional sampling under-allocates the small categories. Algorithm: proportional targets → cap at availability → redistribute slack among uncapped strata, repeat until n_target hit or no slack moves. Verified by unit test.
- **Animal silhouettes use PIL geometric shapes** (body ellipse + leg rectangles + head circle). They give the student counting practice on variable leg-count creatures, but **won't strongly trigger the photo-realist "dog → canonical 4 legs" recognition prior** the way VLMBias photos do. Documented in the file's docstring. If D vs C doesn't move on the Animals topic in eval, this is a known suspect — v2 should consider CLIP-guided / diffusion-based animal generation.
- **Chess pieces are letter glyphs on colored discs** (K/Q/R/B/N/P), not Unicode chess characters. Avoids TTF portability issues while still presenting an 8×8 board with discrete white-vs-black piece counts.
- **Schema bridge**: `precompute_teacher.py` now writes `image_paths` into every output record, so `make_train_parquet.py` doesn't need a per-bucket lookup table for non-virl39k buckets. The legacy virl_lookup path is kept as a fallback for the Day-1.5 smoke jsonls that pre-date this schema.
- **`mixture` is a meta-bucket** in `BUCKET_ITERS`. It reads a pre-built `mixture.jsonl` (from `data/mixture.py`); each row's true bucket lives in `extras["bucket"]`. `process_sample` honors that override so per-bucket monitoring in `on_policy/losses.py` sees the four bucket names (virl39k / pope_style / tallyqa / synthetic), not the meta-name.
- **Synth build is a separate step** (writes ~1500 PNGs + manifest under `out_dir`), not inline in the mixture call. Lets the user inspect images before sampling. The mixture call expects the synth manifest to already exist; `--build-synth` flag at CLI builds on the fly when absent.

### Verification status

- **Local unit tests**: 35 tests across the 6 new files, all passing.
- **End-to-end multi-bucket simulate-without-GPU**: synth → mixture manifest → simulated precompute jsonl → parquet pipeline reads cleanly. ✓
- **CLI surfaces**: `precompute_teacher.py --help`, `mixture.py --help`, `dedup_check.py --help` all show updated bucket choices / args.
- **Layer 2/3 of dedup**: skipped locally (no `imagehash` / `open_clip_torch` on Mac); requires server-side run.
- **Backward compatibility**: the Day-1.5 Config A smoke path (virl39k bucket, no inline `image_paths` in record) still works via the retained virl39k lookup fallback in `make_train_parquet.py`.

### What is NOT in Day 2 (deferred to Day 3+)

- **Server-side data download** (~20 GB): COCO `train2017/` + `annotations/instances_train2017.json`; TallyQA `train.json` + image directory (COCO train2014/val2014 + VG_100K). Pre-req for everything in Day 3 § Right now.
- **`src/eval_tei.py`** — Day 3-4.
- **Actual precompute on the 8K mixture** — Day 3 (estimate: ~5-7h sequential / ~1h with 8-shard parallel on 8×H800).
- **B/C/D smoke on the 8K mixture** — Day 3. The Day-1.5 B/C/D results used 50 ViRL39K samples only; per-bucket `kl_ce_ratio` on the multi-bucket mixture is expected to differ.
- **Full 4-config sweep** — Day 4.
- **R3/R4 long-term cleanups** — pending, not blocking Day 3.

---

## External review (2026-05-13 evening) — E1 goal realignment

After Day 2 close, external GPT review flagged that the planned Day 3 ordering ("8K precompute → 4-config train → eval") was structurally backwards: a few days of training compute *before* producing the metrics E1 was actually designed to test (TEI / Escape / per-topic gain_margin). The realignment is now encoded in:

- A new **E1 Protocol** block at the top of `experiments/E1_filtered_delta_opd/README.md` — 4 hypotheses, 4 configs, primary vs safety metrics, outcome interpretation tree.
- A rewritten "Right now, Day 3" section in `NEXT.md` — eval-first 4-step plan replacing the old "precompute → train" ordering.

Core realignment in one line:

> E1 is the first training-side **causal experiment** for Delta-OPD. The question is not "does the student score higher" — it is "does vanilla OPD inherit teacher wrong patterns, does raw delta amplify wrong-direction image influence, and does correctness-filtered Delta-OPD reduce inheritance".

Day-3 reordered to 4 steps (strict order):

1. **`src/eval_tei.py` (CPU, ~250 LoC)** — VLMBias per-topic + Recognition Aggregate + TEI / Escape + POPE yes-rate + MathVista retention. Unit-testable with fixture jsonls on Mac before any GPU work.
2. **Bucket-3 teacher sanity** — run `precompute_teacher.py --bucket synthetic` on all 1500 synth images; check per-topic teacher accuracy + gain_margin + canonical-prior trigger rate. **Gate**: if teacher accuracy on `animal` is ≥80% and gain_margin doesn't go negative, the silhouettes are too toy — fix data before training.
3. **1K mini-sweep** — drop 8K mixture to 1K (same per-bucket ratio), A/B/C/D × ~50 steps, immediately call eval_tei. Look at *direction*, not magnitude. Cheap (~1.5h vs ~12h for 8K).
4. **8K full sweep** — only after 1-3 pass.

Also locked into the protocol:

- **Primary metrics ordering**: VLMBias Recognition Aggregate / TEI rate / Escape rate / gain_margin. **Safety**: POPE yes-rate / MathVista retention. **Secondary**: loss curves / `kl_ce_ratio` / `delta_t` distribution.
- **Outcome interpretation tree** — ideal / acceptable / danger (`D > A` but `D ≈ C`; CE-on-gold steals credit) / uninformative — each with a fallback action so we don't fish for narratives after the fact.
- **Hard rule**: never report `e1_offline_weighted_sft_*` (off-policy weighted SFT in `src/losses.py`) as an E1 scientific result. Smoke baseline only.

What was already aligned (no change needed):

- Off-policy precompute path already labeled "NOT E1 results" in `README.md` § "Off-policy smoke baseline".
- 72B teacher already deferred to sanity check, not an E1-mini variable.
- Day-1.5 B/C/D smoke (50 ViRL39K) explicitly required to be re-verified on the multi-bucket mixture in Day 3 Step 3 (with eval_tei).

External-review-flagged data risk (carried into Step 2 above):

- **Animal silhouettes may be too toy.** Current PIL geometric shapes (body ellipse + leg rectangles + head circle) won't fire a strong "dog → canonical 4 legs" recognition prior. If Step 2 confirms teacher isn't being mis-led by the silhouettes, the synth pipeline needs a v2 with CLIP-guided / diffusion-based animal generation. Other 4 synth topics (grid / flag / board / chess) are not flagged.

---

## Pitfalls encountered — keep in mind for any new session

### Server environment (NGC PyTorch image — `arc-wlf1-ge103-4`)

See **CLAUDE.md → Environment setup (NGC machine specifics)** for the canonical version. Short version:

1. `PIP_CONSTRAINT=/etc/pip/constraint.txt` locks torch versions — must `unset`. Handled by `activate.sh`.
2. System `/usr/local/lib/python3.12/dist-packages/torch/` is NGC-custom (`2.8.0a0+nv25.6`); leaks into PEP 517 build isolation. TransformerEngine MUST be installed with `--no-build-isolation`.
3. **The `--no-deps` rule**: every `uv pip install -e` on this box must pass `--no-deps`. Otherwise dep re-resolution silently overwrites the hand-built TE binary (30–40 min to rebuild). No exceptions unless explicitly updating deps.
4. `huggingface-hub` auto-upgrades to 1.x → breaks `transformers 4.56.1`. Pin to `>=0.34.0,<1.0`.
5. verl install script downloads a stray `flash_attn-*.whl` to cwd — safe to `rm` after.
6. `pip check` complaint about `decord 0.6.0` is benign — we don't use the video path.
7. **Don't touch the `.venv` symlink** on the server. Machine-specific.

### Code-level footguns we already hit (and fixed)

- **MathVista** PIL image is in column `decoded_image`, not `image` (which is a filename string). Easy bug.
- **VLMBias** has the `expected_bias` column — directly drives metric 3 (visual gain comparison). Don't ignore.
- **Same-wrong matching** must extract the short-form answer (`{Yes}` / `yes` / etc.), not compare full lowered strings. CoT preambles will otherwise make same answers look different. (Fixed in `metric_5a`.)
- **Top-K tokens** need dedup by sample. The naïve global top-50 was flooded by `" sets"` from ~15 Zollner-illusion samples and over-stated the vision-bearing rate.
- **Forced-scoring image alignment**: null image MUST be same resolution as real image, otherwise the number of `<|image_pad|>` tokens differs and the prompt-length offset doesn't line up across conditions. (All-black at original resolution is the safe default.)
- **`torch_dtype=` is deprecated in transformers 4.56**, use `dtype=`. (Fixed.)
- **`Qwen2VLImageProcessor` defaults to fast processor** in recent transformers; slight numerical drift from slow processor. Currently accepted; flag if downstream signals look off.
- **`python -c "..."` multi-line via terminal paste** breaks on zsh/bash because leading whitespace gets eaten. Use heredoc `python <<'PY' ... PY` instead.

### Operational footguns

- The remote box is **shared**. Other users run `vllm serve gemma-4-31B --tensor-parallel-size 8 --gpu-memory-utilization 0.90` and `llamafactory-cli api` jobs that eat ~140 GB / 144 GB on GPUs 0–3 (memory occupied, compute mostly idle). Your nvidia-smi will look weird; this is **not** your jobs misbehaving. See ps + nvidia-smi commands logged in chat history.
- E1 training cannot share the box with these tenants — coordinate or wait.

### Methodological caveats still pending fix

These are noted in `e0_verdict.md` → "Caveats" / "Pending E0.x additions":

- ~~**Option scoring is raw `sum(log p)`**~~ ✅ **Resolved 2026-05-12 (E0.3-B)** — `metrics.py` now reports raw + length-normalized; per-topic direction stable, no sign flips.
- **All-black null image** could itself be OOD for the vision encoder, inflating KL. Step-2 ablation will compare with `image_drop`, Gaussian noise, and an unrelated-but-natural image. **Note for on-policy v1**: `delta_t` at training time is on student rollout prefixes (not teacher's), so the same caveat applies — monitor `delta_t` distribution on student rollouts vs E0's teacher-prefix numbers.
- **PPL-based student metrics** not computed yet (would need another server run, ~15 min on 7B).
- **72B teacher sanity** not run yet.

### Day-1 lessons surfaced today

- **OPD is on-policy in the modern literature** (Thinking Machines, verl GKD recipe). PROGRESS.md and CLAUDE.md initially had "off-policy" as a typo; chain-of-error caused the first cuts of `precompute_teacher.py` and `losses.py` to be built for off-policy. Both are now reframed as smoke baselines; on-policy v1 design is the new canonical (see `on_policy_v1_design.md`).
- **vLLM caps `prompt_logprobs` at `max_logprobs` (default 20)**. For K=50 (our E0 default), engine must be constructed with `max_logprobs=50`. Bake into teacher serving config.
- **HPC-X clusters break triton ldconfig** (Q5 in `docs/migrate-env.md`). `activate.sh.template` now detects + warns. Dev box 1 is affected; dev box 3 is not.
- **ViRL39K answer canonicalization**: `\boxed{4 - 3 = 1}` style answers are common (~89/12K filtered out had no `\boxed{}` at all; rest are mostly well-formed). E1's `verifier_pass` extracts trailing number/letter — passes 12/12 local unit tests including math expression vs digit match.

---

## Git history (key commits, oldest → newest)

| Commit | What it did |
|---|---|
| `d7af8a3a` | Initial commit: experiment repo skeleton with verl submodule. |
| `41050aa3` | NGC server env conventions docs (`activate.sh.template`, three-layer commit workflow). |
| `d5113404` | E0 plan locked in CLAUDE.md (datasets, models, metrics, go/kill). |
| `909c19d4` | E0 forward-only diagnostic skeleton. |
| `e749ca99` | Fix `torch_dtype` deprecation + per-topic VLMBias gain_margin in metrics. |
| `4fbbecc6` | Rich "Conditional GO" verdict + per-topic top-token dedup + analysis/e0_report.py. |
| `c9615363` | E0.3 fixes: 5a answer-extraction bug fix + per-topic 5a + chance baseline. |
| `403ede29` | Added PROGRESS.md / NEXT.md handoff system. |
| `70aac08a` | E0.3-B (length-norm gain_margin) + E1 4-config swap (SFT → Correct-filtered OPD). |
| `7413dd54` | E1 mixture decisions locked + ViRL39K loader + verl backbone choice (FSDP). |
| `1234a472` | E1 `precompute_teacher.py` first cut + `losses.py` stub. |
| `f0ef23d9` | Fix E1 `precompute_teacher.py` cross-experiment import path. |
| `1d358661` | Fix: OPD = on-policy, not off-policy (typo across project docs). |
| `3c7ba382` | E1 pivot to on-policy v1 design; reframe off-policy code as smoke baseline. |
| `0afa00d6` | Add `on_policy_v1_design.md` to new-session reading list. |
| `eac27504` | E1 spike: vLLM multimodal + `prompt_logprobs=K` verification script. |
| `5bd282cf` | `docs/migrate-env.md` runbook; `activate.sh` detects triton ldconfig bug (Q5). |
| `f84fc00b` | Fix(spike): raise vLLM `max_logprobs` to match K. **← Day 1 HEAD** |
| `6ebd1e3e` | E1 Stage 2: on-policy v1 trainer (4-config KL/CE dispatch) — initial drop. |
| `58f34eee` / `58eedbbc` / `7de1225d` | Launcher + Hydra plumbing fixes: shift argv, pkg:// search path, primary-config searchpath. |
| `69f38dbb` | Widen multimodal prompt cap + disable overlong filter (runbook R4). |
| `a4d537a0` | 4 actor + 4 teacher default GPU layout. |
| `5cc8d11c` | `export PYTHONPATH=$PWD` in launcher for Ray-worker module resolution. |
| `7c9f2bf9` / `49120e1d` | Ray worker registration attempts (later abandoned — see `022c465f`). |
| `022c465f` | Switch to lazy loss registration in `DistillationLossConfig.__post_init__` + custom agent-loop manager (runbook R1). Bumps verl submodule to `2c118243`. |
| `73589a43` | `num_workers = NGPUS_PER_NODE` so small-batch smoke doesn't trip the agent-loop chunk assert. |
| `ba655176` | Dummy `compute_score` for `e1_*` data_sources (runbook R5). |
| `80e45a6d` | Expose `DeltaOPDAgentLoop` as a module-level `_target_` (runbook R2). |
| **`b58932b7`** | Default `TORCH_CUDNN_V8_API_DISABLED=1` in launcher (runbook R3). **← Config A smoke passes here.** |

---

## Where things live

| Topic | File |
|---|---|
| Project-level instructions, conventions, env setup, datasets | `CLAUDE.md` |
| This file — what's been done | `PROGRESS.md` |
| What's next, decisions to make | `NEXT.md` |
| **Environment troubleshooting runbook (Q1-Q5)** | `docs/migrate-env.md` |
| **E1 smoke integration runbook (R1-R5)** | `docs/e1_smoke_runbook.md` |
| Canonical E0 verdict | `experiments/E0_image_null_delta/results/e0_verdict.md` |
| Detailed metric table | `experiments/E0_image_null_delta/results/e0_summary.csv` |
| Token category data for metric 4 | `experiments/E0_image_null_delta/results/top_delta_tokens.json` |
| Figures (if matplotlib installed locally) | `experiments/E0_image_null_delta/results/figures/` |
| E1 design / 4-config matrix / mixture / engineering punch list | `experiments/E1_filtered_delta_opd/README.md` |
| **E1 on-policy v1 trainer design (canonical)** | `experiments/E1_filtered_delta_opd/on_policy_v1_design.md` |
| E1 bucket-1 loader (ViRL39K) | `experiments/E1_filtered_delta_opd/data/virl39k_loader.py` |
| **E1 bucket-2 builder (POPE-style on COCO train2017)** | `experiments/E1_filtered_delta_opd/data/pope_style_builder.py` |
| **E1 bucket-3a loader (TallyQA complex)** | `experiments/E1_filtered_delta_opd/data/tallyqa_loader.py` |
| **E1 bucket-3b builder (synthetic counterfactuals)** | `experiments/E1_filtered_delta_opd/data/synthesize_counterfactuals.py` |
| **E1 three-layer dedup check (image_id / pHash / CLIP)** | `experiments/E1_filtered_delta_opd/data/dedup_check.py` |
| **E1 8K mixture sampler** | `experiments/E1_filtered_delta_opd/data/mixture.py` |
| E1 offline precompute (`trajectory_pass` + gold + sample metadata, all 5 buckets) | `experiments/E1_filtered_delta_opd/src/precompute_teacher.py` |
| **E1 on-policy v1 trainer code (Stage 2)** | `experiments/E1_filtered_delta_opd/src/on_policy/` |
| **E1 multi-bucket parquet builder (jsonl → train.parquet)** | `experiments/E1_filtered_delta_opd/data/make_train_parquet.py` |
| **E1 recipe configs (A/B/C/D)** | `experiments/E1_filtered_delta_opd/configs/recipe_*.yaml` |
| **E1 smoke launcher** | `experiments/E1_filtered_delta_opd/scripts/run_e1_recipe_smoke.sh` |
| **E1 eval (Day 3 Step 1): TEI / Escape / VLMBias / POPE / MathVista** | `experiments/E1_filtered_delta_opd/src/eval_tei.py` |
| **E1 train-log extractor (Day 3 logging tool): log → CSV + MD + JSON** | `experiments/E1_filtered_delta_opd/scripts/extract_train_metrics.py` |
| **E1 synth-bucket gate (Day 3 Step 2): per-topic teacher diagnostic + PASS/FAIL** | `experiments/E1_filtered_delta_opd/scripts/synth_sanity.py` |
| Legacy: e1_v1 monitoring grep | `experiments/E1_filtered_delta_opd/scripts/show_e1_metrics.sh` (superseded by extract_train_metrics.py) |
| E1 off-policy smoke-baseline losses (NOT scientific results) | `experiments/E1_filtered_delta_opd/src/losses.py` |
| vLLM dual-forward verification script | `experiments/E1_filtered_delta_opd/src/spike_vllm_dual_forward.py` |
