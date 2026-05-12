# PROGRESS — what has been built and what we found
*Last updated: 2026-05-12 (evening, end of Day 1). Last commit: `f84fc00b`.*

> **For new Claude sessions:** read this file first, then `NEXT.md`, then `CLAUDE.md`. Read the latest `experiments/E0_image_null_delta/results/e0_verdict.md` for the canonical E0 findings; read `experiments/E1_filtered_delta_opd/on_policy_v1_design.md` for the canonical E1 training design.

---

## TL;DR

- **Project**: Delta-OPD — **on-policy** distillation for VLMs reweighted by per-token image-vs-null teacher KL.
- **Phase**: E0 **complete** (Conditional GO + E0.3-A/B done). E1 **Day 1 complete** — design pivoted to on-policy after external review, ViRL39K loader built and verified, verl FSDP integration spike done, vLLM dual-forward verified end-to-end on Qwen2.5-VL-32B. E1 Stage 2 (on-policy trainer code) is next.
- **E0 verdict**: **Conditional GO**. 3 of 5 primary criteria pass; failures (delta-correctness correlation, gain_margin on VLMBias) are *informative* and constrain E1 method choice.
- **E0.3-B finding (today)**: Length-normalized `gain_margin` confirms VLMBias per-topic direction is **stable** (no sign flips). Motivation is not a length-bias artifact.
- **E1 design (locked today)**: On-policy v1. 4 configs A/B/C/D = `VanillaKD` / `RawDeltaKD` / `FilteredKD` / `FilteredDeltaKD`. **C is the critical control** — without it any D > A gain is unattributable to delta. Loss = top-K sparse forward KL on student-rollout prefixes, optionally × normalized `delta_t` × `1[T_correct]` mask, with CE-on-gold branch for teacher-wrong samples.
- **E1 mixture (locked today, GPT-reviewed)**: 8K E1-mini = 4K ViRL39K (PassRate∈[0.3,0.9], single-image) + 1600 self-built POPE-style on COCO train (NOT official POPE — image leakage) + 2400 (1500 synth VLMBias-like + 900 TallyQA complex). Mandatory dedup pipeline pre-launch.
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
| **`f84fc00b`** | Fix(spike): raise vLLM `max_logprobs` to match K. **← Day 1 HEAD** |

---

## Where things live

| Topic | File |
|---|---|
| Project-level instructions, conventions, env setup, datasets | `CLAUDE.md` |
| This file — what's been done | `PROGRESS.md` |
| What's next, decisions to make | `NEXT.md` |
| **Environment troubleshooting runbook (Q1-Q5)** | `docs/migrate-env.md` |
| Canonical E0 verdict | `experiments/E0_image_null_delta/results/e0_verdict.md` |
| Detailed metric table | `experiments/E0_image_null_delta/results/e0_summary.csv` |
| Token category data for metric 4 | `experiments/E0_image_null_delta/results/top_delta_tokens.json` |
| Figures (if matplotlib installed locally) | `experiments/E0_image_null_delta/results/figures/` |
| E1 design / 4-config matrix / mixture / engineering punch list | `experiments/E1_filtered_delta_opd/README.md` |
| **E1 on-policy v1 trainer design (canonical for Stage 2)** | `experiments/E1_filtered_delta_opd/on_policy_v1_design.md` |
| E1 bucket-1 loader (ViRL39K) | `experiments/E1_filtered_delta_opd/data/virl39k_loader.py` |
| E1 offline precompute (smoke + sample-level signals) | `experiments/E1_filtered_delta_opd/src/precompute_teacher.py` |
| E1 smoke-baseline losses (NOT scientific results) | `experiments/E1_filtered_delta_opd/src/losses.py` |
| vLLM dual-forward verification script | `experiments/E1_filtered_delta_opd/src/spike_vllm_dual_forward.py` |
