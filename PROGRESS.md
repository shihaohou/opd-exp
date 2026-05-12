# PROGRESS — what has been built and what we found
*Last updated: 2026-05-12 (morning). Last commit: `c9615363`.*

> **For new Claude sessions:** read this file first, then `NEXT.md`, then `CLAUDE.md`. Read the latest `experiments/E0_image_null_delta/results/e0_verdict.md` for the canonical scientific findings. This file is the prose narrative — verdict.md is the data.

---

## TL;DR

- **Project**: Delta-OPD — on-policy distillation for VLMs reweighted by per-token image-vs-null teacher KL.
- **Phase**: E0 (forward-only diagnostic) **complete**. E1 (training) **not yet started**, design doc exists at `experiments/E1_filtered_delta_opd/README.md`.
- **E0 verdict**: **Conditional GO**. 3 of 5 primary criteria pass; the 2 that fail (delta-correctness correlation, gain_margin on VLMBias) fail in an *informative* way that constrains E1 method choice.
- **Headline**: `delta_t` faithfully tracks image influence, but the *direction* of that influence is task-dependent — it's adversarial on VLMBias recognition topics. So the right E1 recipe is **Filtered Delta-OPD** (weight only on teacher-correct trajectories), with Raw Delta-OPD as negative-control ablation.

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

Two independent passes by GPT (one after 32B-only, one after 7B added). Both converged on the same verdict as Claude. Genuine value-adds:

- **Teacher-Error Inheritance (TEI) metric family** for E1: `Acc_S | T_wrong`, TEI rate, Escape rate. Codified in `experiments/E1_filtered_delta_opd/README.md`.
- **Chance baseline** framing for 5a. Drove the per-topic table.
- **Filtered Delta-OPD cannot fix Animals alone** (teacher acc = 0/546 → no positives to weight) — must combine with ground-truth CE.
- **PPL-based overlap metrics** as complement to answer-overlap (deferred to E1 eval).

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

- **Option scoring is raw `sum(log p)`** over the option's tokens. When `ground_truth` and `expected_bias` tokenize to different lengths, `gain_margin` has a length bias. For VLMBias `main` the strings are short (single-word / digit), so this is plausibly 2nd-order — but real. Fix scheduled for E1 baseline.
- **All-black null image** could itself be OOD for the vision encoder, inflating KL. Step-2 ablation will compare with `image_drop`, Gaussian noise, and an unrelated-but-natural image.
- **PPL-based student metrics** not computed yet (would need another server run, ~15 min on 7B).
- **72B teacher sanity** not run yet.

---

## Git history (key commits, oldest → newest)

| Commit | What it did |
|---|---|
| `d7af8a3a` | Initial commit: experiment repo skeleton with verl submodule. |
| `41050aa3` | NGC server env conventions docs (`activate.sh.template`, three-layer commit workflow). |
| `d5113404` | E0 plan locked in CLAUDE.md (datasets, models, metrics, go/kill). |
| `909c19d4` | E0 forward-only diagnostic skeleton (configs + loaders + dual_forward + metrics + scripts). |
| `e749ca99` | Fix `torch_dtype` deprecation + per-topic VLMBias gain_margin in metrics. |
| `4fbbecc6` | Rich "Conditional GO" verdict + per-topic top-token dedup + analysis/e0_report.py. |
| **`c9615363`** | E0.3 fixes: 5a answer-extraction bug fix + per-topic 5a + chance baseline + 5a wording softened + E1 design doc. **← current HEAD** |

---

## Where things live

| Topic | File |
|---|---|
| Project-level instructions, conventions, env setup, datasets | `CLAUDE.md` |
| This file — what's been done | `PROGRESS.md` |
| What's next, decisions to make | `NEXT.md` |
| Canonical scientific verdict | `experiments/E0_image_null_delta/results/e0_verdict.md` |
| Detailed metric table | `experiments/E0_image_null_delta/results/e0_summary.csv` |
| Token category data for metric 4 | `experiments/E0_image_null_delta/results/top_delta_tokens.json` |
| Figures (if matplotlib installed locally) | `experiments/E0_image_null_delta/results/figures/` |
| E1 plan / design / open questions | `experiments/E1_filtered_delta_opd/README.md` |
