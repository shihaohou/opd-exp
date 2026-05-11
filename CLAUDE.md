# Agent instructions — opd-exp

This is the **experiment workspace** for Delta-OPD (off-policy distillation for VLMs guided by per-token image-vs-null teacher delta). It is *not* the verl framework repo; verl is a git submodule at `verl/`.

## Repo layout

```
opd-exp/
├── verl/                              # submodule -> shihaohou/verl (a fork of verl-project/verl)
├── experiments/
│   └── E0_image_null_delta/           # current focus: forward-only diagnostic
└── (top-level docs, .gitignore, this file)
```

Treat `verl/` as third-party code that we both **use as a library** and may **modify**. When you modify verl source, commit inside `verl/` first (which pushes to `shihaohou/verl`), then bump the submodule pointer in the parent repo.

## Execution environment

- **Local machine** (this directory): macOS, used for editing code, planning, light analysis. No GPUs, no model weights.
- **Remote machine**: `arc-wlf1-ge103-4` (Linux, GPUs). All model forward / generation runs here. Datasets and model weights live here.
  - Models root: `/home/web_server/antispam/project/houshihao/models/`
    - `Qwen2.5-VL-7B-Instruct` (student)
    - `Qwen2.5-VL-32B-Instruct` (primary teacher)
    - `Qwen2.5-VL-72B-Instruct` (optional strong-teacher check)
    - `Qwen2.5-0.5B-Instruct`, `Qwen2.5-1.5B-Instruct`, `Qwen2.5-72B-Instruct` (text-only, not used in E0)

When the user asks to "run" a script, assume they will run it on the remote machine themselves — do **not** invent local Bash commands that touch model weights. Generate the script and let them execute.

## Datasets (E0)

On the remote machine. Three datasets, in priority order:

1. **VLMBias** — main hypothesis: does language prior override visual evidence? Use 500–1000 samples (or full).
2. **POPE-adversarial** — object hallucination under adversarial negatives. 1000 samples.
3. **MathVista-mini** — visual reasoning sanity. 500 samples.

(MMMU and ScienceQA are *not* in the E0 scope. Stay focused.)

## E0: Image-vs-Null Teacher Delta Diagnostic

**Forward-only. No training.** Step 0 of the Delta-OPD plan: verify that image-vs-null KL can distinguish visual contribution before committing to Step 1 training.

### Core procedure

For each sample `(x, I, gold_answer)`:

```
p_T_I    = teacher.forward(x, I)
p_T_null = teacher.forward(x, black_image)
ans_T_I  = teacher.generate(x, I, greedy)
ans_S_I  = student.generate(x, I, greedy)
```

Then forced-score the teacher's own response token-by-token under both image and null conditions:

```
logp_I[t]    = log p_T(y_t | x, I,    y_<t)
logp_null[t] = log p_T(y_t | x, null, y_<t)

delta_t = KL_top50( p_T(. | x, I, y_<t),  p_T(. | x, null, y_<t) )
cmi_t   = logp_I[t] - logp_null[t]
```

Per-sample record (jsonl): question, image_id, gold, ans_T_I, ans_S_I, correctness flags, full `delta_t[]`, `logp_I[t]`, `logp_null[t]`, token list.

### Metrics & go/no-go

| # | Metric | Want to see | If not |
|---|---|---|---|
| 1 | `Acc(T, image)` vs `Acc(T, null)` on VLMBias | `Acc(T, image) > Acc(T, null)` | Teacher isn't using image → Delta-OPD has no basis. **Abort.** |
| 2 | `mean_delta(correct)` vs `mean_delta(wrong)` | correct samples have higher mean delta_t | Delta signal doesn't track visual reliance. **Reconsider.** |
| 3 | Answer-level visual gain: `log P(correct\|I) - log P(correct\|null)` vs same for biased-wrong option | `gain(correct) > gain(biased_wrong)` | Image input is reinforcing wrong prior too → Delta-OPD risk. |
| 4 | Token-level: top-delta tokens | They land on vision-bearing tokens (object names, numbers, yes/no), not formatting | Delta is noise. |
| 5 | Student-teacher wrong-overlap on VLMBias | Overlap high (justifies need for visual masking) | Vanilla OPD already fine. |

**Go criterion for Step 1**: on VLMBias, per-trajectory mean `delta_t` is significantly positively correlated with answer correctness, AND `Acc(T, image) > Acc(T, null)` by a meaningful margin.

### First-day budget

- 32B teacher: full E0 pass on all three datasets.
- 72B teacher: 200–300 sample sanity only. Do **not** launch a full 72B run on day 1.

## Long-term codebase plan

The diagnostic pipeline (`experiments/E0_image_null_delta/src/`) is meant to be reusable: every training checkpoint in later steps will run the same dual-forward diagnostic to track how `mean_delta`, visual gain, and student-teacher overlap evolve. Step 2's mask sensitivity ablation (black / gaussian / shuffle / irrelevant image) also goes through this layer. Keep it decoupled from verl's training loop — it should work on any HF VLM checkpoint.

## Conventions

- **No code written yet.** When asked to implement, default to small, focused modules under `experiments/E0_image_null_delta/src/`. Don't generate Python files until the user asks for a specific component.
- **Results are gitignored** (`experiments/*/results/**`). Save jsonl + summary CSV there, never check in.
- **Greedy decoding** for both teacher generation and forced scoring throughout E0. No sampling.
- **Top-50 KL** for `delta_t` (truncate the support to top-50 of `p_T(.|x,I,y_<t)` ∪ top-50 of `p_T(.|x,null,y_<t)`, renormalize, then KL). Avoids long-tail noise.
- **Null image** = a single all-black PIL image of the same resolution as the real image (resize to match per sample). Cache the encoder output if it speeds things up.

## When in doubt

If a request requires changes to verl source, confirm with the user before editing inside `verl/` — those changes flow to the `shihaohou/verl` fork and affect the submodule pointer in the parent repo.
