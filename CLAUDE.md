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

## Environment setup (NGC machine specifics)

The remote machine is based on an **NVIDIA NGC PyTorch image**. It has two hidden traps that have repeatedly broken `pip install`:

1. **`PIP_CONSTRAINT=/etc/pip/constraint.txt`** is set by the image and pins `torch` / `triton` / etc. to NGC versions. Standard vLLM-compatible `torch` install fails or silently grabs the wrong version until this is `unset`.
2. **`/usr/local/lib/python3.12/dist-packages/torch/`** is a customized NGC torch (`2.8.0a0+nv25.6`). It leaks into PEP 517 build-isolation envs and links compiled extensions (TransformerEngine, flash-attn) against the wrong ABI → `undefined symbol` at import time. The fix is to install those extensions with `--no-build-isolation` inside the project venv.

Both traps are dealt with in `activate.sh`. **Before doing anything on a fresh machine**:

```bash
cp activate.sh.template activate.sh
# edit activate.sh → set OPDEXP_FAST_ROOT (and OPDEXP_VENV if your venv lives elsewhere)
source activate.sh
```

After that, the venv is active and `PIP_CONSTRAINT` / `PIP_CONFIG_FILE` are neutralized for this shell.

### Installing / re-installing packages — the `--no-deps` rule

> **All `pip install -e` and `uv pip install -e` on this machine must pass `--no-deps`.** No exceptions unless the user explicitly says "update dependencies".

```bash
# install verl as editable (correct)
uv pip install --no-deps -e ./verl
```

Why: `verl` declares vLLM / torch / TransformerEngine / flash-attn as dependencies. Without `--no-deps`, uv re-resolves and reinstalls those, which silently overwrites the hand-built TransformerEngine binary. **Rebuilding TE takes 30–40 minutes** and you don't want to repeat it because of a careless install.

If you ever need to rebuild TransformerEngine from scratch:

```bash
pip install --no-deps --no-build-isolation -v \
    git+https://github.com/NVIDIA/TransformerEngine.git@v2.6
```

Both `--no-deps` and `--no-build-isolation` are required: the former protects existing deps, the latter prevents the build env from pulling in the NGC system torch.

### Known pitfalls (already mitigated, listed for memory)

- **`huggingface-hub` auto-upgrades to 1.x** during transitive installs, but `transformers 4.56.1` requires `<1.0`. Pin: `pip install "huggingface-hub>=0.34.0,<1.0"`.
- **verl's install script downloads a `flash_attn-*.whl` into the cwd** as a side effect. Safe to `rm` it after install.
- **`pip check` warns `decord 0.6.0 not supported on this platform`** — ignore. verl uses decord on a video path we don't exercise.
- **Don't touch the `.venv` symlink layout on the server.** It's machine-specific and not part of this repo's contract.

### Verified versions (snapshot — bump only when intentional)

`torch 2.8.0+cu128`, `vllm 0.11.0`, `TransformerEngine 2.6.0+c90a7207`, `megatron-core 0.13.1`, `flash-attn 2.8.1`, `flashinfer 0.3.1`, `numpy 1.26.4` (held below 2.0). Full freeze on the server at `/root/shihao_project/env-snapshots/opd-exp-freeze-20260511.txt`.

### Do not automate this

Environment setup is **documented, not scripted**. Past attempts to wrap it in a single `setup.sh` kept regressing when one dependency version drifted and triggered TE rebuilds. If something needs to change, edit this section of CLAUDE.md, not `activate.sh`.

## Modifying verl or recipe (three-layer commit workflow)

`verl/` is a submodule, and `verl/recipe/` is a nested submodule inside it. Editing code in either of these triggers a chain of commits:

| Where you edit | Repo that owns it | Push target | Then in parent |
|---|---|---|---|
| `verl/recipe/gkd/*.py` (or any `verl/recipe/...`) | `verl-project/verl-recipe` (fork to `shihaohou/verl-recipe` first if you'll push) | the recipe fork | `verl/` records new recipe submodule SHA |
| `verl/...` (not under recipe) | `shihaohou/verl` | `shihaohou/verl` | `opd-exp/` records new verl submodule SHA |
| `experiments/...`, `CLAUDE.md`, etc. | `shihaohou/opd-exp` | `shihaohou/opd-exp` | — |

Concretely for a recipe edit:

```bash
# (one-time) ensure verl/recipe points to your fork
cd verl/recipe
git remote set-url origin git@github.com:shihaohou/verl-recipe.git   # or HTTPS
git checkout -b my-feature
# ... edit ...
git commit -m "..." && git push origin my-feature

cd ..                                   # back to verl/
git add recipe                          # records the new recipe SHA
git commit -m "Bump recipe to <sha>" && git push origin <branch>

cd ..                                   # back to opd-exp/
git add verl                            # records the new verl SHA
git commit -m "Bump verl: <reason>" && git push
```

For a plain `verl/` edit (not under `recipe/`), drop the inner-most step.

If a request requires changes to verl source, confirm with the user before editing inside `verl/` — those changes flow to the fork and ripple up two submodule pointers.

## Datasets (E0)

All three are pre-downloaded under `/home/web_server/antispam/project/houshihao/datasets/` on the remote machine, in HF `Dataset.save_to_disk()` arrow format. Schemas confirmed against the HF dataset cards.

### 1. VLMBias — primary battleground

- Local path: `datasets/VLMBias/` (a `DatasetDict` with sub-configs: `main`, `identification`, `withtitle`, `original`, `remove_background_q1q2`, `remove_background_q3`)
- HF: [`anvo25/vlms-are-biased`](https://huggingface.co/datasets/anvo25/vlms-are-biased)
- **For E0 we use `main` (2,780 rows) as the primary subset.** The `withtitle` / `remove_background_*` subsets are for Step 2 sensitivity ablation.
- Key columns:
  - `image` — PIL image
  - `prompt` — question text (already includes "Answer in curly brackets, e.g., {Yes} or {No}.")
  - `ground_truth` — gold answer (e.g. "Yes")
  - **`expected_bias`** — the biased wrong answer (e.g. "No"). **Use this for metric 3** (`gain(correct) > gain(biased_wrong)`).
  - `topic` / `sub_topic` — category labels for subgroup acc breakdowns
  - `type_of_question` — Q1 / Q2 / Q3 variant
- Budget for E0: use all 2,780 rows of `main`.

### 2. POPE-adversarial — hallucination floor

- Local path: `datasets/POPE-adversarial/` (single arrow file — already the adversarial split, no need to filter on `category`)
- HF: [`lmms-lab/POPE`](https://huggingface.co/datasets/lmms-lab/POPE) with `category == "adversarial"` (3,000 rows in the upstream split; verify local row count)
- Key columns:
  - `image` — PIL image
  - `question` — yes/no question (e.g. "Is there a snowboard in the image?")
  - `answer` — gold "yes" or "no"
  - `category` — should be all "adversarial" since we have the pre-filtered split
- **Hallucination/grounded classification:** when model answers `yes`:
  - gold=`yes` → **grounded** (correct positive)
  - gold=`no` → **hallucinated** (false positive)
- Budget for E0: 1,000 samples (random first 1,000, fixed seed).

### 3. MathVista-mini — visual reasoning sanity

- Local path: `datasets/MathVista-mini/` (single arrow file — already the `testmini` split)
- HF: [`AI4Math/MathVista`](https://huggingface.co/datasets/AI4Math/MathVista) `testmini` (1,000 rows)
- Key columns:
  - **`decoded_image`** — PIL image (NOT `image` — that column is a filename string, easy footgun)
  - `question` — text
  - `choices` — list of strings for `multi_choice`, empty/None for `free_form`
  - `answer` — gold answer (full choice text for multi_choice, not a letter)
  - `question_type` — `multi_choice` or `free_form`
  - `metadata` — dict with `task`, `category`, `skills`, `grade`, `source`
- Budget for E0: 500 samples (random first 500 of testmini, fixed seed).

(MMMU and ScienceQA are *not* in the E0 scope. They show up in E1 evaluation.)

## E0: Image-vs-Null Teacher Delta Diagnostic

**Forward-only. No training.** Step 0 of the Delta-OPD plan: verify that image-vs-null KL can distinguish visual-evidence-driven tokens from language-prior-driven tokens. If this doesn't hold, don't train — pivot to claim-gated OPD instead.

### Backend

- **HF transformers + bf16 + flash-attn-2** for E0. Slow but per-token logits / forced scoring are one-liners and easy to trust.
- vLLM is reserved for E1 rollout. Do not use vLLM in E0 — its prompt-logprobs path is awkward for forced scoring and complicates debugging.

### Models

- **Primary teacher (E0 default)**: Qwen2.5-VL-32B-Instruct — full pass on all three datasets.
- **Sanity teacher**: Qwen2.5-VL-72B-Instruct — 200–300 sample sanity check only. Do **not** launch a 72B full run until the 32B pass shows signal.
- **Student**: Qwen2.5-VL-7B-Instruct — generated answers used only for metric 5a (student/teacher wrong overlap). Student does **not** get forced scoring in E0.

### Null image

For E0 we use **only two** null modes (other masks — Gaussian, patch shuffle, irrelevant image — are Step 2):

1. **`black`** — all-black PIL image, resized to match the real image per sample. This is the default and always works.
2. **`image_drop`** — omit the image entirely from the prompt, if the Qwen2.5-VL processor accepts it cleanly. If image_drop changes the prompt format in a way that introduces an obvious confound (e.g. the `<image>` token still required), fall back to `black` only and note it.

The YAML config controls `null_modes` — start with `[black]` for first run, add `image_drop` once verified on server.

### Core procedure

For each sample `(x, I, gold_answer)`:

```
# generate teacher response with image (greedy)
ans_T_I  = teacher.generate(x, I,    greedy)
# generate student response with image (for metric 5a)
ans_S_I  = student.generate(x, I,    greedy)
# (also record teacher answer under null for metric 1)
ans_T_null = teacher.generate(x, null, greedy)
```

Then forced-score teacher's own image-conditioned response `ans_T_I` token-by-token under both conditions:

```
logp_I[t]    = log p_T(y_t | x, I,    y_<t)
logp_null[t] = log p_T(y_t | x, null, y_<t)
delta_t      = KL_top50( p_T(. | x, I, y_<t),  p_T(. | x, null, y_<t) )
cmi_t        = logp_I[t] - logp_null[t]
```

Per-sample jsonl record: dataset, sample_id, question, gold, ans_T_I, ans_T_null, ans_S_I, correctness flags (each), full `delta_t[]`, `logp_I[t]`, `logp_null[t]`, token strings, and dataset-specific fields (VLMBias `expected_bias`/`topic`/`sub_topic`, POPE `answer`, MathVista `question_type`/`metadata.task`).

### Metrics & go/kill

| # | Metric | Dataset | Want to see |
|---|---|---|---|
| 1 | `Acc(T, image)` vs `Acc(T, null)` | VLMBias | `Acc(T, image) > Acc(T, null)` by meaningful margin |
| 2 | `mean_delta(correct)` vs `mean_delta(wrong)`, plus Spearman(mean_delta, correctness) | VLMBias | correct samples have higher mean delta; Spearman significantly positive |
| 3 | Answer-level visual gain: `log P(ground_truth \| I) - log P(ground_truth \| null)` vs same for `expected_bias` | VLMBias (uses `expected_bias` column) | `gain(ground_truth) > gain(expected_bias)` |
| 4 | Top-delta tokens — qualitative category labelling | All datasets, hand-inspect ~50 | Object / number / color / spatial / yes-no words dominate over connectives, templates, formatting |
| 5a | Student-teacher wrong-overlap (same biased answer rate) | VLMBias | Non-trivial overlap — flags Delta-OPD's *need*, not OPD's failure |
| 5b | `mean_delta(hallucinated)` vs `mean_delta(grounded)` | POPE-adv (when model answers "yes") | hallucinated < grounded — image evidence weaker on hallucinations |

**MathVista-mini** is a sanity check for visual reasoning retention, not a hypothesis-test dataset for E0; we record acc & per-trajectory mean_delta for tracking, not for go/kill.

**Go criterion (proceed to E1 training)**: at least **2 of 5** primary criteria satisfied:
- (1) `Acc(T, image) > Acc(T, null)` by meaningful margin on VLMBias
- (2) trajectory-mean `delta_t` significantly correlates with correctness on VLMBias
- (3) `gain(ground_truth) > gain(expected_bias)` on VLMBias
- (4) high-delta tokens are vision-bearing on qualitative inspection
- (5b) hallucinated samples have lower mean_delta than grounded on POPE-adv

**Kill criterion (pivot to claim-gated OPD)**: 0–1 of the above hold; *or* teacher accuracy is image-invariant on VLMBias (signal source absent at the root); *or* high-delta tokens look random.

### First-day budget

- 32B teacher: full E0 pass on all three datasets (VLMBias `main` 2780 + POPE-adv 1000 + MathVista testmini 500).
- 72B teacher: 200–300 sample sanity only — first 200 of VLMBias `main`. Do **not** launch a full 72B run on day 1.
- Student 7B: same coverage as 32B teacher (greedy, image-conditioned only, for metric 5a).

### Deliverables (what comes out of E0)

In `experiments/E0_image_null_delta/results/`:
- `e0_teacher32b_vlmbias.jsonl` (and equivalents for pope/mathvista, student, 72B-sanity)
- `e0_summary.csv` — one row per (model, dataset), columns = the 5 metrics
- Figures from `analysis/e0_report.ipynb`:
  - Fig 1: VLMBias `mean_delta` distribution split by correctness
  - Fig 2: Top-delta-token histogram by hand-labeled category
  - Fig 3: POPE `mean_delta` distribution split by hallucinated vs grounded
- A short markdown verdict at `results/e0_verdict.md` (one paragraph: go / kill / which criteria passed).

## Long-term codebase plan

The diagnostic pipeline (`experiments/E0_image_null_delta/src/`) is meant to be reusable: every training checkpoint in later steps will run the same dual-forward diagnostic to track how `mean_delta`, visual gain, and student-teacher overlap evolve. Step 2's mask sensitivity ablation (black / gaussian / shuffle / irrelevant image) also goes through this layer. Keep it decoupled from verl's training loop — it should work on any HF VLM checkpoint.

## Conventions

- **No code written yet.** When asked to implement, default to small, focused modules under `experiments/E0_image_null_delta/src/`. Don't generate Python files until the user asks for a specific component.
- **Results are gitignored** (`experiments/*/results/**`). Save jsonl + summary CSV there, never check in.
- **Greedy decoding** for both teacher generation and forced scoring throughout E0. No sampling.
- **Top-50 KL** for `delta_t` (truncate the support to top-50 of `p_T(.|x,I,y_<t)` ∪ top-50 of `p_T(.|x,null,y_<t)`, renormalize, then KL). Avoids long-tail noise.
- **Null image** = a single all-black PIL image of the same resolution as the real image (resize to match per sample). Cache the encoder output if it speeds things up.
- **Editable installs always with `--no-deps`** on the server (see *Environment setup* for the rationale — TE binary protection).
- **No automated environment setup.** If a tool / install / patch needs to be applied to the server, document the command in this file and let the human run it. Do not generate `setup.sh` or wire it into CI.
- **Don't touch the `.venv` symlink layout** on the server — it's machine-specific.
