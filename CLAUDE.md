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
- **Editable installs always with `--no-deps`** on the server (see *Environment setup* for the rationale — TE binary protection).
- **No automated environment setup.** If a tool / install / patch needs to be applied to the server, document the command in this file and let the human run it. Do not generate `setup.sh` or wire it into CI.
- **Don't touch the `.venv` symlink layout** on the server — it's machine-specific.
