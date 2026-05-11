# E0 — Image-vs-Null Teacher Delta Diagnostic

**Status**: skeleton only, no code yet.
**Type**: forward-only diagnostic, no training, no gradient updates.
**Purpose**: validate that per-token *image-vs-null* KL on the teacher tracks visual contribution before committing to Step 1 training in the Delta-OPD plan.

See the project-level [`CLAUDE.md`](../../CLAUDE.md) for the full E0 spec (procedure, metrics, go/no-go criteria, model & data paths on the remote machine).

## Directory layout

```
E0_image_null_delta/
├── configs/      # YAML configs: model paths, dataset paths, sample budgets
├── data/         # dataset loaders (VLMBias, POPE-adv, MathVista-mini)
├── src/          # core: dual_forward, generation, metrics, token_attribution
├── scripts/      # bash entrypoints to launch runs on the remote machine
├── analysis/     # notebooks producing go/no-go figures
└── results/      # jsonl + CSV outputs (gitignored)
```

## Quick model/data reference

Runs on remote machine `arc-wlf1-ge103-4`:

| role | model | path |
|---|---|---|
| Teacher-1 | Qwen2.5-VL-32B-Instruct | `/home/web_server/antispam/project/houshihao/models/Qwen2.5-VL-32B-Instruct` |
| Teacher-2 (sanity 200–300 samples only) | Qwen2.5-VL-72B-Instruct | `/home/web_server/antispam/project/houshihao/models/Qwen2.5-VL-72B-Instruct` |
| Student | Qwen2.5-VL-7B-Instruct | `/home/web_server/antispam/project/houshihao/models/Qwen2.5-VL-7B-Instruct` |

Datasets:

| dataset | samples | rationale |
|---|---|---|
| VLMBias | 500–1000 | main hypothesis (language prior vs visual evidence) |
| POPE-adversarial | 1000 | adversarial hallucination |
| MathVista-mini | 500 | visual reasoning sanity |

## Go/no-go decision

Step 1 (training) is gated on E0. The detailed criteria are in [`CLAUDE.md`](../../CLAUDE.md); the short version is:

- ✅ `Acc(T, image) > Acc(T, null)` on VLMBias by a meaningful margin
- ✅ trajectory-mean `delta_t` significantly positively correlated with answer correctness
- ✅ top-delta tokens land on vision-bearing tokens, not formatting
- ✅ non-trivial student-teacher wrong-overlap on VLMBias (justifies the masking)

If any of these fail, do **not** proceed to Step 1.
