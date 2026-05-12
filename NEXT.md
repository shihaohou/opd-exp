# NEXT — what to do next
*Last updated: 2026-05-12 (morning). Pair this file with `PROGRESS.md`.*

> **For new Claude sessions:** if you're starting fresh, first read `CLAUDE.md`, then `PROGRESS.md` (what we did), then this file. Recommended next concrete action is at **§ Right now, today** below.

---

## Status overview

```
E0 (forward-only diagnostic)   : DONE — Conditional GO
E0.x (optional cleanup)        : pending, low priority
E1 (training)                  : design doc exists, code NOT started
E2 (mask sensitivity ablation) : not started
```

---

## Right now, today (the literal next thing to do)

1. **Verify the overnight ViRL39K download on the server.**

   ```bash
   ssh into arc-wlf1-ge103-4
   tmux a   # or new pane

   cd /home/web_server/antispam/project/houshihao/datasets
   ls -lah ViRL39K/
   tail -20 viRL39K_download.log

   # If huggingface-cli download succeeded, you should see arrow files / metadata.
   # If it failed (auth issue, disk full, etc.), restart with the same command:
   #   nohup huggingface-cli download TIGER-Lab/ViRL39K \
   #       --repo-type dataset --local-dir ./ViRL39K \
   #       > viRL39K_download.log 2>&1 &
   ```

2. **Inspect ViRL39K schema** so we know what the loader needs to do:

   ```bash
   python <<'PY'
   from datasets import load_from_disk
   ds = load_from_disk("/home/web_server/antispam/project/houshihao/datasets/ViRL39K")
   print(type(ds), ds)
   first = ds[0] if hasattr(ds, "__getitem__") else next(iter(ds.values()))[0]
   print(list(first.keys()))
   print({k: type(v).__name__ for k, v in first.items()})
   PY
   ```

   Likely keys (from TIGER-Lab/ViRL39K HF card, **verify**): question, image, answer, source. Confirm before writing the loader.

3. **Decide verl recipe entry point** for E1. See `experiments/E1_filtered_delta_opd/README.md` for context. Open options:

   - `verl/recipe/gkd/` — general knowledge distillation; closest match conceptually.
   - `verl/trainer/distillation/{fsdp,megatron}/losses.py` — primitive reverse-KL utilities.
   - Roll a thin custom trainer wrapping `verl.trainer.sft_trainer`.

   Spike each for 30 min, then commit to one. Don't write significant code without picking one first.

---

## E0.x — pending diagnostic cleanup (deferred, can do in parallel with E1 setup)

Priority is **low** — none of these change the Conditional GO verdict. Do them if the eval framework for E1 needs them anyway.

| Task | Cost | Why |
|---|---|---|
| **Length-normalized option scoring** | Add Qwen tokenizer load in `metrics.py`; ~30 lines; no rerun needed. | Eliminates length bias in `gain_margin`. Will be reused as-is in E1 eval. |
| **PPL_S(teacher_wrong_response \| x, I)** | New ~15 min server run; new script `e0_ppl_student.py`. | Distribution-level overlap proxy. More OPD-faithful than answer-overlap. Useful as an E0 baseline before E1 trained students. |
| **72B teacher sanity** | `bash experiments/E0_image_null_delta/scripts/run_e0_teacher72b_sanity.sh`; ~20 min on 2 H800. | Tells us if the failure modes scale away with larger teacher (probably no) or are property of architecture (probably yes). |
| **Per-topic top tokens validation** | Hand-inspect `top_delta_tokens.json["vlmbias_by_topic"]`; Logos has anomalous `" on"` token, decide if signal or noise. | Confirms metric 4 quality across topics, not just globally. |

---

## E1 — Filtered Delta-OPD training (the main work going forward)

**Read `experiments/E1_filtered_delta_opd/README.md` for the design.** That's the source of truth for the 4-config ablation, training-data composition, loss form, evaluation matrix, and engineering punch list. This section is the *workflow* on top of it.

### 4-config matrix (recap from design doc)

| Config | What | Role |
|---|---|---|
| A. `sft` | `L = −log p_S(y_T \| x, I)` on teacher outputs | Imitation baseline |
| B. `vanilla_opd` | Student rollout + teacher reverse-KL full-prefix | Existing-OPD baseline |
| C. `raw_delta_opd` | `Σ_t delta_t · KL(p_T \|\| p_S)` no filtering | **Negative control** — tests "image influence alone is insufficient" |
| D. `filtered_delta_opd` | Same as C, but `delta_t` zeroed on teacher-wrong trajectories + CE on gold answer tokens | **Primary candidate** |

Compute-budget fallback: drop SFT first, keep B+C+D.

### Day-by-day (rough)

| Day | Goal |
|---|---|
| **Today (Day 1)** | ViRL39K verified + verl recipe picked + multimodal loader for ViRL39K + outline of `precompute_teacher.py`. |
| Day 2 | Implement `precompute_teacher.py`; run on a 500-sample slice to sanity-check teacher correctness rate; if <30% correct, ViRL39K isn't the right primary bucket. |
| Day 3 | Implement SFT trainer (or thin wrapper); smoke-test on 1K samples, 100 steps. |
| Day 4 | Implement Vanilla OPD; Raw Delta-OPD; Filtered Delta-OPD. Eval hooks. |
| Day 5 | Full 4-config sweep on ~8K-sample E1-mini; first eval. |
| Day 6 | Decide v2 hyperparameters. |

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

## Open questions that need decisions BEFORE writing significant E1 code

| # | Question | How to decide |
|---|---|---|
| 1 | verl recipe entry point: `recipe/gkd/` vs thin wrapper? | Spike each for 30 min. Prefer GKD if it already has multimodal data plumbing wired. |
| 2 | Online teacher forward vs precomputed teacher logits? | If memory allows 7B+32B on 8 H800 → online. If not → precompute top-K teacher logits + delta_t per training sample once, cache to disk, replay during training. |
| 3 | ViRL39K starter subsample size? | 8K for E1-mini. Scale later. |
| 4 | Where to find adversarial-recognition counterfactuals (NOT VLMBias eval)? | Look at TallyQA, TangramQA, possibly synthesize custom modified-object samples. Decide after Day 1 spike. |
| 5 | `λ_ans` (CE weight on gold for filtered Delta-OPD when teacher is wrong)? | Start at 1.0 — same scale as KL. Sweep in v2. |
| 6 | Top-K for KL in training (vs E0's 50)? | 50 by default for continuity, but consider 100 to reduce truncation bias. Document the choice. |

---

## Hard rules — **DO NOT**

1. **Don't launch training without a recipe decision** (open question #1). You'll waste GPU hours debugging integration.
2. **Don't generate teacher data on the VLMBias eval set.** That's evaluation, not training. Use held-out adversarial-recognition data only.
3. **Don't change the null mode** (still only `black`) until E2 ablation. Changing it mid-experiment poisons cross-run comparisons.
4. **Don't `pip install -e` anything without `--no-deps`** on the server. Rebuilding TransformerEngine costs 30–40 minutes.
5. **Don't touch `.venv` symlink** on the server.
6. **Don't share the server** with other tenants during E1 training. Coordinate.
7. **Don't trust same-wrong overlap rate as a primary signal** — it conflates baseline shared prior with potential distillation effect. Use the TEI metric family instead.

---

## Pending coordination items

- Confirm with operations team / `gemma-4-31B vllm serve` owner: when does that long-running serve come down? (Currently 19+ days running on the same box.)
- Confirm `Qwen3-VL-8B-Instruct` download (PID 164459 in earlier ps output) — is someone going to use it on the same box?
- If E1 training needs dedicated GPUs, schedule a window.

---

## How to resume a fresh Claude session

1. Open this repo.
2. Read in order: `CLAUDE.md` → `PROGRESS.md` → `NEXT.md`.
3. Skim `experiments/E0_image_null_delta/results/e0_verdict.md` for the canonical numbers.
4. Open `experiments/E1_filtered_delta_opd/README.md` for the E1 design.
5. Check `git log --oneline -10` to confirm where we left off (current HEAD: `c9615363` as of this writing).
6. Pick up at the next un-checked item in **§ Right now, today** above.
