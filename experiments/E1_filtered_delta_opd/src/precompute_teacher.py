"""
Offline teacher pass for E1.

⚠️  Scope of usefulness:

This script was written in an early phase of E1 design when the loss form
was assumed to be off-policy weighted CE (student force-scores teacher's
greedy response). After realizing OPD = On-Policy Distillation, the real
E1 v1 training loop uses student rollouts and computes delta_t on the
**student's** rollout prefix, not on teacher's response. So the per-token
fields produced here (`teacher_logp_I`, `teacher_logp_null`, `delta_t`)
are NOT used by the on-policy v1 trainer.

What v1 DOES use from this output:
  - sample_id, question, gold, ans_T_I, correct_I (= trajectory_pass)
  - bucket / category / source / pass_rate_32b for monitoring
  - response_token_ids of teacher's greedy — useful as a cold-start SFT
    init or for a quick "weighted-SFT vs gold-CE" smoke baseline

What v1 does NOT use:
  - teacher_logp_I / teacher_logp_null / delta_t (these are on the wrong
    trajectory — must be recomputed at training time on the student's
    rollout prefix)

The per-token forced-score pass takes ~70% of the wall time per sample;
once v1 is wired in we can add a `--no-forced-score` mode that drops it
and reduces precompute cost ≈ 5×. For now the full output stays because
it is still consumed by:
  - the off-policy "weighted_sft" smoke baseline (`src/losses.py`)
  - E0-style sanity inspection of teacher's response

Per-sample flow:
  1. Greedy-generate a teacher response under (question, image).
  2. Forced-score the response under (question, image)        → logp_I, topK_I
  3. Forced-score the response under (question, null_image)   → logp_null, topK_null
  4. Compute per-token delta_t = KL_topK( p_T(.|x,I) || p_T(.|x,null) )
  5. Parse correctness vs gold to derive `trajectory_pass`.

Outputs one jsonl line per sample with everything the student trainer needs:

  sample_id, bucket, source, category, pass_rate_32b
  question, gold
  ans_T_I                       # teacher's generated response (text)
  correct_I                     # bool; vs gold
  trajectory_pass               # alias of correct_I for now (sample-level filter)
  response_token_ids            # the trajectory student force-scores against
  response_tokens               # decoded per-token (debug)
  teacher_logp_I[t]             # per-token teacher log-prob under image — REVERSE-KL TARGET
  teacher_logp_null[t]          # under null — kept for analysis / future use
  delta_t[t]                    # per-token KL-top50 between I and null distributions
  (optional) teacher_topk_log_probs_I, teacher_topk_token_ids_I   # if --keep-topk

This script intentionally reuses E0's `dual_forward.py` helpers
(greedy_generate, forced_score, kl_topk_union, prepare_null) by `sys.path`-
injecting `experiments/E0_image_null_delta/src/`. The logic for teacher
forward is identical to E0; only the input shape (E1's bucket-loader samples)
and the output record schema differ.

TODO once E2 (mask sensitivity ablation) starts: refactor these helpers into
`experiments/_common/teacher_forward.py` so both E0 and E1 import from one
location.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

# --- Repo-root injection ----------------------------------------------------
# E0's dual_forward uses absolute imports rooted at `experiments.E0_image_null_delta....`,
# which only resolve when the repo root is on sys.path. When this script is
# launched as `python experiments/E1_filtered_delta_opd/src/precompute_teacher.py`,
# Python adds only the script's directory (src/), not the repo root, to sys.path.
# Inject the repo root here so cross-experiment imports work.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --- E1 bucket loader registry ---------------------------------------------
# Each entry is a (loader_kwargs_dict, build_fn) — build_fn yields
# (sample_id: str, question: str, image_paths: list[Path], gold: str, extras: dict).
# Wrapping everything to a uniform tuple lets the precompute driver stay loader-
# agnostic.

def _virl39k_iter(
    dataset_root: str,
    pass_rate_min: Optional[float] = 0.3,
    pass_rate_max: Optional[float] = 0.9,
) -> Iterator[tuple[str, str, list[Path], str, dict]]:
    # Lazy import — avoids dragging pyarrow when the user only inspects this file.
    from experiments.E1_filtered_delta_opd.data.virl39k_loader import iter_virl39k

    for s in iter_virl39k(
        dataset_root=dataset_root,
        pass_rate_min=pass_rate_min,
        pass_rate_max=pass_rate_max,
        single_image_only=True,
        require_boxed=True,
    ):
        yield (s.sample_id, s.question, s.image_paths, s.gold, dict(s.extras))


def _pope_style_iter(
    coco_train_root: str,
    annotations_path: str,
    pope_adv_root: Optional[str] = None,
    n_samples: int = 1600,
    seed: int = 42,
    **kwargs,
) -> Iterator[tuple[str, str, list[Path], str, dict]]:
    """POPE-style bucket: yields self-built yes/no samples on COCO train2017.

    `pope_adv_root` (when provided) is used to extract POPE-adv eval image
    ids so the loader can filter them out of the eligible pool.
    """
    from experiments.E1_filtered_delta_opd.data.pope_style_builder import (
        iter_pope_style, load_pope_adv_image_ids,
    )

    pope_ids = load_pope_adv_image_ids(pope_adv_root) if pope_adv_root else set()
    for s in iter_pope_style(
        coco_train_root=coco_train_root,
        annotations_path=annotations_path,
        pope_adv_eval_ids=pope_ids,
        n_samples=n_samples,
        seed=seed,
        **kwargs,
    ):
        yield (s.sample_id, s.question, s.image_paths, s.gold, dict(s.extras))


def _tallyqa_iter(
    json_path: str,
    images_root: str,
    pope_adv_root: Optional[str] = None,
    n_max: int = 900,
    seed: int = 42,
    **kwargs,
) -> Iterator[tuple[str, str, list[Path], str, dict]]:
    """TallyQA complex bucket with COCO-id leakage filter."""
    from experiments.E1_filtered_delta_opd.data.tallyqa_loader import iter_tallyqa
    from experiments.E1_filtered_delta_opd.data.pope_style_builder import load_pope_adv_image_ids

    pope_ids = load_pope_adv_image_ids(pope_adv_root) if pope_adv_root else set()
    for s in iter_tallyqa(
        json_path=json_path,
        images_root=images_root,
        coco_eval_image_ids=pope_ids,
        n_max=n_max,
        seed=seed,
        **kwargs,
    ):
        yield (s.sample_id, s.question, s.image_paths, s.gold, dict(s.extras))


def _synthetic_iter(
    out_dir: str,
) -> Iterator[tuple[str, str, list[Path], str, dict]]:
    """Synthetic VLMBias-like counterfactuals; reads from manifest.jsonl."""
    from experiments.E1_filtered_delta_opd.data.synthesize_counterfactuals import (
        iter_synthetic_counterfactuals,
    )

    for s in iter_synthetic_counterfactuals(out_dir):
        yield (s.sample_id, s.question, s.image_paths, s.gold, dict(s.extras))


def _mixture_iter(
    manifest_path: str,
) -> Iterator[tuple[str, str, list[Path], str, dict]]:
    """Mixture meta-bucket: reads a pre-built 8K manifest from `mixture.py`.

    The original per-sample bucket (virl39k / pope_style / tallyqa /
    synthetic) lands in `extras["bucket"]`; `process_sample` then
    overrides the function-arg bucket with that per-sample value so the
    output record's `bucket` field reflects the true source.
    """
    from experiments.E1_filtered_delta_opd.data.mixture import iter_mixture_manifest

    for entry in iter_mixture_manifest(manifest_path):
        extras = dict(entry.get("extras", {}))
        # Canonicalize bucket inside extras so process_sample can override.
        extras["bucket"] = entry["bucket"]
        yield (
            entry["sample_id"],
            entry["question"],
            [Path(entry["image_path"])],
            entry["gold"],
            extras,
        )


BUCKET_ITERS = {
    "virl39k": _virl39k_iter,
    "pope_style": _pope_style_iter,
    "tallyqa": _tallyqa_iter,
    "synthetic": _synthetic_iter,
    "mixture": _mixture_iter,
}


# --- E1 verifier ------------------------------------------------------------

_BOXED_OUTER_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_TRAILING_NUMBER_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*$")
_LETTER_CHOICE_RE = re.compile(r"^\(?([A-Ea-e])\)?$")


def _extract_boxed(s: str) -> Optional[str]:
    m = _BOXED_OUTER_RE.search(s)
    return m.group(1).strip() if m else None


def _canon(s: str) -> str:
    """Lower/strip + collapse whitespace + drop trailing punctuation."""
    s = s.strip().lower().rstrip(".!?,;:")
    s = re.sub(r"\s+", " ", s)
    return s


def verifier_pass(teacher_text: str, gold: str) -> bool:
    """E1 verifier: did the teacher's generation match `gold`?

    Strategy (cheap, deterministic, deliberately imperfect):
      1. Extract `\\boxed{...}` from both teacher_text and gold; if either is
         missing a box, fall back to using the raw string.
      2. Canonicalize: lower / strip / collapse whitespace / drop trailing
         punctuation.
      3. If both canonical strings match a multiple-choice letter pattern
         (`A`/`B`/`(C)`), compare on the letter.
      4. If gold canonicalizes to a number AND the teacher's canonical string
         ends with that same number (or vice versa), pass — handles
         `gold="4 - 3 = 1"` matched by teacher saying just `"1"`.
      5. Otherwise, equality on canonical strings, or substring containment
         (gold-in-teacher) as a lenient fallback.

    Diagnostic-only — this is NOT a competition verifier. For final eval,
    `eval_tei.py` will reuse this with stricter rules.
    """
    t_box = _extract_boxed(teacher_text) or teacher_text
    g_box = _extract_boxed(gold) or gold

    t = _canon(t_box)
    g = _canon(g_box)
    if not t or not g:
        return False

    # (3) multiple-choice letter
    tm = _LETTER_CHOICE_RE.match(t.replace(" ", ""))
    gm = _LETTER_CHOICE_RE.match(g.replace(" ", ""))
    if tm and gm:
        return tm.group(1).lower() == gm.group(1).lower()

    # (4) trailing-number match (handles "4 - 3 = 1" vs "1")
    t_num = _TRAILING_NUMBER_RE.search(t)
    g_num = _TRAILING_NUMBER_RE.search(g)
    if t_num and g_num:
        try:
            return float(t_num.group(1)) == float(g_num.group(1))
        except ValueError:
            pass

    # (5) string equality / substring
    return t == g or g in t


# --- Per-sample driver ------------------------------------------------------

@dataclass
class TeacherForwardConfig:
    """Knobs for the per-sample teacher forward."""

    max_new_tokens: int = 256
    top_k: int = 50
    null_mode: str = "black"        # see E0 null_image.py
    keep_topk_in_record: bool = False  # disk-cost flag; default off for now


def process_sample(
    bucket: str,
    sample_id: str,
    question: str,
    image_paths: list[Path],
    gold: str,
    extras: dict,
    model,
    processor,
    tf_cfg: TeacherForwardConfig,
) -> dict[str, Any]:
    """Run the dual-forward teacher pass on one E1 training sample."""
    from PIL import Image
    from experiments.E0_image_null_delta.src.dual_forward import (
        greedy_generate,
        forced_score,
        kl_topk_union,
        prepare_null,
    )

    assert len(image_paths) == 1, "E1 v1 supports single-image samples only"
    image = Image.open(image_paths[0]).convert("RGB")
    null_img = prepare_null(image, tf_cfg.null_mode)

    # (1) greedy teacher gen under image
    ans_T_I, _ = greedy_generate(model, processor, question, image, tf_cfg.max_new_tokens)

    # (2)(3) dual forced scoring on the teacher's own response
    score_I = forced_score(model, processor, question, image, ans_T_I, tf_cfg.top_k)
    score_null = forced_score(model, processor, question, null_img, ans_T_I, tf_cfg.top_k)

    T = min(len(score_I["logp"]), len(score_null["logp"]))
    delta_t = [
        kl_topk_union(
            score_I["topk_log_probs"][t], score_I["topk_token_ids"][t],
            score_null["topk_log_probs"][t], score_null["topk_token_ids"][t],
        )
        for t in range(T)
    ]

    correct_I = verifier_pass(ans_T_I, gold)

    # For the `mixture` bucket the function-arg `bucket` is the meta-name;
    # the true per-sample bucket lives in extras. Honor the per-sample value
    # so downstream monitoring (per-bucket metrics in losses.py, parquet
    # dispatch in make_train_parquet.py) sees the original bucket name.
    out_bucket = extras.get("bucket", bucket)

    record: dict[str, Any] = {
        "bucket": out_bucket,
        "sample_id": sample_id,
        # Propagate image_paths so make_train_parquet.py can read bytes
        # directly without per-bucket lookup tables.
        "image_paths": [str(p) for p in image_paths],
        "category": extras.get("category"),
        "source": extras.get("source"),
        "pass_rate_32b": extras.get("pass_rate_32b"),
        "pass_rate_7b": extras.get("pass_rate_7b"),
        "question": question,
        "gold": gold,
        "ans_T_I": ans_T_I,
        "correct_I": correct_I,
        "trajectory_pass": correct_I,
        "response_token_ids": score_I["response_token_ids"][:T],
        "response_tokens": score_I["response_tokens"][:T],
        "teacher_logp_I": score_I["logp"][:T],
        "teacher_logp_null": score_null["logp"][:T],
        "delta_t": delta_t,
        "T": T,
        # Bucket-specific metadata (kept verbose for downstream eval / monitoring).
        "extras": extras,
    }
    if tf_cfg.keep_topk_in_record:
        record["teacher_topk_log_probs_I"] = score_I["topk_log_probs"][:T]
        record["teacher_topk_token_ids_I"] = score_I["topk_token_ids"][:T]
    return record


# --- CLI --------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1 offline teacher precompute")
    p.add_argument(
        "--bucket", required=True, choices=sorted(BUCKET_ITERS.keys()),
        help="Which loader to use. For the full 8K E1-mini run, use "
             "'mixture' with a pre-built manifest from data/mixture.py.",
    )
    p.add_argument(
        "--loader-kwargs", default="{}",
        help="JSON dict of kwargs forwarded to the bucket loader. Examples:\n"
             '  virl39k:    \'{"dataset_root":"/path/to/ViRL39K","pass_rate_min":0.3,"pass_rate_max":0.9}\'\n'
             '  pope_style: \'{"coco_train_root":"/path/to/coco","annotations_path":".../instances_train2017.json","pope_adv_root":".../POPE-adversarial","n_samples":1600}\'\n'
             '  tallyqa:    \'{"json_path":".../train.json","images_root":".../tallyqa_images","pope_adv_root":".../POPE-adversarial","n_max":900}\'\n'
             '  synthetic:  \'{"out_dir":".../synth_v1"}\' (requires manifest.jsonl already built)\n'
             '  mixture:    \'{"manifest_path":".../mixture.jsonl"}\'',
    )
    p.add_argument("--model-path", required=True, help="HF teacher model path / id")
    p.add_argument("--output", required=True, help="Output jsonl path (shard-specific)")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="Cap samples (post-shard)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--null-mode", default="black")
    p.add_argument(
        "--keep-topk", action="store_true",
        help="Store teacher_topk_log_probs_I in the record. Off by default; "
             "the reverse-KL student loss only needs the chosen-token logp.",
    )
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device-map", default="auto", help="HF device_map for the teacher")
    return p.parse_args()


def main():
    args = parse_args()

    # Heavy imports only here; keeps the file lightweight to inspect.
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    loader_kwargs = json.loads(args.loader_kwargs)
    iter_fn = BUCKET_ITERS[args.bucket]

    # Materialize the bucket so we can shard + count. The yielded tuples are
    # tiny (paths, not PIL images) so memory is not a concern.
    print(f"[precompute] enumerating bucket={args.bucket} ...", flush=True)
    all_samples = list(iter_fn(**loader_kwargs))
    print(f"[precompute] bucket={args.bucket} total_after_filter={len(all_samples)}", flush=True)

    # Round-robin shard.
    if args.num_shards > 1:
        all_samples = all_samples[args.shard_index::args.num_shards]
    if args.limit is not None:
        all_samples = all_samples[: args.limit]
    print(f"[precompute] shard={args.shard_index}/{args.num_shards} "
          f"limit={args.limit} -> processing {len(all_samples)} samples", flush=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"[precompute] loading teacher: {args.model_path} dtype={args.dtype}", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=args.device_map,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    print(f"[precompute] teacher loaded", flush=True)

    tf_cfg = TeacherForwardConfig(
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        null_mode=args.null_mode,
        keep_topk_in_record=args.keep_topk,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_done = 0
    n_pass = 0
    with open(out_path, "w") as f:
        for sample_id, question, image_paths, gold, extras in all_samples:
            try:
                rec = process_sample(
                    bucket=args.bucket,
                    sample_id=sample_id,
                    question=question,
                    image_paths=image_paths,
                    gold=gold,
                    extras=extras,
                    model=model,
                    processor=processor,
                    tf_cfg=tf_cfg,
                )
            except Exception as e:
                rec = {
                    "bucket": args.bucket,
                    "sample_id": sample_id,
                    "error": f"{type(e).__name__}: {e}",
                }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_done += 1
            if rec.get("trajectory_pass"):
                n_pass += 1
            if n_done % 25 == 0:
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1e-6)
                eta_s = (len(all_samples) - n_done) / max(rate, 1e-6)
                print(
                    f"[precompute] {n_done}/{len(all_samples)} "
                    f"({rate:.2f} it/s, ETA {eta_s/60:.1f}min) "
                    f"trajectory_pass_rate={n_pass/n_done:.3f}",
                    flush=True,
                )

    elapsed = time.time() - t0
    pass_rate = n_pass / max(n_done, 1)
    print(
        f"[precompute] DONE shard={args.shard_index}: {n_done} samples in "
        f"{elapsed/60:.1f}min ({n_done/max(elapsed,1e-6):.2f} it/s); "
        f"trajectory_pass_rate={pass_rate:.3f} -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
