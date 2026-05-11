"""
E0 dual-forward driver — per-sample image-vs-null teacher delta on a single GPU.

For each sample in this shard, this script:

  1. Generates the teacher response greedily under image and under null.
  2. Forced-scores the image-conditioned response under BOTH conditions, token
     by token, recording log p_T^I(y_t), log p_T^null(y_t), and
     delta_t = KL_topK( p_T^I(. | y_<t)  ||  p_T^null(. | y_<t) )
     using the union of top-K per position to truncate noise.
  3. (VLMBias / POPE only) Forced-scores fixed answer-option strings under
     both conditions for metric 3 (answer-level visual gain).
  4. Writes one jsonl line per sample.

This script is single-process / single-model. Data parallelism comes from
launching multiple processes (one per CUDA device) via the bash launchers
under experiments/E0_image_null_delta/scripts/.

Run on the remote box, NOT locally:

  CUDA_VISIBLE_DEVICES=0 python -m experiments.E0_image_null_delta.src.dual_forward \\
      --config experiments/E0_image_null_delta/configs/e0_default.yaml \\
      --model teacher32b \\
      --dataset vlmbias_main \\
      --shard-index 0 --num-shards 8 \\
      --output experiments/E0_image_null_delta/results/e0_teacher32b_vlmbias_main.shard0.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from tqdm import tqdm

# Imports are written so this file can be run as a module from the repo root.
from experiments.E0_image_null_delta.data.loaders import Sample, load_dataset
from experiments.E0_image_null_delta.src.null_image import make_null_image
from experiments.E0_image_null_delta.src.prompting import (
    build_forced_scoring_inputs,
    build_generation_inputs,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_cfg: dict, models_root: str):
    """Load a Qwen2.5-VL model + processor per the config block."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_path = os.path.join(models_root, model_cfg["path_relative"])
    dtype = getattr(torch, model_cfg["dtype"])

    device_map_cfg = model_cfg["device_map"]
    if device_map_cfg == "single":
        # Pin everything to cuda:0 of whatever CUDA_VISIBLE_DEVICES the caller set.
        device_map = {"": 0}
    elif device_map_cfg == "auto":
        device_map = "auto"
    else:
        device_map = device_map_cfg

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation=model_cfg["attn_impl"],
        device_map=device_map,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


# ---------------------------------------------------------------------------
# Image / null prep
# ---------------------------------------------------------------------------

def prepare_null(sample_image, null_mode: str):
    """Return the null image (or None for image_drop)."""
    if null_mode == "image_drop":
        return None
    return make_null_image(sample_image, mode=null_mode)


# ---------------------------------------------------------------------------
# Greedy generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_generate(
    model,
    processor,
    question: str,
    image,
    max_new_tokens: int,
) -> tuple[str, torch.Tensor]:
    """
    Greedy-generate a response.

    Returns (decoded_text, response_token_ids_tensor_1d_on_cpu).
    """
    inputs = build_generation_inputs(processor, question, image)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[1])

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    response_ids = output[0, prompt_len:].detach().cpu()
    response_text = processor.tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text, response_ids


# ---------------------------------------------------------------------------
# Forced scoring of a fixed response
# ---------------------------------------------------------------------------

@torch.no_grad()
def forced_score(
    model,
    processor,
    question: str,
    image,
    response_text: str,
    top_k: int,
) -> dict[str, Any]:
    """
    Forced-score `response_text` under (question, image).

    Returns dict with:
      response_token_ids : list[int]
      response_tokens    : list[str]  (decoded per-token, includes leading-space markers)
      logp               : list[float]  log p(y_t | y_<t) for each response token
      topk_log_probs     : list[list[float]]  per-position top-K log-probs
      topk_token_ids     : list[list[int]]    per-position top-K token ids
    """
    full_inputs, prompt_len = build_forced_scoring_inputs(
        processor, question, image, response_text
    )
    full_inputs = {k: v.to(model.device) for k, v in full_inputs.items()}
    input_ids = full_inputs["input_ids"]
    full_len = int(input_ids.shape[1])

    if full_len <= prompt_len:
        # response_text tokenized to zero new tokens — degenerate case
        return {
            "response_token_ids": [],
            "response_tokens": [],
            "logp": [],
            "topk_log_probs": [],
            "topk_token_ids": [],
        }

    outputs = model(**full_inputs)
    logits = outputs.logits  # [1, L, V]
    # logits at position i predict token i+1.
    # Response tokens occupy positions [prompt_len, full_len). To score them,
    # we use logits[prompt_len-1 : full_len-1].
    response_logits = logits[0, prompt_len - 1 : full_len - 1, :]  # [T, V]
    response_ids = input_ids[0, prompt_len:full_len]               # [T]

    log_probs = torch.log_softmax(response_logits.float(), dim=-1)   # [T, V]
    logp_per_token = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)  # [T]

    topk_lp, topk_ids = log_probs.topk(top_k, dim=-1)  # [T, K], [T, K]

    return {
        "response_token_ids": response_ids.detach().cpu().tolist(),
        "response_tokens": [
            processor.tokenizer.decode([tid]) for tid in response_ids.detach().cpu().tolist()
        ],
        "logp": logp_per_token.detach().cpu().tolist(),
        "topk_log_probs": topk_lp.detach().cpu().tolist(),
        "topk_token_ids": topk_ids.detach().cpu().tolist(),
    }


# ---------------------------------------------------------------------------
# KL on the union of per-position top-K supports
# ---------------------------------------------------------------------------

def kl_topk_union(
    topk_lp_p: list[float],
    topk_ids_p: list[int],
    topk_lp_q: list[float],
    topk_ids_q: list[int],
) -> float:
    """
    KL(P || Q) computed on the union of P's and Q's top-K supports.

    Restricting to the union (instead of full vocab) avoids long-tail noise and
    is cheap. Probabilities are renormalized over the union before the KL.
    Tokens missing from one side are floored at the smallest top-K log-prob on
    that side, then renormalized — this is a conservative finite KL estimate.
    """
    p_map: dict[int, float] = {tid: lp for tid, lp in zip(topk_ids_p, topk_lp_p)}
    q_map: dict[int, float] = {tid: lp for tid, lp in zip(topk_ids_q, topk_lp_q)}
    union = sorted(set(p_map) | set(q_map))

    # Use the smallest top-K log-prob on each side as the floor for missing ids.
    p_floor = min(topk_lp_p)
    q_floor = min(topk_lp_q)

    log_p = [p_map.get(tid, p_floor) for tid in union]
    log_q = [q_map.get(tid, q_floor) for tid in union]

    # Convert to probabilities and renormalize.
    p = [math.exp(lp) for lp in log_p]
    q = [math.exp(lq) for lq in log_q]
    sp = sum(p)
    sq = sum(q)
    if sp <= 0 or sq <= 0:
        return 0.0
    p = [v / sp for v in p]
    q = [v / sq for v in q]

    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0 and qi > 0.0:
            kl += pi * (math.log(pi) - math.log(qi))
    return float(kl)


# ---------------------------------------------------------------------------
# Option-level forced scoring (for metric 3)
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_option(
    model,
    processor,
    question: str,
    image,
    option_text: str,
) -> float:
    """
    Return sum_t log p_T(option_token_t | question, image, option_<t).

    Used for VLMBias (ground_truth, expected_bias) and POPE (yes, no).
    """
    full_inputs, prompt_len = build_forced_scoring_inputs(
        processor, question, image, option_text
    )
    full_inputs = {k: v.to(model.device) for k, v in full_inputs.items()}
    input_ids = full_inputs["input_ids"]
    full_len = int(input_ids.shape[1])
    if full_len <= prompt_len:
        return 0.0

    outputs = model(**full_inputs)
    logits = outputs.logits[0, prompt_len - 1 : full_len - 1, :]   # [T, V]
    response_ids = input_ids[0, prompt_len:full_len]                # [T]

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    logp = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
    return float(logp.sum().item())


# ---------------------------------------------------------------------------
# Per-sample driver
# ---------------------------------------------------------------------------

def options_for_sample(sample: Sample) -> Optional[dict[str, str]]:
    """
    Return {label: option_string} for metric-3 option scoring, or None to skip.

    VLMBias: ground_truth vs expected_bias.
    POPE:    "yes" vs "no".
    MathVista: skipped (metric 3 is VLMBias-specific; choices for MV vary).
    """
    if sample.dataset.startswith("vlmbias"):
        return {
            "ground_truth": sample.gold,
            "expected_bias": sample.extras["expected_bias"],
        }
    if sample.dataset == "pope_adversarial":
        return {"yes": "yes", "no": "no"}
    return None


def parse_correctness(sample: Sample, response_text: str) -> bool:
    """
    Cheap, dataset-specific correctness parser. For E0 we don't try to be
    bulletproof — these are diagnostic, not leaderboard runs.
    """
    rt = response_text.strip().lower()
    gold = sample.gold.strip().lower()

    if sample.dataset.startswith("vlmbias"):
        # VLMBias prompts ask for {Yes} / {No} (or other short answers).
        # Extract the first {...} payload; fall back to substring match.
        import re
        m = re.search(r"\{([^}]+)\}", response_text)
        candidate = (m.group(1) if m else response_text).strip().lower()
        return candidate == gold or gold in candidate

    if sample.dataset == "pope_adversarial":
        # Answers are short yes/no. First occurrence of yes vs no wins.
        # Tie-breaker: if neither word appears, mark wrong.
        first_yes = rt.find("yes")
        first_no = rt.find("no")
        if first_yes == -1 and first_no == -1:
            return False
        if first_yes == -1:
            return gold == "no"
        if first_no == -1:
            return gold == "yes"
        predicted = "yes" if first_yes < first_no else "no"
        return predicted == gold

    if sample.dataset == "mathvista_mini":
        # Loose: exact-match or substring on lowered+stripped strings.
        return rt == gold or gold in rt

    return rt == gold


def process_sample(
    sample: Sample,
    model,
    processor,
    cfg: dict,
    null_mode: str,
) -> dict[str, Any]:
    gen_cfg = cfg["generation"]
    top_k = int(cfg["forced_scoring"]["top_k"])

    null_img = prepare_null(sample.image, null_mode)

    # (1) Greedy generation under both conditions.
    ans_T_I, _ = greedy_generate(
        model, processor, sample.question, sample.image, gen_cfg["max_new_tokens"]
    )
    ans_T_null, _ = greedy_generate(
        model, processor, sample.question, null_img, gen_cfg["max_new_tokens"]
    )

    # (2) Forced-score ans_T_I under both conditions.
    score_I = forced_score(model, processor, sample.question, sample.image, ans_T_I, top_k)
    score_null = forced_score(model, processor, sample.question, null_img, ans_T_I, top_k)

    # Sanity: the two scorings should produce the same response token sequence
    # (text identical; tokenization deterministic). If they diverge in length,
    # we trim to the min and record the discrepancy.
    T = min(len(score_I["logp"]), len(score_null["logp"]))
    delta_t = [
        kl_topk_union(
            score_I["topk_log_probs"][t], score_I["topk_token_ids"][t],
            score_null["topk_log_probs"][t], score_null["topk_token_ids"][t],
        )
        for t in range(T)
    ]

    # (3) Option-level forced scoring (metric 3 inputs).
    option_logP = None
    options = options_for_sample(sample) if cfg.get("option_scoring", {}).get("enabled", False) else None
    if options is not None:
        option_logP_I = {
            label: score_option(model, processor, sample.question, sample.image, text)
            for label, text in options.items()
        }
        option_logP_null = {
            label: score_option(model, processor, sample.question, null_img, text)
            for label, text in options.items()
        }
        option_logP = {"I": option_logP_I, "null": option_logP_null}

    # (4) Correctness.
    correct_I = parse_correctness(sample, ans_T_I)
    correct_null = parse_correctness(sample, ans_T_null)

    record: dict[str, Any] = {
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "null_mode": null_mode,
        "question": sample.question,
        "gold": sample.gold,
        "extras": sample.extras,
        "ans_T_I": ans_T_I,
        "ans_T_null": ans_T_null,
        "correct_I": correct_I,
        "correct_null": correct_null,
        "response_tokens": score_I["response_tokens"][:T],
        "logp_I": score_I["logp"][:T],
        "logp_null": score_null["logp"][:T],
        "delta_t": delta_t,
        "T": T,
        "T_image_only": len(score_I["logp"]),
        "T_null_only": len(score_null["logp"]),
    }
    if option_logP is not None:
        record["option_logP"] = option_logP
    return record


# ---------------------------------------------------------------------------
# Shard helper
# ---------------------------------------------------------------------------

def shard_samples(samples: list[Sample], shard_index: int, num_shards: int) -> list[Sample]:
    """Round-robin shard so per-shard difficulty distributions are similar."""
    if num_shards <= 1:
        return samples
    return samples[shard_index::num_shards]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to e0_default.yaml")
    p.add_argument("--model", required=True, help="Model key from cfg['models']")
    p.add_argument("--dataset", required=True, help="Dataset key from cfg['datasets']")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument(
        "--null-mode",
        default=None,
        help="Override null mode. Defaults to cfg['null_modes'][0].",
    )
    p.add_argument("--output", required=True, help="Output jsonl path (shard-specific).")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on samples (after sharding) — for sanity / smoke tests.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    random.seed(cfg.get("seed", 42))
    torch.manual_seed(cfg.get("seed", 42))

    null_mode = args.null_mode or cfg["null_modes"][0]

    print(f"[dual_forward] config={args.config} model={args.model} dataset={args.dataset} "
          f"shard={args.shard_index}/{args.num_shards} null_mode={null_mode}", flush=True)

    samples = load_dataset(args.dataset, cfg)
    samples = shard_samples(samples, args.shard_index, args.num_shards)
    if args.limit is not None:
        samples = samples[: args.limit]
    print(f"[dual_forward] {len(samples)} samples in this shard", flush=True)

    model, processor = load_model(cfg["models"][args.model], cfg["paths"]["models_root"])
    print(f"[dual_forward] model loaded on device(s): "
          f"{set(p.device for p in model.parameters())}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(out_path, "w") as fout:
        for i, sample in enumerate(tqdm(samples, desc=f"shard{args.shard_index}")):
            try:
                record = process_sample(sample, model, processor, cfg, null_mode)
            except Exception as e:
                # Don't kill the whole shard for one bad sample — record the failure.
                record = {
                    "dataset": sample.dataset,
                    "sample_id": sample.sample_id,
                    "null_mode": null_mode,
                    "error": f"{type(e).__name__}: {e}",
                }
                print(f"[dual_forward] sample {sample.sample_id} failed: {e}", file=sys.stderr, flush=True)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                fout.flush()

    print(f"[dual_forward] done in {time.time() - t0:.1f}s. wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
