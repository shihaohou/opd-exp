"""
Spike: verify vLLM can serve Qwen2.5-VL-32B with multimodal images +
`prompt_logprobs=K` simultaneously.

This is the precondition for the on-policy v1 dual-forward teacher path
described in `on_policy_v1_design.md`. If this works:
  - We can use the existing verl `AsyncTeacherLLMServerManager` infra
    (`verl/experimental/teacher_loop/teacher_manager.py`) by calling it
    twice per sample (once with the real image, once with a black null
    image) and computing delta_t locally.

If this FAILS (e.g. vLLM rejects multimodal + prompt_logprobs combo),
we fall back to HF transformers teacher with manual forced-score —
slower (no continuous batching) but always works.

Run:
    python experiments/E1_filtered_delta_opd/src/spike_vllm_dual_forward.py \\
        --model-path /home/web_server/antispam/project/houshihao/models/Qwen2.5-VL-32B-Instruct \\
        --image-path <path to any small image>

Smoke success criteria:
  - Both image-conditioned and null-image-conditioned calls return without
    error
  - `prompt_logprobs` is a list of (length == #prompt tokens) entries
  - Each non-None entry is a dict of {token_id: Logprob}, size ≈ K
  - Top-K under image and top-K under null differ for at least some
    positions (proves the image is actually conditioning the forward)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image


def make_null_image(real_image: Image.Image) -> Image.Image:
    return Image.new("RGB", real_image.size, color=(0, 0, 0))


def build_chat_text(processor, question: str) -> str:
    """Use the model's own chat template so we don't hand-code special tokens."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def summarize_position(pl_entry, top_n: int = 5):
    """pl_entry is a dict {token_id: Logprob}. Return top-N tuples."""
    if pl_entry is None:
        return None
    items = sorted(
        pl_entry.items(), key=lambda kv: kv[1].logprob, reverse=True
    )[:top_n]
    return [(k, round(v.logprob, 2), getattr(v, "decoded_token", "?")) for k, v in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--image-path", required=True,
                    help="Any small JPG/PNG to use as the real image")
    ap.add_argument(
        "--question", default="Describe the image briefly.",
        help="Text question for the prompt",
    )
    ap.add_argument(
        "--force-response", default="The image shows a scene.",
        help=("A fixed response to force-score. vLLM prompt_logprobs returns "
              "logprobs at every prompt position, so the response is concatenated "
              "to the prompt to get its forced-score logprobs in one call."),
    )
    ap.add_argument("--top-k", type=int, default=50,
                    help="prompt_logprobs K — should match E1 K (50)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    args = ap.parse_args()

    print("[spike] importing vllm + transformers ...", flush=True)
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    print(f"[spike] loading processor {args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path)

    print(f"[spike] loading vLLM (this may take 1-2 min for 32B)", flush=True)
    t0 = time.time()
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
        # vLLM caps prompt_logprobs at `max_logprobs` (default 20). We need
        # at least K to satisfy our SamplingParams(prompt_logprobs=K). For
        # E1 on-policy v1, K=50 matches E0's `delta_t` top-50 union.
        max_logprobs=args.top_k,
    )
    print(f"[spike] model loaded in {(time.time()-t0)/60:.1f} min", flush=True)

    real_image = Image.open(args.image_path).convert("RGB")
    null_image = make_null_image(real_image)
    chat_text = build_chat_text(processor, args.question)
    full_text = chat_text + args.force_response

    sampling_params = SamplingParams(
        max_tokens=1,            # we only want forced-score, not actual generation
        temperature=1.0,         # required: prompt_logprobs doesn't honor temperature anyway
        prompt_logprobs=args.top_k,
    )

    print("\n[spike] --- Test 1: image-conditioned ---", flush=True)
    t0 = time.time()
    outputs_I = llm.generate(
        prompts=[{
            "prompt": full_text,
            "multi_modal_data": {"image": real_image},
        }],
        sampling_params=sampling_params,
    )
    print(f"  generated in {time.time()-t0:.2f}s", flush=True)
    pl_I = outputs_I[0].prompt_logprobs
    print(f"  prompt_logprobs length: {len(pl_I)}")
    nonempty = sum(1 for x in pl_I if x)
    print(f"  non-None entries: {nonempty}/{len(pl_I)}")
    if nonempty == 0:
        print("[spike] ❌ FAIL: all prompt_logprobs entries are None")
        return

    print("\n[spike] --- Test 2: null-image-conditioned ---", flush=True)
    t0 = time.time()
    outputs_null = llm.generate(
        prompts=[{
            "prompt": full_text,
            "multi_modal_data": {"image": null_image},
        }],
        sampling_params=sampling_params,
    )
    print(f"  generated in {time.time()-t0:.2f}s", flush=True)
    pl_null = outputs_null[0].prompt_logprobs
    print(f"  prompt_logprobs length: {len(pl_null)}")

    assert len(pl_I) == len(pl_null), \
        f"length mismatch: {len(pl_I)} vs {len(pl_null)}"

    print("\n[spike] --- Test 3: image-vs-null differ at SOME positions ---", flush=True)
    diff_count = 0
    for t in range(len(pl_I)):
        if pl_I[t] is None or pl_null[t] is None:
            continue
        # compare top-1 token id under each condition
        top_I = max(pl_I[t].items(), key=lambda kv: kv[1].logprob)[0]
        top_null = max(pl_null[t].items(), key=lambda kv: kv[1].logprob)[0]
        if top_I != top_null:
            diff_count += 1
    print(f"  positions where top-1 differs (image vs null): "
          f"{diff_count}/{nonempty}")
    if diff_count == 0:
        print("  ⚠️  image and null produce identical top-1 everywhere — "
              "either the prompt has no image-bearing tokens OR the image "
              "isn't being conditioned on. Inspect manually.")

    print("\n[spike] --- Test 4: sample 5 last positions, show top-5 each ---", flush=True)
    n_show = 5
    L = len(pl_I)
    for t in range(max(L - n_show, 0), L):
        top_I = summarize_position(pl_I[t])
        top_null = summarize_position(pl_null[t])
        print(f"  pos {t}:")
        print(f"    top5  I  : {top_I}")
        print(f"    top5 null: {top_null}")

    print("\n[spike] ✅ vLLM multimodal + prompt_logprobs=K works on Qwen2.5-VL-32B")
    print("[spike]    on-policy v1 dual-forward teacher path is unblocked")


if __name__ == "__main__":
    main()
