"""
Qwen2.5-VL prompt construction.

Two callers:
  * generation     — build prompt with image (or no image), ask processor to
                     produce input_ids ready for model.generate().
  * forced scoring — build prompt + appended assistant response, ask processor
                     to produce input_ids whose tail equals the response. The
                     caller can then run model.forward() and slice logits at
                     the response positions.

Important: for forced scoring under (image vs null), we want the same response
token sequence under both conditions. The image content differs but the text
sequence remains identical, so the slice offsets (prompt_len) only depend on
the prompt up to the assistant cue and the response. We pass the IDs in
explicitly rather than re-tokenizing, to avoid drift.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image


def _user_content(question: str, image: Optional[Image.Image]) -> list[dict]:
    if image is None:
        return [{"type": "text", "text": question}]
    return [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]


def build_generation_inputs(processor, question: str, image: Optional[Image.Image]):
    """
    Return processor(...) inputs ready for model.generate().

    `image=None` corresponds to the image_drop null mode; the chat template
    is asked to produce a text-only message.
    """
    messages = [{"role": "user", "content": _user_content(question, image)}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if image is None:
        return processor(text=[text], return_tensors="pt", padding=True)
    return processor(
        text=[text], images=[image], return_tensors="pt", padding=True
    )


def build_forced_scoring_inputs(
    processor,
    question: str,
    image: Optional[Image.Image],
    response_text: str,
):
    """
    Return (full_inputs, prompt_len) for forced scoring.

    full_inputs: processor output with the assistant response appended.
    prompt_len:  length of the prompt portion (up to and including the
                 `<|im_start|>assistant\\n` cue). Response logits live at
                 full_inputs.input_ids[0, prompt_len:].

    Note: prompt_len is computed off the same processor call with
    add_generation_prompt=True, so it includes whatever image-pad token
    expansion the image triggers — which differs between image and null when
    `image_drop` is used, but is identical between (real image, black null)
    since black is the same resolution.
    """
    # Prompt-only (with generation cue) — used to find where the response starts.
    prompt_messages = [{"role": "user", "content": _user_content(question, image)}]
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    # Full prompt + response (no generation cue; we manually appended).
    full_text = prompt_text + response_text

    if image is None:
        prompt_inputs = processor(text=[prompt_text], return_tensors="pt", padding=True)
        full_inputs = processor(text=[full_text], return_tensors="pt", padding=True)
    else:
        prompt_inputs = processor(
            text=[prompt_text], images=[image], return_tensors="pt", padding=True
        )
        full_inputs = processor(
            text=[full_text], images=[image], return_tensors="pt", padding=True
        )

    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    return full_inputs, prompt_len
