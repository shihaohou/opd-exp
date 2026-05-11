"""
Null-image generation.

Used to build the `p_T(. | x, null)` condition for the dual-forward
diagnostic. The Qwen2.5-VL processor maps image resolution to a specific
number of <|image_pad|> tokens, so the null image MUST be the same size as
the real image — otherwise the prompt tokenization shifts and forced-scoring
across (image, null) becomes misaligned.
"""

from __future__ import annotations

from PIL import Image


def make_null_image(real: Image.Image, mode: str = "black") -> Image.Image:
    """
    Build the null counterpart of `real`.

    mode='black' — all-black RGB image at the same resolution. This is the
    default and currently the only supported mode in E0.

    Other modes (gaussian, patch_shuffle, irrelevant) are Step 2 ablations
    and are intentionally NotImplemented here.
    """
    if mode == "black":
        return Image.new("RGB", real.size, color=(0, 0, 0))
    if mode == "image_drop":
        # image_drop is handled at the prompt level (omit image from messages),
        # not by returning a PIL image. Callers should branch on mode before
        # calling this function.
        raise ValueError("image_drop is a prompt-level mode, not a pixel mode")
    raise NotImplementedError(f"null mode '{mode}' is not implemented in E0 (Step 2 ablation)")
