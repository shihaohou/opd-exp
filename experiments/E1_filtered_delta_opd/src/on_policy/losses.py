"""
E1 on-policy v1 distillation losses — KL/CE per-sample dispatch.

Loss math (per response token ``t``)
------------------------------------
At each response position, the FSDP logits processor has precomputed
``raw_kl_t = KL_topK(P_T^I(.|s_t) || P_S(.|s_t))`` for KL samples (it
runs on all samples; for CE samples we wrote dummy teacher tensors in
the agent-loop worker patch, so its output is meaningless and gets
masked away). For each sample:

* **KL branch** (``loss_branch == "kl"``):
      ``loss_t = w_t · raw_kl_t``
  where ``w_t = clip(delta_t, p95) / mean(clip)`` if the recipe uses
  delta, else ``w_t = 1``.

* **CE branch** (``loss_branch == "ce"``, only reachable when the recipe
  filters AND teacher was wrong):
      ``loss_t = β · (-student_logp_t)``
  i.e. β-weighted NLL of the gold token at response position ``t``.
  This is the "Final answer: \\boxed{X}." suffix the agent loop set as
  ``response_ids`` for these samples; ``response_mask`` is 1 over that
  span and 0 over padding, so verl's outer ``agg_loss`` aggregates the
  CE correctly.

Per-sample dispatch is via ``is_kl`` / ``is_ce`` masks ``(bsz, 1)``
built from ``data["loss_branch"]``:
    per_token = is_kl · kl_loss + is_ce · ce_loss

Both branches share the same student forward — verl's FSDP engine
always produces ``model_output["log_probs"]`` regardless of
``use_topk`` (verified at ``verl/workers/engine/fsdp/transformer_impl.py:1197``),
so the CE branch reads it for free.

β default (= 0.1) is set in ``e1_base.yaml`` and plumbed per-sample via
``data["beta"]``; all samples in a batch share the same value (agent
loop reads it once at init).

Mandatory monitoring (per design § 5 + GPT review)
--------------------------------------------------
* ``e1_v1/effective_kl_tokens``  — Σ ``response_mask · is_kl``  (A, B, C, D)
* ``e1_v1/effective_ce_tokens``  — Σ ``response_mask · is_ce``  (C, D)
* ``e1_v1/effective_ce_samples`` — # samples on CE branch          (C, D)
* ``e1_v1/kl_loss_sum``           — Σ kl_loss · response_mask · is_kl  (all)
* ``e1_v1/ce_loss_sum``           — Σ ce_loss · response_mask · is_ce  (C, D)
* ``e1_v1/kl_ce_ratio``           — kl_sum / (kl_sum + ce_sum)         (C, D)
* ``e1_v1/delta_t_{mean_pre_norm, p99_pre_norm, mean_post_norm}``       (B, D)
* ``e1_v1/trajectory_pass_rate``  — fraction with T_correct=1 (filtered configs)

Per-bucket breakdowns of ``effective_kl_tokens``, ``effective_ce_samples``,
``teacher_correct_rate``, ``kl_loss_sum``, ``ce_loss_sum`` flow under
``e1_v1/bucket/<name>/<metric>``.

Attribution guard: if ``kl_ce_ratio → 0`` (CE dominates) on Config D,
its improvement over Config C is NOT attributable to delta. This is the
key check that determines whether D-vs-C results are interpretable.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ===========================================================================
# Pure-math helpers — unit-testable without verl.
# ===========================================================================

def normalize_delta(
    delta_t: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    p95_quantile: float = 0.95,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """``clip(delta_t, p95) / mean(clip)`` over the valid response tokens.

    Computed in float32; cast back to input dtype. If the mask selects
    zero valid tokens, returns ones (vanilla KD equivalence) — so a
    micro-batch with no KL samples doesn't NaN out delta-config training.

    Returns (w_t, stats) where stats has mean_pre_norm, p99_pre_norm,
    mean_post_norm for monitoring.
    """
    mask_bool = response_mask.bool()
    valid = delta_t[mask_bool].float()
    if valid.numel() == 0:
        return torch.ones_like(delta_t), {
            "mean_pre_norm": 0.0,
            "p99_pre_norm": 0.0,
            "mean_post_norm": 1.0,
        }

    p95 = torch.quantile(valid, p95_quantile)
    p99 = torch.quantile(valid, 0.99)
    clipped = torch.clamp(delta_t.float(), max=p95)
    clipped_valid = clipped[mask_bool]
    mean_clipped = clipped_valid.mean().clamp_min(eps)
    w_t = (clipped / mean_clipped).to(delta_t.dtype)
    stats = {
        "mean_pre_norm": clipped_valid.mean().item(),
        "p99_pre_norm": p99.item(),
        "mean_post_norm": (clipped_valid / mean_clipped).mean().item(),
    }
    return w_t, stats


# ===========================================================================
# Data extraction helpers — verl-side, behind a lazy import.
# ===========================================================================

def _get_nts(data, key: str, default=None):
    """``tu.get`` wrapper that unwraps NonTensorStack to Python list."""
    from verl.utils import tensordict_utils as tu

    return tu.get(data, key, default=default)


def _stack_delta_t(
    data,
    *,
    batch_size: int,
    response_width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    raw = _get_nts(data, "delta_t", default=None)
    if raw is None:
        return None
    arr = np.stack([np.asarray(d, dtype=np.float32) for d in raw], axis=0)
    if arr.shape != (batch_size, response_width):
        raise ValueError(
            f"delta_t shape {arr.shape} != (bsz, response_width)="
            f"({batch_size}, {response_width}); check agent-loop padding."
        )
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def _stack_per_sample_str(data, key: str, batch_size: int, default: str) -> list[str]:
    raw = _get_nts(data, key, default=None)
    if raw is None:
        return [default] * batch_size
    return [str(x) if x is not None else default for x in raw]


def _stack_per_sample_bool(data, key: str, batch_size: int, default: bool) -> np.ndarray:
    raw = _get_nts(data, key, default=None)
    if raw is None:
        return np.full((batch_size,), float(default), dtype=np.float32)
    return np.asarray([float(bool(x)) for x in raw], dtype=np.float32)


def _read_beta(data, default: float = 0.1) -> float:
    raw = _get_nts(data, "beta", default=None)
    if raw is None:
        return default
    # All samples in a batch share the same β.
    try:
        return float(raw[0])
    except (IndexError, TypeError, ValueError):
        return default


# ===========================================================================
# verl tensor helpers.
# ===========================================================================

def _read_raw_kl(model_output, data) -> torch.Tensor:
    """Per-token raw forward-KL the FSDP logits processor produced.

    For CE samples we wrote dummy teacher tensors in the worker patch, so
    the per-token KL at those positions is non-zero but meaningless;
    ``is_kl`` masks it away before aggregation.
    """
    from verl.workers.utils.padding import no_padding_2_padding

    losses = no_padding_2_padding(model_output["distillation_losses"], data)
    return losses.clamp_min(0.0)


def _read_student_log_probs(model_output, data) -> torch.Tensor:
    """Per-token student log-prob at the chosen response token (response_ids[t]).

    For CE samples ``response_ids = gold_token_ids`` (set by the agent
    loop), so ``-log_probs[sample, t]`` is exactly the NLL of the gold
    token. For KL samples we don't use it, but log_probs is always
    computed by the FSDP engine so reading it is free.
    """
    from verl.workers.utils.padding import no_padding_2_padding

    log_probs = no_padding_2_padding(model_output["log_probs"], data)
    return log_probs


def _read_response_mask(data) -> torch.Tensor:
    rm = data["response_mask"]
    if hasattr(rm, "is_nested") and rm.is_nested:
        rm = rm.to_padded_tensor(False)
    return rm


# ===========================================================================
# Per-bucket monitoring helper.
# ===========================================================================

def _emit_per_bucket(
    metrics: dict[str, Any],
    *,
    buckets: list[str],
    response_mask_f: torch.Tensor,         # (bsz, response_width)
    is_kl_b: torch.Tensor,                 # (bsz, 1)
    is_ce_b: torch.Tensor,                 # (bsz, 1)
    pass_rate_b: torch.Tensor,             # (bsz, 1)
    kl_loss_full: torch.Tensor,            # (bsz, response_width)
    ce_loss_full: torch.Tensor,            # (bsz, response_width)
) -> None:
    """Slice the per-token tensors per unique bucket name; emit summed metrics.

    Cheap: ~3 buckets, one bool mask per bucket. Done on CPU after a single
    ``.cpu()`` would also work but tensors are small so just keep on device.
    """
    device = response_mask_f.device
    for bk in sorted(set(buckets)):
        bk_mask_1d = torch.tensor(
            [1.0 if b == bk else 0.0 for b in buckets],
            device=device,
            dtype=response_mask_f.dtype,
        ).unsqueeze(-1)  # (bsz, 1)
        n_samples = bk_mask_1d.sum().item()
        if n_samples <= 0:
            continue

        kl_tokens = (bk_mask_1d * is_kl_b * response_mask_f).sum().item()
        ce_tokens = (bk_mask_1d * is_ce_b * response_mask_f).sum().item()
        ce_samples = (bk_mask_1d * is_ce_b).sum().item()
        kl_sum = (kl_loss_full * bk_mask_1d * is_kl_b * response_mask_f).sum().item()
        ce_sum = (ce_loss_full * bk_mask_1d * is_ce_b * response_mask_f).sum().item()
        t_correct = (bk_mask_1d.squeeze(-1) * pass_rate_b.squeeze(-1)).sum().item()

        prefix = f"e1_v1/bucket/{bk}"
        metrics[f"{prefix}/n_samples"] = n_samples
        metrics[f"{prefix}/effective_kl_tokens"] = kl_tokens
        metrics[f"{prefix}/effective_ce_tokens"] = ce_tokens
        metrics[f"{prefix}/effective_ce_samples"] = ce_samples
        metrics[f"{prefix}/kl_loss_sum"] = kl_sum
        metrics[f"{prefix}/ce_loss_sum"] = ce_sum
        metrics[f"{prefix}/teacher_correct_rate"] = t_correct / max(n_samples, 1.0)


# ===========================================================================
# The loss factory. Per-sample dispatch + monitoring; same body for all 4
# configs to keep their behaviours strictly aligned.
# ===========================================================================

def _build_loss_fn(*, apply_delta: bool, apply_pass_filter: bool):
    """4 configs differ in (apply_delta × apply_pass_filter):
    A: F, F   B: T, F   C: F, T   D: T, T
    """

    def loss_fn(config, distillation_config, model_output, data) -> tuple[torch.Tensor, dict[str, Any]]:
        # ---- read tensors ----
        raw_kl = _read_raw_kl(model_output, data)                  # (bsz, response_width)
        log_probs = _read_student_log_probs(model_output, data)    # (bsz, response_width)
        response_mask = _read_response_mask(data)                  # (bsz, response_width) bool/int

        assert raw_kl.shape == log_probs.shape == response_mask.shape, (
            f"shape mismatch: raw_kl={tuple(raw_kl.shape)} "
            f"log_probs={tuple(log_probs.shape)} "
            f"response_mask={tuple(response_mask.shape)}"
        )

        bsz, response_width = raw_kl.shape
        device = raw_kl.device
        dtype = raw_kl.dtype
        response_mask_f = response_mask.to(dtype)

        # ---- per-sample fields (from agent_loop extra_fields → non_tensor_batch → NonTensorStack) ----
        loss_branch_list = _stack_per_sample_str(data, "loss_branch", bsz, default="kl")
        buckets = _stack_per_sample_str(data, "bucket", bsz, default="unknown")
        pass_arr = _stack_per_sample_bool(data, "trajectory_pass", bsz, default=True)
        beta = _read_beta(data, default=0.1) if apply_pass_filter else 0.0

        is_ce_np = np.asarray([1.0 if lb == "ce" else 0.0 for lb in loss_branch_list], dtype=np.float32)
        is_kl_np = 1.0 - is_ce_np
        is_ce_b = torch.from_numpy(is_ce_np).to(device=device, dtype=dtype).unsqueeze(-1)   # (bsz, 1)
        is_kl_b = torch.from_numpy(is_kl_np).to(device=device, dtype=dtype).unsqueeze(-1)   # (bsz, 1)
        pass_rate_b = torch.from_numpy(pass_arr).to(device=device, dtype=dtype).unsqueeze(-1)

        # ---- KL branch loss ----
        kl_loss = raw_kl
        delta_stats: Optional[dict[str, float]] = None
        if apply_delta:
            delta_t = _stack_delta_t(
                data,
                batch_size=bsz,
                response_width=response_width,
                device=device,
                dtype=dtype,
            )
            if delta_t is None:
                logger.warning(
                    "[e1_onpolicy] apply_delta=True but no delta_t in data; "
                    "falling back to vanilla KL for this batch."
                )
            else:
                # Normalize over KL-branch response tokens only — CE positions
                # have delta_t=0 from the worker patch and shouldn't pull the
                # mean down.
                kl_token_mask = response_mask_f * is_kl_b
                w_t, delta_stats = normalize_delta(delta_t, kl_token_mask)
                kl_loss = kl_loss * w_t

        # ---- CE branch loss ----
        # β · NLL of gold tokens. log_probs at CE positions is log p_S(gold_t | prefix).
        if apply_pass_filter:
            ce_loss = beta * (-log_probs)
        else:
            ce_loss = torch.zeros_like(raw_kl)

        # ---- per-sample combination ----
        per_token_loss = is_kl_b * kl_loss + is_ce_b * ce_loss

        # ---- batch-level metrics ----
        metrics: dict[str, Any] = {}

        # Effective token / sample counts
        kl_token_mask = is_kl_b * response_mask_f
        ce_token_mask = is_ce_b * response_mask_f
        metrics["e1_v1/effective_kl_tokens"] = kl_token_mask.sum().item()
        if apply_pass_filter:
            metrics["e1_v1/effective_ce_tokens"] = ce_token_mask.sum().item()
            metrics["e1_v1/effective_ce_samples"] = is_ce_b.sum().item()

        # Loss-side contributions (pre-aggregation; raw sums for the ratio)
        kl_loss_sum = (kl_loss * kl_token_mask).sum().item()
        metrics["e1_v1/kl_loss_sum"] = kl_loss_sum
        if apply_pass_filter:
            ce_loss_sum = (ce_loss * ce_token_mask).sum().item()
            metrics["e1_v1/ce_loss_sum"] = ce_loss_sum
            total = kl_loss_sum + ce_loss_sum
            metrics["e1_v1/kl_ce_ratio"] = (kl_loss_sum / total) if total > 1e-12 else 0.0

        # delta_t monitors (B / D only)
        if apply_delta and delta_stats is not None:
            metrics["e1_v1/delta_t_mean_pre_norm"] = delta_stats["mean_pre_norm"]
            metrics["e1_v1/delta_t_p99_pre_norm"] = delta_stats["p99_pre_norm"]
            metrics["e1_v1/delta_t_mean_post_norm"] = delta_stats["mean_post_norm"]

        # trajectory_pass rate (filtered configs)
        if apply_pass_filter:
            metrics["e1_v1/trajectory_pass_rate"] = float(pass_arr.mean())

        # β echo (so eval logs can correlate gradients to the knob)
        if apply_pass_filter:
            metrics["e1_v1/beta"] = beta

        # Per-bucket breakdowns
        _emit_per_bucket(
            metrics,
            buckets=buckets,
            response_mask_f=response_mask_f,
            is_kl_b=is_kl_b,
            is_ce_b=is_ce_b,
            pass_rate_b=pass_rate_b,
            kl_loss_full=kl_loss,
            ce_loss_full=ce_loss,
        )

        return per_token_loss, metrics

    return loss_fn


# ===========================================================================
# Verl registration. Idempotent.
# ===========================================================================

_LOSSES_REGISTERED = False

_E1_ONPOLICY_LOSSES = [
    # (name, apply_delta, apply_pass_filter)
    ("e1_onpolicy_vanilla_kd",        False, False),
    ("e1_onpolicy_raw_delta_kd",      True,  False),
    ("e1_onpolicy_filtered_kd",       False, True),
    ("e1_onpolicy_filtered_delta_kd", True,  True),
]


def register_e1_onpolicy_losses() -> None:
    """Register the four E1 on-policy KD losses with verl's distillation registry."""
    global _LOSSES_REGISTERED
    if _LOSSES_REGISTERED:
        return

    from verl.trainer.distillation.losses import (
        DistillationLossSettings,
        register_distillation_loss,
    )

    for name, ud, up in _E1_ONPOLICY_LOSSES:
        fn = _build_loss_fn(apply_delta=ud, apply_pass_filter=up)
        register_distillation_loss(
            DistillationLossSettings(names=[name], use_topk=True)
        )(fn)

    _LOSSES_REGISTERED = True
    logger.info(
        "[delta_opd] registered 4 on-policy losses: %s", [n for n, _, _ in _E1_ONPOLICY_LOSSES]
    )


# ===========================================================================
# Local smoke test.
# Tests:
#   1. normalize_delta math (existing).
#   2. Per-sample KL/CE dispatch math (new): is_kl × kl_loss + is_ce × ce_loss
#      produces the right per-row totals.
#   3. β knob: CE loss scales linearly with β.
#   4. kl_ce_ratio computation.
# ===========================================================================

def _smoke_test() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    # --- 1. normalize_delta ---
    bsz, T = 4, 8
    delta = torch.zeros(bsz, T)
    delta[0, 2] = 5.0
    delta[1, 5] = 3.0
    delta[2, 0] = 1.0
    response_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 0],
         [1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1]],
        dtype=torch.bool,
    )
    w_t, stats = normalize_delta(delta, response_mask)
    assert abs(stats["mean_post_norm"] - 1.0) < 1e-4
    assert w_t.shape == delta.shape

    # --- 2. KL/CE dispatch math ---
    raw_kl = torch.full((bsz, T), 2.0)
    log_probs = torch.full((bsz, T), -1.0)
    beta = 0.1
    # samples 0,2 → kl; samples 1,3 → ce
    is_ce = torch.tensor([0.0, 1.0, 0.0, 1.0]).unsqueeze(-1)
    is_kl = 1.0 - is_ce
    kl_loss = raw_kl  # vanilla (no delta weighting)
    ce_loss = beta * (-log_probs)  # = 0.1 * 1.0 = 0.1 per token
    per_token = is_kl * kl_loss + is_ce * ce_loss
    # samples 0,2 → 2.0 per token; samples 1,3 → 0.1 per token
    assert torch.allclose(per_token[0], torch.full((T,), 2.0))
    assert torch.allclose(per_token[1], torch.full((T,), 0.1))
    assert torch.allclose(per_token[2], torch.full((T,), 2.0))
    assert torch.allclose(per_token[3], torch.full((T,), 0.1))

    # --- 3. β knob ---
    for b in [0.0, 0.05, 0.1, 0.3]:
        ce = b * (-log_probs)
        assert torch.allclose(ce, torch.full((bsz, T), b * 1.0))

    # --- 4. kl_ce_ratio ---
    response_mask_f = response_mask.float()
    kl_token_mask = is_kl * response_mask_f
    ce_token_mask = is_ce * response_mask_f
    kl_sum = (kl_loss * kl_token_mask).sum().item()
    ce_sum = (ce_loss * ce_token_mask).sum().item()
    ratio = kl_sum / (kl_sum + ce_sum)
    # kl: samples 0,2 with 5+2=7 valid tokens, each contributing 2.0 → 14.0
    # ce: samples 1,3 with 7+8=15 valid tokens, each contributing 0.1 → 1.5
    assert abs(kl_sum - 14.0) < 1e-5, kl_sum
    assert abs(ce_sum - 1.5) < 1e-5, ce_sum
    assert abs(ratio - 14.0 / 15.5) < 1e-5

    # --- 5. delta weighting on KL branch only ---
    delta_tt = torch.ones(bsz, T) * 2.0
    delta_tt[1] = 10.0  # CE sample has high delta — should NOT affect normalization
    kl_token_mask_for_norm = response_mask_f * is_kl
    w_t2, _ = normalize_delta(delta_tt, kl_token_mask_for_norm)
    # Over KL-branch valid tokens (samples 0,2): all delta=2.0 → uniform after clip+norm
    valid_kl = (response_mask_f * is_kl).bool()
    assert torch.allclose(w_t2[valid_kl], torch.ones_like(w_t2[valid_kl]), atol=1e-5)

    # --- 6. empty mask edge case ---
    empty_mask = torch.zeros_like(response_mask)
    w_empty, stats_empty = normalize_delta(delta, empty_mask)
    assert torch.allclose(w_empty, torch.ones_like(delta))

    print("losses.py smoke tests: 6/6 passed (on-policy v1 KL/CE dispatch)")


if __name__ == "__main__":
    _smoke_test()
