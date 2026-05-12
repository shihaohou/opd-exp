"""
E1 LOSSES — smoke-test baselines, NOT the main E1 experiment.

⚠️  Read before using:

The losses in this file are **off-policy weighted SFT** — student is
force-scored on the *teacher's* greedy response, and per-token NLL is
re-weighted by `delta_t` and/or masked by `trajectory_pass`. This is
mathematically a sequence-level SFT-on-teacher-mode objective with token
re-weighting, NOT on-policy distillation. The Delta-OPD project name's
"OPD" stands for On-Policy Distillation; the real E1 experiment uses
student rollouts and a teacher dual-forced-score path (see
`on_policy_v1_design.md` for the design).

These losses exist for one purpose only:

    100–300 step engineering smoke test to validate
      - data loader + mixture sampling
      - per-bucket monitoring hooks
      - delta_t clipping / normalization stability
      - CE-on-gold vs KL loss-magnitude balance
      - TEI eval pipeline can read this trainer's checkpoints

They are NOT the scientific result of E1. Results from a 4-config run
using THESE losses cannot answer the central E1 question (whether OPD
teacher-error inheritance happens on student-visited states, and whether
filtered Delta-OPD blocks it). For that, the on-policy v1 loop is
required.

# Math (off-policy MC estimator of KL(p_T || p_S))

For each training sample we have teacher's greedy response y_T (precomputed)
and student's per-token log-prob on those same tokens. The single-sample
MC estimator of KL(p_T || p_S) when y_T ~ p_T is

    L = -student_logp[t] + const

with delta-weighting / filtering applied multiplicatively. teacher_logp
does not appear in the loss (it is a constant w.r.t. student parameters).

# Per-sample dispatch via precompute conventions

Filtered configs need CE-on-gold for teacher-wrong samples. Precompute
should write:

    if trajectory_pass:  (teacher got the answer right)
        response_token_ids = teacher's greedy response
        delta_t            = real per-token delta_t values
        trajectory_pass    = 1
    else:  (teacher wrong; only used by filtered configs)
        response_token_ids = gold answer tokens
        delta_t            = uniform 1.0 across gold trajectory
        trajectory_pass    = 1   (we DO want this sample's CE gradient)

This way each loss is a per-token product of weights × NLL.
"""

from __future__ import annotations

from typing import Any

import torch


# ===========================================================================
# Pure-math helpers (no verl dependency — unit-testable on any machine)
# ===========================================================================

def weighted_token_nll(
    student_log_probs: torch.Tensor,
    delta_t: torch.Tensor | None = None,
    trajectory_pass: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-token weighted NLL.

    Args:
        student_log_probs: (bsz, resp_len) — student's log-prob at the
            chosen token (teacher's response for KL samples, gold for CE
            samples).
        delta_t: (bsz, resp_len) or None. None → uniform 1.0.
        trajectory_pass: (bsz,) or (bsz, 1) bool/float or None. None → all 1.

    Returns:
        (bsz, resp_len) per-token loss tensor. Caller (verl's `agg_loss`)
        applies `response_mask` reduction.
    """
    losses = -student_log_probs

    if delta_t is not None:
        if delta_t.shape != losses.shape:
            raise ValueError(
                f"delta_t shape {tuple(delta_t.shape)} does not match "
                f"losses shape {tuple(losses.shape)}"
            )
        losses = losses * delta_t

    if trajectory_pass is not None:
        tp = trajectory_pass.float()
        if tp.dim() == 1:
            tp = tp.unsqueeze(-1)
        losses = losses * tp

    return losses


# ===========================================================================
# Verl loss registry — register the 4 SMOKE-BASELINE losses with explicit
# `offline_weighted_sft` naming so they cannot be confused with on-policy
# OPD losses (to be added in a separate `on_policy_losses.py` file once
# the on-policy v1 design is wired in).
# ===========================================================================

def _verl_imports():
    from verl.trainer.distillation.losses import (
        register_distillation_loss,
        DistillationLossSettings,
    )
    from verl.workers.utils.padding import no_padding_2_padding
    return register_distillation_loss, DistillationLossSettings, no_padding_2_padding


def _build_loss_fn(use_delta: bool, use_pass_mask: bool):
    """Factory returning a verl-compatible loss function with given knobs."""
    _, _, no_padding_2_padding = _verl_imports()

    def loss_fn(config, distillation_config, model_output, data) -> tuple[torch.Tensor, dict]:
        student_log_probs = no_padding_2_padding(model_output["log_probs"], data)

        delta_t = None
        if use_delta:
            delta_t = no_padding_2_padding(data["delta_t"], data)
            if delta_t.dim() == 3 and delta_t.shape[-1] == 1:
                delta_t = delta_t.squeeze(-1)

        pass_mask = None
        if use_pass_mask:
            pass_mask = data["trajectory_pass"]

        losses = weighted_token_nll(
            student_log_probs=student_log_probs,
            delta_t=delta_t,
            trajectory_pass=pass_mask,
        )

        metrics: dict[str, Any] = {}
        if use_delta and delta_t is not None:
            metrics["e1_smoke/delta_t_mean"] = delta_t.detach().mean().item()
            metrics["e1_smoke/delta_t_var"] = delta_t.detach().var().item()
        if use_pass_mask and pass_mask is not None:
            tp = pass_mask.float()
            metrics["e1_smoke/trajectory_pass_rate"] = tp.mean().item()

        return losses, metrics

    return loss_fn


def register_offline_smoke_losses() -> None:
    """Register the 4 off-policy smoke-baseline configs.

    Naming is deliberately verbose (`e1_offline_weighted_sft_*`) so config
    files distinguish these from on-policy losses without ambiguity. The
    real E1 experiment will register a separate set of names like
    `e1_onpolicy_vanilla_kd`, `e1_onpolicy_raw_delta_kd`, etc.
    """
    global _LOSSES_REGISTERED
    if _LOSSES_REGISTERED:
        return

    register_distillation_loss, DistillationLossSettings, _ = _verl_imports()

    configs = [
        # name                                                   use_delta  use_pass_mask
        ("e1_offline_weighted_sft_vanilla",                      False,     False),
        ("e1_offline_weighted_sft_raw_delta",                    True,      False),
        ("e1_offline_weighted_sft_correct_filtered",             False,     True),
        ("e1_offline_weighted_sft_correct_filtered_delta",       True,      True),
    ]
    for name, ud, up in configs:
        fn = _build_loss_fn(use_delta=ud, use_pass_mask=up)
        register_distillation_loss(
            DistillationLossSettings(names=[name], use_estimator=True)
        )(fn)

    _LOSSES_REGISTERED = True


_LOSSES_REGISTERED = False


# ===========================================================================
# Local smoke test (run as: `python experiments/E1_filtered_delta_opd/src/losses.py`)
# Tests pure math without requiring verl.
# ===========================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    bsz, T = 4, 6
    student_lp = torch.randn(bsz, T)
    delta = torch.rand(bsz, T)
    pass_mask = torch.tensor([1, 0, 1, 1], dtype=torch.float32)

    out = weighted_token_nll(student_lp)
    assert torch.allclose(out, -student_lp), "vanilla failed"

    out = weighted_token_nll(student_lp, delta_t=delta)
    assert torch.allclose(out, -student_lp * delta), "delta-weight failed"

    out = weighted_token_nll(student_lp, trajectory_pass=pass_mask)
    expected = -student_lp * pass_mask[:, None]
    assert torch.allclose(out, expected), "pass_mask failed"
    assert torch.allclose(out[1], torch.zeros(T)), "pass_mask zero-row failed"

    out = weighted_token_nll(student_lp, delta_t=delta, trajectory_pass=pass_mask)
    expected = -student_lp * delta * pass_mask[:, None]
    assert torch.allclose(out, expected), "combined failed"

    try:
        weighted_token_nll(student_lp, delta_t=torch.rand(bsz, T + 1))
    except ValueError as e:
        print(f"  ok — shape mismatch correctly raised: {e}")
    else:
        raise AssertionError("expected ValueError on shape mismatch")

    print("losses.py smoke tests: 5/5 passed (offline_weighted_sft baseline)")
