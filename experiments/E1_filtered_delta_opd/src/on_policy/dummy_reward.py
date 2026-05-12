"""
Dummy reward function for E1 on-policy distillation.

Why this exists
---------------
verl's AgentLoop unconditionally calls ``_compute_score`` after each
rollout, which routes through ``default_compute_score`` and matches on
``data_source``. Our parquet uses ``data_source='e1_virl39k'`` (and will
add ``e1_pope_style`` / ``e1_synth`` / ``e1_tallyqa`` in Day 2), none of
which are registered in verl's hardcoded reward registry, so the run
crashes with ``NotImplementedError: Reward function is not implemented
for data_source='e1_virl39k'`` before the first training step.

For Delta-OPD we set ``distillation.distillation_loss.use_task_rewards=
false`` — reward is irrelevant to the loss; only distillation contributes
to the gradient. So we just need verl's reward plumbing to return
*something* (any finite number) and move on.

Wiring
------
In ``e1_base.yaml``:
    custom_reward_function:
      path: experiments/E1_filtered_delta_opd/src/on_policy/dummy_reward.py
      name: compute_score

verl's RewardLoopWorker then imports this module and calls
``compute_score`` instead of consulting ``default_compute_score``.

Signature matches verl's contract (see
``verl/experimental/reward_loop/reward_manager/naive.py``):
positional ``data_source``, ``solution_str``, ``ground_truth``,
optional ``extra_info``, plus ``**kwargs`` for fields verl may add
later (reward_router_address, reward_model_tokenizer, etc.).
"""

from __future__ import annotations

from typing import Any


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> float:
    """Always 0. Distillation loss carries the actual gradient signal.

    Returning a scalar (not a dict) keeps the downstream ``score = result``
    branch in ``naive.py`` simple; verl casts to float either way.
    """
    return 0.0
