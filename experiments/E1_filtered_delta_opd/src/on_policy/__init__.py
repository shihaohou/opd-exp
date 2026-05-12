"""
E1 on-policy v1 trainer plumbing.

Public API:
    * ``enable()`` — applies the ``AgentLoopWorker._compute_teacher_logprobs``
      monkey-patch (so teacher dual-forward + ``delta_t`` happens at training
      time) and registers ``DeltaOPDAgentLoop`` + the 4 KD losses with verl.

Call ``enable()`` from your entrypoint *before* invoking
``verl.trainer.main_ppo.main()``. The provided launcher
``experiments/E1_filtered_delta_opd/src/on_policy/entrypoint.py`` does
exactly that.
"""

from __future__ import annotations


def enable() -> None:
    """Wire Delta-OPD into verl (idempotent).

    - Monkey-patches ``AgentLoopWorker._compute_teacher_logprobs`` to do
      teacher dual-forward + per-token ``delta_t``.
    - Registers ``DeltaOPDAgentLoop`` (``delta_opd_single_turn``).
    - Registers the 4 on-policy KD losses (``e1_onpolicy_*``).
    """
    # Imports happen here, not at module top, so the package is inspectable
    # on machines without verl (e.g., the developer's Mac).
    from .agent_loop import enable_delta_opd
    from .losses import register_e1_onpolicy_losses

    enable_delta_opd()
    register_e1_onpolicy_losses()


__all__ = ["enable"]
