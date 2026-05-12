"""
E1 on-policy v1 trainer plumbing.

Public API:

* ``enable()`` — driver-side: registers the agent loop, applies the worker
  monkey-patch, and registers the 4 losses.

* ``install_late_hook()`` — compatibility no-op. Do not use Ray
  ``worker_process_setup_hook`` for E1 wiring.

Why a "late hook"
-----------------
Earlier we tried registering via Ray's ``runtime_env.worker_process_setup_hook``,
which fires at worker process startup. That's too early: Ray assigns
``CUDA_VISIBLE_DEVICES`` to the worker only after the hook runs, so any
``torch.cuda.*`` call inside the hook (or its transitive imports — verl's
``verl.utils.device.get_device_capability`` will do one) caches an 8-GPU
view. By the time Ray sets ``CUDA_VISIBLE_DEVICES='N'``, it's already too
late — torch.cuda still reports 8 GPUs, ``current_device()`` falls to 0
for every worker, and FSDP's 4 ranks all map to physical GPU 0
(``Duplicate GPU detected``).

The replacement wiring avoids Ray setup hooks entirely:

* E1 losses are lazy-registered from ``verl.workers.config.distillation``
  when an E1 ``loss_mode`` is materialized.
* The E1 agent-loop worker behavior is selected via
  ``rollout.agent.agent_loop_manager_class``.
"""

from __future__ import annotations

_LATE_HOOK_INSTALLED = False


def install_late_hook() -> None:
    """Compatibility no-op for older configs.

    Importing ``verl.workers.config`` here is not CUDA-safe: importing any
    ``verl.*`` submodule first executes ``verl/__init__.py``, which imports
    ``verl.utils.device`` and calls ``torch.cuda.is_available()`` at module
    import time. Under Ray ``worker_process_setup_hook`` this can happen
    before Ray narrows ``CUDA_VISIBLE_DEVICES`` for the assigned GPU.
    """
    global _LATE_HOOK_INSTALLED
    _LATE_HOOK_INSTALLED = True


def enable() -> None:
    """Wire Delta-OPD into verl (idempotent).

    Driver-side: registers ``DeltaOPDAgentLoop`` + applies the local worker
    patch + registers the 4 losses immediately.
    """
    from .agent_loop import enable_delta_opd
    from .losses import register_e1_onpolicy_losses

    enable_delta_opd()
    register_e1_onpolicy_losses()


__all__ = ["enable", "install_late_hook"]
