"""
E1 on-policy v1 trainer plumbing.

Public API:

* ``enable()`` — driver-side: registers the agent loop, applies the worker
  monkey-patch, registers the 4 losses, AND installs the "late hook" so
  Ray workers register on their own.

* ``install_late_hook()`` — runs only the late hook (monkey-patches
  ``DistillationLossConfig.__post_init__`` to lazy-register on first
  lookup). Workers call this themselves.

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

The late hook patches ``DistillationLossConfig.__post_init__`` so it
lazy-imports + registers our losses on first lookup. This patch
installation only imports ``verl.workers.config``, which does NOT cascade
into ``torch.cuda`` calls. The patched body fires inside
``WorkerDict.actor_rollout_init_model`` — well after Ray has assigned the
GPU to this actor — so the unavoidable torch.cuda init from the cascade
happens with the right ``CUDA_VISIBLE_DEVICES`` already in place.
"""

from __future__ import annotations

_LATE_HOOK_INSTALLED = False


def install_late_hook() -> None:
    """Patch ``DistillationLossConfig.__post_init__`` to lazy-register E1 losses.

    Idempotent. Safe to call from both the driver (via ``enable()``) and
    from every Ray worker process (via a no-import-required runtime_env
    setup). The patch body only fires when verl actually resolves a
    DistillationLossConfig — by then the actor is post-CVD-assignment.

    Imports here intentionally avoid torch.cuda-init-heavy verl modules.
    ``verl.workers.config`` only imports omegaconf + dataclass machinery.
    """
    global _LATE_HOOK_INSTALLED
    if _LATE_HOOK_INSTALLED:
        return

    from verl.workers.config import DistillationLossConfig

    original_post_init = DistillationLossConfig.__post_init__

    def _patched_post_init(self):
        loss_mode = getattr(self, "loss_mode", "")
        if loss_mode.startswith("e1_onpolicy_"):
            # Late import — runs inside an actor that has its GPU assigned, so
            # the verl import cascade's torch.cuda init sees the correct CVD.
            from verl.trainer.distillation.losses import DISTILLATION_LOSS_REGISTRY
            if loss_mode not in DISTILLATION_LOSS_REGISTRY:
                from .losses import register_e1_onpolicy_losses
                register_e1_onpolicy_losses()
        # Delegate to the original (which sets loss_settings, validates, etc.)
        original_post_init(self)

    DistillationLossConfig.__post_init__ = _patched_post_init
    _LATE_HOOK_INSTALLED = True


def enable() -> None:
    """Wire Delta-OPD into verl (idempotent).

    - Driver-side: registers ``DeltaOPDAgentLoop`` + applies worker patch +
      registers the 4 losses immediately (driver process is already past
      CUDA init, so the verl import cascade is harmless).
    - Installs the late hook so Ray workers register on their own at the
      right time (after each actor has its GPU assigned by Ray).
    """
    from .agent_loop import enable_delta_opd
    from .losses import register_e1_onpolicy_losses

    enable_delta_opd()
    register_e1_onpolicy_losses()
    install_late_hook()


__all__ = ["enable", "install_late_hook"]
