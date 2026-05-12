"""
Delta-OPD agent loop + worker subclass for E1 on-policy v1.

Two pieces of plumbing into verl:

1. ``DeltaOPDAgentLoop``  (subclass of ``SingleTurnAgentLoop``).
   Per-sample dispatch in ``run()``:

   * **KL branch** (T_correct=True, or recipe doesn't filter):
     standard vLLM student rollout. ``extra_fields["loss_branch"] = "kl"``.

   * **CE branch** (T_correct=False AND recipe filters):
     **no vLLM rollout**. We build the ``AgentLoopOutput`` by hand —
     ``response_ids = gold_token_ids`` (the answer the student should
     learn), ``response_mask = [1] * len(gold_token_ids)``,
     ``response_logprobs = None`` (no rollout happened),
     ``multi_modal_data`` keeps the real image so the student forward
     during training is image-conditioned. ``extra_fields["loss_branch"]
     = "ce"``.

   ``apply_pass_filter`` (= "filtered" in the loss_mode name) is read
   from the trainer config once at agent-loop init; "filtered" → C/D
   configs do filtering, A/B don't.

2. ``DeltaOPDAgentLoopWorker._compute_teacher_logprobs`` override.
   Three cases, depending on the sample's branch and whether delta is
   needed by the recipe:

   * **CE samples**: no teacher forward at all. We write dummies
     (pad_token_id ids, zero logprobs, zero delta_t) so the rest of the
     verl plumbing — which assumes ``teacher_ids`` / ``teacher_logprobs``
     are present whenever distillation is enabled — keeps flowing.
     The loss layer masks them out via ``is_kl``.

   * **KL samples, recipe doesn't need delta** (Config A "vanilla_kd"
     or Config C "filtered_kd"): image-conditioned teacher forward only.
     Skip the null forward — GPT-confirmed optimization (saves ~50%
     teacher time on A/C).

   * **KL samples, recipe needs delta** (Config B "raw_delta_kd" or
     Config D "filtered_delta_kd"): both image and null teacher
     forwards run concurrently via ``asyncio.gather``; per-position
     ``delta_t = KL_topK_union(P_T^I, P_T^null)`` is computed via E0's
     ``kl_topk_union``.

Index convention for ``delta_t`` (matches verl's ``extract_prompt_logprobs``):
position 0 of the returned teacher tensor is dropped by vLLM (no prior
context) and the last position is padded with a dummy, so entry ``i``
carries the distribution that predicts ``sequence_ids[i+1]``. To score
response token ``t`` (= ``sequence_ids[prompt_len + t]``) we read entry
``prompt_len + t - 1`` — equivalently, slice
``[prompt_len-1 : prompt_len-1+response_len]``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# E0 helper re-use (lazy import — keeps this module loadable without E0 deps).
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _import_kl_topk_union():
    from experiments.E0_image_null_delta.src.dual_forward import kl_topk_union

    return kl_topk_union


# ---------------------------------------------------------------------------
# Loss-mode introspection. The recipe yaml's loss_mode encodes the per-config
# behaviour, so we don't need extra config keys: "filtered" → apply_pass_filter,
# "delta" → need_delta.
# ---------------------------------------------------------------------------

def loss_mode_apply_pass_filter(loss_mode: str) -> bool:
    return "filtered" in loss_mode


def loss_mode_need_delta(loss_mode: str) -> bool:
    return "delta" in loss_mode


# ---------------------------------------------------------------------------
# Null image construction (same-resolution all-black; matches E0 mode="black").
# Same resolution preserves vision-token count → ``sequence_ids`` length is
# identical between image and null forwards.
# ---------------------------------------------------------------------------

def _make_null_multi_modal_data(multi_modal_data: Optional[dict]) -> Optional[dict]:
    if not multi_modal_data or "images" not in multi_modal_data:
        return multi_modal_data
    null = dict(multi_modal_data)
    null["images"] = [
        Image.new("RGB", img.size, color=(0, 0, 0)) for img in multi_modal_data["images"]
    ]
    return null


# ---------------------------------------------------------------------------
# kwargs unwrapping. RLHFDataset → AgentLoopWorker stashes non-tensor row
# fields in kwargs as numpy objects; cast defensively to native Python.
# ---------------------------------------------------------------------------

def _to_python_scalar(v, default=None):
    if v is None:
        return default
    if hasattr(v, "item"):
        return v.item()
    return v


def _to_python_list_int(v) -> list[int]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.astype(int).tolist()
    raise TypeError(f"cannot convert {type(v)} to list[int]")


# ---------------------------------------------------------------------------
# Per-position delta_t from two top-K teacher distributions.
# ---------------------------------------------------------------------------

def _compute_delta_t_response(
    teacher_ids_I: torch.Tensor,
    teacher_logprobs_I: torch.Tensor,
    teacher_ids_null: torch.Tensor,
    teacher_logprobs_null: torch.Tensor,
    prompt_len: int,
    response_len: int,
) -> np.ndarray:
    if response_len <= 0:
        return np.zeros((0,), dtype=np.float32)

    start = prompt_len - 1
    end = start + response_len
    kl_topk_union = _import_kl_topk_union()
    lp_I = teacher_logprobs_I[start:end].tolist()
    ids_I = teacher_ids_I[start:end].tolist()
    lp_n = teacher_logprobs_null[start:end].tolist()
    ids_n = teacher_ids_null[start:end].tolist()

    delta = [
        kl_topk_union(lp_I[t], ids_I[t], lp_n[t], ids_n[t]) for t in range(response_len)
    ]
    return np.asarray(delta, dtype=np.float32)


def _pad_delta_to_response_width(
    delta_response: np.ndarray, response_width: int
) -> np.ndarray:
    out = np.zeros((response_width,), dtype=np.float32)
    if delta_response.size:
        copy_len = min(delta_response.size, response_width)
        out[:copy_len] = delta_response[:copy_len]
    return out


# ---------------------------------------------------------------------------
# Patched ``AgentLoopWorker._compute_teacher_logprobs``.
# Looks at ``output.extra_fields["loss_branch"]`` (set by DeltaOPDAgentLoop.run)
# and dispatches: CE → dummies, KL+no-delta → single image forward, KL+delta →
# concurrent image+null forwards.
# ---------------------------------------------------------------------------

def _resolve_routing_key(teacher_key: str, sample_kwargs: Optional[dict]) -> Optional[Any]:
    if sample_kwargs is None:
        return None
    value = sample_kwargs.get(teacher_key)
    if value is None:
        return None
    return value.item() if hasattr(value, "item") else value


def _topk(self) -> int:
    return int(self.distillation_config.distillation_loss.topk)


def _write_ce_dummies(
    self,
    output,
    prompt_ids: list[int],
    response_ids: list[int],
) -> None:
    """For CE samples, fill teacher_ids/logprobs/delta_t with shapes verl expects.

    verl's ``_pad_teacher_outputs`` pads ``(seq_len, K)`` to
    ``(prompt_width + response_width, K)``; we supply the unpadded shape
    (= ``seq_len`` rows, K columns) with pad_token_id ids and zero logprobs.
    ``delta_t`` is response-aligned (= ``response_width``).
    """
    K = _topk(self)
    seq_len = len(prompt_ids) + len(response_ids)
    pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
    response_width = self.rollout_config.response_length

    output.extra_fields["teacher_ids"] = torch.full((seq_len, K), pad_id, dtype=torch.int32)
    output.extra_fields["teacher_logprobs"] = torch.zeros((seq_len, K), dtype=torch.float32)
    output.extra_fields["delta_t"] = np.zeros((response_width,), dtype=np.float32)


async def _dual_forward_compute_teacher_logprobs(
    self,
    output,
    prompt_ids: list[int],
    response_ids: list[int],
    validate: bool,
    sample_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    """Replacement for ``AgentLoopWorker._compute_teacher_logprobs``."""
    if not getattr(self, "distillation_enabled", False) or validate:
        return

    loss_branch = output.extra_fields.get("loss_branch", "kl")
    if loss_branch == "ce":
        _write_ce_dummies(self, output, prompt_ids, response_ids)
        return

    loss_mode = self.config.distillation.distillation_loss.loss_mode
    need_delta = loss_mode_need_delta(loss_mode)

    routing_key = _resolve_routing_key(getattr(self, "teacher_key", "data_source"), sample_kwargs)
    sequence_ids = prompt_ids + response_ids
    prompt_len = len(prompt_ids)
    response_len = len(response_ids)
    response_width = self.rollout_config.response_length

    image_call = self.teacher_server_manager.compute_teacher_logprobs_single(
        sequence_ids=sequence_ids,
        multi_modal_data=output.multi_modal_data,
        routing_key=routing_key,
    )

    if need_delta:
        null_mm = _make_null_multi_modal_data(output.multi_modal_data)
        null_call = self.teacher_server_manager.compute_teacher_logprobs_single(
            sequence_ids=sequence_ids,
            multi_modal_data=null_mm,
            routing_key=routing_key,
        )
        (teacher_ids_I, teacher_logprobs_I), (teacher_ids_null, teacher_logprobs_null) = (
            await asyncio.gather(image_call, null_call)
        )
        if teacher_ids_I.shape[0] != teacher_ids_null.shape[0]:
            raise ValueError(
                "Image and null teacher returned different sequence lengths "
                f"({teacher_ids_I.shape[0]} vs {teacher_ids_null.shape[0]}); "
                "same-resolution null should preserve length."
            )
        delta_response = _compute_delta_t_response(
            teacher_ids_I=teacher_ids_I,
            teacher_logprobs_I=teacher_logprobs_I,
            teacher_ids_null=teacher_ids_null,
            teacher_logprobs_null=teacher_logprobs_null,
            prompt_len=prompt_len,
            response_len=response_len,
        )
        delta_padded = _pad_delta_to_response_width(delta_response, response_width)
    else:
        # A/C: image forward only. delta_t isn't used by the loss; emit zeros
        # so the loss-side stacking shape is consistent across the batch.
        teacher_ids_I, teacher_logprobs_I = await image_call
        delta_padded = np.zeros((response_width,), dtype=np.float32)

    output.extra_fields["teacher_ids"] = teacher_ids_I
    output.extra_fields["teacher_logprobs"] = teacher_logprobs_I
    output.extra_fields["delta_t"] = delta_padded


# ---------------------------------------------------------------------------
# DeltaOPDAgentLoop (registration deferred to ``register_delta_opd_agent_loop``).
# ---------------------------------------------------------------------------

_AGENT_LOOP_REGISTERED = False
_PATCH_INSTALLED = False


def register_delta_opd_agent_loop() -> type:
    """Register ``DeltaOPDAgentLoop`` under ``delta_opd_single_turn`` (idempotent)."""
    global _AGENT_LOOP_REGISTERED
    if _AGENT_LOOP_REGISTERED:
        return globals()["DeltaOPDAgentLoop"]

    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopMetrics,
        AgentLoopOutput,
        register,
    )
    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

    class DeltaOPDAgentLoop(SingleTurnAgentLoop):
        """Single-turn rollout with per-sample KL/CE dispatch for Delta-OPD."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            loss_mode = self.config.distillation.distillation_loss.loss_mode
            self._apply_pass_filter = loss_mode_apply_pass_filter(loss_mode)
            self._need_delta = loss_mode_need_delta(loss_mode)
            # β read once from the config; injected per-sample into extra_fields
            # so the loss layer can access it without a separate plumbing path.
            try:
                self._beta = float(self.config.e1.beta)
            except Exception:
                self._beta = 0.1
                logger.warning(
                    "[delta_opd] config.e1.beta not found; defaulting to 0.1. "
                    "Set it in recipe yaml's e1: {beta: <float>} block."
                )

        def _decide_loss_branch(self, kwargs: dict) -> str:
            """Per-sample branch: 'ce' iff recipe filters AND teacher was wrong."""
            if not self._apply_pass_filter:
                return "kl"
            t_pass = _to_python_scalar(kwargs.get("trajectory_pass"), default=True)
            return "kl" if bool(t_pass) else "ce"

        async def _run_ce_branch(self, kwargs: dict) -> AgentLoopOutput:
            """Build an ``AgentLoopOutput`` for a CE sample (no vLLM rollout)."""
            messages = list(kwargs["raw_prompt"])
            multi_modal_data = await self.process_vision_info(messages)
            images = multi_modal_data.get("images")
            videos = multi_modal_data.get("videos")
            prompt_ids = await self.apply_chat_template(
                messages, images=images, videos=videos
            )

            gold_token_ids = _to_python_list_int(kwargs.get("gold_token_ids"))
            if not gold_token_ids:
                # Safety: gold missing — should have been filtered at parquet build time.
                # Fall back to a 1-token EOS so verl's padding doesn't choke; loss will
                # contribute almost nothing.
                eos = self.tokenizer.eos_token_id or 0
                gold_token_ids = [eos]
                logger.warning(
                    "[delta_opd] CE sample has no gold_token_ids; using [eos] fallback."
                )

            # Truncate to fit configured response width.
            if len(gold_token_ids) > self.response_length:
                gold_token_ids = gold_token_ids[: self.response_length]

            response_mask = [1] * len(gold_token_ids)

            output = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=gold_token_ids,
                response_mask=response_mask,
                response_logprobs=None,
                multi_modal_data=multi_modal_data,
                num_turns=1,
                metrics=AgentLoopMetrics(),
                extra_fields={
                    # Shape-consistent with SingleTurnAgentLoop's run() output.
                    "turn_scores": [],
                    "tool_rewards": [],
                },
            )
            self._attach_e1_extra_fields(output, kwargs, loss_branch="ce")
            return output

        def _attach_e1_extra_fields(
            self, output: AgentLoopOutput, kwargs: dict, loss_branch: str
        ) -> None:
            """Plumb per-sample E1 metadata so the loss layer can read it.

            ``loss_branch``, ``bucket``, ``beta`` flow through extra_fields →
            non_tensor_batch → DataProto.to_tensordict → NonTensorStack in the
            loss's TensorDict.
            """
            output.extra_fields["loss_branch"] = loss_branch
            output.extra_fields["bucket"] = str(
                _to_python_scalar(kwargs.get("bucket"), default="unknown")
            )
            output.extra_fields["beta"] = float(self._beta)
            # Also stash trajectory_pass for monitoring / per-bucket teacher_correct_rate.
            output.extra_fields["trajectory_pass"] = bool(
                _to_python_scalar(kwargs.get("trajectory_pass"), default=True)
            )

        async def run(self, sampling_params, **kwargs):  # type: ignore[override]
            loss_branch = self._decide_loss_branch(kwargs)
            if loss_branch == "ce":
                return await self._run_ce_branch(kwargs)

            output = await super().run(sampling_params, **kwargs)
            self._attach_e1_extra_fields(output, kwargs, loss_branch="kl")
            return output

    DeltaOPDAgentLoop.__qualname__ = "DeltaOPDAgentLoop"
    globals()["DeltaOPDAgentLoop"] = DeltaOPDAgentLoop
    register("delta_opd_single_turn")(DeltaOPDAgentLoop)
    _AGENT_LOOP_REGISTERED = True
    return DeltaOPDAgentLoop


def apply_delta_opd_worker_patch() -> None:
    """Replace ``AgentLoopWorker._compute_teacher_logprobs`` with the dispatch version."""
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return

    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    original = AgentLoopWorker._compute_teacher_logprobs
    AgentLoopWorker._compute_teacher_logprobs = _dual_forward_compute_teacher_logprobs
    AgentLoopWorker._compute_teacher_logprobs_original = original

    # Also stash distillation_config on the worker (the patched method reads
    # config.distillation.distillation_loss.topk via self.distillation_config).
    # Worker already has self.config; we add a property-like alias.
    _ensure_distillation_config_property(AgentLoopWorker)

    _PATCH_INSTALLED = True
    logger.info(
        "[delta_opd] patched AgentLoopWorker._compute_teacher_logprobs "
        "for KL/CE dispatch + selective null forward"
    )


def _ensure_distillation_config_property(AgentLoopWorker_cls) -> None:
    """Make ``self.distillation_config`` accessible on the worker (read-only)."""
    if hasattr(AgentLoopWorker_cls, "distillation_config"):
        return
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.config import DistillationConfig

    def _get_distillation_config(self):
        # Cache on first access; dataclass conversion is non-trivial.
        cached = self.__dict__.get("_e1_distillation_config")
        if cached is None:
            cached = omega_conf_to_dataclass(self.config.distillation)
            self.__dict__["_e1_distillation_config"] = cached
        return cached

    AgentLoopWorker_cls.distillation_config = property(_get_distillation_config)


def enable_delta_opd() -> None:
    """Apply the worker patch and register the agent loop (idempotent)."""
    apply_delta_opd_worker_patch()
    register_delta_opd_agent_loop()


def __getattr__(name: str):
    """Materialize the lazily defined Hydra target on demand."""
    if name == "DeltaOPDAgentLoop":
        return register_delta_opd_agent_loop()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from verl.experimental.agent_loop.agent_loop import (  # noqa: E402
    AgentLoopManager as _VerlAgentLoopManager,
    AgentLoopWorker as _VerlAgentLoopWorker,
)


class DeltaOPDAgentLoopWorker(_VerlAgentLoopWorker):
    """AgentLoopWorker variant with E1 teacher-logprob dispatch."""

    @property
    def distillation_config(self):
        cached = self.__dict__.get("_e1_distillation_config")
        if cached is None:
            from verl.utils.config import omega_conf_to_dataclass

            cached = omega_conf_to_dataclass(self.config.distillation)
            self.__dict__["_e1_distillation_config"] = cached
        return cached

    async def _compute_teacher_logprobs(
        self,
        output,
        prompt_ids: list[int],
        response_ids: list[int],
        validate: bool,
        sample_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        await _dual_forward_compute_teacher_logprobs(
            self,
            output=output,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            validate=validate,
            sample_kwargs=sample_kwargs,
        )


class DeltaOPDAgentLoopManager(_VerlAgentLoopManager):
    """AgentLoopManager that launches the E1 worker subclass."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        import ray

        self.agent_loop_workers_class = ray.remote(DeltaOPDAgentLoopWorker)
