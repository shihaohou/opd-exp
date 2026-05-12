"""
E1 distillation losses — placeholder.

Will register the four E1 configs into verl's
`verl.trainer.distillation.DISTILLATION_LOSS_REGISTRY`:

    vanilla_opd                   per-token reverse-KL
    raw_delta_opd                 per-token reverse-KL  × delta_t
    correct_filtered_opd          per-token reverse-KL  × trajectory_pass(sample)
                                   + CE(gold_tokens)    where trajectory_pass == 0
    correct_filtered_delta_opd    per-token reverse-KL  × delta_t × trajectory_pass
                                   + CE(gold_tokens)    where trajectory_pass == 0

All four sit on top of verl's existing reverse-KL estimator
(`compute_distillation_loss_reverse_kl_estimator`) which uses single-sample
`kl_penalty(student_logp, teacher_logp, mode)`. Mode default = k1 (matches the
classic KL definition; k2/k3 are biased variants).

**Why this file is a stub**: the four losses each multiply a per-token weight
into the existing reverse-KL kernel. The interesting part — how `delta_t` and
`trajectory_pass` arrive in the loss function as fields of the `data`
TensorDict — depends on verl's data-collation path inside the FSDP actor
worker (in particular how `rl_dataset.RLHFDataset` packs extra columns onto
the DataProto). Reading that plumbing is a precondition for writing this
file correctly.

Equally open: the CE-on-gold branch for `correct_filtered_*` configs needs
either (a) a per-sample sub-batch split + dispatch in the trainer, or (b) a
two-trajectory data shape where each sample carries BOTH the teacher's
response and the gold answer. Both are tractable; both require trainer-level
work first.

TODO (after the verl data-flow spike):
1. Locate where verl FSDP actor reads `data["teacher_logprobs"]` and confirm
   the path for adding `data["delta_t"]` + `data["trajectory_pass"]`.
2. Decide CE-on-gold dispatch: trainer-level sub-batch split (preferred for
   simplicity) vs in-loss per-sample branching.
3. Write the four `@register_distillation_loss(...)` functions.
4. Smoke-test on a 1K subset: vanilla_opd should reproduce verl's existing
   `k1` mode numerics within fp32 epsilon. correct_filtered_* should produce
   well-defined gradients on a mix of teacher-correct and teacher-wrong
   samples.
"""

# Intentionally empty. See module docstring.
