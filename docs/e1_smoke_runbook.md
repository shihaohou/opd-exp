# E1 on-policy smoke runbook

Companion to `docs/migrate-env.md`. Where migrate-env covers the **per-machine
environment traps**, this file covers the **per-stage integration traps**
that surfaced while wiring `experiments/E1_filtered_delta_opd/src/on_policy/`
into verl's FSDP distillation path. All five issues below were hit on
`arc-wlf1-ge103-1` between Day 1 and Day 1.5 (2026-05-12). Config A smoke
(`Vanilla KD`, 50 ViRL39K samples, 12 train steps + validation + checkpoint)
runs end-to-end with the mitigations in place.

## Quick reference: which problem are you looking at?

| Symptom | R |
|---|---|
| `NCCL error: Duplicate GPU detected : rank N and rank 0 both on CUDA device <id>` at FSDP init | [R1](#r1-fsdp-ranks-all-bind-to-the-same-physical-gpu) |
| Hydra error: `Could not instantiate target ... .<locals>.DeltaOPDAgentLoop` | [R2](#r2-hydra-instantiate-rejects-functionlocal-class-as-_target_) |
| `RuntimeError: cudnn_status: CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` during `compute_log_prob` / FSDP exit | [R3](#r3-cudnn-v9-sublibrary-load-failure-in-qwen25-vl-conv3d) |
| `filter dataset len: 0` → `ValueError: batch_size should be a positive integer value, but got batch_size=0` | [R4](#r4-multimodal-prompts-truncated-out-of-the-dataset) |
| `NotImplementedError: Reward function is not implemented for data_source='e1_virl39k'` | [R5](#r5-verl-rewards-registry-doesnt-know-e1_-data_sources) |

---

## R1: FSDP ranks all bind to the same physical GPU

### Symptom
```
NCCL error in: ... NCCLUtils.cpp:77, invalid usage
ncclInvalidUsage: This usually reflects invalid usage of NCCL library.
Last error:
Duplicate GPU detected : rank 1 and rank 0 both on CUDA device <bus_id>
Duplicate GPU detected : rank 2 and rank 0 both on CUDA device <bus_id>
Duplicate GPU detected : rank 3 and rank 0 both on CUDA device <bus_id>
```
Fires inside `FSDP._init_param_handle_from_module → _sync_params_and_buffers → dist._broadcast_coalesced` at student model init.

### Trigger
The Delta-OPD package used `Ray.runtime_env.worker_process_setup_hook` to register E1 losses + the agent-loop subclass in every Ray worker. The hook function called `enable()`, which transitively imported `verl.trainer.distillation.losses` → `verl.utils.device.get_device_capability()`. That helper calls `torch.cuda.is_available()` at module-load time, which **initializes the CUDA driver before Ray narrows `CUDA_VISIBLE_DEVICES` for the actor**. Torch caches an 8-GPU view; Ray's later `CUDA_VISIBLE_DEVICES='N'` does nothing because the cache is frozen. Every FSDP rank then defaults to logical `cuda:0`, which now refers to physical GPU 0 for all of them.

### Verification
Add a debug print at the entry of `verl/workers/engine_workers.py:WorkerDict.actor_rollout_init_model`:
```python
import os, torch
print(f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')!r}"
      f" device_count={torch.cuda.device_count()}"
      f" current={torch.cuda.current_device()}", flush=True)
```
A poisoned worker prints `CVD='2' device_count=8 current=0`. A healthy worker prints `CVD='2' device_count=1 current=0`. A standalone `@ray.remote(num_gpus=1)` probe will be healthy even when verl is poisoned — confirming Ray itself is fine, the import order is the bug.

### Fix
Remove the Ray setup hook entirely. Lazy-register E1 losses from inside `verl.workers.config.distillation.DistillationLossConfig.__post_init__` (which runs in the actor only after Ray has set `CUDA_VISIBLE_DEVICES`):
```python
# verl/workers/config/distillation.py
def __post_init__(self):
    self._mutable_fields.add("loss_settings")
    if self.loss_mode.startswith("e1_onpolicy_"):
        from experiments.E1_filtered_delta_opd.src.on_policy.losses import register_e1_onpolicy_losses
        register_e1_onpolicy_losses()
    from verl.trainer.distillation.losses import get_distillation_loss_settings
    self.loss_settings = get_distillation_loss_settings(self.loss_mode)
```
The agent loop subclass goes through `rollout.agent.agent_loop_manager_class` (Hydra `_target_`), which is also actor-scope.

Carried in `verl@2c118243` (branch `e1-loss-lazy-register`) + opd-exp `022c465f`.

### Long-term
None needed — the lazy-register pattern is the right answer. Treat this as a permanent rule: **do not use `worker_process_setup_hook` for anything that imports verl**, ever.

---

## R2: Hydra `instantiate` rejects function-local class as `_target_`

### Symptom
```
hydra.errors.InstantiationException: Error in call to target
'experiments.E1_filtered_delta_opd.src.on_policy.agent_loop.register_delta_opd_agent_loop.<locals>.DeltaOPDAgentLoop':
... could not be imported by importlib.
```

### Trigger
`DeltaOPDAgentLoop` was originally defined inside `register_delta_opd_agent_loop()` so the verl `@register` decorator was evaluated lazily (avoiding verl imports at module top-level). Python's `__qualname__` for that class is `register_delta_opd_agent_loop.<locals>.DeltaOPDAgentLoop`. Hydra's `_target_` resolution uses `importlib`, which only finds module-level attributes — function-local classes are unreachable.

### Verification
```bash
python3 -c "from experiments.E1_filtered_delta_opd.src.on_policy.agent_loop import register_delta_opd_agent_loop, DeltaOPDAgentLoop; \
            print(DeltaOPDAgentLoop.__qualname__)"
# Bad: register_delta_opd_agent_loop.<locals>.DeltaOPDAgentLoop
# Good: DeltaOPDAgentLoop
```

### Fix
Define the class at module top-level (and overwrite `__qualname__` after `@register` runs to keep introspection clean):
```python
# experiments/E1_filtered_delta_opd/src/on_policy/agent_loop.py
class DeltaOPDAgentLoop(SingleTurnAgentLoop):
    ...

DeltaOPDAgentLoop.__qualname__ = "DeltaOPDAgentLoop"
register("delta_opd_single_turn")(DeltaOPDAgentLoop)
```
Carried in `80e45a6d`.

---

## R3: cuDNN v9 sublibrary load failure in Qwen2.5-VL Conv3d

### Symptom
```
RuntimeError: CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed
ptrDesc->finalize() cudnn_status: CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
```
Surfaces during the first `compute_log_prob` after rollout. May cascade into a secondary `AssertionError` inside `verl/utils/fsdp_utils.py:161 offload_fsdp_model_to_cpu` — that's a downstream symptom of the FSDP context-manager's cleanup running while the cuDNN exception is propagating; **do not chase the assertion**.

### Trigger
Qwen2.5-VL's vision-tower `patch_embed` is a BF16 `Conv3d`. cuDNN v9 is a split library — backend ops dlopen sublibs (e.g. `libcudnn_engines_runtime_compiled.so`, `libcudnn_heuristic.so`) lazily at `cudnnBackendFinalize()` time, not at `cudnnCreate()`. The split-lib loader fails in this NGC PyTorch image when LD_LIBRARY_PATH mixes paths from the NGC-system Python's `torch/lib` and the venv-pip-installed cuDNN wheel. Basic cuDNN works (independent conv2d + SDPA test passes); only the Conv3d backend descriptor path trips the loader.

### Verification
A standalone test of cuDNN's basic ops will pass:
```python
import torch
torch.backends.cudnn.benchmark = True
# conv2d, SDPA — both succeed
x = torch.randn(2, 16, 32, 32, device="cuda", dtype=torch.bfloat16)
torch.nn.functional.conv2d(x, torch.randn(16, 16, 3, 3, device="cuda", dtype=torch.bfloat16))
torch.nn.functional.scaled_dot_product_attention(*[torch.randn(2,8,128,64, device="cuda", dtype=torch.bfloat16)]*3)
```
Only the verl + Qwen2.5-VL path fails. Confirms cuDNN itself is fine; the problem is one specific sublib that gets called via the v9 backend API.

### Fix
```bash
export TORCH_CUDNN_V8_API_DISABLED=1
```
Falls back to the v7 cuDNN API, which doesn't use the split-library loader. Qwen2.5-VL attention goes through `flash-attn-2` already (verl's `attn_implementation='flash_attention_2'`), so the perf hit from disabling v8 is negligible — only the Conv3d patch_embed and any cuDNN conv ops slow down marginally.

Baked into `experiments/E1_filtered_delta_opd/scripts/run_e1_recipe_smoke.sh` (commit `b58932b7`) as a launcher default.

### Long-term
The real fix is to clean up `LD_LIBRARY_PATH` so the NGC system Python's torch lib (`/usr/local/lib/python3.12/dist-packages/torch/lib`) doesn't leak into the venv. Either via `activate.sh` filtering, or by setting `LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"` in the launcher. Not done yet — keep the `TORCH_CUDNN_V8_API_DISABLED=1` workaround until verified on a clean env.

---

## R4: Multimodal prompts truncated out of the dataset

### Symptom
```
Filtering prompts longer than 1024 tokens (num_proc=1): 100%|██████████| 50/50
filter dataset len: 0
ValueError: batch_size should be a positive integer value, but got batch_size=0
```
Often preceded by `AssertionError: Cannot have both 'bytes' and 'image'` from `verl/utils/dataset/vision_utils.py:process_image`.

### Trigger
Two bugs stacking:
1. **Qwen2.5-VL image processor expands each image into ~1000-1500 `<|image_pad|>` tokens** (one per spatial patch at default resolution). The default `data.max_prompt_length=1024` cap drops every multimodal sample on the floor.
2. When `filter_overlong_prompts: true`, the filter calls `doc2len` which invokes `_build_messages` (mutates `image` dict to add an `"image"` key on top of `"bytes"`) and then `process_image` (which asserts the dict does NOT have both `"bytes"` and `"image"`). The mutation pollutes the dict for the second call — verl ordering bug.

The assertion is the actual error; `doc2len` then returns `max_prompt_length + 1` for every row, the filter drops everything, the dataloader fails with `batch_size=0`.

### Verification
```bash
# How many prompts the multimodal cap drops:
python3 -c "
import pyarrow.parquet as pq
t = pq.read_table('/tmp/e1_smoke.parquet')
# rough estimate; actual cap depends on processor
print('row count:', t.num_rows)
"
```
And read the traceback above the `batch_size=0` line — `AssertionError: Cannot have both 'bytes' and 'image'` is the give-away.

### Fix
Raise the cap and disable the overlong-prompt filter for smoke (the filter is what triggers the mutation bug):
```yaml
# e1_base.yaml
data:
  max_prompt_length: 4096       # was 1024
  max_response_length: 2048
  filter_overlong_prompts: false  # was true
actor_rollout_ref:
  rollout:
    max_model_len: 6145         # = 4096 + 2048 + 1
```
Baked into `69f38dbb`.

### Long-term
The filter's mutation bug is a real verl issue. A clean fix would be to patch `_build_messages` to not mutate the input dict in-place (clone it, add the PIL handle to the clone). Not pursued — until production needs `filter_overlong_prompts: true` to cull synthetic outliers, leaving it disabled is fine.

---

## R5: verl rewards registry doesn't know `e1_*` data_sources

### Symptom
```
NotImplementedError: Reward function is not implemented for data_source='e1_virl39k'
```
Fires at the first agent-loop `_compute_score` call (after rollout, before the first training step).

### Trigger
`make_train_parquet.py` sets `data_source = f"e1_{bucket}"` for trace/aggregation purposes. verl's `default_compute_score` in `verl/utils/reward_score/__init__.py` matches `data_source` against a hardcoded `if/elif` chain (NOT a registry decorator), and raises if no branch matches. Even though we have `distillation.distillation_loss.use_task_rewards=false` — meaning the reward never enters the loss — verl's AgentLoop still calls `_compute_score` unconditionally to populate `rm_scores` in the DataProto.

### Verification
```python
from verl.utils.reward_score import default_compute_score
default_compute_score(data_source="e1_virl39k", solution_str="...", ground_truth="...")
# → NotImplementedError
```

### Fix
Point verl at a dummy reward function via `custom_reward_function.path`. The dummy always returns `0.0`; distillation loss carries the actual gradient:
```yaml
# e1_base.yaml
custom_reward_function:
  path: experiments/E1_filtered_delta_opd/src/on_policy/dummy_reward.py
  name: compute_score
```
The `compute_score` function signature must match verl's contract (`data_source`, `solution_str`, `ground_truth`, `extra_info=None`, `**kwargs`). Carried in `ba655176`.

### Long-term
When E1 evaluation runs land (`src/eval_tei.py`), replace the dummy with a real `compute_score` that calls `verifier_pass` from `precompute_teacher.py` to flag teacher-correct samples — gives us live `Acc_S` in training logs even though the reward doesn't drive the loss.

---

## Cross-cutting: launcher defaults baked in after smoke

The smoke launcher (`experiments/E1_filtered_delta_opd/scripts/run_e1_recipe_smoke.sh`) now ships several non-obvious defaults from this debugging. Keep them when copying the launcher for production:

| Default | Why |
|---|---|
| `export PYTHONPATH="$PWD:${PYTHONPATH:-}"` | Ray workers / Hydra `_target_` resolution need to find `experiments.*` |
| `export TORCH_CUDNN_V8_API_DISABLED=1` | R3 — cuDNN v9 sublib load failure on this NGC image |
| `NGPUS_PER_NODE=4 + TEACHER_WORLD_SIZE=4` | actor/teacher pools are disjoint, 4+4=8 fits one H800 box |
| `AGENT_LOOP_WORKERS=$NGPUS_PER_NODE` | `train_batch_size % num_workers == 0` assert in `AgentLoopManager.generate_sequences` |
| `MAX_PROMPT_LENGTH=4096`, `MAX_RESPONSE_LENGTH=2048` | R4 — multimodal image_pad tokens explode prompt length |
| `data.filter_overlong_prompts: false` (via yaml) | R4 — sidesteps verl's `_build_messages` mutation bug |
