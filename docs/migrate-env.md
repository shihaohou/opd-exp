# Environment migration troubleshooting

Runbook for setting up `opd-exp` on a new machine. Companion to the canonical
`CLAUDE.md → Environment setup (NGC machine specifics)` section, which lists
the **mitigations** baked into `activate.sh.template`. This doc captures
**diagnoses and one-off fixes** for issues that surface during setup or after
an environment drift.

## Quick reference: which problem are you looking at?

| Symptom | Q |
|---|---|
| `pip install` silently pulls NGC-pinned torch/triton/etc. | [Q1](#q1-pip_constraint-locks-torchtriton-to-ngc-versions) |
| TransformerEngine fails to import with `undefined symbol` | [Q2](#q2-pep-517-build-isolation-leaks-the-system-ngc-torch) |
| `uv pip install -e ./verl` silently overwrites your hand-built TE binary | [Q3](#q3-the---no-deps-rule-for-editable-installs) |
| `transformers` import fails after `pip install`; `huggingface-hub` is now 1.x | [Q4](#q4-huggingface-hub-auto-upgrades-to-1x-and-breaks-transformers) |
| vLLM crashes during `profile_run` with `UnicodeDecodeError 0xc4` from triton | [Q5](#q5-triton-ldconfig-unicodedecodeerror-on-machines-with-hpc-x) |

---

## Q1: `PIP_CONSTRAINT` locks torch/triton to NGC versions

### Symptom
`pip install` (or `uv pip install`) appears to succeed, but `python -c "import torch; print(torch.__version__)"` returns an NGC-tagged version like `2.8.0a0+nv25.6` instead of the version you asked for. Or your `vllm` install fails with a torch version conflict.

### Trigger
NVIDIA NGC PyTorch images set `PIP_CONSTRAINT=/etc/pip/constraint.txt` and `PIP_CONFIG_FILE=/etc/pip.conf` to pin torch/triton/etc. to NGC-built versions.

### Verification
```bash
echo "$PIP_CONSTRAINT"   # if non-empty, you're on an NGC image
cat /etc/pip/constraint.txt | head -10
```

### Fix
`activate.sh.template` already unsets both. If you bypass `activate.sh` (e.g. shell out via `bash -l`), re-export manually:
```bash
unset PIP_CONSTRAINT
export PIP_CONFIG_FILE=/dev/null
```

---

## Q2: PEP 517 build isolation leaks the system NGC torch

### Symptom
After building TransformerEngine or flash-attn from source, the resulting binary fails to import with `undefined symbol` errors on torch C++ ABI.

### Trigger
`/usr/local/lib/python3.12/dist-packages/torch/` on NGC images is a customized NGC torch (`2.8.0a0+nv25.6`). PEP 517 build isolation creates a temporary venv with `pip install build-deps`, and that temporary venv ends up linking against the *system* NGC torch instead of the project venv's torch. The compiled extension has the wrong ABI.

### Verification
```bash
ls /usr/local/lib/python3.12/dist-packages/torch/   # NGC torch exists here
python -c "import transformer_engine"               # fails with undefined symbol
```

### Fix
Always install TE / flash-attn with `--no-build-isolation`:
```bash
pip install --no-deps --no-build-isolation -v \
    git+https://github.com/NVIDIA/TransformerEngine.git@v2.6
```

Both `--no-deps` (see Q3) and `--no-build-isolation` are required.

### Cost of getting this wrong
Rebuilding TransformerEngine from scratch takes **30-40 minutes**.

---

## Q3: The `--no-deps` rule for editable installs

### Symptom
After `uv pip install -e ./verl` your TransformerEngine import suddenly fails. Or `flash-attn` is mysteriously the wrong version.

### Trigger
`verl` declares `vllm`, `torch`, `TransformerEngine`, and `flash-attn` as dependencies in `pyproject.toml`. Without `--no-deps`, `uv` resolves and reinstalls those, silently overwriting the hand-built binaries.

### Verification
```bash
pip show transformer_engine | grep Location  # changed after a verl install?
```

### Fix
**Every** `pip install -e` and `uv pip install -e` on this machine must pass `--no-deps`:
```bash
uv pip install --no-deps -e ./verl
```

No exceptions unless explicitly asked to update deps.

### Cost of getting this wrong
30-40 min to rebuild TransformerEngine (see Q2).

---

## Q4: `huggingface-hub` auto-upgrades to 1.x and breaks transformers

### Symptom
`from transformers import AutoModel` fails with an attribute error about `huggingface_hub.HfFileSystem` or similar.

### Trigger
Some transitive install (often via `vllm` or `datasets`) pulls in `huggingface-hub>=1.0`, which dropped API surface that `transformers 4.56.1` still depends on.

### Verification
```bash
pip show huggingface-hub | grep Version   # 1.x → broken
```

### Fix
Pin to `<1.0` after the offending install:
```bash
pip install --no-deps "huggingface-hub>=0.34.0,<1.0"
```

---

## Q5: Triton `ldconfig` UnicodeDecodeError on machines with HPC-X

### Symptom
vLLM crashes during `profile_run` (before serving any request). Model weights have already loaded. The crash is in the multimodal vision encoder's rotary kernel:

```
File ".../triton/backends/nvidia/driver.py", line 25, in libcuda_dirs
    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode()
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc4 in position 67587: invalid continuation byte
```

The error fires on the **first triton kernel JIT** — for VLM models that's typically `apply_rotary_pos_emb_vision` in `qwen2_5_vl.py`. Pure-text models on the same machine may not hit this until later, but the underlying bug is the same.

### Trigger
NVIDIA HPC-X (the InfiniBand / NCCL plugin stack — installed on some training clusters) registers libraries like `ncclnet_plugin` into `/etc/ld.so.cache` with non-UTF-8 bytes (e.g. `\x04\xc4`) somewhere in the binary cache. When triton's CUDA backend calls `subprocess.check_output(["/sbin/ldconfig", "-p"]).decode()`, the bytes-to-str conversion uses Python's default UTF-8 decoder, which fails on the invalid byte.

Machines without HPC-X (typical workstations, including the project's other dev boxes) are unaffected — `ldconfig -p` outputs pure ASCII.

### Verification
```bash
/sbin/ldconfig -p | head -c 70000 | tail -c 500 | od -c | grep -E '\\[0-9]{3}' | head
# Look for octal escapes like \004 or \304 — those are the non-ASCII bytes.
```

Or directly reproduce the failure:
```bash
python -c "import subprocess; subprocess.check_output(['/sbin/ldconfig','-p']).decode()"
# UnicodeDecodeError on HPC-X machines; clean on others.
```

### Fix
Patch the triton driver to tolerate non-UTF-8 bytes (one-time edit to the venv, no repo changes):

```bash
sed -i 's/decode()/decode("utf-8", errors="ignore")/' \
    "${VIRTUAL_ENV:?activate venv first}/lib/python3.12/site-packages/triton/backends/nvidia/driver.py"
```

Re-run the offending vLLM / triton call. Profile_run completes; multimodal models load and serve normally.

### Notes
- **`LC_ALL=C` does NOT work.** Python's `bytes.decode()` defaults to UTF-8 regardless of locale; `LC_ALL=C` would only affect how the *child* `ldconfig` process emits output, but `/etc/ld.so.cache` is a binary file whose contents `ldconfig -p` prints raw — locale doesn't enter the picture.
- **The fix is venv-local.** If you reinstall triton (e.g. via `pip install -U triton` or as a transitive of a torch reinstall), the patch is overwritten — re-apply. `activate.sh.template` has a startup check that warns when this happens (see § Detection in activate.sh below).
- **Upstream**: this is a known triton bug. As of triton 3.x there's no fix shipped. Track via the triton repo if relevant.

---

## Detection in `activate.sh`

`activate.sh.template` runs a non-fatal startup check for Q5 (and a few other
common environment issues). The check is **detect-only** — it warns and
points at this doc, but never modifies the venv. Auto-patching from
`activate.sh` would violate the project's standing policy ("no automation in
activate.sh" — see top of the template) because previous automation attempts
silently broke hand-built TransformerEngine binaries.

If you see the warning, run the fix command yourself.
