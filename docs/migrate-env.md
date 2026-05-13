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
| After restoring a venv backup, `python` is `/usr/bin/python` and half the packages "disappear" | [Q6](#q6-venv-binpython-is-a-dangling-symlink-to-uv-managed-python) |

Cross-machine setup: see also the [**Migration runbook**](#migration-runbook-copy-venv-to-a-new-dev-machine) below.

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

## Q6: venv `bin/python` is a dangling symlink to uv-managed Python

### Symptom
After restoring a venv backup to a new machine (or after a fast-disk wipe), the venv *looks* intact but Python isn't actually running from it:

- `source activate.sh` reports `python: /usr/bin/python` (system Python), not `/root/shihao_project/env/opd-exp/bin/python`.
- `python -c "import sys; print(sys.executable)"` confirms the same.
- The 11/11 sanity check fails on most packages (`vllm`, `transformers`, `ray`, `hf_hub`, `flashinfer`, …) — they look "missing" but they're actually fine in the venv site-packages; system Python just can't see that site-packages.
- Even after running the Q5 sed patch on the venv triton, `import transformer_engine.pytorch` *still* fails with the same Q5 `UnicodeDecodeError`, because system Python loads system triton (unpatched).

The smoking gun:

```bash
$ ls -la /root/shihao_project/env/opd-exp/bin/python
lrwxrwxrwx ... /root/shihao_project/env/opd-exp/bin/python ->
    /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12
$ /root/shihao_project/env/opd-exp/bin/python --version
bash: /root/shihao_project/env/opd-exp/bin/python: No such file or directory
```

The symlink target `/root/.local/share/uv/python/…` doesn't exist on the new machine.

### Trigger
The venv was created by `uv venv`, and its `bin/python` is a **symlink into uv's managed Python store** (`~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/`). On NGC dev machines this path is on the **fast (volatile) disk** that gets wiped on reboot / reimage. A `tar` of only `env/opd-exp/` and `env-snapshots/` packs the venv shell but **misses the actual Python interpreter** the symlink resolves to. After restore on a fresh machine, the symlink dangles; the venv's `bin/python` doesn't exec; the shell's PATH falls through to `/usr/bin/python`, which reads from `/usr/local/lib/python3.12/dist-packages/` — a completely different site-packages with a different set of installed packages (NGC system packages, not the project's venv).

### Verification
```bash
ls -la /root/shihao_project/env/opd-exp/bin/python              # see the symlink
[ -e "$(readlink -f /root/shihao_project/env/opd-exp/bin/python)" ] \
    && echo "target exists" || echo "TARGET MISSING — this is the bug"
ls /root/.local/share/uv/python/ 2>&1 | head -3
```

After `source activate.sh`, the diagnostic check:
```bash
python -c "import sys; print(sys.executable)"
# Healthy:  /root/shihao_project/env/opd-exp/bin/python
# Bug:      /usr/bin/python
```

### Fix

Two options depending on whether `uv` itself is available on this machine:

**(A) `uv` is installed → reinstall the matching Python in place:**
```bash
which uv                           # if this prints a path, you're set
uv python install 3.12             # downloads cpython-3.12 to ~/.local/share/uv/python/
/root/shihao_project/env/opd-exp/bin/python --version   # symlink now resolves
```

This is ~30 seconds and doesn't touch the venv's site-packages.

**(B) `uv` is missing → repack from the source machine with uv-python included.**

The source-machine `tar` command must include `.local/share/uv/python/` (see the [Migration runbook](#migration-runbook-copy-venv-to-a-new-dev-machine) below — its `Step 1` is the corrected version).

### Cost of getting this wrong
Identifying this without the symptom checklist above can burn hours chasing
"missing modules" that aren't actually missing — they're just invisible
because Python's running from the wrong location. Once the symptom is
recognized, fix is ~30 sec (option A) or ~10-15 min (option B re-pack).

---

## Migration runbook: copy venv to a new dev machine

**When to use**: you have a working `opd-exp` env on machine `SRC` (passed `11/11 sanity` per Step 4 below) and want to bring up the same env on machine `DST`.

**When NOT to use** (rebuild from scratch instead, following `CLAUDE.md` § Environment setup):
- `DST` has a different GPU arch (H800 → A100 / H100 = different `sm` = different sgl-kernel / flashinfer / xformers binaries)
- `DST` has a different CUDA major version
- `DST` is not based on the same NGC PyTorch image
- `DST` architecture differs (rare; everything here is `x86_64-linux-gnu`)

### What to pack — the critical bit

Pack **three** directories from `SRC`'s root filesystem:

1. `/root/shihao_project/env/opd-exp/` — the venv shell (symlinks + site-packages)
2. `/root/shihao_project/env-snapshots/` — `pip freeze` records, useful for diff diagnostics later
3. `/root/.local/share/uv/python/` — **the actual Python interpreter the venv `bin/python` symlinks to** (see [Q6](#q6-venv-binpython-is-a-dangling-symlink-to-uv-managed-python) above for why missing this breaks everything silently)

A backup that omits the third path will look like it worked but Python will fall back to `/usr/bin/python` on `DST` and the entire venv becomes invisible.

### Step 1 (SRC): pack

```bash
tmux new -s pack    # don't lose this if SSH disconnects
which pigz || apt-get install -y pigz   # multi-core gzip, 3-5× faster

mkdir -p /home/web_server/antispam/project/houshihao/transfer/

cd /root
tar -I pigz -cf /home/web_server/antispam/project/houshihao/transfer/opd-exp-venv-$(date +%Y%m%d).tar.gz \
    shihao_project/env/opd-exp/ \
    shihao_project/env-snapshots/ \
    .local/share/uv/python/

ls -lh /home/web_server/antispam/project/houshihao/transfer/opd-exp-venv-*.tar.gz
# Expect 8-10 GB (13 GB raw venv compresses well; uv-python adds ~200 MB)
```

Typical wall-clock: pigz ~3-5 min (NFS write is the bottleneck). Plain `gzip` ~10-15 min.

### Step 2 (DST): clean + unpack

```bash
# Clean any partial / broken state from a previous attempt
rm -rf /root/shihao_project/env /root/shihao_project/env-snapshots /root/.local/share/uv/python

mkdir -p /root/shihao_project/{.uv,build}
cd /root
tar -I pigz -xf /home/web_server/antispam/project/houshihao/transfer/opd-exp-venv-YYYYMMDD.tar.gz

# Critical verification — the venv's Python binary must actually exec
ls -la /root/shihao_project/env/opd-exp/bin/python
/root/shihao_project/env/opd-exp/bin/python --version    # MUST print "Python 3.12.x"
```

If `--version` errors with `No such file or directory`, `.local/share/uv/python/` was missed on the `SRC` pack — see [Q6](#q6-venv-binpython-is-a-dangling-symlink-to-uv-managed-python) for the in-place rescue using `uv python install 3.12`.

### Step 3 (DST): activate + Q5 sed

```bash
cd /home/web_server/antispam/project/houshihao/opd-exp
cp -n activate.sh.template activate.sh     # only if missing
source activate.sh
```

Expected output should include:
```
[activate.sh] opd-exp env active
  venv:    /root/shihao_project/env/opd-exp
  python:  /root/shihao_project/env/opd-exp/bin/python       ← NOT /usr/bin/python
  ...
```

If `python:` shows `/usr/bin/python`, **stop** and go back to Step 2's verification — see [Q6](#q6-venv-binpython-is-a-dangling-symlink-to-uv-managed-python).

If `activate.sh` prints the Q5 warning, apply the sed (one-time per venv; will need re-applying if triton ever gets reinstalled):
```bash
sed -i 's/decode()/decode("utf-8", errors="ignore")/' \
    /root/shihao_project/env/opd-exp/lib/python3.12/site-packages/triton/backends/nvidia/driver.py
```

### Step 4 (DST): 11/11 sanity

```bash
python -c "
checks = [
    ('numpy<2',     'import numpy; assert numpy.__version__.startswith(\"1.\")'),
    ('torch',       'import torch; assert torch.cuda.is_available()'),
    ('torch.gpus',  'import torch; assert torch.cuda.device_count()==8'),
    ('vllm',        'import vllm'),
    ('transformers','import transformers; from transformers import PreTrainedTokenizer'),
    ('TE',          'import transformer_engine.pytorch'),
    ('megatron',    'import megatron.core'),
    ('flash_attn',  'import flash_attn'),
    ('flashinfer',  'import flashinfer'),
    ('verl',        'import verl; assert \"opd-exp/verl\" in verl.__file__'),
    ('hf_hub',      'import huggingface_hub; assert huggingface_hub.__version__.startswith(\"0.\")'),
]
ok = 0
for name, code in checks:
    try: exec(code); print(f'  OK  {name}'); ok += 1
    except Exception as e: print(f'  FAIL {name}: {type(e).__name__}: {e}')
print(f'{ok}/{len(checks)} passed')
"
```

**11/11 = ready.** Anything less, debug per Q1-Q6 above before running real workloads.

Common failure → section:

| Failure | Section |
|---|---|
| `python:` is `/usr/bin/python` after `source activate.sh` | [Q6](#q6-venv-binpython-is-a-dangling-symlink-to-uv-managed-python) |
| `import torch` succeeds but version has `nv` suffix | [Q1](#q1-pip_constraint-locks-torchtriton-to-ngc-versions) + [Q2](#q2-pep-517-build-isolation-leaks-the-system-ngc-torch) |
| TE import fails with `UnicodeDecodeError` | [Q5](#q5-triton-ldconfig-unicodedecodeerror-on-machines-with-hpc-x) |
| `import transformers` fails after some recent `pip install` | [Q4](#q4-huggingface-hub-auto-upgrades-to-1x-and-breaks-transformers) |
| TE / flash-attn import fails with `undefined symbol` | [Q2](#q2-pep-517-build-isolation-leaks-the-system-ngc-torch) |

### Step 5 (DST → new SRC): re-backup after first end-to-end success

The original `SRC` backup is a snapshot in time. After your first end-to-end success on `DST` (e.g. one full E1 Phase 3 config train + merge + eval), pack `DST` as the new canonical backup. **Always timestamp the filename** so you don't overwrite a known-good backup with a partial / broken one:

```bash
DATE=$(date +%Y%m%d)
cd /root
tar -I pigz -cf /home/web_server/antispam/project/houshihao/transfer/opd-exp-venv-${DATE}-validated.tar.gz \
    shihao_project/env/opd-exp/ \
    shihao_project/env-snapshots/ \
    .local/share/uv/python/
ls -lh /home/web_server/antispam/project/houshihao/transfer/
```

Only delete old backups after the new one has been verified by a full round-trip migration **or** after a few weeks of stability. Disk on NFS is cheap compared to a half-day env rebuild.

### Step 6: don't forget the per-machine `activate.sh`

`activate.sh` is **per-machine** (paths, GPU arch list, anything site-specific). It's gitignored. `activate.sh.template` is committed and tracked. After a migration:
- If `DST` has no `activate.sh` yet: `cp activate.sh.template activate.sh` then edit env vars
- If `DST` already had one from a prior install: it survives the venv re-extract (it's in the NFS-mounted repo, not the volatile root fs)

---

## Detection in `activate.sh`

`activate.sh.template` runs a non-fatal startup check for Q5 (and a few other
common environment issues). The check is **detect-only** — it warns and
points at this doc, but never modifies the venv. Auto-patching from
`activate.sh` would violate the project's standing policy ("no automation in
activate.sh" — see top of the template) because previous automation attempts
silently broke hand-built TransformerEngine binaries.

If you see the warning, run the fix command yourself.
