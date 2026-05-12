# opd-exp

Experiment workspace for **Delta-OPD** — on-policy distillation for VLMs guided by per-token *image-vs-null* teacher delta.

The verl training framework is vendored as a git submodule at `verl/` (forked from [`verl-project/verl`](https://github.com/verl-project/verl) at [`shihaohou/verl`](https://github.com/shihaohou/verl)). Experiments live under `experiments/`.

## Layout

```
opd-exp/
├── verl/                              # submodule -> shihaohou/verl
└── experiments/
    └── E0_image_null_delta/           # forward-only diagnostic (no training)
        ├── configs/
        ├── data/
        ├── src/
        ├── scripts/
        ├── analysis/
        └── results/                   # gitignored
```

## Clone

```bash
git clone --recurse-submodules https://github.com/shihaohou/opd-exp.git
# or, if already cloned without submodules:
git submodule update --init --recursive
```

## verl submodule workflow

Modify verl source:

```bash
cd verl
git checkout -b my-feature
# edit ...
git commit -m "..."
git push origin my-feature              # pushes to shihaohou/verl
cd ..
git add verl
git commit -m "Bump verl submodule"     # parent records new submodule hash
```

To pull upstream verl changes:

```bash
cd verl
git remote add upstream https://github.com/verl-project/verl.git   # first time only
git fetch upstream
git merge upstream/main
cd ..
git add verl && git commit -m "Bump verl: merge upstream"
```

## Experiments

See [`experiments/E0_image_null_delta/README.md`](experiments/E0_image_null_delta/README.md) for the first experiment (no training, forward-only teacher diagnostic).
