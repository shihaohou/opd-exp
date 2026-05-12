"""
Sample the 8K E1-mini training mixture from the 3 buckets.

Locked recipe (from `experiments/E1_filtered_delta_opd/README.md`):
    Bucket 1: 4000 ViRL39K subset
        - PassRate_32BTrained ∈ [0.3, 0.9]
        - single-image, \\boxed{}-parseable
        - stratified by `category` (8 ViRL39K categories)
    Bucket 2: 1600 self-built POPE-style on COCO train
        - yes:no = 1:1
        - negatives mixed random / popular / cooccur
        - image-level disjoint with POPE-adv eval
    Bucket 3: 2400
        - 1500 synthetic VLMBias-like counterfactuals
        -  900 TallyQA `complex` (COCO-id-filtered vs POPE-adv)

Output: a unified mixture manifest at ``--output``. One jsonl row per
sample with the minimum fields ``precompute_teacher.py`` needs to run the
teacher dual-forward + verifier:

    {
        "sample_id"  : str,        # unique across mixture
        "bucket"     : str,        # virl39k / pope_style / tallyqa / synthetic
        "question"   : str,
        "gold"       : str,
        "image_path" : str,        # absolute path on disk
        "extras"     : dict,       # passthrough metadata (category, neg_type, …)
    }

This manifest is the **input** to precompute_teacher.py's ``mixture``
bucket (added in Task #6 of Day 2). Precompute then writes another jsonl
with teacher fields added; that jsonl feeds make_train_parquet.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

# Repo-root injection for cross-experiment imports
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-bucket targets locked by `experiments/E1_filtered_delta_opd/README.md`.
# ---------------------------------------------------------------------------

DEFAULT_TARGETS: dict[str, int] = {
    "virl39k": 4000,
    "pope_style": 1600,
    "tallyqa": 900,
    "synthetic": 1500,
}


# ---------------------------------------------------------------------------
# Stratified sampling helper.
# ---------------------------------------------------------------------------

def stratified_sample(
    items_by_strata: dict[str, list[Any]],
    n_target: int,
    rng: random.Random,
) -> list[Any]:
    """Proportional stratified sample, capped at per-stratum availability.

    First pass picks ``round(n_target * stratum_size / total)`` per stratum,
    capped at ``len(stratum)``. Slack from capped strata is redistributed
    among the others.

    Returns a list of items from across all strata. Item order within a
    stratum is shuffled; the inter-stratum order matches ``items_by_strata``.
    """
    total = sum(len(v) for v in items_by_strata.values())
    if total == 0 or n_target <= 0:
        return []

    targets: dict[str, int] = {}
    for stratum, items in items_by_strata.items():
        prop = int(round(n_target * len(items) / total))
        targets[stratum] = min(prop, len(items))

    # Redistribute slack from capped strata, in rounds, until we either
    # reach n_target or run out of capacity.
    while True:
        current = sum(targets.values())
        slack = n_target - current
        if slack <= 0:
            break
        not_capped = [s for s, items in items_by_strata.items() if targets[s] < len(items)]
        if not not_capped:
            break
        # Distribute slack uniformly across not-capped strata.
        per = slack // len(not_capped)
        rem = slack - per * len(not_capped)
        progressed = False
        for s in not_capped:
            cap = len(items_by_strata[s])
            new_target = min(targets[s] + per, cap)
            if new_target > targets[s]:
                progressed = True
                targets[s] = new_target
        # 1-per-stratum remainder pass
        for s in not_capped:
            if rem <= 0:
                break
            cap = len(items_by_strata[s])
            if targets[s] < cap:
                targets[s] += 1
                rem -= 1
                progressed = True
        if not progressed:
            break

    if sum(targets.values()) > n_target:
        # Round-down trim to honor n_target exactly.
        excess = sum(targets.values()) - n_target
        for s in sorted(targets, key=lambda k: -targets[k]):
            if excess <= 0:
                break
            take = min(targets[s], excess)
            targets[s] -= take
            excess -= take

    out: list[Any] = []
    for stratum, n in targets.items():
        items = list(items_by_strata[stratum])
        rng.shuffle(items)
        out.extend(items[:n])
    logger.info(
        "[stratify] n_target=%d total_available=%d → emitted=%d (per-stratum: %s)",
        n_target, total, len(out), targets,
    )
    return out


# ---------------------------------------------------------------------------
# Manifest schema + sample → dict adapter.
# ---------------------------------------------------------------------------

def sample_to_manifest_entry(sample, bucket: str) -> dict:
    """Convert a bucket-specific Sample dataclass into a uniform manifest dict.

    All four bucket dataclasses (``ViRL39KSample`` / ``POPEStyleSample`` /
    ``TallyQASample`` / ``SyntheticSample``) share the same field names,
    so this adapter is single-shape.
    """
    if not sample.image_paths:
        raise ValueError(f"sample {sample.sample_id} has no image_paths")
    if len(sample.image_paths) > 1:
        raise ValueError(
            f"sample {sample.sample_id} has {len(sample.image_paths)} images; "
            "v1 is single-image only"
        )
    return {
        "sample_id": sample.sample_id,
        "bucket": bucket,
        "question": sample.question,
        "gold": sample.gold,
        "image_path": str(sample.image_paths[0]),
        "extras": dict(sample.extras),
    }


# ---------------------------------------------------------------------------
# Bucket collectors.
# ---------------------------------------------------------------------------

def collect_virl39k(
    dataset_root: Path | str,
    n_target: int,
    seed: int,
    *,
    pass_rate_min: float = 0.3,
    pass_rate_max: float = 0.9,
) -> list[dict]:
    """Bucket 1: stratified-by-category sampling from ViRL39K."""
    from experiments.E1_filtered_delta_opd.data.virl39k_loader import iter_virl39k

    samples = list(iter_virl39k(
        dataset_root=dataset_root,
        pass_rate_min=pass_rate_min,
        pass_rate_max=pass_rate_max,
        single_image_only=True,
        require_boxed=True,
    ))
    logger.info("[bucket=virl39k] eligible after pre-filter: %d", len(samples))

    by_cat: dict[str, list] = defaultdict(list)
    for s in samples:
        by_cat[s.extras.get("category") or "UNKNOWN"].append(s)

    rng = random.Random(seed)
    sampled = stratified_sample(by_cat, n_target, rng)
    return [sample_to_manifest_entry(s, "virl39k") for s in sampled]


def collect_pope_style(
    coco_train_root: Path | str,
    annotations_path: Path | str,
    pope_adv_root: Path | str,
    n_target: int,
    seed: int,
) -> list[dict]:
    """Bucket 2: POPE-style yes/no from COCO train2017."""
    from experiments.E1_filtered_delta_opd.data.pope_style_builder import (
        iter_pope_style, load_pope_adv_image_ids,
    )

    pope_adv_ids = load_pope_adv_image_ids(pope_adv_root)
    logger.info("[bucket=pope_style] POPE-adv eval ids loaded: %d", len(pope_adv_ids))
    samples = list(iter_pope_style(
        coco_train_root=coco_train_root,
        annotations_path=annotations_path,
        pope_adv_eval_ids=pope_adv_ids,
        n_samples=n_target,
        seed=seed,
    ))
    return [sample_to_manifest_entry(s, "pope_style") for s in samples]


def collect_tallyqa(
    json_path: Path | str,
    images_root: Path | str,
    pope_adv_root: Path | str,
    n_target: int,
    seed: int,
) -> list[dict]:
    """Bucket 3a: TallyQA complex subset with COCO-id leakage filter."""
    from experiments.E1_filtered_delta_opd.data.tallyqa_loader import iter_tallyqa
    from experiments.E1_filtered_delta_opd.data.pope_style_builder import load_pope_adv_image_ids

    pope_adv_ids = load_pope_adv_image_ids(pope_adv_root)
    logger.info("[bucket=tallyqa] POPE-adv eval ids loaded: %d", len(pope_adv_ids))
    samples = list(iter_tallyqa(
        json_path=json_path,
        images_root=images_root,
        coco_eval_image_ids=pope_adv_ids,
        complex_only=True,
        n_max=n_target,
        seed=seed,
    ))
    return [sample_to_manifest_entry(s, "tallyqa") for s in samples]


def collect_synthetic(
    synth_dir: Path | str,
    n_target: int,
    seed: int,
    *,
    build_if_missing: bool = False,
) -> list[dict]:
    """Bucket 3b: synthetic counterfactuals manifest.

    Expects ``synth_dir/manifest.jsonl`` to exist (built by
    ``synthesize_counterfactuals.build_synthetic_counterfactuals``). If
    ``build_if_missing`` is True, build it with ``n_target`` samples on
    the fly; otherwise fail loudly when missing.
    """
    from experiments.E1_filtered_delta_opd.data.synthesize_counterfactuals import (
        iter_synthetic_counterfactuals, build_synthetic_counterfactuals,
    )

    manifest_path = Path(synth_dir) / "manifest.jsonl"
    if not manifest_path.exists():
        if not build_if_missing:
            raise FileNotFoundError(
                f"No synth manifest at {manifest_path}. Either run "
                f"`python -m experiments.E1_filtered_delta_opd.data.synthesize_counterfactuals "
                f"--out-dir {synth_dir} --n-samples {n_target}` first, or pass --build-synth."
            )
        build_synthetic_counterfactuals(
            out_dir=synth_dir, n_samples=n_target, seed=seed,
        )

    samples = list(iter_synthetic_counterfactuals(synth_dir))
    # If the on-disk manifest has more than n_target (e.g., previously built
    # at a larger size), sub-sample deterministically.
    if len(samples) > n_target:
        rng = random.Random(seed)
        rng.shuffle(samples)
        samples = samples[:n_target]
    elif len(samples) < n_target:
        logger.warning(
            "[bucket=synthetic] manifest has %d samples but target=%d — taking all",
            len(samples), n_target,
        )
    return [sample_to_manifest_entry(s, "synthetic") for s in samples]


# ---------------------------------------------------------------------------
# Mixture driver.
# ---------------------------------------------------------------------------

def build_mixture(
    *,
    output_manifest: Path | str,
    virl39k_root: Optional[Path | str] = None,
    coco_train_root: Optional[Path | str] = None,
    coco_annotations_path: Optional[Path | str] = None,
    pope_adv_root: Optional[Path | str] = None,
    tallyqa_json_path: Optional[Path | str] = None,
    tallyqa_images_root: Optional[Path | str] = None,
    synth_dir: Optional[Path | str] = None,
    targets: Optional[dict[str, int]] = None,
    seed: int = 42,
    build_synth_if_missing: bool = False,
) -> Path:
    """Materialize the 8K E1-mini manifest at ``output_manifest``.

    Per-bucket arguments are optional — if a bucket's required paths are
    not provided, that bucket is **skipped with a warning**, and the
    remaining buckets are emitted unchanged. This lets you build a
    partial mixture during day-2 (e.g., synth-only, or POPE+synth).
    """
    targets = dict(targets or DEFAULT_TARGETS)
    output_manifest = Path(output_manifest)

    all_entries: list[dict] = []

    # --- Bucket 1: ViRL39K ---
    if targets.get("virl39k", 0) > 0:
        if not virl39k_root:
            logger.warning("[bucket=virl39k] virl39k_root not provided; skipping")
        else:
            entries = collect_virl39k(
                dataset_root=virl39k_root,
                n_target=targets["virl39k"],
                seed=seed,
            )
            all_entries.extend(entries)
            logger.info("[bucket=virl39k] emitted %d entries", len(entries))

    # --- Bucket 2: POPE-style on COCO train ---
    if targets.get("pope_style", 0) > 0:
        if not (coco_train_root and coco_annotations_path and pope_adv_root):
            logger.warning(
                "[bucket=pope_style] missing one of "
                "coco_train_root / coco_annotations_path / pope_adv_root; skipping"
            )
        else:
            entries = collect_pope_style(
                coco_train_root=coco_train_root,
                annotations_path=coco_annotations_path,
                pope_adv_root=pope_adv_root,
                n_target=targets["pope_style"],
                seed=seed + 1,    # distinct seed for distinct bucket
            )
            all_entries.extend(entries)
            logger.info("[bucket=pope_style] emitted %d entries", len(entries))

    # --- Bucket 3a: TallyQA complex ---
    if targets.get("tallyqa", 0) > 0:
        if not (tallyqa_json_path and tallyqa_images_root and pope_adv_root):
            logger.warning(
                "[bucket=tallyqa] missing one of "
                "tallyqa_json_path / tallyqa_images_root / pope_adv_root; skipping"
            )
        else:
            entries = collect_tallyqa(
                json_path=tallyqa_json_path,
                images_root=tallyqa_images_root,
                pope_adv_root=pope_adv_root,
                n_target=targets["tallyqa"],
                seed=seed + 2,
            )
            all_entries.extend(entries)
            logger.info("[bucket=tallyqa] emitted %d entries", len(entries))

    # --- Bucket 3b: Synthetic counterfactuals ---
    if targets.get("synthetic", 0) > 0:
        if not synth_dir:
            logger.warning("[bucket=synthetic] synth_dir not provided; skipping")
        else:
            entries = collect_synthetic(
                synth_dir=synth_dir,
                n_target=targets["synthetic"],
                seed=seed + 3,
                build_if_missing=build_synth_if_missing,
            )
            all_entries.extend(entries)
            logger.info("[bucket=synthetic] emitted %d entries", len(entries))

    if not all_entries:
        raise RuntimeError("No bucket emitted any entries — check arguments")

    # --- Write the manifest ---
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # --- Summary ---
    bucket_counts = Counter(e["bucket"] for e in all_entries)
    logger.info("[mixture] wrote %d entries → %s", len(all_entries), output_manifest)
    logger.info("[mixture] per-bucket counts: %s", dict(bucket_counts))
    return output_manifest


def iter_mixture_manifest(manifest_path: Path | str):
    """Yield mixture-manifest entries; lightweight (no PIL load)."""
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the 8K E1-mini training mixture manifest")
    p.add_argument(
        "--output", required=True,
        help="Output mixture-manifest jsonl path",
    )
    p.add_argument("--seed", type=int, default=42)
    # Bucket 1 (ViRL39K)
    p.add_argument(
        "--virl39k-root",
        default="/home/web_server/antispam/project/houshihao/datasets/ViRL39K",
        help="ViRL39K dataset root (contains 39Krelease.parquet and images/)",
    )
    # Bucket 2 (POPE-style)
    p.add_argument(
        "--coco-train-root",
        default="/home/web_server/antispam/project/houshihao/datasets/coco",
        help="COCO root containing train2017/ and annotations/",
    )
    p.add_argument(
        "--coco-annotations", default=None,
        help="Path to instances_train2017.json (default: <coco-train-root>/annotations/instances_train2017.json)",
    )
    p.add_argument(
        "--pope-adv-root",
        default="/home/web_server/antispam/project/houshihao/datasets/POPE-adversarial",
    )
    # Bucket 3a (TallyQA)
    p.add_argument(
        "--tallyqa-json", default=None,
        help="Path to TallyQA train.json (or train.jsonl)",
    )
    p.add_argument(
        "--tallyqa-images-root", default=None,
        help="Root containing train2014/ val2014/ VG_100K/ ... (TallyQA image base)",
    )
    # Bucket 3b (synthetic)
    p.add_argument(
        "--synth-dir", default=None,
        help="Directory holding synthesize_counterfactuals output (with manifest.jsonl)",
    )
    p.add_argument(
        "--build-synth", action="store_true",
        help="Build the synth manifest on-the-fly if it doesn't exist",
    )
    # Per-bucket target overrides
    p.add_argument("--n-virl39k", type=int, default=DEFAULT_TARGETS["virl39k"])
    p.add_argument("--n-pope-style", type=int, default=DEFAULT_TARGETS["pope_style"])
    p.add_argument("--n-tallyqa", type=int, default=DEFAULT_TARGETS["tallyqa"])
    p.add_argument("--n-synthetic", type=int, default=DEFAULT_TARGETS["synthetic"])
    p.add_argument(
        "--skip", nargs="*", default=[],
        help="Bucket names to skip (virl39k / pope_style / tallyqa / synthetic). "
             "Useful when not all data is on disk yet.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    targets = {
        "virl39k": args.n_virl39k,
        "pope_style": args.n_pope_style,
        "tallyqa": args.n_tallyqa,
        "synthetic": args.n_synthetic,
    }
    for s in args.skip:
        if s not in targets:
            raise ValueError(f"Unknown --skip bucket {s!r}; choose from {sorted(targets)}")
        targets[s] = 0

    coco_ann = Path(args.coco_annotations) if args.coco_annotations else (
        Path(args.coco_train_root) / "annotations" / "instances_train2017.json"
    )

    build_mixture(
        output_manifest=args.output,
        virl39k_root=args.virl39k_root,
        coco_train_root=args.coco_train_root,
        coco_annotations_path=coco_ann,
        pope_adv_root=args.pope_adv_root,
        tallyqa_json_path=args.tallyqa_json,
        tallyqa_images_root=args.tallyqa_images_root,
        synth_dir=args.synth_dir,
        targets=targets,
        seed=args.seed,
        build_synth_if_missing=args.build_synth,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
