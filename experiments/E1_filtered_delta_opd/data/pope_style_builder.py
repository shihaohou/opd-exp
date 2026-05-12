"""
POPE-style yes/no builder on COCO train2017.

Why this exists:
    Official POPE `random` / `popular` / `adversarial` splits all live on
    the SAME ~500 COCO val2014 images (varying only in negative-sampling
    strategy). Using them for training would leak images into POPE-adv eval.
    Additionally, COCO train2017 is NOT split-disjoint from val2014 by
    image id — train2017 = train2014 ∪ (val2014 \\ minival5k), so the
    35k "remainder" val2014 ids are *inside* train2017. The disjointness
    check is therefore not automatic; we filter explicitly against POPE-adv
    image ids loaded at runtime.

Spec (from `experiments/E1_filtered_delta_opd/README.md` § Bucket 2):
    Base images   : COCO train2017 (NOT val)
    Annotations   : COCO instance annotations (80 stuff categories)
    Yes/No ratio  : 1:1
    Negative mix  : balanced random / popular / co-occurring (POPE-adv scheme)
    Template      : "Is there a {object} in the image?" → \\boxed{Yes/No}
    Disjoint check: image_id ∩ POPE-adv == ∅ (asserted at smoke test)

This module exposes:
    * ``COCOInstanceIndex``      — reusable indices over the COCO instances JSON
    * ``POPEStyleSample``        — dataclass matching ``ViRL39KSample`` shape
    * ``load_pope_adv_image_ids`` — extract COCO image ids from POPE-adv on disk
    * ``iter_pope_style``        — generator of ``POPEStyleSample``, balanced sampling

``POPEStyleSample`` mirrors the ViRL39K sample dataclass so the rest of the
pipeline (precompute, parquet builder, mixture) doesn't need to know which
bucket emitted a row.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from PIL import Image


logger = logging.getLogger(__name__)


DEFAULT_NEG_WEIGHTS: dict[str, float] = {
    "random": 1.0,
    "popular": 1.0,
    "cooccur": 1.0,
}


# ---------------------------------------------------------------------------
# Sample dataclass — mirrors ViRL39KSample for downstream compatibility.
# ---------------------------------------------------------------------------

@dataclass
class POPEStyleSample:
    """One POPE-style yes/no row built on COCO train2017.

    The shape is intentionally identical to ``ViRL39KSample`` so callers
    (precompute, parquet builder, mixture) can iterate uniformly.
    """

    sample_id: str
    question: str
    image_paths: list[Path]
    gold: str                      # "Yes" or "No" (matches POPE answer space)
    extras: dict[str, Any] = field(default_factory=dict)

    def load_images(self) -> list[Image.Image]:
        return [Image.open(p).convert("RGB") for p in self.image_paths]


# ---------------------------------------------------------------------------
# COCO instances index.
# ---------------------------------------------------------------------------

@dataclass
class COCOInstanceIndex:
    """Indices over a COCO instance annotations JSON.

    Build via ``COCOInstanceIndex.from_json(instances_train2017.json)``.

    `popularity` is sorted descending by total instance count across the
    train set (this matches POPE's "popular" negative sampling, which
    picks the most common categories absent from the image).

    `cooccur_top` is the image-level co-occurrence ranking: for category A,
    `cooccur_top[A]` lists other categories ordered by how many train
    images contain both A and B. This is POPE-adv's negative pool — pick
    a top co-occurrent of an in-image category that is NOT itself in the
    image.
    """

    cat_id_to_name: dict[int, str]
    img_to_cats: dict[int, set[int]]
    img_to_file: dict[int, str]
    image_ids: list[int]
    all_cat_ids: list[int]
    popularity: list[int]                # cat_ids, descending
    cooccur_top: dict[int, list[int]]    # cat_id → ordered list of co-occurring cat_ids

    @classmethod
    def from_json(cls, annotations_path: Path | str) -> "COCOInstanceIndex":
        path = Path(annotations_path)
        logger.info("[coco] loading %s ...", path)
        with open(path) as f:
            data = json.load(f)

        cat_id_to_name: dict[int, str] = {c["id"]: c["name"] for c in data["categories"]}
        all_cat_ids: list[int] = sorted(cat_id_to_name.keys())

        img_to_cats: dict[int, set[int]] = defaultdict(set)
        cat_count: Counter[int] = Counter()
        for ann in data["annotations"]:
            img_to_cats[ann["image_id"]].add(ann["category_id"])
            cat_count[ann["category_id"]] += 1

        img_to_file: dict[int, str] = {img["id"]: img["file_name"] for img in data["images"]}
        # Only images with ≥1 instance annotation — pure-background images
        # are useless for "yes" questions and ambiguous for "no".
        image_ids: list[int] = sorted(img_to_cats.keys())

        popularity: list[int] = [c for c, _ in cat_count.most_common()]
        # Pad with any categories not seen in this split (rare but safe).
        seen = set(popularity)
        for c in all_cat_ids:
            if c not in seen:
                popularity.append(c)

        cooccur_count: dict[int, Counter[int]] = {c: Counter() for c in all_cat_ids}
        for cats in img_to_cats.values():
            cat_list = list(cats)
            for i, a in enumerate(cat_list):
                for b in cat_list[i + 1:]:
                    cooccur_count[a][b] += 1
                    cooccur_count[b][a] += 1
        cooccur_top: dict[int, list[int]] = {
            c: [other for other, _ in cnt.most_common()]
            for c, cnt in cooccur_count.items()
        }

        logger.info(
            "[coco] loaded: %d images with annotations, %d categories",
            len(image_ids), len(all_cat_ids),
        )
        return cls(
            cat_id_to_name=cat_id_to_name,
            img_to_cats=dict(img_to_cats),
            img_to_file=img_to_file,
            image_ids=image_ids,
            all_cat_ids=all_cat_ids,
            popularity=popularity,
            cooccur_top=cooccur_top,
        )

    def pick_negative(
        self,
        image_cats: set[int],
        neg_type: str,
        rng: random.Random,
    ) -> Optional[int]:
        """Return a category id NOT in ``image_cats`` per POPE sampling strategy.

        Returns ``None`` only when no valid negative exists (which only
        happens when the image contains every category — essentially never
        on COCO).
        """
        if neg_type == "random":
            candidates = [c for c in self.all_cat_ids if c not in image_cats]
            return rng.choice(candidates) if candidates else None

        if neg_type == "popular":
            for c in self.popularity:
                if c not in image_cats:
                    return c
            return None

        if neg_type == "cooccur":
            if not image_cats:
                return None
            seed_cat = rng.choice(list(image_cats))
            for other in self.cooccur_top[seed_cat]:
                if other not in image_cats:
                    return other
            # Fall back to popular if no top co-occurrent is absent (rare).
            for c in self.popularity:
                if c not in image_cats:
                    return c
            return None

        raise ValueError(f"unknown neg_type: {neg_type!r}")


# ---------------------------------------------------------------------------
# POPE-adv eval ID extraction (for image-level disjointness check).
# ---------------------------------------------------------------------------

_COCO_FILENAME_ID_RE = re.compile(r"(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def load_pope_adv_image_ids(pope_adv_root: Path | str) -> set[int]:
    """Extract COCO numeric image ids from POPE-adv saved with ``save_to_disk``.

    POPE on HF is ``lmms-lab/POPE``. The on-disk schema varies between
    snapshots — sometimes the image filename is in ``image_source`` /
    ``image_path`` / ``image_name``; sometimes only the decoded PIL is
    stored under ``image``. We probe common columns and parse the trailing
    digit run (``COCO_val2014_000000005802.jpg`` → ``5802``).

    Returns an empty set when no parseable column is found — the caller
    should then pass ids explicitly via ``--pope-eval-ids-file`` or accept
    the missing-filter warning.
    """
    pope_adv_root = Path(pope_adv_root)
    try:
        from datasets import load_from_disk
    except ImportError:
        logger.warning("[pope-adv] `datasets` package not installed; returning empty set")
        return set()

    try:
        ds = load_from_disk(str(pope_adv_root))
    except Exception as e:
        logger.warning("[pope-adv] load_from_disk failed: %s", e)
        return set()

    if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
        # DatasetDict → union ids across splits
        all_ids: set[int] = set()
        for split_name in ds.keys():
            all_ids |= _extract_ids_from_split(ds[split_name])
        return all_ids

    return _extract_ids_from_split(ds)


def _extract_ids_from_split(ds_split) -> set[int]:
    """Probe id-bearing columns in a single HF Dataset split."""
    cols = set(ds_split.column_names)
    ids: set[int] = set()

    # Preferred: explicit image_id / image_path / image_source columns.
    for col in ("image_id", "image_path", "image_source", "image_name", "file_name"):
        if col in cols:
            for v in ds_split[col]:
                ids |= _parse_image_id_from_value(v)
            if ids:
                return ids

    # Fallback: `image` column may be a dict with a `path` entry.
    if "image" in cols and len(ds_split) > 0:
        first = ds_split[0]["image"]
        if isinstance(first, dict) and ("path" in first or "filename" in first):
            for img in ds_split["image"]:
                ids |= _parse_image_id_from_value(img.get("path") or img.get("filename") or "")
            if ids:
                return ids

    return ids


def _parse_image_id_from_value(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, str):
        m = _COCO_FILENAME_ID_RE.search(value)
        if m:
            return {int(m.group(1))}
        try:
            return {int(value)}
        except ValueError:
            return set()
    return set()


# ---------------------------------------------------------------------------
# Main sampler.
# ---------------------------------------------------------------------------

def iter_pope_style(
    coco_train_root: Path | str,
    annotations_path: Path | str,
    pope_adv_eval_ids: Optional[set[int]] = None,
    *,
    n_samples: int = 1600,
    yes_no_ratio: float = 0.5,
    neg_type_weights: Optional[dict[str, float]] = None,
    seed: int = 42,
    min_objects_per_image: int = 1,
    coco_images_subdir: str = "train2017",
) -> Iterator[POPEStyleSample]:
    """Yield POPE-style yes/no samples drawn from COCO train2017.

    Emission scheme: for each shuffled eligible image, emit one yes (random
    in-image category) and one no (negative category sampled per a
    quota-weighted type chosen from ``neg_type_weights``). One yes + one no
    per image avoids per-image overuse while preserving the 1:1 global
    yes:no ratio.

    Args:
        coco_train_root: Path containing the ``train2017/`` images
            sub-directory (default ``coco_images_subdir``).
        annotations_path: Path to ``instances_train2017.json``.
        pope_adv_eval_ids: Set of COCO image ids reserved for POPE-adv
            evaluation; eligible-image set is filtered against these. Pass
            ``set()`` to opt out — only valid for smoke tests.
        n_samples: Target sample count (default 1600 = E1 bucket-2 budget).
        yes_no_ratio: Fraction of samples that should be ``Yes``. 0.5 = balanced.
        neg_type_weights: ``{random/popular/cooccur: weight}``. Default uniform.
        seed: RNG seed for reproducibility.
        min_objects_per_image: Drop images with fewer than this many
            distinct categories — POPE-adv's "cooccur" mode needs ≥1 GT.
        coco_images_subdir: Sub-directory under ``coco_train_root`` holding
            the image files. COCO ships images under ``train2017/`` by
            default; pass ``""`` if images are directly in the root.
    """
    coco_train_root = Path(coco_train_root)
    annotations_path = Path(annotations_path)
    rng = random.Random(seed)

    index = COCOInstanceIndex.from_json(annotations_path)

    eval_ids = pope_adv_eval_ids or set()
    eligible_ids = [
        iid for iid in index.image_ids
        if iid not in eval_ids
        and len(index.img_to_cats[iid]) >= min_objects_per_image
    ]
    n_filtered_for_eval = sum(1 for iid in index.image_ids if iid in eval_ids)
    n_filtered_no_objects = sum(
        1 for iid in index.image_ids
        if iid not in eval_ids and len(index.img_to_cats[iid]) < min_objects_per_image
    )
    logger.info(
        "[pope_style] eligible images: %d (filtered: %d for POPE-adv overlap, %d for too few objects)",
        len(eligible_ids), n_filtered_for_eval, n_filtered_no_objects,
    )

    rng.shuffle(eligible_ids)

    weights = dict(neg_type_weights or DEFAULT_NEG_WEIGHTS)
    neg_types = list(weights.keys())
    total_w = sum(weights.values())

    target_yes = int(round(n_samples * yes_no_ratio))
    target_no = n_samples - target_yes
    target_no_per_type: dict[str, int] = {}
    assigned = 0
    for i, t in enumerate(neg_types):
        if i == len(neg_types) - 1:
            target_no_per_type[t] = target_no - assigned
        else:
            n = int(round(target_no * weights[t] / total_w))
            target_no_per_type[t] = n
            assigned += n
    logger.info(
        "[pope_style] targets: yes=%d, no=%d (per-type: %s), seed=%d",
        target_yes, target_no, target_no_per_type, seed,
    )

    n_yes_emitted = 0
    n_no_emitted: dict[str, int] = {t: 0 for t in neg_types}

    for image_id in eligible_ids:
        n_no_total = sum(n_no_emitted.values())
        if n_yes_emitted >= target_yes and n_no_total >= target_no:
            break

        file_name = index.img_to_file[image_id]
        if coco_images_subdir:
            image_path = coco_train_root / coco_images_subdir / file_name
        else:
            image_path = coco_train_root / file_name
        image_cats = index.img_to_cats[image_id]

        # --- Yes question (one per image) ---
        if n_yes_emitted < target_yes and image_cats:
            cat_id = rng.choice(list(image_cats))
            cat_name = index.cat_id_to_name[cat_id]
            yield POPEStyleSample(
                sample_id=f"pope_style_{image_id}_yes_{cat_id}",
                question=f"Is there a {cat_name} in the image?",
                image_paths=[image_path],
                gold="Yes",
                extras={
                    "neg_type": None,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "category": cat_name,
                    "source": "coco_train2017",
                },
            )
            n_yes_emitted += 1

        # --- No question (one per image, pick a type with remaining quota) ---
        types_in_random_order = neg_types[:]
        rng.shuffle(types_in_random_order)
        types_with_quota = [
            t for t in types_in_random_order
            if n_no_emitted[t] < target_no_per_type[t]
        ]
        for t in types_with_quota:
            neg_cat = index.pick_negative(image_cats, t, rng)
            if neg_cat is None:
                continue
            neg_name = index.cat_id_to_name[neg_cat]
            yield POPEStyleSample(
                sample_id=f"pope_style_{image_id}_no_{t}_{neg_cat}",
                question=f"Is there a {neg_name} in the image?",
                image_paths=[image_path],
                gold="No",
                extras={
                    "neg_type": t,
                    "image_id": image_id,
                    "category_id": neg_cat,
                    "category": neg_name,
                    "source": "coco_train2017",
                },
            )
            n_no_emitted[t] += 1
            break

    total_emitted = n_yes_emitted + sum(n_no_emitted.values())
    if total_emitted < n_samples:
        logger.warning(
            "[pope_style] short on samples: yielded %d/%d (yes=%d, no=%s); "
            "consider lowering `min_objects_per_image` or relaxing eval-id filter",
            total_emitted, n_samples, n_yes_emitted,
            {t: n_no_emitted[t] for t in neg_types},
        )
    else:
        logger.info(
            "[pope_style] done: yielded yes=%d no=%s total=%d",
            n_yes_emitted, n_no_emitted, total_emitted,
        )


# ---------------------------------------------------------------------------
# CLI smoke test — runs on the server after COCO train2017 is on disk.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from collections import Counter as _Counter

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="POPE-style builder smoke test")
    p.add_argument(
        "--coco-root",
        default="/home/web_server/antispam/project/houshihao/datasets/coco",
        help="COCO root containing train2017/ and annotations/",
    )
    p.add_argument(
        "--annotations", default=None,
        help="Path to instances_train2017.json (defaults to <coco-root>/annotations/instances_train2017.json)",
    )
    p.add_argument(
        "--pope-adv-root",
        default="/home/web_server/antispam/project/houshihao/datasets/POPE-adversarial",
        help="Path to POPE-adversarial save_to_disk dir for image-id disjointness check",
    )
    p.add_argument("--n-samples", type=int, default=20, help="Sample count (tiny for smoke)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-show", type=int, default=5, help="How many samples to dump verbatim")
    p.add_argument(
        "--no-image-load", action="store_true",
        help="Skip the load_images() probe (use when COCO images are not yet on disk)",
    )
    args = p.parse_args()

    ann_path = Path(args.annotations) if args.annotations else (
        Path(args.coco_root) / "annotations" / "instances_train2017.json"
    )

    pope_adv_ids = load_pope_adv_image_ids(args.pope_adv_root)
    print(f"[pope-adv] loaded {len(pope_adv_ids)} image ids from {args.pope_adv_root}")

    samples = list(iter_pope_style(
        coco_train_root=args.coco_root,
        annotations_path=ann_path,
        pope_adv_eval_ids=pope_adv_ids,
        n_samples=args.n_samples,
        seed=args.seed,
    ))

    print(f"[smoke] total samples: {len(samples)}")
    yes_count = sum(1 for s in samples if s.gold == "Yes")
    print(f"[smoke] yes={yes_count} no={len(samples) - yes_count}")
    neg_dist = _Counter(s.extras["neg_type"] for s in samples if s.gold == "No")
    print(f"[smoke] no breakdown by neg_type: {dict(neg_dist)}")
    cat_dist = _Counter(s.extras["category"] for s in samples)
    print(f"[smoke] top categories: {dict(cat_dist.most_common(10))}")

    train_ids = {s.extras["image_id"] for s in samples}
    overlap = train_ids & pope_adv_ids
    assert not overlap, f"POPE-adv image-id overlap detected: {sorted(overlap)[:5]}"
    print(f"[disjoint] OK — 0 overlap with POPE-adv ({len(pope_adv_ids)} eval ids)")

    print(f"\nfirst {args.n_show} samples:")
    for s in samples[:args.n_show]:
        print(f"--- {s.sample_id} ---")
        print(f"  question : {s.question!r}")
        print(f"  gold     : {s.gold}")
        print(f"  category : {s.extras['category']} (id={s.extras['category_id']})")
        print(f"  neg_type : {s.extras['neg_type']}")
        print(f"  image    : {s.image_paths[0]}")
        if not args.no_image_load:
            try:
                img = s.load_images()[0]
                print(f"  size     : {img.size}  mode={img.mode}")
            except FileNotFoundError as e:
                print(f"  LOAD ERR : {e}")
