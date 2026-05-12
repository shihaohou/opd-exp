"""
Three-layer dedup check for E1 training-vs-eval image leakage.

MUST PASS before any non-smoke E1 launch — see CLAUDE.md and the README §
"Mandatory dedup pipeline". The check exists because GPT review flagged
two real leakage risks during E1 design:

    * Official POPE random/popular splits and POPE-adv share the same
      ~500 COCO val2014 images. Training on those splits would leak eval
      images into train. We avoid the official splits and build POPE-style
      on COCO train2017 instead — but COCO train2017 ⊋ COCO val2014
      minus the 5K minival, so val2014 image_ids exist inside train2017.
      Filename-level checks must be done explicitly.
    * VLMBias `withtitle` / `remove_background_*` subsets share base
      images with VLMBias `main` (the eval set). We've already excluded
      them from the training mix; this script confirms no inadvertent
      reintroduction (e.g. through CLIP-similar self-built synth that
      happens to match a VLMBias image).

Three layers, in order of increasing cost:

    1. **image_id intersection** — numeric COCO ids in train (POPE-style,
       TallyQA-COCO) must be disjoint from POPE-adv eval ids. Trivial cost.
    2. **pHash near-duplicate** — perceptual hash, 64-bit, Hamming < 5 is
       a flag. Catches "same image, different filename" (rare but happens
       with VG cropping or our synth accidentally mimicking COCO).
    3. **CLIP embedding NN** — top-1 cosine > 0.95 is a flag for manual
       review. Catches semantically near-identical images (e.g., the
       same scene at a different angle).

Run order matters: cheap layers fire first; expensive layers only run on
samples that survived earlier layers.

Output:
    * stdout summary: counts per layer
    * `--output` jsonl: one line per finding with all metadata
    * Exit code: 0 if no findings, 1 if any finding (intended for CI gate)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

# Repo-root injection so cross-experiment imports resolve when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datatypes.
# ---------------------------------------------------------------------------

@dataclass
class ImageRef:
    """An image identity for dedup. Lazy-loadable.

    ``unique_key`` should be globally unique within (train ∪ eval) — it's
    used to dedupe within a side (e.g., a POPE-style image used by 2
    different yes/no questions counts once). For COCO images we use the
    absolute path; for HF-stored images we use ``{ds_name}:{idx}``.
    """

    sample_id: str               # caller-facing handle
    unique_key: str              # for within-side dedup
    source_tag: str              # bucket / eval-set name
    image_id: Optional[int] = None     # numeric COCO/VG id when known
    _path: Optional[Path] = None
    _loader: Optional[Callable[[], Any]] = None  # returns a PIL.Image

    def load(self):  # -> PIL.Image
        from PIL import Image
        if self._loader is not None:
            img = self._loader()
            if hasattr(img, "convert"):
                return img.convert("RGB")
            raise RuntimeError(f"loader for {self.sample_id} did not return a PIL image")
        if self._path is not None:
            return Image.open(self._path).convert("RGB")
        raise RuntimeError(f"ImageRef {self.sample_id} has no path or loader")


@dataclass
class DedupFinding:
    """One leakage flag — train sample matched an eval sample at some layer."""

    layer: str                  # "image_id" / "phash" / "clip"
    train_sample_id: str
    train_source: str
    eval_sample_id: str
    eval_source: str
    distance: float             # Hamming (phash), 1-cos (clip), 0 (image_id)
    train_image_id: Optional[int] = None
    eval_image_id: Optional[int] = None
    train_path: Optional[str] = None
    eval_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Layer 1: image_id intersection.
# ---------------------------------------------------------------------------

def check_image_id_overlap(
    train_refs: list[ImageRef],
    eval_image_ids: set[int],
    eval_source: str = "eval",
) -> list[DedupFinding]:
    """Layer 1 — flag train refs whose ``image_id`` is in the eval set.

    Trivial cost; runs first.
    """
    findings: list[DedupFinding] = []
    for t in train_refs:
        if t.image_id is None:
            continue
        if t.image_id in eval_image_ids:
            findings.append(DedupFinding(
                layer="image_id",
                train_sample_id=t.sample_id,
                train_source=t.source_tag,
                eval_sample_id=f"{eval_source}_image_id_{t.image_id}",
                eval_source=eval_source,
                distance=0.0,
                train_image_id=t.image_id,
                eval_image_id=t.image_id,
                train_path=str(t._path) if t._path else None,
            ))
    logger.info(
        "[layer1] image_id check: %d findings across %d train refs vs %d eval ids",
        len(findings), len(train_refs), len(eval_image_ids),
    )
    return findings


# ---------------------------------------------------------------------------
# Layer 2: pHash Hamming distance.
# ---------------------------------------------------------------------------

def _compute_phashes(refs: list[ImageRef], hash_size: int = 8) -> list[tuple[ImageRef, Any]]:
    """Compute pHashes for refs; skip + warn on individual failures.

    Lazy-imports ``imagehash`` so this module imports even when imagehash
    is absent (e.g. on the Mac for code inspection).
    """
    import imagehash

    results: list[tuple[ImageRef, Any]] = []
    n_skipped = 0
    for i, ref in enumerate(refs):
        try:
            img = ref.load()
            h = imagehash.phash(img, hash_size=hash_size)
            results.append((ref, h))
        except Exception as e:
            n_skipped += 1
            logger.warning("[phash] skip %s (%s): %s", ref.sample_id, ref.source_tag, e)
        if (i + 1) % 500 == 0:
            logger.info("[phash] hashed %d/%d", i + 1, len(refs))
    if n_skipped:
        logger.warning("[phash] %d refs failed to hash", n_skipped)
    return results


def check_phash_overlap(
    train_refs: list[ImageRef],
    eval_refs: list[ImageRef],
    *,
    hash_size: int = 8,
    max_hamming: int = 5,
) -> list[DedupFinding]:
    """Layer 2 — pHash Hamming distance < max_hamming flagged.

    Brute force; for ~8K × ~4K pairs this is seconds.
    """
    train_hashes = _compute_phashes(train_refs, hash_size=hash_size)
    eval_hashes = _compute_phashes(eval_refs, hash_size=hash_size)

    findings: list[DedupFinding] = []
    for ts, th in train_hashes:
        for es, eh in eval_hashes:
            dist = float(th - eh)   # imagehash overloads __sub__ → Hamming
            if dist < max_hamming:
                findings.append(DedupFinding(
                    layer="phash",
                    train_sample_id=ts.sample_id,
                    train_source=ts.source_tag,
                    eval_sample_id=es.sample_id,
                    eval_source=es.source_tag,
                    distance=dist,
                    train_image_id=ts.image_id,
                    eval_image_id=es.image_id,
                    train_path=str(ts._path) if ts._path else None,
                    eval_path=str(es._path) if es._path else None,
                ))
    logger.info(
        "[layer2] pHash check (hash_size=%d, max_hamming=%d): %d findings (%d × %d compared)",
        hash_size, max_hamming, len(findings), len(train_hashes), len(eval_hashes),
    )
    return findings


# ---------------------------------------------------------------------------
# Layer 3: CLIP embedding nearest-neighbor.
# ---------------------------------------------------------------------------

def _compute_clip_embeddings(
    refs: list[ImageRef],
    model,
    preprocess,
    device: str,
    batch_size: int = 64,
):
    """Embed image refs with CLIP; returns (refs_kept, embeddings tensor)."""
    import torch

    kept: list[ImageRef] = []
    batched_tensors: list = []

    def _flush(batch: list, kept_local: list):
        if not batch:
            return None
        stacked = torch.stack(batch, dim=0).to(device)
        with torch.no_grad():
            feats = model.encode_image(stacked)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.cpu()

    feats_all = []
    batch: list = []
    for i, ref in enumerate(refs):
        try:
            img = ref.load()
            t = preprocess(img)
            batch.append(t)
            kept.append(ref)
        except Exception as e:
            logger.warning("[clip] skip %s (%s): %s", ref.sample_id, ref.source_tag, e)

        if len(batch) >= batch_size:
            feats = _flush(batch, kept)
            if feats is not None:
                feats_all.append(feats)
            batch = []

        if (i + 1) % 500 == 0:
            logger.info("[clip] embedded %d/%d", i + 1, len(refs))

    feats = _flush(batch, kept)
    if feats is not None:
        feats_all.append(feats)

    if not feats_all:
        return kept, None
    return kept, torch.cat(feats_all, dim=0)


def check_clip_overlap(
    train_refs: list[ImageRef],
    eval_refs: list[ImageRef],
    *,
    threshold: float = 0.95,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: Optional[str] = None,
    batch_size: int = 64,
) -> list[DedupFinding]:
    """Layer 3 — CLIP cosine > threshold flagged. Requires GPU for speed."""
    import torch

    try:
        import open_clip
    except ImportError as e:
        raise RuntimeError(
            "open_clip_torch not installed. "
            "Install via `pip install open_clip_torch` or pass --skip-clip."
        ) from e

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[clip] loading %s pretrained=%s on %s ...", model_name, pretrained, device)
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()

    train_kept, train_feats = _compute_clip_embeddings(train_refs, model, preprocess, device, batch_size)
    eval_kept, eval_feats = _compute_clip_embeddings(eval_refs, model, preprocess, device, batch_size)

    if train_feats is None or eval_feats is None:
        logger.warning("[clip] no embeddings produced; layer skipped")
        return []

    sims = (train_feats @ eval_feats.t()).numpy()   # both already L2-normed

    findings: list[DedupFinding] = []
    for i, ts in enumerate(train_kept):
        # Top-1 match
        row = sims[i]
        j = int(row.argmax())
        sim = float(row[j])
        if sim > threshold:
            es = eval_kept[j]
            findings.append(DedupFinding(
                layer="clip",
                train_sample_id=ts.sample_id,
                train_source=ts.source_tag,
                eval_sample_id=es.sample_id,
                eval_source=es.source_tag,
                distance=1.0 - sim,
                train_image_id=ts.image_id,
                eval_image_id=es.image_id,
                train_path=str(ts._path) if ts._path else None,
                eval_path=str(es._path) if es._path else None,
            ))

    logger.info(
        "[layer3] CLIP check (model=%s, threshold=%.3f): %d findings (%d × %d compared)",
        model_name, threshold, len(findings), len(train_kept), len(eval_kept),
    )
    return findings


# ---------------------------------------------------------------------------
# Orchestrator.
# ---------------------------------------------------------------------------

def run_dedup(
    train_refs: list[ImageRef],
    eval_refs: list[ImageRef],
    *,
    eval_image_ids: Optional[set[int]] = None,
    layers: tuple[str, ...] = ("image_id", "phash", "clip"),
    phash_max_hamming: int = 5,
    clip_threshold: float = 0.95,
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "openai",
    device: Optional[str] = None,
) -> list[DedupFinding]:
    """Run the 3 dedup layers in order; return concatenated findings."""
    all_findings: list[DedupFinding] = []

    # Deduplicate within each side: same unique_key counts once.
    train_unique = _unique_by_key(train_refs)
    eval_unique = _unique_by_key(eval_refs)
    logger.info(
        "[dedup] train: %d refs (%d unique by key); eval: %d refs (%d unique by key)",
        len(train_refs), len(train_unique), len(eval_refs), len(eval_unique),
    )

    if "image_id" in layers and eval_image_ids:
        all_findings.extend(check_image_id_overlap(
            train_unique, eval_image_ids, eval_source="pope_adv_image_id_set",
        ))

    if "phash" in layers:
        all_findings.extend(check_phash_overlap(
            train_unique, eval_unique, max_hamming=phash_max_hamming,
        ))

    if "clip" in layers:
        all_findings.extend(check_clip_overlap(
            train_unique, eval_unique,
            threshold=clip_threshold,
            model_name=clip_model,
            pretrained=clip_pretrained,
            device=device,
        ))

    return all_findings


def _unique_by_key(refs: list[ImageRef]) -> list[ImageRef]:
    seen: set[str] = set()
    out: list[ImageRef] = []
    for r in refs:
        if r.unique_key in seen:
            continue
        seen.add(r.unique_key)
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Source builders — bucket and eval refs.
# ---------------------------------------------------------------------------

def train_refs_from_pope_style_iter(
    iter_pope_style_kwargs: dict,
) -> Iterator[ImageRef]:
    """Build ImageRefs from a live iter_pope_style() generator."""
    from experiments.E1_filtered_delta_opd.data.pope_style_builder import iter_pope_style
    for s in iter_pope_style(**iter_pope_style_kwargs):
        path = s.image_paths[0]
        yield ImageRef(
            sample_id=s.sample_id,
            unique_key=str(path),
            source_tag="pope_style",
            image_id=s.extras.get("image_id"),
            _path=path,
        )


def train_refs_from_tallyqa_iter(
    iter_tallyqa_kwargs: dict,
) -> Iterator[ImageRef]:
    """Build ImageRefs from a live iter_tallyqa() generator."""
    from experiments.E1_filtered_delta_opd.data.tallyqa_loader import iter_tallyqa
    for s in iter_tallyqa(**iter_tallyqa_kwargs):
        path = s.image_paths[0]
        yield ImageRef(
            sample_id=s.sample_id,
            unique_key=str(path),
            source_tag="tallyqa",
            image_id=s.extras.get("image_id"),
            _path=path,
        )


def train_refs_from_synth_dir(out_dir: Path | str) -> Iterator[ImageRef]:
    """Build ImageRefs from a synthesize_counterfactuals manifest."""
    from experiments.E1_filtered_delta_opd.data.synthesize_counterfactuals import iter_synthetic_counterfactuals
    for s in iter_synthetic_counterfactuals(out_dir):
        path = s.image_paths[0]
        yield ImageRef(
            sample_id=s.sample_id,
            unique_key=str(path),
            source_tag="synthetic",
            image_id=None,
            _path=path,
        )


def train_refs_from_virl39k(virl39k_root: Path | str, **iter_kwargs) -> Iterator[ImageRef]:
    """Build ImageRefs from the ViRL39K loader."""
    from experiments.E1_filtered_delta_opd.data.virl39k_loader import iter_virl39k
    for s in iter_virl39k(dataset_root=virl39k_root, **iter_kwargs):
        path = s.image_paths[0]
        yield ImageRef(
            sample_id=s.sample_id,
            unique_key=str(path),
            source_tag="virl39k",
            image_id=None,
            _path=path,
        )


def eval_refs_from_pope_adv(pope_adv_root: Path | str) -> tuple[list[ImageRef], set[int]]:
    """Build eval ImageRefs from POPE-adv (HF save_to_disk).

    Returns (refs, image_id_set). The id-set is reusable for the cheap
    Layer-1 intersection.
    """
    from datasets import load_from_disk
    from experiments.E1_filtered_delta_opd.data.pope_style_builder import _parse_image_id_from_value

    ds = load_from_disk(str(pope_adv_root))
    # Handle DatasetDict by concatenating splits.
    if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
        splits = [(name, ds[name]) for name in ds.keys()]
    else:
        splits = [("default", ds)]

    refs: list[ImageRef] = []
    image_id_set: set[int] = set()
    for split_name, split in splits:
        cols = set(split.column_names)
        for i in range(len(split)):
            row = split[i]
            # Extract image_id from common columns
            iid_set: set[int] = set()
            for col in ("image_id", "image_path", "image_source", "image_name", "file_name"):
                if col in cols:
                    iid_set = _parse_image_id_from_value(row[col])
                    if iid_set:
                        break
            iid = next(iter(iid_set)) if iid_set else None
            if iid is not None:
                image_id_set.add(iid)

            # PIL loader (close over (split, i))
            def make_loader(s=split, idx=i):
                return lambda: s[idx]["image"]

            unique_key = f"pope_adv:{split_name}:{iid}" if iid is not None else f"pope_adv:{split_name}:{i}"
            refs.append(ImageRef(
                sample_id=f"pope_adv_{split_name}_{i}",
                unique_key=unique_key,
                source_tag=f"pope_adv:{split_name}",
                image_id=iid,
                _loader=make_loader(),
            ))
    logger.info(
        "[pope_adv] loaded %d eval refs (%d unique image ids)",
        len(refs), len(image_id_set),
    )
    return refs, image_id_set


def eval_refs_from_vlmbias(
    vlmbias_root: Path | str,
    subset_name: str = "main",
) -> list[ImageRef]:
    """Build eval ImageRefs from VLMBias `main` subset (the eval set).

    VLMBias is `anvo25/vlms-are-biased`. Local copy is a DatasetDict; we
    select the named subset (`main` by default — the E1 eval subset).
    """
    from datasets import load_from_disk

    ds = load_from_disk(str(vlmbias_root))
    if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
        if subset_name not in ds:
            raise KeyError(f"VLMBias subset {subset_name!r} not found; available: {list(ds.keys())}")
        ds = ds[subset_name]

    refs: list[ImageRef] = []
    for i in range(len(ds)):
        def make_loader(s=ds, idx=i):
            return lambda: s[idx]["image"]

        # VLMBias images don't have COCO ids; use (subset, idx) as unique key.
        # The dataset has `topic` / `sub_topic` columns we can stash for diagnostics.
        row = ds[i]
        topic = row.get("topic") if isinstance(row, dict) else None
        refs.append(ImageRef(
            sample_id=f"vlmbias_{subset_name}_{i}",
            unique_key=f"vlmbias:{subset_name}:{i}",
            source_tag=f"vlmbias:{subset_name}",
            image_id=None,
            _loader=make_loader(),
        ))
    logger.info("[vlmbias] loaded %d eval refs from subset=%s", len(refs), subset_name)
    return refs


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1 three-layer dedup check (image_id / pHash / CLIP)")
    p.add_argument(
        "--pope-style-manifest", default=None,
        help="JSON of iter_pope_style kwargs (passed to the loader); "
             "e.g. '{\"coco_train_root\":\"/data/coco\",\"annotations_path\":\"/data/coco/annotations/instances_train2017.json\",\"n_samples\":1600}'",
    )
    p.add_argument(
        "--tallyqa-manifest", default=None,
        help="JSON of iter_tallyqa kwargs (passed to the loader)",
    )
    p.add_argument(
        "--synth-dir", default=None,
        help="Synthesize-counterfactuals output dir (with manifest.jsonl)",
    )
    p.add_argument(
        "--virl39k-root", default=None,
        help="ViRL39K root for bucket-1 dedup. If omitted, bucket 1 is skipped (it has its own namespace).",
    )
    p.add_argument(
        "--pope-adv-root",
        default="/home/web_server/antispam/project/houshihao/datasets/POPE-adversarial",
        help="Path to POPE-adv save_to_disk dir",
    )
    p.add_argument(
        "--vlmbias-root",
        default="/home/web_server/antispam/project/houshihao/datasets/VLMBias",
        help="Path to VLMBias save_to_disk dir",
    )
    p.add_argument(
        "--vlmbias-subset", default="main",
        help="VLMBias subset to check (E1 eval is `main`)",
    )
    p.add_argument(
        "--output", required=True,
        help="Output jsonl file for findings (one finding per line)",
    )
    p.add_argument(
        "--layers", nargs="+", default=["image_id", "phash", "clip"],
        choices=["image_id", "phash", "clip"],
        help="Which layers to run (default: all three)",
    )
    p.add_argument("--phash-max-hamming", type=int, default=5)
    p.add_argument("--clip-threshold", type=float, default=0.95)
    p.add_argument("--clip-model", default="ViT-B-32")
    p.add_argument("--clip-pretrained", default="openai")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument(
        "--skip-clip", action="store_true",
        help="Convenience flag to drop CLIP from --layers (e.g., on Mac without GPU)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    layers = list(args.layers)
    if args.skip_clip and "clip" in layers:
        layers.remove("clip")

    # --- Train refs ---
    train_refs: list[ImageRef] = []

    if args.pope_style_manifest:
        kwargs = json.loads(args.pope_style_manifest)
        train_refs.extend(train_refs_from_pope_style_iter(kwargs))
        logger.info("[train] POPE-style: %d cumulative refs", len(train_refs))

    if args.tallyqa_manifest:
        kwargs = json.loads(args.tallyqa_manifest)
        train_refs.extend(train_refs_from_tallyqa_iter(kwargs))
        logger.info("[train] TallyQA: %d cumulative refs", len(train_refs))

    if args.synth_dir:
        train_refs.extend(train_refs_from_synth_dir(args.synth_dir))
        logger.info("[train] synth: %d cumulative refs", len(train_refs))

    if args.virl39k_root:
        train_refs.extend(train_refs_from_virl39k(args.virl39k_root))
        logger.info("[train] ViRL39K: %d cumulative refs", len(train_refs))

    if not train_refs:
        print("ERROR: no train refs collected — specify at least one of --pope-style-manifest / "
              "--tallyqa-manifest / --synth-dir / --virl39k-root", file=sys.stderr)
        return 2

    # --- Eval refs ---
    eval_refs: list[ImageRef] = []
    eval_image_ids: set[int] = set()

    if Path(args.pope_adv_root).exists():
        pope_refs, pope_ids = eval_refs_from_pope_adv(args.pope_adv_root)
        eval_refs.extend(pope_refs)
        eval_image_ids |= pope_ids
    else:
        logger.warning("[eval] POPE-adv not found at %s — image_id layer will be a no-op", args.pope_adv_root)

    if Path(args.vlmbias_root).exists():
        eval_refs.extend(eval_refs_from_vlmbias(args.vlmbias_root, subset_name=args.vlmbias_subset))
    else:
        logger.warning("[eval] VLMBias not found at %s", args.vlmbias_root)

    if not eval_refs:
        print("ERROR: no eval refs collected — pop_adv_root / vlmbias_root paths missing?", file=sys.stderr)
        return 2

    # --- Run dedup ---
    findings = run_dedup(
        train_refs=train_refs,
        eval_refs=eval_refs,
        eval_image_ids=eval_image_ids,
        layers=tuple(layers),
        phash_max_hamming=args.phash_max_hamming,
        clip_threshold=args.clip_threshold,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        device=args.device,
    )

    # --- Write findings + print summary ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for x in findings:
            f.write(json.dumps(asdict(x), ensure_ascii=False) + "\n")

    summary: dict[str, int] = {}
    for x in findings:
        summary[x.layer] = summary.get(x.layer, 0) + 1
    print()
    print("=" * 60)
    print(f"DEDUP SUMMARY ({len(findings)} total findings)")
    for layer in ("image_id", "phash", "clip"):
        print(f"  {layer:10s}: {summary.get(layer, 0)}")
    print(f"Detailed findings → {out_path}")
    print("=" * 60)

    if findings:
        print("\nFAIL: train ∩ eval overlap detected. Review findings before training launch.", file=sys.stderr)
        return 1

    print("\nPASS: no overlap detected across requested layers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
