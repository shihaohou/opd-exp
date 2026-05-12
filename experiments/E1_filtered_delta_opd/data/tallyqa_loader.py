"""
TallyQA `complex`-subset loader with COCO image-id leakage filter.

TallyQA (Acharya et al. 2018) is a 287K-question counting QA dataset over
~165K images drawn from COCO (via the original VQA dataset) and Visual
Genome. The ``complex`` subset (rows with ``issimple == False``) requires
multi-step counting — a skill closer to VLMBias's animal-leg-count
failure mode than the simple ``how many <obj>`` patterns. We use the
complex subset to give the student a natural-image counting backbone
inside bucket 3.

Why this loader exists:
    Counting is the most underrepresented skill in our other buckets
    (ViRL39K is mostly math / charts / spatial; POPE-style is binary
    yes/no; synthetic counterfactuals cover specific failure topics). The
    TallyQA complex subset is also the natural complement to the
    synthetic counterfactuals (which use clean templated images) — it
    forces the student to handle counting on real photographs.

Spec (from `experiments/E1_filtered_delta_opd/README.md` § Bucket 3):
    Source       : `manoja328/TallyQA_dataset` (HF / GitHub release)
    Subset       : `issimple == False` only
    Target n     : ~900 (the rest of bucket 3 is synth counterfactuals)
    Image filter : drop samples whose image is from COCO **and** whose
                   `image_id` appears in POPE-adv eval set
    Answer fmt   : rewrap integer count → `\\boxed{N}` for verifier

The loader is JSON-first because the official TallyQA release ships as
two large JSON files (``train.json`` / ``test.json``). A jsonl variant
is also accepted in case the user has pre-split the corpus.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from PIL import Image


logger = logging.getLogger(__name__)


# TallyQA marks COCO-sourced rows under one of these labels depending on
# the snapshot. Anything else (most commonly "visualgenome" / "vg") is
# treated as Visual Genome — the VG image-id namespace is separate from
# COCO so we don't need to filter against POPE-adv on those rows.
_COCO_SOURCES = {"imageqa", "vqa", "coco", "mscoco", "coco2014", "coco2017"}


# ---------------------------------------------------------------------------
# Sample dataclass — same shape as ViRL39KSample / POPEStyleSample.
# ---------------------------------------------------------------------------

@dataclass
class TallyQASample:
    sample_id: str
    question: str
    image_paths: list[Path]
    gold: str                       # stringified integer count
    extras: dict[str, Any] = field(default_factory=dict)

    def load_images(self) -> list[Image.Image]:
        return [Image.open(p).convert("RGB") for p in self.image_paths]


# ---------------------------------------------------------------------------
# Raw record ingest.
# ---------------------------------------------------------------------------

def _iter_raw_records(path: Path) -> Iterator[dict]:
    """Yield rows from a TallyQA JSON or JSONL file.

    The official release ships a JSON list. Some users split the list into
    JSONL for streaming; both shapes are accepted.
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict) and "data" in data:
        yield from data["data"]
    else:
        raise ValueError(
            f"Unsupported TallyQA JSON shape at {path!r}: "
            f"expected list or {{'data': list}}; got {type(data).__name__}"
        )


def _is_coco_source(data_source: Optional[str]) -> bool:
    if not data_source:
        return False
    return data_source.lower() in _COCO_SOURCES


# ---------------------------------------------------------------------------
# Main loader.
# ---------------------------------------------------------------------------

def iter_tallyqa(
    json_path: Path | str,
    images_root: Path | str,
    *,
    complex_only: bool = True,
    coco_eval_image_ids: Optional[set[int]] = None,
    n_max: Optional[int] = 900,
    seed: int = 42,
    answer_max: int = 50,
    answer_min: int = 0,
    require_image_field: bool = True,
    shuffle: bool = True,
) -> Iterator[TallyQASample]:
    """Stream filtered TallyQA samples.

    Filtering order:
      1. ``complex_only``: drop ``issimple == True``.
      2. ``coco_eval_image_ids``: drop rows whose source is COCO **and**
         whose ``image_id`` is in that set.
      3. ``answer_min`` / ``answer_max``: drop counts outside this range.
         TallyQA has rare 100+ outliers that produce silly tokenized
         responses; default 0–50 covers ≥99% of complex.
      4. ``require_image_field``: drop rows without an ``image`` field
         (very rare; sanity guard).
      5. ``shuffle`` (seed-deterministic), then cap to ``n_max``.

    Args:
        json_path: TallyQA ``train.json`` (or jsonl variant).
        images_root: Directory that the row's ``image`` path is joined
            against. The official tarball lays out ``train2014/...``,
            ``val2014/...``, and ``VG_100K/...`` under one root.
        coco_eval_image_ids: image-ids reserved by POPE-adv eval — pass
            via ``load_pope_adv_image_ids(...)``. ``None`` opts out; only
            valid for smoke tests.
        n_max: cap on emitted samples (post-filter, post-shuffle). Default
            900 matches the E1 bucket-3 sub-budget.
        answer_max / answer_min: integer count bounds.
        require_image_field: defensive guard against records missing the
            ``image`` field (rare but exists in some TallyQA snapshots).
        shuffle: shuffle the eligible pool before capping (recommended for
            balanced category sampling).
    """
    json_path = Path(json_path)
    images_root = Path(images_root)
    rng = random.Random(seed)

    eval_ids = coco_eval_image_ids or set()

    raw_rows: list[dict] = []
    skipped = {"simple": 0, "eval_leak": 0, "answer_range": 0, "no_image": 0, "bad_answer": 0}

    for rec in _iter_raw_records(json_path):
        if complex_only and rec.get("issimple", False):
            skipped["simple"] += 1
            continue

        if require_image_field and not rec.get("image"):
            skipped["no_image"] += 1
            continue

        data_source = rec.get("data_source") or rec.get("source") or ""
        image_id = rec.get("image_id")
        if _is_coco_source(data_source) and image_id is not None and image_id in eval_ids:
            skipped["eval_leak"] += 1
            continue

        answer = rec.get("answer")
        # TallyQA answers are integers but sometimes stored as strings.
        try:
            answer_int = int(answer)
        except (TypeError, ValueError):
            skipped["bad_answer"] += 1
            continue
        if not (answer_min <= answer_int <= answer_max):
            skipped["answer_range"] += 1
            continue

        raw_rows.append({
            "question": rec.get("question", "").strip(),
            "answer": answer_int,
            "image": rec["image"],
            "image_id": image_id,
            "data_source": data_source,
            "question_id": rec.get("question_id"),
            "issimple": rec.get("issimple"),
        })

    logger.info(
        "[tallyqa] eligible after filters: %d (skipped: simple=%d eval_leak=%d answer_range=%d no_image=%d bad_answer=%d)",
        len(raw_rows), skipped["simple"], skipped["eval_leak"],
        skipped["answer_range"], skipped["no_image"], skipped["bad_answer"],
    )

    if shuffle:
        rng.shuffle(raw_rows)

    if n_max is not None:
        if len(raw_rows) < n_max:
            logger.warning(
                "[tallyqa] only %d eligible after filters; requested n_max=%d",
                len(raw_rows), n_max,
            )
        raw_rows = raw_rows[:n_max]

    n_emitted = 0
    for rec in raw_rows:
        # Question_id sometimes missing — fall back to a stable hash of image+question.
        qid = rec["question_id"]
        if qid is None:
            qid = f"{rec['image_id']}_{abs(hash(rec['question']))}"

        image_path = images_root / rec["image"]

        yield TallyQASample(
            sample_id=f"tallyqa_{qid}",
            question=rec["question"],
            image_paths=[image_path],
            gold=str(rec["answer"]),
            extras={
                "image_id": rec["image_id"],
                "data_source": rec["data_source"],
                "issimple": rec["issimple"],
                "answer_int": rec["answer"],
                "source": "tallyqa_complex",
            },
        )
        n_emitted += 1

    logger.info("[tallyqa] yielded %d samples", n_emitted)


# ---------------------------------------------------------------------------
# CLI smoke test.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from collections import Counter

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="TallyQA complex loader smoke test")
    p.add_argument(
        "--json-path", required=True,
        help="Path to TallyQA train.json (or train.jsonl)",
    )
    p.add_argument(
        "--images-root", required=True,
        help="Root directory holding train2014/ val2014/ VG_100K/ ...",
    )
    p.add_argument(
        "--pope-adv-root",
        default="/home/web_server/antispam/project/houshihao/datasets/POPE-adversarial",
        help="Path to POPE-adversarial save_to_disk dir for image-id disjointness check",
    )
    p.add_argument("--n-max", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-show", type=int, default=5)
    p.add_argument(
        "--no-image-load", action="store_true",
        help="Skip the load_images() probe (use when images are not yet on disk)",
    )
    args = p.parse_args()

    # Reuse the POPE-style helper for eval id extraction.
    from experiments.E1_filtered_delta_opd.data.pope_style_builder import load_pope_adv_image_ids
    pope_ids = load_pope_adv_image_ids(args.pope_adv_root)
    print(f"[pope-adv] loaded {len(pope_ids)} image ids from {args.pope_adv_root}")

    samples = list(iter_tallyqa(
        json_path=args.json_path,
        images_root=args.images_root,
        coco_eval_image_ids=pope_ids,
        n_max=args.n_max,
        seed=args.seed,
    ))
    print(f"[smoke] sampled {len(samples)}")

    src_dist = Counter(s.extras["data_source"] for s in samples)
    print(f"[smoke] data_source distribution: {dict(src_dist)}")
    ans_dist = Counter(s.extras["answer_int"] for s in samples)
    print(f"[smoke] answer distribution: {dict(sorted(ans_dist.items()))}")
    issimple_dist = Counter(s.extras["issimple"] for s in samples)
    print(f"[smoke] issimple distribution: {dict(issimple_dist)} (should be all False)")
    assert all(s.extras["issimple"] is False for s in samples), \
        "complex_only=True should not yield simple rows"

    # Disjointness assertion for COCO-sourced rows
    leaked = [
        s for s in samples
        if _is_coco_source(s.extras["data_source"])
        and s.extras["image_id"] in pope_ids
    ]
    assert not leaked, f"POPE-adv overlap detected on COCO TallyQA rows: {len(leaked)} leaks"
    print(f"[disjoint] OK — 0 COCO-sourced rows overlap with POPE-adv ({len(pope_ids)} eval ids)")

    print(f"\nfirst {args.n_show} samples:")
    for s in samples[:args.n_show]:
        print(f"--- {s.sample_id} ---")
        print(f"  question : {s.question!r}")
        print(f"  gold     : {s.gold!r}")
        print(f"  src/imgid: {s.extras['data_source']}/{s.extras['image_id']}")
        print(f"  image    : {s.image_paths[0]}")
        if not args.no_image_load:
            try:
                img = s.load_images()[0]
                print(f"  size     : {img.size}  mode={img.mode}")
            except FileNotFoundError as e:
                print(f"  LOAD ERR : {e}")
