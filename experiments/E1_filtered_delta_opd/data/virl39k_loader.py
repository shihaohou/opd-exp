"""
ViRL39K loader for E1 training data.

HF dataset: TIGER-Lab/ViRL39K (38,870 verifiable multimodal QA pairs).
On-disk layout (after the raw-repo download to /datasets/ViRL39K):
    39Krelease.parquet   # all rows
    images.zip           # 1.7G zipped; unzip → images/<source>-<qid>-<idx>.{jpg,png}
    README.md
    .gitattributes

Parquet schema (verified 2026-05-12):
    question              str   — includes "<image>\\n" placeholder(s)
    answer                str   — answers are wrapped in `\\boxed{...}`
    PassRate_32BTrained   float — pre-measured pass rate of VL-Rethinker's
                                  trained 32B model. 0.0 = always wrong;
                                  1.0 = always right. ALL rows have a value
                                  (no -1 sentinels in this snapshot).
    PassRate_7BBase       float — same for 7B base
    category              str   — 8 categories (GradeSchool Math, Geometric,
                                  Tables/Charts, Spatial Reasoning, etc.)
    source                str   — origin dataset name (Processed, MMK12,
                                  MMMath, M3CoT, dvqa, ai2d, ...)
    qid                   str   — unique sample id
    image                 list[str] — relative paths inside images/.
                                      94.1% are single-image; rest is 2-8.

`PassRate_32BTrained` is treated as a cheap pre-filter for E1: rows where it
is 1.0 give Filtered Delta-OPD no advantage over Vanilla OPD (teacher always
passes → filtering is a no-op), and rows near 0.0 are unlikely to give our
own 32B teacher useful trajectories either. The default `pass_rate ∈ [0.3, 0.9]`
window keeps the rows where filtering actually does work (~13.4K / 38.9K).

Footguns this loader handles:
  * `<image>` placeholders in the question text: Qwen2.5-VL uses `<|image_pad|>`
    internally and wants images delivered via the chat-template image slot,
    NOT the literal `<image>` token. Strip them before passing to the
    processor.
  * `image` is `list[str]`, not `str`. v1 of the loader filters to
    `len(image)==1` for simplicity; multi-image support is deferred.
  * `\boxed{...}` answers may contain nested braces (e.g. `\boxed{\\frac{1}{2}}`).
    The simple greedy regex fails on those; we count and report skipped rows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from PIL import Image

# `\boxed{<content>}` with depth-1 brace tracking — handles a single layer of
# nested braces inside the box (e.g. `\boxed{\frac{1}{2}}` and `\boxed{A}` both
# parse). For deeper nesting the regex still fails; those rows are skipped and
# counted.
_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_IMAGE_PLACEHOLDER_RE = re.compile(r"<image>\s*")


@dataclass
class ViRL39KSample:
    """One ViRL39K row, image-decoding deferred.

    Use `load_images()` to materialize PIL images at the point of forward.
    Holding paths instead of decoded tensors keeps the per-sample footprint
    small enough to materialize the full filtered subset in memory.
    """

    sample_id: str
    question: str             # `<image>` placeholders stripped
    image_paths: list[Path]   # absolute paths
    gold: str                 # extracted from `\boxed{...}`
    extras: dict[str, Any] = field(default_factory=dict)

    def load_images(self) -> list[Image.Image]:
        return [Image.open(p).convert("RGB") for p in self.image_paths]


def parse_boxed(answer: str) -> Optional[str]:
    """Extract content from `\\boxed{...}`. Returns None if absent or unparseable."""
    m = _BOXED_RE.search(answer)
    return m.group(1).strip() if m else None


def strip_image_placeholders(question: str) -> str:
    """Remove `<image>` tokens (and trailing whitespace) from the question text."""
    return _IMAGE_PLACEHOLDER_RE.sub("", question).strip()


def iter_virl39k(
    dataset_root: Path | str,
    pass_rate_min: Optional[float] = 0.3,
    pass_rate_max: Optional[float] = 0.9,
    single_image_only: bool = True,
    require_boxed: bool = True,
    parquet_name: str = "39Krelease.parquet",
) -> Iterator[ViRL39KSample]:
    """Stream ViRL39K samples with optional pre-filtering.

    Default filter regime is appropriate for E1 training:
        pass_rate_min=0.3, pass_rate_max=0.9, single_image_only=True,
        require_boxed=True.

    Set `pass_rate_min=None` and `pass_rate_max=None` to disable the pass-rate
    filter entirely.

    Image paths in the parquet are relative (`images/<filename>`), resolved
    against `dataset_root`. After unzipping `images.zip` in place, these point
    to the unpacked files.
    """
    import pyarrow.parquet as pq

    dataset_root = Path(dataset_root)
    parquet_path = dataset_root / parquet_name

    t = pq.read_table(parquet_path)
    # Materialize columns once; ~38K rows fits trivially.
    cols = {name: t.column(name).to_pylist() for name in t.column_names}
    n_rows = t.num_rows

    n_filtered = {"image_count": 0, "pass_rate": 0, "no_boxed": 0}
    n_yielded = 0

    for i in range(n_rows):
        images = cols["image"][i] or []
        if single_image_only and len(images) != 1:
            n_filtered["image_count"] += 1
            continue

        pr32 = cols["PassRate_32BTrained"][i]
        if pass_rate_min is not None and pr32 < pass_rate_min:
            n_filtered["pass_rate"] += 1
            continue
        if pass_rate_max is not None and pr32 > pass_rate_max:
            n_filtered["pass_rate"] += 1
            continue

        gold = parse_boxed(cols["answer"][i])
        if require_boxed and gold is None:
            n_filtered["no_boxed"] += 1
            continue

        yield ViRL39KSample(
            sample_id=cols["qid"][i],
            question=strip_image_placeholders(cols["question"][i]),
            image_paths=[dataset_root / p for p in images],
            gold=gold or "",
            extras={
                "pass_rate_32b": pr32,
                "pass_rate_7b": cols["PassRate_7BBase"][i],
                "category": cols["category"][i],
                "source": cols["source"][i],
                "raw_answer": cols["answer"][i],
            },
        )
        n_yielded += 1

    # One-shot summary on the last iteration. Caller can also inspect by
    # wrapping in `list(...)` and re-counting; this is informational.
    print(
        f"[virl39k] yielded {n_yielded}/{n_rows}; "
        f"filtered: image_count={n_filtered['image_count']}, "
        f"pass_rate={n_filtered['pass_rate']}, no_boxed={n_filtered['no_boxed']}"
    )


# ---------------------------------------------------------------------------
# CLI smoke test — run on the server after `unzip` finishes to verify the
# loader can actually open images.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from collections import Counter

    p = argparse.ArgumentParser(description="ViRL39K loader smoke test")
    p.add_argument(
        "--dataset-root",
        default="/home/web_server/antispam/project/houshihao/datasets/ViRL39K",
        help="Path to ViRL39K root (containing 39Krelease.parquet and images/)",
    )
    p.add_argument("--n-show", type=int, default=5, help="Print this many samples")
    p.add_argument(
        "--no-pass-rate-filter", action="store_true",
        help="Disable PassRate_32BTrained filtering (count everything)",
    )
    args = p.parse_args()

    kwargs: dict[str, Any] = {"dataset_root": args.dataset_root}
    if args.no_pass_rate_filter:
        kwargs["pass_rate_min"] = None
        kwargs["pass_rate_max"] = None

    samples = list(iter_virl39k(**kwargs))
    print(f"[virl39k] loaded {len(samples)} samples")

    cat_counts = Counter(s.extras["category"] for s in samples)
    print("\ncategory distribution (post-filter):")
    for cat, n in cat_counts.most_common():
        print(f"  {n:>5}  {cat}")

    src_counts = Counter(s.extras["source"] for s in samples)
    print(f"\nsource distribution (top 10 of {len(src_counts)}):")
    for src, n in src_counts.most_common(10):
        print(f"  {n:>5}  {src}")

    print(f"\nfirst {args.n_show} samples:")
    for s in samples[: args.n_show]:
        print(f"--- {s.sample_id} ---")
        print(f"  category   : {s.extras['category']}")
        print(f"  source     : {s.extras['source']}")
        print(f"  pass_rate  : 32B={s.extras['pass_rate_32b']:.3f} 7B={s.extras['pass_rate_7b']:.3f}")
        print(f"  gold       : {s.gold!r}")
        print(f"  raw_answer : {s.extras['raw_answer']!r}")
        print(f"  question[:200]: {s.question[:200]!r}")
        print(f"  image_paths: {s.image_paths}")
        # Try to actually open the image — fails loudly if unzip not done.
        try:
            imgs = s.load_images()
            print(f"  image size : {imgs[0].size} mode={imgs[0].mode}")
        except FileNotFoundError as e:
            print(f"  image LOAD FAILED: {e}")
