"""
Build the E1 training parquet from precompute_teacher.py's jsonl output.

What this script does:

1. Reads one or more shard jsonls from ``precompute_teacher.py``. Those
   carry the sample-level signals we need (``trajectory_pass`` from the
   verifier, ``gold``, ``question``, ``bucket``) plus a bunch of fields
   we discard (per-token teacher logp / delta_t — those are stale because
   v1 recomputes them online on the *student's* rollout, not the
   teacher's).

2. Looks up each sample's image bytes from the source dataset (v1: only
   ViRL39K — bucket 2/3 loaders land in Day 2). Stored as a
   ``{"bytes": ...}`` dict so verl's ``RLHFDataset._build_messages``
   can decode it via PIL.

3. Formats the gold answer per the GPT-locked v1 template:

       gold_response_text = "Final answer: \\boxed{<answer>}."

   The "Final answer: " prefix is the **fixed suffix** GPT recommended
   (not a bare ``\\boxed{X}``) — keeps the response shape consistent
   with what the student actually outputs at eval time and avoids
   training a fake "answer-only no-rationale" style across the whole
   prompt. See on_policy_v1_design.md § 6 + the May-12 GPT review.

4. Tokenizes ``gold_response_text`` with the **student** tokenizer
   (Qwen2.5-VL-7B-Instruct). Even though the 32B teacher shares the
   tokenizer family, we lock to student tokenizer to avoid downstream
   ABI surprises if the families ever drift.

5. Emits a parquet with one row per sample, schema designed for verl's
   ``RLHFDataset`` + our ``DeltaOPDAgentLoop`` (which reads
   ``trajectory_pass``, ``gold_token_ids``, ``bucket`` from the row).

Run on the server (needs the ViRL39K images on disk and the student
tokenizer locally cached):

    python -m experiments.E1_filtered_delta_opd.data.make_train_parquet \\
        --jsonl /path/to/precompute_shard_*.jsonl \\
        --virl39k-root $DATASETS/ViRL39K \\
        --student-tokenizer $MODELS/Qwen2.5-VL-7B-Instruct \\
        --output /path/to/e1_train.parquet \\
        --limit 100   # smoke first; drop for full run

The output parquet is the input to
``scripts/run_e1_recipe_smoke.sh`` via ``E1_TRAIN_PARQUET``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator

# Inject repo root so cross-experiment imports (E1 ViRL39K loader) resolve
# when the script is run as ``python -m experiments.E1_filtered_delta_opd...``.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("E1_LOG_LEVEL", "INFO"), format="%(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Gold response formatting. The "Final answer: " prefix is the GPT-locked
# choice (May-12 review). If we ever want to sweep formats, change here +
# rebuild the parquet.
# ---------------------------------------------------------------------------

ANSWER_FORMAT_VERSION = "final_answer_boxed_v1"
GOLD_RESPONSE_TEMPLATE = "Final answer: \\boxed{{{answer}}}."


def format_gold_response(answer: str) -> str:
    return GOLD_RESPONSE_TEMPLATE.format(answer=answer.strip())


# ---------------------------------------------------------------------------
# Shard jsonl ingest.
# ---------------------------------------------------------------------------

def iter_precompute_records(jsonl_paths: Iterable[Path]) -> Iterator[dict]:
    """Yield precompute records, skipping per-shard error lines."""
    for path in jsonl_paths:
        n_total, n_error = 0, 0
        with open(path) as f:
            for line in f:
                n_total += 1
                rec = json.loads(line)
                if rec.get("error"):
                    n_error += 1
                    continue
                yield rec
        logger.info(
            "[ingest] %s — %d records (%d errors)", path.name, n_total, n_error
        )


# ---------------------------------------------------------------------------
# Image lookup: build sample_id → image_paths map from the ViRL39K loader.
# We re-walk the source parquet because the precompute jsonl doesn't carry
# image paths (only sample_id + the bucket name).
# ---------------------------------------------------------------------------

def build_virl39k_image_lookup(dataset_root: Path) -> dict[str, list[Path]]:
    """Map ViRL39K sample_id (qid) → list of absolute image paths.

    We disable the loader's pass-rate filter so every sample_id is indexable.
    The filter belongs to precompute (it already ran), not to this stage.
    """
    from experiments.E1_filtered_delta_opd.data.virl39k_loader import iter_virl39k

    lookup: dict[str, list[Path]] = {}
    for s in iter_virl39k(
        dataset_root=dataset_root,
        pass_rate_min=None,
        pass_rate_max=None,
        single_image_only=True,
        require_boxed=False,
    ):
        lookup[s.sample_id] = s.image_paths
    logger.info("[virl39k] image lookup built: %d entries", len(lookup))
    return lookup


def load_image_bytes(path: Path) -> bytes:
    return path.read_bytes()


# ---------------------------------------------------------------------------
# Row builder.
# ---------------------------------------------------------------------------

def build_row(
    rec: dict,
    *,
    index: int,
    image_bytes_per_sample: list[bytes],
    gold_token_ids: list[int],
    gold_response_text: str,
    tokenizer_name: str,
    chat_template_version: str,
) -> dict:
    """Compose one verl-shape RLHFDataset row for a precompute record.

    verl's ``_build_messages`` requires ``<image>`` placeholders in the
    message content matching the number of entries in ``images`` exactly
    (asserts ``image_offset == len(images)``). We prepend a single
    ``<image>\\n`` (v1 is single-image-only — multi-image filtered by
    ``virl39k_loader.single_image_only=True``).
    """
    bucket = rec["bucket"]
    question_clean = rec["question"]  # already <image>-stripped by virl39k_loader
    content = "<image>\n" + question_clean

    return {
        "data_source": f"e1_{bucket}",
        "prompt": [{"role": "user", "content": content}],
        "images": [{"bytes": b} for b in image_bytes_per_sample],
        "ability": "vl_distill",
        # ---- Delta-OPD per-sample fields read by the agent loop / loss ----
        "bucket": bucket,
        "trajectory_pass": bool(rec.get("trajectory_pass", rec.get("correct_I", False))),
        "gold_text": rec["gold"],
        "gold_response_text": gold_response_text,
        "gold_token_ids": gold_token_ids,
        "gold_token_len": len(gold_token_ids),
        # ---- Passthrough metadata (useful for per-category eval reports) ----
        "sample_id": rec["sample_id"],
        "category": rec.get("category") or "",
        "source": rec.get("source") or "",
        "pass_rate_32b": float(rec.get("pass_rate_32b") or 0.0),
        # ---- verl convention ----
        "reward_model": {"style": "rule", "ground_truth": rec["gold"]},
        "extra_info": {
            "index": index,
            "answer": rec["gold"],
            "question": question_clean,
            "tokenizer_name": tokenizer_name,
            "chat_template_version": chat_template_version,
            "answer_format": ANSWER_FORMAT_VERSION,
        },
    }


# ---------------------------------------------------------------------------
# Main pipeline.
# ---------------------------------------------------------------------------

def build_parquet(
    jsonl_paths: list[Path],
    virl39k_root: Path,
    student_tokenizer_path: str,
    output_parquet: Path,
    limit: int | None = None,
    chat_template_version: str = "qwen2_5_vl_default",
) -> None:
    # Heavy imports localized so the module imports on Mac for inspection.
    from transformers import AutoTokenizer
    import pyarrow as pa
    import pyarrow.parquet as pq

    tokenizer = AutoTokenizer.from_pretrained(student_tokenizer_path, trust_remote_code=True)
    logger.info(
        "[tokenizer] loaded %s vocab_size=%d", student_tokenizer_path, tokenizer.vocab_size
    )

    virl_lookup = build_virl39k_image_lookup(virl39k_root)

    rows: list[dict] = []
    counts = {"yielded": 0, "missing_image": 0, "unsupported_bucket": 0, "empty_gold": 0}

    for idx, rec in enumerate(iter_precompute_records(jsonl_paths)):
        if limit is not None and counts["yielded"] >= limit:
            break

        bucket = rec.get("bucket", "")
        if bucket == "virl39k":
            image_paths = virl_lookup.get(rec["sample_id"])
            if not image_paths:
                counts["missing_image"] += 1
                continue
            image_bytes = [load_image_bytes(p) for p in image_paths]
        else:
            # Bucket 2/3 builders land in Day 2 (POPE-style, synth, TallyQA).
            counts["unsupported_bucket"] += 1
            continue

        gold = (rec.get("gold") or "").strip()
        if not gold:
            counts["empty_gold"] += 1
            continue

        gold_response_text = format_gold_response(gold)
        gold_token_ids = tokenizer.encode(gold_response_text, add_special_tokens=False)

        row = build_row(
            rec,
            index=counts["yielded"],
            image_bytes_per_sample=image_bytes,
            gold_token_ids=gold_token_ids,
            gold_response_text=gold_response_text,
            tokenizer_name=student_tokenizer_path,
            chat_template_version=chat_template_version,
        )
        rows.append(row)
        counts["yielded"] += 1

        if counts["yielded"] % 500 == 0:
            logger.info("[build] %d rows queued", counts["yielded"])

    logger.info("[build] done — counts: %s", counts)

    if not rows:
        raise RuntimeError("No rows to write — check the jsonl path / virl39k root / limit.")

    table = pa.Table.from_pylist(rows)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_parquet, compression="zstd")
    logger.info("[write] %s rows → %s", len(rows), output_parquet)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build E1 training parquet from precompute jsonls")
    p.add_argument(
        "--jsonl", nargs="+", required=True,
        help="One or more precompute_teacher.py jsonl shards. Glob patterns supported by shell.",
    )
    p.add_argument(
        "--virl39k-root", required=True,
        help="Path to ViRL39K root (contains 39Krelease.parquet and images/)",
    )
    p.add_argument(
        "--student-tokenizer", required=True,
        help="Path or HF id for the student tokenizer (Qwen2.5-VL-7B-Instruct on the server)",
    )
    p.add_argument(
        "--output", required=True,
        help="Output parquet path (will be overwritten)",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap rows for smoke testing; omit for full run",
    )
    p.add_argument(
        "--chat-template-version", default="qwen2_5_vl_default",
        help="Free-form string baked into extra_info for future tokenizer-drift checks",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_parquet(
        jsonl_paths=[Path(p) for p in args.jsonl],
        virl39k_root=Path(args.virl39k_root),
        student_tokenizer_path=args.student_tokenizer,
        output_parquet=Path(args.output),
        limit=args.limit,
        chat_template_version=args.chat_template_version,
    )


if __name__ == "__main__":
    main()
