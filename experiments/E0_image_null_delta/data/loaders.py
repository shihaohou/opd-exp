"""
Dataset loaders for E0.

All three datasets are stored on the remote box in HF arrow format (output of
Dataset.save_to_disk / DatasetDict.save_to_disk). Each loader returns a list of
Sample objects with a unified shape so dual_forward.py doesn't need to
special-case dataset structure (only answer-parsing and option scoring care).

Footguns this module handles for you:
  * MathVista: PIL image is in `decoded_image`, not `image`.
  * VLMBias:   `image` is direct PIL; `expected_bias` is the biased wrong answer
               needed by metric 3.
  * POPE-adv:  Already a single split — no need to filter on `category`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset, load_from_disk
from PIL import Image


@dataclass
class Sample:
    dataset: str            # canonical dataset key, e.g. "vlmbias_main"
    sample_id: str          # stable string ID from the source row
    question: str           # raw prompt/question text to feed the model
    image: Image.Image      # PIL image (already decoded)
    gold: str               # gold answer as a string
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# VLMBias
# ---------------------------------------------------------------------------

def load_vlmbias(
    path: str,
    subset: str = "main",
    n_samples: int | None = None,
) -> list[Sample]:
    """
    Load a subset of VLMBias from `path` (a DatasetDict.save_to_disk directory).

    Columns used: ID, image, prompt, ground_truth, expected_bias, topic,
    sub_topic, type_of_question.
    """
    ds = load_from_disk(path)
    if subset not in ds:
        raise KeyError(f"VLMBias subset '{subset}' not in {list(ds.keys())}")
    rows: Dataset = ds[subset]
    if n_samples is not None:
        rows = rows.select(range(min(n_samples, len(rows))))

    samples: list[Sample] = []
    for row in rows:
        samples.append(Sample(
            dataset=f"vlmbias_{subset}",
            sample_id=str(row["ID"]),
            question=row["prompt"],
            image=row["image"],
            gold=row["ground_truth"],
            extras={
                "expected_bias": row["expected_bias"],
                "topic": row["topic"],
                "sub_topic": row["sub_topic"],
                "type_of_question": row["type_of_question"],
                "pixel": row.get("pixel"),
            },
        ))
    return samples


# ---------------------------------------------------------------------------
# POPE-adversarial
# ---------------------------------------------------------------------------

def load_pope(
    path: str,
    n_samples: int | None = 1000,
) -> list[Sample]:
    """
    Load the POPE adversarial split from `path` (single Dataset.save_to_disk dir).

    Columns used: id, question_id, question, answer, image_source, image, category.
    """
    rows: Dataset = load_from_disk(path)
    if n_samples is not None:
        rows = rows.select(range(min(n_samples, len(rows))))

    samples: list[Sample] = []
    for row in rows:
        samples.append(Sample(
            dataset="pope_adversarial",
            sample_id=str(row.get("id") or row["question_id"]),
            question=row["question"],
            image=row["image"],
            gold=row["answer"].strip().lower(),  # "yes" / "no"
            extras={
                "question_id": row["question_id"],
                "image_source": row["image_source"],
                "category": row.get("category", "adversarial"),
            },
        ))
    return samples


# ---------------------------------------------------------------------------
# MathVista-mini
# ---------------------------------------------------------------------------

def load_mathvista(
    path: str,
    n_samples: int | None = 500,
) -> list[Sample]:
    """
    Load MathVista testmini from `path`.

    Columns used: pid, question, decoded_image, choices, answer, question_type,
    answer_type, unit, metadata.
    """
    rows: Dataset = load_from_disk(path)
    if n_samples is not None:
        rows = rows.select(range(min(n_samples, len(rows))))

    samples: list[Sample] = []
    for row in rows:
        choices = row.get("choices") or []
        # Pull metadata.task / metadata.category if available
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            # Some saved datasets keep metadata as a JSON string
            import json
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}

        samples.append(Sample(
            dataset="mathvista_mini",
            sample_id=str(row["pid"]),
            question=row["question"],
            image=row["decoded_image"],   # NOT row["image"] (that is a filename)
            gold=str(row["answer"]),
            extras={
                "choices": list(choices),
                "question_type": row["question_type"],
                "answer_type": row.get("answer_type"),
                "unit": row.get("unit"),
                "task": meta.get("task"),
                "category": meta.get("category"),
                "source": meta.get("source"),
            },
        ))
    return samples


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def load_dataset(
    name: str,
    cfg: dict,
) -> list[Sample]:
    """
    Load a dataset by its canonical key from the config dict.

    `cfg` is the full parsed e0_default.yaml; we read `cfg["paths"]` and
    `cfg["datasets"][name]` here so the caller doesn't have to.
    """
    if name not in cfg["datasets"]:
        raise KeyError(f"Dataset '{name}' not in config. Available: {list(cfg['datasets'])}")

    ds_cfg = cfg["datasets"][name]
    path = os.path.join(cfg["paths"]["datasets_root"], ds_cfg["path_relative"])
    n = ds_cfg.get("n_samples")
    loader = ds_cfg["loader"]

    if loader == "vlmbias":
        return load_vlmbias(path, subset=ds_cfg.get("subset", "main"), n_samples=n)
    if loader == "pope":
        return load_pope(path, n_samples=n)
    if loader == "mathvista":
        return load_mathvista(path, n_samples=n)
    raise ValueError(f"Unknown loader '{loader}' for dataset '{name}'")
