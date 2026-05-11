"""
E0 report figures.

Reads the jsonl shards produced by dual_forward.py and renders the three
figures referenced by `CLAUDE.md` and the verdict:

  Fig 1  VLMBias per-topic gain_margin bar chart
         (highlights that only Optical Illusion is positive; all
         recognition-style topics push image toward the biased wrong answer)
  Fig 2  VLMBias mean_delta distribution split by correctness
         (overlapping density / histogram)
  Fig 3  POPE-adversarial mean_delta distribution split by hallucinated vs
         grounded (when the teacher answered "yes")

Runs CPU-only on a Mac with the jsonl files present at results/. Requires
matplotlib and numpy; pandas is optional. No GPU, no model load.

Usage (from repo root):
    python -m experiments.E0_image_null_delta.analysis.e0_report

Outputs three PNGs into `results/figures/`.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(
        "matplotlib is required. Install it in the Mac venv:\n"
        "  pip install matplotlib numpy\n"
        f"Original error: {e}"
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path) -> Iterable[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_records(paths: list[Path]) -> list[dict]:
    records = []
    for p in paths:
        for r in iter_jsonl(p):
            if "error" in r:
                continue
            records.append(r)
    return records


def per_record_mean_delta(r: dict) -> float | None:
    deltas = r.get("delta_t") or []
    if not deltas:
        return None
    return float(statistics.mean(deltas))


# ---------------------------------------------------------------------------
# Figure 1 — per-topic gain margin
# ---------------------------------------------------------------------------

def fig1_per_topic_gain_margin(vlmbias_records: list[dict], out_path: Path) -> None:
    """Bar chart of per-topic mean(gain_gt - gain_bias), sorted ascending."""
    by_topic: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in vlmbias_records:
        topic = (r.get("extras") or {}).get("topic")
        opt = r.get("option_logP") or {}
        I = opt.get("I", {})
        N = opt.get("null", {})
        if topic is None or "ground_truth" not in I or "expected_bias" not in I:
            continue
        gain_gt = I["ground_truth"] - N["ground_truth"]
        gain_bias = I["expected_bias"] - N["expected_bias"]
        by_topic[topic].append((gain_gt, gain_bias))

    rows = []
    for topic, pairs in by_topic.items():
        gts = [g for g, _ in pairs]
        biases = [b for _, b in pairs]
        margin = statistics.mean(gts) - statistics.mean(biases)
        rows.append((topic, len(pairs), margin, statistics.mean(gts), statistics.mean(biases)))
    rows.sort(key=lambda r: r[2])  # ascending — worst first

    topics = [r[0] for r in rows]
    margins = [r[2] for r in rows]
    ns = [r[1] for r in rows]
    colors = ["#d62728" if m < 0 else "#2ca02c" for m in margins]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(topics))
    ax.barh(y, margins, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y, labels=[f"{t} (n={n})" for t, n in zip(topics, ns)])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(r"$\overline{\mathrm{gain}}(\mathrm{gt}) - \overline{\mathrm{gain}}(\mathrm{bias})$"
                  "  (positive = image favours the right answer)")
    ax.set_title("VLMBias per-topic image-vs-null gain margin (32B teacher)\n"
                 "Negative = image pushes the biased wrong answer up more than the right answer")
    ax.grid(axis="x", linewidth=0.3, alpha=0.5)
    for yi, m in zip(y, margins):
        ax.text(m + (0.1 if m >= 0 else -0.1), yi, f"{m:+.2f}",
                va="center", ha="left" if m >= 0 else "right", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[fig1] {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — VLMBias mean_delta split by correctness
# ---------------------------------------------------------------------------

def fig2_vlmbias_mean_delta_by_correctness(records: list[dict], out_path: Path) -> None:
    correct, wrong = [], []
    for r in records:
        md = per_record_mean_delta(r)
        if md is None:
            continue
        (correct if r.get("correct_I") else wrong).append(md)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, max(max(correct or [1]), max(wrong or [1])), 60)
    ax.hist(correct, bins=bins, alpha=0.55, color="#2ca02c",
            label=f"correct (n={len(correct)}, mean={np.mean(correct):.3f})",
            density=True)
    ax.hist(wrong, bins=bins, alpha=0.55, color="#d62728",
            label=f"wrong (n={len(wrong)}, mean={np.mean(wrong):.3f})",
            density=True)
    ax.axvline(float(np.mean(correct)), color="#2ca02c", linestyle="--", linewidth=1.2)
    ax.axvline(float(np.mean(wrong)), color="#d62728", linestyle="--", linewidth=1.2)
    ax.set_xlabel("per-trajectory mean delta_t  (KL_top50 averaged over response tokens)")
    ax.set_ylabel("density")
    ax.set_title("VLMBias: mean delta_t distribution, split by image-conditioned correctness\n"
                 "Wrong samples have HIGHER mean delta — image is wrong-direction on adversarial topics")
    ax.legend()
    ax.grid(linewidth=0.3, alpha=0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[fig2] {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — POPE mean_delta: grounded vs hallucinated
# ---------------------------------------------------------------------------

def fig3_pope_grounded_vs_hallucinated(records: list[dict], out_path: Path) -> None:
    grounded, hallucinated = [], []
    for r in records:
        md = per_record_mean_delta(r)
        if md is None:
            continue
        rt = (r.get("ans_T_I") or "").strip().lower()
        first_yes = rt.find("yes")
        first_no = rt.find("no")
        if first_yes == -1:
            continue
        if first_no != -1 and first_no < first_yes:
            continue
        gold = (r.get("gold") or "").strip().lower()
        if gold == "yes":
            grounded.append(md)
        elif gold == "no":
            hallucinated.append(md)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    pool = (grounded or []) + (hallucinated or [])
    if not pool:
        ax.text(0.5, 0.5, "No POPE 'yes' responses found", ha="center", va="center")
    else:
        bins = np.linspace(0, max(pool), 50)
        ax.hist(grounded, bins=bins, alpha=0.55, color="#1f77b4",
                label=f"grounded yes (gold=yes, n={len(grounded)}, mean={np.mean(grounded):.3f})",
                density=True)
        ax.hist(hallucinated, bins=bins, alpha=0.55, color="#ff7f0e",
                label=f"hallucinated yes (gold=no, n={len(hallucinated)}, mean={np.mean(hallucinated):.3f})",
                density=True)
        if grounded:
            ax.axvline(float(np.mean(grounded)), color="#1f77b4", linestyle="--", linewidth=1.2)
        if hallucinated:
            ax.axvline(float(np.mean(hallucinated)), color="#ff7f0e", linestyle="--", linewidth=1.2)
    ax.set_xlabel("per-trajectory mean delta_t")
    ax.set_ylabel("density")
    ax.set_title("POPE-adversarial: mean delta_t when teacher answered 'yes'\n"
                 "Grounded > hallucinated supports the central hypothesis (metric 5b)")
    ax.legend()
    ax.grid(linewidth=0.3, alpha=0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[fig3] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="experiments/E0_image_null_delta/results")
    p.add_argument("--teacher-glob", default="e0_teacher32b_*.jsonl")
    p.add_argument("--out-dir", default=None, help="Default: <results-dir>/figures")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "figures"

    paths = sorted(results_dir.glob(args.teacher_glob))
    print(f"[e0_report] loading {len(paths)} shard jsonls from {results_dir}")
    records = load_records(paths)
    print(f"[e0_report] {len(records)} records")

    vlmbias = [r for r in records if r["dataset"].startswith("vlmbias")]
    pope = [r for r in records if r["dataset"] == "pope_adversarial"]
    print(f"[e0_report] vlmbias={len(vlmbias)}  pope={len(pope)}")

    fig1_per_topic_gain_margin(vlmbias, out_dir / "fig1_vlmbias_per_topic_gain_margin.png")
    fig2_vlmbias_mean_delta_by_correctness(vlmbias, out_dir / "fig2_vlmbias_mean_delta_by_correctness.png")
    fig3_pope_grounded_vs_hallucinated(pope, out_dir / "fig3_pope_grounded_vs_hallucinated.png")


if __name__ == "__main__":
    main()
