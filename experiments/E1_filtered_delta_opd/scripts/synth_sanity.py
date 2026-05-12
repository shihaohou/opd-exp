"""
Day 3 Step 2 gate: aggregate teacher behaviour on the synthetic bucket.

Input: a jsonl produced by

    python -m experiments.E1_filtered_delta_opd.src.precompute_teacher \\
        --bucket synthetic \\
        --loader-kwargs '{"out_dir": "$DATASETS/e1_synth_v1"}' \\
        --model-path $MODELS/Qwen2.5-VL-32B-Instruct \\
        --output $RESULTS/e1_synth_teacher_sanity.jsonl

Each record carries `correct_I`, `ans_T_I`, `gold`, and the synth `extras`
({topic, q_variant, ...}). This script computes per-topic teacher accuracy
and — for the `animal` topic — the canonical-prior trigger rate, then
emits a PASS/FAIL verdict against the gate defined in E1 README § "E1
Protocol" and NEXT.md § "Day 3 Step 2".

Gate logic (animal-only; the other topics are reported for context):
    - count "counterfactual" samples = `extras.is_counterfactual` True
      (animal silhouette has `n_legs != canonical_legs`).
    - accuracy_on_counterfactual = correct_I rate on those samples.
    - prior_trigger_rate = among teacher-wrong counterfactual samples,
      fraction where the teacher's integer answer equals
      `extras.canonical_legs` (instead of the actual `n_legs`).
    - PASS = `accuracy_on_counterfactual < 0.80` AND
             `prior_trigger_rate >= 0.30`
      → the silhouette IS triggering the prior; bucket-3 will train the
      recognition-failure mode.
    - FAIL = accuracy too high (silhouette doesn't fire the prior; the
      teacher just counts) OR prior_trigger_rate too low (wrong for
      reasons other than the prior). In either case, fix the synth
      pipeline before launching Step 3 / Step 4.

Outputs three files (markdown for GPT, JSON for scripts, CSV for plots):
    <out>.md   — verdict + per-topic table
    <out>.json — structured aggregate
    <out>.csv  — per-topic row table
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Answer parsing for synthetic samples (integers, sometimes inside \boxed{}).
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_INT_RE = re.compile(r"-?\d+")


def extract_int_answer(text: str) -> Optional[int]:
    """Extract the most plausible integer answer from the teacher's response.

    Strategy:
      1. If `\\boxed{...}` is present, parse the integer inside it.
      2. Else, take the LAST integer in the response (Qwen often emits
         "There are 5 legs visible." — last int = the answer).

    Returns None if no integer found.
    """
    if not text:
        return None
    m = _BOXED_RE.search(text)
    if m:
        inner = m.group(1).strip()
        try:
            return int(inner)
        except ValueError:
            ints = _INT_RE.findall(inner)
            if ints:
                try:
                    return int(ints[-1])
                except ValueError:
                    pass
    ints = _INT_RE.findall(text)
    if not ints:
        return None
    try:
        return int(ints[-1])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Per-topic aggregation.
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error"):
                continue
            out.append(rec)
    return out


def topic_of(rec: dict) -> str:
    e = rec.get("extras") or {}
    return e.get("topic") or "unknown"


def aggregate(records: list[dict]) -> dict[str, Any]:
    """Per-topic teacher accuracy + animal-specific prior-trigger metric."""
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_topic[topic_of(r)].append(r)

    per_topic: dict[str, dict[str, Any]] = {}
    for topic, recs in sorted(by_topic.items()):
        n = len(recs)
        n_correct = sum(1 for r in recs if r.get("correct_I"))

        entry: dict[str, Any] = {
            "n": n,
            "accuracy": n_correct / n if n else None,
        }

        # Animal-specific: prior-trigger rate
        if topic == "animal":
            n_cf = 0
            n_correct_cf = 0
            n_canon = 0
            n_correct_canon = 0
            n_teacher_wrong_cf = 0
            n_prior_trigger = 0

            for r in recs:
                e = r.get("extras") or {}
                canonical = e.get("canonical_legs")
                n_legs = e.get("n_legs")
                is_cf = bool(e.get("is_counterfactual"))

                if is_cf:
                    n_cf += 1
                    if r.get("correct_I"):
                        n_correct_cf += 1
                    else:
                        n_teacher_wrong_cf += 1
                        teacher_int = extract_int_answer(r.get("ans_T_I") or "")
                        if teacher_int is not None and canonical is not None and teacher_int == canonical:
                            n_prior_trigger += 1
                else:
                    n_canon += 1
                    if r.get("correct_I"):
                        n_correct_canon += 1

            entry.update({
                "n_counterfactual": n_cf,
                "accuracy_on_counterfactual": n_correct_cf / n_cf if n_cf else None,
                "n_canonical": n_canon,
                "accuracy_on_canonical": n_correct_canon / n_canon if n_canon else None,
                "n_teacher_wrong_on_counterfactual": n_teacher_wrong_cf,
                "prior_trigger_count": n_prior_trigger,
                "prior_trigger_rate": (
                    n_prior_trigger / n_teacher_wrong_cf if n_teacher_wrong_cf else None
                ),
            })

        # Chess-specific: canonical-position trigger (start position = 16 white + 16 black)
        if topic == "chess_position":
            n_can16_w = 0
            n_can16_b = 0
            n_canon_pair = 0  # full canonical (16, 16)
            for r in recs:
                e = r.get("extras") or {}
                nw = e.get("n_white"); nb = e.get("n_black")
                if nw == 16: n_can16_w += 1
                if nb == 16: n_can16_b += 1
                if nw == 16 and nb == 16: n_canon_pair += 1
            entry.update({
                "n_with_n_white_16": n_can16_w,
                "n_with_n_black_16": n_can16_b,
                "n_full_canonical": n_canon_pair,
            })

        per_topic[topic] = entry

    return {
        "n_total": len(records),
        "per_topic": per_topic,
    }


# ---------------------------------------------------------------------------
# Verdict.
# ---------------------------------------------------------------------------

def evaluate_gate(
    agg: dict,
    acc_cf_threshold: float = 0.80,
    prior_rate_threshold: float = 0.30,
) -> dict[str, Any]:
    """Apply the animal-bucket gate from NEXT.md § Day 3 Step 2.

    Returns {verdict: 'PASS'/'FAIL'/'INCONCLUSIVE', reason: str, details: dict}.
    """
    animal = agg["per_topic"].get("animal")
    if animal is None:
        return {
            "verdict": "INCONCLUSIVE",
            "reason": "No `animal` topic samples in input — re-run precompute with --bucket synthetic",
            "details": {},
        }

    acc_cf = animal.get("accuracy_on_counterfactual")
    prior_rate = animal.get("prior_trigger_rate")
    if acc_cf is None:
        return {
            "verdict": "INCONCLUSIVE",
            "reason": "No counterfactual animal samples — synth pipeline produced 100% canonical",
            "details": animal,
        }

    too_easy = acc_cf >= acc_cf_threshold
    weak_prior = (prior_rate is None) or (prior_rate < prior_rate_threshold)

    if too_easy:
        return {
            "verdict": "FAIL",
            "reason": (
                f"Teacher accuracy on counterfactual animal silhouettes = {acc_cf:.3f}, "
                f"≥ threshold {acc_cf_threshold}. The teacher is correctly counting the "
                f"actual legs, which means the silhouette is NOT triggering the canonical "
                f"prior. Bucket-3 won't train the recognition failure mode. Fix the synth "
                f"pipeline (e.g., CLIP-guided animal generation) before training."
            ),
            "details": animal,
        }
    if weak_prior:
        return {
            "verdict": "FAIL",
            "reason": (
                f"Teacher acc on counterfactual = {acc_cf:.3f} (below threshold, good), but "
                f"prior_trigger_rate = {prior_rate} (< {prior_rate_threshold}). Teacher errors "
                f"are not from the canonical prior — they're from generic miscount or "
                f"degenerate output. Bucket-3 doesn't test inheritance of the canonical-prior "
                f"failure. Fix the synth pipeline."
            ),
            "details": animal,
        }
    return {
        "verdict": "PASS",
        "reason": (
            f"acc_on_counterfactual = {acc_cf:.3f} (< {acc_cf_threshold}) AND "
            f"prior_trigger_rate = {prior_rate:.3f} (≥ {prior_rate_threshold}): canonical "
            f"prior is being triggered as designed. Bucket-3 is ready for training."
        ),
        "details": animal,
    }


# ---------------------------------------------------------------------------
# Output writers.
# ---------------------------------------------------------------------------

def write_markdown(agg: dict, gate: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Synthetic-bucket teacher sanity (Day 3 Step 2)", ""]
    lines.append(f"**Verdict**: **{gate['verdict']}**")
    lines.append("")
    lines.append(f"_{gate['reason']}_")
    lines.append("")
    lines.append(f"Total synth samples scored: **{agg['n_total']}**")
    lines.append("")
    lines.append("## Per-topic teacher accuracy")
    lines.append("")
    lines.append("| topic | n | teacher accuracy |")
    lines.append("|---|---:|---:|")
    for topic, e in agg["per_topic"].items():
        acc = e["accuracy"]
        acc_s = f"{acc:.3f}" if acc is not None else "—"
        lines.append(f"| `{topic}` | {e['n']} | {acc_s} |")
    lines.append("")

    animal = agg["per_topic"].get("animal")
    if animal:
        lines.append("## Animal topic — canonical-prior trigger detail")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for k in ("n_counterfactual", "accuracy_on_counterfactual",
                  "n_canonical", "accuracy_on_canonical",
                  "n_teacher_wrong_on_counterfactual",
                  "prior_trigger_count", "prior_trigger_rate"):
            v = animal.get(k)
            v_s = f"{v:.3f}" if isinstance(v, float) else str(v)
            lines.append(f"| `{k}` | {v_s} |")
        lines.append("")

    chess = agg["per_topic"].get("chess_position")
    if chess:
        lines.append("## Chess topic — canonical-position counts (informational)")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for k in ("n_with_n_white_16", "n_with_n_black_16", "n_full_canonical"):
            v = chess.get(k)
            lines.append(f"| `{k}` | {v} |")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_json(agg: dict, gate: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"aggregate": agg, "gate": gate}, f, indent=2, ensure_ascii=False)


def write_csv(agg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Union of keys across topics so the header covers animal-specific cols too.
    cols: list[str] = ["topic"]
    seen: set[str] = set()
    for e in agg["per_topic"].values():
        for k in e:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for topic, e in agg["per_topic"].items():
            w.writerow({"topic": topic, **e})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Day-3 Step-2 gate: teacher sanity on the synthetic bucket",
    )
    p.add_argument("jsonl", help="precompute_teacher.py output for --bucket synthetic")
    p.add_argument(
        "--out", required=True,
        help="Output prefix; writes <out>.md, <out>.json, <out>.csv",
    )
    p.add_argument(
        "--acc-cf-threshold", type=float, default=0.80,
        help="Animal accuracy_on_counterfactual must be < this to PASS (default 0.80)",
    )
    p.add_argument(
        "--prior-rate-threshold", type=float, default=0.30,
        help="Animal prior_trigger_rate must be >= this to PASS (default 0.30)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"ERROR: input jsonl not found: {jsonl_path}", file=sys.stderr)
        return 2

    records = load_jsonl(jsonl_path)
    print(f"[sanity] loaded {len(records)} records from {jsonl_path}")

    agg = aggregate(records)
    gate = evaluate_gate(
        agg,
        acc_cf_threshold=args.acc_cf_threshold,
        prior_rate_threshold=args.prior_rate_threshold,
    )

    out_prefix = Path(args.out)
    write_markdown(agg, gate, out_prefix.with_suffix(".md"))
    write_json(agg, gate, out_prefix.with_suffix(".json"))
    write_csv(agg, out_prefix.with_suffix(".csv"))

    print(f"[sanity] verdict: {gate['verdict']}")
    print(f"[sanity] reason : {gate['reason']}")
    print(f"[sanity] wrote:")
    print(f"  {out_prefix.with_suffix('.md')}")
    print(f"  {out_prefix.with_suffix('.json')}")
    print(f"  {out_prefix.with_suffix('.csv')}")

    return 0 if gate["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
