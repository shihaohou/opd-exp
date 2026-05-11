"""
E0 metrics aggregator.

CPU-only. Reads the jsonl shards written by dual_forward.py and produces:

  results/e0_summary.csv   — one row per (model, dataset), columns = key metrics
  results/e0_verdict.md    — draft one-paragraph go/kill recommendation

Five metrics:
  1   Acc(T, image) vs Acc(T, null)                          [VLMBias]
  2   mean_delta(correct) vs mean_delta(wrong) + Spearman    [VLMBias]
  3   gain(ground_truth) vs gain(expected_bias)              [VLMBias, needs option_logP]
  4   top-K delta tokens                                     [all datasets — dumped as list]
  5a  student-teacher wrong-overlap                          [VLMBias, needs student jsonl]
  5b  mean_delta(hallucinated) vs mean_delta(grounded)       [POPE-adv, model answers yes]

This module is intentionally independent of torch / transformers so it can run
on the Mac for inspection without touching the venv on the server.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path) -> Iterable[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_records(paths: list[Path]) -> list[dict]:
    """Concatenate shard jsonls; skip lines that recorded an exception."""
    records = []
    for p in paths:
        for r in iter_jsonl(p):
            if "error" in r:
                continue
            records.append(r)
    return records


# ---------------------------------------------------------------------------
# Stat helpers
# ---------------------------------------------------------------------------

def safe_mean(xs: list[float]) -> Optional[float]:
    return statistics.mean(xs) if xs else None


def spearman(xs: list[float], ys: list[float]) -> Optional[float]:
    """Spearman rank correlation. Returns None if degenerate."""
    if len(xs) < 3 or len(ys) < 3 or len(xs) != len(ys):
        return None

    def ranks(vs: list[float]) -> list[float]:
        # Average rank for ties.
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        ranks = [0.0] * len(vs)
        i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1  # 1-indexed rank
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = ranks(xs)
    ry = ranks(ys)
    mx = statistics.mean(rx)
    my = statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    denom = (dx * dy) ** 0.5
    if denom == 0:
        return None
    return num / denom


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metric_1_accuracy(records: list[dict]) -> dict[str, Any]:
    """Acc(T, image) vs Acc(T, null) on records (e.g. VLMBias subset)."""
    n = len(records)
    if n == 0:
        return {"n": 0, "acc_I": None, "acc_null": None, "delta_acc": None}
    acc_I = sum(1 for r in records if r["correct_I"]) / n
    acc_null = sum(1 for r in records if r["correct_null"]) / n
    return {
        "n": n,
        "acc_I": acc_I,
        "acc_null": acc_null,
        "delta_acc": acc_I - acc_null,
    }


def per_record_mean_delta(r: dict) -> Optional[float]:
    deltas = r.get("delta_t") or []
    if not deltas:
        return None
    return float(statistics.mean(deltas))


def metric_2_mean_delta_by_correctness(records: list[dict]) -> dict[str, Any]:
    """mean_delta split by correct_I, plus Spearman(mean_delta, correct_I)."""
    correct_deltas: list[float] = []
    wrong_deltas: list[float] = []
    paired_x: list[float] = []
    paired_y: list[float] = []
    for r in records:
        md = per_record_mean_delta(r)
        if md is None:
            continue
        paired_x.append(md)
        paired_y.append(1.0 if r["correct_I"] else 0.0)
        if r["correct_I"]:
            correct_deltas.append(md)
        else:
            wrong_deltas.append(md)
    return {
        "n_with_delta": len(paired_x),
        "n_correct": len(correct_deltas),
        "n_wrong": len(wrong_deltas),
        "mean_delta_correct": safe_mean(correct_deltas),
        "mean_delta_wrong": safe_mean(wrong_deltas),
        "delta_gap": (
            safe_mean(correct_deltas) - safe_mean(wrong_deltas)
            if correct_deltas and wrong_deltas else None
        ),
        "spearman_delta_correctness": spearman(paired_x, paired_y),
    }


def metric_3_visual_gain(records: list[dict]) -> dict[str, Any]:
    """
    For records that carry option_logP (VLMBias):
        gain(label) = option_logP['I'][label] - option_logP['null'][label]
    Compare mean gain on ground_truth vs expected_bias.
    """
    gains_gt: list[float] = []
    gains_bias: list[float] = []
    for r in records:
        opt = r.get("option_logP")
        if not opt:
            continue
        I = opt.get("I", {})
        N = opt.get("null", {})
        if "ground_truth" in I and "ground_truth" in N:
            gains_gt.append(I["ground_truth"] - N["ground_truth"])
        if "expected_bias" in I and "expected_bias" in N:
            gains_bias.append(I["expected_bias"] - N["expected_bias"])
    return {
        "n_gt": len(gains_gt),
        "n_bias": len(gains_bias),
        "mean_gain_ground_truth": safe_mean(gains_gt),
        "mean_gain_expected_bias": safe_mean(gains_bias),
        "gain_margin": (
            safe_mean(gains_gt) - safe_mean(gains_bias)
            if gains_gt and gains_bias else None
        ),
    }


def metric_4_top_delta_tokens(records: list[dict], top_n: int = 50) -> list[dict]:
    """
    Across all records in this group, surface the top-N (token, delta) pairs
    for hand-labeling. Returns a list of {"token", "delta", "context_sample_id"}.
    """
    pool: list[tuple[float, str, str]] = []
    for r in records:
        toks = r.get("response_tokens") or []
        deltas = r.get("delta_t") or []
        for tok, d in zip(toks, deltas):
            pool.append((d, tok, r["sample_id"]))
    pool.sort(key=lambda t: t[0], reverse=True)
    return [
        {"token": tok, "delta": d, "context_sample_id": sid}
        for d, tok, sid in pool[:top_n]
    ]


def metric_5a_student_teacher_overlap(
    teacher_records: list[dict],
    student_records: list[dict],
) -> dict[str, Any]:
    """
    Among VLMBias samples where the teacher is wrong AND the student is wrong,
    what fraction predicted the SAME answer string?

    We pair on sample_id (assumes student and teacher are both run on the same
    samples; mismatched sample sets are silently dropped).
    """
    by_id_s = {r["sample_id"]: r for r in student_records}
    n_both_wrong = 0
    n_same_wrong = 0
    for tr in teacher_records:
        sr = by_id_s.get(tr["sample_id"])
        if sr is None:
            continue
        if tr["correct_I"] or sr["correct_I"]:
            continue
        n_both_wrong += 1
        if tr["ans_T_I"].strip().lower() == sr["ans_T_I"].strip().lower():
            n_same_wrong += 1
    return {
        "n_both_wrong": n_both_wrong,
        "n_same_wrong": n_same_wrong,
        "overlap_rate": (n_same_wrong / n_both_wrong) if n_both_wrong else None,
    }


def metric_5b_hallucinated_vs_grounded_delta(records: list[dict]) -> dict[str, Any]:
    """
    On POPE-adv where the model answered 'yes':
      grounded     = gold == 'yes' (object actually present)
      hallucinated = gold == 'no'  (object absent — false positive)
    """
    grounded: list[float] = []
    hallucinated: list[float] = []
    for r in records:
        md = per_record_mean_delta(r)
        if md is None:
            continue
        # Did the model say yes?
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
    return {
        "n_grounded": len(grounded),
        "n_hallucinated": len(hallucinated),
        "mean_delta_grounded": safe_mean(grounded),
        "mean_delta_hallucinated": safe_mean(hallucinated),
        "delta_gap": (
            safe_mean(grounded) - safe_mean(hallucinated)
            if grounded and hallucinated else None
        ),
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def draft_verdict(metrics: dict[str, Any]) -> str:
    """Render a short go/kill recommendation based on the 5 metrics."""
    passes = []
    fails = []

    v = metrics["vlmbias"]
    m1 = v["m1_acc"]
    if m1["delta_acc"] is not None and m1["delta_acc"] > 0.0:
        passes.append(f"(1) Acc(T,I)={m1['acc_I']:.3f} > Acc(T,null)={m1['acc_null']:.3f}")
    else:
        fails.append(f"(1) Acc(T,I)={m1['acc_I']} vs Acc(T,null)={m1['acc_null']} — no margin")

    m2 = v["m2_mean_delta"]
    if (m2["delta_gap"] is not None and m2["delta_gap"] > 0
            and m2["spearman_delta_correctness"] is not None
            and m2["spearman_delta_correctness"] > 0):
        passes.append(
            f"(2) mean_delta(correct)-mean_delta(wrong)={m2['delta_gap']:.4f}, "
            f"Spearman={m2['spearman_delta_correctness']:.3f}"
        )
    else:
        fails.append(f"(2) delta gap={m2['delta_gap']}, Spearman={m2['spearman_delta_correctness']}")

    m3 = v["m3_visual_gain"]
    if m3["gain_margin"] is not None and m3["gain_margin"] > 0:
        passes.append(f"(3) gain margin (ground_truth vs expected_bias) = {m3['gain_margin']:.3f}")
    else:
        fails.append(f"(3) gain margin = {m3['gain_margin']}")

    fails.append("(4) Token-category analysis requires hand-labeling — see top_delta_tokens.json")

    if metrics.get("pope"):
        m5b = metrics["pope"]["m5b_hallucinated_grounded"]
        if m5b["delta_gap"] is not None and m5b["delta_gap"] > 0:
            passes.append(
                f"(5b) POPE mean_delta(grounded) - mean_delta(hallucinated) = {m5b['delta_gap']:.4f}"
            )
        else:
            fails.append(f"(5b) POPE delta gap = {m5b['delta_gap']}")

    threshold = 2
    decision = "GO (proceed to E1)" if len(passes) >= threshold else "KILL (pivot to claim-gated OPD)"

    body = ["# E0 verdict (draft)", "", f"**Decision**: {decision}", ""]
    body.append("**Criteria passed**:")
    for p in passes:
        body.append(f"- {p}")
    body.append("")
    body.append("**Criteria failed / pending**:")
    for f in fails:
        body.append(f"- {f}")
    body.append("")
    body.append("Threshold for GO: ≥ 2 of the 5 primary criteria. "
                "Hand-inspect top_delta_tokens.json before finalizing the verdict — "
                "metric (4) cannot be automated.")
    return "\n".join(body)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-dir",
        default="experiments/E0_image_null_delta/results",
        help="Directory containing dual_forward shard jsonls",
    )
    p.add_argument(
        "--teacher-glob",
        default="e0_teacher32b_*.jsonl",
        help="Glob for teacher shards (relative to --results-dir)",
    )
    p.add_argument(
        "--student-glob",
        default="e0_student7b_*.jsonl",
        help="Glob for student shards (relative to --results-dir)",
    )
    p.add_argument(
        "--out-csv",
        default=None,
        help="Where to write the summary CSV. Default: <results-dir>/e0_summary.csv",
    )
    p.add_argument(
        "--out-verdict",
        default=None,
        help="Where to write the verdict markdown. Default: <results-dir>/e0_verdict.md",
    )
    p.add_argument(
        "--out-tokens",
        default=None,
        help="Where to write top-delta token JSON. Default: <results-dir>/top_delta_tokens.json",
    )
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_csv = Path(args.out_csv) if args.out_csv else results_dir / "e0_summary.csv"
    out_verdict = Path(args.out_verdict) if args.out_verdict else results_dir / "e0_verdict.md"
    out_tokens = Path(args.out_tokens) if args.out_tokens else results_dir / "top_delta_tokens.json"

    teacher_paths = sorted(results_dir.glob(args.teacher_glob))
    student_paths = sorted(results_dir.glob(args.student_glob))
    print(f"[metrics] teacher shards: {len(teacher_paths)}; student shards: {len(student_paths)}")

    teacher_records = load_records(teacher_paths)
    student_records = load_records(student_paths)
    print(f"[metrics] teacher records: {len(teacher_records)}; student records: {len(student_records)}")

    # Group by dataset.
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in teacher_records:
        by_dataset[r["dataset"]].append(r)

    vlmbias_records = [r for ds, recs in by_dataset.items() if ds.startswith("vlmbias") for r in recs]
    pope_records = by_dataset.get("pope_adversarial", [])
    mathvista_records = by_dataset.get("mathvista_mini", [])

    student_vlmbias = [r for r in student_records if r["dataset"].startswith("vlmbias")]

    metrics: dict[str, Any] = {}
    metrics["vlmbias"] = {
        "m1_acc": metric_1_accuracy(vlmbias_records),
        "m2_mean_delta": metric_2_mean_delta_by_correctness(vlmbias_records),
        "m3_visual_gain": metric_3_visual_gain(vlmbias_records),
        "m5a_student_teacher_overlap": metric_5a_student_teacher_overlap(
            vlmbias_records, student_vlmbias
        ),
    }
    if pope_records:
        metrics["pope"] = {
            "m1_acc": metric_1_accuracy(pope_records),
            "m2_mean_delta": metric_2_mean_delta_by_correctness(pope_records),
            "m5b_hallucinated_grounded": metric_5b_hallucinated_vs_grounded_delta(pope_records),
        }
    if mathvista_records:
        metrics["mathvista"] = {
            "m1_acc": metric_1_accuracy(mathvista_records),
            "m2_mean_delta": metric_2_mean_delta_by_correctness(mathvista_records),
        }

    # Top-delta tokens — one bucket per dataset, dumped to JSON for hand-inspection.
    top_tokens = {
        "vlmbias": metric_4_top_delta_tokens(vlmbias_records),
        "pope_adversarial": metric_4_top_delta_tokens(pope_records),
        "mathvista_mini": metric_4_top_delta_tokens(mathvista_records),
    }
    out_tokens.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tokens, "w") as f:
        json.dump(top_tokens, f, indent=2, ensure_ascii=False)
    print(f"[metrics] wrote {out_tokens}")

    # Flat CSV summary.
    rows: list[dict[str, Any]] = []
    for ds_key, ds_metrics in metrics.items():
        flat = {"group": ds_key}
        for mkey, mval in ds_metrics.items():
            for sub, v in mval.items():
                flat[f"{mkey}.{sub}"] = v
        rows.append(flat)

    all_cols = sorted({k for row in rows for k in row})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[metrics] wrote {out_csv}")

    with open(out_verdict, "w") as f:
        f.write(draft_verdict(metrics))
    print(f"[metrics] wrote {out_verdict}")


if __name__ == "__main__":
    main()
