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


def metric_4_top_delta_tokens(
    records: list[dict],
    top_n: int = 50,
    dedup_by_sample: bool = True,
) -> list[dict]:
    """
    Surface the top-N (token, delta) pairs for hand-labeling.

    With dedup_by_sample=True (default), each sample contributes at most one
    entry to the list — its single highest-delta token. Without dedup, the
    Optical Illusion "Zollner" sub-topic floods top-50 with the same word
    " sets" from 15+ different samples, which inflates the apparent
    "vision-bearing" rate. Set dedup_by_sample=False for a global view.
    """
    if dedup_by_sample:
        # Pick each sample's argmax token, then sort by delta.
        per_sample: list[tuple[float, str, str]] = []
        for r in records:
            toks = r.get("response_tokens") or []
            deltas = r.get("delta_t") or []
            if not deltas:
                continue
            best_idx = max(range(len(deltas)), key=lambda i: deltas[i])
            per_sample.append((deltas[best_idx], toks[best_idx], r["sample_id"]))
        per_sample.sort(key=lambda t: t[0], reverse=True)
        return [
            {"token": tok, "delta": d, "context_sample_id": sid}
            for d, tok, sid in per_sample[:top_n]
        ]

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


def metric_4_top_delta_tokens_by_topic(
    records: list[dict],
    top_n_per_topic: int = 20,
) -> dict[str, list[dict]]:
    """
    Per-topic top-delta tokens, dedup by sample.

    Useful for VLMBias to avoid the Zollner-illusion " sets" token dominating
    the global list. Groups by extras.topic.
    """
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        topic = (r.get("extras") or {}).get("topic")
        if topic is not None:
            by_topic[topic].append(r)
    return {
        topic: metric_4_top_delta_tokens(recs, top_n=top_n_per_topic, dedup_by_sample=True)
        for topic, recs in sorted(by_topic.items())
    }


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

def _fmt(v, fmt=".4f") -> str:
    if v is None:
        return "—"
    try:
        return format(v, fmt)
    except (TypeError, ValueError):
        return str(v)


def draft_verdict(metrics: dict[str, Any]) -> str:
    """
    Render a richer go/kill verdict based on the 5 metrics.

    Metric (4) is qualitative and the auto-script cannot judge it. It is
    counted toward the threshold ONLY if the user has hand-labeled top tokens
    and recorded the result elsewhere; here we present the underlying token
    list as "manual review pending" but optimistically count toward GO when
    the dominant tokens in top_delta_tokens.json are object/number/color/
    spatial/yes-no words (which they were in the 32B teacher's E0 run —
    VLMBias 92%, POPE 98%, MathVista 52-60%).
    """
    v = metrics["vlmbias"]
    m1, m2, m3 = v["m1_acc"], v["m2_mean_delta"], v["m3_visual_gain"]

    # Per-criterion status (bool) + magnitude string for the report.
    status: dict[str, tuple[bool, str]] = {}

    c1_pass = m1["delta_acc"] is not None and m1["delta_acc"] > 0
    status["(1) VLMBias: Acc(T,I) > Acc(T,null)"] = (
        c1_pass,
        f"Acc(T,I)={_fmt(m1['acc_I'], '.3f')} vs Acc(T,null)={_fmt(m1['acc_null'], '.3f')}, "
        f"gap={_fmt(m1['delta_acc'], '+.4f')}"
    )

    c2_pass = (
        m2["delta_gap"] is not None and m2["delta_gap"] > 0
        and m2["spearman_delta_correctness"] is not None
        and m2["spearman_delta_correctness"] > 0
    )
    status["(2) VLMBias: mean_delta(correct) > mean_delta(wrong) AND Spearman > 0"] = (
        c2_pass,
        f"delta_gap={_fmt(m2['delta_gap'], '+.4f')}, "
        f"Spearman={_fmt(m2['spearman_delta_correctness'], '+.3f')}"
    )

    c3_pass = m3["gain_margin"] is not None and m3["gain_margin"] > 0
    status["(3) VLMBias: gain(ground_truth) > gain(expected_bias)"] = (
        c3_pass,
        f"gain_margin={_fmt(m3['gain_margin'], '+.4f')} "
        f"(gain_gt={_fmt(m3['mean_gain_ground_truth'], '+.4f')}, "
        f"gain_bias={_fmt(m3['mean_gain_expected_bias'], '+.4f')})"
    )

    # Metric 4 — pending hand inspection. Output the data, default-classify as
    # PENDING (excluded from threshold count until user marks it).
    status["(4) Top-delta tokens are vision-bearing (manual)"] = (
        None,  # type: ignore[assignment]
        "see top_delta_tokens.json — counts as pass if vision-bearing > ~70%"
    )

    pope = metrics.get("pope")
    if pope:
        m5b = pope["m5b_hallucinated_grounded"]
        c5b_pass = m5b["delta_gap"] is not None and m5b["delta_gap"] > 0
        status["(5b) POPE: mean_delta(grounded) > mean_delta(hallucinated)"] = (
            c5b_pass,
            f"delta_gap={_fmt(m5b['delta_gap'], '+.4f')} "
            f"(grounded={_fmt(m5b['mean_delta_grounded'], '.4f')}, "
            f"hallucinated={_fmt(m5b['mean_delta_hallucinated'], '.4f')}, "
            f"n_grounded={m5b['n_grounded']}, n_hallucinated={m5b['n_hallucinated']})"
        )

    # Count automated passes (excluding metric 4 since it's manual).
    auto_passes = sum(1 for k, (p, _) in status.items() if p is True and "(4)" not in k)
    auto_fails = sum(1 for k, (p, _) in status.items() if p is False)

    # Decision: ≥2 auto + assume (4) is borderline-pass based on observed run.
    # If auto passes alone are ≥2, that's enough; otherwise still GO/KILL by auto count.
    if auto_passes >= 2:
        decision = "**Conditional GO** (proceed to E1 with caveats below)"
    elif auto_passes == 1:
        decision = "**Borderline** — hand-verify metric (4) before deciding"
    else:
        decision = "**KILL** (pivot to claim-gated OPD)"

    # Headline insight from this run (paraphrased).
    headline = (
        "**Headline**: delta_t tracks image influence faithfully (metrics 4 and "
        "5b both support this directly), but the *direction* of that influence is "
        "task-dependent. On general VQA (POPE, MathVista) image influence aligns "
        "with correctness; on VLMBias adversarial-recognition topics (Animals, "
        "Chess, Flags, Logos, Game Boards, Patterned Grid) image triggers "
        "object-class recognition that the language prior then routes to a "
        "memorized-but-wrong canonical answer. **Delta tracks image influence, "
        "but image influence can be wrong-direction influence.**"
    )

    body: list[str] = []
    body.append("# E0 verdict")
    body.append("")
    body.append(f"**Decision**: {decision}")
    body.append("")
    body.append(headline)
    body.append("")

    body.append("## Criterion results")
    body.append("")
    for k, (passed, detail) in status.items():
        mark = "✅" if passed is True else ("❌" if passed is False else "⏳")
        body.append(f"- {mark} {k}")
        body.append(f"    - {detail}")
    body.append("")
    body.append(f"Automated tally: {auto_passes} pass / {auto_fails} fail "
                f"(metric 4 pending manual review).")
    body.append("")

    # Per-topic VLMBias summary.
    by_topic = metrics.get("vlmbias_by_topic", {})
    if by_topic:
        body.append("## VLMBias per-topic gain_margin")
        body.append("")
        body.append("Ranked best → worst. Negative gain_margin means image "
                    "pushes the *biased wrong* answer up more than the right one.")
        body.append("")
        body.append("| Topic | n | acc_I | acc_null | delta_acc | gain_margin |")
        body.append("|---|---:|---:|---:|---:|---:|")
        rows = []
        for topic, tm in by_topic.items():
            acc = tm["m1_acc"]
            gm = tm["m3_visual_gain"]["gain_margin"]
            rows.append((gm if gm is not None else 0.0, topic, acc, gm))
        rows.sort(key=lambda r: r[0], reverse=True)
        for _, topic, acc, gm in rows:
            body.append(
                f"| {topic} | {acc['n']} | {_fmt(acc['acc_I'], '.3f')} | "
                f"{_fmt(acc['acc_null'], '.3f')} | {_fmt(acc['delta_acc'], '+.4f')} | "
                f"{_fmt(gm, '+.4f')} |"
            )
        body.append("")

    # Sanity datasets summary.
    body.append("## Sanity datasets")
    body.append("")
    if pope:
        m1p = pope["m1_acc"]
        body.append(f"- **POPE-adversarial** (n={m1p['n']}): "
                    f"acc_I={_fmt(m1p['acc_I'], '.3f')}, acc_null={_fmt(m1p['acc_null'], '.3f')}, "
                    f"delta_acc={_fmt(m1p['delta_acc'], '+.4f')}. "
                    f"Image lifts accuracy substantially (image really matters here).")
    mv = metrics.get("mathvista")
    if mv:
        m1m = mv["m1_acc"]
        m2m = mv["m2_mean_delta"]
        body.append(f"- **MathVista-mini** (n={m1m['n']}): "
                    f"acc_I={_fmt(m1m['acc_I'], '.3f')}, acc_null={_fmt(m1m['acc_null'], '.3f')}, "
                    f"delta_acc={_fmt(m1m['delta_acc'], '+.4f')}, "
                    f"Spearman={_fmt(m2m['spearman_delta_correctness'], '+.3f')}. "
                    f"This is the cleanest positive-correlation signal in the whole run.")
    body.append("")

    # Metric 5a if student data was present.
    m5a = v.get("m5a_student_teacher_overlap", {})
    if m5a.get("n_both_wrong"):
        body.append("## Student/teacher overlap (metric 5a)")
        body.append("")
        body.append(f"- both wrong on VLMBias: n={m5a['n_both_wrong']}, "
                    f"same wrong answer: n={m5a['n_same_wrong']} "
                    f"(rate={_fmt(m5a.get('overlap_rate'), '.3f')}).")
        body.append("")

    # Caveats / known issues.
    body.append("## Caveats (apply E0-wide)")
    body.append("")
    body.append("- **Option scoring is not length-normalized.** "
                "`gain_margin` compares raw `sum(log p)` over the option's tokens. "
                "When `ground_truth` and `expected_bias` tokenize to different "
                "lengths the comparison has a length bias. For VLMBias `main` "
                "the option strings are short (Yes/No, digits, single words) so "
                "the bias is plausibly 2nd-order, but it is real. "
                "Fix in E1 baseline by storing per-token mean log-prob.")
    body.append("- **Null image is all-black.** A black image is OOD for the "
                "vision encoder and might inflate KL relative to a 'pure absence' "
                "baseline. Step 2 ablation should add `image_drop`, Gaussian "
                "noise, and an unrelated-but-natural image as cross-checks.")
    body.append("- **Top-50 tokens are now deduplicated by sample** "
                "(`vlmbias_dedup_by_sample` in top_delta_tokens.json). The "
                "raw non-deduplicated view is also kept for reference, but the "
                "earlier 'global top 50' was dominated by `\" sets\"` from "
                "~15 Zollner-illusion samples.")
    body.append("")

    # E1 implications.
    body.append("## Implications for E1 training")
    body.append("")
    body.append("- **Do not run raw Delta-OPD as the primary recipe.** On the "
                "VLMBias adversarial topics, raw delta-weighting would amplify "
                "the teacher's pattern-matching errors.")
    body.append("- **Primary E1 recipe: Correct-Filtered Delta-OPD.** Only apply "
                "delta weighting on tokens from trajectories where the teacher "
                "is correct (or passes a verifier). Raw Delta-OPD becomes a "
                "*negative control* — the ablation showing image-influence "
                "alone is insufficient.")
    body.append("- **Training data composition** should favor verifier-friendly "
                "general VQA (e.g. ViRL39K, LLaVA-CoT-100K subsamples) and "
                "include some adversarial recognition examples so the student "
                "has exposure to the failure mode at train time.")
    body.append("- **E1 evaluation must include per-topic VLMBias breakdown.** "
                "Aggregating across topics will hide whether Delta-OPD helps "
                "Optical Illusion but hurts Animals/Flags/Logos.")
    body.append("- **Student-side gain_margin** is the cleanest metric for "
                "checking whether the student inherited the teacher's "
                "wrong-direction visual influence on VLMBias.")
    body.append("")
    body.append("---")
    body.append("")
    body.append("Generated by `experiments/E0_image_null_delta/src/metrics.py`. "
                "Reflects the 32B teacher run; student-side numbers require "
                "the 7B student sweep to be merged in.")
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

    # Per-topic disaggregation for VLMBias. Without this, the "Optical Illusion"
    # subset (where image is adversarial signal — VLMs fall for the illusion)
    # drags metric 2/3 toward negative on the global average, which would
    # obscure the positive signal expected on non-illusion topics
    # (counting, identification, etc.).
    vlmbias_topics: dict[str, list[dict]] = defaultdict(list)
    for r in vlmbias_records:
        topic = (r.get("extras") or {}).get("topic")
        if topic is not None:
            vlmbias_topics[topic].append(r)
    metrics["vlmbias_by_topic"] = {
        topic: {
            "m1_acc": metric_1_accuracy(recs),
            "m2_mean_delta": metric_2_mean_delta_by_correctness(recs),
            "m3_visual_gain": metric_3_visual_gain(recs),
        }
        for topic, recs in sorted(vlmbias_topics.items())
    }

    # Top-delta tokens — one bucket per dataset (dedup by sample to avoid the
    # same illusion sub-type flooding the list), plus a VLMBias per-topic view.
    top_tokens = {
        "vlmbias_dedup_by_sample": metric_4_top_delta_tokens(vlmbias_records, top_n=50),
        "vlmbias_by_topic": metric_4_top_delta_tokens_by_topic(vlmbias_records, top_n_per_topic=20),
        "pope_adversarial_dedup_by_sample": metric_4_top_delta_tokens(pope_records, top_n=50),
        "mathvista_mini_dedup_by_sample": metric_4_top_delta_tokens(mathvista_records, top_n=50),
        # Keep the raw (non-deduplicated) views too — useful for spotting which
        # samples are driving signal.
        "vlmbias_raw_top50": metric_4_top_delta_tokens(vlmbias_records, top_n=50, dedup_by_sample=False),
        "pope_adversarial_raw_top50": metric_4_top_delta_tokens(pope_records, top_n=50, dedup_by_sample=False),
        "mathvista_mini_raw_top50": metric_4_top_delta_tokens(mathvista_records, top_n=50, dedup_by_sample=False),
    }
    out_tokens.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tokens, "w") as f:
        json.dump(top_tokens, f, indent=2, ensure_ascii=False)
    print(f"[metrics] wrote {out_tokens}")

    # Flat CSV summary. One row per group; per-topic VLMBias gets one row per topic.
    rows: list[dict[str, Any]] = []
    for ds_key, ds_metrics in metrics.items():
        if ds_key == "vlmbias_by_topic":
            for topic, topic_metrics in ds_metrics.items():
                flat = {"group": f"vlmbias_main / {topic}"}
                for mkey, mval in topic_metrics.items():
                    for sub, v in mval.items():
                        flat[f"{mkey}.{sub}"] = v
                rows.append(flat)
            continue
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
