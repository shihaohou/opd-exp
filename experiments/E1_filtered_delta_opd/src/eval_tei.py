"""
E1 evaluation: TEI / Escape / VLMBias per-topic / POPE / MathVista.

E1 Protocol § Primary metrics + Safety metrics → this is the script that
materializes them on a trained student checkpoint. Required output (per
the protocol):

  PRIMARY
    1. VLMBias Recognition Aggregate accuracy
    2. TEI rate     = P(S_after = T_wrong_answer | T_wrong)        ↓ better
    3. Escape rate  = P(S_after = GT | T_wrong AND S_base = T_wrong) ↑ better
    4. VLMBias per-topic student-side length-normalized gain_margin

  SAFETY
    POPE-adv: accuracy, F1, yes-rate, hallucinated-yes rate
    MathVista-mini: accuracy, response length p50/p95, parse success rate

Architecture (two layers, intentionally decoupled):

  METRIC LAYER (pure CPU, no torch).
    Consumes pre-generated jsonls in the E0 schema (sample_id, gold,
    ans_T_I, correct_I, extras, option_logP). Mac-testable with the
    existing E0 32B teacher + 7B student jsonls under
    `experiments/E0_image_null_delta/results/`.

  INFERENCE LAYER (server, GPU).
    Loads a student HF checkpoint (after `verl.model_merger merge
    --backend fsdp ...` consolidation), runs greedy_generate +
    forced_score + score_option on the eval datasets, writes jsonls in
    the same shape the metric layer consumes. Reuses E0
    `src/dual_forward.py` helpers verbatim (no need to re-implement).

Validation built into `python eval_tei.py test --e0-results-dir <dir>`:
when student_after = student_base (= the un-distilled E0 7B run), TEI
rate must match the E0-reported same-wrong / T-wrong ratio
(1353 / 2179 ≈ 0.621). If it doesn't, the metric layer is broken.

Day-3 ordering: this file (Step 1) → Bucket-3 teacher sanity (Step 2)
→ 1K mini-sweep + run this script on each checkpoint (Step 3) → 8K full
sweep (Step 4). Do NOT run the 8K precompute before this script is
green; see NEXT.md § "Right now, Day 3".
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

# Cross-experiment imports (E0 helpers).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — VLMBias topic taxonomy (locked by E0 dataset).
# ---------------------------------------------------------------------------

# Exact spelling from VLMBias `main` subset extras.topic. The 7 topics
# split into one "illusion" group + 6 "recognition" groups; per the
# E1 Protocol, we report the Recognition Aggregate separately from
# Optical Illusion because they have opposite-sign signals in E0.

VLMBIAS_TOPICS_ALL = (
    "Optical Illusion",
    "Animals",
    "Chess Pieces",
    "Flags",
    "Game Boards",
    "Logos",
    "Patterned Grid",
)

VLMBIAS_RECOGNITION_TOPICS = (
    "Animals",
    "Chess Pieces",
    "Flags",
    "Game Boards",
    "Logos",
    "Patterned Grid",
)

# ---------------------------------------------------------------------------
# Small utilities.
# ---------------------------------------------------------------------------

def safe_mean(xs: list[float]) -> Optional[float]:
    """Mean, ignoring None and non-finite (NaN/inf) values.

    Non-finite scores leak in from `score_option` when option text tokenizes
    to a degenerate sequence (very rare; saw it on a handful of Logos rows
    in E0 student jsonls). Drop them rather than propagating NaN through
    every topic table.
    """
    import math
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return float(sum(xs) / len(xs)) if xs else None


def percentile(xs: list[float], p: float) -> Optional[float]:
    """Linear-interpolated percentile. `p` in [0, 100]."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return float(xs[lo] * (1 - frac) + xs[hi] * frac)


def load_records(paths: Iterable[Path | str]) -> list[dict]:
    """Load records from one or more jsonl shards; skip rows with `error`."""
    out: list[dict] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"jsonl not found: {p}")
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("error"):
                    continue
                out.append(rec)
    return out


def shard_glob(jsonl_dir: Path | str, prefix: str) -> list[Path]:
    """Expand `<dir>/<prefix>.shard*.jsonl` (E0's naming convention)."""
    jsonl_dir = Path(jsonl_dir)
    matches = sorted(jsonl_dir.glob(f"{prefix}.shard*.jsonl"))
    if not matches:
        # Fallback: maybe a single non-sharded file
        single = jsonl_dir / f"{prefix}.jsonl"
        if single.exists():
            return [single]
    return matches


# ---------------------------------------------------------------------------
# Answer extraction — forked from E0 metrics.py with the same logic so the
# metric layer is self-contained (no implicit dependency on E0 src layout).
# ---------------------------------------------------------------------------

_BOXED_PAYLOAD_RE = re.compile(r"\{([^}]+)\}")
_TOKEN_RE = re.compile(r"[\w\-]+")


def extract_answer(record: dict) -> str:
    """
    Best-effort short-form answer extraction from a record's `ans_T_I`.

    Behaviour:
      * vlmbias_*       → first {…} payload (e.g. "{No}" → "no"); fall back
                          to the last word-like token.
      * pope_adversarial → earlier of "yes" / "no" in the lowered string.
      * mathvista_mini  → lowered stripped string (caller does loose matching).
      * otherwise       → lowered stripped string.
    """
    text = (record.get("ans_T_I") or "").strip()
    if not text:
        return ""
    ds = record.get("dataset") or ""
    rt_low = text.lower()

    if ds.startswith("vlmbias"):
        m = _BOXED_PAYLOAD_RE.search(text)
        if m:
            return m.group(1).strip().lower()
        toks = _TOKEN_RE.findall(text)
        return toks[-1].lower() if toks else rt_low

    if ds == "pope_adversarial":
        first_yes = rt_low.find("yes")
        first_no = rt_low.find("no")
        if first_yes == -1 and first_no == -1:
            return rt_low
        if first_yes == -1:
            return "no"
        if first_no == -1:
            return "yes"
        return "yes" if first_yes < first_no else "no"

    return rt_low


def _gold_canonical(record: dict) -> str:
    """Lowercased / stripped gold answer."""
    return (record.get("gold") or "").strip().lower()


# ---------------------------------------------------------------------------
# Optional: tokenizer-backed option-length fn for length-normalized gain.
# Matches E0 metrics.py:make_option_len_fn (boundary trick).
# ---------------------------------------------------------------------------

def make_option_len_fn(tokenizer_path: str) -> Optional[Callable[[str], int]]:
    """Boundary-trick `option_text -> n_tokens` for length-norm gain_margin.

    Returns None if transformers / the tokenizer fail to load — the caller
    should then skip the length-normalized branch (raw gain_margin is
    still reported either way).
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.warning("[eval_tei] transformers not installed; skipping length-norm gain")
        return None
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        logger.warning("[eval_tei] tokenizer load failed (%s); skipping length-norm gain", e)
        return None

    newline_len = len(tok.encode("\n", add_special_tokens=False))

    from functools import lru_cache

    @lru_cache(maxsize=4096)
    def n_tokens(option_text: str) -> int:
        return len(tok.encode("\n" + option_text, add_special_tokens=False)) - newline_len

    logger.info("[eval_tei] loaded tokenizer %s for length-norm gain", tokenizer_path)
    return n_tokens


# ===========================================================================
# METRIC LAYER — pure CPU, no torch dependency.
# ===========================================================================

# ---------------------------------------------------------------------------
# VLMBias: per-topic accuracy + Recognition Aggregate + gain_margin.
# ---------------------------------------------------------------------------

def _compute_gain_margin(
    records: list[dict],
    option_len_fn: Optional[Callable[[str], int]] = None,
) -> dict[str, Any]:
    """Compute raw + optional length-normalized gain_margin from option_logP.

    Forked from E0 metrics.metric_3_visual_gain so the metric layer has
    no implicit dependency on the E0 src tree. Behaviour is identical.
    """
    gains_gt: list[float] = []
    gains_bias: list[float] = []
    gains_gt_per_tok: list[float] = []
    gains_bias_per_tok: list[float] = []
    n_skipped_lengthnorm = 0

    for r in records:
        opt = r.get("option_logP")
        if not opt:
            continue
        I = opt.get("I", {})
        N = opt.get("null", {})
        extras = r.get("extras") or {}

        if "ground_truth" in I and "ground_truth" in N:
            g = I["ground_truth"] - N["ground_truth"]
            gains_gt.append(g)
            if option_len_fn is not None:
                gt_text = r.get("gold") or extras.get("ground_truth")
                if gt_text:
                    L = option_len_fn(gt_text)
                    if L > 0:
                        gains_gt_per_tok.append(g / L)
                    else:
                        n_skipped_lengthnorm += 1
                else:
                    n_skipped_lengthnorm += 1

        if "expected_bias" in I and "expected_bias" in N:
            g = I["expected_bias"] - N["expected_bias"]
            gains_bias.append(g)
            if option_len_fn is not None:
                bias_text = extras.get("expected_bias")
                if bias_text:
                    L = option_len_fn(bias_text)
                    if L > 0:
                        gains_bias_per_tok.append(g / L)
                    else:
                        n_skipped_lengthnorm += 1
                else:
                    n_skipped_lengthnorm += 1

    out: dict[str, Any] = {
        "n_gt": len(gains_gt),
        "n_bias": len(gains_bias),
        "mean_gain_ground_truth": safe_mean(gains_gt),
        "mean_gain_expected_bias": safe_mean(gains_bias),
        "gain_margin": (
            safe_mean(gains_gt) - safe_mean(gains_bias)
            if gains_gt and gains_bias else None
        ),
    }
    if option_len_fn is not None:
        out.update({
            "mean_gain_per_tok_ground_truth": safe_mean(gains_gt_per_tok),
            "mean_gain_per_tok_expected_bias": safe_mean(gains_bias_per_tok),
            "gain_margin_lengthnorm": (
                safe_mean(gains_gt_per_tok) - safe_mean(gains_bias_per_tok)
                if gains_gt_per_tok and gains_bias_per_tok else None
            ),
            "n_skipped_lengthnorm": n_skipped_lengthnorm,
        })
    return out


def compute_vlmbias_metrics(
    records: list[dict],
    option_len_fn: Optional[Callable[[str], int]] = None,
) -> dict[str, Any]:
    """Per-topic accuracy, Recognition Aggregate, per-topic + aggregate gain_margin."""
    n_total = len(records)
    n_correct = sum(1 for r in records if r.get("correct_I"))

    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        topic = (r.get("extras") or {}).get("topic")
        if topic is not None:
            by_topic[topic].append(r)

    per_topic: dict[str, dict[str, Any]] = {}
    for topic, recs in sorted(by_topic.items()):
        n = len(recs)
        n_corr = sum(1 for r in recs if r.get("correct_I"))
        gain = _compute_gain_margin(recs, option_len_fn=option_len_fn)
        per_topic[topic] = {
            "n": n,
            "accuracy": n_corr / n if n else None,
            **gain,
        }

    # Recognition Aggregate = the 6 non-illusion topics, pooled.
    recog_records = [
        r for r in records
        if (r.get("extras") or {}).get("topic") in VLMBIAS_RECOGNITION_TOPICS
    ]
    n_recog = len(recog_records)
    n_recog_corr = sum(1 for r in recog_records if r.get("correct_I"))
    recog_gain = _compute_gain_margin(recog_records, option_len_fn=option_len_fn)

    return {
        "n_samples": n_total,
        "global_accuracy": n_correct / n_total if n_total else None,
        "per_topic": per_topic,
        "recognition_aggregate": {
            "n": n_recog,
            "accuracy": n_recog_corr / n_recog if n_recog else None,
            **recog_gain,
        },
    }


# ---------------------------------------------------------------------------
# TEI family: Acc_S | T_wrong, TEI rate, Escape rate.
# ---------------------------------------------------------------------------

def compute_tei(
    teacher_records: list[dict],
    student_after_records: list[dict],
    student_base_records: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """TEI / Escape on the teacher-wrong subset.

    Definitions (per `experiments/E1_filtered_delta_opd/README.md` §
    "Primary metrics"):

        T_wrong   = {s : teacher (32B) was wrong on sample s}
        TEI rate  = P(S_after answer = T's wrong answer | T_wrong)
        Escape    = P(S_after correct | T_wrong AND S_base answer = T's wrong answer)
        Acc_S|Tw  = P(S_after correct | T_wrong)

    `student_base_records` is optional; if absent, Escape rate is None.
    Validation invariant (used by `--test`): if student_after = student_base
    and teacher_records is the E0 32B run on VLMBias `main`, TEI rate must
    match E0's reported same-wrong / T-wrong = 1353 / 2179 ≈ 0.621.
    """
    by_id_s = {r["sample_id"]: r for r in student_after_records}
    by_id_b = {r["sample_id"]: r for r in (student_base_records or [])}

    t_wrong = [tr for tr in teacher_records if not tr.get("correct_I")]
    n_t_wrong = len(t_wrong)
    n_t_wrong_with_student = 0
    n_s_correct_on_t_wrong = 0
    n_s_matches_t_wrong_answer = 0

    n_base_inherited = 0
    n_escaped = 0

    for tr in t_wrong:
        sr = by_id_s.get(tr["sample_id"])
        if sr is None:
            continue
        n_t_wrong_with_student += 1

        t_ans = extract_answer(tr)
        s_after_ans = extract_answer(sr)
        s_after_correct = bool(sr.get("correct_I"))

        if s_after_correct:
            n_s_correct_on_t_wrong += 1
        if s_after_ans and s_after_ans == t_ans:
            n_s_matches_t_wrong_answer += 1

        if by_id_b:
            sb = by_id_b.get(tr["sample_id"])
            if sb is not None and not sb.get("correct_I"):
                s_base_ans = extract_answer(sb)
                if s_base_ans and s_base_ans == t_ans:
                    n_base_inherited += 1
                    if s_after_correct:
                        n_escaped += 1

    out: dict[str, Any] = {
        "n_t_wrong": n_t_wrong,
        "n_t_wrong_with_student": n_t_wrong_with_student,
        "acc_s_on_t_wrong": (
            n_s_correct_on_t_wrong / n_t_wrong_with_student
            if n_t_wrong_with_student else None
        ),
        "tei_rate": (
            n_s_matches_t_wrong_answer / n_t_wrong_with_student
            if n_t_wrong_with_student else None
        ),
        "n_s_matches_t_wrong_answer": n_s_matches_t_wrong_answer,
    }
    if by_id_b:
        out["n_base_inherited"] = n_base_inherited
        out["escape_rate"] = (
            n_escaped / n_base_inherited if n_base_inherited else None
        )
        out["n_escaped"] = n_escaped
    return out


# ---------------------------------------------------------------------------
# POPE: accuracy, F1, yes-rate, hallucinated-yes.
# ---------------------------------------------------------------------------

def compute_pope_metrics(records: list[dict]) -> dict[str, Any]:
    """Standard POPE binary metrics on yes/no answers."""
    n_total = len(records)
    n_correct = 0
    tp = fp = fn = tn = 0
    n_pred_yes = 0
    n_grounded_yes = 0
    n_hallucinated_yes = 0

    for r in records:
        pred = extract_answer(r)
        gold = _gold_canonical(r)
        is_pred_yes = pred == "yes"
        is_gold_yes = gold == "yes"

        if pred == gold:
            n_correct += 1
        if is_pred_yes:
            n_pred_yes += 1
        if is_pred_yes and is_gold_yes:
            tp += 1
            n_grounded_yes += 1
        elif is_pred_yes and not is_gold_yes:
            fp += 1
            n_hallucinated_yes += 1
        elif not is_pred_yes and is_gold_yes:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    return {
        "n_samples": n_total,
        "accuracy": n_correct / n_total if n_total else None,
        "precision_yes": precision,
        "recall_yes": recall,
        "f1_yes": f1,
        "yes_rate": n_pred_yes / n_total if n_total else None,
        "grounded_yes": n_grounded_yes,
        "hallucinated_yes": n_hallucinated_yes,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# MathVista: accuracy, response length, parse success.
# ---------------------------------------------------------------------------

def _mathvista_parse_success(record: dict) -> bool:
    """A response 'parses' if we can recover a non-empty short-form answer.

    For `multi_choice` questions: any choice substring appears in the response
    OR a recognizable letter A-E appears.
    For `free_form`: response is non-empty after stripping.
    """
    text = (record.get("ans_T_I") or "").strip()
    if not text:
        return False
    extras = record.get("extras") or {}
    if extras.get("question_type") == "multi_choice":
        choices = extras.get("choices") or []
        if any(str(c).strip().lower() in text.lower() for c in choices if c):
            return True
        if re.search(r"\b[A-Ea-e]\b", text):
            return True
        return False
    return True


def compute_mathvista_metrics(records: list[dict]) -> dict[str, Any]:
    n_total = len(records)
    n_correct = sum(1 for r in records if r.get("correct_I"))
    lengths = [len((r.get("ans_T_I") or "")) for r in records]
    n_parsed = sum(1 for r in records if _mathvista_parse_success(r))
    return {
        "n_samples": n_total,
        "accuracy": n_correct / n_total if n_total else None,
        "response_length_p50": percentile(lengths, 50),
        "response_length_p95": percentile(lengths, 95),
        "response_length_mean": safe_mean([float(x) for x in lengths]),
        "parse_success_rate": n_parsed / n_total if n_total else None,
    }


# ---------------------------------------------------------------------------
# Aggregator.
# ---------------------------------------------------------------------------

def evaluate(
    student_after_vlmbias: list[dict],
    teacher_vlmbias: list[dict],
    student_after_pope: list[dict],
    student_after_mathvista: list[dict],
    student_base_vlmbias: Optional[list[dict]] = None,
    tokenizer_path: Optional[str] = None,
) -> dict[str, Any]:
    """Single entry point for the metric layer.

    Each dataset's records carry the E0 jsonl schema: at minimum
    `sample_id`, `gold`, `ans_T_I`, `correct_I`, `dataset`, `extras`,
    plus `option_logP` on VLMBias for gain_margin.
    """
    option_len_fn = make_option_len_fn(tokenizer_path) if tokenizer_path else None

    return {
        "vlmbias": compute_vlmbias_metrics(student_after_vlmbias, option_len_fn=option_len_fn),
        "tei": compute_tei(teacher_vlmbias, student_after_vlmbias, student_base_vlmbias),
        "pope": compute_pope_metrics(student_after_pope),
        "mathvista": compute_mathvista_metrics(student_after_mathvista),
        "_counts": {
            "vlmbias_student_after": len(student_after_vlmbias),
            "vlmbias_teacher": len(teacher_vlmbias),
            "vlmbias_student_base": len(student_base_vlmbias or []),
            "pope_student_after": len(student_after_pope),
            "mathvista_student_after": len(student_after_mathvista),
        },
    }


# ===========================================================================
# INFERENCE LAYER — server, GPU. Lazy-imported.
# ===========================================================================

def _load_e0_helpers():
    """Lazy-import E0 dual_forward helpers and sample loaders.

    E0 loader names are `load_vlmbias` / `load_pope` / `load_mathvista`
    (no `_adversarial` / `_mini` suffix). Both POPE and MathVista loaders
    have non-None defaults for `n_samples` (1000 and 500), so passing
    `n_samples=None` explicitly returns ALL rows in the on-disk dataset.
    """
    from experiments.E0_image_null_delta.src.dual_forward import (
        greedy_generate, forced_score, score_option,
        options_for_sample, parse_correctness,
    )
    from experiments.E0_image_null_delta.data.loaders import (
        load_vlmbias, load_pope, load_mathvista,
    )
    return {
        "greedy_generate": greedy_generate,
        "forced_score": forced_score,
        "score_option": score_option,
        "options_for_sample": options_for_sample,
        "parse_correctness": parse_correctness,
        "load_vlmbias": load_vlmbias,
        "load_pope": load_pope,
        "load_mathvista": load_mathvista,
    }


def load_student_hf(checkpoint_dir: Path | str, dtype: str = "bfloat16"):
    """Load a merged HF student checkpoint (after `verl.model_merger merge`).

    Returns (model, processor). Imports are inside this function so the
    metric layer doesn't drag torch in on Mac.
    """
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    torch_dtype = getattr(torch, dtype)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(checkpoint_dir),
        dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(str(checkpoint_dir))
    return model, processor


def run_inference_on_dataset(
    model,
    processor,
    samples_iter,
    output_jsonl: Path,
    *,
    dataset_name: str,
    max_new_tokens: int = 256,
    top_k: int = 50,
    score_options: bool = True,
    limit: Optional[int] = None,
) -> int:
    """Greedy + forced + option-score every sample; dump E0-shape jsonl.

    `samples_iter` yields E0 `Sample` objects (from loaders.py). The output
    schema matches `e0_student7b_*` jsonls so the metric layer reads it
    without modification.
    """
    helpers = _load_e0_helpers()
    greedy_generate = helpers["greedy_generate"]
    forced_score = helpers["forced_score"]
    score_option = helpers["score_option"]
    options_for_sample = helpers["options_for_sample"]
    parse_correctness = helpers["parse_correctness"]

    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    with open(output_jsonl, "w") as f:
        for sample in samples_iter:
            if limit is not None and n_done >= limit:
                break
            try:
                ans_T_I, _ = greedy_generate(model, processor, sample.question, sample.image, max_new_tokens)
                score_I = forced_score(model, processor, sample.question, sample.image, ans_T_I, top_k)
                correct_I = parse_correctness(sample, ans_T_I)

                option_logP = None
                if score_options:
                    opts = options_for_sample(sample)
                    if opts:
                        option_logP = {
                            "I": {
                                label: score_option(model, processor, sample.question, sample.image, text)
                                for label, text in opts.items()
                            },
                            # We leave the "null" side empty here — E1 eval doesn't need it
                            # (only student-side gain is wanted, and that gain compares
                            # different OPTION TEXTS under the same conditioning, so we
                            # store both options under "I" and leave "null" as zeros that
                            # cancel out in _compute_gain_margin). To preserve the E0
                            # schema shape, we also store the same scores under "null"
                            # so the gain subtraction is 0 — gain_margin is then driven
                            # entirely by the difference between option texts under image
                            # conditioning, which is the right student-side analogue.
                            #
                            # NOTE: this differs from E0's two-condition gain (image - null).
                            # For E1 eval the "image only" student-side gain is what the
                            # protocol asks for; using zero baselines under "null" makes
                            # gain_margin = mean(gain_gt) - mean(gain_bias) reduce to
                            # mean(I[gt]) - mean(I[bias]), which is the per-sample
                            # log-prob preference between gold and the bias option. That
                            # IS the student-side "do you prefer GT over the canonical-
                            # wrong answer" signal the README asks for.
                            "null": {label: 0.0 for label in opts},
                        }

                record = {
                    "dataset": dataset_name,
                    "sample_id": sample.sample_id,
                    "gold": sample.gold,
                    "extras": dict(sample.extras),
                    "question": sample.question,
                    "ans_T_I": ans_T_I,
                    "correct_I": correct_I,
                    "response_tokens": score_I["response_tokens"],
                    "logp_I": score_I["logp"],
                    "T": len(score_I["logp"]),
                }
                if option_logP is not None:
                    record["option_logP"] = option_logP
            except Exception as e:
                record = {
                    "dataset": dataset_name,
                    "sample_id": getattr(sample, "sample_id", f"__err_{n_done}"),
                    "error": f"{type(e).__name__}: {e}",
                }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            n_done += 1
    logger.info("[infer] %s: wrote %d records → %s", dataset_name, n_done, output_jsonl)
    return n_done


# ===========================================================================
# Unit-test driver — Mac-local, uses E0 jsonls as fixture.
# ===========================================================================

def run_self_test(e0_results_dir: Path | str) -> int:
    """Self-test: when student_after = student_base = E0 7B, TEI rate must
    match E0-reported same-wrong / T-wrong = 1353 / 2179 ≈ 0.621.

    Returns 0 on success, non-zero on validation failure.
    """
    e0_results_dir = Path(e0_results_dir)

    print(f"[test] loading E0 jsonls from {e0_results_dir}")
    teacher_vlm = load_records(shard_glob(e0_results_dir, "e0_teacher32b_vlmbias_main"))
    student_vlm = load_records(shard_glob(e0_results_dir, "e0_student7b_vlmbias_main"))
    teacher_pope = load_records(shard_glob(e0_results_dir, "e0_teacher32b_pope_adversarial"))
    student_pope = load_records(shard_glob(e0_results_dir, "e0_student7b_pope_adversarial"))
    student_mv = load_records(shard_glob(e0_results_dir, "e0_student7b_mathvista_mini"))

    print(f"[test] teacher VLMBias: {len(teacher_vlm)}; student VLMBias: {len(student_vlm)}")
    print(f"[test] teacher POPE: {len(teacher_pope)}; student POPE: {len(student_pope)}")
    print(f"[test] student MathVista: {len(student_mv)}")

    fails = 0

    # ---------- VLMBias metric layer ----------
    vlm = compute_vlmbias_metrics(student_vlm)
    print(f"\n[test] vlmbias (student 7B) global_acc={vlm['global_accuracy']:.3f} "
          f"n={vlm['n_samples']}")
    for t in VLMBIAS_TOPICS_ALL:
        if t not in vlm["per_topic"]:
            continue
        pt = vlm["per_topic"][t]
        print(f"  {t:<18s} n={pt['n']:>4d} acc={pt['accuracy']:.3f} "
              f"gain={pt['gain_margin']}")
    print(f"  Recognition Aggregate: n={vlm['recognition_aggregate']['n']} "
          f"acc={vlm['recognition_aggregate']['accuracy']:.3f}")

    # ---------- TEI: student_after = student_base = E0 7B ----------
    tei = compute_tei(teacher_vlm, student_vlm, student_base_records=student_vlm)
    print(f"\n[test] TEI (student_after=student_base=E0 7B):")
    print(f"  n_t_wrong               : {tei['n_t_wrong']}")
    print(f"  acc_s_on_t_wrong        : {tei['acc_s_on_t_wrong']:.3f}")
    print(f"  tei_rate                : {tei['tei_rate']:.3f}")
    print(f"  n_base_inherited        : {tei['n_base_inherited']}")
    print(f"  escape_rate             : {tei['escape_rate']}")

    # ---- Validation 1: TEI rate vs E0 reported 1353/2179 ≈ 0.621 ----
    expected_lo, expected_hi = 0.60, 0.64
    if not (expected_lo <= tei["tei_rate"] <= expected_hi):
        print(f"  ✗ TEI rate {tei['tei_rate']:.3f} NOT in expected [{expected_lo}, {expected_hi}] "
              f"(E0 reports 1353/2179 = 0.621)")
        fails += 1
    else:
        print(f"  ✓ TEI rate {tei['tei_rate']:.3f} matches E0 reported ~0.621")

    # ---- Validation 2: Escape rate trivially 0 (student_after = student_base) ----
    if tei["escape_rate"] != 0.0:
        print(f"  ✗ Escape rate should be 0.0 when student_after = student_base, "
              f"got {tei['escape_rate']}")
        fails += 1
    else:
        print(f"  ✓ Escape rate = 0.0 (trivially correct for student_after = student_base)")

    # ---------- POPE ----------
    pope = compute_pope_metrics(student_pope)
    print(f"\n[test] POPE (student 7B): acc={pope['accuracy']:.3f} "
          f"f1={pope['f1_yes']:.3f} yes_rate={pope['yes_rate']:.3f}")
    print(f"  grounded_yes={pope['grounded_yes']} hallucinated_yes={pope['hallucinated_yes']}")
    if not (0.0 <= pope["accuracy"] <= 1.0):
        print(f"  ✗ POPE accuracy out of [0,1]")
        fails += 1
    else:
        print(f"  ✓ POPE accuracy in [0,1]")

    # ---------- MathVista ----------
    mv = compute_mathvista_metrics(student_mv)
    print(f"\n[test] MathVista (student 7B): acc={mv['accuracy']:.3f} "
          f"n={mv['n_samples']} resp_p50={mv['response_length_p50']:.0f} "
          f"resp_p95={mv['response_length_p95']:.0f} "
          f"parse_success={mv['parse_success_rate']:.3f}")
    if not (0.0 <= mv["accuracy"] <= 1.0):
        print(f"  ✗ MathVista accuracy out of [0,1]")
        fails += 1
    else:
        print(f"  ✓ MathVista accuracy in [0,1]")

    # ---------- End-to-end aggregate ----------
    print(f"\n[test] aggregate evaluate() OK; producing JSON summary...")
    summary = evaluate(
        student_after_vlmbias=student_vlm,
        teacher_vlmbias=teacher_vlm,
        student_after_pope=student_pope,
        student_after_mathvista=student_mv,
        student_base_vlmbias=student_vlm,
    )
    # Sanity: every top-level key populated
    for k in ("vlmbias", "tei", "pope", "mathvista", "_counts"):
        if k not in summary:
            print(f"  ✗ aggregate missing key: {k}")
            fails += 1
    if all(k in summary for k in ("vlmbias", "tei", "pope", "mathvista", "_counts")):
        print(f"  ✓ aggregate produces all 5 sections")

    print(f"\n[test] {'PASS' if fails == 0 else f'FAIL ({fails} failures)'}")
    return 0 if fails == 0 else 1


# ===========================================================================
# CLI.
# ===========================================================================

def _cmd_metrics(args: argparse.Namespace) -> int:
    """Compute metrics from pre-generated jsonls; write summary JSON."""
    student_vlm = load_records(args.student_vlmbias_jsonl)
    teacher_vlm = load_records(args.teacher_vlmbias_jsonl)
    student_pope = load_records(args.student_pope_jsonl)
    student_mv = load_records(args.student_mathvista_jsonl)
    student_base_vlm = load_records(args.student_base_vlmbias_jsonl) if args.student_base_vlmbias_jsonl else None

    summary = evaluate(
        student_after_vlmbias=student_vlm,
        teacher_vlmbias=teacher_vlm,
        student_after_pope=student_pope,
        student_after_mathvista=student_mv,
        student_base_vlmbias=student_base_vlm,
        tokenizer_path=args.tokenizer_path,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[metrics] wrote summary → {out_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    """Run student inference on one dataset; write E0-shape jsonl."""
    helpers = _load_e0_helpers()
    model, processor = load_student_hf(args.checkpoint, dtype=args.dtype)

    # When sharding, load ALL rows so the shard slicing covers the full dataset.
    # `--limit` then caps each shard's own work (mostly useful for smoke).
    loader_n = None if args.num_shards > 1 else args.limit

    if args.dataset == "vlmbias":
        samples_iter = helpers["load_vlmbias"](args.dataset_root, subset=args.subset or "main", n_samples=loader_n)
        ds_name = f"vlmbias_{args.subset or 'main'}"
    elif args.dataset == "pope":
        samples_iter = helpers["load_pope"](args.dataset_root, n_samples=loader_n)
        ds_name = "pope_adversarial"
    elif args.dataset == "mathvista":
        samples_iter = helpers["load_mathvista"](args.dataset_root, n_samples=loader_n)
        ds_name = "mathvista_mini"
    else:
        raise ValueError(f"unknown dataset {args.dataset!r}")

    # Round-robin sharding (same scheme as precompute_teacher.py).
    samples_list = list(samples_iter)
    if args.num_shards > 1:
        total = len(samples_list)
        samples_list = samples_list[args.shard_index::args.num_shards]
        logger.info(
            "[shard] %d/%d → %d samples for shard %d",
            len(samples_list), total, len(samples_list), args.shard_index,
        )

    run_inference_on_dataset(
        model=model,
        processor=processor,
        samples_iter=samples_list,
        output_jsonl=Path(args.output),
        dataset_name=ds_name,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        score_options=(args.dataset in {"vlmbias", "pope"}),
        limit=args.limit,
    )
    return 0


def _cmd_test(args: argparse.Namespace) -> int:
    return run_self_test(args.e0_results_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1 evaluation: TEI / Escape / VLMBias / POPE / MathVista")
    sub = p.add_subparsers(dest="cmd", required=True)

    # metrics
    pm = sub.add_parser("metrics", help="Compute metrics from pre-generated jsonls")
    pm.add_argument("--student-vlmbias-jsonl", nargs="+", required=True,
                    help="Student-after VLMBias jsonl shard(s)")
    pm.add_argument("--teacher-vlmbias-jsonl", nargs="+", required=True,
                    help="Teacher (E0 32B) VLMBias jsonl shard(s) — source of T_wrong")
    pm.add_argument("--student-pope-jsonl", nargs="+", required=True,
                    help="Student-after POPE-adv jsonl shard(s)")
    pm.add_argument("--student-mathvista-jsonl", nargs="+", required=True,
                    help="Student-after MathVista-mini jsonl shard(s)")
    pm.add_argument("--student-base-vlmbias-jsonl", nargs="+", default=None,
                    help="Optional baseline (un-distilled) student VLMBias jsonl(s) for Escape rate")
    pm.add_argument("--tokenizer-path", default=None,
                    help="Tokenizer path for length-normalized gain_margin (e.g. Qwen2.5-VL-7B-Instruct)")
    pm.add_argument("--output", required=True, help="Output summary JSON")
    pm.set_defaults(func=_cmd_metrics)

    # infer
    pi = sub.add_parser("infer", help="Run student inference on a dataset (server-side, GPU)")
    pi.add_argument("--checkpoint", required=True, help="Merged HF student checkpoint dir")
    pi.add_argument("--dataset", required=True, choices=["vlmbias", "pope", "mathvista"])
    pi.add_argument("--dataset-root", required=True, help="Path to the save_to_disk dir")
    pi.add_argument("--subset", default=None, help="(VLMBias only) sub-config; defaults to 'main'")
    pi.add_argument("--output", required=True, help="Output jsonl path")
    pi.add_argument("--limit", type=int, default=None, help="Cap samples (smoke)")
    pi.add_argument("--shard-index", type=int, default=0,
                    help="0-indexed shard (multi-GPU parallel). Round-robin from full dataset.")
    pi.add_argument("--num-shards", type=int, default=1,
                    help="Total shard count. Run N processes with different --shard-index "
                         "on different CUDA_VISIBLE_DEVICES, then `cat shard*.jsonl > all.jsonl`.")
    pi.add_argument("--max-new-tokens", type=int, default=256)
    pi.add_argument("--top-k", type=int, default=50)
    pi.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    pi.set_defaults(func=_cmd_infer)

    # test
    pt = sub.add_parser("test", help="Self-test the metric layer against E0 jsonls on Mac")
    pt.add_argument("--e0-results-dir",
                    default="experiments/E0_image_null_delta/results",
                    help="Path to the E0 results dir containing shard jsonls")
    pt.set_defaults(func=_cmd_test)

    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
