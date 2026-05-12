"""
Parse a verl smoke-run log file into structured per-step metrics.

Why this exists:
    A single `bash run_e1_recipe_smoke.sh X` produces ~5-10K lines of verl
    output (tqdm, NCCL chatter, vLLM warnings, FSDP comms). The signal
    we actually care about is in lines containing `training/global_step:`,
    which carry space-separated `key=value` pairs for every metric (loss
    components, e1_v1/* monitoring, response_length, etc.).

    This script extracts those lines into three artifacts:

      <out>.csv   — one row per training step; columns = metric names.
                    Ready for pandas / matplotlib / GPT analysis.
      <out>.json  — programmatic dump (all rows + a "summary" with
                    last-step values, min/max/mean per metric, and a
                    health-check verdict per E1 Protocol § Outcome).
      <out>.md    — terse markdown report: last-step metric table,
                    per-metric range, health-check verdict. Copy-paste
                    straight into GPT for analysis.

    The verdict block applies the E1 Protocol attribution-guard band
    (`kl_ce_ratio ∈ [0.3, 0.7]` for filtered configs) and flags out-of-band
    bucket-level ratios so you don't have to scan the table by eye.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


# A verl step line looks like:
#   ... training/global_step: 5 actor/loss=0.23 actor/e1_v1/kl_loss_sum=12.3 ...
# Some verl builds use "training/global_step:5" without a space. Both are
# accepted by the regex below.
_STEP_LINE_RE = re.compile(r"training/global_step\s*[:=]\s*(\d+)")

# Matches `key=value` where value is a float, scientific-notation number, or
# an integer. Values that aren't numeric (e.g. "True", strings) are skipped.
_KV_RE = re.compile(
    r"([A-Za-z_][\w./\-]*)\s*[:=]\s*"
    r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan|-?inf)"
)

# Default metric prefixes we want to keep. Everything else is dropped to
# keep the CSV narrow. Override with --include-prefix on the CLI.
DEFAULT_INCLUDE_PREFIXES = (
    "actor/e1_v1/",
    "actor/distillation/",
    "actor/loss",
    "actor/grad_norm",
    "actor/policy_loss",
    "actor/kl_loss",
    "actor/entropy_loss",
    "response_length/",
    "training/epoch",
    "training/global_step",
    "perf/",
)


def parse_log(log_path: Path, include_prefixes: tuple[str, ...]) -> list[dict]:
    """Yield one dict per training-step line in the log.

    Each dict has at least `step` (int); other keys are arbitrary metrics.
    """
    rows: list[dict] = []
    with open(log_path, errors="replace") as f:
        for line in f:
            m = _STEP_LINE_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            row: dict[str, Any] = {"step": step}
            for k, v in _KV_RE.findall(line):
                if k == "training/global_step":
                    continue
                if not any(k.startswith(p) for p in include_prefixes):
                    continue
                try:
                    fv = float(v)
                except ValueError:
                    continue
                row[k] = fv
            rows.append(row)
    # Dedup by step in case verl logs the same step twice (rare but happens
    # on retries). Keep the latest occurrence; that's the post-update value.
    by_step: dict[int, dict] = {}
    for r in rows:
        by_step[r["step"]] = r
    return [by_step[k] for k in sorted(by_step)]


def metric_columns(rows: list[dict]) -> list[str]:
    """Stable column order: `step` first, then metrics sorted alphabetically."""
    cols: set[str] = set()
    for r in rows:
        cols.update(r.keys())
    cols.discard("step")
    return ["step"] + sorted(cols)


def per_metric_range(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Per-metric {min, max, mean, last} across all step rows."""
    by_metric: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for k, v in r.items():
            if k == "step":
                continue
            if v is not None and math.isfinite(v):
                by_metric[k].append(v)
    return {
        k: {
            "min": float(min(vs)),
            "max": float(max(vs)),
            "mean": float(statistics.fmean(vs)),
            "last": float(vs[-1]),
            "n_steps": len(vs),
        }
        for k, vs in sorted(by_metric.items())
    }


def health_check(
    rows: list[dict],
    config_letter: Optional[str],
    kl_ce_band: tuple[float, float] = (0.3, 0.7),
) -> list[dict]:
    """Apply E1 Protocol attribution-guard checks; return a list of findings.

    Each finding: {severity: 'ok'/'warn'/'fail', metric, observed, expected, comment}.
    A run with zero `fail` findings is healthy.
    """
    findings: list[dict] = []
    if not rows:
        findings.append({
            "severity": "fail", "metric": "step_count", "observed": 0,
            "expected": ">=1", "comment": "no training/global_step lines parsed",
        })
        return findings

    last = rows[-1]

    def _check(metric: str, lo: float, hi: float, severity: str, note: str):
        v = last.get(metric)
        if v is None:
            return
        ok = lo <= v <= hi
        findings.append({
            "severity": "ok" if ok else severity,
            "metric": metric, "observed": v, "expected": f"[{lo}, {hi}]",
            "comment": note,
        })

    # delta_t normalization should land near 1.0 for B/D
    if config_letter in (None, "B", "D"):
        _check("actor/e1_v1/delta_t_mean_post_norm", 0.95, 1.05, "warn",
               "clip-then-rescale should land near 1.0; outside means normalization is broken")

    # kl_ce_ratio in attribution-guard band for C/D
    if config_letter in (None, "C", "D"):
        _check("actor/e1_v1/kl_ce_ratio", kl_ce_band[0], kl_ce_band[1], "fail",
               f"outside [{kl_ce_band[0]}, {kl_ce_band[1]}]: D > C would not be attributable to delta")
        # Per-bucket variants if present
        for b in ("virl39k", "pope_style", "tallyqa", "synthetic"):
            _check(f"actor/e1_v1/{b}_kl_ce_ratio", kl_ce_band[0], kl_ce_band[1], "warn",
                   f"bucket {b}: outside attribution band")

    # Loss sanity — should be finite + decreasing (very loose: just check finite)
    for metric in ("actor/loss", "actor/distillation/loss"):
        v = last.get(metric)
        if v is None:
            continue
        ok = math.isfinite(v) and v < 100
        findings.append({
            "severity": "ok" if ok else "fail",
            "metric": metric, "observed": v, "expected": "finite, <100",
            "comment": "loss should be finite and within sane range",
        })

    return findings


def write_csv(rows: list[dict], cols: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(
    rows: list[dict],
    ranges: dict,
    findings: list[dict],
    meta: dict,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "meta": meta,
            "rows": rows,
            "per_metric_range": ranges,
            "health_check": findings,
        }, f, indent=2, ensure_ascii=False)


def write_markdown(
    rows: list[dict],
    ranges: dict,
    findings: list[dict],
    meta: dict,
    path: Path,
) -> None:
    """Compact markdown report — copy-paste into GPT."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# E1 run report — {meta.get('config', '(unknown config)')}")
    lines.append("")
    lines.append(f"- log file: `{meta['log_path']}`")
    lines.append(f"- config: `{meta.get('config', '?')}` ({meta.get('config_name', '?')})")
    lines.append(f"- steps observed: {meta['n_steps']}")
    lines.append(f"- first step: {meta['first_step']}")
    lines.append(f"- last step: {meta['last_step']}")
    lines.append("")
    lines.append("## Health check")
    lines.append("")
    if not findings:
        lines.append("_(no health-check entries)_")
    else:
        lines.append("| severity | metric | observed | expected | comment |")
        lines.append("|---|---|---|---|---|")
        for f in findings:
            obs = f["observed"]
            obs_s = f"{obs:.4f}" if isinstance(obs, float) else str(obs)
            lines.append(
                f"| {f['severity']} | `{f['metric']}` | {obs_s} | "
                f"{f['expected']} | {f['comment']} |"
            )
    lines.append("")
    lines.append("## Last-step metrics")
    lines.append("")
    if rows:
        last = rows[-1]
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for k in sorted(last.keys()):
            if k == "step":
                continue
            v = last[k]
            v_s = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"| `{k}` | {v_s} |")
    lines.append("")
    lines.append("## Per-metric range across all steps")
    lines.append("")
    lines.append("| metric | min | max | mean | last | n |")
    lines.append("|---|---|---|---|---|---|")
    for k, r in ranges.items():
        lines.append(
            f"| `{k}` | {r['min']:.4f} | {r['max']:.4f} | "
            f"{r['mean']:.4f} | {r['last']:.4f} | {r['n_steps']} |"
        )
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract per-step metrics from a verl smoke-run log",
    )
    p.add_argument("log", help="Path to the run log file (e.g. /tmp/c.log)")
    p.add_argument(
        "--out", required=True,
        help="Output path prefix; writes <out>.csv, <out>.json, <out>.md",
    )
    p.add_argument(
        "--config", default=None,
        help="A/B/C/D config letter — drives the health-check rules (see E1 Protocol)",
    )
    p.add_argument(
        "--config-name", default=None,
        help="Recipe name (e.g., recipe_C_filtered_kd); informational only",
    )
    p.add_argument(
        "--include-prefix", nargs="*", default=list(DEFAULT_INCLUDE_PREFIXES),
        help="Metric-name prefixes to keep (everything else dropped)",
    )
    p.add_argument(
        "--kl-ce-band", nargs=2, type=float, default=[0.3, 0.7], metavar=("LO", "HI"),
        help="Attribution-guard band for kl_ce_ratio (default 0.3 0.7)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: log file not found: {log_path}", file=sys.stderr)
        return 2

    rows = parse_log(log_path, tuple(args.include_prefix))
    if not rows:
        print(f"WARN: no `training/global_step:` lines found in {log_path}")
    ranges = per_metric_range(rows)
    findings = health_check(rows, args.config, tuple(args.kl_ce_band))

    meta = {
        "log_path": str(log_path),
        "config": args.config,
        "config_name": args.config_name,
        "n_steps": len(rows),
        "first_step": rows[0]["step"] if rows else None,
        "last_step": rows[-1]["step"] if rows else None,
    }

    out_prefix = Path(args.out)
    cols = metric_columns(rows)
    write_csv(rows, cols, out_prefix.with_suffix(".csv"))
    write_json(rows, ranges, findings, meta, out_prefix.with_suffix(".json"))
    write_markdown(rows, ranges, findings, meta, out_prefix.with_suffix(".md"))

    print(f"[extract] {len(rows)} step rows, {len(cols)-1} metrics")
    print(f"[extract] wrote:")
    print(f"  {out_prefix.with_suffix('.csv')}")
    print(f"  {out_prefix.with_suffix('.json')}")
    print(f"  {out_prefix.with_suffix('.md')}")

    # Health-check exit code: 0 if no `fail`, 1 otherwise.
    n_fail = sum(1 for f in findings if f["severity"] == "fail")
    n_warn = sum(1 for f in findings if f["severity"] == "warn")
    print(f"[extract] health check: {len(findings)} entries "
          f"({n_fail} fail, {n_warn} warn)")
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
