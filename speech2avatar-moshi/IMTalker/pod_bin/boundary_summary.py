#!/usr/bin/env python3
"""
Summarise boundary_l2 (and DBG/mimi, DBG/x0 if present) from a live WS debug log.

Usage:
    boundary_summary.py <path/to/live_ws_debug.log>

Prints one block of stats, nothing else. Safe to quote into a chat.
"""

from __future__ import annotations

import re
import statistics as stats
import sys
from pathlib import Path


BOUNDARY_RE = re.compile(r"chunk\s+(\d+).*?boundary_l2=([0-9.]+)")
MIMI_RE = re.compile(r"\[DBG/mimi\].*?full_samples=(\d+).*?remainder=(\d+)")
X0_RE = re.compile(r"\[DBG/x0\].*?head_norm=([0-9.]+)")
CHUNK_AGE_RE = re.compile(r"chunk_age=([0-9.]+)")


def percentile(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main(path: str) -> int:
    text = Path(path).read_text(errors="replace")

    b_pairs = [(int(m.group(1)), float(m.group(2))) for m in BOUNDARY_RE.finditer(text)]
    boundaries = [v for _, v in b_pairs]

    print(f"log: {path}")
    print(f"chunks: {len(boundaries)}")

    if boundaries:
        print("boundary_l2:")
        print(f"  mean   : {stats.mean(boundaries):.3f}")
        print(f"  median : {stats.median(boundaries):.3f}")
        print(f"  p90    : {percentile(boundaries, 0.90):.3f}")
        print(f"  p95    : {percentile(boundaries, 0.95):.3f}")
        print(f"  p99    : {percentile(boundaries, 0.99):.3f}")
        print(f"  max    : {max(boundaries):.3f}")

        spikes = sorted(((c, v) for c, v in b_pairs if v > 1.0), key=lambda x: -x[1])[:10]
        print(f"  spikes (>1.0, top 10 by value): {len(spikes)} total")
        for c, v in spikes:
            print(f"    chunk {c:5d}  boundary_l2={v:.3f}")

    mimi = [(int(m.group(1)), int(m.group(2))) for m in MIMI_RE.finditer(text)]
    if mimi:
        full = [f for f, _ in mimi]
        rem = [r for _, r in mimi]
        print(f"mimi (from DBG/mimi, {len(mimi)} entries):")
        print(f"  full_samples mean={stats.mean(full):.0f} min={min(full)} max={max(full)}")
        print(f"  remainder    mean={stats.mean(rem):.0f} min={min(rem)} max={max(rem)}")
        print(f"  empty-chunk events (full_samples=0): {sum(1 for f in full if f == 0)}")

    x0_norms = [float(m.group(1)) for m in X0_RE.finditer(text)]
    if x0_norms:
        print(f"x0 head_norm ({len(x0_norms)} entries):")
        print(f"  mean={stats.mean(x0_norms):.3f} min={min(x0_norms):.3f} max={max(x0_norms):.3f}")

    ages = [float(m.group(1)) for m in CHUNK_AGE_RE.finditer(text)]
    if ages:
        print(f"chunk_age ({len(ages)} entries):")
        print(f"  mean={stats.mean(ages):.3f} p95={percentile(ages, 0.95):.3f} max={max(ages):.3f}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: boundary_summary.py <path/to/log>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
