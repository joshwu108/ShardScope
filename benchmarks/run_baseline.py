"""Warm, UNPROFILED throughput benchmark (requirement SS-04 / SS-06).

Status: STUB (Week 3 deliverable, hardened in Week 7).

This script must never enable torch.profiler: profiling overhead would make
its steps/s and examples/s numbers unusable as before/after evidence.
It records warmup, repeats, median and p95 step time, examples/s and peak
VRAM alongside the environment snapshot from ``shardscope.env``.
"""

from __future__ import annotations


def main() -> int:
    raise NotImplementedError("Week 3: unprofiled benchmark harness.")


if __name__ == "__main__":
    raise SystemExit(main())
