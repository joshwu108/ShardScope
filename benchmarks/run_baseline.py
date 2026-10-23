"""Warm, UNPROFILED throughput benchmark (requirement SS-04 / SS-06).

Status: STUB (Week 3 deliverable, hardened in Week 7).

This script must never enable torch.profiler: profiling overhead would make
its steps/s and examples/s numbers unusable as before/after evidence.
It records warmup, repeats, median and p95 step time, examples/s and peak
VRAM alongside the environment snapshot from ``shardscope.env``.
"""

from __future__ import annotations

import sys

#: The decision this harness waits on. Same wording as the ``blocked_on``
#: text ``shardscope bench`` prints, because they are the same blocker and
#: two spellings of it would eventually disagree.
BLOCKED_ON = (
    "the benchmark methodology -- warmup count, repeat count and the test used "
    "to call a difference real (GATE 7.1) are student decisions"
)


def main() -> int:
    """Refuse the way the rest of the project refuses: one line, exit 2.

    A ``NotImplementedError`` traceback here would read as a crash and exit
    1, which is what a broken harness looks like. This is not broken; it is
    waiting on a decision, and the message has to say which one.
    """
    print(
        "error: benchmarks/run_baseline.py is not implemented yet -- it is a Week 3 "
        f"deliverable. Blocked on: {BLOCKED_ON}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
