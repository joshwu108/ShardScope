"""Unprofiled throughput benchmark entry point.

Not implemented. Requires defined warmup, repeat and variance criteria before
step rates, examples per second or peak memory can be reported. Benchmark runs
must never enable torch.profiler because its overhead distorts throughput.
"""

from __future__ import annotations

import sys

#: The decision this harness waits on. Same wording as the ``blocked_on``
#: text ``shardscope bench`` prints, because they are the same blocker and
#: two spellings of it would eventually disagree.
BLOCKED_ON = (
    "the benchmark methodology -- warmup count, repeat count and the test used "
    "to distinguish a change from noise remain undefined"
)


def main() -> int:
    """Refuse the way the rest of the project refuses: one line, exit 2.

    A ``NotImplementedError`` traceback here would read as a crash and exit
    1, which is what a broken harness looks like. This is not broken; it is
    waiting on a decision, and the message has to say which one.
    """
    print(
        f"error: benchmarks/run_baseline.py is not implemented yet. Blocked on: {BLOCKED_ON}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
