"""Compare two unprofiled benchmark results and report whether the delta
exceeds observed run-to-run variance (PRD v2.0 section 9, "Statistical
trust").

Status: STUB (Week 7 deliverable).

A comparison that cannot distinguish the change from noise must say so
rather than reporting a headline percentage.
"""

from __future__ import annotations

import sys

from run_baseline import BLOCKED_ON


def main() -> int:
    """Refuse the way the rest of the project refuses: one line, exit 2.

    Blocked on the same GATE 7.1 methodology as the harness it compares the
    output of: without an agreed variance test this command could only
    report a percentage it cannot defend.
    """
    print(
        "error: benchmarks/compare.py is not implemented yet -- it is a Week 7 "
        f"deliverable. Blocked on: {BLOCKED_ON}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
