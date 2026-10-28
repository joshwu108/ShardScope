"""Compare unprofiled runs against observed run-to-run variance.

Not implemented. Requires the same methodology as the throughput harness.
A change that cannot be distinguished from noise must not become a headline
percentage improvement.
"""

from __future__ import annotations

import sys

from run_baseline import BLOCKED_ON


def main() -> int:
    """Refuse the way the rest of the project refuses: one line, exit 2.

    Requires the same benchmark methodology as the harness it compares the
    output of: without an agreed variance test this command could only
    report a percentage it cannot defend.
    """
    print(
        f"error: benchmarks/compare.py is not implemented yet. Blocked on: {BLOCKED_ON}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
