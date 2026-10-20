"""Tiny single-device training loop -- the fixture every ShardScope run profiles.

Status: STUB (Week 1 deliverable, requirement SS-01).

Ownership (PRD section 12.1): the student writes this loop by hand in
Session A -- fixed-seed synthetic 28x28 inputs held in memory, a very small
network, 20-100 iterations, and a stopwatch around the whole iteration. The
agent may scaffold argument parsing and result formatting around it on
request, and must not introduce device-side or hot-path changes unasked.

Contract the rest of the package will rely on once this exists:
  * a fixed seed makes step timings comparable between runs;
  * the loop is importable (no work at import time) so tests can drive a
    handful of steps without a subprocess.
"""

from __future__ import annotations


def main() -> int:
    raise NotImplementedError("Week 1: student-authored training loop. See the module docstring.")


if __name__ == "__main__":
    raise SystemExit(main())
