"""Trace capture: wrap a bounded window of training steps with torch.profiler.

Status: SKELETON (Week 2 deliverable).

AWAITING STUDENT DECISION (PRD section 12.1 approval gate) -- the agent must
not settle these on the student's behalf:
  * profiling scope: which activities (CPU / CUDA) and whether to record
    shapes, stacks, memory, flops (each costs overhead and trace size);
  * the schedule: wait / warmup / active / repeat step counts;
  * which phases get ``record_function`` markers and their exact names,
    since every downstream rule keys off those names;
  * whether traces are exported per rank to one directory or merged.

Nothing here touches the GPU hot path; hot-path changes stay student-owned.
"""

from __future__ import annotations

__all__: list[str] = []

# TODO(student): approve the profiler schedule and activity set, then the
# agent implements capture_window(...) against it.
