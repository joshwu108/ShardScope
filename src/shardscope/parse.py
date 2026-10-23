"""Trace + per-step metric parsing and normalization.

Status: SKELETON (Week 3 deliverable).

Scope, once the capture contract is approved:
  * read exported Chrome/Kineto traces and per-step records;
  * normalize events onto (rank, step_index) keys -- never onto raw
    cross-host wall-clock timestamps (PRD v2.0 section 8: analysis must not
    imply precise cross-host clock synchronization);
  * emit tidy per-step rows for CSV/JSON export.

AWAITING STUDENT DECISION: the per-step row schema (which of step_time_ms,
data_wait_ms, h2d_copy_ms, ... are measured directly vs. derived, and how a
derived field records its uncertainty).
"""

from __future__ import annotations

__all__: list[str] = []
