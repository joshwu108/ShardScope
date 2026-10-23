"""Trace event parsing and normalization.

Status: SKELETON (Week 3 deliverable, trace half only).

The artifact half of Week 3 is implemented elsewhere and is not waiting on
anything: :mod:`shardscope.artifacts` discovers, loads and validates the
per-step CSV, provenance dump and recorded statistics a run writes, and
:mod:`shardscope.report` renders them. This module is specifically the
*trace* reader, and it stays empty until there are traces to read.

Scope, once the capture contract is approved:
  * read exported Chrome/Kineto traces;
  * normalize events onto (rank, step_index) keys -- never onto raw
    cross-host wall-clock timestamps (PRD v2.0 section 8: analysis must not
    imply precise cross-host clock synchronization);
  * join trace-derived phase timings onto the per-step rows.

AWAITING STUDENT DECISION: the ``record_function`` marker names every rule
keys off (GATE 2.3), and the per-step row schema -- which of step_time_ms,
data_wait_ms, h2d_copy_ms are measured directly vs. derived, and how a
derived field records its uncertainty (GATE 3.1), plus what ``data_wait_ms``
means operationally (GATE 3.2).

Until then the per-step CSV reader in :mod:`shardscope.artifacts` refuses a
file carrying profiled columns rather than dropping them, so a trace-era
artifact cannot be half-read by this era's code.
"""

from __future__ import annotations

__all__: list[str] = []
