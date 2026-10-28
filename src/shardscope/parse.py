"""Trace event parsing and normalization.

Not implemented. Requires a capture contract with stable phase markers and a
per-step schema that distinguishes measured durations from derived values.
Loader-wait timing also needs an operational definition.

Trace events must align by (rank, step_index), without assuming synchronized
cross-host wall clocks. Artifact loading and validation are implemented in
:mod:`shardscope.artifacts`; unsupported profiled columns are rejected rather
than silently dropped.
"""

from __future__ import annotations

__all__: list[str] = []
