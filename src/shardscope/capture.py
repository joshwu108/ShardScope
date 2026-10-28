"""Bounded training-step capture with torch.profiler.

Not implemented. The capture contract must define the activity set, optional
shape/stack/memory collection, wait/warmup/active/repeat schedule, phase marker
names, and per-rank trace layout before capture can be enabled. These choices
affect profiling overhead and the interpretation of downstream measurements.
"""

from __future__ import annotations

__all__: list[str] = []
