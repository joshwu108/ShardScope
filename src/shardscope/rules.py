"""Evidence-based bottleneck interpretation.

Not implemented. Diagnostic thresholds and confidence levels require defined
evidence criteria. Rules must cite measurements and trace locations, preserve
unobservable activity as ``NOT_OBSERVED``, and never sum overlapping kernel,
copy, or collective durations into a fraction of step time.
"""

from __future__ import annotations

from shardscope.env import NOT_OBSERVED

__all__ = ["NOT_OBSERVED"]
