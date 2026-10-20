"""Transparent, rule-based bottleneck interpretation.

Status: SKELETON (Weeks 4-6 deliverable).

Hard constraints carried over from the PRD and not open to relaxation by an
agent:
  * rules state evidence ("loader wait rose from X to Y ms across steps
    N..M in trace T"), not verdicts stated as certainty;
  * no rule may sum overlapping kernel/copy/collective durations and present
    the result as a fraction of step time;
  * an unobservable event class yields ``NOT_OBSERVED``, never a number.

AWAITING STUDENT DECISION: the thresholds and the confidence vocabulary
(e.g. likely / uncertain / not observed) each rule is allowed to emit.
"""

from __future__ import annotations

from shardscope.env import NOT_OBSERVED

__all__ = ["NOT_OBSERVED"]
