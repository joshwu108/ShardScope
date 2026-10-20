"""Trace/metric parser tests -- Week 3.

Kept as explicit skips so the suite never implies coverage it does not have.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Week 3: blocked on student-approved per-step metric schema (parse.py)."
)


def test_normalizes_events_by_rank_and_step(): ...


def test_unobservable_event_class_is_reported_not_observed(): ...
