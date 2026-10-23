"""Trace parser tests -- Week 3, trace half.

Kept as explicit skips so the suite never implies coverage it does not have.
The artifact half of Week 3 is implemented and tested in
``tests/test_artifacts.py`` (loading, validation, serialization) and
``tests/test_report.py`` (rendering); what is skipped here is reading an
exported trace, which needs the capture contract first.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Week 3: blocked on student-approved trace markers and per-step metric schema."
)


def test_normalizes_events_by_rank_and_step(): ...


def test_unobservable_event_class_is_reported_not_observed(): ...
