"""Trace parsing contract tests.

Skipped until the capture markers and per-step schema are defined. Artifact
loading, serialization and report rendering have separate active test suites.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Trace markers and per-step metric schema are not defined.")


def test_normalizes_events_by_rank_and_step(): ...


def test_unobservable_event_class_is_reported_not_observed(): ...
