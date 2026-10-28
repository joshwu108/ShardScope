"""Trace-backed detection of an injected data-loader stall.

A synthetic loader delay must produce an input-wait diagnostic with evidence
linking the warning to the affected trace events.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Requires trace capture, rule implementation and diagnostic thresholds."
)


def test_injected_delay_raises_loader_wait_warning(): ...


def test_no_warning_on_clean_baseline(): ...
