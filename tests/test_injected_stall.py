"""Injected data-loader stall detection -- Week 4 (requirement SS-05).

Acceptance test from PRD section 6: a documented artificial ~50-100 ms
loader delay must produce a slow-input warning backed by a trace difference.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Week 4: blocked on capture + rules implementation and student thresholds."
)


def test_injected_delay_raises_loader_wait_warning(): ...


def test_no_warning_on_clean_baseline(): ...
