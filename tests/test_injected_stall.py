"""Injected data-loader stall detection -- Week 4 (requirement SS-05).

Acceptance test from PRD v2.0 section 8 ("MVP acceptance"): a documented
artificial ~50-100 ms loader delay must raise an evidence-linked input-wait
diagnostic backed by a trace difference.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Week 4: blocked on capture + rules implementation and student thresholds."
)


def test_injected_delay_raises_loader_wait_warning(): ...


def test_no_warning_on_clean_baseline(): ...
