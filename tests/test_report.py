"""Report rendering tests -- Week 3.

The key invariant under test later: a report must label profiled diagnostic
figures separately from unprofiled throughput figures (requirement SS-04).
"""

import pytest

pytestmark = pytest.mark.skip(reason="Week 3: blocked on report.py implementation.")


def test_report_separates_profiled_and_unprofiled_sections(): ...


def test_report_never_sums_overlapping_durations_into_a_percentage(): ...
