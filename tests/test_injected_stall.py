"""Synthetic host-wait comparisons; real capture acceptance remains separate."""

import pytest

from shardscope.errors import ArtifactError
from shardscope.parse import measurement
from shardscope.rules import RuleConfig, evaluate_rules


def synthetic_step(value):
    return {
        "training_step": 12,
        "measurements": {
            "host_data_wait_ms": measurement(
                value,
                basis="host_range" if value is not None else None,
                reason="synthetic host range" if value is not None else "marker missing",
                evidence=[{"file": "synthetic.json", "event_index": 3}]
                if value is not None
                else [],
            )
        },
    }


@pytest.mark.parametrize(
    "value,status,result",
    [
        (1, "below_threshold", False),
        (5, "possible_input_wait", True),
        (3, "possible_input_wait", True),
        (None, "unavailable", None),
    ],
)
def test_synthetic_host_wait_comparisons(value, status, result):
    output = evaluate_rules([synthetic_step(value)], RuleConfig(3))[0]
    assert output["status"] == status
    assert output["comparison"] == {
        "operator": ">=",
        "threshold_ms": 3,
        "value_ms": value,
        "result": result,
    }
    assert output["steps"] == [12]
    if value is not None:
        assert output["evidence"] == [{"file": "synthetic.json", "event_index": 3}]
    else:
        assert output["evidence"] == []


def test_no_configuration_or_no_evidence_is_not_a_diagnosis():
    assert evaluate_rules([synthetic_step(100)], None)[0]["status"] == "unconfigured"
    row = synthetic_step(100)
    row["measurements"]["host_data_wait_ms"]["evidence"] = []
    assert evaluate_rules([row], RuleConfig(3))[0]["status"] == "unavailable"
    assert (
        evaluate_rules([{"training_step": 1, "measurements": {}}], RuleConfig(3))[0]["status"]
        == "unavailable"
    )


@pytest.mark.parametrize("threshold", [True, -1, float("nan"), float("inf"), "3"])
def test_threshold_configuration_is_validated(threshold):
    with pytest.raises(ArtifactError):
        RuleConfig(threshold)


@pytest.mark.gpu
@pytest.mark.skip(
    reason="Requires capture integration, hardware traces and measured delay acceptance tolerances."
)
def test_real_injected_iterator_delay_acceptance():
    """Pending hardware test: iterator delay must resolve to actual trace evidence."""
