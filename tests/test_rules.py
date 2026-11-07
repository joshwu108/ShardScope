"""Rule evaluation over labelled synthetic measurements; nothing here reads a trace.

Each test builds analysis step rows by hand so the comparisons, denominators
and failure modes are visible in the test itself. The numbers are invented.
"""

from __future__ import annotations

import json

import pytest

from shardscope.errors import ArtifactError
from shardscope.parse import DEVICE_METRICS, HOST_PHASES, measurement
from shardscope.rules import (
    LEGACY_RULES,
    RULE_ID,
    RULE_IDS,
    RULES,
    TRIGGERED_STATUSES,
    RuleConfig,
    describe_rules,
    evaluate_rules,
    parse_threshold,
)

TRACE = "trace-synthetic.json"
SCOPE = "rank=0; trace=trace-synthetic.json; pid=1; tid=1"
FINDING_KEYS = {"rule_id", "status", "steps", "inputs", "comparison", "evidence", "interpretation"}


def recorded(value_ms, *, index=1, scope=SCOPE, basis="host_range"):
    return measurement(
        value_ms,
        basis=basis,
        evidence=[{"file": TRACE, "event_index": index}],
        scope=scope,
        reason="synthetic host range",
    )


def absent(reason="missing marker"):
    return measurement(None, reason=reason, scope=SCOPE)


def row(training_step, mapped=True, **metrics):
    """A step row with every metric unavailable except the ones given."""
    measurements = {name: absent() for name in (*HOST_PHASES.values(), *DEVICE_METRICS)}
    for name, value in metrics.items():
        measurements[name] = value if isinstance(value, dict) else recorded(value)
    return {
        "training_step": training_step,
        "boundary": {"file": TRACE, "event_index": 0} if mapped else None,
        "measurements": measurements,
    }


def only(findings, rule_id):
    return [f for f in findings if f["rule_id"] == rule_id]


# --- registry ----------------------------------------------------------------------


def test_registry_is_consistent_and_serializable():
    assert tuple(spec.rule_id for spec in RULES) == RULE_IDS
    assert len(set(RULE_IDS)) == len(RULE_IDS) and RULE_IDS[0] == RULE_ID
    assert LEGACY_RULES == (RULE_ID,)
    described = describe_rules()
    assert json.loads(json.dumps(described)) == described
    owners = {}
    for spec in RULES:
        assert "unconfigured" in spec.statuses
        assert len(set(spec.statuses) & TRIGGERED_STATUSES) == 1
        assert spec.required_measurements and spec.next_investigation and spec.interpretation
        for item in spec.config:
            assert item["key"] in set(RuleConfig.keys())
            assert {"key", "unit", "type", "range", "compares"} <= item.keys()
            assert item["key"] not in owners, "a config key belongs to exactly one rule"
            owners[item["key"]] = spec.rule_id
    assert set(owners) == set(RuleConfig.keys())
    # The text never promises what the evidence cannot support.
    for spec in RULES:
        text = " ".join([spec.description, spec.interpretation, spec.next_investigation])
        for banned in ("root cause", "confidence", "speedup", "utilization"):
            assert banned not in text.lower(), (spec.rule_id, banned)


# --- configuration -----------------------------------------------------------------


def test_legacy_positional_config_and_sparse_serialization():
    config = RuleConfig(3)
    assert config.loader_wait_threshold_ms == 3 and config.to_dict() == {
        "loader_wait_threshold_ms": 3
    }
    assert RuleConfig().to_dict() == {}
    full = RuleConfig(0, 0.5, 1.5, 3)
    assert full.to_dict() == {
        "loader_wait_threshold_ms": 0,
        "h2d_share_threshold": 0.5,
        "step_variability_ratio": 1.5,
        "step_variability_min_steps": 3,
    }
    assert RuleConfig.from_mapping(full.to_dict()) == full


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(loader_wait_threshold_ms=-1), "loader_wait_threshold_ms"),
        (dict(loader_wait_threshold_ms=float("nan")), "loader_wait_threshold_ms"),
        (dict(loader_wait_threshold_ms=float("inf")), "loader_wait_threshold_ms"),
        (dict(loader_wait_threshold_ms=True), "loader_wait_threshold_ms"),
        (dict(loader_wait_threshold_ms="3"), "loader_wait_threshold_ms"),
        (dict(h2d_share_threshold=1.0001), "between 0 and 1"),
        (dict(h2d_share_threshold=-0.1), "between 0 and 1"),
        (dict(h2d_share_threshold=float("nan")), "between 0 and 1"),
        (dict(step_variability_ratio=0.99), "at least 1"),
        (dict(step_variability_ratio=float("inf")), "at least 1"),
        (dict(step_variability_ratio=1.5, step_variability_min_steps=1), "at least 2"),
        (dict(step_variability_ratio=1.5, step_variability_min_steps=2.5), "at least 2"),
        (dict(step_variability_ratio=1.5, step_variability_min_steps=True), "at least 2"),
        (dict(step_variability_min_steps=3), "requires step_variability_ratio"),
    ],
)
def test_invalid_configuration_is_refused(kwargs, match):
    with pytest.raises(ArtifactError, match=match):
        RuleConfig(**kwargs)


def test_configuration_boundaries_are_accepted():
    assert RuleConfig(loader_wait_threshold_ms=0).loader_wait_threshold_ms == 0
    assert RuleConfig(h2d_share_threshold=0).h2d_share_threshold == 0
    assert RuleConfig(h2d_share_threshold=1).h2d_share_threshold == 1
    assert RuleConfig(step_variability_ratio=1).step_variability_ratio == 1
    assert RuleConfig(step_variability_ratio=1, step_variability_min_steps=2)


def test_from_mapping_refuses_unknown_keys_and_non_objects():
    with pytest.raises(ArtifactError, match="unsupported key"):
        RuleConfig.from_mapping({"loader_wait_threshold_ms": 1, "gpu_idle_fraction": 0.5})
    with pytest.raises(ArtifactError, match="must be an object"):
        RuleConfig.from_mapping([1])


@pytest.mark.parametrize(
    "text,expected",
    [
        ("loader_wait_threshold_ms=5", ("loader_wait_threshold_ms", 5)),
        (" h2d_share_threshold = 0.25 ", ("h2d_share_threshold", 0.25)),
        ("step_variability_min_steps=4", ("step_variability_min_steps", 4)),
        ("step_variability_ratio=1e0", ("step_variability_ratio", 1.0)),
    ],
)
def test_parse_threshold_reads_plain_numbers(text, expected):
    key, value = parse_threshold(text)
    assert (key, value) == expected and type(value) is type(expected[1])


@pytest.mark.parametrize(
    "text,match",
    [
        ("loader_wait_threshold_ms", "KEY=VALUE"),
        ("=5", "KEY=VALUE"),
        ("loader_wait_threshold_ms=", "KEY=VALUE"),
        ("gpu_idle=0.5", "unknown threshold key"),
        ("loader_wait_threshold_ms=five", "plain finite number"),
        ("loader_wait_threshold_ms=1_0", "plain finite number"),
        ("loader_wait_threshold_ms=nan", "plain finite number"),
    ],
)
def test_parse_threshold_refuses_malformed_items(text, match):
    with pytest.raises(ArtifactError, match=match):
        parse_threshold(text)


# --- evaluation framing --------------------------------------------------------------


def test_unconfigured_rules_report_once_and_legacy_subset_is_honoured():
    steps = [row(0, host_data_wait_ms=4, host_step_ms=10, host_h2d_ms=5)]
    findings = evaluate_rules(steps, None)
    assert [f["rule_id"] for f in findings] == list(RULE_IDS)
    assert all(f["status"] == "unconfigured" and set(f) == FINDING_KEYS for f in findings)
    assert findings[0] == {
        "rule_id": RULE_ID,
        "status": "unconfigured",
        "steps": [],
        "inputs": [],
        "comparison": None,
        "evidence": [],
        "interpretation": "No explicit loader-wait threshold configured; no diagnosis.",
    }
    legacy = evaluate_rules(steps, RuleConfig(3), rules=LEGACY_RULES)
    assert [f["rule_id"] for f in legacy] == [RULE_ID]
    assert legacy == only(evaluate_rules(steps, RuleConfig(3)), RULE_ID)
    with pytest.raises(ArtifactError, match="unknown or repeated"):
        evaluate_rules(steps, None, rules=("host_loader_wait", "gpu_idle"))
    with pytest.raises(ArtifactError, match="unknown or repeated"):
        evaluate_rules(steps, None, rules=(RULE_ID, RULE_ID))


def test_every_finding_has_the_persisted_shape():
    steps = [row(0, host_data_wait_ms=4, host_step_ms=10, host_h2d_ms=5), row(1), row(2, False)]
    config = RuleConfig(3, 0.4, 1.2, 2)
    for finding in evaluate_rules(steps, config):
        assert set(finding) == FINDING_KEYS
        assert finding["rule_id"] in RULE_IDS
        assert finding["status"] in {s for spec in RULES for s in spec.statuses}
        assert isinstance(finding["evidence"], list)
        assert (finding["status"] in TRIGGERED_STATUSES) == bool(
            finding["comparison"] and finding["comparison"]["result"]
        )


# --- host_loader_wait: the legacy shape, byte for byte ---------------------------------


def test_loader_wait_shape_is_unchanged():
    wait = recorded(3.0, index=7)
    steps = [row(5, host_data_wait_ms=wait), row(6, host_data_wait_ms=recorded(2.999)), row(7)]
    findings = only(evaluate_rules(steps, RuleConfig(3)), RULE_ID)
    assert findings[0] == {
        "rule_id": "host_loader_wait",
        "status": "possible_input_wait",
        "steps": [5],
        "inputs": [{"training_step": 5, "metric": "host_data_wait_ms", "measurement": wait}],
        "comparison": {"operator": ">=", "threshold_ms": 3, "value_ms": 3.0, "result": True},
        "evidence": [{"file": TRACE, "event_index": 7}],
        "interpretation": (
            "Host wait meets the configured threshold: possible input waiting; "
            "does not establish GPU starvation or a confirmed root cause."
        ),
    }
    assert (
        findings[1]["status"] == "below_threshold" and findings[1]["comparison"]["result"] is False
    )
    assert findings[2]["status"] == "unavailable"
    assert findings[2]["comparison"] == {
        "operator": ">=",
        "threshold_ms": 3,
        "value_ms": None,
        "result": None,
    }
    assert (
        findings[2]["evidence"] == []
        and findings[2]["inputs"][0]["measurement"]["value_ms"] is None
    )


@pytest.mark.parametrize(
    "measured",
    [
        recorded(9, basis="device_interval_union"),
        {**recorded(9), "evidence": []},
        {**recorded(9), "status": "not_observed"},
        {**recorded(-1)},
        {**recorded(float("nan"))},
    ],
)
def test_loader_wait_requires_a_validated_host_range(measured):
    (finding,) = only(evaluate_rules([row(0, host_data_wait_ms=measured)], RuleConfig(1)), RULE_ID)
    assert finding["status"] == "unavailable" and finding["evidence"] == []


# --- host_h2d_share ----------------------------------------------------------------------


def h2d_finding(step_row, threshold):
    (finding,) = only(
        evaluate_rules([step_row], RuleConfig(h2d_share_threshold=threshold)), "host_h2d_share"
    )
    return finding


def test_h2d_share_denominator_is_the_same_steps_host_step_range():
    step_row = row(
        3,
        host_h2d_ms=recorded(4.0, index=11),
        host_step_ms=recorded(10.0, index=8),
        host_forward_ms=99.0,
    )
    finding = h2d_finding(step_row, 0.4)
    assert finding["status"] == "possible_host_copy_wait" and finding["steps"] == [3]
    assert finding["comparison"] == {
        "operator": ">=",
        "statistic": "host_h2d_ms / host_step_ms",
        "threshold": 0.4,
        "value": 0.4,
        "numerator_ms": 4.0,
        "denominator_ms": 10.0,
        "result": True,
    }
    assert finding["evidence"] == [
        {"file": TRACE, "event_index": 11},
        {"file": TRACE, "event_index": 8},
    ]
    assert [i["metric"] for i in finding["inputs"]] == ["host_h2d_ms", "host_step_ms"]
    assert "0.400 of the host step range" in finding["interpretation"]
    assert "not device transfer time" in finding["interpretation"]
    assert "Next:" in finding["interpretation"]


def test_h2d_share_exact_threshold_boundary():
    step_row = row(0, host_h2d_ms=3.999, host_step_ms=10.0)
    below = h2d_finding(step_row, 0.4)
    assert below["status"] == "below_threshold" and below["comparison"]["result"] is False
    assert below["evidence"] and "does not establish" in below["interpretation"]
    full = h2d_finding(row(0, host_h2d_ms=10.0, host_step_ms=10.0), 1)
    assert full["status"] == "possible_host_copy_wait" and full["comparison"]["value"] == 1.0
    zero = h2d_finding(row(0, host_h2d_ms=0.0, host_step_ms=10.0), 0)
    assert zero["status"] == "possible_host_copy_wait" and zero["comparison"]["value"] == 0.0


@pytest.mark.parametrize(
    "metrics,missing",
    [
        ({"host_step_ms": 10.0}, "host_h2d_ms"),
        ({"host_h2d_ms": 4.0}, "host_step_ms"),
        ({}, "host_h2d_ms, host_step_ms"),
        (
            {"host_h2d_ms": recorded(4.0, basis="device_interval_union"), "host_step_ms": 10.0},
            "host_h2d_ms",
        ),
    ],
)
def test_h2d_share_missing_inputs_are_unavailable_not_zero(metrics, missing):
    finding = h2d_finding(row(0, **metrics), 0.1)
    assert finding["status"] == "unavailable"
    assert finding["comparison"]["value"] is None and finding["comparison"]["result"] is None
    assert finding["evidence"] == []
    assert f"not recorded: {missing}" in finding["interpretation"]
    assert finding["comparison"]["denominator_ms"] == (10.0 if "host_step_ms" in metrics else None)


@pytest.mark.parametrize(
    "metrics,problem",
    [
        (
            {
                "host_h2d_ms": recorded(4.0, scope="rank=0; trace=other.json; pid=1; tid=1"),
                "host_step_ms": 10.0,
            },
            "different scopes",
        ),
        ({"host_h2d_ms": 0.0, "host_step_ms": 0.0}, "zero-length host step range"),
        ({"host_h2d_ms": 11.0, "host_step_ms": 10.0}, "longer than its host step range"),
    ],
)
def test_h2d_share_inconsistent_inputs_are_ambiguous(metrics, problem):
    finding = h2d_finding(row(0, **metrics), 0.1)
    assert finding["status"] == "ambiguous" and finding["evidence"] == []
    assert finding["comparison"]["value"] is None and finding["comparison"]["result"] is None
    assert problem in finding["interpretation"]
    # Recorded inputs are still reported even when the share is withheld.
    assert finding["comparison"]["numerator_ms"] is not None
    assert finding["comparison"]["denominator_ms"] is not None


def test_h2d_share_is_evaluated_per_step_and_unmapped_steps_are_unavailable():
    steps = [
        row(0, host_h2d_ms=5.0, host_step_ms=10.0),
        row(1, host_h2d_ms=1.0, host_step_ms=10.0),
        row(2, mapped=False),
    ]
    findings = only(evaluate_rules(steps, RuleConfig(h2d_share_threshold=0.5)), "host_h2d_share")
    assert [f["status"] for f in findings] == [
        "possible_host_copy_wait",
        "below_threshold",
        "unavailable",
    ]
    assert [f["steps"] for f in findings] == [[0], [1], [2]]


# --- step_duration_variability ------------------------------------------------------------


def variability(steps, ratio, min_steps=None):
    config = RuleConfig(step_variability_ratio=ratio, step_variability_min_steps=min_steps)
    (finding,) = only(evaluate_rules(steps, config), "step_duration_variability")
    return finding


def three_steps(values=(10.0, 10.0, 15.0)):
    return [
        row(index, host_step_ms=recorded(value, index=index + 1), host_forward_ms=1.0)
        for index, value in enumerate(values)
    ]


def test_variability_exact_threshold_names_the_slowest_step_and_cites_every_step():
    finding = variability(three_steps(), 1.5)
    assert finding["status"] == "possible_step_variability"
    assert finding["steps"] == [0, 1, 2]
    assert finding["comparison"] == {
        "operator": ">=",
        "statistic": "max(host_step_ms) / median(host_step_ms)",
        "threshold": 1.5,
        "value": 1.5,
        "result": True,
        "recorded_steps": 3,
        "required_steps": 2,
        "min_ms": 10.0,
        "median_ms": 10.0,
        "max_ms": 15.0,
        "slowest_training_step": 2,
    }
    assert finding["evidence"] == [{"file": TRACE, "event_index": i} for i in (1, 2, 3)]
    assert [i["training_step"] for i in finding["inputs"]] == [0, 1, 2]
    assert "Training step 2 took 1.500 times the median" in finding["interpretation"]
    assert "does not compare ranks" in finding["interpretation"]
    just_above = variability(three_steps(), 1.5000001)
    assert just_above["status"] == "below_threshold" and just_above["comparison"]["result"] is False
    assert (
        just_above["evidence"] and "does not establish steady steps" in just_above["interpretation"]
    )


def test_variability_median_is_a_median_not_a_mean():
    # mean would be 20; median is 10, so the ratio is 4, not 2.
    finding = variability(three_steps((10.0, 10.0, 40.0)), 3)
    assert finding["comparison"]["median_ms"] == 10.0 and finding["comparison"]["value"] == 4.0
    even = variability(three_steps((10.0, 20.0, 30.0, 40.0)), 1)
    assert even["comparison"]["median_ms"] == 25.0 and even["comparison"]["value"] == 1.6


def test_variability_insufficient_steps_forms_no_statistic():
    finding = variability(three_steps(), 1.1, min_steps=4)
    assert finding["status"] == "insufficient" and finding["steps"] == [0, 1, 2]
    assert finding["comparison"]["recorded_steps"] == 3
    assert finding["comparison"]["required_steps"] == 4
    assert finding["comparison"]["value"] is None and finding["comparison"]["max_ms"] is None
    assert finding["evidence"] == [] and "4 required" in finding["interpretation"]
    single = variability(three_steps((10.0,)), 1.0)
    assert single["status"] == "insufficient" and single["comparison"]["required_steps"] == 2
    two = variability(three_steps((10.0, 10.0)), 1.0)
    assert two["status"] == "possible_step_variability" and two["comparison"]["value"] == 1.0


def test_variability_without_any_recorded_step_is_unavailable():
    finding = variability([row(0), row(1, mapped=False)], 1.0)
    assert finding["status"] == "unavailable" and finding["steps"] == []
    assert finding["inputs"] == [] and finding["evidence"] == []
    assert finding["comparison"]["recorded_steps"] == 0


def test_variability_skips_steps_without_host_step_and_counts_only_recorded_ones():
    steps = three_steps() + [row(3, host_forward_ms=1.0), row(4, mapped=False)]
    finding = variability(steps, 1.5)
    assert finding["status"] == "possible_step_variability" and finding["steps"] == [0, 1, 2]
    assert finding["comparison"]["recorded_steps"] == 3


def test_variability_refuses_to_compare_steps_with_different_phase_sets():
    steps = three_steps()
    steps[2]["measurements"]["host_data_wait_ms"] = recorded(50.0)
    finding = variability(steps, 1.0)
    assert finding["status"] == "ambiguous" and finding["evidence"] == []
    assert finding["comparison"]["value"] is None and finding["comparison"]["result"] is None
    assert finding["comparison"]["phase_sets"] == [
        ["host_data_wait_ms", "host_forward_ms", "host_step_ms"],
        ["host_forward_ms", "host_step_ms"],
    ]
    assert "not known to be equivalent logical work" in finding["interpretation"]


def test_variability_zero_median_is_ambiguous():
    finding = variability(three_steps((0.0, 0.0, 0.0)), 1.0)
    assert finding["status"] == "ambiguous" and "undefined" in finding["interpretation"]
    assert finding["comparison"]["median_ms"] == 0.0 and finding["comparison"]["value"] is None
