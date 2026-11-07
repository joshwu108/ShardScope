"""Transfer attribution, kernel classes, overlap measurements and their rules.

Synthetic Kineto-shaped traces drive exact arithmetic: every union and
intersection here can be checked by hand. Nothing is timed by the tests.
"""

from __future__ import annotations

import copy
import json

import pytest

from shardscope.analysis import validate_analysis, write_analysis
from shardscope.artifacts import load_run
from shardscope.attribution import (
    _LEGACY_CLASSIFICATION_KEYS,
    _LEGACY_LAUNCHES,
    ATTRIBUTED_KERNEL_METRIC,
    ATTRIBUTED_TRANSFER_METRIC,
    ATTRIBUTION_BASIS,
    OVERLAP_BASIS,
    TRANSFER_OVERLAP_METRIC,
    classify_device_events,
    is_kernel_event,
    is_transfer_event,
    transfer_direction,
)
from shardscope.availability import field_statuses
from shardscope.cli import main
from shardscope.errors import ArtifactError
from shardscope.parse import interval_intersection_us, measurement, parse_trace
from shardscope.report import render_html, render_json
from shardscope.rules import RULE_IDS, RuleConfig, evaluate_rules, parse_threshold
from test_kernel_attribution import (
    AUTOGRAD,
    CUDA_ENVIRONMENT,
    MAIN,
    TRACE,
    Builder,
    analyze,
    host,
    kernel,
    launch,
    row,
    step,
    write_run,
)

HTOD = "Memcpy HtoD (Pageable -> Device)"
DTOH = "Memcpy DtoH (Device -> Pageable)"
DTOD = "Memcpy DtoD (Device -> Device)"
NCCL = "ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>)"


def copy_call(ts, correlation, *, tid=MAIN, dur=10):
    return launch(ts, correlation, tid=tid, name="cudaMemcpyAsync", dur=dur)


def memcpy(ts, dur, correlation, *, name=HTOD, device=0, stream=7, size=None):
    event = kernel(ts, dur, correlation, device=device, stream=stream, cat="gpu_memcpy", name=name)
    if size is not None:
        event["args"]["bytes"] = size
    return event


def driver_launch(ts, correlation, *, tid=AUTOGRAD):
    return launch(ts, correlation, tid=tid, name="cuLaunchKernelEx", cat="cuda_driver")


def c10d_op(ts, dur, *, tid=AUTOGRAD, name="c10d::allreduce_"):
    return host(name, ts, dur, tid=tid, cat="cpu_op")


def metrics(payload, training_step):
    return row(payload, training_step)["measurements"]


# --- classification helpers ---------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        (HTOD, ("h2d", "HtoD")),
        (DTOH, ("d2h", "DtoH")),
        (DTOD, ("other", "DtoD")),
        ("Memcpy PtoP (Device -> Device)", ("other", "PtoP")),
        ("Memcpy <unknown> (Device -> Device)", ("other", "<unknown>")),
        ("Memcpy HtoD", ("unknown", None)),
        ("something else", ("unknown", None)),
    ],
)
def test_transfer_direction_follows_the_kineto_name_form(name, expected):
    assert transfer_direction(memcpy(0, 1, 1, name=name)) == expected


def test_transfer_and_kernel_categories_are_disjoint():
    assert is_transfer_event(memcpy(0, 1, 1)) and not is_kernel_event(memcpy(0, 1, 1))
    assert is_kernel_event(kernel(0, 1, 1)) and not is_transfer_event(kernel(0, 1, 1))
    memset = kernel(0, 1, 1, cat="gpu_memset", name="Memset (Device)")
    assert not is_transfer_event(memset) and not is_kernel_event(memset)


def test_interval_intersection_is_over_unions_not_pairs():
    first = [(0, 10), (5, 20)]  # union [0, 20)
    second = [(15, 30), (18, 19)]  # union [15, 30)
    assert interval_intersection_us(first, second) == 5
    assert interval_intersection_us([], second) == 0
    assert interval_intersection_us([(0, 5)], [(5, 10)]) == 0
    with pytest.raises(ArtifactError):
        interval_intersection_us([(5, 0)], second)


# --- transfer attribution -----------------------------------------------------------


def transfer_trace():
    """Step 1: an H2D copy that a compute kernel partly overlaps; step 2: a copy
    overlapped by a kernel launched in step 1 that ran late, plus a D2H copy.
    """
    b = Builder()
    s1 = b.extend(step(1, 1000, 1000))[0]
    s2 = b.extend(step(2, 3000, 1000))[0]
    ix = {
        "h2d1": b.add(memcpy(1500, 100, 21, size=2000)),  # [1500, 1600)
        "k1": b.add(kernel(1550, 150, 11)),  # [1550, 1700): overlaps h2d1 by 50
        "late": b.add(kernel(3100, 200, 12)),  # launched in step 1, runs [3100, 3300)
        "h2d2": b.add(memcpy(3200, 200, 22, size=4000)),  # [3200, 3400): overlap 100
        "d2h2": b.add(memcpy(3500, 50, 23, name=DTOH)),  # [3500, 3550): no bytes
    }
    b.add(copy_call(1100, 21))
    b.add(launch(1200, 11))
    b.add(launch(1900, 12))
    b.add(copy_call(3050, 22))
    b.add(copy_call(3450, 23))
    return b, s1, s2, ix


def test_transfers_are_attributed_by_copy_call_and_overlap_any_compute_kernel(make_run):
    b, s1, s2, ix = transfer_trace()
    directory = write_run(make_run, {TRACE: b.events}, [(TRACE, s1, 1), (TRACE, s2, 2)])
    payload = analyze(directory)
    one, two = metrics(payload, 1), metrics(payload, 2)

    transfer = one[ATTRIBUTED_TRANSFER_METRIC]
    assert transfer["status"] == "recorded" and transfer["basis"] == ATTRIBUTION_BASIS
    assert transfer["value_ms"] == pytest.approx(0.1)
    assert transfer["evidence"] == [{"file": TRACE, "event_index": ix["h2d1"]}]
    assert transfer["scope"] == one[ATTRIBUTED_KERNEL_METRIC]["scope"]
    assert transfer["coverage"]["directions"]["h2d"] == {
        "events": 1,
        "ms": pytest.approx(0.1),
        "bytes": 2000,
    }
    assert transfer["coverage"]["completeness"] == "complete"
    overlap = one[TRANSFER_OVERLAP_METRIC]
    assert overlap["status"] == "recorded" and overlap["basis"] == OVERLAP_BASIS
    assert overlap["value_ms"] == pytest.approx(0.05)
    assert overlap["evidence"] == [
        {"file": TRACE, "event_index": ix["h2d1"]},
        {"file": TRACE, "event_index": ix["k1"]},
    ]
    assert overlap["coverage"]["exposed_ms"] == pytest.approx(0.05)

    # Step 2: the union of two copies in different directions, and the overlap
    # comes from a kernel attributed to step 1 (the device computed while copying).
    transfer = two[ATTRIBUTED_TRANSFER_METRIC]
    assert transfer["value_ms"] == pytest.approx(0.25)
    assert transfer["coverage"]["directions"]["d2h"] == {
        "events": 1,
        "ms": pytest.approx(0.05),
        "bytes": None,
    }
    assert two[TRANSFER_OVERLAP_METRIC]["value_ms"] == pytest.approx(0.1)
    assert {"file": TRACE, "event_index": ix["late"]} in two[TRANSFER_OVERLAP_METRIC]["evidence"]
    result = payload["kernel_attribution"]["traces"][TRACE]
    assert result["transfer_events"] == {
        "total": 3,
        "attributed": 3,
        "unmatched": 0,
        "ambiguous": 0,
    }
    assert [r["direction"] for r in result["transfers_attributed"]] == ["h2d", "h2d", "d2h"]
    assert "no bandwidth" in payload["kernel_attribution"]["transfer_scope"].lower() or (
        "bandwidth" in payload["kernel_attribution"]["transfer_scope"]
    )
    assert not any("GB/s" in w for w in payload["warnings"])


def test_overlap_is_unavailable_without_a_compute_kernel_on_the_device(make_run):
    b = Builder()
    s1 = b.extend(step(1, 1000, 1000))[0]
    b.add(memcpy(1500, 100, 21))
    b.add(copy_call(1100, 21))
    directory = write_run(make_run, {TRACE: b.events}, [(TRACE, s1, 1)])
    one = metrics(analyze(directory), 1)
    assert one[ATTRIBUTED_TRANSFER_METRIC]["status"] == "recorded"
    overlap = one[TRANSFER_OVERLAP_METRIC]
    assert overlap["status"] == "not_recorded" and overlap["value_ms"] is None
    assert "missing kernel collection" in overlap["reason"]


def test_copy_calls_outside_a_mapped_step_or_with_bad_ids_are_listed_not_guessed(make_run):
    b = Builder()
    s1 = b.extend(step(1, 1000, 1000))[0]
    bucket = b.add(memcpy(1500, 20, 31, name=DTOD))  # autograd-thread bucket copy
    dup_a = b.add(memcpy(1600, 20, 32))
    orphan = b.add(memcpy(1700, 20, 33))
    nodev = b.add(memcpy(1800, 20, 34, device=None))
    inside = b.add(memcpy(1900, 20, 35))
    b.add(copy_call(1100, 31, tid=AUTOGRAD))
    b.add(copy_call(1110, 32))
    b.add(copy_call(1120, 32))  # same correlation twice: conflict
    b.add(copy_call(1130, 34))
    b.add(copy_call(1140, 35))
    directory = write_run(make_run, {TRACE: b.events}, [(TRACE, s1, 1)])
    payload = analyze(directory)
    result = payload["kernel_attribution"]["traces"][TRACE]
    kinds = {r["transfer_event_index"]: r["kind"] for r in result["transfers_unmatched"]}
    assert kinds == {
        bucket: "launch_outside_mapped_step",
        orphan: "no_host_api_event",
        nodev: "missing_device",
    }
    assert [r["transfer_event_index"] for r in result["transfers_ambiguous"]] == [dup_a]
    assert result["transfers_ambiguous"][0]["kind"] == "conflicting_host_events"
    assert [r["transfer_event_index"] for r in result["transfers_attributed"]] == [inside]
    transfer = metrics(payload, 1)[ATTRIBUTED_TRANSFER_METRIC]
    assert transfer["value_ms"] == pytest.approx(0.02)
    assert transfer["coverage"]["completeness"] == "partial"
    assert any("copy-call events outside_mapped_step" in w for w in payload["warnings"])


def test_transfers_on_two_devices_withhold_the_single_figure(make_run):
    b = Builder()
    s1 = b.extend(step(1, 1000, 1000))[0]
    b.add(memcpy(1500, 100, 21, device=0))
    b.add(memcpy(1500, 300, 22, device=1))
    b.add(copy_call(1100, 21))
    b.add(copy_call(1110, 22))
    environment = copy.deepcopy(CUDA_ENVIRONMENT)
    environment["device_count"] = 2
    directory = write_run(make_run, {TRACE: b.events}, [(TRACE, s1, 1)], environment=environment)
    payload = analyze(directory)
    (entry,) = payload["kernel_attribution"]["traces"][TRACE]["steps"]
    assert [d["attributed_transfer_ms"] for d in entry["per_device"]] == [
        pytest.approx(0.1),
        pytest.approx(0.3),
    ]
    transfer = metrics(payload, 1)[ATTRIBUTED_TRANSFER_METRIC]
    assert transfer["status"] == "not_recorded" and "span devices [0, 1]" in transfer["reason"]
    assert metrics(payload, 1)[TRANSFER_OVERLAP_METRIC]["status"] == "not_recorded"


def test_cpu_only_environment_reports_transfer_metrics_not_observed(make_run):
    b = Builder()
    s1 = b.extend(step(1, 1000, 1000))[0]
    directory = write_run(
        make_run,
        {TRACE: b.events},
        [(TRACE, s1, 1)],
        environment=copy.deepcopy(CUDA_ENVIRONMENT) | {"cuda_available": False, "device_count": 0},
    )
    one = metrics(analyze(directory), 1)
    for metric in (ATTRIBUTED_KERNEL_METRIC, ATTRIBUTED_TRANSFER_METRIC, TRANSFER_OVERLAP_METRIC):
        assert one[metric]["status"] == "not_observed"


# --- collective classification --------------------------------------------------------


def collective_trace():
    """One compute kernel [1500, 1700) and one NCCL kernel [1650, 1750) on
    device 0, launched from the autograd thread inside c10d::allreduce_; plus a
    candidate with the name but no c10d op, and a candidate with the op but a
    plain name.
    """
    b = Builder()
    s1 = b.extend(step(1, 1000, 1000))[0]
    ix = {
        "compute": b.add(kernel(1500, 200, 11)),
        "nccl": b.add(kernel(1650, 100, 41, stream=17, name=NCCL)),
        "name_only": b.add(kernel(1800, 10, 42, stream=17, name=NCCL)),
        "op_only": b.add(kernel(1820, 10, 43, name="bucket_copy_kernel")),
    }
    b.add(launch(1200, 11))
    b.add(c10d_op(1400, 100))  # [1400, 1500) wraps the collective's launch
    b.add(driver_launch(1420, 41))
    b.add(driver_launch(1600, 42))  # not inside any c10d op
    b.add(c10d_op(1700, 50))
    b.add(launch(1710, 43, tid=AUTOGRAD))  # ordinary launch inside a c10d op
    return b, s1, ix


def test_collectives_need_both_signals_and_are_unioned_per_device_window(make_run):
    b, s1, ix = collective_trace()
    environment = copy.deepcopy(CUDA_ENVIRONMENT)
    environment["device_count"] = 2
    directory = write_run(make_run, {TRACE: b.events}, [(TRACE, s1, 1)], environment=environment)
    payload = analyze(directory)
    result = payload["kernel_attribution"]["traces"][TRACE]
    classes = {
        r["kernel_event_index"]: r["class"] for r in result["attributed"] + result["unmatched"]
    }
    assert classes == {
        ix["compute"]: "compute",
        ix["nccl"]: "collective",
        ix["name_only"]: "collective_candidate",
        ix["op_only"]: "collective_candidate",
    }
    (device,) = result["devices"]
    assert device["compute_kernel_event_indices"] == [ix["compute"]]
    assert device["collective_kernel_event_indices"] == [ix["nccl"]]
    assert sorted(device["candidate_kernel_event_indices"]) == [ix["name_only"], ix["op_only"]]
    (activity,) = result["device_activity"]
    assert activity == {
        "device": 0,
        "compute_kernel_events": 1,
        "collective_kernel_events": 1,
        "compute_kernel_ms": pytest.approx(0.2),
        "collective_kernel_ms": pytest.approx(0.1),
        "collective_compute_overlap_ms": pytest.approx(0.05),
        "overlapping_compute_kernel_event_indices": [ix["compute"]],
    }
    # Autograd-thread launches are outside every mapped step: nothing per step.
    assert [r["kernel_event_index"] for r in result["attributed"]] == [ix["compute"]]
    assert metrics(payload, 1)["device_collective_ms"]["status"] == "not_recorded"
    assert "window-level" in metrics(payload, 1)["device_collective_ms"]["reason"]
    write_analysis(directory, payload)
    states = {s.field: s for s in field_statuses(load_run(directory))}
    assert states["collective_activity"].state == "recorded"
    assert "1 collective kernel(s)" in states["collective_activity"].detail
    assert "2 candidate(s)" in states["collective_activity"].detail


def test_collective_rule_fires_at_the_exact_share_and_is_unavailable_without_collectives(
    make_run,
):
    b, s1, ix = collective_trace()
    environment = copy.deepcopy(CUDA_ENVIRONMENT)
    environment["device_count"] = 2
    directory = write_run(make_run, {TRACE: b.events}, [(TRACE, s1, 1)], environment=environment)
    payload = analyze(directory)
    steps, block = payload["steps"], payload["kernel_attribution"]

    def findings(config):
        return evaluate_rules(
            steps, config, rules=("collective_compute_overlap",), attribution=block
        )

    (hit,) = findings(RuleConfig(collective_exposed_share=0.5))
    assert hit["status"] == "possible_exposed_collective" and hit["steps"] == [1]
    assert hit["comparison"]["value"] == pytest.approx(0.5)
    assert hit["comparison"]["numerator_ms"] == pytest.approx(0.05)
    assert hit["comparison"]["denominator_ms"] == pytest.approx(0.1)
    assert hit["comparison"]["device"] == 0 and hit["comparison"]["trace"] == TRACE
    assert hit["evidence"] == [{"file": TRACE, "event_index": ix["nccl"]}]
    assert "not network" in hit["interpretation"]
    (miss,) = findings(RuleConfig(collective_exposed_share=0.51))
    assert miss["status"] == "below_threshold" and miss["evidence"] == hit["evidence"]
    (none,) = findings(None)
    assert none["status"] == "unconfigured"
    (legacy,) = evaluate_rules(
        steps, RuleConfig(collective_exposed_share=0.5), rules=("collective_compute_overlap",)
    )
    assert legacy["status"] == "unavailable" and "schema 2" in legacy["interpretation"]
    stripped = copy.deepcopy(block)
    stripped["traces"][TRACE]["device_activity"] = []
    (empty,) = evaluate_rules(
        steps,
        RuleConfig(collective_exposed_share=0.5),
        rules=("collective_compute_overlap",),
        attribution=stripped,
    )
    assert empty["status"] == "unavailable" and "classified" in empty["interpretation"]


def test_collective_rule_reports_each_trace_and_device_and_undefined_shares(make_run):
    b, s1, ix = collective_trace()
    # A second device carries a collective and no compute at all.
    b.add(kernel(1650, 100, 51, device=1, stream=17, name=NCCL))
    b.add(c10d_op(1400, 100, tid=AUTOGRAD + 1))
    b.add(driver_launch(1420, 51, tid=AUTOGRAD + 1))
    environment = copy.deepcopy(CUDA_ENVIRONMENT)
    environment["device_count"] = 2
    directory = write_run(make_run, {TRACE: b.events}, [(TRACE, s1, 1)], environment=environment)
    payload = analyze(directory)
    found = evaluate_rules(
        payload["steps"],
        RuleConfig(collective_exposed_share=0.9),
        rules=("collective_compute_overlap",),
        attribution=payload["kernel_attribution"],
    )
    assert [(f["comparison"]["device"], f["status"]) for f in found] == [
        (0, "below_threshold"),
        (1, "possible_exposed_collective"),
    ]
    assert found[1]["comparison"]["value"] == 1.0
    forged = copy.deepcopy(payload["kernel_attribution"])
    forged["traces"][TRACE]["device_activity"][0]["collective_kernel_ms"] = 0.0
    (zero,) = evaluate_rules(
        payload["steps"],
        RuleConfig(collective_exposed_share=0.9),
        rules=("collective_compute_overlap",),
        attribution=forged,
    )[:1]
    assert zero["status"] == "ambiguous" and "zero-length" in zero["interpretation"]


# --- device_transfer_overlap rule ------------------------------------------------------


SCOPE = "rank=0; training_step=1; trace=trace-00000.json; device=0"
REF = [{"file": TRACE, "event_index": 3}]


def synthetic_step(transfer_ms, overlap_ms, *, overlap_scope=SCOPE):
    measurements = {
        ATTRIBUTED_TRANSFER_METRIC: measurement(
            transfer_ms, basis=ATTRIBUTION_BASIS, evidence=REF, scope=SCOPE
        )
        if transfer_ms is not None
        else measurement(None, reason="none", scope=SCOPE),
        TRANSFER_OVERLAP_METRIC: measurement(
            overlap_ms,
            basis=OVERLAP_BASIS,
            evidence=REF + [{"file": TRACE, "event_index": 4}],
            scope=overlap_scope,
        )
        if overlap_ms is not None
        else measurement(None, reason="none", scope=SCOPE),
    }
    measurements[ATTRIBUTED_TRANSFER_METRIC]["coverage"] = {
        "directions": {"h2d": {"events": 1, "ms": transfer_ms, "bytes": None}},
        "completeness": "complete",
    }
    return {
        "training_step": 1,
        "boundary": {"file": TRACE, "event_index": 0},
        "measurements": measurements,
    }


def transfer_rule(step, config):
    (finding,) = evaluate_rules([step], config, rules=("device_transfer_overlap",))
    return finding


@pytest.mark.parametrize(
    "transfer_ms, overlap_ms, threshold, status, value",
    [
        (0.4, 0.1, 0.75, "possible_exposed_transfer", 0.75),  # exact boundary
        (0.4, 0.1, 0.7501, "below_threshold", 0.75),
        (0.4, 0.4, 0.0, "possible_exposed_transfer", 0.0),  # fully hidden still meets 0
        (0.4, 0.0, 1.0, "possible_exposed_transfer", 1.0),
    ],
)
def test_transfer_rule_compares_the_exposed_share(
    transfer_ms, overlap_ms, threshold, status, value
):
    finding = transfer_rule(
        synthetic_step(transfer_ms, overlap_ms), RuleConfig(transfer_exposed_share=threshold)
    )
    assert finding["status"] == status
    assert finding["comparison"]["value"] == pytest.approx(value)
    assert finding["comparison"]["numerator_ms"] == pytest.approx(transfer_ms - overlap_ms)
    assert finding["comparison"]["denominator_ms"] == transfer_ms
    assert finding["comparison"]["directions"]["h2d"]["events"] == 1
    # Shared transfer refs are cited once, the overlapping kernel once.
    assert finding["evidence"] == REF + [{"file": TRACE, "event_index": 4}]


def test_transfer_rule_declines_undefined_or_inconsistent_shares():
    config = RuleConfig(transfer_exposed_share=0.5)
    zero = transfer_rule(synthetic_step(0.0, 0.0), config)
    assert zero["status"] == "ambiguous" and "zero-length" in zero["interpretation"]
    longer = transfer_rule(synthetic_step(0.1, 0.2), config)
    assert longer["status"] == "ambiguous" and "longer than" in longer["interpretation"]
    other = transfer_rule(synthetic_step(0.4, 0.1, overlap_scope=SCOPE + "x"), config)
    assert other["status"] == "ambiguous" and "different scopes" in other["interpretation"]
    missing = transfer_rule(synthetic_step(0.4, None), config)
    assert missing["status"] == "unavailable" and missing["evidence"] == []
    assert TRANSFER_OVERLAP_METRIC in missing["interpretation"]
    assert transfer_rule(synthetic_step(0.4, 0.1), None)["status"] == "unconfigured"


# --- configuration ------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"transfer_exposed_share": 1.5},
        {"transfer_exposed_share": -0.1},
        {"collective_exposed_share": float("nan")},
        {"rank_imbalance_ratio": 0.5},
        {"rank_imbalance_ratio": float("inf")},
    ],
)
def test_out_of_range_device_and_rank_thresholds_are_refused(kwargs):
    with pytest.raises(ArtifactError):
        RuleConfig(**kwargs)


def test_threshold_keys_are_bound_to_the_command_that_evaluates_them(make_run, capsys):
    assert parse_threshold("transfer_exposed_share=0.5") == ("transfer_exposed_share", 0.5)
    assert parse_threshold("rank_imbalance_ratio=2") == ("rank_imbalance_ratio", 2)
    with pytest.raises(ArtifactError, match="shardscope ranks"):
        RuleConfig(rank_imbalance_ratio=2.0).restrict(RULE_IDS, command="shardscope analyze")
    RuleConfig(transfer_exposed_share=0.5).restrict(RULE_IDS, command="shardscope analyze")
    b = Builder()
    s1 = b.extend(step(1, 1000, 1000))[0]
    directory = write_run(make_run, {TRACE: b.events}, [(TRACE, s1, 1)])
    code = main(
        [
            "analyze",
            "--run",
            str(directory),
            "--step-map",
            "map.json",
            "--threshold",
            "rank_imbalance_ratio=2",
        ]
    )
    assert code == 2 and "not evaluated by shardscope analyze" in capsys.readouterr().err
    assert not (directory / "analysis.json").exists()


# --- persistence, validation and presentation -------------------------------------


@pytest.fixture
def saved(make_run):
    b, s1, s2, ix = transfer_trace()
    # Add a collective so every section of the block is populated.
    nccl = b.add(kernel(1650, 100, 41, stream=17, name=NCCL))
    b.add(c10d_op(1400, 100))
    b.add(driver_launch(1420, 41))
    ix["nccl"], ix["s1"], ix["s2"] = nccl, s1, s2
    environment = copy.deepcopy(CUDA_ENVIRONMENT)
    environment["device_count"] = 2
    directory = write_run(
        make_run, {TRACE: b.events}, [(TRACE, s1, 1), (TRACE, s2, 2)], environment=environment
    )
    config = RuleConfig(
        transfer_exposed_share=0.5, collective_exposed_share=0.5, h2d_share_threshold=0.1
    )
    payload = analyze_with(directory, config)
    write_analysis(directory, payload)
    return directory, payload, ix


def analyze_with(directory, config):
    from shardscope.analysis import analyze_run

    return analyze_run(directory, rank=0, step_map_file="map.json", config=config)


def test_saved_analysis_round_trips_and_reports_render_every_section(saved):
    directory, payload, ix = saved
    run = load_run(directory)
    assert run.analysis == payload
    html = render_html(run)
    for text in (
        "Device transfer attribution",
        "Device kernel classes and collective overlap",
        "attributed transfer ms",
        "collective/compute overlap ms",
        "gpu_memcpy events",
        "possible_exposed_transfer",
    ):
        assert text in html, text
    assert "schema 1 attribution" not in html
    rendered = json.loads(render_json(run))
    assert rendered["analysis"]["kernel_attribution"]["schema"] == 2
    states = {s.field: s for s in field_statuses(run)}
    assert states[ATTRIBUTED_TRANSFER_METRIC].state == "recorded"
    assert "attributed directions: d2h 1, h2d 2" in states[ATTRIBUTED_TRANSFER_METRIC].detail
    assert states[TRANSFER_OVERLAP_METRIC].state == "recorded"
    assert ATTRIBUTED_TRANSFER_METRIC in states["h2d_copy_ms"].detail
    assert states["collective_activity"].state == "recorded"


@pytest.mark.parametrize(
    "edit, match",
    [
        (
            lambda p, ix: p["steps"][1]["measurements"][ATTRIBUTED_TRANSFER_METRIC].update(
                value_ms=0.2
            ),
            "disagrees with the attribution block",
        ),
        (
            lambda p, ix: p["steps"][1]["measurements"][TRANSFER_OVERLAP_METRIC].update(
                value_ms=0.0
            ),
            "disagrees with the attribution block",
        ),
        (
            lambda p, ix: p["steps"][1]["measurements"][TRANSFER_OVERLAP_METRIC][
                "evidence"
            ].__setitem__(0, {"file": TRACE, "event_index": ix["nccl"] + 1}),
            "disagrees with the attribution block|not a gpu_memcpy or device kernel event",
        ),
        (
            lambda p, ix: p["steps"][1]["measurements"][ATTRIBUTED_TRANSFER_METRIC].update(
                basis=OVERLAP_BASIS
            ),
            "device_interval_union",
        ),
        (
            lambda p, ix: p["kernel_attribution"]["traces"][TRACE]["device_activity"][0].update(
                collective_compute_overlap_ms=0.0
            ),
            "collective overlap disagrees with its overlapping kernel list",
        ),
        (
            lambda p, ix: p["kernel_attribution"]["traces"][TRACE]["transfers_attributed"][
                0
            ].update(direction="d2h"),
            "disagrees with the trace",
        ),
        (
            lambda p, ix: [s["measurements"].pop(ATTRIBUTED_TRANSFER_METRIC) for s in p["steps"]],
            "all three attributed device metrics or none",
        ),
        (
            lambda p, ix: p["kernel_attribution"]["traces"][TRACE]["attributed"][0].update(
                {"class": "x"}
            ),
            "needs a class|disagrees",
        ),
        (
            lambda p, ix: p["findings"].__setitem__(
                next(
                    i
                    for i, f in enumerate(p["findings"])
                    if f["rule_id"] == "collective_compute_overlap"
                ),
                {
                    **next(
                        f for f in p["findings"] if f["rule_id"] == "collective_compute_overlap"
                    ),
                    "status": "below_threshold",
                },
            ),
            "findings disagree",
        ),
    ],
)
def test_tampered_transfer_or_collective_evidence_is_refused(saved, edit, match):
    directory, payload, ix = saved
    forged = copy.deepcopy(payload)
    edit(forged, ix)
    if match != "findings disagree":
        # Findings are rechecked first; keep them consistent with the forged
        # measurements so the attribution relationship check is what refuses.
        forged["findings"] = evaluate_rules(
            forged["steps"],
            RuleConfig.from_mapping(forged["rule_config"]),
            attribution=forged["kernel_attribution"],
        )
    with pytest.raises(ArtifactError, match=match):
        validate_analysis(forged, directory=directory, steps={0, 1, 2, 3})


def test_schema_one_sidecar_without_transfers_still_loads_and_is_labelled(saved):
    """A sidecar written by the kernels-only reader: no copies, classes or activity."""
    directory, payload, ix = saved
    legacy = copy.deepcopy(payload)
    block = legacy["kernel_attribution"]
    block["schema"] = 1
    block["supported_launch_events"] = block["supported_launch_events"][:2]
    for key in (
        "supported_copy_events",
        "transfer_category",
        "collective_classification",
        "transfer_scope",
        "overlap_scope",
        "collective_scope",
    ):
        block.pop(key)
    # What the kernels-only reader computed: the driver-API launch of the
    # collective was unsupported to it, and records carried no class.
    old = classify_device_events(
        parse_trace(directory / TRACE), [(1, ix["s1"]), (2, ix["s2"])], launches=_LEGACY_LAUNCHES
    )
    result = {key: old[key] for key in _LEGACY_CLASSIFICATION_KEYS}
    for label in ("attributed", "unmatched", "ambiguous"):
        result[label] = [{k: v for k, v in r.items() if k != "class"} for r in result[label]]
    legacy_keys = ("device", "attributed_kernel_ms", "kernel_event_indices", "launch_event_indices")
    result["steps"] = [
        {
            **entry,
            "per_device": [
                {k: d[k] for k in (*legacy_keys, "streams")} for d in entry["per_device"]
            ],
        }
        for entry in block["traces"][TRACE]["steps"]
    ]
    block["traces"][TRACE] = result
    for row_ in legacy["steps"]:
        for metric in (ATTRIBUTED_TRANSFER_METRIC, TRANSFER_OVERLAP_METRIC):
            row_["measurements"].pop(metric)
    legacy["rules"] = legacy["rules"][:3]
    legacy["rule_config"] = {"h2d_share_threshold": 0.1}
    legacy["findings"] = evaluate_rules(
        legacy["steps"], RuleConfig(h2d_share_threshold=0.1), rules=legacy["rules"]
    )
    legacy["warnings"] = [
        w for w in legacy["warnings"] if "transfer" not in w and "collective" not in w
    ]
    write_analysis(directory, legacy, overwrite=True)
    run = load_run(directory)
    assert run.analysis == legacy
    html = render_html(run)
    assert "schema 1 attribution" in html and "Device transfer attribution" not in html
    states = {s.field: s for s in field_statuses(run)}
    assert ATTRIBUTED_TRANSFER_METRIC not in states
    assert states["collective_activity"].state == "not_recorded"
    assert main(["findings", "--run", str(directory), "--rule", "collective_compute_overlap"]) == 0


def test_findings_cli_presents_device_rules(saved, capsys):
    directory, payload, ix = saved
    assert main(["findings", "--run", str(directory), "--rule", "device_transfer_overlap"]) == 0
    text = capsys.readouterr().out
    assert "device_transfer_overlap [transfer_exposed_share=0.5]" in text
    assert "possible_exposed_transfer: steps 1-2" in text
    assert "not bandwidth" in text
    assert (
        main(
            [
                "findings",
                "--run",
                str(directory),
                "--format",
                "json",
                "--status",
                "possible_exposed_collective",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    collective = next(r for r in summary["rules"] if r["rule_id"] == "collective_compute_overlap")
    assert collective["counts"] == {"possible_exposed_collective": 1}
    assert collective["findings"][0]["comparison"]["device"] == 0
    assert main(["rules"]) == 0
    text = capsys.readouterr().out
    assert "device_transfer_overlap" in text and "collective_compute_overlap" in text
    assert "rank_step_imbalance  (" in text and "shardscope ranks" in text
