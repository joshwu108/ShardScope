"""Kernel-to-step attribution through launch correlation ids; synthetic traces only.

Every trace here is hand-built in the shape of a real ``torch.profiler`` CUDA
capture (host ``cuda_runtime`` launches carrying ``args.correlation``, device
``kernel`` ranges on ``pid 0`` carrying the same id plus ``device``/``stream``),
with invented timings. Nothing here measures hardware; the real-GPU acceptance
path is ``tests/test_capture_execution.py::test_real_cuda_capture`` plus a
manual ``shardscope analyze`` of a captured run.
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from conftest import GOOD_ENVIRONMENT
from shardscope.analysis import analyze_run, validate_analysis, write_analysis
from shardscope.artifacts import RunArtifacts, load_run
from shardscope.attribution import (
    ATTRIBUTED_KERNEL_METRIC,
    attribute_trace,
    is_kernel_event,
    is_supported_launch,
)
from shardscope.availability import field_statuses
from shardscope.errors import ArtifactError
from shardscope.parse import StepMapping, analyze_traces, parse_trace
from shardscope.report import render_html, render_json, render_steps_csv
from shardscope.validate import validate_run

METRIC = ATTRIBUTED_KERNEL_METRIC
TRACE = "trace-00000.json"
PID = 4242
MAIN = 4242
AUTOGRAD = 4300

CUDA_ENVIRONMENT = {
    **GOOD_ENVIRONMENT,
    "cuda_available": True,
    "cuda_version": "13.2",
    "device_count": 1,
    "device_names": ["synthetic-device"],
    "extra": {**GOOD_ENVIRONMENT["extra"], "device_resolved": "cuda:0", "profiled": True},
}


# --- event builders --------------------------------------------------------------


def host(name, ts, dur, *, tid=MAIN, cat="user_annotation"):
    return dict(ph="X", cat=cat, name=name, pid=PID, tid=tid, ts=ts, dur=dur, args={})


def launch(ts, correlation, *, tid=MAIN, name="cudaLaunchKernel", cat="cuda_runtime", dur=10):
    args = {"cbid": 211}
    if correlation is not None:
        args["correlation"] = correlation
    return dict(ph="X", cat=cat, name=name, pid=PID, tid=tid, ts=ts, dur=dur, args=args)


def kernel(ts, dur, correlation, *, device=0, stream=7, cat="kernel", name="k"):
    args = {"stream": stream, "queued": 0}
    if correlation is not None:
        args["correlation"] = correlation
    if device is not None:
        args["device"] = device
    return dict(ph="X", cat=cat, name=name, pid=0, tid=stream, ts=ts, dur=dur, args=args)


def step(training_step, ts, dur, *, tid=MAIN):
    """A mapped step boundary plus its nested identity marker."""
    return [
        host("shardscope/step", ts, dur, tid=tid),
        host(f"shardscope/training_step/{training_step}", ts + 1, dur - 2, tid=tid),
    ]


class Builder:
    """Append events and remember the original index of each."""

    def __init__(self):
        self.events = []

    def add(self, event):
        self.events.append(event)
        return len(self.events) - 1

    def extend(self, events):
        return [self.add(event) for event in events]


def write_run(make_run, traces: dict[str, list], mappings, *, steps=4, environment=None):
    rows = "step,loss,step_time_ms,data_residency,synchronized\n" + "".join(
        f"{index},1.0,10.0,host,True\n" for index in range(steps)
    )
    directory = make_run(environment=environment or CUDA_ENVIRONMENT, steps_csv=rows, summary=None)
    for name, events in traces.items():
        (directory / name).write_text(json.dumps({"traceEvents": events, "synthetic": True}))
    (directory / "map.json").write_text(
        json.dumps(
            {
                "mappings": [
                    dict(trace_file=file, event_index=index, training_step=training_step)
                    for file, index, training_step in mappings
                ]
            }
        )
    )
    return directory


def analyze(directory):
    return analyze_run(directory, rank=0, step_map_file="map.json")


def row(payload, training_step):
    return next(r for r in payload["steps"] if r["training_step"] == training_step)


def basic_trace():
    """Step 1 on [1000, 2000) with two launches; step 2 on [3000, 4000) with one.

    Kernel ``b`` runs after step 1's boundary has ended, kernel ``c`` overlaps
    ``a`` on the same device, and the kernels are listed before their launches
    as Kineto does.
    """
    b = Builder()
    s1 = b.extend(step(1, 1000, 1000))[0]
    s2 = b.extend(step(2, 3000, 1000))[0]
    k = {
        "a": b.add(kernel(1500, 100, 11)),
        "c": b.add(kernel(1550, 100, 12)),
        "b": b.add(kernel(2500, 200, 13)),
    }
    la = b.add(launch(1200, 11))
    lc = b.add(launch(1300, 12))
    lb = b.add(launch(1900, 13))
    return b.events, dict(s1=s1, s2=s2, k=k, la=la, lc=lc, lb=lb)


# --- attribution -----------------------------------------------------------------


def test_launches_tie_kernels_to_steps_including_kernels_after_the_step_ends(make_run):
    events, ix = basic_trace()
    directory = write_run(make_run, {TRACE: events}, [(TRACE, ix["s1"], 1), (TRACE, ix["s2"], 2)])
    payload = analyze(directory)
    result = payload["kernel_attribution"]["traces"][TRACE]
    assert result["kernel_events"] == {"total": 3, "attributed": 3, "unmatched": 0, "ambiguous": 0}
    assert result["launch_events"]["attributed"] == 3
    assert result["launch_lookup"] == [
        dict(
            correlation=11,
            training_step=1,
            launch_event_index=ix["la"],
            boundary_event_index=ix["s1"],
        ),
        dict(
            correlation=12,
            training_step=1,
            launch_event_index=ix["lc"],
            boundary_event_index=ix["s1"],
        ),
        dict(
            correlation=13,
            training_step=1,
            launch_event_index=ix["lb"],
            boundary_event_index=ix["s1"],
        ),
    ]
    late = next(r for r in result["attributed"] if r["kernel_event_index"] == ix["k"]["b"])
    assert late["training_step"] == 1 and late["launch_event_index"] == ix["lb"]
    assert events[ix["k"]["b"]]["ts"] > events[ix["s1"]]["ts"] + events[ix["s1"]]["dur"]

    metric = row(payload, 1)["measurements"][METRIC]
    # a [1500,1600) and c [1550,1650) overlap: union 150us, not 200; plus b 200us.
    assert metric["status"] == "recorded"
    assert metric["value_ms"] == pytest.approx(0.35)
    assert metric["basis"] == "device_interval_union"
    assert metric["evidence"] == [
        {"file": TRACE, "event_index": ix["k"][name]} for name in ("a", "c", "b")
    ]
    assert metric["scope"] == f"rank=0; training_step=1; trace={TRACE}; device=0"
    assert metric["coverage"]["completeness"] == "complete"
    assert (
        "not utilization" in metric["reason"] and "not total device busy time" in metric["reason"]
    )
    assert "every kernel event in this trace was attributed" in metric["reason"]
    # Step 2 launched nothing: not measured, and never zero.
    other = row(payload, 2)["measurements"][METRIC]
    assert other["status"] == "not_recorded" and other["value_ms"] is None
    assert "no kernel event attributed" in other["reason"]
    assert other["coverage"]["step_kernel_events"] == 0
    # Unmapped steps say so; the four complete-activity device metrics stay unavailable.
    assert row(payload, 0)["measurements"][METRIC]["reason"] == "no explicit mapped step boundary"
    for name in ("device_compute_ms", "device_busy_ms"):
        assert row(payload, 1)["measurements"][name]["status"] == "not_recorded"
    assert (
        "device_attributed_kernel_ms"
        in row(payload, 1)["measurements"]["device_compute_ms"]["reason"]
    )


def test_kernels_are_never_attributed_by_timestamp_containment(make_run):
    b = Builder()
    s1 = b.extend(step(1, 1000, 1000))[0]
    inside = b.add(kernel(1500, 100, None))  # inside the step window, no id
    orphan = b.add(kernel(1600, 100, 99))  # id matches no host event
    directory = write_run(make_run, {TRACE: b.events}, [(TRACE, s1, 1)])
    payload = analyze(directory)
    result = payload["kernel_attribution"]["traces"][TRACE]
    assert result["kernel_events"] == {"total": 2, "attributed": 0, "unmatched": 2, "ambiguous": 0}
    kinds = {r["kernel_event_index"]: r["kind"] for r in result["unmatched"]}
    assert kinds == {inside: "missing_correlation", orphan: "no_host_api_event"}
    metric = row(payload, 1)["measurements"][METRIC]
    assert metric["status"] == "not_recorded" and metric["value_ms"] is None
    assert "2 of 2 kernel event(s) in this trace unattributed" in metric["reason"]
    assert metric["coverage"]["completeness"] == "none"
    assert any("missing_correlation" in w and f"event[{inside}]" in w for w in payload["warnings"])


def test_duplicate_correlation_ids_are_conflicts_not_choices(make_run):
    b = Builder()
    s1 = b.extend(step(1, 1000, 1000))[0]
    k = b.add(kernel(1500, 100, 11))
    first = b.add(launch(1200, 11))
    second = b.add(launch(1300, 11))
    directory = write_run(make_run, {TRACE: b.events}, [(TRACE, s1, 1)])
    payload = analyze(directory)
    result = payload["kernel_attribution"]["traces"][TRACE]
    assert result["launch_events"]["duplicate_correlation"] == 2
    assert result["launch_events"]["attributed"] == 0
    assert result["launch_lookup"] == []
    assert result["excluded_launches"] == [
        {"kind": "duplicate_correlation", "launch_event_indices": [first, second]}
    ]
    (record,) = result["ambiguous"]
    assert record["kernel_event_index"] == k
    assert record["kind"] == "conflicting_host_events"
    assert record["host_event_indices"] == [first, second]
    assert "no candidate is chosen" in record["reason"]
    assert row(payload, 1)["measurements"][METRIC]["value_ms"] is None
    assert any("1 ambiguous" in w for w in payload["warnings"])


def test_correlation_ids_are_looked_up_per_trace(make_run):
    """The same ids in two capture cycles resolve inside their own trace only."""
    first, ix1 = basic_trace()
    b = Builder()
    s3 = b.extend(step(3, 1000, 1000))[0]
    k = b.add(kernel(1500, 50, 11))  # id 11 again, a different kernel
    b.add(launch(1200, 11))
    directory = write_run(
        make_run,
        {TRACE: first, "trace-00001.json": b.events},
        [(TRACE, ix1["s1"], 1), (TRACE, ix1["s2"], 2), ("trace-00001.json", s3, 3)],
    )
    payload = analyze(directory)
    traces = payload["kernel_attribution"]["traces"]
    assert traces[TRACE]["kernel_events"]["attributed"] == 3
    assert traces["trace-00001.json"]["kernel_events"]["attributed"] == 1
    assert traces["trace-00001.json"]["attributed"][0]["kernel_event_index"] == k
    assert row(payload, 3)["measurements"][METRIC]["value_ms"] == pytest.approx(0.05)
    assert row(payload, 3)["measurements"][METRIC]["evidence"] == [
        {"file": "trace-00001.json", "event_index": k}
    ]
    assert row(payload, 1)["measurements"][METRIC]["value_ms"] == pytest.approx(0.35)


def test_launches_outside_or_inside_several_mapped_steps_are_not_attributed(make_run):
    b = Builder()
    s1 = b.extend(step(1, 1000, 1000))[0]
    before = b.add(kernel(900, 10, 21))
    other_thread = b.add(kernel(1500, 10, 22))
    straddle = b.add(kernel(2100, 10, 23))
    b.add(launch(500, 21))  # before any mapped step
    autograd = b.add(launch(1200, 22, tid=AUTOGRAD))  # inside the window, other thread
    b.add(launch(1995, 23))  # [1995, 2005) crosses the boundary end
    # Overlapping mapped boundaries on the same thread: step 2 nests inside
    # step 3. Neither carries an identity marker, which analysis discloses.
    s3 = b.add(host("shardscope/step", 3000, 1000))
    s2 = b.add(host("shardscope/step", 3100, 200))
    nested = b.add(kernel(3500, 10, 24))
    nested_launch = b.add(launch(3150, 24))
    directory = write_run(
        make_run, {TRACE: b.events}, [(TRACE, s1, 1), (TRACE, s2, 2), (TRACE, s3, 3)]
    )
    payload = analyze(directory)
    result = payload["kernel_attribution"]["traces"][TRACE]
    assert result["launch_events"] == {
        "attributed": 0,
        "outside_mapped_step": 3,
        "ambiguous_boundary": 1,
        "duplicate_correlation": 0,
        "missing_correlation": 0,
    }
    unmatched = {r["kernel_event_index"]: r for r in result["unmatched"]}
    assert set(unmatched) == {before, other_thread, straddle}
    assert all(r["kind"] == "launch_outside_mapped_step" for r in unmatched.values())
    assert f"tid={AUTOGRAD}" in unmatched[other_thread]["reason"]
    assert unmatched[other_thread]["launch_event_index"] == autograd
    (ambiguous,) = result["ambiguous"]
    assert ambiguous["kernel_event_index"] == nested
    assert ambiguous["kind"] == "launch_in_several_boundaries"
    assert ambiguous["candidate_training_steps"] == [2, 3]
    assert ambiguous["launch_event_index"] == nested_launch
    for training_step in (1, 2, 3):
        assert row(payload, training_step)["measurements"][METRIC]["value_ms"] is None
    thread_warning = next(
        w for w in payload["warnings"] if "kernel-launch events outside_mapped_step" in w
    )
    assert f"tid={AUTOGRAD}" in thread_warning and f"tid={MAIN}" in thread_warning


def test_several_kernels_may_share_one_unambiguous_launch(make_run):
    b = Builder()
    s1 = b.extend(step(1, 1000, 1000))[0]
    k1 = b.add(kernel(1500, 100, 11))
    k2 = b.add(kernel(1700, 100, 11))
    la = b.add(launch(1200, 11))
    directory = write_run(make_run, {TRACE: b.events}, [(TRACE, s1, 1)])
    payload = analyze(directory)
    result = payload["kernel_attribution"]["traces"][TRACE]
    assert [r["kernel_event_index"] for r in result["attributed"]] == [k1, k2]
    assert all(r["launch_event_index"] == la for r in result["attributed"])
    assert result["launch_events"]["attributed"] == 1
    metric = row(payload, 1)["measurements"][METRIC]
    assert metric["value_ms"] == pytest.approx(0.2)
    assert metric["coverage"] == {
        "step_kernel_events": 2,
        "step_launch_events": 1,
        "devices": [0],
        "trace_kernel_events": 2,
        "trace_attributed_kernel_events": 2,
        "trace_unattributed_kernel_events": 0,
        "completeness": "complete",
    }


def test_separate_devices_are_never_unioned_together(make_run):
    b = Builder()
    s1 = b.extend(step(1, 1000, 1000))[0]
    d0 = b.add(kernel(1500, 100, 11, device=0, stream=7))
    d0b = b.add(kernel(1550, 100, 12, device=0, stream=8))  # overlaps d0, other stream
    d1 = b.add(kernel(1500, 300, 13, device=1, stream=7))
    b.add(launch(1200, 11))
    b.add(launch(1210, 12))
    b.add(launch(1220, 13))
    environment = copy.deepcopy(CUDA_ENVIRONMENT)
    environment["device_count"] = 2
    directory = write_run(make_run, {TRACE: b.events}, [(TRACE, s1, 1)], environment=environment)
    payload = analyze(directory)
    result = payload["kernel_attribution"]["traces"][TRACE]
    (entry,) = result["steps"]
    assert entry["per_device"] == [
        {
            "device": 0,
            "attributed_kernel_ms": pytest.approx(0.15),
            "kernel_event_indices": [d0, d0b],
            "launch_event_indices": [5, 6],
            "streams": ["7", "8"],
        },
        {
            "device": 1,
            "attributed_kernel_ms": pytest.approx(0.3),
            "kernel_event_indices": [d1],
            "launch_event_indices": [7],
            "streams": ["7"],
        },
    ]
    metric = row(payload, 1)["measurements"][METRIC]
    assert metric["status"] == "not_recorded" and metric["value_ms"] is None
    assert "span devices [0, 1]" in metric["reason"] and "not combined" in metric["reason"]
    assert metric["coverage"]["devices"] == [0, 1]


def test_gpu_annotations_copies_and_unsupported_forms_are_not_kernels(make_run):
    b = Builder()
    s1 = b.extend(step(1, 1000, 1000))[0]
    real = b.add(kernel(1500, 100, 11))
    mirror = b.add(kernel(1500, 900, 11, cat="gpu_user_annotation", name="shardscope/forward"))
    copy_event = b.add(kernel(1600, 50, 12, cat="gpu_memcpy", name="Memcpy HtoD"))
    memset = b.add(kernel(1700, 5, 13, cat="gpu_memset", name="Memset (Device)"))
    driver = b.add(kernel(1800, 100, 14))
    exc = b.add(kernel(1900, 100, 15))
    b.add(launch(1200, 11))
    b.add(launch(1210, 12, name="cudaMemcpyAsync"))
    b.add(launch(1220, 13, name="cudaMemsetAsync"))
    b.add(launch(1230, 14, name="cuLaunchKernel", cat="cuda_driver"))
    b.add(launch(1240, 15, name="cudaLaunchKernelExC"))
    assert not is_kernel_event(b.events[mirror]) and not is_kernel_event(b.events[copy_event])
    assert not is_kernel_event(b.events[memset])
    assert not is_supported_launch(b.events[-2]) and is_supported_launch(b.events[-1])
    directory = write_run(make_run, {TRACE: b.events}, [(TRACE, s1, 1)])
    payload = analyze(directory)
    result = payload["kernel_attribution"]["traces"][TRACE]
    assert result["kernel_events"]["total"] == 3  # real, driver-launched, ExC-launched
    assert [r["kernel_event_index"] for r in result["attributed"]] == [real, exc]
    (unsupported,) = result["unmatched"]
    assert unsupported["kernel_event_index"] == driver
    assert unsupported["kind"] == "unsupported_host_event"
    assert "cuda_driver:cuLaunchKernel" in unsupported["reason"]
    metric = row(payload, 1)["measurements"][METRIC]
    # The 900us annotation mirror and the copies never enter the union.
    assert metric["value_ms"] == pytest.approx(0.2)
    assert metric["coverage"]["completeness"] == "partial"
    assert "1 of 3 kernel event(s) in this trace unattributed" in metric["reason"]
    assert row(payload, 1)["measurements"]["device_h2d_ms"]["status"] == "not_recorded"


def test_missing_device_id_withholds_the_kernel_from_any_device_union(make_run):
    b = Builder()
    s1 = b.extend(step(1, 1000, 1000))[0]
    k = b.add(kernel(1500, 100, 11, device=None))
    b.add(launch(1200, 11))
    directory = write_run(make_run, {TRACE: b.events}, [(TRACE, s1, 1)])
    result = analyze(directory)["kernel_attribution"]["traces"][TRACE]
    assert result["unmatched"][0]["kernel_event_index"] == k
    assert result["unmatched"][0]["kind"] == "missing_device"
    assert result["steps"] == []


def test_cpu_only_environment_reports_not_observed(make_run):
    events, ix = basic_trace()
    directory = write_run(
        make_run, {TRACE: events}, [(TRACE, ix["s1"], 1)], environment=GOOD_ENVIRONMENT
    )
    payload = analyze(directory)
    metric = row(payload, 1)["measurements"][METRIC]
    assert metric["status"] == "not_observed" and metric["value_ms"] is None
    assert metric["evidence"] == []
    assert "no CUDA device" in metric["reason"]
    # The attribution block still records what the trace contained.
    assert payload["kernel_attribution"]["traces"][TRACE]["kernel_events"]["attributed"] == 3


def test_attribute_trace_is_deterministic_plain_json(tmp_path):
    events, ix = basic_trace()
    path = tmp_path / TRACE
    path.write_text(json.dumps({"traceEvents": events}))
    trace = parse_trace(path)
    result = attribute_trace(trace, [(1, ix["s1"]), (2, ix["s2"])])
    assert json.loads(json.dumps(result)) == result
    assert attribute_trace(trace, [(1, ix["s1"]), (2, ix["s2"])]) == result
    with pytest.raises(ArtifactError, match="duplicate training step"):
        attribute_trace(trace, [(1, ix["s1"]), (1, ix["s2"])])
    # analyze_traces alone still returns host rows without the attributed metric.
    rows, _ = analyze_traces((trace,), rank=0, step_map=(StepMapping(TRACE, ix["s1"], 1),))
    assert METRIC not in rows[0]["measurements"]


# --- persistence and validation -------------------------------------------------


@pytest.fixture
def saved(make_run):
    events, ix = basic_trace()
    directory = write_run(make_run, {TRACE: events}, [(TRACE, ix["s1"], 1), (TRACE, ix["s2"], 2)])
    payload = analyze(directory)
    write_analysis(directory, payload)
    return directory, payload, ix


def test_saved_analysis_round_trips_through_load_validate_and_json_report(saved):
    directory, payload, _ = saved
    run = load_run(directory)
    assert run.analysis == payload
    assert validate_run(directory).ok
    restored = RunArtifacts.from_dict(json.loads(render_json(run)))
    assert restored == run
    assert (
        json.loads(render_json(run))["analysis"]["kernel_attribution"]
        == payload["kernel_attribution"]
    )
    assert render_steps_csv(run).startswith("step,loss,")
    states = {s.field: s for s in field_statuses(run)}
    assert states[METRIC].state == "recorded"
    assert "1/4 step(s) recorded" in states[METRIC].detail
    assert "Coverage: 3/3 kernel event(s) attributed across 1 analyzed trace(s)" in (
        states[METRIC].detail
    )
    assert "not utilization" in states[METRIC].detail
    assert states["gpu_activity"].state == "not_recorded"
    assert "not total device busy time" in states["gpu_activity"].detail
    assert states["evidence_refs"].state == "recorded"


def hide_kernel_b(payload, ix):
    """Consistently relabel kernel ``b`` as unmatched everywhere in the sidecar.

    Structure and counts stay coherent, so only recomputing the classification
    from the trace bytes can tell that ``b`` was attributable.
    """
    result = payload["kernel_attribution"]["traces"][TRACE]
    record = next(r for r in result["attributed"] if r["kernel_event_index"] == ix["k"]["b"])
    result["attributed"].remove(record)
    result["unmatched"].append(
        {
            "kernel_event_index": ix["k"]["b"],
            "correlation": 13,
            "kind": "no_host_api_event",
            "reason": "forged",
        }
    )
    result["kernel_events"].update(attributed=2, unmatched=1)
    device = result["steps"][0]["per_device"][0]
    device["kernel_event_indices"].remove(ix["k"]["b"])
    device["launch_event_indices"].remove(ix["lb"])
    device["attributed_kernel_ms"] = 0.15
    metric = payload["steps"][1]["measurements"][METRIC]
    metric["value_ms"] = 0.15
    metric["evidence"] = [e for e in metric["evidence"] if e["event_index"] != ix["k"]["b"]]


@pytest.mark.parametrize(
    "edit, match",
    [
        (
            lambda p, ix: p["kernel_attribution"]["traces"][TRACE]["attributed"][0].update(
                training_step=2, boundary_event_index=ix["s2"]
            ),
            "not attributed to it",
        ),
        (
            lambda p, ix: p["kernel_attribution"]["traces"][TRACE]["attributed"][0].update(
                training_step=2
            ),
            "not the mapped boundary",
        ),
        (
            lambda p, ix: p["steps"][1]["measurements"][METRIC].update(value_ms=0.36),
            "disagrees with the attribution block",
        ),
        (
            lambda p, ix: p["steps"][1]["measurements"][METRIC]["evidence"].pop(),
            "disagrees with the attribution block",
        ),
        (
            lambda p, ix: p["steps"][1]["measurements"][METRIC]["evidence"].__setitem__(
                0, {"file": TRACE, "event_index": ix["la"]}
            ),
            "not a device kernel event",
        ),
        (
            lambda p, ix: p["steps"][1]["measurements"][METRIC].update(basis="host_range"),
            "device_interval_union",
        ),
        (
            lambda p, ix: p["steps"][2]["measurements"][METRIC].update(
                status="recorded",
                value_ms=0.0,
                basis="device_interval_union",
                evidence=[{"file": TRACE, "event_index": ix["k"]["a"]}],
            ),
            "disagrees with the attribution block",
        ),
        (lambda p, ix: hide_kernel_b(p, ix), "disagrees with the trace"),
        (
            lambda p, ix: p["kernel_attribution"]["traces"][TRACE]["steps"][0]["per_device"][0][
                "kernel_event_indices"
            ].reverse(),
            "disagrees with the attribution block|per-step kernel groups",
        ),
        (lambda p, ix: p.pop("kernel_attribution"), "requires a kernel_attribution block"),
        (lambda p, ix: p["kernel_attribution"].update(schema=2), "unsupported schema"),
        (
            lambda p, ix: p["kernel_attribution"]["traces"].update({"other.json": {}}),
            "must name sources",
        ),
        (
            lambda p, ix: p["kernel_attribution"]["traces"][TRACE]["steps"][0]["per_device"][
                0
            ].update(attributed_kernel_ms=-1),
            "finite and nonnegative",
        ),
        (
            lambda p, ix: p["steps"][1]["measurements"][METRIC].pop("coverage"),
            "needs coverage",
        ),
        (
            lambda p, ix: p["steps"][1]["measurements"][METRIC].update(
                status="not_recorded", value_ms=None, basis=None, evidence=[]
            ),
            "withholds them",
        ),
    ],
)
def test_tampered_attribution_is_refused(saved, edit, match):
    directory, payload, ix = saved
    forged = copy.deepcopy(payload)
    edit(forged, ix)
    with pytest.raises(ArtifactError, match=match):
        validate_analysis(forged, directory=directory, steps={0, 1, 2, 3})


def test_structural_validation_without_sources_checks_relationships(saved):
    _, payload, ix = saved
    assert validate_analysis(copy.deepcopy(payload)) == payload
    forged = copy.deepcopy(payload)
    forged["kernel_attribution"]["traces"][TRACE]["attributed"][0]["training_step"] = 2
    with pytest.raises(ArtifactError, match="not the mapped boundary"):
        validate_analysis(forged)
    forged = copy.deepcopy(payload)
    forged["kernel_attribution"]["traces"][TRACE]["unmatched"].append(
        {"kernel_event_index": ix["k"]["a"], "kind": "no_host_api_event", "reason": "x"}
    )
    with pytest.raises(ArtifactError, match="both attributed and not"):
        validate_analysis(forged)


def test_edited_trace_is_stale_even_with_matching_digest_and_block_mismatch(saved):
    directory, payload, ix = saved
    trace_path = directory / TRACE
    trace = json.loads(trace_path.read_text())
    # Move launch 'b' onto the autograd thread: the digest changes (stale), and
    # a sidecar re-digested by hand still disagrees with the recomputation.
    trace["traceEvents"][ix["lb"]]["tid"] = AUTOGRAD
    trace_path.write_text(json.dumps(trace))
    with pytest.raises(ArtifactError, match="stale source"):
        load_run(directory)
    forged = copy.deepcopy(payload)
    forged["source"][TRACE] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    with pytest.raises(ArtifactError, match="disagrees with the trace"):
        validate_analysis(forged, directory=directory)
    fresh = analyze(directory)
    assert fresh["kernel_attribution"]["traces"][TRACE]["kernel_events"]["unmatched"] == 1
    assert row(fresh, 1)["measurements"][METRIC]["value_ms"] == pytest.approx(0.15)
    assert row(fresh, 1)["measurements"][METRIC]["coverage"]["completeness"] == "partial"


def test_legacy_sidecar_without_attribution_still_loads(saved):
    directory, payload, _ = saved
    legacy = copy.deepcopy(payload)
    del legacy["kernel_attribution"]
    for item in legacy["steps"]:
        del item["measurements"][METRIC]
    legacy["warnings"] = [w for w in legacy["warnings"] if "kernel" not in w]
    write_analysis(directory, legacy, overwrite=True)
    run = load_run(directory)
    assert run.analysis == legacy
    assert validate_run(directory).ok
    assert METRIC not in {s.field for s in field_statuses(run)}
    rendered = render_html(run)
    assert "predates kernel attribution" in rendered
    assert "attributed kernel ms" not in rendered


# --- rendering ---------------------------------------------------------------------


def test_html_shows_timing_coverage_and_evidence_from_persisted_results(saved, monkeypatch):
    directory, payload, ix = saved
    run = load_run(directory)
    import shardscope.attribution as module

    def refuse(*args, **kwargs):
        pytest.fail("rendering recomputed kernel attribution")

    monkeypatch.setattr(module, "attribute_trace", refuse)
    monkeypatch.setattr(module, "attribute_traces", refuse)
    monkeypatch.setattr(module, "interval_union_us", refuse)
    rendered = render_html(run)
    assert "<h2>Kernel attribution</h2>" in rendered
    assert "cuda_runtime:cudaLaunchKernel" in rendered
    assert "not utilization, not total step latency, not total device busy time" in rendered
    assert "complete; 3/3 kernel events in trace" in rendered
    assert (
        "0.35</td>" in rendered
        or repr(payload["steps"][1]["measurements"][METRIC]["value_ms"]) in rendered
    )
    assert f"{TRACE} event[{ix['k']['b']}]" in rendered
    assert rendered.count('href="#evidence-') >= 3
    assert "lower bound on attributed kernel execution" in rendered
    assert "device_attributed_kernel_ms" in rendered


def test_html_summarizes_unattributed_kernels_per_reason(make_run):
    b = Builder()
    s1 = b.extend(step(1, 1000, 1000))[0]
    for offset in range(7):
        b.add(kernel(1500 + offset, 1, None))
    directory = write_run(make_run, {TRACE: b.events}, [(TRACE, s1, 1)])
    write_analysis(directory, analyze(directory))
    rendered = render_html(load_run(directory))
    assert "missing_correlation" in rendered
    assert "7 event(s): event[2], event[3], event[4], event[5], event[6], ... 2 more" in rendered
    assert "unavailable</td>" in rendered
    assert "none; 0/7 kernel events in trace" in rendered
