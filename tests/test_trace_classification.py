"""Classification and warning aggregation against a CUDA-trace-shaped fixture.

``tests/fixtures/a100_like_cuda_trace.json`` reproduces the *structure* seen in
a real ``torch.profiler`` CUDA capture (Kineto schemaVersion 1): top-level
metadata, ``M`` metadata events, ``s``/``f`` flow events, ``i`` instants with
string pid/tid, host ``user_annotation`` markers, device-side
``gpu_user_annotation`` copies of the same ``shardscope/*`` names on a GPU
stream, and kernels/copies on that stream. Every timing in it is invented;
nothing here is a hardware measurement or a device-attribution claim.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from conftest import GOOD_ENVIRONMENT
from shardscope.analysis import analyze_run, validate_analysis, write_analysis
from shardscope.artifacts import load_run
from shardscope.errors import ArtifactError
from shardscope.parse import (
    DEVICE_METRICS,
    HOST_PHASES,
    StepMapping,
    analyze_traces,
    parse_trace,
)
from shardscope.report import render_html
from shardscope.validate import validate_run

FIXTURE = Path(__file__).parent / "fixtures" / "a100_like_cuda_trace.json"
TRACE = "trace-00000.json"

#: Original event-array indices of the host phase markers for training step 4,
#: as laid out in the fixture. Evidence must point at exactly these.
STEP_4_HOST_EVIDENCE = {
    "host_step_ms": 8,
    "host_data_wait_ms": 10,
    "host_h2d_ms": 11,
    "host_forward_ms": 17,
    "host_backward_ms": 24,
    "host_optimizer_ms": 27,
}


def fixture():
    return json.loads(FIXTURE.read_text())


def write_trace(directory, trace):
    (directory / TRACE).write_text(json.dumps(trace))


@pytest.fixture
def captured_run(make_run):
    """A run directory whose environment and trace follow the fixture."""
    data = fixture()
    environment = copy.deepcopy(GOOD_ENVIRONMENT)
    environment.update(data["environment"])
    environment["extra"].update(data["environment_extra"])
    rows = "step,loss,step_time_ms,data_residency,synchronized\n" + "".join(
        f"{step},1.0,10.0,host,True\n" for step in range(8)
    )
    directory = make_run(environment=environment, steps_csv=rows, summary=None)
    write_trace(directory, data["trace"])
    return directory


def edit_trace(directory, change):
    trace = json.loads((directory / TRACE).read_text())
    change(trace["traceEvents"])
    write_trace(directory, trace)


def edit_capture(directory, change):
    path = directory / "environment.json"
    environment = json.loads(path.read_text())
    change(environment["extra"]["capture"])
    path.write_text(json.dumps(environment))


def device_annotations(events):
    return [i for i, e in enumerate(events) if e.get("cat") == "gpu_user_annotation"]


def shape(warning):
    """A warning with its numbers blanked, to detect per-event repetition."""
    return re.sub(r"\d+", "N", warning)


# --- parsing: metadata, flow and instant events ------------------------------


def test_metadata_flow_and_instant_events_are_retained_and_summarized_once(captured_run):
    raw = fixture()["trace"]
    trace = parse_trace(captured_run / TRACE)
    # Every raw event, unknown field and original position survives.
    assert list(trace.events) == raw["traceEvents"]
    assert trace.metadata == {k: v for k, v in raw.items() if k != "traceEvents"}
    assert trace.metadata["deviceProperties"][0]["name"] == "synthetic-device"
    assert trace.metadata["baseTimeNanoseconds"] == 1700000000000000000
    # One line per retained non-X form, never one per event.
    forms = sorted({e["ph"] for e in raw["traceEvents"] if e["ph"] != "X"})
    assert forms == ["M", "f", "i", "s"]
    assert len(trace.warnings) == len(forms)
    by_form = {re.search(r"\('(\w)'\)", w).group(1): w for w in trace.warnings}
    assert set(by_form) == set(forms)
    assert by_form["M"].startswith("metadata")
    assert (
        "6 event(s): event[0], event[1], event[2], event[3], event[4], ... 1 more" in by_form["M"]
    )
    assert "2 event(s): event[45], event[46]" in by_form["i"]
    assert "flow-start" in by_form["s"] and "flow-end" in by_form["f"]
    # Kernels, copies and cpu_ops are complete X ranges: retained silently.
    assert not any("kernel" in w or "cpu_op" in w for w in trace.warnings)


# --- host measurement stays host-only -----------------------------------------


def test_host_phases_measured_from_host_markers_with_stable_indices(captured_run):
    payload = analyze_run(captured_run)
    events = fixture()["trace"]["traceEvents"]
    rows = {row["training_step"]: row for row in payload["steps"]}
    assert payload["mapping_source"] == "capture"
    assert [s for s, row in rows.items() if row["boundary"]] == [4, 5]
    assert rows[4]["boundary"] == {"file": TRACE, "event_index": 8}

    step4 = rows[4]["measurements"]
    expected_ms = {
        "host_step_ms": 0.88,
        "host_data_wait_ms": 0.05,
        "host_h2d_ms": 0.1,
        "host_forward_ms": 0.2,
        "host_backward_ms": 0.25,
        "host_optimizer_ms": 0.15,
    }
    for metric, value in expected_ms.items():
        assert step4[metric]["status"] == "recorded"
        assert step4[metric]["basis"] == "host_range"
        assert step4[metric]["value_ms"] == pytest.approx(value)
        index = STEP_4_HOST_EVIDENCE[metric]
        assert step4[metric]["evidence"] == [{"file": TRACE, "event_index": index}]
        assert events[index]["cat"] == "user_annotation"
    assert rows[5]["measurements"]["host_data_wait_ms"]["value_ms"] == pytest.approx(0.06)

    # No evidence anywhere resolves to a device-side annotation, a kernel or a copy.
    cited = {
        ref["event_index"]
        for row in payload["steps"]
        for metric in row["measurements"].values()
        for ref in metric["evidence"]
    }
    assert cited.isdisjoint(device_annotations(events))
    assert all(events[i]["pid"] == 4242 for i in cited)


def test_device_measurements_stay_unavailable_despite_device_annotations(captured_run):
    """GPU-side copies of the markers are evidence of nothing until correlated."""
    payload = analyze_run(captured_run)
    for row in payload["steps"]:
        for metric in DEVICE_METRICS:
            value = row["measurements"][metric]
            assert value["value_ms"] is None
            assert value["evidence"] == []
            assert value["basis"] is None
        if row["boundary"] is not None:
            assert row["measurements"]["device_busy_ms"]["status"] == "not_recorded"
            assert (
                "no validated device-to-step correlation"
                in (row["measurements"]["device_busy_ms"]["reason"])
            )
            # One recorded device; no collective could have run.
            assert row["measurements"]["device_collective_ms"]["status"] == "not_observed"


def test_device_annotation_on_the_host_thread_is_still_not_a_host_phase(tmp_path):
    """Classification is by category; timestamp containment proves nothing."""
    boundary = dict(ph="X", name="shardscope/step", pid=1, tid=1, ts=0, dur=1000)
    disguised = dict(
        ph="X",
        cat="gpu_user_annotation",
        name="shardscope/forward",
        pid=1,
        tid=1,
        ts=10,
        dur=100,
    )
    path = tmp_path / TRACE
    path.write_text(json.dumps({"traceEvents": [boundary, disguised]}))
    trace = parse_trace(path)
    rows, warnings = analyze_traces((trace,), rank=0, step_map=(StepMapping(TRACE, 0, 1),))
    forward = rows[0]["measurements"]["host_forward_ms"]
    assert forward["value_ms"] is None
    assert "missing complete shardscope/forward marker" in forward["reason"]
    assert any("device annotation (gpu_user_annotation)" in w and "event[1]" in w for w in warnings)
    assert not any("unmapped" in w for w in warnings)


# --- warning aggregation -------------------------------------------------------


def test_device_annotations_are_summarized_per_trace_not_reported_as_unmapped(captured_run):
    payload = analyze_run(captured_run)
    warnings = payload["warnings"]
    device = [w for w in warnings if "device annotation (gpu_user_annotation)" in w]
    assert len(device) == 3  # one line each for backward, forward, h2d
    h2d = next(w for w in device if "shardscope/h2d" in w)
    assert "2 event(s): event[40], event[44]" in h2d
    assert "not attributed to a step" in h2d
    assert not any("unmapped" in w or "ambiguous" in w for w in warnings)
    # ProfilerStep#N device annotations are not host phases and raise nothing.
    assert not any("ProfilerStep" in w for w in warnings)


def test_warnings_do_not_repeat_per_event(captured_run):
    payload = analyze_run(captured_run)
    warnings = payload["warnings"]
    # 47 raw events, 29 of them not host-measurable; the summary stays short.
    assert len(warnings) <= 14
    assert len({shape(w) for w in warnings}) == len(warnings)
    # The two standing caveats about unavailable device metrics and thresholds remain.
    assert any(w.startswith("device metrics unavailable") for w in warnings)
    assert any("thresholds are caller configuration" in w for w in warnings)


def test_partial_map_summarizes_unmapped_host_markers_with_indices(captured_run):
    (captured_run / "only-step-4.json").write_text(
        json.dumps({"mappings": [dict(trace_file=TRACE, event_index=8, training_step=4)]})
    )
    payload = analyze_run(captured_run, step_map_file="only-step-4.json")
    rows = {row["training_step"]: row for row in payload["steps"]}
    assert rows[5]["boundary"] is None
    assert (
        rows[5]["measurements"]["host_forward_ms"]["reason"] == "no explicit mapped step boundary"
    )
    unmapped = [w for w in payload["warnings"] if "unmapped" in w]
    assert len(unmapped) == len(HOST_PHASES)  # one per marker, not one per event
    forward = next(w for w in unmapped if "shardscope/forward" in w)
    assert "1 event(s): event[34]" in forward


def test_unsupported_host_form_is_reported_once_and_withholds_every_mapped_step(captured_run):
    def add_begin_end(events):
        events.append(dict(ph="B", name="shardscope/optimizer", pid=4242, tid=4242, ts=1900))
        events.append(dict(ph="E", pid=4242, tid=4242, ts=1950))

    edit_trace(captured_run, add_begin_end)
    payload = analyze_run(captured_run)
    for row in payload["steps"]:
        if row["boundary"] is not None:
            optimizer = row["measurements"]["host_optimizer_ms"]
            assert optimizer["value_ms"] is None
            assert "unsupported host marker form" in optimizer["reason"]
            assert optimizer["evidence"] == []
            assert row["measurements"]["host_forward_ms"]["status"] == "recorded"
    unsupported = [w for w in payload["warnings"] if "unsupported host marker form" in w]
    assert len(unsupported) == 1
    assert "shardscope/optimizer" in unsupported[0]
    # The E event carries no name, so only the named B event is a marker.
    assert "['B']" in unsupported[0]
    assert "1 event(s): event[47]" in unsupported[0]
    assert "host_optimizer_ms withheld" in unsupported[0]


# --- mapping validation ------------------------------------------------------------


def test_mapping_to_a_device_profiler_step_annotation_is_refused(captured_run):
    edit_capture(captured_run, lambda c: c["step_map"][0].update(step_event_index=41))
    with pytest.raises(ArtifactError, match="device annotation"):
        analyze_run(captured_run)


def test_mapping_that_contradicts_the_identity_marker_is_refused(captured_run):
    # Step 6 exists in steps.csv, so only the nested identity marker can object.
    edit_capture(captured_run, lambda c: c["step_map"][0].update(training_step=6))
    with pytest.raises(ArtifactError, match="identity marker names training step 4"):
        analyze_run(captured_run)
    (captured_run / "wrong.json").write_text(
        json.dumps({"mappings": [dict(trace_file=TRACE, event_index=8, training_step=6)]})
    )
    with pytest.raises(ArtifactError, match="identity marker names training step 4"):
        analyze_run(captured_run, step_map_file="wrong.json")


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda e: e[9].update(name="shardscope/training_step/four"), "malformed identity"),
        (
            lambda e: e.append({**e[9], "name": "shardscope/training_step/6"}),
            "several training steps",
        ),
    ],
)
def test_malformed_or_ambiguous_identity_markers_fail_analysis(captured_run, change, message):
    edit_trace(captured_run, change)
    with pytest.raises(ArtifactError, match=message):
        analyze_run(captured_run)


def test_boundary_without_identity_marker_is_accepted_and_disclosed(captured_run):
    edit_trace(captured_run, lambda e: e[9].update(name="not/an/identity"))
    payload = analyze_run(captured_run)
    assert payload["steps"][4]["measurements"]["host_forward_ms"]["status"] == "recorded"
    disclosed = [w for w in payload["warnings"] if "without a nested" in w]
    assert len(disclosed) == 1
    assert "1 event(s): event[8]" in disclosed[0]


def test_sidecar_cannot_cite_device_annotations_or_contradict_identity(captured_run):
    payload = analyze_run(captured_run)
    step4 = payload["steps"][4]
    # A boundary pointing at the GPU-side ProfilerStep#4 copy.
    forged = copy.deepcopy(payload)
    forged["steps"][4]["boundary"]["event_index"] = 41
    with pytest.raises(ArtifactError, match="device annotation"):
        validate_analysis(forged, directory=captured_run)
    # Host evidence replaced by the GPU-side forward copy inside the same step.
    forged = copy.deepcopy(payload)
    forged["steps"][4]["measurements"]["host_forward_ms"]["evidence"] = [
        {"file": TRACE, "event_index": 42}
    ]
    with pytest.raises(ArtifactError, match="expected complete host marker"):
        validate_analysis(forged, directory=captured_run)
    # Trace and sidecar edited together so digests agree, but the nested
    # identity marker now names another step than the one the sidecar claims.
    edit_trace(captured_run, lambda e: e[9].update(name="shardscope/training_step/6"))
    forged = copy.deepcopy(payload)
    forged["source"][TRACE] = hashlib.sha256((captured_run / TRACE).read_bytes()).hexdigest()
    assert forged["steps"][4] == step4
    with pytest.raises(ArtifactError, match="identity marker names step 6"):
        validate_analysis(forged, directory=captured_run)


# --- presentation -------------------------------------------------------------


def test_report_and_validate_present_the_aggregated_warnings(captured_run):
    payload = analyze_run(captured_run)
    write_analysis(captured_run, payload)
    run = load_run(captured_run)
    rendered = render_html(run)
    assert f"Analysis warnings ({len(payload['warnings'])})" in rendered
    assert "representative original event indices" in rendered
    assert rendered.count("device annotation (gpu_user_annotation)") == 3
    result = validate_run(captured_run)
    assert result.ok
    assert set(payload["warnings"]) <= set(result.warnings)
