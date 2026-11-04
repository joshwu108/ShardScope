"""Small synthetic Chrome/Kineto inputs; these do not establish CUDA visibility."""

import gzip
import hashlib
import json

import pytest

from shardscope.errors import ArtifactError
from shardscope.parse import StepMapping, analyze_traces, interval_union_us, parse_trace


def synthetic_trace(tmp_path, events, **metadata):
    path = tmp_path / "synthetic.trace.json"
    path.write_text(json.dumps({"traceEvents": events, **metadata}))
    return path


def event(name="shardscope/step", ts=0, dur=10000, **extra):
    return dict(ph="X", name=name, ts=ts, dur=dur, pid=1, tid=2, **extra)


def test_structure_preserves_unknown_metadata_and_original_indices(tmp_path):
    events = [
        dict(ph="M", name="process_name", args={"name": "synthetic"}),
        event(),
        dict(ph="future", something=[1, 2]),
    ]
    path = synthetic_trace(tmp_path, events, displayTimeUnit="ms", unknown={"x": 1})
    trace = parse_trace(path)
    assert list(trace.events) == events
    assert trace.metadata["unknown"] == {"x": 1}
    assert trace.reference(1) == {"file": path.name, "event_index": 1}
    assert trace.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(trace.warnings) == 2
    with pytest.raises(ArtifactError, match="event_index"):
        trace.reference(-1)


def test_legacy_array_empty_and_gzip(tmp_path):
    path = tmp_path / "synthetic.trace.json.gz"
    path.write_bytes(gzip.compress(b"[]"))
    assert "empty trace" in parse_trace(path).warnings[0]
    path.write_bytes(gzip.compress(json.dumps([event()]).encode()))
    assert len(parse_trace(path).events) == 1
    path.write_bytes(b"broken gzip")
    with pytest.raises(ArtifactError, match="cannot read trace"):
        parse_trace(path)


@pytest.mark.parametrize(
    "raw",
    [
        "{",
        "{}",
        "null",
        '{"traceEvents":{}}',
        "[null]",
        "[{}]",
        '[{"ph":false}]',
        '[{"ph":"X"}]',
        '[{"ph":"M","ts":NaN}]',
    ],
)
def test_malformed_container_and_truncation(tmp_path, raw):
    path = tmp_path / "synthetic.json"
    path.write_text(raw)
    with pytest.raises(ArtifactError, match=str(path)):
        parse_trace(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ts", True),
        ("dur", -1),
        ("ts", "0"),
        ("dur", float("inf")),
        ("pid", []),
        ("name", None),
        ("args", []),
        ("cat", 3),
    ],
)
def test_bad_range_fields(tmp_path, field, value):
    item = event()
    item[field] = value
    with pytest.raises(ArtifactError, match="event|JSON"):
        parse_trace(synthetic_trace(tmp_path, [item]))


def test_unit_override_is_refused_but_display_unit_is_not_timestamp_unit(tmp_path):
    with pytest.raises(ArtifactError, match="timestampUnit"):
        parse_trace(synthetic_trace(tmp_path, [], timestampUnit="ns"))
    assert (
        parse_trace(synthetic_trace(tmp_path, [event()], displayTimeUnit="ns")).events[0]["dur"]
        == 10000
    )


def test_read_failure_and_unclosed_range(tmp_path):
    with pytest.raises(ArtifactError, match="cannot read trace"):
        parse_trace(tmp_path / "absent")
    begin = dict(ph="B", name="range", ts=0, pid=1, tid=2)
    with pytest.raises(ArtifactError, match="unclosed"):
        parse_trace(synthetic_trace(tmp_path, [begin]))
    end = dict(ph="E", ts=10, pid=1, tid=2)
    assert len(parse_trace(synthetic_trace(tmp_path, [begin, end])).warnings) == 2
    with pytest.raises(ArtifactError, match="unmatched"):
        parse_trace(synthetic_trace(tmp_path, [end]))


def test_interval_union_handles_nested_overlapping_adjacent_and_zero():
    assert interval_union_us([(10, 20), (0, 12), (3, 5), (20, 22), (30, 30)]) == 22
    assert interval_union_us([]) == 0
    for intervals in [[(1, 0)], [(0, True)], [(0, float("inf"))]]:
        with pytest.raises(ArtifactError, match="interval union"):
            interval_union_us(intervals)


def test_explicit_mapping_host_unions_and_missing_activity(tmp_path):
    events = [
        event("ProfilerStep#99"),
        event(),
        event("shardscope/data_wait", 100, 3000),
        event("shardscope/data_wait", 1000, 4000),
        event("shardscope/h2d", 5000, 1000),
        event("nccl_guessed_name", 0, 9000, cat="unknown_device"),
    ]
    trace = parse_trace(synthetic_trace(tmp_path, events))
    for rank in (0, 7):
        rows, warnings = analyze_traces(
            (trace,), rank=rank, step_map=(StepMapping(trace.path.name, 0, 42),)
        )
        row = rows[0]
        assert row["training_step"] == 42
        assert row["boundary"]["event_index"] == 0
        assert row["measurements"]["host_data_wait_ms"]["value_ms"] == 4.9
        assert row["measurements"]["host_h2d_ms"]["value_ms"] == 1
        assert row["measurements"]["host_backward_ms"]["value_ms"] is None
        assert row["measurements"]["device_collective_ms"]["value_ms"] is None
        assert row["measurements"]["host_data_wait_ms"]["evidence"][1]["event_index"] == 3
        assert any("multiple" in message for message in warnings)
    rows, warnings = analyze_traces((trace,), rank=0, step_map=())
    assert rows == []
    assert any("no explicit step mapping" in warning for warning in warnings)


def test_cycles_and_ambiguous_boundaries(tmp_path):
    trace = parse_trace(
        synthetic_trace(
            tmp_path, [event(), event(ts=2000), event("shardscope/data_wait", 3000, 1000)]
        )
    )
    rows, _ = analyze_traces(
        (trace,),
        rank=0,
        step_map=(StepMapping(trace.path.name, 0, 0), StepMapping(trace.path.name, 1, 1)),
    )
    assert all(row["measurements"]["host_data_wait_ms"]["value_ms"] is None for row in rows)
    other = tmp_path / "synthetic-cycle2.json"
    other.write_text(json.dumps([event(), event("shardscope/data_wait", 0, 100)]))
    second = parse_trace(other)
    rows, _ = analyze_traces(
        (trace, second),
        rank=1,
        step_map=(StepMapping(trace.path.name, 0, 3), StepMapping(other.name, 0, 8)),
        unavailable={"device_busy_ms": "CUDA unavailable in recorded environment"},
    )
    assert [row["training_step"] for row in rows] == [3, 8]
    assert rows[1]["measurements"]["host_data_wait_ms"]["value_ms"] == 0.1
    assert rows[0]["measurements"]["device_busy_ms"]["status"] == "not_observed"


@pytest.mark.parametrize(
    "rank,mappings",
    [
        (True, ()),
        (-1, ()),
        (0, (StepMapping("absent", 0, 1),)),
        (0, (StepMapping("synthetic.trace.json", 100, 1),)),
        (0, (StepMapping("synthetic.trace.json", 0, True),)),
        (0, (StepMapping("synthetic.trace.json", 0, 1), StepMapping("synthetic.trace.json", 0, 1))),
        (0, (StepMapping("synthetic.trace.json", 1, 1),)),
    ],
)
def test_invalid_mapping_is_not_inferred(tmp_path, rank, mappings):
    trace = parse_trace(synthetic_trace(tmp_path, [event(), event("unknown")]))
    with pytest.raises(ArtifactError):
        analyze_traces((trace,), rank=rank, step_map=mappings)
