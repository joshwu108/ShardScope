"""Small synthetic Chrome/Kineto inputs; these do not establish CUDA visibility."""

import gzip
import hashlib
import json

import pytest

from shardscope.errors import ArtifactError
from shardscope.parse import parse_trace


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
