"""Sidecar loading, validation and rendering; all traces here are synthetic."""

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from shardscope.analysis import analyze_run, main, write_analysis
from shardscope.artifacts import RunArtifacts, load_run
from shardscope.availability import field_statuses
from shardscope.errors import ArtifactError, ReportError
from shardscope.report import render_html, render_json, render_steps_csv, write_report
from shardscope.rules import RuleConfig
from shardscope.validate import validate_run


def synthetic_sidecar(directory, *, name="trace-synthetic.json"):
    events = [
        dict(ph="X", name="shardscope/step", pid=1, tid=1, ts=0, dur=10000),
        dict(ph="X", name="shardscope/data_wait", pid=1, tid=1, ts=100, dur=5000),
    ]
    (directory / name).write_text(json.dumps({"traceEvents": events, "synthetic": True}))
    (directory / "map.json").write_text(
        json.dumps({"mappings": [dict(trace_file=name, event_index=0, training_step=0)]})
    )
    payload = analyze_run(directory, rank=0, step_map_file="map.json", config=RuleConfig(3))
    write_analysis(directory, payload)
    return payload


def test_load_validate_json_roundtrip_and_csv_compatibility(make_run):
    directory = make_run(summary=None)
    legacy = load_run(directory)
    assert legacy.analysis is None
    assert "analysis" not in legacy.to_dict()
    payload = synthetic_sidecar(directory)
    run = load_run(directory)
    assert run.analysis == payload
    result = validate_run(directory)
    assert result.ok
    assert result.to_dict()["analysis"] == {"present": True, "validated": True}
    assert set(payload["warnings"]) <= set(result.warnings)
    restored = RunArtifacts.from_dict(json.loads(render_json(run)))
    assert restored == run
    assert render_steps_csv(run) == render_steps_csv(legacy)
    assert run.summary is None
    states = {s.field: s for s in field_statuses(run)}
    assert states["host_data_wait_ms"].state == "recorded"
    assert "1/3" in states["host_data_wait_ms"].detail
    assert states["device_busy_ms"].state == "not_observed"
    assert states["evidence_refs"].state == "recorded"
    assert states["suspected_bottleneck"].state == "recorded"
    assert states["per_rank_step_ms"].state == "recorded"


def test_html_escapes_analysis_and_evidence_filenames(make_run):
    directory = make_run()
    name = "trace-<img onerror=alert(1)>.json"
    payload = synthetic_sidecar(directory, name=name)
    attack = '<script>alert("unsafe")</script>'
    payload["warnings"].append(attack)
    payload["steps"][0]["measurements"]["host_step_ms"]["scope"] = attack
    write_analysis(directory, payload, overwrite=True)
    rendered = render_html(load_run(directory))
    assert attack not in rendered
    assert "<img onerror" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&lt;img onerror" in rendered
    assert "original event-array index 1" in rendered
    assert 'href="#evidence-' in rendered
    assert "SHA-256" in rendered
    assert "Host and device durations are distinct, in milliseconds" in rendered
    assert "possible_input_wait" in rendered
    assert "threshold_ms" in rendered
    assert "%3Cimg%20onerror%3Dalert%281%29%3E" in rendered


@pytest.mark.parametrize(
    "source", ["steps.csv", "environment.json", "summary.json", "trace-synthetic.json", "map.json"]
)
def test_stale_analysis_is_never_a_current_diagnosis(make_run, source):
    directory = make_run()
    synthetic_sidecar(directory)
    run = load_run(directory)
    artifact = directory / source
    artifact.write_bytes(artifact.read_bytes() + b"\n")
    with pytest.raises(ArtifactError, match="stale source"):
        load_run(directory)
    result = validate_run(directory)
    assert not result.ok
    assert "stale source" in result.errors[0]
    for renderer in (render_html, render_json, render_steps_csv, field_statuses):
        with pytest.raises(ArtifactError, match="stale source"):
            renderer(run)
    # Explicit analysis, unlike rendering, can regenerate a stale sidecar.
    assert (
        main(
            [
                "--run",
                str(directory),
                "--rank",
                "0",
                "--step-map",
                "map.json",
                "--loader-wait-threshold-ms",
                "3",
                "--overwrite",
            ]
        )
        == 0
    )
    assert validate_run(directory).ok


@pytest.mark.parametrize(
    "content,match",
    [
        ("null", "expected analysis object"),
        ("{", "malformed"),
        ("[]", "expected object"),
        ('{"schema_version":99}', "schema_version"),
    ],
)
def test_invalid_sidecar_refuses_loading_and_validation(make_run, content, match):
    directory = make_run()
    (directory / "analysis.json").write_text(content)
    with pytest.raises(ArtifactError, match=match):
        load_run(directory)
    assert not validate_run(directory).ok
    assert load_run(directory, include_analysis=False).analysis is None


def test_analysis_directory_and_corrupt_serialized_evidence_are_refused(make_run):
    directory = make_run()
    (directory / "analysis.json").mkdir()
    with pytest.raises(ArtifactError, match="not a regular file"):
        load_run(directory)
    (directory / "analysis.json").rmdir()
    synthetic_sidecar(directory)
    data = load_run(directory).to_dict()
    data["analysis"]["steps"][0]["boundary"]["event_index"] = -1
    with pytest.raises(ArtifactError, match="event_index"):
        RunArtifacts.from_dict(data)


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
@pytest.mark.parametrize("name", ["analysis.json", "trace-synthetic.json", "map.json"])
def test_report_cannot_replace_analysis_or_its_sources(make_run, tmp_path, alias, name):
    directory = make_run()
    synthetic_sidecar(directory)
    run = load_run(directory)
    source = directory / name
    original = source.read_bytes()
    target = source
    if alias != "direct":
        target = tmp_path / "alias.html"
        if alias == "symlink":
            target.symlink_to(source)
        else:
            os.link(source, target)
    # Collision protection includes source names even after deserialization
    # without the optional directory listing.
    run = replace(run, artifact_names=())
    with pytest.raises(ReportError, match="refusing"):
        write_report(run, target, fmt="html")
    assert source.read_bytes() == original


def test_report_rendering_never_derives_new_timings(make_run, monkeypatch):
    directory = make_run()
    synthetic_sidecar(directory)
    run = load_run(directory)
    import shardscope.parse

    def refuse(*args, **kwargs):
        pytest.fail("report attempted new trace timing analysis")

    monkeypatch.setattr(shardscope.parse, "analyze_traces", refuse)
    monkeypatch.setattr(shardscope.parse, "interval_union_us", refuse)
    assert "Persisted trace analysis" in render_html(run)
    assert json.loads(render_json(run))["analysis"] == run.analysis
    assert render_steps_csv(run).startswith("step,loss,")


def test_reporting_imports_no_torch_and_installed_commands_use_sidecar(make_run):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import shardscope.report, shardscope.analysis; import sys; "
            'assert "torch" not in sys.modules',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    directory = make_run()
    synthetic_sidecar(directory)
    for args in [("validate", "--format", "json"), ("report", "--format", "json")]:
        command = [
            sys.executable,
            "-m",
            "shardscope.cli",
            args[0],
            "--run",
            str(directory),
            *args[1:],
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "analysis" in json.loads(result.stdout)


def test_unconfigured_analysis_and_invalid_presence_are_explicit(make_run):
    directory = make_run()
    payload = analyze_run(directory, rank=0)
    write_analysis(directory, payload)
    run = load_run(directory)
    status = next(s for s in field_statuses(run) if s.field == "suspected_bottleneck")
    assert status.state == "not_recorded"
    assert "unconfigured" in status.detail
    (directory / "analysis.json").write_text("{")
    result = validate_run(directory)
    assert not result.ok
    assert result.to_dict()["analysis"] == {"present": True, "validated": False}
