"""Offline sidecar tests using labelled synthetic host ranges only."""

import copy
import hashlib
import json
import subprocess
import sys

import pytest

from shardscope.analysis import analyze_run, load_step_map, main, validate_analysis, write_analysis
from shardscope.errors import ArtifactError
from shardscope.rules import RuleConfig


def synthetic_inputs(directory, wait_us=4000):
    events = [
        dict(ph="X", name="ProfilerStep#90", ts=0, dur=10000, pid=1, tid=1),
        dict(ph="X", name="shardscope/step", ts=0, dur=10000, pid=1, tid=1),
        dict(ph="X", name="shardscope/data_wait", ts=100, dur=wait_us, pid=1, tid=1),
    ]
    (directory / "trace-synthetic.json").write_text(
        json.dumps({"traceEvents": events, "synthetic": True})
    )
    (directory / "synthetic-map.json").write_text(
        json.dumps(
            {"mappings": [dict(trace_file="trace-synthetic.json", event_index=0, training_step=1)]}
        )
    )


def build(directory, **kwargs):
    return analyze_run(directory, rank=0, step_map_file="synthetic-map.json", **kwargs)


def test_analysis_digests_units_identity_and_explicit_configuration(make_run):
    directory = make_run()
    synthetic_inputs(directory)
    payload = build(directory, config=RuleConfig(3))
    assert payload["schema_version"] == 1
    assert payload["rank"] == 0
    assert payload["steps"][1]["measurements"]["host_data_wait_ms"]["value_ms"] == 4
    assert payload["steps"][0]["measurements"]["host_data_wait_ms"]["value_ms"] is None
    assert payload["steps"][1]["boundary"]["event_index"] == 0
    assert payload["findings"][1]["status"] == "possible_input_wait"
    assert "does not establish GPU starvation" in payload["findings"][1]["interpretation"]
    for name, digest in payload["source"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    original = {name: (directory / name).read_bytes() for name in payload["source"]}
    target = write_analysis(directory, payload)
    assert json.loads(target.read_text()) == payload
    assert original == {name: (directory / name).read_bytes() for name in original}
    with pytest.raises(ArtifactError, match="exists"):
        write_analysis(directory, payload)
    write_analysis(directory, payload, overwrite=True)
    assert not list(directory.glob(".analysis-*.tmp"))


def test_missing_map_and_rules_remain_unknown(make_run):
    directory = make_run(summary=None)
    payload = analyze_run(directory, rank=4)
    assert len(payload["steps"]) == 3
    assert payload["findings"][0]["status"] == "unconfigured"
    assert any("no explicit step mapping" in text for text in payload["warnings"])
    assert "summary.json" not in payload["source"]
    with pytest.raises(ArtifactError, match="requires --rank"):
        analyze_run(directory)


def test_rank_consistency_and_step_identity(make_run):
    directory = make_run()
    synthetic_inputs(directory)
    envpath = directory / "environment.json"
    env = json.loads(envpath.read_text())
    env["extra"]["rank"] = 2
    envpath.write_text(json.dumps(env))
    assert analyze_run(directory)["rank"] == 2
    with pytest.raises(ArtifactError, match="disagrees"):
        build(directory)
    mapping = directory / "synthetic-map.json"
    value = json.loads(mapping.read_text())
    value["mappings"][0]["training_step"] = 99
    mapping.write_text(json.dumps(value))
    with pytest.raises(ArtifactError, match="absent from steps.csv"):
        analyze_run(directory, step_map_file=mapping.name)


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda p: p.update(schema_version=2), "schema_version"),
        (lambda p: p.update(rank=True), "rank"),
        (lambda p: p.update(steps={}), "steps must"),
        (lambda p: p.update(warnings=[1]), "warnings"),
        (lambda p: p.update(source={}), "source"),
        (lambda p: p["source"].update({"../escape": "a" * 64}), "filenames"),
        (lambda p: p["source"].update({"bad.json": "no"}), "SHA-256"),
        (lambda p: p["steps"][1].update(training_step=90), "absent from"),
        (lambda p: p["steps"][1]["boundary"].update(event_index=100), "event_index"),
        (lambda p: p["steps"][1]["boundary"].update(event_index=True), "event_index"),
        (lambda p: p["steps"][1]["boundary"].update(event_index=2), "step marker"),
        (lambda p: p["steps"][1]["boundary"].update(file="unknown.json"), "trace sources"),
        (
            lambda p: p["steps"][1]["measurements"]["host_data_wait_ms"]["evidence"][0].update(
                event_index=1
            ),
            "expected complete host",
        ),
        (
            lambda p: p["steps"][1]["measurements"]["host_data_wait_ms"].update(value_ms=-1),
            "finite nonnegative",
        ),
        (
            lambda p: p["steps"][0]["measurements"]["host_data_wait_ms"].update(value_ms=0),
            "unavailable measurement",
        ),
        (
            lambda p: p["steps"][1]["measurements"]["host_data_wait_ms"].update(
                basis="device_interval_union"
            ),
            "host_range",
        ),
        (lambda p: p["findings"][0].update(status="confirmed"), "findings disagree"),
    ],
)
def test_sidecar_rejects_corrupt_schema_and_evidence(make_run, edit, match):
    directory = make_run()
    synthetic_inputs(directory)
    payload = build(directory)
    edit(payload)
    with pytest.raises(ArtifactError, match=match):
        validate_analysis(payload, directory=directory, steps={0, 1, 2})


def test_stale_sources_and_atomic_failure_preserve_existing(make_run, monkeypatch):
    directory = make_run()
    synthetic_inputs(directory)
    payload = build(directory)
    target = write_analysis(directory, payload)
    original = target.read_bytes()
    import shardscope.analysis as module

    def fail(*args):
        raise OSError("synthetic publish failure")

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(ArtifactError, match="publish failure"):
        write_analysis(directory, payload, overwrite=True)
    assert target.read_bytes() == original
    assert not list(directory.glob(".analysis-*.tmp"))
    trace = directory / "trace-synthetic.json"
    trace.write_text(trace.read_text() + "\n")
    with pytest.raises(ArtifactError, match="stale source"):
        write_analysis(directory, payload, overwrite=True)
    assert target.read_bytes() == original


def test_mapping_validation_and_path_safety(make_run):
    directory = make_run()
    synthetic_inputs(directory)
    for content in [
        {},
        [],
        {"mappings": {}},
        {"mappings": [{}]},
        {"mappings": [dict(trace_file="../x", event_index=0, training_step=1)]},
    ]:
        path = directory / "bad-map.json"
        path.write_text(json.dumps(content))
        with pytest.raises(ArtifactError):
            load_step_map(path)
    for kwargs in [
        dict(step_map_file="../x"),
        dict(trace_files=("trace-synthetic.json",) * 2),
        dict(trace_files=("analysis.json",)),
    ]:
        with pytest.raises(ArtifactError):
            analyze_run(directory, rank=0, **kwargs)


def test_module_entrypoint_and_error_paths(make_run, capsys):
    directory = make_run()
    synthetic_inputs(directory)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shardscope.analysis",
            "--run",
            str(directory),
            "--rank",
            "0",
            "--step-map",
            "synthetic-map.json",
            "--loader-wait-threshold-ms",
            "3",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (directory / "analysis.json").is_file()
    assert main(["--run", str(directory), "--rank", "0"]) == 2
    assert "--overwrite" in capsys.readouterr().err
    assert (
        main(
            [
                "--run",
                str(directory),
                "--rank",
                "0",
                "--overwrite",
                "--trace",
                "trace-synthetic.json",
                "--step-map",
                "synthetic-map.json",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--run",
                str(directory),
                "--rank",
                "0",
                "--overwrite",
                "--loader-wait-threshold-ms",
                "nan",
            ]
        )
        == 2
    )


def test_output_aliases_never_replace_sources(make_run):
    directory = make_run()
    synthetic_inputs(directory)
    payload = build(directory)
    target = directory / "analysis.json"
    source = directory / "trace-synthetic.json"
    target.symlink_to(source)
    with pytest.raises(ArtifactError, match="symlink"):
        write_analysis(directory, payload, overwrite=True)
    target.unlink()
    import os

    os.link(source, target)
    with pytest.raises(ArtifactError, match="collision"):
        write_analysis(directory, payload, overwrite=True)


def test_serialized_validation_retains_unknown_fields(make_run):
    directory = make_run()
    synthetic_inputs(directory)
    payload = build(directory)
    payload["future_metadata"] = {"x": ["preserved"]}
    clone = copy.deepcopy(payload)
    assert validate_analysis(clone) == payload
