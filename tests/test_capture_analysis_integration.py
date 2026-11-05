"""Producer-map integration using synthetic traces, without hardware claims."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import GOOD_ENVIRONMENT
from shardscope.analysis import analyze_run, validate_analysis, write_analysis
from shardscope.artifacts import load_run
from shardscope.errors import ArtifactError
from shardscope.report import render


@pytest.fixture
def captured_run(make_run):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "synthetic_capture_interface.json").read_text()
    )
    environment = copy.deepcopy(GOOD_ENVIRONMENT)
    environment["extra"].update(fixture["environment_extra"])
    environment["extra"]["synthetic"] = True
    rows = "step,loss,step_time_ms,data_residency,synchronized\n" + "".join(
        f"{step},1.0,10.0,host,True\n" for step in range(8)
    )
    directory = make_run(environment=environment, steps_csv=rows, summary=None)
    for name, trace in fixture["traces"].items():
        (directory / name).write_text(json.dumps(trace))
    return directory


def edit_capture(directory, change):
    path = directory / "environment.json"
    environment = json.loads(path.read_text())
    change(environment["extra"]["capture"])
    path.write_text(json.dumps(environment))


def test_embedded_map_matches_explicit_map_across_capture_cycles(captured_run):
    embedded = analyze_run(captured_run)
    capture = json.loads((captured_run / "environment.json").read_text())["extra"]["capture"]
    explicit = {
        "mappings": [
            {
                "trace_file": item["trace_file"],
                "event_index": item["step_event_index"],
                "training_step": item["training_step"],
            }
            for item in capture["step_map"]
        ]
    }
    (captured_run / "consumer-map.json").write_text(json.dumps(explicit))
    from_file = analyze_run(captured_run, step_map_file="consumer-map.json")
    assert embedded["steps"] == from_file["steps"]
    assert embedded["mapping_source"] == "capture"
    assert from_file["mapping_source"] == "file"
    assert [row["training_step"] for row in embedded["steps"] if row["boundary"]] == [2, 3, 6, 7]
    write_analysis(captured_run, embedded)
    run = load_run(captured_run)
    assert run.analysis["mapping_source"] == "capture"
    assert "host_forward_ms" in render(run, "json")
    assert "<!DOCTYPE html>" in render(run, "html")


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda c: c.update(step_map_schema=2), "step_map_schema"),
        (lambda c: c.update(step_map_schema=True), "step_map_schema"),
        (lambda c: c.update(step_map={}), "step_map type"),
        (lambda c: c["step_map"][0].update(rank=1), "rank"),
        (lambda c: c["step_map"][0].update(step_event_index=True), "step_event_index"),
        (lambda c: c["step_map"][0].update(step_event_index=-1), "step_event_index"),
        (lambda c: c["step_map"][0].update(step_event_index=999), "event_index"),
        (lambda c: c["step_map"][0].update(step_event_index=3), "step marker"),
        (lambda c: c["step_map"][0].update(trace_file="../escape.json"), "local filename"),
        (lambda c: c["step_map"][0].update(training_step=True), "training_step"),
        (lambda c: c["step_map"][1].update(training_step=2), "duplicate"),
        (lambda c: c["step_map"][1].update(step_event_index=1), "multiple training steps"),
    ],
)
def test_invalid_embedded_mapping_is_refused(captured_run, change, message):
    edit_capture(captured_run, change)
    with pytest.raises(ArtifactError, match=message):
        analyze_run(captured_run)


def test_explicit_file_takes_precedence_without_invalid_file_fallback(captured_run):
    edit_capture(captured_run, lambda c: c.update(step_map_schema=999))
    mapping = captured_run / "consumer-map.json"
    mapping.write_text(json.dumps({"mappings": []}))
    assert analyze_run(captured_run, step_map_file=mapping.name)["mapping_source"] == "file"
    mapping.write_text("{}")
    with pytest.raises(ArtifactError, match="mappings array"):
        analyze_run(captured_run, step_map_file=mapping.name)


def test_sidecar_assignments_must_match_embedded_map(captured_run):
    payload = analyze_run(captured_run)
    payload["steps"][2]["boundary"] = copy.deepcopy(payload["steps"][3]["boundary"])
    with pytest.raises(ArtifactError, match="disagrees with explicit step map"):
        validate_analysis(payload, directory=captured_run)


def test_embedded_mapping_change_invalidates_saved_analysis(captured_run):
    write_analysis(captured_run, analyze_run(captured_run))
    edit_capture(captured_run, lambda c: c["step_map"][0].update(training_step=1))
    with pytest.raises(ArtifactError, match="stale source environment.json"):
        load_run(captured_run)


def test_legacy_explicit_sidecar_without_source_label_still_validates(captured_run):
    (captured_run / "consumer-map.json").write_text(json.dumps({"mappings": []}))
    payload = analyze_run(captured_run, step_map_file="consumer-map.json")
    del payload["mapping_source"]
    assert validate_analysis(payload, directory=captured_run) is payload


def test_module_analyzes_embedded_map_without_external_file(captured_run):
    result = subprocess.run(
        [sys.executable, "-m", "shardscope.analysis", "--run", str(captured_run)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads((captured_run / "analysis.json").read_text())
    assert payload["mapping_source"] == "capture"
    assert payload["mapping_file"] is None
    assert payload["steps"][2]["boundary"] == {"file": "trace-00000.json", "event_index": 1}


@pytest.mark.parametrize("source", ["none", "unsupported", [], None])
def test_sidecar_cannot_disguise_embedded_mapping_source(captured_run, source):
    payload = analyze_run(captured_run)
    payload["mapping_source"] = source
    with pytest.raises(ArtifactError, match="mapping_source|provenance"):
        validate_analysis(payload, directory=captured_run)
