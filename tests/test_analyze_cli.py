"""Installed explicit-map analysis workflows using synthetic host timings only."""

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

from conftest import GOOD_ENVIRONMENT
from test_analysis import synthetic_inputs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def console():
    script = shutil.which("shardscope")
    if script is None:
        pytest.skip("installed shardscope console script is unavailable")
    return [script]


@pytest.fixture
def synthetic_run(make_run):
    environment = copy.deepcopy(GOOD_ENVIRONMENT)
    environment["extra"].update(
        synthetic=True, description="Synthetic software inputs, not measurements"
    )
    directory = make_run(
        environment=environment,
        summary=None,
        steps_csv=(
            "step,loss,step_time_ms,data_residency,synchronized\n"
            "0,1.0,10.0,host,True\n1,1.0,10.0,host,True\n2,1.0,10.0,host,True\n"
        ),
    )
    synthetic_inputs(directory)
    return directory


def command(prefix, *args):
    return subprocess.run(
        [*prefix, *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
    )


def analyze(console, directory, *args):
    return command(console, "analyze", "--run", directory, "--rank", "0", *args)


def test_installed_analyze_matches_module_output_and_overwrite(console, synthetic_run):
    directory = synthetic_run
    flags = ["--step-map", "synthetic-map.json", "--loader-wait-threshold-ms", "3"]
    result = analyze(console, directory, *flags)
    target = directory / "analysis.json"
    assert (result.returncode, result.stdout, result.stderr) == (0, f"{target}\n", "")
    original = target.read_bytes()
    module = [sys.executable, "-m", "shardscope.analysis"]
    refused = analyze(console, directory, *flags)
    module_refused = command(module, "--run", directory, "--rank", "0", *flags)
    assert (refused.returncode, refused.stdout, refused.stderr) == (
        module_refused.returncode,
        module_refused.stdout,
        module_refused.stderr,
    )
    assert refused.returncode == 2 and refused.stdout == ""
    assert "--overwrite" in refused.stderr and target.read_bytes() == original
    overwritten = command(module, "--run", directory, "--rank", "0", *flags, "--overwrite")
    assert (overwritten.returncode, overwritten.stdout, overwritten.stderr) == (
        result.returncode,
        result.stdout,
        result.stderr,
    )
    assert target.read_bytes() == original
    console_overwritten = analyze(console, directory, *flags, "--overwrite")
    assert (
        console_overwritten.returncode,
        console_overwritten.stdout,
        console_overwritten.stderr,
    ) == (result.returncode, result.stdout, result.stderr)
    assert target.read_bytes() == original
    without_threshold = analyze(
        console, directory, "--step-map", "synthetic-map.json", "--overwrite"
    )
    assert without_threshold.returncode == 0, without_threshold.stderr
    refreshed = json.loads(target.read_text())
    assert refreshed["rule_config"] is None
    assert refreshed["findings"][0]["status"] == "unconfigured"
    assert refreshed["steps"][1]["measurements"]["host_data_wait_ms"]["value_ms"] == 4


def test_installed_help_and_required_arguments(console):
    help_result = command(console, "analyze", "--help")
    assert help_result.returncode == 0 and help_result.stderr == ""
    for flag in (
        "--run",
        "--rank",
        "--step-map",
        "--trace",
        "--loader-wait-threshold-ms",
        "--overwrite",
    ):
        assert flag in help_result.stdout
    for prefix, args in [
        (console, ["analyze"]),
        ([sys.executable, "-m", "shardscope.analysis"], []),
    ]:
        result = command(prefix, *args)
        assert result.returncode == 2 and result.stdout == ""
        assert "required" in result.stderr and "--run" in result.stderr


@pytest.mark.parametrize(
    "args,expected",
    [
        (["--step-map", "../escape.json"], "local filenames"),
        (["--trace", "missing.json"], "regular source file"),
        (["--trace", "trace-synthetic.json", "--trace", "trace-synthetic.json"], "duplicate trace"),
        (["--loader-wait-threshold-ms", "nan"], "finite and nonnegative"),
        (["--loader-wait-threshold-ms", "-1"], "finite and nonnegative"),
        (["--rank", "-1"], "nonnegative"),
    ],
)
def test_invalid_analysis_inputs_preserve_module_errors(console, synthetic_run, args, expected):
    module = [sys.executable, "-m", "shardscope.analysis"]
    left = analyze(console, synthetic_run, *args)
    right = command(module, "--run", synthetic_run, "--rank", "0", *args)
    assert (left.returncode, left.stdout, left.stderr) == (
        right.returncode,
        right.stdout,
        right.stderr,
    )
    assert left.returncode == 2 and left.stdout == "" and expected in left.stderr
    assert left.stderr.startswith("error: ") and "Traceback" not in left.stderr
    assert not (synthetic_run / "analysis.json").exists()


class PreBlocks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.blocks = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag == "pre":
            self.current = []

    def handle_data(self, data):
        if self.current is not None:
            self.current.append(data)

    def handle_endtag(self, tag):
        if tag == "pre" and self.current is not None:
            self.blocks.append("".join(self.current))
            self.current = None


def test_explicit_repeated_traces_through_validate_and_reports(console, synthetic_run):
    directory = synthetic_run
    second_name = "trace-synthetic-second.json"
    second = json.loads((directory / "trace-synthetic.json").read_text())
    second["traceEvents"][2]["dur"] = 2000
    (directory / second_name).write_text(json.dumps(second))
    mapping_path = directory / "synthetic-map.json"
    mapping = json.loads(mapping_path.read_text())
    mapping["mappings"].append(dict(trace_file=second_name, event_index=1, training_step=2))
    mapping_path.write_text(json.dumps(mapping))
    source_names = [
        "steps.csv",
        "environment.json",
        "synthetic-map.json",
        "trace-synthetic.json",
        second_name,
    ]
    original = {name: (directory / name).read_bytes() for name in source_names}
    result = analyze(
        console,
        directory,
        "--step-map",
        mapping_path.name,
        "--trace",
        "trace-synthetic.json",
        "--trace",
        second_name,
        "--loader-wait-threshold-ms",
        "3",
    )
    assert result.returncode == 0, result.stderr
    sidecar = json.loads((directory / "analysis.json").read_text())
    assert sidecar["source"] == {
        name: hashlib.sha256(raw).hexdigest() for name, raw in original.items()
    }
    assert sidecar["rule_config"] == {"loader_wait_threshold_ms": 3}
    assert sidecar["steps"][1]["measurements"]["host_data_wait_ms"]["value_ms"] == 4
    assert sidecar["steps"][2]["measurements"]["host_data_wait_ms"]["value_ms"] == 2
    assert sidecar["findings"][1]["comparison"] == dict(
        operator=">=", threshold_ms=3, value_ms=4, result=True
    )
    assert sidecar["findings"][2]["status"] == "below_threshold"
    missing = sidecar["steps"][0]["measurements"]["host_data_wait_ms"]
    assert missing["value_ms"] is None and missing["status"] == "not_recorded"
    validated = command(console, "validate", "--run", directory, "--format", "json")
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["analysis"] == dict(present=True, validated=True)
    reported = command(console, "report", "--run", directory, "--format", "json")
    assert reported.returncode == 0, reported.stderr
    assert json.loads(reported.stdout)["analysis"] == sidecar
    html_result = command(console, "report", "--run", directory, "--format", "html")
    assert html_result.returncode == 0, html_result.stderr
    parser = PreBlocks()
    parser.feed(html_result.stdout)
    comparisons = [json.loads(block) for block in parser.blocks if '"comparison"' in block]
    assert comparisons == [
        {"inputs": f["inputs"], "comparison": f["comparison"]} for f in sidecar["findings"]
    ]
    for finding in sidecar["findings"]:
        for ref in finding["evidence"]:
            assert f"{ref['file']} event[{ref['event_index']}]" in html_result.stdout
            assert f"original event-array index {ref['event_index']}" in html_result.stdout
    assert "Host and device durations are distinct" in html_result.stdout
    csv_result = command(console, "report", "--run", directory, "--format", "csv")
    assert csv_result.returncode == 0 and csv_result.stdout == original["steps.csv"].decode()
    assert original == {name: (directory / name).read_bytes() for name in source_names}


def test_unmapped_analysis_stays_unavailable_and_unconfigured(console, synthetic_run):
    result = analyze(console, synthetic_run)
    assert result.returncode == 0, result.stderr
    reported = command(console, "report", "--run", synthetic_run, "--format", "json")
    assert reported.returncode == 0, reported.stderr
    payload = json.loads(reported.stdout)["analysis"]
    assert payload["mapping_file"] is None and payload["rule_config"] is None
    assert payload["findings"][0]["status"] == "unconfigured"
    for row in payload["steps"]:
        metric = row["measurements"]["host_data_wait_ms"]
        assert metric["value_ms"] is None and metric["status"] == "not_recorded"


def test_stale_source_cli_refusal_and_explicit_refresh(console, synthetic_run):
    directory = synthetic_run
    flags = ["--step-map", "synthetic-map.json", "--loader-wait-threshold-ms", "3"]
    assert analyze(console, directory, *flags).returncode == 0
    sidecar = directory / "analysis.json"
    before = sidecar.read_bytes()
    trace = directory / "trace-synthetic.json"
    trace.write_bytes(trace.read_bytes() + b"\n")
    validated = command(console, "validate", "--run", directory, "--format", "json")
    assert validated.returncode == 2
    assert not json.loads(validated.stdout)["ok"] and "stale source" in validated.stdout
    for fmt in ("html", "json", "csv"):
        result = command(console, "report", "--run", directory, "--format", fmt)
        assert result.returncode == 2 and result.stdout == ""
        assert "stale source" in result.stderr
    assert sidecar.read_bytes() == before
    assert analyze(console, directory, *flags, "--overwrite").returncode == 0
    refreshed = json.loads(sidecar.read_text())
    assert refreshed["source"][trace.name] == hashlib.sha256(trace.read_bytes()).hexdigest()
    assert command(console, "validate", "--run", directory).returncode == 0


def test_legacy_run_renders_without_analysis(console, synthetic_run):
    validated = command(console, "validate", "--run", synthetic_run)
    assert validated.returncode == 0, validated.stderr
    for fmt in ("html", "json", "csv"):
        reported = command(console, "report", "--run", synthetic_run, "--format", fmt)
        assert reported.returncode == 0, reported.stderr
        if fmt == "json":
            assert "analysis" not in json.loads(reported.stdout)
        if fmt == "csv":
            assert reported.stdout == (synthetic_run / "steps.csv").read_text()
    assert not (synthetic_run / "analysis.json").exists()


def test_help_and_reporting_imports_do_not_load_torch(console):
    script = (
        "import sys; import shardscope.report; "
        "from shardscope.cli import build_parser; "
        "parser = build_parser(); "
        "assert 'torch' not in sys.modules; "
        "parser.print_help(); "
        "assert 'torch' not in sys.modules"
    )
    result = command([sys.executable, "-c", script])
    assert result.returncode == 0, result.stderr
    for args in [("--help",), ("report", "--help"), ("analyze", "--help")]:
        result = command(console, *args)
        assert result.returncode == 0 and result.stderr == ""
