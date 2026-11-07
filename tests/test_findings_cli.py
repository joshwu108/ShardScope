"""The ``rules`` and ``findings`` commands, ``--threshold`` configuration and sidecar
round trips, over synthetic host ranges only.

Traces here are hand-written Chrome event arrays with invented microsecond
timings; nothing is a hardware measurement.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys

import pytest

from conftest import GOOD_ENVIRONMENT
from shardscope.analysis import analyze_run, validate_analysis, write_analysis
from shardscope.artifacts import load_run
from shardscope.availability import field_statuses
from shardscope.cli import main
from shardscope.errors import ArtifactError
from shardscope.findings import render_findings_text, summarize_findings
from shardscope.report import render_html, render_json
from shardscope.rules import LEGACY_RULES, RULE_ID, RULE_IDS, RuleConfig, evaluate_rules

TRACE = "trace-synthetic.json"
MAP = "map.json"
STEP_US = 10000


def marker(name, ts, dur):
    return dict(ph="X", name=name, pid=1, tid=1, ts=ts, dur=dur)


def write_trace(directory, *, h2d_us=(5000, 2000, 2000), data_wait_us=(None, None, None)):
    """Three mapped steps of 10 ms each, 20 ms apart, plus one unmapped step row."""
    events, mappings = [], []
    for index, (h2d, wait) in enumerate(zip(h2d_us, data_wait_us, strict=True)):
        ts = index * 2 * STEP_US
        mappings.append(dict(trace_file=TRACE, event_index=len(events), training_step=index))
        events.append(marker("shardscope/step", ts, STEP_US))
        events.append(marker(f"shardscope/training_step/{index}", ts + 1, STEP_US - 2))
        if h2d is not None:
            events.append(marker("shardscope/h2d", ts + 10, h2d))
        if wait is not None:
            events.append(marker("shardscope/data_wait", ts + 2, wait))
        events.append(marker("shardscope/forward", ts + 8000, 500))
    (directory / TRACE).write_text(json.dumps({"traceEvents": events, "synthetic": True}))
    (directory / MAP).write_text(json.dumps({"mappings": mappings}))


@pytest.fixture
def synthetic_run(make_run):
    environment = copy.deepcopy(GOOD_ENVIRONMENT)
    environment["extra"].update(synthetic=True, steps=4)
    rows = "step,loss,step_time_ms,data_residency,synchronized\n" + "".join(
        f"{index},1.0,10.0,host,True\n" for index in range(4)
    )
    directory = make_run(environment=environment, steps_csv=rows, summary=None)
    write_trace(directory)
    return directory


def analyze_cli(directory, *flags):
    return main(["analyze", "--run", str(directory), "--rank", "0", "--step-map", MAP, *flags])


THRESHOLDS = (
    "--threshold",
    "h2d_share_threshold=0.4",
    "--threshold",
    "step_variability_ratio=1.5",
)


# --- rules ---------------------------------------------------------------------------


def test_rules_text_and_json_describe_the_registry(capsys):
    assert main(["rules"]) == 0
    text = capsys.readouterr().out
    for rule_id in RULE_IDS:
        assert rule_id in text
    for key in list(RuleConfig.keys()):
        assert f"{key}=<" in text
    assert "never calibrated defaults" in text and "requires" in text
    assert main(["rules", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [rule["rule_id"] for rule in payload["rules"]] == list(RULE_IDS)
    keys = {item["key"] for rule in payload["rules"] for item in rule["config"]}
    assert keys == set(RuleConfig.keys())
    for rule in payload["rules"]:
        assert {"required_measurements", "statuses", "next_investigation", "scope"} <= rule.keys()


# --- analyze --threshold -------------------------------------------------------------


def test_threshold_items_configure_rules_and_persist_the_rule_set(synthetic_run, capsys):
    assert analyze_cli(synthetic_run, "--loader-wait-threshold-ms", "3", *THRESHOLDS) == 0
    capsys.readouterr()
    sidecar = json.loads((synthetic_run / "analysis.json").read_text())
    assert sidecar["rules"] == list(RULE_IDS)
    assert sidecar["rule_config"] == {
        "loader_wait_threshold_ms": 3,
        "h2d_share_threshold": 0.4,
        "step_variability_ratio": 1.5,
    }
    by_rule = {}
    for finding in sidecar["findings"]:
        by_rule.setdefault(finding["rule_id"], []).append(finding["status"])
    # Loader wait: no data_wait marker anywhere, so nothing is zero.
    assert by_rule[RULE_ID] == ["unavailable"] * 4
    assert by_rule["host_h2d_share"] == [
        "possible_host_copy_wait",
        "below_threshold",
        "below_threshold",
        "unavailable",
    ]
    assert by_rule["step_duration_variability"] == ["below_threshold"]
    h2d = next(f for f in sidecar["findings"] if f["rule_id"] == "host_h2d_share")
    assert h2d["comparison"]["value"] == 0.5 and h2d["comparison"]["denominator_ms"] == 10.0
    assert h2d["evidence"] == [
        {"file": TRACE, "event_index": 2},
        {"file": TRACE, "event_index": 0},
    ]


@pytest.mark.parametrize(
    "flags,match",
    [
        (["--threshold", "gpu_idle=0.5"], "unknown threshold key"),
        (["--threshold", "h2d_share_threshold=lots"], "plain finite number"),
        (["--threshold", "h2d_share_threshold=1.5"], "between 0 and 1"),
        (["--threshold", "step_variability_ratio=0.5"], "at least 1"),
        (["--threshold", "step_variability_min_steps=3"], "requires step_variability_ratio"),
        (
            [
                "--threshold",
                "loader_wait_threshold_ms=3",
                "--threshold",
                "loader_wait_threshold_ms=4",
            ],
            "given twice",
        ),
        (
            ["--loader-wait-threshold-ms", "3", "--threshold", "loader_wait_threshold_ms=4"],
            "given twice",
        ),
    ],
)
def test_invalid_threshold_items_fail_analysis_before_writing(synthetic_run, capsys, flags, match):
    assert analyze_cli(synthetic_run, *flags) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.startswith("error: ") and match in captured.err
    assert not (synthetic_run / "analysis.json").exists()


def test_repeating_the_same_threshold_value_is_not_a_conflict(synthetic_run, capsys):
    flags = ["--loader-wait-threshold-ms", "3", "--threshold", "loader_wait_threshold_ms=3"]
    assert analyze_cli(synthetic_run, *flags) == 0
    capsys.readouterr()
    sidecar = json.loads((synthetic_run / "analysis.json").read_text())
    assert sidecar["rule_config"] == {"loader_wait_threshold_ms": 3}


# --- findings ----------------------------------------------------------------------------


def test_findings_text_summarizes_triggered_steps_with_values_and_evidence(synthetic_run, capsys):
    assert analyze_cli(synthetic_run, *THRESHOLDS) == 0
    capsys.readouterr()
    assert main(["findings", "--run", str(synthetic_run)]) == 0
    text = capsys.readouterr().out
    assert "3 of 4 training step(s) mapped" in text
    assert "rules evaluated: " + ", ".join(RULE_IDS) in text
    assert "thresholds: h2d_share_threshold=0.4, step_variability_ratio=1.5" in text
    assert "host_loader_wait: unconfigured (set --threshold loader_wait_threshold_ms" in text
    assert (
        "host_h2d_share [h2d_share_threshold=0.4]: 2 below_threshold, "
        "1 possible_host_copy_wait, 1 unavailable"
    ) in text
    assert "possible_host_copy_wait: steps 0" in text
    assert (
        f"steps 0: host_h2d_ms / host_step_ms 0.5 >= 0.4; evidence {TRACE} event[2], "
        f"{TRACE} event[0]"
    ) in text
    assert "below_threshold: steps 1-2" in text and "unavailable: steps 3" in text
    assert "step_duration_variability [step_variability_ratio=1.5]: 1 below_threshold" in text
    assert "Missing measurements are not zero." in text
    assert "not a confirmed cause" in text


def test_findings_json_is_the_scripting_interface_and_filters_keep_counts(synthetic_run, capsys):
    assert analyze_cli(synthetic_run, *THRESHOLDS) == 0
    capsys.readouterr()
    assert main(["findings", "--run", str(synthetic_run), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rules_evaluated"] == list(RULE_IDS)
    assert payload["mapped_steps"] == 3 and payload["total_steps"] == 4 and payload["rank"] == 0
    assert set(payload["source"]) == {"steps.csv", "environment.json", TRACE, MAP}
    rules = {rule["rule_id"]: rule for rule in payload["rules"]}
    assert set(rules) == set(RULE_IDS)
    assert rules[RULE_ID]["configured"] is False and rules[RULE_ID]["counts"] == {"unconfigured": 1}
    h2d = rules["host_h2d_share"]
    assert h2d["configured"] and h2d["config"] == {"h2d_share_threshold": 0.4}
    assert h2d["counts"] == {"possible_host_copy_wait": 1, "below_threshold": 2, "unavailable": 1}
    assert h2d["triggered_steps"] == [0] and len(h2d["findings"]) == 4
    sidecar = json.loads((synthetic_run / "analysis.json").read_text())
    assert h2d["findings"] == [f for f in sidecar["findings"] if f["rule_id"] == "host_h2d_share"]

    assert (
        main(
            [
                "findings",
                "--run",
                str(synthetic_run),
                "--format",
                "json",
                "--rule",
                "host_h2d_share",
                "--status",
                "possible_host_copy_wait",
            ]
        )
        == 0
    )
    filtered = json.loads(capsys.readouterr().out)
    assert [rule["rule_id"] for rule in filtered["rules"]] == ["host_h2d_share"]
    (rule,) = filtered["rules"]
    assert rule["counts"] == h2d["counts"]
    assert [f["status"] for f in rule["findings"]] == ["possible_host_copy_wait"]


@pytest.mark.parametrize(
    "flags,match",
    [
        (["--rule", "gpu_idle"], "unknown rule id"),
        (["--status", "confirmed"], "unknown status"),
    ],
)
def test_findings_refuses_unknown_filters(synthetic_run, capsys, flags, match):
    assert analyze_cli(synthetic_run, *THRESHOLDS) == 0
    capsys.readouterr()
    assert main(["findings", "--run", str(synthetic_run), *flags]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and match in captured.err


def test_findings_without_a_sidecar_or_with_stale_sources_is_an_input_error(synthetic_run, capsys):
    assert main(["findings", "--run", str(synthetic_run)]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "has no analysis.json" in captured.err
    assert "shardscope analyze" in captured.err
    assert analyze_cli(synthetic_run, *THRESHOLDS) == 0
    capsys.readouterr()
    trace = synthetic_run / TRACE
    trace.write_bytes(trace.read_bytes() + b"\n")
    assert main(["findings", "--run", str(synthetic_run), "--format", "json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "stale source" in captured.err


def test_findings_exit_code_does_not_depend_on_what_was_found(synthetic_run, capsys):
    # A triggered h2d share and a run where nothing fires both exit 0.
    assert analyze_cli(synthetic_run, "--threshold", "h2d_share_threshold=0.1") == 0
    assert main(["findings", "--run", str(synthetic_run)]) == 0
    assert "possible_host_copy_wait: steps 0-2" in capsys.readouterr().out
    assert analyze_cli(synthetic_run, "--overwrite", "--threshold", "h2d_share_threshold=1") == 0
    assert main(["findings", "--run", str(synthetic_run)]) == 0
    text = capsys.readouterr().out
    assert "possible_host_copy_wait" not in text and "below_threshold: steps 0-2" in text


# --- persisted sidecars --------------------------------------------------------------------


def test_saved_analysis_round_trips_through_load_validate_and_reports(synthetic_run):
    config = RuleConfig(h2d_share_threshold=0.4, step_variability_ratio=1.5)
    payload = analyze_run(synthetic_run, rank=0, step_map_file=MAP, config=config)
    write_analysis(synthetic_run, payload)
    run = load_run(synthetic_run)
    assert run.analysis == payload and payload["rules"] == list(RULE_IDS)
    assert json.loads(render_json(run))["analysis"] == payload
    html = render_html(run)
    assert "host_h2d_share" in html and "possible_host_copy_wait" in html
    assert "step_duration_variability" in html
    status = next(s for s in field_statuses(run) if s.field == "suspected_bottleneck")
    assert status.state == "recorded" and "shardscope findings" in status.detail
    summary = summarize_findings(run)
    assert render_findings_text(summary).startswith("diagnostics for")
    # Offline structural validation accepts the serialized copy too.
    assert validate_analysis(copy.deepcopy(payload)) == payload


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda p: p.update(rules=["gpu_idle"]), "rules must list"),
        (lambda p: p.update(rules=[RULE_ID, RULE_ID]), "rules must list"),
        (lambda p: p.update(rules="host_loader_wait"), "rules must list"),
        (lambda p: p.update(rules=[RULE_ID]), "findings disagree"),
        (lambda p: p.update(rule_config={}), "invalid rule_config"),
        (lambda p: p["rule_config"].update(gpu_idle_fraction=0.5), "unsupported key"),
        (lambda p: p["rule_config"].update(h2d_share_threshold=2), "between 0 and 1"),
        (lambda p: p["rule_config"].update(h2d_share_threshold=0.5), "findings disagree"),
        (
            lambda p: next(f for f in p["findings"] if f["rule_id"] == "host_h2d_share").update(
                status="below_threshold"
            ),
            "findings disagree",
        ),
        (
            lambda p: next(f for f in p["findings"] if f["rule_id"] == "step_duration_variability")[
                "comparison"
            ].update(value=9),
            "findings disagree",
        ),
    ],
)
def test_corrupt_rule_fields_are_refused(synthetic_run, edit, match):
    config = RuleConfig(h2d_share_threshold=0.4, step_variability_ratio=1.5)
    payload = analyze_run(synthetic_run, rank=0, step_map_file=MAP, config=config)
    edit(payload)
    with pytest.raises(ArtifactError, match=match):
        validate_analysis(payload, directory=synthetic_run, steps={0, 1, 2, 3})


def test_legacy_sidecar_without_a_rule_set_still_loads_and_is_labelled(synthetic_run, capsys):
    payload = analyze_run(synthetic_run, rank=0, step_map_file=MAP, config=RuleConfig(3))
    legacy = {k: v for k, v in payload.items() if k != "rules"}
    legacy["findings"] = evaluate_rules(payload["steps"], RuleConfig(3), rules=LEGACY_RULES)
    assert [f["rule_id"] for f in legacy["findings"]] == [RULE_ID] * 4
    write_analysis(synthetic_run, legacy)
    run = load_run(synthetic_run)
    assert run.analysis == legacy and "rules" not in run.analysis
    assert main(["findings", "--run", str(synthetic_run)]) == 0
    text = capsys.readouterr().out
    assert "rules evaluated: host_loader_wait\n" in text
    assert "host_h2d_share: not evaluated (this analysis.json predates the rule" in text
    assert "step_duration_variability: not evaluated" in text
    assert main(["findings", "--run", str(synthetic_run), "--format", "json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["rules_evaluated"] == [RULE_ID]
    assert [(r["rule_id"], r["evaluated"]) for r in summary["rules"]] == [
        (RULE_ID, True),
        ("host_h2d_share", False),
        ("step_duration_variability", False),
    ]
    # A legacy unconfigured sidecar carries exactly one finding.
    unconfigured = {k: v for k, v in payload.items() if k != "rules"}
    unconfigured.update(
        rule_config=None, findings=evaluate_rules(payload["steps"], None, rules=LEGACY_RULES)
    )
    assert len(unconfigured["findings"]) == 1
    write_analysis(synthetic_run, unconfigured, overwrite=True)
    assert load_run(synthetic_run).analysis == unconfigured


def test_variability_triggers_on_a_slow_step_and_names_it(make_run, capsys):
    environment = copy.deepcopy(GOOD_ENVIRONMENT)
    environment["extra"].update(synthetic=True, steps=4)
    rows = "step,loss,step_time_ms,data_residency,synchronized\n" + "".join(
        f"{index},1.0,10.0,host,True\n" for index in range(4)
    )
    directory = make_run(environment=environment, steps_csv=rows, summary=None)
    write_trace(directory, h2d_us=(2000, 2000, 2000))
    trace = json.loads((directory / TRACE).read_text())
    # Stretch step 1's boundary and identity marker to 20 ms; other steps stay 10 ms.
    for event in trace["traceEvents"]:
        if event["ts"] in (2 * STEP_US, 2 * STEP_US + 1) and event["name"] != "shardscope/h2d":
            event["dur"] *= 2
    (directory / TRACE).write_text(json.dumps(trace))
    assert analyze_cli(directory, "--threshold", "step_variability_ratio=2") == 0
    assert main(["findings", "--run", str(directory), "--rule", "step_duration_variability"]) == 0
    text = capsys.readouterr().out
    assert "possible_step_variability: steps 0-2" in text
    assert "max(host_step_ms) / median(host_step_ms) 2 >= 2" in text
    assert "Training step 1 took 2.000 times the median" in text
    sidecar = json.loads((directory / "analysis.json").read_text())
    finding = next(f for f in sidecar["findings"] if f["rule_id"] == "step_duration_variability")
    assert finding["comparison"]["slowest_training_step"] == 1
    assert finding["comparison"]["median_ms"] == 10.0 and finding["comparison"]["max_ms"] == 20.0


def test_findings_and_rules_import_no_torch_and_run_as_console_script(synthetic_run):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import shardscope.findings, shardscope.rules, shardscope.cli; import sys; "
            'assert "torch" not in sys.modules',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    script = shutil.which("shardscope")
    if script is None:
        pytest.skip("installed shardscope console script is unavailable")
    assert analyze_cli(synthetic_run, *THRESHOLDS) == 0
    listed = subprocess.run([script, "rules", "--format", "json"], capture_output=True, text=True)
    assert listed.returncode == 0 and listed.stderr == ""
    assert [r["rule_id"] for r in json.loads(listed.stdout)["rules"]] == list(RULE_IDS)
    shown = subprocess.run(
        [script, "findings", "--run", str(synthetic_run), "--format", "json"],
        capture_output=True,
        text=True,
    )
    assert shown.returncode == 0, shown.stderr
    assert json.loads(shown.stdout)["rules_evaluated"] == list(RULE_IDS)
