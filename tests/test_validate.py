"""``shardscope validate``: say whether a run directory can be reported.

The command is the artifact reader exposed on its own. It reuses
:func:`shardscope.artifacts.load_run` and the availability statuses, so what
it accepts is exactly what ``report`` accepts, with one deliberate
difference: a step count that disagrees between ``summary.json`` and
``steps.csv`` is shown by ``report`` and is a *failure* here, because the
question validate answers is "can this directory be trusted as a recording".

Nothing is rendered, written or rewritten; the tests check that too.
"""

from __future__ import annotations

import json

import pytest

from conftest import GOOD_ENVIRONMENT, GOOD_STEPS_CSV, GOOD_SUMMARY
from shardscope.cli import PENDING_COMMANDS, main
from shardscope.errors import ShardScopeError, ValidationError
from shardscope.validate import ValidationResult, validate_run

HEADER = "step,loss,step_time_ms,data_residency,synchronized"


def _csv(*rows: str) -> str:
    return "\n".join((HEADER, *rows)) + "\n"


def _snapshot(run_dir):
    return {path.name: path.read_bytes() for path in sorted(run_dir.iterdir())}


class TestValidateRun:
    def test_a_valid_run_is_ok_with_no_errors(self, make_run):
        result = validate_run(make_run())

        assert result.ok is True
        assert result.errors == ()
        assert result.rows == 3
        assert result.step_range == (0, 2)

    def test_an_absent_summary_is_valid_and_described_as_absent(self, make_run):
        result = validate_run(make_run(summary=None))

        assert result.ok is True
        assert result.summary_present is False
        assert result.summary_steps is None
        assert "absent" in result.to_text()
        assert result.warnings == ()

    def test_a_step_count_disagreement_is_a_failure_not_a_warning(self, make_run):
        result = validate_run(make_run(summary={**GOOD_SUMMARY, "steps": 99.0}))

        assert result.ok is False
        assert len(result.errors) == 1
        assert "99" in result.errors[0] and "3 row" in result.errors[0]
        # Neither number is adjusted to match the other.
        assert result.summary_steps == 99
        assert result.rows == 3

    @pytest.mark.parametrize("payload", ['{"steps": "three"}', '{"steps": NaN}', '{"steps": 3.5}'])
    def test_an_unreadable_recorded_count_is_a_warning_not_agreement(self, make_run, payload):
        result = validate_run(make_run(summary=payload))

        assert result.ok is True
        assert result.summary_present is True
        assert result.summary_steps is None
        assert any(
            "steps" in warning and "cannot be checked" in warning for warning in result.warnings
        )

    def test_an_unrecorded_profiling_status_is_a_warning(self, make_run):
        extra = {k: v for k, v in GOOD_ENVIRONMENT["extra"].items() if k != "profiled"}

        result = validate_run(make_run(environment={**GOOD_ENVIRONMENT, "extra": extra}))

        assert result.ok is True
        assert result.profiled is None
        assert any("profiled" in warning for warning in result.warnings)

    def test_rows_that_disagree_about_how_they_were_measured_are_a_warning(self, make_run):
        run_dir = make_run(
            steps_csv=_csv("0,1.0,2.0,host,True", "1,1.0,2.0,device,False"),
            summary={**GOOD_SUMMARY, "steps": 2.0},
        )

        result = validate_run(run_dir)

        assert result.ok is True
        joined = " ".join(result.warnings)
        assert "synchronized" in joined
        assert "data_residency" in joined

    def test_a_malformed_run_reports_the_loaders_message(self, make_run):
        run_dir = make_run(steps_csv=_csv("0,x,2.0,host,True"))

        result = validate_run(run_dir)

        assert result.ok is False
        assert "row 2" in result.errors[0]
        assert "loss" in result.errors[0]
        assert result.rows is None

    def test_a_missing_directory_is_a_failure(self, tmp_path):
        result = validate_run(tmp_path / "absent")

        assert result.ok is False
        assert "no such run directory" in result.errors[0]

    def test_validation_reads_and_never_writes(self, make_run):
        run_dir = make_run("sound")
        damaged = make_run("damaged", summary={**GOOD_SUMMARY, "steps": 99.0})
        before = _snapshot(run_dir), _snapshot(damaged)

        validate_run(run_dir)
        validate_run(damaged)

        assert (_snapshot(run_dir), _snapshot(damaged)) == before

    def test_the_result_carries_the_availability_statuses_not_metrics(self, make_run):
        result = validate_run(make_run())
        payload = result.to_dict()

        fields = {entry["field"]: entry["state"] for entry in payload["availability"]}
        assert fields["step_time_ms"] == "recorded"
        assert fields["data_wait_ms"] == "not_recorded"
        # No statistic is computed or copied into the result: the summary's
        # recorded figures stay in summary.json, and nothing is derived.
        text = json.dumps(payload)
        for statistic in ("median_step_ms", "p95_step_ms", "min_step_ms", "max_step_ms"):
            assert statistic not in text
        assert "17.92" not in text  # the fixture's recorded max/first step time

    def test_the_json_view_is_strict_json_even_for_a_damaged_summary(self, make_run):
        # json.dumps spells nan as the non-standard token NaN; a validation
        # document for automation must never contain one.
        result = validate_run(make_run(summary='{"steps": NaN}'))

        text = result.to_json()

        assert "NaN" not in text
        payload = json.loads(text)
        assert payload["ok"] is True
        assert payload["summary"]["steps"] is None

    def test_text_and_json_describe_the_same_result(self, make_run):
        result = validate_run(make_run(summary={**GOOD_SUMMARY, "steps": 99.0}))
        payload = json.loads(result.to_json())

        assert payload["ok"] is False
        assert payload["errors"] == list(result.errors)
        assert payload["warnings"] == list(result.warnings)
        assert payload["path"] == result.path
        assert result.errors[0] in result.to_text()

    def test_the_result_is_immutable_evidence(self, make_run):
        result = validate_run(make_run())

        with pytest.raises(AttributeError):
            result.ok = False  # type: ignore[misc]

    def test_a_validation_error_is_a_shardscope_error(self):
        assert issubclass(ValidationError, ShardScopeError)
        assert isinstance(ValidationResult, type)


class TestValidateCommand:
    def test_a_valid_run_exits_zero_with_a_concise_text_result(self, make_run, capsys):
        run_dir = make_run("baseline")

        assert main(["validate", "--run", str(run_dir)]) == 0

        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out.startswith("valid run: ")
        assert "baseline" in captured.out
        assert "3 row(s)" in captured.out
        assert "summary.json" in captured.out
        assert "profiled=false" in captured.out
        assert len(captured.out.splitlines()) <= 10

    def test_an_absent_summary_text_mentions_absent(self, make_run, capsys):
        main(["validate", "--run", str(make_run(summary=None))])

        out = capsys.readouterr().out
        assert "absent" in out
        assert "optional" in out

    def test_a_step_count_disagreement_exits_two_with_one_line(self, make_run, capsys):
        run_dir = make_run(summary={**GOOD_SUMMARY, "steps": 99.0})

        assert main(["validate", "--run", str(run_dir)]) == 2

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("error: ")
        assert captured.err.count("\n") == 1
        assert "99" in captured.err and "3 row" in captured.err

    def test_a_malformed_run_exits_two_with_the_loaders_line(self, make_run, capsys):
        run_dir = make_run(steps_csv=_csv("0,x,2.0,host,True"))

        assert main(["validate", "--run", str(run_dir)]) == 2

        err = capsys.readouterr().err
        assert err.startswith("error: ")
        assert "row 2" in err

    def test_warnings_are_listed_and_do_not_fail_validation(self, make_run, capsys):
        extra = {k: v for k, v in GOOD_ENVIRONMENT["extra"].items() if k != "profiled"}

        code = main(
            ["validate", "--run", str(make_run(environment={**GOOD_ENVIRONMENT, "extra": extra}))]
        )

        assert code == 0
        out = capsys.readouterr().out
        assert "warning" in out
        assert "profiled" in out

    def test_json_output_for_a_valid_run(self, make_run, capsys):
        assert main(["validate", "--run", str(make_run()), "--format", "json"]) == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is True
        assert payload["errors"] == []
        assert payload["steps"]["rows"] == 3
        assert payload["summary"] == {"present": True, "steps": 3, "agrees_with_rows": True}

    def test_json_output_for_a_failure_goes_to_stdout_and_exits_two(self, make_run, capsys):
        run_dir = make_run(summary={**GOOD_SUMMARY, "steps": 99.0})

        assert main(["validate", "--run", str(run_dir), "--format", "json"]) == 2

        captured = capsys.readouterr()
        assert captured.err == ""
        payload = json.loads(captured.out)
        assert payload["ok"] is False
        assert len(payload["errors"]) == 1
        assert "Traceback" not in captured.out

    def test_json_output_for_an_unreadable_run_is_still_a_document(self, make_run, capsys):
        run_dir = make_run(steps_csv="not,a,csv\n")

        assert main(["validate", "--run", str(run_dir), "--format", "json"]) == 2

        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is False
        assert payload["steps"] is None

    def test_validate_writes_nothing(self, make_run, tmp_path, capsys):
        run_dir = make_run()
        before = _snapshot(run_dir)
        outside = set(tmp_path.iterdir())

        main(["validate", "--run", str(run_dir)])
        main(["validate", "--run", str(run_dir), "--format", "json"])

        assert _snapshot(run_dir) == before
        assert set(tmp_path.iterdir()) == outside

    def test_validate_is_implemented_and_the_pending_commands_still_refuse(self):
        assert "validate" not in [command.name for command in PENDING_COMMANDS]
        assert [command.name for command in PENDING_COMMANDS] == ["profile", "bench", "compare"]

    def test_help_does_not_describe_validate_as_unimplemented(self, capsys):
        with pytest.raises(SystemExit):
            main(["validate", "--help"])

        out = capsys.readouterr().out
        assert "not implemented" not in out.lower()
        assert "--format" in out

    def test_the_fixture_csv_is_the_shape_this_file_assumes(self):
        assert GOOD_STEPS_CSV.startswith(HEADER)
