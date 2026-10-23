"""Tests for the command-line surface."""

from __future__ import annotations

import json
import os

import pytest

from conftest import GOOD_STEPS_CSV
from shardscope import __version__
from shardscope.cli import EXIT_BROKEN_PIPE, PENDING_COMMANDS, build_parser, main


def test_env_text_output(capsys):
    assert main(["env"]) == 0
    out = capsys.readouterr().out
    assert "ShardScope environment" in out
    assert "ddp capable" in out


def test_env_json_output_is_parseable(capsys):
    assert main(["env", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "device_count" in payload
    assert "torch_version" in payload


PENDING_NAMES = [command.name for command in PENDING_COMMANDS]


@pytest.mark.parametrize("command", PENDING_NAMES)
def test_unimplemented_commands_fail_loudly(command, capsys):
    # Requirement SS-04 depends on these staying distinct commands; a stub
    # must exit non-zero rather than silently appear to succeed.
    assert main([command]) == 2
    assert "not implemented yet" in capsys.readouterr().err


@pytest.mark.parametrize("command", PENDING_NAMES)
def test_unimplemented_commands_name_the_decision_they_wait_on(command, capsys):
    # "Unimplemented" and "waiting on a decision the student owns" are
    # different states; conflating them hides the approval gate.
    main([command])
    err = capsys.readouterr().err

    assert "Blocked on:" in err
    assert "GATE" in err


@pytest.mark.parametrize("command", PENDING_COMMANDS, ids=PENDING_NAMES)
def test_unimplemented_commands_have_no_side_effects(command, tmp_path, capsys):
    # A refused command must not leave a half-built run directory behind.
    before = set(tmp_path.iterdir())
    argv = [command.name]
    for option in command.options:
        argv += [option.flag, str(tmp_path / f"{option.flag.lstrip('-')}-target")]

    assert main(argv) == 2
    assert set(tmp_path.iterdir()) == before


def test_the_prd_section_11_interface_target_is_documented(capsys):
    # PRD v2.0 section 11 names these flags. Declaring them keeps --help
    # honest about the target shape while the command is still refused.
    expected = {
        "profile": ["--config", "--out"],
        "bench": ["--config", "--out"],
        "compare": ["--baseline", "--candidate"],
    }
    declared = {
        command.name: [option.flag for option in command.options] for command in PENDING_COMMANDS
    }

    assert declared == expected


def test_report_is_implemented_and_the_rest_still_are_not():
    assert "report" not in PENDING_NAMES
    assert PENDING_NAMES == ["profile", "bench", "compare"]


def test_report_accepts_the_prd_section_11_flags():
    args = build_parser().parse_args(["report", "--run", "runs/x", "--out", "reports/x.html"])

    assert str(args.run) == "runs/x"
    assert str(args.out) == "reports/x.html"


def test_memory_lab_commands_are_not_offered():
    # PRD v2.0 section 10 gates the whole Memory Lab on a Week 7 review that
    # has not happened; offering the commands would imply otherwise.
    assert "memory-sweep" not in PENDING_NAMES
    assert "memory-report" not in PENDING_NAMES


def test_help_marks_pending_commands_as_not_implemented(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out

    for name in PENDING_NAMES:
        assert name in out
    assert "not implemented" in out


class TestReportCommand:
    def test_writes_an_html_report_and_says_where(self, make_run, tmp_path, capsys):
        out = tmp_path / "reports" / "baseline.html"

        assert main(["report", "--run", str(make_run("baseline")), "--out", str(out)]) == 0

        assert "baseline" in out.read_text()
        assert str(out) in capsys.readouterr().out

    def test_the_format_follows_the_output_suffix(self, make_run, tmp_path):
        run_dir = make_run()
        as_json = tmp_path / "r.json"
        as_csv = tmp_path / "r.csv"

        assert main(["report", "--run", str(run_dir), "--out", str(as_json)]) == 0
        assert main(["report", "--run", str(run_dir), "--out", str(as_csv)]) == 0

        assert json.loads(as_json.read_text())["run_id"] == "run"
        assert as_csv.read_text() == GOOD_STEPS_CSV

    def test_an_explicit_format_overrides_the_suffix(self, make_run, tmp_path):
        out = tmp_path / "r.txt"

        assert (
            main(
                [
                    "report",
                    "--run",
                    str(make_run()),
                    "--out",
                    str(out),
                    "--format",
                    "json",
                ]
            )
            == 0
        )

        assert json.loads(out.read_text())["steps"]

    def test_an_unknown_suffix_asks_for_a_format_and_writes_nothing(
        self, make_run, tmp_path, capsys
    ):
        out = tmp_path / "r.pdf"

        assert main(["report", "--run", str(make_run()), "--out", str(out)]) == 2

        assert "--format" in capsys.readouterr().err
        assert not out.exists()

    def test_without_an_output_the_report_goes_to_stdout(self, make_run, capsys):
        assert main(["report", "--run", str(make_run()), "--format", "csv"]) == 0

        assert capsys.readouterr().out == GOOD_STEPS_CSV

    def test_the_row_cap_is_honoured(self, make_run, capsys):
        assert main(["report", "--run", str(make_run()), "--max-rows", "1"]) == 0

        assert "1 of 3" in capsys.readouterr().out

    def test_a_zero_row_cap_is_refused_and_writes_nothing(self, make_run, tmp_path, capsys):
        out = tmp_path / "reports" / "r.html"

        assert main(["report", "--run", str(make_run()), "--out", str(out), "--max-rows", "0"]) == 2

        assert "max_rows" in capsys.readouterr().err
        assert not out.exists()
        assert not out.parent.exists()

    def test_an_unwritable_output_exits_two_with_one_line(self, make_run, tmp_path, capsys):
        out = tmp_path / "report.html"
        out.mkdir()

        assert main(["report", "--run", str(make_run()), "--out", str(out)]) == 2

        err = capsys.readouterr().err
        assert err.startswith("error: ")
        assert "cannot write" in err

    def test_a_zero_row_cap_is_refused_for_a_csv_report_too(self, make_run, capsys):
        assert main(["report", "--run", str(make_run()), "--format", "csv", "--max-rows", "0"]) == 2

        assert "max_rows" in capsys.readouterr().err

    def test_a_missing_run_directory_exits_two_with_one_line(self, tmp_path, capsys):
        assert main(["report", "--run", str(tmp_path / "absent")]) == 2

        err = capsys.readouterr().err
        assert err.startswith("error: ")
        assert "no such run directory" in err

    def test_a_malformed_step_row_names_the_row(self, make_run, capsys):
        run_dir = make_run(
            steps_csv="step,loss,step_time_ms,data_residency,synchronized\n0,x,1,host,True\n"
        )

        assert main(["report", "--run", str(run_dir)]) == 2

        assert "row 2" in capsys.readouterr().err

    def test_a_run_without_provenance_is_refused(self, make_run, capsys):
        assert main(["report", "--run", str(make_run(environment=None))]) == 2

        assert "provenance" in capsys.readouterr().err

    def test_pointing_at_a_parent_directory_lists_the_runs_found(self, make_run, tmp_path, capsys):
        make_run("alpha")

        assert main(["report", "--run", str(tmp_path)]) == 2

        assert "alpha" in capsys.readouterr().err

    def test_a_real_fixture_run_reports_end_to_end(self, tmp_path, capsys):
        import train_single

        run_dir = tmp_path / "fixture"
        assert (
            train_single.main(
                ["--steps", "3", "--device", "cpu", "--quiet", "--out-dir", str(run_dir)]
            )
            == 0
        )
        out = tmp_path / "report.html"

        assert main(["report", "--run", str(run_dir), "--out", str(out)]) == 0

        html = out.read_text()
        assert "Unprofiled run" in html
        assert "examples/train_single.py" in html
        assert "report generated" in html

    def test_help_does_not_describe_report_as_unimplemented(self, capsys):
        with pytest.raises(SystemExit):
            main(["report", "--help"])
        out = capsys.readouterr().out

        assert "not implemented" not in out.lower()
        assert "--format" in out


class TestAClosedStdoutPipe:
    """``shardscope report ... | head`` must not end in a traceback.

    Reporting to stdout is a documented way to use this command, and the
    reader of a pipe is allowed to stop reading. That is not a ShardScope
    bug and must not be printed as one.
    """

    @staticmethod
    def _broken_stdout(monkeypatch):
        class BrokenStdout:
            def write(self, _text):
                raise BrokenPipeError(32, "Broken pipe")

            def flush(self):
                pass

        monkeypatch.setattr("sys.stdout", BrokenStdout())

    def test_exits_with_the_sigpipe_code_and_says_nothing(self, make_run, monkeypatch, capsys):
        run = make_run()
        self._broken_stdout(monkeypatch)

        code = main(["report", "--run", str(run), "--format", "csv"])

        assert code == EXIT_BROKEN_PIPE
        assert capsys.readouterr().err == ""

    def test_a_pipe_that_closes_during_the_exit_flush_is_handled_too(
        self, make_run, monkeypatch, capsys
    ):
        # A short report fits in the pipe buffer, so the write succeeds and
        # only the flush fails. Left to the interpreter that is an
        # "Exception ignored" line and exit code 120.
        written: list[str] = []

        class BufferedStdout:
            def write(self, text):
                written.append(text)

            def flush(self):
                raise BrokenPipeError(32, "Broken pipe")

        run = make_run()
        monkeypatch.setattr("sys.stdout", BufferedStdout())

        code = main(["report", "--run", str(run), "--format", "csv"])

        assert code == EXIT_BROKEN_PIPE
        assert written, "nothing was written before the flush failed"
        assert capsys.readouterr().err == ""

    def test_remaining_output_is_sent_to_devnull(self, make_run, tmp_path, monkeypatch):
        # The interpreter flushes stdout at shutdown. If that descriptor is
        # still the dead pipe, the flush complains again -- after we already
        # decided this was not an error -- so it is redirected away.
        sink = tmp_path / "sink.txt"
        handle = sink.open("w")
        fileno = handle.fileno()

        class BrokenButRealStdout:
            def write(self, _text):
                raise BrokenPipeError(32, "Broken pipe")

            def flush(self):
                pass

            def fileno(self):
                return fileno

        run = make_run()
        monkeypatch.setattr("sys.stdout", BrokenButRealStdout())

        code = main(["report", "--run", str(run), "--format", "csv"])
        os.write(fileno, b"this must not reach the file")
        handle.close()

        assert code == EXIT_BROKEN_PIPE
        assert sink.read_text() == ""

    def test_the_redirect_does_not_leak_a_descriptor(self, make_run, tmp_path, monkeypatch):
        # dup2 duplicates the descriptor, so the one os.open returned is a
        # second reference. Harmless in a one-shot process, a leak per call
        # in anything that calls main() more than once.
        sink = tmp_path / "sink.txt"
        handle = sink.open("w")
        fileno = handle.fileno()

        class BrokenButRealStdout:
            def write(self, _text):
                raise BrokenPipeError(32, "Broken pipe")

            def flush(self):
                pass

            def fileno(self):
                return fileno

        run = make_run()
        monkeypatch.setattr("sys.stdout", BrokenButRealStdout())

        before = _open_descriptor_count()
        for _ in range(3):
            assert main(["report", "--run", str(run), "--format", "csv"]) == EXIT_BROKEN_PIPE
        after = _open_descriptor_count()
        handle.close()

        assert after == before

    def test_the_code_is_distinct_from_a_user_error_and_from_success(self):
        # 0 success, 2 user error, 141 = 128 + SIGPIPE. Collapsing the third
        # into either of the first two would make a script unable to tell
        # "you typed a bad path" from "the reader hung up".
        assert EXIT_BROKEN_PIPE == 141


def _open_descriptor_count() -> int:
    """How many file descriptors this process holds open.

    /proc/self/fd on Linux, /dev/fd on macOS; both are directories whose
    entries are the open descriptors.
    """
    for path in ("/proc/self/fd", "/dev/fd"):
        if os.path.isdir(path):
            return len(os.listdir(path))
    pytest.skip("no /proc/self/fd or /dev/fd on this platform")


def test_version_is_reported(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])

    assert __version__ in capsys.readouterr().out


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        main(["nope"])


def test_missing_command_is_rejected():
    with pytest.raises(SystemExit):
        main([])


def test_an_unexpected_error_is_not_swallowed(monkeypatch):
    # Only ShardScopeError becomes a tidy message; a real bug must still
    # surface as a traceback instead of a quiet exit code.
    def boom():
        raise RuntimeError("kaboom")

    monkeypatch.setattr("shardscope.cli.capture_environment", boom)

    with pytest.raises(RuntimeError, match="kaboom"):
        main(["env"])
