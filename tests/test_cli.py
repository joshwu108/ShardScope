"""Tests for the command-line surface."""

from __future__ import annotations

import json

import pytest

from shardscope import __version__
from shardscope.cli import PENDING_COMMANDS, main


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
        "report": ["--run", "--out"],
        "bench": ["--config", "--out"],
        "compare": ["--baseline", "--candidate"],
    }
    declared = {
        command.name: [option.flag for option in command.options] for command in PENDING_COMMANDS
    }

    assert declared == expected


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
