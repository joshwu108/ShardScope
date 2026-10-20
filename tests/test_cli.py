"""Tests for the command-line surface."""

from __future__ import annotations

import json

import pytest

from shardscope.cli import main


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


@pytest.mark.parametrize("command", ["profile", "report", "bench"])
def test_unimplemented_commands_fail_loudly(command, capsys):
    # Requirement SS-04 depends on these staying distinct commands; a stub
    # must exit non-zero rather than silently appear to succeed.
    assert main([command]) == 2
    assert "not implemented yet" in capsys.readouterr().err


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        main(["nope"])


def test_missing_command_is_rejected():
    with pytest.raises(SystemExit):
        main([])
