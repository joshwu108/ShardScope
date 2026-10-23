"""End-to-end tests that drive the real processes, not the functions.

Every other test file calls ``main([...])`` in-process. That is fast and it
is where the detail lives, but it cannot catch the failures that only exist
once something is *installed*: a console script pointing at a stale copy of
the package, an entry point that does not import, a run directory the
producer writes and the reader then refuses, an exit code the wrapper
swallows, or a traceback on a closed pipe.

So these tests spawn ``shardscope`` off ``PATH`` and
``python examples/train_single.py`` as separate processes and assert on exit
codes, stdout, stderr and the files left on disk. They are the pytest half
of the ``smoke`` check in ``tools/dev.py``: that check proves the path runs,
these prove it runs *correctly*.

Scope: only workflows that are implemented and carry no open measurement
decision -- ``env``, the training fixture, and ``report``. The unimplemented
commands are checked for refusing cleanly, which is the whole of their
contract today.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

from shardscope.artifacts import RunArtifacts

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SINGLE = REPO_ROOT / "examples" / "train_single.py"
BENCHMARKS = REPO_ROOT / "benchmarks"

#: The installed console script, or None when the package is not installed.
#: Resolved once: a skip must say "not installed", not fail as "not found".
SHARDSCOPE = shutil.which("shardscope")

requires_install = pytest.mark.skipif(
    SHARDSCOPE is None,
    reason="the 'shardscope' console script is not on PATH; install with "
    'uv pip install -e ".[dev]"',
)


def _run(argv: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` from the repo root, capturing text output."""
    return subprocess.run(
        argv, cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", **kwargs
    )


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    assert SHARDSCOPE is not None  # guarded by requires_install
    return _run([SHARDSCOPE, *args])


def _fixture(*args: str) -> subprocess.CompletedProcess[str]:
    # sys.executable, not "python": the interpreter running the tests is the
    # one the package is installed into.
    return _run([sys.executable, str(TRAIN_SINGLE), *args])


@pytest.fixture(scope="module")
def recorded_run(tmp_path_factory) -> Path:
    """One real training run, recorded to disk by the real producer process.

    Module-scoped because it costs a process start plus a torch import, and
    because every reader test wants the *same* artifacts: if two tests each
    produced their own run, a reader bug that depends on the run's contents
    would be harder to pin down, not easier.
    """
    out_dir = tmp_path_factory.mktemp("integration") / "run"
    completed = _fixture("--steps", "5", "--device", "cpu", "--quiet", "--out-dir", str(out_dir))

    assert completed.returncode == 0, completed.stderr
    return out_dir


class TestInstallation:
    """What only an installed package can get wrong."""

    @requires_install
    def test_the_console_script_reports_the_installed_version(self):
        # A stale editable install, or another shardscope earlier on PATH,
        # shows up here and nowhere else: every in-process test imports the
        # package directly and so can never disagree with itself.
        completed = _cli("--version")

        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == f"shardscope {version('shardscope')}"

    @requires_install
    def test_help_lists_every_command_and_marks_the_pending_ones(self):
        completed = _cli("--help")

        assert completed.returncode == 0, completed.stderr
        for name in ("env", "report", "profile", "bench", "compare"):
            assert name in completed.stdout
        assert "not implemented" in completed.stdout

    @requires_install
    def test_an_unknown_command_exits_two_without_a_traceback(self):
        completed = _cli("nope")

        assert completed.returncode == 2
        assert "Traceback" not in completed.stderr


class TestEnvCommand:
    @requires_install
    def test_the_text_snapshot_names_the_fields_a_reader_needs(self):
        completed = _cli("env")

        assert completed.returncode == 0, completed.stderr
        for label in ("python", "torch", "accelerator", "cuda devices", "ddp capable"):
            assert label in completed.stdout

    @requires_install
    def test_the_json_snapshot_parses_and_carries_the_device_inventory(self):
        completed = _cli("env", "--format", "json")

        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert isinstance(payload["cuda_available"], bool)
        assert isinstance(payload["device_count"], int)
        assert isinstance(payload["device_names"], list)

    @requires_install
    def test_the_device_count_and_the_inventory_always_agree(self):
        # The honesty invariant of the snapshot, and it holds on every host:
        # a count that outruns the named inventory is how a DDP result gets
        # attributed to a machine with one GPU. Asserted unconditionally,
        # because a check that only runs on CUDA hosts is no check here.
        payload = json.loads(_cli("env", "--format", "json").stdout)

        assert len(payload["device_names"]) == payload["device_count"]
        if not payload["cuda_available"]:
            assert payload["device_count"] == 0
            # No CUDA also means no CUDA-only version string invented.
            assert payload["cudnn_version"] is None
            assert payload["nccl_version"] is None

    @requires_install
    def test_ddp_capability_follows_the_device_count(self):
        # One rule for "can this host run DDP", shared by the snapshot and
        # any report that reads a recorded environment.
        from shardscope.env import distributed_capable

        payload = json.loads(_cli("env", "--format", "json").stdout)
        expected = distributed_capable(
            cuda_available=payload["cuda_available"], device_count=payload["device_count"]
        )

        assert ("ddp capable     True" in _cli("env").stdout) is expected


class TestTrainingFixtureProcess:
    def test_a_recorded_run_contains_the_three_artifacts(self, recorded_run):
        assert (recorded_run / "steps.csv").is_file()
        assert (recorded_run / "environment.json").is_file()
        assert (recorded_run / "summary.json").is_file()

    def test_the_recorded_run_says_it_was_not_profiled(self, recorded_run):
        extra = json.loads((recorded_run / "environment.json").read_text(encoding="utf-8"))["extra"]

        # Requirement SS-04: an artifact must say which side of the
        # profiled/unprofiled split it came from, and `profiled` must be a
        # recorded false rather than a missing key read as one.
        assert extra["profiled"] is False
        assert extra["source"] == "examples/train_single.py"

    def test_a_bad_config_refuses_before_training_and_writes_nothing(self, tmp_path):
        config = tmp_path / "bad.json"
        config.write_text('{"steps": 0}\n', encoding="utf-8")
        out_dir = tmp_path / "never"

        completed = _fixture("--config", str(config), "--quiet", "--out-dir", str(out_dir))

        assert completed.returncode == 2
        assert completed.stderr.startswith("error: ")
        assert "Traceback" not in completed.stderr
        assert "median=" not in completed.stdout
        assert not out_dir.exists()

    def test_an_unusable_out_dir_refuses_before_training(self, tmp_path):
        occupied = tmp_path / "occupied"
        occupied.write_text("not a directory\n", encoding="utf-8")

        completed = _fixture("--steps", "5", "--quiet", "--out-dir", str(occupied))

        assert completed.returncode == 2
        assert "Traceback" not in completed.stderr
        # No summary line means the refusal came before the first step, so a
        # mistyped path costs a second rather than a whole run.
        assert "median=" not in completed.stdout
        assert occupied.read_text(encoding="utf-8") == "not a directory\n"


class TestReportProcess:
    """The producer-to-reader path, across two separate processes."""

    @requires_install
    def test_a_recorded_run_renders_to_an_html_file(self, recorded_run, tmp_path):
        out = tmp_path / "report.html"

        completed = _cli("report", "--run", str(recorded_run), "--out", str(out))

        assert completed.returncode == 0, completed.stderr
        assert f"wrote {out}" in completed.stdout
        document = out.read_text(encoding="utf-8")
        assert document.startswith("<!DOCTYPE html>")
        assert document.rstrip().endswith("</html>")

    @requires_install
    def test_the_html_report_states_the_run_was_not_profiled(self, recorded_run, tmp_path):
        out = tmp_path / "report.html"
        _cli("report", "--run", str(recorded_run), "--out", str(out))

        # The guardrail that matters most in a shared artifact: a reader must
        # not be able to quote this document as a throughput result.
        document = out.read_text(encoding="utf-8")
        assert "Unprofiled run." in document
        assert "not throughput results" in document

    @requires_install
    def test_the_csv_view_is_byte_identical_to_the_recorded_file(self, recorded_run):
        completed = _run(
            [SHARDSCOPE, "report", "--run", str(recorded_run), "--format", "csv"],
        )

        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == (recorded_run / "steps.csv").read_text(encoding="utf-8")

    @requires_install
    def test_the_json_view_loads_back_into_run_artifacts(self, recorded_run):
        completed = _cli("report", "--run", str(recorded_run), "--format", "json")

        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        reloaded = RunArtifacts.from_dict(payload)

        assert len(reloaded.steps) == 5
        assert [status["field"] for status in payload["availability"]]

    @requires_install
    def test_a_missing_run_exits_two_with_one_line_and_writes_nothing(self, tmp_path):
        out = tmp_path / "report.html"

        completed = _cli("report", "--run", str(tmp_path / "absent"), "--out", str(out))

        assert completed.returncode == 2
        assert completed.stdout == ""
        assert "Traceback" not in completed.stderr
        assert len(completed.stderr.strip().splitlines()) == 1
        assert not out.exists()

    @requires_install
    def test_a_closed_stdout_pipe_is_not_an_error(self, recorded_run):
        # 'shardscope report --format csv | head' is documented; the reader
        # is allowed to stop reading, and that must not read as a crash.
        process = subprocess.Popen(
            [SHARDSCOPE, "report", "--run", str(recorded_run), "--format", "csv"],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        process.stdout.close()
        stderr = process.stderr.read().decode() if process.stderr else ""

        assert process.wait() == 141
        assert stderr == ""


class TestUnimplementedScriptsRefuseCleanly:
    """A stub that waits on a decision must not look like a crash.

    These scripts are runnable, so someone will run them. Exiting 1 with a
    ``NotImplementedError`` traceback reads as a broken harness; exiting 2
    with one line naming the open decision reads as what it is.
    """

    @pytest.mark.parametrize("script", ["run_baseline.py", "compare.py"])
    def test_they_exit_two_naming_gate_7_1(self, script):
        completed = _run([sys.executable, str(BENCHMARKS / script)])

        assert completed.returncode == 2
        assert completed.stdout == ""
        assert "Traceback" not in completed.stderr
        assert len(completed.stderr.strip().splitlines()) == 1
        assert "GATE 7.1" in completed.stderr
        assert "not implemented" in completed.stderr

    def test_both_name_the_same_blocker(self):
        # One methodology decision blocks both. Two spellings of it would
        # drift, and a reader would not know whether that mattered.
        first = _run([sys.executable, str(BENCHMARKS / "run_baseline.py")]).stderr
        second = _run([sys.executable, str(BENCHMARKS / "compare.py")]).stderr

        assert first.split("Blocked on:")[1] == second.split("Blocked on:")[1]


class TestPendingCommandsRefuseCleanly:
    @requires_install
    @pytest.mark.parametrize("command", ["profile", "bench", "compare"])
    def test_they_exit_two_naming_a_decision_and_create_nothing(self, command, tmp_path):
        target = tmp_path / "should-not-exist"
        argv = ["--out", str(target)] if command != "compare" else ["--baseline", str(target)]

        completed = _cli(command, *argv)

        assert completed.returncode == 2
        assert "GATE" in completed.stderr
        assert "Traceback" not in completed.stderr
        assert not target.exists()
        assert os.listdir(tmp_path) == []
