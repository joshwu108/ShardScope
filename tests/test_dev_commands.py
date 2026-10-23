"""Developer-command and CI parity.

``tools/dev.py`` exists so that "it passed locally" and "it passed in CI"
mean the same commands ran. These tests are what keeps that true: if a CI
step stops going through the script, or a check drops out of the CI sequence,
the parity test fails rather than the gate quietly going missing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dev = _load("dev")


class TestRegistry:
    def test_check_names_are_unique(self):
        names = [check.name for check in dev.CHECKS]

        assert len(names) == len(set(names))

    def test_the_ci_sequence_is_exactly_the_in_ci_checks_in_order(self):
        assert tuple(c.name for c in dev.CHECKS if c.in_ci) == dev.CI_SEQUENCE

    @pytest.mark.parametrize(
        "name", ["lint", "format-check", "readme-check", "cov", "ignore-policy", "smoke"]
    )
    def test_the_gates_that_must_run_in_ci_are_in_the_ci_sequence(self, name):
        assert name in dev.CI_SEQUENCE

    def test_the_coverage_gate_is_the_one_ci_runs(self):
        # 'test' skips the coverage gate; CI must use 'cov' or the 80%
        # threshold stops being enforced anywhere.
        assert "cov" in dev.CI_SEQUENCE
        assert "test" not in dev.CI_SEQUENCE
        assert "--cov" in dev.BY_NAME["cov"].argv

    def test_file_rewriting_checks_never_run_in_ci(self):
        # 'format' rewrites files; running it in CI would hide a formatting
        # violation instead of reporting it.
        assert dev.BY_NAME["format"].in_ci is False
        assert "format" not in dev.CI_SEQUENCE

    def test_every_check_is_either_a_command_or_a_callable(self):
        for check in dev.CHECKS:
            assert bool(check.argv) != (check.run is not None), check.name

    def test_subprocess_checks_use_the_current_interpreter(self):
        # A bare "python" would run whatever is first on PATH, which is not
        # necessarily the interpreter the package was installed into.
        for check in dev.CHECKS:
            if not check.argv:
                continue
            assert check.argv[0] == sys.executable, check.name


class TestCiParity:
    def test_the_workflow_exists(self):
        assert WORKFLOW.is_file()

    @pytest.mark.parametrize("name", dev.CI_SEQUENCE)
    def test_the_workflow_invokes_every_ci_check_through_dev_py(self, name):
        assert f"python tools/dev.py {name}" in WORKFLOW.read_text()

    def test_the_workflow_does_not_reimplement_a_gate_inline(self):
        # Re-spelling a gate in YAML is how CI and local checks drift apart.
        text = WORKFLOW.read_text()

        assert "ruff check ." not in text
        assert "pytest -q" not in text
        assert "git check-ignore" not in text


class TestReadmeGate:
    """README.rst is the package's readme and the only public document."""

    def test_the_real_readme_parses(self):
        dev.check_readme()

    def test_a_malformed_readme_fails_with_the_parser_output(self, tmp_path, monkeypatch):
        (tmp_path / "README.rst").write_text(
            "Title\n=====\n\n.. code-block:: console\n\nno content block\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(dev, "ROOT", tmp_path)

        with pytest.raises(dev.CheckFailed) as caught:
            dev.check_readme()

        assert "does not parse cleanly" in str(caught.value)
        assert "code-block" in str(caught.value)

    def test_a_missing_docutils_is_reported_as_a_missing_dependency(self, monkeypatch):
        # Not as a README problem, and never as a silent pass: a gate that
        # quietly does nothing is worse than no gate.
        def capture(_argv, **_kwargs):
            return _Completed(returncode=1, stdout="", stderr="No module named docutils")

        monkeypatch.setattr(dev, "_capture", capture)

        with pytest.raises(dev.CheckFailed, match="docutils is not installed"):
            dev.check_readme()


class TestInstalledScriptCheck:
    """The console script and the interpreter must be the same install."""

    def test_a_version_mismatch_is_refused(self, monkeypatch):
        monkeypatch.setattr(dev.shutil, "which", lambda _name: "/somewhere/bin/shardscope")
        monkeypatch.setattr(dev, "_installed_version", lambda: "9.9.9")
        monkeypatch.setattr(
            dev,
            "_capture",
            lambda _argv, **_kwargs: _Completed(returncode=0, stdout="shardscope 0.0.1\n"),
        )

        with pytest.raises(dev.CheckFailed) as caught:
            dev.check_smoke()

        message = str(caught.value)
        assert "shardscope 0.0.1" in message
        assert "shardscope 9.9.9" in message

    def test_an_uninstalled_package_is_named_as_such(self, monkeypatch):
        monkeypatch.setattr(
            dev,
            "_capture",
            lambda _argv, **_kwargs: _Completed(
                returncode=1, stdout="", stderr="No module named shardscope"
            ),
        )

        with pytest.raises(dev.CheckFailed, match="not installed for this interpreter"):
            dev._installed_version()

    def test_a_missing_console_script_says_how_to_install(self, monkeypatch):
        monkeypatch.setattr(dev.shutil, "which", lambda _name: None)

        with pytest.raises(dev.CheckFailed, match="not on PATH"):
            dev.check_smoke()


class TestIgnorePolicy:
    def test_the_policy_holds_in_this_working_tree(self):
        assert dev.ignore_policy_failures() == []

    def test_a_non_ignored_planning_path_is_reported(self):
        def capture(argv, **_):
            # Pretend nothing is ignored.
            return _Completed(returncode=1 if argv[1] == "check-ignore" else 0, stdout="")

        failures = dev.ignore_policy_failures(capture)

        assert any("is NOT git-ignored" in failure for failure in failures)
        assert any("planning/ShardScope_PRD.md" in failure for failure in failures)

    def test_a_tracked_markdown_file_is_reported(self):
        def capture(argv, **_):
            if argv[1] == "ls-files" and "--error-unmatch" not in argv:
                return _Completed(returncode=0, stdout="README.md\nplanning/design.md\n")
            return _Completed(returncode=0, stdout="")

        failures = dev.ignore_policy_failures(capture)

        assert any("README.md" in failure and "tracked" in failure for failure in failures)

    def test_an_untracked_gitignore_is_reported(self):
        def capture(argv, **_):
            if "--error-unmatch" in argv:
                return _Completed(returncode=1, stdout="")
            return _Completed(returncode=0, stdout="")

        failures = dev.ignore_policy_failures(capture)

        assert any(
            ".gitignore" in failure and "must stay tracked" in failure for failure in failures
        )

    def test_the_auth_file_and_run_artifacts_are_covered_by_the_policy(self):
        # Secrets and machine-specific artifacts are the two things whose
        # leaking matters most, so name them explicitly.
        assert ".shardscope-commit-auth.json" in dev.MUST_BE_IGNORED
        assert any(path.startswith("runs/") for path in dev.MUST_BE_IGNORED)
        # A rendered report carries the producing machine's paths and
        # operation names; PRD v2.0 section 11 keeps it local until reviewed.
        assert any(path.startswith("reports/") for path in dev.MUST_BE_IGNORED)


class TestCli:
    def test_listing_the_checks_succeeds(self, capsys):
        assert dev.main(["--list"]) == 0
        out = capsys.readouterr().out

        for check in dev.CHECKS:
            assert check.name in out

    def test_no_argument_lists_the_checks_and_fails(self, capsys):
        # Nothing ran, so exiting 0 would be a false pass.
        assert dev.main([]) == 2
        assert "checks" in capsys.readouterr().out

    def test_an_unknown_check_is_rejected(self):
        with pytest.raises(SystemExit):
            dev.main(["nope"])

    def test_a_failing_check_reports_which_one_failed(self, capsys, monkeypatch):
        def explode():
            raise dev.CheckFailed("synthetic failure")

        monkeypatch.setitem(
            dev.BY_NAME,
            "ignore-policy",
            dev.Check("ignore-policy", "synthetic", run=explode),
        )

        assert dev.run_checks(["ignore-policy"]) == 1
        assert "FAILED ignore-policy" in capsys.readouterr().err


class _Completed:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
