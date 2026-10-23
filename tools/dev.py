#!/usr/bin/env python3
"""One place that defines every check this project runs.

CI and a developer's terminal used to describe the same gates in two
different dialects, which is how a check quietly stops matching the thing it
is supposed to gate. ``.github/workflows/ci.yml`` now shells out to this
script, so "it passed locally" and "it passed in CI" mean the same commands
ran. ``tests/test_dev_commands.py`` asserts that parity.

Usage::

    python tools/dev.py --list
    python tools/dev.py lint
    python tools/dev.py format          # rewrites files
    python tools/dev.py ci              # everything CI runs, in order

Stdlib only, and it never installs anything: a check that silently fixes its
own environment cannot tell you the environment is broken.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

PY = [sys.executable]

#: Paths that must be ignored by git, with the reason each one matters.
MUST_BE_IGNORED = {
    "planning/ShardScope_PRD.md": "planning documents stay local (PRD v2.0 section 13.2)",
    "planning/notes/scratch.md": "all Markdown stays local (PRD v2.0 section 13.2)",
    ".agents/skills/commit/SKILL.md": "the generated commit skill is regenerable, not tracked",
    ".shardscope-commit-auth.json": "local commit authorization is never shared",
    "runs/baseline/steps.csv": "run artifacts are machine-specific and regenerable",
    "reports/baseline.html": "a rendered report embeds local paths and must be reviewed first",
}

#: Globs that must match nothing in the index.
MUST_NOT_BE_TRACKED = ("*.md", "*.markdown", "*.mdx", "planning/*")

#: Paths that must stay tracked; .gitignore does not untrack itself.
MUST_BE_TRACKED = (".gitignore", "README.rst")


class CheckFailed(RuntimeError):
    """A check did not pass. The message is shown to the user verbatim."""


@dataclass(frozen=True)
class Check:
    """One named gate, either a subprocess or a Python callable."""

    name: str
    summary: str
    argv: tuple[str, ...] = ()
    run: Callable[[], None] | None = None
    in_ci: bool = True

    def execute(self) -> None:
        if self.run is not None:
            self.run()
            return
        print(f"$ {' '.join(self.argv)}", flush=True)
        completed = subprocess.run(list(self.argv), cwd=ROOT)
        if completed.returncode != 0:
            raise CheckFailed(f"{self.name}: exited {completed.returncode}")


def _capture(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), cwd=ROOT, capture_output=True, text=True)


def ignore_policy_failures(capture: Callable[..., Any] = _capture) -> list[str]:
    """Return a list of ignore-policy violations, empty when the policy holds.

    Split out from the check itself so the predicates are testable against
    injected git output rather than only against this working tree.
    """
    failures: list[str] = []

    for path, reason in MUST_BE_IGNORED.items():
        if capture(["git", "check-ignore", "-q", path]).returncode != 0:
            failures.append(f"{path} is NOT git-ignored but must be: {reason}")

    tracked = capture(["git", "ls-files", *MUST_NOT_BE_TRACKED]).stdout.split()
    if tracked:
        failures.append(
            "these paths are tracked but the policy forbids it (needs the student's"
            " approval to 'git rm --cached'): " + ", ".join(sorted(tracked))
        )

    for path in MUST_BE_TRACKED:
        if capture(["git", "ls-files", "--error-unmatch", path]).returncode != 0:
            failures.append(f"{path} must stay tracked but is not in the index")

    return failures


def check_ignore_policy() -> None:
    print("$ git check-ignore / ls-files (ignore policy)", flush=True)
    failures = ignore_policy_failures()
    if failures:
        raise CheckFailed("ignore policy violated:\n  - " + "\n  - ".join(failures))
    print(f"ok: {len(MUST_BE_IGNORED)} ignored paths, {len(MUST_BE_TRACKED)} tracked paths")


def check_bootstrap() -> None:
    """The commit skill must regenerate byte-identically from tracked sources.

    Exit code 3 from the bootstrap means "written, but the reference skill at
    ~/beacon-climb is absent" -- an honest, expected state (PRD v2.0 section
    13.1), not a failure. Any other non-zero code is a real problem.
    """
    script = [*PY, "tools/bootstrap_commit_skill.py"]
    print(f"$ {' '.join(script)}", flush=True)
    written = _capture(script)
    if written.returncode not in (0, 3):
        raise CheckFailed(
            f"bootstrap_commit_skill.py exited {written.returncode}:\n"
            f"{written.stdout}{written.stderr}".strip()
        )
    if written.returncode == 3:
        print("note: reference skill at ~/beacon-climb absent; banner recorded in SKILL.md")

    target = ROOT / ".agents" / "skills" / "commit" / "SKILL.md"
    if not target.is_file():
        raise CheckFailed(f"bootstrap did not produce {target}")

    print(f"$ {' '.join(script)} --check", flush=True)
    verified = _capture([*script, "--check"])
    if verified.returncode != 0:
        raise CheckFailed(
            "generated commit skill is stale or missing:\n"
            f"{verified.stdout}{verified.stderr}".strip()
        )


def check_readme() -> None:
    """README.rst must parse cleanly with no warnings.

    It is the project's only public document *and* the package's
    ``readme``, so a malformed directive is both a broken install page and a
    broken doc. docutils is a dev dependency; a missing one is reported as a
    missing dependency rather than silently skipped, because a check that
    quietly does nothing is worse than no check.
    """
    readme = ROOT / "README.rst"
    argv = [*PY, "-m", "docutils", "--report=warning", "--halt=warning", str(readme), os.devnull]
    print(f"$ {' '.join(argv)}", flush=True)
    completed = _capture(argv)
    if completed.returncode == 0:
        print(f"ok: {readme.name} parses with no warnings")
        return
    if "No module named docutils" in completed.stderr:
        raise CheckFailed(
            'docutils is not installed. Install the dev extras: uv pip install -e ".[dev]"'
        )
    raise CheckFailed(
        f"{readme.name} does not parse cleanly:\n{completed.stdout}{completed.stderr}".strip()
    )


def _installed_version() -> str:
    """The version of the shardscope distribution this interpreter imports."""
    argv = [*PY, "-c", "from importlib.metadata import version; print(version('shardscope'))"]
    completed = _capture(argv)
    if completed.returncode != 0:
        raise CheckFailed(
            "the shardscope distribution is not installed for this interpreter:\n"
            f"{completed.stdout}{completed.stderr}".strip()
        )
    return completed.stdout.strip()


def check_smoke() -> None:
    """The installed console script, the fixture, and the report path all run.

    The last two commands are the end-to-end artifact path: a real run
    writes its triple into a throwaway directory, and ``shardscope report``
    renders it. A unit test can pass against a fixture that no producer
    writes; this cannot.
    """
    script = shutil.which("shardscope")
    if script is None:
        raise CheckFailed(
            "the 'shardscope' console script is not on PATH. Install the package"
            ' first: uv pip install -e ".[dev]"'
        )

    # The console script and the interpreter's import must resolve to the
    # same installed distribution. A stale editable install, or another
    # shardscope earlier on PATH, is invisible to every in-process test --
    # they import the package directly and so can never disagree.
    expected = f"shardscope {_installed_version()}"
    reported = _capture([script, "--version"]).stdout.strip()
    if reported != expected:
        raise CheckFailed(
            f"the console script at {script} reports {reported!r} but this "
            f"interpreter imports {expected!r}. Reinstall with "
            'uv pip install -e ".[dev]" and check PATH for another shardscope.'
        )
    print(f"ok: {script} reports {reported}")
    with tempfile.TemporaryDirectory(prefix="shardscope-smoke-") as scratch:
        run_dir = Path(scratch) / "smoke-run"
        report = Path(scratch) / "smoke-report.html"
        for argv in (
            ["shardscope", "--version"],
            ["shardscope", "env"],
            [*PY, "examples/train_single.py", "--steps", "3", "--device", "cpu", "--quiet"],
            # Also exercises --config and flag-over-config precedence.
            [
                *PY,
                "examples/train_single.py",
                "--config",
                "examples/configs/cpu_baseline.json",
                "--steps",
                "3",
                "--quiet",
            ],
            [
                *PY,
                "examples/train_single.py",
                "--steps",
                "3",
                "--device",
                "cpu",
                "--quiet",
                "--out-dir",
                str(run_dir),
            ],
            ["shardscope", "report", "--run", str(run_dir), "--out", str(report)],
        ):
            print(f"$ {' '.join(argv)}", flush=True)
            completed = _capture(argv)
            if completed.returncode != 0:
                raise CheckFailed(
                    f"smoke command failed ({completed.returncode}): {' '.join(argv)}\n"
                    f"{completed.stdout}{completed.stderr}".strip()
                )

        if not report.is_file() or not report.read_text().startswith("<!DOCTYPE html>"):
            raise CheckFailed(f"report smoke run did not produce an HTML document: {report}")


CHECKS: tuple[Check, ...] = (
    Check("lint", "ruff lint", argv=(*PY, "-m", "ruff", "check", ".")),
    Check(
        "format-check", "ruff format --check", argv=(*PY, "-m", "ruff", "format", "--check", ".")
    ),
    Check("readme-check", "README.rst parses with no docutils warnings", run=check_readme),
    Check(
        "cov",
        "pytest with the 80% coverage gate",
        argv=(*PY, "-m", "pytest", "-q", "--cov", "--cov-report=term-missing"),
    ),
    Check("bootstrap-check", "commit skill regenerates from tracked sources", run=check_bootstrap),
    Check("ignore-policy", "Markdown/planning/secrets stay out of git", run=check_ignore_policy),
    Check("smoke", "installed CLI, its version, and the report path run", run=check_smoke),
    # Developer conveniences; not part of the CI sequence.
    Check(
        "test", "pytest without the coverage gate", argv=(*PY, "-m", "pytest", "-q"), in_ci=False
    ),
    Check(
        "format",
        "ruff format (rewrites files)",
        argv=(*PY, "-m", "ruff", "format", "."),
        in_ci=False,
    ),
)

BY_NAME = {check.name: check for check in CHECKS}

#: The exact sequence .github/workflows/ci.yml runs, in order.
CI_SEQUENCE: tuple[str, ...] = tuple(check.name for check in CHECKS if check.in_ci)


def run_checks(names: Sequence[str]) -> int:
    """Run ``names`` in order, stopping at the first failure."""
    for index, name in enumerate(names, start=1):
        check = BY_NAME[name]
        print(f"\n=== [{index}/{len(names)}] {check.name}: {check.summary} ===", flush=True)
        try:
            check.execute()
        except CheckFailed as exc:
            print(f"\nFAILED {check.name}\n{exc}", file=sys.stderr)
            return 1
    print(f"\nall {len(names)} check(s) passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/dev.py",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {check.name:<16} {check.summary}" + ("" if check.in_ci else "  (not in CI)")
            for check in CHECKS
        ),
    )
    parser.add_argument(
        "check",
        nargs="?",
        choices=(*BY_NAME, "ci"),
        help="check to run; 'ci' runs the full CI sequence",
    )
    parser.add_argument("--list", action="store_true", help="list the checks and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list or args.check is None:
        print("checks (ci runs: " + ", ".join(CI_SEQUENCE) + ")")
        for check in CHECKS:
            marker = " " if check.in_ci else "-"
            print(f"  {marker} {check.name:<16} {check.summary}")
        return 0 if args.list else 2

    return run_checks(CI_SEQUENCE if args.check == "ci" else [args.check])


if __name__ == "__main__":
    raise SystemExit(main())
