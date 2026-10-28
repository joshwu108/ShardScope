"""Packaging metadata consistency checks.

The installed distribution and package attribute must report the same version
so recorded provenance identifies a single build.
"""

from __future__ import annotations

import re
import sys
from importlib.metadata import entry_points, version
from pathlib import Path

import pytest

import shardscope

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on the 3.10 CI leg
    import tomli as tomllib

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def test_installed_version_matches_the_package_attribute():
    assert version("shardscope") == shardscope.__version__


def test_pyproject_declares_the_version_dynamically(pyproject):
    # A literal `version = "..."` here would be a second source of truth.
    assert "version" in pyproject["project"]["dynamic"]
    assert "version" not in pyproject["project"]
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "shardscope.__version__"
    }


def test_console_script_points_at_the_cli_entry_point():
    scripts = {ep.name: ep.value for ep in entry_points(group="console_scripts")}
    assert scripts.get("shardscope") == "shardscope.cli:main"


def _supported_minors(requires_python: str) -> set[int]:
    """Minor versions of Python 3 allowed by a ``>=3.a,<3.b`` specifier."""
    lower, upper = (part.strip() for part in requires_python.split(","))
    low = int(lower.removeprefix(">=").split(".")[1])
    high = int(upper.removeprefix("<").split(".")[1])
    return set(range(low, high))


def test_ci_matrix_only_tests_supported_python_versions(pyproject):
    # requires-python and the CI matrix drifting apart means CI either tests
    # a version the package disclaims or skips one it promises.
    workflow = (PYPROJECT.parent / ".github" / "workflows" / "ci.yml").read_text()
    supported = _supported_minors(pyproject["project"]["requires-python"])
    matrix = {int(m) for m in re.findall(r'"3\.(\d+)"', workflow)}

    assert matrix, "no python-version matrix found in ci.yml"
    assert matrix <= supported, f"CI tests unsupported versions: {sorted(matrix - supported)}"
    assert min(supported) in matrix, "CI does not test the oldest supported Python"
    assert max(supported) in matrix, "CI does not test the newest supported Python"


def test_the_dev_extra_declares_every_tool_a_ci_gate_shells_out_to(pyproject):
    # A gate that cannot run after `uv pip install -e ".[dev]"` is a gate
    # that fails for the wrong reason, which trains people to ignore it.
    declared = {
        re.split(r"[<>=!;\s]", entry, maxsplit=1)[0]
        for entry in pyproject["project"]["optional-dependencies"]["dev"]
    }

    assert {"pytest", "pytest-cov", "ruff", "docutils"} <= declared


def test_every_marker_the_suite_relies_on_is_declared(pyproject):
    # --strict-markers turns an undeclared marker into an error; naming the
    # expected set here makes that failure legible.
    ini = pyproject["tool"]["pytest"]["ini_options"]
    declared = {entry.split(":", 1)[0] for entry in ini["markers"]}

    assert {"gpu", "distributed", "slow"} <= declared
    assert "--strict-markers" in ini["addopts"]
