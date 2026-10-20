"""Tests for the tracked commit tooling (PRD section 12.2 / 12.3).

These guard the safety rules, which matter more than the convenience: the
deny list must beat the allow list, Markdown and planning documents must
never be stageable, and the bootstrap must be honest about the missing
reference skill.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


commit_safe = _load("commit_safe")
bootstrap = _load("bootstrap_commit_skill")


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "notes.markdown",
        "page.mdx",
        "planning/design.md",
        "planning/PRD.docx",
        "plans/week1.txt",
        "ShardScope_PRD.md",
        "scratch.plan",
        ".agents/skills/commit/SKILL.md",
        ".shardscope-commit-auth.json",
        "runs/rank0.pt.trace.json",
    ],
)
def test_denied_paths_are_never_approved(path):
    approved, skipped = commit_safe.partition([path], commit_safe.DEFAULT_ALLOW_PATTERNS)
    assert approved == []
    assert skipped == [path]


@pytest.mark.parametrize(
    "path",
    [
        ".gitignore",
        "pyproject.toml",
        "README.rst",
        "src/shardscope/cli.py",
        "tests/test_env.py",
        "examples/train_single.py",
        "benchmarks/results.json",
        "tools/commit_safe.py",
        ".github/workflows/ci.yml",
    ],
)
def test_project_source_paths_are_approved(path):
    approved, skipped = commit_safe.partition([path], commit_safe.DEFAULT_ALLOW_PATTERNS)
    assert approved == [path]
    assert skipped == []


def test_deny_list_overrides_an_overbroad_allow_list():
    # Even if someone widens the allowlist to everything, policy holds.
    approved, _ = commit_safe.partition(["planning/secret.md"], ["*", "**"])
    assert approved == []


def test_load_auth_refuses_when_not_authorized(tmp_path):
    (tmp_path / commit_safe.AUTH_FILE).write_text(
        json.dumps({"authorized": False, "author_name": "A", "author_email": "a@b.c"})
    )
    with pytest.raises(commit_safe.CommitRefused, match="authorized=false"):
        commit_safe.load_auth(tmp_path)


def test_load_auth_refuses_when_file_is_absent(tmp_path):
    with pytest.raises(commit_safe.CommitRefused, match="missing"):
        commit_safe.load_auth(tmp_path)


def test_load_auth_refuses_when_identity_is_blank(tmp_path):
    (tmp_path / commit_safe.AUTH_FILE).write_text(
        json.dumps({"authorized": True, "author_name": "", "author_email": "a@b.c"})
    )
    with pytest.raises(commit_safe.CommitRefused, match="author_name"):
        commit_safe.load_auth(tmp_path)


def test_bootstrap_reports_missing_reference_rather_than_claiming_review(monkeypatch):
    monkeypatch.setattr(bootstrap, "find_reference", lambda: None)
    status, found = bootstrap.reference_status()
    assert not found
    assert "NOT FOUND" in status


def test_bootstrap_reports_the_reference_when_present(monkeypatch, tmp_path):
    reference = tmp_path / "SKILL.md"
    reference.write_text("x")
    monkeypatch.setattr(bootstrap, "find_reference", lambda: reference)
    status, found = bootstrap.reference_status()
    assert found
    assert str(reference) in status


def test_rendered_skill_has_no_unfilled_placeholder():
    template = (TOOLS / "commit_skill_template.txt").read_text()
    rendered = bootstrap.render(template, "reviewed from somewhere")
    assert "@@REFERENCE_STATUS@@" not in rendered
    assert "Never `git push`" in rendered
