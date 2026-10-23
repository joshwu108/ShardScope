"""Tests for the tracked commit tooling (PRD v2.0 sections 13.1 / 13.2).

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


class TestAuthorizationIsNeverOverwritten:
    """The auth file is the student's recorded consent, not agent state.

    An agent that could rewrite it could grant itself permission, so
    ``--authorize`` only ever creates the file, and the template it creates
    is unauthorized.
    """

    def test_an_existing_auth_file_is_left_byte_for_byte_alone(self, tmp_path, monkeypatch, capsys):
        recorded = (
            '{\n  "authorized": true,\n  "author_name": "student",\n'
            '  "author_email": "s@example.com",\n  "allow_patterns": ["src/**"]\n}\n'
        )
        auth = tmp_path / commit_safe.AUTH_FILE
        auth.write_text(recorded, encoding="utf-8")
        monkeypatch.setattr(commit_safe, "repo_root", lambda: tmp_path)

        assert commit_safe.main(["--authorize"]) == 0

        assert auth.read_text(encoding="utf-8") == recorded
        assert "leaving it untouched" in capsys.readouterr().out

    def test_the_created_template_is_not_pre_authorized(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(commit_safe, "repo_root", lambda: tmp_path)

        assert commit_safe.main(["--authorize"]) == 0

        payload = json.loads((tmp_path / commit_safe.AUTH_FILE).read_text(encoding="utf-8"))
        assert payload["authorized"] is False
        assert payload["author_name"] == ""


class TestBootstrapWritesOnlyTheSkill:
    """Regenerating the skill must not disturb anything else on disk.

    ``.agents/`` can hold hand-written instructions and the repo root holds
    the auth file. Both are git-ignored, so git would not notice them being
    clobbered -- this test is the only thing that would.
    """

    @staticmethod
    def _snapshot(root: Path) -> dict[str, bytes]:
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    @pytest.fixture
    def sandbox(self, tmp_path, monkeypatch):
        """A miniature repo: the tracked template plus local-only files."""
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "commit_skill_template.txt").write_text(
            "# skill\n\nReference: @@REFERENCE_STATUS@@\n", encoding="utf-8"
        )
        (tmp_path / commit_safe.AUTH_FILE).write_text(
            '{"authorized": true, "author_name": "student"}\n', encoding="utf-8"
        )
        (tmp_path / ".agents").mkdir()
        (tmp_path / ".agents" / "custom-instructions.md").write_text(
            "hand-written, local only\n", encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def test_only_the_skill_file_appears(self, sandbox, capsys):
        before = self._snapshot(sandbox)

        # 3 is the honest "written, but ~/beacon-climb is absent" code; 0 is
        # the same write with the reference present. Either is a write.
        assert bootstrap.main([]) in (0, 3)

        after = self._snapshot(sandbox)
        assert set(after) - set(before) == {str(bootstrap.TARGET)}
        unchanged = {path: blob for path, blob in after.items() if path in before}
        assert unchanged == before

    def test_the_regenerated_skill_verifies_as_current(self, sandbox, capsys):
        bootstrap.main([])

        assert bootstrap.main(["--check"]) == 0

    def test_check_reports_a_hand_edited_skill_as_stale(self, sandbox, capsys):
        bootstrap.main([])
        (sandbox / bootstrap.TARGET).write_text("edited by hand\n", encoding="utf-8")

        assert bootstrap.main(["--check"]) == 1
        assert "stale" in capsys.readouterr().err

    def test_check_does_not_write_the_skill(self, sandbox, capsys):
        assert bootstrap.main(["--check"]) == 1
        assert not (sandbox / bootstrap.TARGET).exists()
        assert "missing" in capsys.readouterr().err
