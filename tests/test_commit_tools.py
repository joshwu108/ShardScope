"""Commit tooling checks for path restrictions and local authorization.

Denied paths must override allowlists. Bootstrap must preserve unrelated local
files and report unavailable reference conventions accurately.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
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
    """Authorization is explicit local configuration.

    ``--authorize`` must preserve existing settings and create disabled
    templates so initialization cannot silently enable commits.
    """

    def test_an_existing_auth_file_is_left_byte_for_byte_alone(self, tmp_path, monkeypatch, capsys):
        recorded = (
            '{\n  "authorized": true,\n  "author_name": "Developer",\n'
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
            '{"authorized": true, "author_name": "Developer"}\n', encoding="utf-8"
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


#: File names git must carry through the helper as single paths. Each one
#: breaks whitespace splitting in a different way: a space, a tab, a newline
#: and a non-ASCII name that ``str.split`` leaves alone but a locale decode
#: might not.
AWKWARD_NAMES = (
    "src/with space.py",
    "src/with\ttab.py",
    "src/with\nnewline.py",
    "src/größe-ünïcode.py",
)

AUTHOR = ("Test Author", "author@example.invalid")


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        encoding="utf-8",
        errors="surrogateescape",
    )
    if check and result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stdout}{result.stderr}")
    return result


def _staged(repo: Path) -> list[str]:
    out = _git(repo, "diff", "--cached", "--name-only", "-z", "--no-renames").stdout
    return sorted(p for p in out.split("\0") if p)


def _committed(repo: Path) -> list[str]:
    out = _git(repo, "show", "--name-only", "--format=", "-z", "--no-renames", "HEAD").stdout
    return sorted(p for p in out.split("\0") if p)


def _write(repo: Path, relative: str, text: str = "x = 1\n") -> Path:
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A disposable repository the helper can commit into.

    Identity, authorization and the test command are all local to this
    directory; nothing here reads or writes the developer's own
    configuration. The test command is a no-op interpreter call so these
    tests exercise the staging rules, not the project's test suite.
    """
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "--local", "user.name", AUTHOR[0])
    _git(root, "config", "--local", "user.email", AUTHOR[1])
    _git(root, "config", "--local", "commit.gpgsign", "false")
    _write(root, "src/base.py", "base = 1\n")
    _write(root, "README.rst", "Readme\n")
    _write(root, ".gitignore", f"{commit_safe.AUTH_FILE}\n")
    _git(root, "add", "--", "src/base.py", "README.rst", ".gitignore")
    _git(root, "commit", "-q", "-m", "initial")
    (root / commit_safe.AUTH_FILE).write_text(
        json.dumps(
            {
                "authorized": True,
                "author_name": AUTHOR[0],
                "author_email": AUTHOR[1],
                "allow_patterns": list(commit_safe.DEFAULT_ALLOW_PATTERNS),
                "gpu_review_required": ["src/gpu_only.py"],
                "test_command": [sys.executable, "-c", "pass"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(root)
    return root


class TestPathsSurviveWhitespace:
    """Git prints paths NUL-delimited; the helper must read them that way."""

    def test_changed_paths_lists_awkward_names_whole(self, repo):
        for name in AWKWARD_NAMES:
            _write(repo, name)
        _write(repo, "src/base.py", "base = 2\n")

        assert commit_safe.changed_paths(repo) == sorted([*AWKWARD_NAMES, "src/base.py"])

    def test_a_commit_carries_awkward_names_as_single_paths(self, repo, capsys):
        for name in AWKWARD_NAMES:
            _write(repo, name)

        assert commit_safe.main(["--message", "feat: awkward names"]) == 0

        assert _committed(repo) == sorted(AWKWARD_NAMES)
        assert _git(repo, "status", "--porcelain").stdout == ""
        out = capsys.readouterr().out
        assert "committed locally (not pushed)" in out
        for name in AWKWARD_NAMES:
            assert commit_safe._show(name) in out

    def test_a_move_stages_both_the_deletion_and_the_new_name(self, repo):
        (repo / "src" / "base.py").rename(repo / "src" / "renamed.py")

        assert commit_safe.main(["--message", "refactor: rename"]) == 0

        assert _committed(repo) == ["src/base.py", "src/renamed.py"]
        tracked = _git(repo, "ls-files", "-z").stdout.split("\0")
        assert "src/base.py" not in tracked
        assert "src/renamed.py" in tracked


class TestDryRunLeavesTheIndexAlone:
    def test_dry_run_reports_the_staging_without_staging(self, repo, capsys):
        _write(repo, "src/new.py")
        _write(repo, AWKWARD_NAMES[0])

        assert commit_safe.main(["--message", "feat: x", "--dry-run"]) == 0

        assert _staged(repo) == []
        out = capsys.readouterr().out
        assert "src/new.py" in out
        assert commit_safe._show(AWKWARD_NAMES[0]) in out
        assert "the index was not modified" in out
        assert _git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "1"

    def test_dry_run_keeps_a_pre_staged_entry_byte_for_byte(self, repo):
        _write(repo, "src/base.py", "base = 2\n")
        _git(repo, "add", "--", "src/base.py")
        before = _git(repo, "ls-files", "--stage", "-z").stdout

        assert commit_safe.main(["--message", "fix: base", "--dry-run"]) == 0

        assert _git(repo, "ls-files", "--stage", "-z").stdout == before

    def test_dry_run_works_before_the_first_commit(self, tmp_path, monkeypatch, capsys):
        root = tmp_path / "unborn"
        root.mkdir()
        _git(root, "init", "-q", "-b", "main")
        _git(root, "config", "--local", "user.name", AUTHOR[0])
        _git(root, "config", "--local", "user.email", AUTHOR[1])
        _write(root, ".gitignore", f"{commit_safe.AUTH_FILE}\n")
        _write(root, "src/first.py")
        (root / commit_safe.AUTH_FILE).write_text(
            json.dumps(
                {
                    "authorized": True,
                    "author_name": AUTHOR[0],
                    "author_email": AUTHOR[1],
                    "allow_patterns": list(commit_safe.DEFAULT_ALLOW_PATTERNS),
                    "test_command": [sys.executable, "-c", "pass"],
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.chdir(root)

        assert commit_safe.main(["--message", "feat: first", "--dry-run"]) == 0

        assert _staged(root) == []
        assert "src/first.py" in capsys.readouterr().out

    def test_dry_run_still_catches_whitespace_damage(self, repo, capsys):
        _write(repo, "src/trailing.py", "x = 1   \n")

        assert commit_safe.main(["--message", "feat: x", "--dry-run"]) == 1

        assert "whitespace" in capsys.readouterr().err
        assert _staged(repo) == []


class TestUnrelatedStagedWorkIsPreserved:
    """The helper never resets the index; it refuses instead."""

    def test_refuses_when_the_index_holds_a_path_outside_the_commit(self, repo, capsys):
        _write(repo, "notes.txt", "someone else's staged work\n")
        _git(repo, "add", "--", "notes.txt")
        before = _git(repo, "ls-files", "--stage", "-z").stdout
        _write(repo, "src/new.py")

        assert commit_safe.main(["--message", "feat: x"]) == 1

        err = capsys.readouterr().err
        assert "does not select" in err
        assert "notes.txt" in err
        assert _git(repo, "ls-files", "--stage", "-z").stdout == before
        assert _git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "1"

    def test_refuses_a_partially_staged_selected_file(self, repo, capsys):
        _write(repo, "src/base.py", "base = 2\n")
        _git(repo, "add", "--", "src/base.py")
        _write(repo, "src/base.py", "base = 3\n")
        before = _git(repo, "ls-files", "--stage", "-z").stdout

        assert commit_safe.main(["--message", "fix: base"]) == 1

        assert "partially staged" in capsys.readouterr().err
        assert _git(repo, "ls-files", "--stage", "-z").stdout == before

    def test_a_fully_staged_selected_file_is_committed(self, repo):
        _write(repo, "src/base.py", "base = 2\n")
        _git(repo, "add", "--", "src/base.py")

        assert commit_safe.main(["--message", "fix: base"]) == 0

        assert _committed(repo) == ["src/base.py"]

    def test_a_wildcard_name_cannot_touch_a_staged_sibling(self, repo, capsys):
        # ``--`` paths are glob pathspecs unless told otherwise: without
        # GIT_LITERAL_PATHSPECS, ``src/a[1].py`` also matched ``src/a1.py``,
        # so the undo after a failed check unstaged the sibling's work.
        _write(repo, "src/a1.py", "staged = 1\n")
        _git(repo, "add", "--", "src/a1.py")
        before = _git(repo, "ls-files", "--stage", "-z").stdout
        _write(repo, "src/a[1].py", "x = 1   \n")
        _write(repo, "src/a?.py", "y = 1\n")

        assert commit_safe.main(["--message", "fix: x"]) == 1

        assert "whitespace" in capsys.readouterr().err
        assert _git(repo, "ls-files", "--stage", "-z").stdout == before

    def test_a_wildcard_name_is_committed_as_itself(self, repo):
        _write(repo, "src/a1.py")
        _write(repo, "src/a[1].py")

        assert commit_safe.main(["--message", "feat: x", "--path", "src/a[1].py"]) == 0

        assert _committed(repo) == ["src/a[1].py"]
        assert _staged(repo) == []

    def test_a_failed_check_unstages_only_what_this_run_staged(self, repo, capsys):
        _write(repo, "src/base.py", "base = 2\n")
        _git(repo, "add", "--", "src/base.py")
        _write(repo, "src/trailing.py", "x = 1   \n")
        before = _git(repo, "ls-files", "--stage", "-z").stdout

        assert commit_safe.main(["--message", "fix: x"]) == 1

        assert "whitespace" in capsys.readouterr().err
        assert _staged(repo) == ["src/base.py"]
        assert _git(repo, "ls-files", "--stage", "-z").stdout == before


class TestExplicitPathSelection:
    def test_path_limits_the_commit_to_one_slice(self, repo, capsys):
        _write(repo, "src/one.py")
        _write(repo, "src/two.py")
        _write(repo, "tests/test_one.py")

        assert commit_safe.main(["--message", "feat: one", "--path", "src/one.py"]) == 0

        assert _committed(repo) == ["src/one.py"]
        assert _staged(repo) == []
        out = capsys.readouterr().out
        assert "not selected by --path" in out
        assert "src/two.py" in out
        assert "tests/test_one.py" in out

    def test_a_directory_selector_covers_its_changed_files(self, repo):
        _write(repo, "src/one.py")
        _write(repo, "tests/test_one.py")
        _write(repo, "tests/test_two.py")

        assert commit_safe.main(["--message", "test: both", "--path", "tests/"]) == 0

        assert _committed(repo) == ["tests/test_one.py", "tests/test_two.py"]

    def test_a_selector_matching_nothing_is_refused(self, repo, capsys):
        _write(repo, "src/one.py")

        assert commit_safe.main(["--message", "feat: x", "--path", "src/typo.py"]) == 1

        err = capsys.readouterr().err
        assert "matches no changed path" in err
        assert _git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "1"

    def test_a_selector_cannot_widen_the_allowlist(self, repo, capsys):
        _write(repo, "planning/plan.md", "secret plan\n")
        _write(repo, "src/one.py")

        assert commit_safe.main(["--message", "feat: x", "--path", "planning/plan.md"]) == 1

        assert "cannot widen" in capsys.readouterr().err
        assert _staged(repo) == []

    def test_gpu_paths_still_need_review_when_selected(self, repo, capsys):
        _write(repo, "src/gpu_only.py")

        assert commit_safe.main(["--message", "feat: gpu", "--path", "src/gpu_only.py"]) == 1

        assert "GPU review" in capsys.readouterr().err
        assert _staged(repo) == []
