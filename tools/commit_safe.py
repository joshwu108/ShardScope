#!/usr/bin/env python3
"""Create test-verified local commits with path and identity checks.

What this script will do:
  * confirm the repository-local git identity matches the authorized author;
  * run the project's tests;
  * select changed paths that match an explicit allowlist (optionally
    narrowed further with ``--path``);
  * refuse, before touching the index, if the index already holds staged
    work this commit does not select, or a partially staged file;
  * stage the selected paths, run ``git diff --check`` on the result, and
    re-read ``git diff --cached --name-only`` to confirm nothing else is
    staged;
  * create one descriptive local commit.

What this script will never do:
  push, merge, rebase, amend, force-push, change remotes, create PRs, run
  ``git add -A``, ``git reset`` the index, stage ignored or force-added files,
  or commit while tests fail. Those stay manual and need separate approval
  each time.

Paths are read from git with NUL delimiters, so a file name containing a
space, tab, newline or non-ASCII character is staged as one path rather than
as several fragments or not at all.

Usage::

    python tools/commit_safe.py --message "feat: add per-step CSV export"
    python tools/commit_safe.py --message "..." --dry-run
    python tools/commit_safe.py --message "..." --path src/shardscope/cli.py --path tests
    python tools/commit_safe.py --authorize   # write the auth file template

``--dry-run`` stages into a scratch index file and leaves the real index
untouched, so its report is what the commit would stage, not an estimate.

Authorization lives in ``.shardscope-commit-auth.json`` (git-ignored) and
must be enabled explicitly before automated commits are allowed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from fnmatch import fnmatch
from pathlib import Path

AUTH_FILE = ".shardscope-commit-auth.json"

#: Paths that must never reach a commit, whatever the allowlist says.
#: Keep local configuration, planning and generated artifacts out of commits.
DENY_PATTERNS = (
    "*.md",
    "*.markdown",
    "*.mdx",
    "planning/*",
    "planning/**",
    ".planning/*",
    ".planning/**",
    "plans/*",
    "plans/**",
    "docs/planning/**",
    "docs/plans/**",
    "*_PRD.*",
    "*.prd",
    "*.plan",
    ".agents/**",
    AUTH_FILE,
    "*.trace.json",
    "*.trace.json.gz",
    "runs/**",
    "artifacts/**",
)

DEFAULT_ALLOW_PATTERNS = (
    ".gitignore",
    "pyproject.toml",
    "README.rst",
    "*.rst",
    "src/**",
    "tests/**",
    "examples/**",
    "benchmarks/**",
    "tools/**",
    ".github/**",
)

AUTH_TEMPLATE = {
    "_comment": "Local authorization for tools/commit_safe.py. Enable explicitly.",
    "authorized": False,
    "author_name": "",
    "author_email": "",
    "allow_patterns": list(DEFAULT_ALLOW_PATTERNS),
    "gpu_review_required": ["examples/train_ddp.py", "src/shardscope/capture.py"],
    "test_command": ["pytest", "-q"],
}


class CommitRefused(RuntimeError):
    """Raised when a safety precondition fails. Always fatal."""


def run(
    args: Sequence[str],
    *,
    check: bool = True,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one command, capturing text with ``surrogateescape``.

    A path git prints is bytes; decoding it strictly would turn a file name
    that is not valid UTF-8 into a crash, while ``surrogateescape`` carries
    the bytes through and ``subprocess`` encodes them back the same way when
    the path is handed to the next git command.
    """
    result = subprocess.run(
        list(args),
        capture_output=True,
        encoding="utf-8",
        errors="surrogateescape",
        cwd=cwd,
        env=env,
    )
    if check and result.returncode != 0:
        raise CommitRefused(
            f"command failed: {' '.join(args)}\n{result.stdout}{result.stderr}".strip()
        )
    return result


def repo_root() -> Path:
    return Path(run(["git", "rev-parse", "--show-toplevel"]).stdout.strip())


def load_auth(root: Path) -> dict:
    path = root / AUTH_FILE
    if not path.exists():
        raise CommitRefused(
            f"missing {AUTH_FILE}. Run 'python tools/commit_safe.py --authorize',"
            " then configure the author identity and enable local commits."
        )
    try:
        auth = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CommitRefused(f"{AUTH_FILE} is not valid JSON: {exc}") from exc

    if not auth.get("authorized"):
        raise CommitRefused(f"{AUTH_FILE} has authorized=false. Automated commits are not enabled.")
    for field in ("author_name", "author_email"):
        if not auth.get(field):
            raise CommitRefused(f"{AUTH_FILE} is missing '{field}'.")
    return auth


def verify_identity(auth: dict, root: Path | None = None) -> None:
    name = run(["git", "config", "--local", "user.name"], check=False, cwd=root).stdout.strip()
    email = run(["git", "config", "--local", "user.email"], check=False, cwd=root).stdout.strip()
    if not name or not email:
        raise CommitRefused(
            "local git identity is unset. Set 'git config --local user.name/.email'"
            " to the configured author identity before committing."
        )
    if (name, email) != (auth["author_name"], auth["author_email"]):
        raise CommitRefused(
            "local git identity does not match the authorized identity:\n"
            f"  repo: {name} <{email}>\n"
            f"  auth: {auth['author_name']} <{auth['author_email']}>"
        )


def matches_any(path: str, patterns) -> bool:
    return any(fnmatch(path, pattern) for pattern in patterns)


def _git_paths(args: Sequence[str], *, root: Path | None, env: dict[str, str] | None = None):
    """Run a path-listing git command with ``-z`` and return its paths.

    Every caller passes ``-z`` so the output is NUL-delimited and unquoted.
    Splitting on whitespace instead would break a name containing a space
    into fragments that git has never heard of, and a newline would hide
    the rest of the name entirely.
    """
    output = run(args, check=False, cwd=root, env=env).stdout
    return [path for path in output.split("\0") if path]


def changed_paths(root: Path) -> list[str]:
    """Unstaged + staged + untracked paths, root-relative, that git does not ignore.

    ``--no-renames`` lists a moved file as its deleted source *and* its new
    destination. With rename detection on, only the destination would be
    listed, the source deletion would never be staged, and the commit would
    leave the old file in the tree.
    """
    against_head = run(
        ["git", "diff", "--name-only", "-z", "--no-renames", "HEAD"], check=False, cwd=root
    )
    if against_head.returncode == 0:
        tracked = [path for path in against_head.stdout.split("\0") if path]
    else:
        # No HEAD yet (an unborn branch): compare the work tree with the index.
        tracked = _git_paths(["git", "diff", "--name-only", "-z", "--no-renames"], root=root)
    staged = staged_paths(root)
    untracked = _git_paths(["git", "ls-files", "--others", "--exclude-standard", "-z"], root=root)
    return sorted({*tracked, *staged, *untracked})


def staged_paths(root: Path, env: dict[str, str] | None = None) -> list[str]:
    """Paths whose index entry differs from HEAD."""
    return _git_paths(
        ["git", "diff", "--cached", "--name-only", "-z", "--no-renames"], root=root, env=env
    )


def unstaged_paths(root: Path) -> list[str]:
    """Tracked paths whose work-tree contents differ from their index entry."""
    return _git_paths(["git", "diff", "--name-only", "-z", "--no-renames"], root=root)


def partition(paths: list[str], allow_patterns) -> tuple[list[str], list[str]]:
    approved, skipped = [], []
    for path in paths:
        if matches_any(path, DENY_PATTERNS) or not matches_any(path, allow_patterns):
            skipped.append(path)
        else:
            approved.append(path)
    return approved, skipped


def normalize_selector(selector: str) -> str:
    """Turn a ``--path`` argument into the root-relative form git prints."""
    cleaned = selector.replace(os.sep, "/").strip("/")
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def select_paths(
    changed: list[str], approved: list[str], selectors: Iterable[str]
) -> tuple[list[str], list[str]]:
    """Narrow ``approved`` to the paths named by ``selectors``.

    A selector names one changed file, or a directory whose changed files
    it covers. Returns ``(selected, unselected)``. Refuses a selector that
    matches no changed path, because a typo that silently selects nothing
    would commit less than the caller believes, and a selector whose
    matches include a denied or non-allowlisted path, because ``--path``
    narrows the allowlist and never widens it.
    """
    selected: list[str] = []
    approved_set = set(approved)
    for raw in selectors:
        selector = normalize_selector(raw)
        matched = [p for p in changed if p == selector or p.startswith(selector + "/")]
        if not matched:
            raise CommitRefused(
                f"--path {_show(raw)} matches no changed path. Changed paths:\n  "
                + "\n  ".join(_show(p) for p in changed)
            )
        refused = [p for p in matched if p not in approved_set]
        if refused:
            raise CommitRefused(
                f"--path {_show(raw)} selects path(s) outside the allowlist or on the "
                "deny list; it cannot widen what the allowlist permits:\n  "
                + "\n  ".join(_show(p) for p in refused)
            )
        selected.extend(p for p in matched if p not in selected)
    unselected = [p for p in approved if p not in selected]
    return sorted(selected), unselected


def index_conflicts(root: Path, selected: list[str]) -> list[str]:
    """Reasons the current index cannot take this commit, empty when it can.

    Checked before anything is staged. Two conditions are refused:

    * staged paths this commit does not select. The helper never resets the
      index, so the only way to keep them out of the commit is to stop here.
    * a selected path that is staged *and* has further unstaged changes.
      Staging the whole file would fold the unstaged hunks into the commit,
      silently changing what someone chose to stage.

    A selected path that is already fully staged is fine: staging it again
    changes nothing.
    """
    staged = staged_paths(root)
    selected_set = set(selected)
    reasons: list[str] = []

    foreign = [p for p in staged if p not in selected_set]
    if foreign:
        reasons.append(
            "the index already holds staged path(s) this commit does not select. This "
            "helper does not reset the index or sweep other staged work into a commit; "
            "commit or unstage them first, or select them with --path:\n  "
            + "\n  ".join(_show(p) for p in foreign)
        )

    partially = [p for p in staged if p in selected_set and p in set(unstaged_paths(root))]
    if partially:
        reasons.append(
            "these selected path(s) are partially staged (the index holds a different "
            "version than the work tree). Staging the whole file would fold the unstaged "
            "hunks into the commit; stage all of it or unstage it first:\n  "
            + "\n  ".join(_show(p) for p in partially)
        )
    return reasons


def assert_no_tracked_planning_files(root: Path | None = None) -> None:
    """Ignore rules do not remove files that are already tracked."""
    tracked = _git_paths(["git", "ls-files", "-z"], root=root)
    offenders = [p for p in tracked if matches_any(p, DENY_PATTERNS)]
    if offenders:
        raise CommitRefused(
            "these Markdown/planning paths are already tracked; review and remove them"
            " from the index before enabling automated commits:\n  "
            + "\n  ".join(_show(p) for p in offenders)
        )


def check_staged(
    root: Path, selected: list[str], *, env: dict[str, str] | None = None
) -> list[str]:
    """Whitespace-check the staged content and confirm only ``selected`` is staged.

    Runs against whichever index ``env`` points at: the real one when
    committing, a scratch copy during a dry run. Returns the staged paths.
    """
    damage = run(["git", "diff", "--check", "--cached"], check=False, cwd=root, env=env)
    if damage.returncode != 0:
        raise CommitRefused(
            "git diff --check found whitespace or conflict-marker damage:\n"
            f"{damage.stdout}{damage.stderr}".strip()
        )
    staged = staged_paths(root, env=env)
    unexpected = [p for p in staged if p not in selected or matches_any(p, DENY_PATTERNS)]
    if unexpected:
        raise CommitRefused(
            "unexpected staged paths:\n  " + "\n  ".join(_show(p) for p in unexpected)
        )
    return staged


def stage_in_scratch_index(root: Path, selected: list[str]) -> list[str]:
    """Stage ``selected`` into a temporary index and check it; the real index is untouched.

    ``GIT_INDEX_FILE`` points git at a scratch file seeded from HEAD, so a
    dry run exercises the exact ``git add`` and ``git diff --check`` the
    commit would, including untracked files, without staging anything the
    caller then has to undo.
    """
    with tempfile.TemporaryDirectory(prefix="commit-safe-index-") as scratch:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(scratch) / "index")}
        run(["git", "read-tree", "HEAD"], cwd=root, env=env)
        run(["git", "add", "--", *selected], cwd=root, env=env)
        return check_staged(root, selected, env=env)


def stage_for_commit(root: Path, selected: list[str]) -> list[str]:
    """Stage ``selected`` into the real index and check it.

    If the check fails, only the paths this call newly staged are unstaged
    again (``git restore --staged`` on exactly those), which returns the
    index to the state it was in before the call. Paths that were already
    fully staged are not touched either way.
    """
    before = set(staged_paths(root))
    run(["git", "add", "--", *selected], cwd=root)
    try:
        return check_staged(root, selected)
    except CommitRefused as exc:
        newly = [p for p in selected if p not in before]
        if newly:
            undo = run(["git", "restore", "--staged", "--", *newly], check=False, cwd=root)
            if undo.returncode != 0:
                raise CommitRefused(
                    f"{exc}\n\nunstaging the paths this run staged also failed; the index "
                    "now holds them:\n  "
                    + "\n  ".join(_show(p) for p in newly)
                    + f"\n{undo.stderr.strip()}"
                ) from exc
        raise


def _show(path: str) -> str:
    """Spell a path unambiguously in a message.

    Plain names are shown as-is. A name containing whitespace or a
    non-printable character is shown in Python repr form so a tab, a
    newline or a trailing space is visible rather than silently rendered.
    """
    if path.isprintable() and not any(ch.isspace() for ch in path):
        return path
    return repr(path)


def _listing(paths: Iterable[str]) -> str:
    return "\n  ".join(_show(p) for p in paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--message", "-m", help="commit message (conventional commits)")
    parser.add_argument("--dry-run", action="store_true", help="report, do not commit")
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "limit the commit to this changed file or directory (repeatable); the "
            "allowlist and deny list still apply"
        ),
    )
    parser.add_argument(
        "--authorize",
        action="store_true",
        help=f"write a {AUTH_FILE} template for local commit configuration",
    )
    parser.add_argument(
        "--gpu-reviewed",
        action="store_true",
        help="confirm staged GPU-path changes have completed review",
    )
    parser.add_argument("--skip-tests", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        root = repo_root()
        if args.authorize:
            path = root / AUTH_FILE
            if path.exists():
                print(f"{AUTH_FILE} already exists; leaving it untouched.")
                return 0
            path.write_text(json.dumps(AUTH_TEMPLATE, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {path}. Configure the author and set authorized=true to enable commits.")
            return 0

        if not args.message:
            raise CommitRefused("--message is required")

        auth = load_auth(root)
        verify_identity(auth, root)
        assert_no_tracked_planning_files(root)

        changed = changed_paths(root)
        approved, skipped = partition(changed, auth["allow_patterns"])
        if args.path:
            selected, unselected = select_paths(changed, approved, args.path)
        else:
            selected, unselected = approved, []
        if not selected:
            print("nothing to commit within the allowlist.")
            return 0

        gpu_paths = [p for p in selected if matches_any(p, auth.get("gpu_review_required", []))]
        if gpu_paths and not args.gpu_reviewed:
            raise CommitRefused(
                "these paths require a completed GPU review"
                " (rerun with --gpu-reviewed after review):\n  " + _listing(gpu_paths)
            )

        conflicts = index_conflicts(root, selected)
        if conflicts:
            raise CommitRefused("\n".join(conflicts))

        if not args.skip_tests:
            test_cmd = auth.get("test_command") or ["pytest", "-q"]
            print(f"running tests: {' '.join(test_cmd)}")
            tests = subprocess.run(test_cmd, cwd=root)
            if tests.returncode != 0:
                raise CommitRefused("tests failed; refusing to commit")

        if args.dry_run:
            staged = stage_in_scratch_index(root, selected)
        else:
            staged = stage_for_commit(root, selected)

        print("staged:\n  " + _listing(staged))
        if unselected:
            print("not staged (allowed, but not selected by --path):\n  " + _listing(unselected))
        if skipped:
            print("not staged (outside allowlist or denied):\n  " + _listing(skipped))

        if args.dry_run:
            print("dry run: nothing committed; the index was not modified")
            return 0

        run(["git", "commit", "-m", args.message], cwd=root)
        print("committed locally (not pushed):")
        print(run(["git", "log", "-1", "--stat", "--format=%h %an <%ae> %s"], cwd=root).stdout)
        return 0
    except CommitRefused as exc:
        print(f"commit refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
