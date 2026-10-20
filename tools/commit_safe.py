#!/usr/bin/env python3
"""Safe, auditable local commits for ShardScope (PRD section 12.2).

What this script will do:
  * confirm the repository's local git identity matches the identity the
    student authorized;
  * run the project's tests;
  * run ``git diff --check`` for whitespace/conflict damage;
  * stage only paths matching an explicit allowlist;
  * re-read ``git diff --cached --name-only`` and abort if anything outside
    the allowlist, or anything Markdown/planning/ignored, is staged;
  * create one descriptive local commit.

What this script will never do:
  push, merge, rebase, amend, force-push, change remotes, create PRs, run
  ``git add -A``, stage ignored or force-added files, or commit while tests
  fail. Those stay manual and need separate approval each time.

Usage::

    python tools/commit_safe.py --message "feat: add per-step CSV export"
    python tools/commit_safe.py --message "..." --dry-run
    python tools/commit_safe.py --authorize   # write the auth file template

Authorization lives in ``.shardscope-commit-auth.json`` (git-ignored) and
must be created or edited by the student, not by an agent.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

AUTH_FILE = ".shardscope-commit-auth.json"

#: Paths that must never reach a commit, whatever the allowlist says.
#: Mirrors the .gitignore policy in PRD section 12.3.
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
    "_comment": "Student-owned authorization for tools/commit_safe.py. Edit by hand.",
    "authorized": False,
    "author_name": "",
    "author_email": "",
    "allow_patterns": list(DEFAULT_ALLOW_PATTERNS),
    "gpu_review_required": ["examples/train_ddp.py", "src/shardscope/capture.py"],
    "test_command": ["pytest", "-q"],
}


class CommitRefused(RuntimeError):
    """Raised when a safety precondition fails. Always fatal."""


def run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, capture_output=True, text=True)
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
            " then have the student fill in and approve it."
        )
    try:
        auth = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CommitRefused(f"{AUTH_FILE} is not valid JSON: {exc}") from exc

    if not auth.get("authorized"):
        raise CommitRefused(f"{AUTH_FILE} has authorized=false. Automated commits are not enabled.")
    for field in ("author_name", "author_email"):
        if not auth.get(field):
            raise CommitRefused(f"{AUTH_FILE} is missing '{field}'.")
    return auth


def verify_identity(auth: dict) -> None:
    name = run(["git", "config", "--local", "user.name"], check=False).stdout.strip()
    email = run(["git", "config", "--local", "user.email"], check=False).stdout.strip()
    if not name or not email:
        raise CommitRefused(
            "local git identity is unset. Set 'git config --local user.name/.email'"
            " to the student's verified identity before committing."
        )
    if (name, email) != (auth["author_name"], auth["author_email"]):
        raise CommitRefused(
            "local git identity does not match the authorized identity:\n"
            f"  repo: {name} <{email}>\n"
            f"  auth: {auth['author_name']} <{auth['author_email']}>"
        )


def matches_any(path: str, patterns) -> bool:
    return any(fnmatch(path, pattern) for pattern in patterns)


def changed_paths(root: Path) -> list[str]:
    """Unstaged + staged + untracked paths that git does not ignore."""
    tracked = run(["git", "diff", "--name-only", "HEAD"], check=False).stdout.split()
    if not tracked:
        tracked = run(["git", "diff", "--name-only"], check=False).stdout.split()
    staged = run(["git", "diff", "--cached", "--name-only"], check=False).stdout.split()
    untracked = run(
        ["git", "ls-files", "--others", "--exclude-standard"], check=False
    ).stdout.split()
    return sorted({*tracked, *staged, *untracked})


def partition(paths: list[str], allow_patterns) -> tuple[list[str], list[str]]:
    approved, skipped = [], []
    for path in paths:
        if matches_any(path, DENY_PATTERNS) or not matches_any(path, allow_patterns):
            skipped.append(path)
        else:
            approved.append(path)
    return approved, skipped


def assert_no_tracked_planning_files() -> None:
    """PRD 12.3: .gitignore does not untrack files already tracked."""
    tracked = run(["git", "ls-files"], check=False).stdout.split()
    offenders = [p for p in tracked if matches_any(p, DENY_PATTERNS)]
    if offenders:
        raise CommitRefused(
            "these Markdown/planning paths are already tracked and need the student's"
            " approval to 'git rm --cached' before automated commits are safe:\n  "
            + "\n  ".join(offenders)
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--message", "-m", help="commit message (conventional commits)")
    parser.add_argument("--dry-run", action="store_true", help="report, do not commit")
    parser.add_argument(
        "--authorize",
        action="store_true",
        help=f"write a {AUTH_FILE} template for the student to fill in",
    )
    parser.add_argument(
        "--gpu-reviewed",
        action="store_true",
        help="assert the student has reviewed and approved staged GPU-path changes",
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
            path.write_text(json.dumps(AUTH_TEMPLATE, indent=2) + "\n")
            print(f"wrote {path}. The student must set authorized=true and the identity.")
            return 0

        if not args.message:
            raise CommitRefused("--message is required")

        auth = load_auth(root)
        verify_identity(auth)
        assert_no_tracked_planning_files()

        approved, skipped = partition(changed_paths(root), auth["allow_patterns"])
        if not approved:
            print("nothing to commit within the allowlist.")
            return 0

        gpu_paths = [p for p in approved if matches_any(p, auth.get("gpu_review_required", []))]
        if gpu_paths and not args.gpu_reviewed:
            raise CommitRefused(
                "these paths are marked GPU-review-required and need the student's explicit"
                " sign-off (rerun with --gpu-reviewed once approved):\n  " + "\n  ".join(gpu_paths)
            )

        if not args.skip_tests:
            test_cmd = auth.get("test_command") or ["pytest", "-q"]
            print(f"running tests: {' '.join(test_cmd)}")
            tests = subprocess.run(test_cmd, cwd=root)
            if tests.returncode != 0:
                raise CommitRefused("tests failed; refusing to commit")

        run(["git", "reset"], check=False)
        run(["git", "add", "--", *approved])
        run(["git", "diff", "--check", "--cached"])

        staged = run(["git", "diff", "--cached", "--name-only"]).stdout.split()
        unexpected = [p for p in staged if p not in approved or matches_any(p, DENY_PATTERNS)]
        if unexpected:
            run(["git", "reset"], check=False)
            raise CommitRefused("unexpected staged paths:\n  " + "\n  ".join(unexpected))

        print("staged:\n  " + "\n  ".join(staged))
        if skipped:
            print("not staged (outside allowlist or denied):\n  " + "\n  ".join(skipped))

        if args.dry_run:
            run(["git", "reset"], check=False)
            print("dry run: nothing committed")
            return 0

        run(["git", "commit", "-m", args.message])
        print("committed locally (not pushed):")
        print(run(["git", "log", "-1", "--stat", "--format=%h %an <%ae> %s"]).stdout)
        return 0
    except CommitRefused as exc:
        print(f"commit refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
