#!/usr/bin/env python3
"""Regenerate the git-ignored local commit skill from tracked sources.

The commit skill lives at ``.agents/skills/commit/SKILL.md``, which the
project's .gitignore excludes (PRD section 12.3 ignores all Markdown). This
script is the tracked, non-Markdown path that recreates it on a fresh
checkout from ``tools/commit_skill_template.txt``.

It also reports -- rather than silently assumes -- whether the reference
committing skill named in the PRD (``~/beacon-climb``) is present on this
machine.

Usage::

    python tools/bootstrap_commit_skill.py
    python tools/bootstrap_commit_skill.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TEMPLATE = Path("tools/commit_skill_template.txt")
TARGET = Path(".agents/skills/commit/SKILL.md")

#: PRD section 12.2 names this as the convention to derive from. Candidate
#: locations are checked read-only; nothing is ever copied out of them.
REFERENCE_CANDIDATES = (
    Path.home() / "beacon-climb" / ".agents" / "skills" / "commit" / "SKILL.md",
    Path.home() / "beacon-climb" / ".claude" / "skills" / "commit" / "SKILL.md",
    Path.home() / "beacon-climb" / "AGENTS.md",
    Path.home() / "beacon-climb" / "CLAUDE.md",
)

MISSING_REFERENCE = (
    "NOT FOUND on this machine. ~/beacon-climb was absent when this skill was "
    "generated, so the workflow below was derived from the ShardScope PRD "
    "(section 12.2) alone and has NOT been reconciled with the reference "
    "convention. Re-run this bootstrap once ~/beacon-climb is available."
)


def find_reference() -> Path | None:
    for candidate in REFERENCE_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def reference_status() -> tuple[str, bool]:
    found = find_reference()
    if found is None:
        return MISSING_REFERENCE, False
    return f"reviewed from {found}", True


def render(template_text: str, status: str) -> str:
    return template_text.replace("@@REFERENCE_STATUS@@", status)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the generated skill is present and current; do not write",
    )
    args = parser.parse_args(argv)

    if not TEMPLATE.is_file():
        print(f"error: missing template {TEMPLATE} (run from the repo root)", file=sys.stderr)
        return 1

    status, found = reference_status()
    rendered = render(TEMPLATE.read_text(), status)

    if args.check:
        if not TARGET.is_file():
            print(f"missing: {TARGET} -- run 'python {Path(__file__).name}'", file=sys.stderr)
            return 1
        if TARGET.read_text() != rendered:
            print(f"stale: {TARGET} differs from the tracked template", file=sys.stderr)
            return 1
        print(f"ok: {TARGET} is current")
        print(f"reference skill: {status}")
        return 0

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(rendered)
    print(f"wrote {TARGET}")
    print(f"reference skill: {status}")
    if not found:
        print(
            "\nPrerequisite not met: the PRD requires reviewing the committing skill at"
            "\n~/beacon-climb before deriving this project's convention. Supply that"
            "\ndirectory (or the skill file) and re-run to reconcile.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
