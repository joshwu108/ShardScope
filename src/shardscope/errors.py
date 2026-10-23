"""Exception types shared by the ShardScope library and CLI.

One base class exists so the CLI can turn any expected, user-caused failure
into a short actionable line on stderr and a non-zero exit, while a genuine
bug still escapes as an unhandled traceback. Swallowing the difference would
hide real defects behind a tidy error message.
"""

from __future__ import annotations

__all__ = [
    "ArtifactError",
    "BlockedOnDecisionError",
    "ConfigError",
    "ReportError",
    "ShardScopeError",
]


class ShardScopeError(Exception):
    """Base class for failures a user can act on.

    The message must say what was wrong *and* what to do about it. ``str()``
    of this exception is printed verbatim to the user.
    """


class ConfigError(ShardScopeError):
    """A run configuration file is missing, malformed, or out of range."""


class ArtifactError(ShardScopeError):
    """A run artifact is missing, unreadable, or does not match its schema.

    Raised while discovering, loading or validating the files a run writes.
    The message must name the offending file -- and the row, key or column
    inside it -- because the usual cause is a hand-edited or half-written
    run directory, and "invalid artifact" alone does not say which one.
    """


class ReportError(ShardScopeError):
    """A report cannot be rendered as asked for.

    Covers an unknown output format and rendering options that would produce
    a misleading document (for example a row cap of zero, which would render
    a run as if it had measured nothing).
    """


class BlockedOnDecisionError(ShardScopeError):
    """A command exists as an interface target but cannot run yet.

    Raised where a feature is blocked on a decision the student owns (PRD
    v2.0 section 13 approval gate) rather than on unwritten plumbing. The
    message names the blocking decision so the gap is legible instead of
    looking like a crash.
    """
