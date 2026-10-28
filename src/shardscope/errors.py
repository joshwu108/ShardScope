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
    "ValidationError",
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


class ValidationError(ShardScopeError):
    """A run directory loaded, but its artifacts do not describe one run.

    Raised by ``shardscope validate`` when the recorded artifacts disagree --
    today, when ``summary.json`` records a step count that ``steps.csv``
    does not have. Loading succeeded, so this is not an :class:`ArtifactError`;
    reporting still works and shows the disagreement, but the directory
    cannot be vouched for as a recording of a single run.
    """


class BlockedOnDecisionError(ShardScopeError):
    """A command exists as an interface target but cannot run yet.

    Raised when a required measurement contract has not been defined. The
    message names the blocking decision so the gap is legible instead of
    looking like a crash.
    """
