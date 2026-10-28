"""Command-line entry point for ShardScope.

Subcommand split mirrors PRD requirement SS-04: diagnostic (profiled) runs
and throughput (unprofiled) benchmarks are different commands, and their
numbers are never presented as interchangeable.

    shardscope env                 # reproducibility snapshot (implemented)
    shardscope validate ...        # say whether a run directory can be reported (implemented)
    shardscope report ...          # render HTML/JSON/CSV from a run (implemented)
    shardscope profile ...         # bounded profiled diagnostic run (not implemented)
    shardscope bench ...           # warm, unprofiled throughput run (not implemented)
    shardscope compare ...         # before/after with variance (not implemented)

``report`` renders the artifacts a run already wrote -- the per-step CSV,
the provenance dump and the recorded statistics -- and states, field by
field, what the run does *not* contain and why. It adds no measurement: the
profiled per-step schema is GATE 3.1/3.2 and has no reader yet, so a CSV
carrying those columns is refused rather than partially rendered.

``profile``, ``bench`` and ``compare`` are declared but unimplemented. The
argument names come from PRD v2.0 section 11, so ``--help`` documents the
agreed interface target; invoking one exits non-zero and names the decision
it waits on, rather than pretending to work. Memory Lab commands
(``memory-sweep``, ``memory-report``) are deliberately absent: PRD v2.0
section 10 gates that whole block on a review that has not happened.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from shardscope import __version__
from shardscope.artifacts import load_run
from shardscope.env import capture_environment
from shardscope.errors import BlockedOnDecisionError, ShardScopeError, ValidationError
from shardscope.report import (
    DEFAULT_MAX_ROWS,
    REPORT_FORMATS,
    format_for_path,
    render,
    write_report,
)
from shardscope.validate import validate_run

#: Exit code for "the reader of our stdout pipe hung up": 128 + SIGPIPE, the
#: shell convention. Distinct from 2 (the user's mistake) and 0 (success), so
#: a calling script can tell a short read from a bad argument.
EXIT_BROKEN_PIPE = 141


@dataclass(frozen=True)
class Option:
    """One declared command-line option for a not-yet-implemented command."""

    flag: str
    metavar: str
    help: str


@dataclass(frozen=True)
class PendingCommand:
    """A command that exists as an interface target but cannot run yet.

    ``blocked_on`` must name the specific open decision, not just say
    "unimplemented": the difference between "nobody has written this" and
    "this is waiting on a measurement contract the owner has not settled" is
    the whole point of the PRD v2.0 section 13 approval gate.
    """

    name: str
    summary: str
    blocked_on: str
    options: tuple[Option, ...] = field(default_factory=tuple)

    @property
    def help(self) -> str:
        return f"{self.summary} (not implemented)"

    def message(self) -> str:
        return f"'{self.name}' is not implemented yet. Blocked on: {self.blocked_on}"


_CONFIG_OPTION = Option(
    "--config", "FILE", "JSON run config; see shardscope.config for the validated keys"
)

PENDING_COMMANDS: tuple[PendingCommand, ...] = (
    PendingCommand(
        name="profile",
        summary="bounded profiled diagnostic run",
        blocked_on=(
            "the capture contract in src/shardscope/capture.py -- the profiler "
            "schedule (GATE 2.1), the activity set and extras (GATE 2.2) and the "
            "record_function marker names every later rule keys off (GATE 2.3) "
            "are owner decisions"
        ),
        options=(
            _CONFIG_OPTION,
            Option("--out", "DIR", "per-run artifact directory to create"),
        ),
    ),
    PendingCommand(
        name="bench",
        summary="warm, never-profiled throughput run",
        blocked_on=(
            "the benchmark methodology in benchmarks/run_baseline.py -- warmup count, "
            "repeat count and the test used to call a difference real (GATE 7.1) are "
            "owner decisions"
        ),
        options=(
            _CONFIG_OPTION,
            Option("--out", "DIR", "unprofiled benchmark record to write"),
        ),
    ),
    PendingCommand(
        name="compare",
        summary="before/after comparison of two unprofiled runs",
        blocked_on=(
            "the same GATE 7.1 methodology decision as 'bench': without an agreed "
            "variance test this command could only report a percentage it cannot "
            "defend"
        ),
        options=(
            Option("--baseline", "DIR", "unprofiled baseline run directory"),
            Option("--candidate", "DIR", "unprofiled candidate run directory"),
        ),
    ),
)

_PENDING_BY_NAME = {command.name: command for command in PENDING_COMMANDS}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shardscope",
        description="Diagnose where a PyTorch training step loses time.",
        epilog=(
            "Throughput claims come only from unprofiled runs ('bench'/'compare'); "
            "'profile' output is diagnostic and must not be quoted as throughput."
        ),
    )
    parser.add_argument("--version", action="version", version=f"shardscope {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    env_cmd = sub.add_parser("env", help="print the reproducibility snapshot for this machine")
    env_cmd.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )

    validate_cmd = sub.add_parser(
        "validate",
        help="check that a run directory can be reported; writes nothing",
        description=(
            "Load a run directory exactly as 'report' does and say whether that worked, "
            "what the run records and what it does not. Renders nothing and writes "
            "nothing. Exit 0 for a loadable run whose artifacts agree; exit 2 when the "
            "directory will not load or summary.json and steps.csv disagree about the "
            "step count."
        ),
    )
    validate_cmd.add_argument(
        "--run",
        metavar="DIR",
        type=Path,
        required=True,
        help="run directory holding steps.csv and environment.json",
    )
    validate_cmd.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help=(
            "output format (default: text). json always writes a document to stdout, "
            "with ok=false and the error on a failure, and still exits 2"
        ),
    )

    report_cmd = sub.add_parser(
        "report",
        help="render a recorded run as HTML, JSON or CSV",
        description=(
            "Render the artifacts a run directory already contains. Shows the "
            "recorded provenance, statistics and per-step rows, and states which "
            "PRD v2.0 section 5 report fields the run does not contain, with the "
            "reason for each. Computes no new measurement."
        ),
    )
    report_cmd.add_argument(
        "--run",
        metavar="DIR",
        type=Path,
        required=True,
        help="run directory holding steps.csv and environment.json",
    )
    report_cmd.add_argument(
        "--out",
        metavar="FILE",
        type=Path,
        default=None,
        help="report file to write; without it the report goes to stdout",
    )
    report_cmd.add_argument(
        "--format",
        choices=REPORT_FORMATS,
        default=None,
        help="output format (default: inferred from --out, else html)",
    )
    report_cmd.add_argument(
        "--max-rows",
        metavar="N",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=(
            f"per-step rows to show in the HTML table (default {DEFAULT_MAX_ROWS}); a "
            "truncated table says how many rows it is hiding and where the rest are. "
            "The json and csv views always carry every row"
        ),
    )

    for command in PENDING_COMMANDS:
        # Options are declared but optional: the command cannot run either
        # way, so demanding its arguments would only obscure the real reason
        # it refuses.
        pending = sub.add_parser(
            command.name,
            help=command.help,
            description=f"{command.summary}. NOT IMPLEMENTED. {command.blocked_on}.",
        )
        for option in command.options:
            pending.add_argument(option.flag, metavar=option.metavar, help=option.help)

    return parser


def _render_env_text() -> str:
    env = capture_environment()
    cudnn = env.cudnn_version if env.cudnn_version is not None else "not available"
    lines = [
        "ShardScope environment",
        f"  python          {env.python_version}",
        f"  platform        {env.platform}",
        f"  machine         {env.machine}",
        f"  torch           {env.torch_version or 'not installed'}",
        f"  accelerator     {env.accelerator}",
        f"  cuda            {env.cuda_version or 'not available'}",
        f"  cudnn           {cudnn}",
        f"  nccl            {env.nccl_version or 'not available'}",
        f"  cuda devices    {env.device_count}",
    ]
    for index, name in enumerate(env.device_names):
        lines.append(f"    [{index}] {name}")
    lines.append(f"  ddp capable     {env.supports_distributed()}")
    if not env.supports_distributed():
        lines.append("  note            DDP/NCCL results must not be reported from this host.")
    return "\n".join(lines)


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "env":
        env = capture_environment()
        print(env.to_json() if args.format == "json" else _render_env_text())
        return 0

    if args.command == "validate":
        return _run_validate(args)

    if args.command == "report":
        return _run_report(args)

    raise BlockedOnDecisionError(_PENDING_BY_NAME[args.command].message())


def _run_validate(args: argparse.Namespace) -> int:
    """Describe one run directory without touching it.

    Text mode follows the rest of the CLI: a failure is one ``error:`` line
    on stderr and exit 2. JSON mode is for automation, so the document goes
    to stdout whatever the outcome and the exit code carries the verdict;
    an ``ok=false`` document with the error inside it is more useful to a
    script than an empty stdout.
    """
    result = validate_run(args.run)
    if args.format == "json":
        sys.stdout.write(result.to_json())
        return 0 if result.ok else 2
    if not result.ok:
        raise ValidationError(result.errors[0])
    sys.stdout.write(result.to_text())
    return 0


def _run_report(args: argparse.Namespace) -> int:
    """Render one recorded run. Nothing is written unless rendering succeeds."""
    run = load_run(args.run)
    fmt = args.format or (format_for_path(args.out) if args.out is not None else "html")
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if args.out is None:
        # No trailing newline of our own: every renderer already ends in one.
        sys.stdout.write(render(run, fmt, generated_at=generated_at, max_rows=args.max_rows))
        return 0

    written = write_report(
        run, args.out, fmt=fmt, generated_at=generated_at, max_rows=args.max_rows
    )
    print(f"wrote {written} ({fmt}) from {run.path}")
    return 0


def _discard_further_stdout() -> None:
    """Point stdout at devnull so the interpreter's exit-time flush stays quiet.

    Without this, a closed pipe produces a second complaint -- "Exception
    ignored in: <_io.TextIOWrapper ...>" -- from the flush CPython runs at
    shutdown, long after we decided the situation was not an error.

    A captured or substituted stdout has no file descriptor to redirect;
    there is nothing to silence in that case either.
    """
    try:
        fileno = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fileno)
    finally:
        # dup2 duplicates the descriptor; the one os.open handed back is a
        # second reference and has to be released. Harmless in a one-shot
        # process, a leak per call in anything that calls main() twice.
        os.close(devnull)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code = _dispatch(args)
        # Flush inside the try: a short report fits in the pipe buffer, so a
        # closed reader is only noticed when the buffer drains. Left to the
        # interpreter's exit-time flush it becomes an "Exception ignored"
        # line on stderr and exit code 120, which no caller can act on.
        sys.stdout.flush()
        return code
    except BrokenPipeError:
        # 'shardscope report --format csv | head' is a documented use, and
        # the reader is allowed to stop reading. Printing a traceback would
        # blame ShardScope for someone else's early exit.
        _discard_further_stdout()
        return EXIT_BROKEN_PIPE
    except ShardScopeError as exc:
        # One actionable line; a genuine bug still escapes as a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
