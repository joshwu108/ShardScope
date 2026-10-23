"""Command-line entry point for ShardScope.

Subcommand split mirrors PRD requirement SS-04: diagnostic (profiled) runs
and throughput (unprofiled) benchmarks are different commands, and their
numbers are never presented as interchangeable.

    shardscope env                 # reproducibility snapshot (implemented)
    shardscope profile ...         # bounded profiled diagnostic run (Week 2-3)
    shardscope report ...          # render CSV/JSON/HTML from a run (Week 3)
    shardscope bench ...           # warm, unprofiled throughput run (Week 3-7)
    shardscope compare ...         # before/after with variance (Week 7)

Everything except ``env`` is declared but unimplemented. The argument names
come from PRD v2.0 section 11, so ``--help`` documents the agreed interface
target; invoking one exits non-zero and names the decision it waits on,
rather than pretending to work. Memory Lab commands (``memory-sweep``,
``memory-report``) are deliberately absent: PRD v2.0 section 10 gates that
whole block on a Week 7 review that has not happened.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field

from shardscope import __version__
from shardscope.env import capture_environment
from shardscope.errors import BlockedOnDecisionError, ShardScopeError


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
    "this is waiting on a measurement contract the student owns" is the
    whole point of the PRD v2.0 section 13 approval gate.
    """

    name: str
    week: str
    summary: str
    blocked_on: str
    options: tuple[Option, ...] = field(default_factory=tuple)

    @property
    def help(self) -> str:
        return f"{self.summary} (not implemented; Week {self.week})"

    def message(self) -> str:
        return (
            f"'{self.name}' is not implemented yet -- it is a Week {self.week} "
            f"deliverable. Blocked on: {self.blocked_on}"
        )


_CONFIG_OPTION = Option(
    "--config", "FILE", "JSON run config; see shardscope.config for the validated keys"
)

PENDING_COMMANDS: tuple[PendingCommand, ...] = (
    PendingCommand(
        name="profile",
        week="2-3",
        summary="bounded profiled diagnostic run",
        blocked_on=(
            "the capture contract in src/shardscope/capture.py -- the profiler "
            "schedule (GATE 2.1), the activity set and extras (GATE 2.2) and the "
            "record_function marker names every later rule keys off (GATE 2.3) "
            "are student decisions"
        ),
        options=(
            _CONFIG_OPTION,
            Option("--out", "DIR", "per-run artifact directory to create"),
        ),
    ),
    PendingCommand(
        name="report",
        week="3",
        summary="render CSV/JSON/HTML from a recorded run",
        blocked_on=(
            "the per-step row schema in src/shardscope/parse.py -- which fields are "
            "measured directly versus derived and how a derived field carries its "
            "uncertainty (GATE 3.1), and what data_wait_ms means operationally "
            "(GATE 3.2)"
        ),
        options=(
            Option("--run", "DIR", "run directory produced by 'shardscope profile'"),
            Option("--out", "FILE", "report file to write"),
        ),
    ),
    PendingCommand(
        name="bench",
        week="3-7",
        summary="warm, never-profiled throughput run",
        blocked_on=(
            "the benchmark methodology in benchmarks/run_baseline.py -- warmup count, "
            "repeat count and the test used to call a difference real (GATE 7.1) are "
            "student decisions"
        ),
        options=(
            _CONFIG_OPTION,
            Option("--out", "DIR", "unprofiled benchmark record to write"),
        ),
    ),
    PendingCommand(
        name="compare",
        week="7",
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

    raise BlockedOnDecisionError(_PENDING_BY_NAME[args.command].message())


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except ShardScopeError as exc:
        # One actionable line; a genuine bug still escapes as a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
