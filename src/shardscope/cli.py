"""Command-line entry point for ShardScope.

Subcommand split mirrors PRD requirement SS-04: diagnostic (profiled) runs
and throughput (unprofiled) benchmarks are different commands, and their
numbers are never presented as interchangeable.

    shardscope env                 # reproducibility snapshot (implemented)
    shardscope profile ...         # bounded profiled diagnostic run (Week 2-3)
    shardscope report ...          # render CSV/JSON/HTML from a run (Week 3)
    shardscope bench ...           # warm, unprofiled throughput run (Week 3-7)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from shardscope import __version__
from shardscope.env import capture_environment

_PENDING = (
    "'{cmd}' is not implemented yet. It is a Week {week} deliverable and is "
    "blocked on the student approving the capture/metric contract "
    "(see src/shardscope/{module}.py)."
)

_PENDING_COMMANDS = {
    "profile": ("2-3", "capture"),
    "report": ("3", "report"),
    "bench": ("3-7", "parse"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shardscope",
        description="Diagnose where a PyTorch training step loses time.",
    )
    parser.add_argument("--version", action="version", version=f"shardscope {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    env_cmd = sub.add_parser("env", help="print the reproducibility snapshot for this machine")
    env_cmd.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )

    for name, (week, _module) in _PENDING_COMMANDS.items():
        sub.add_parser(name, help=f"(not implemented; Week {week})")

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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "env":
        env = capture_environment()
        print(env.to_json() if args.format == "json" else _render_env_text())
        return 0

    week, module = _PENDING_COMMANDS[args.command]
    print(_PENDING.format(cmd=args.command, week=week, module=module), file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
