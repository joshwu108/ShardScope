"""Reproducible single-device training fixture with fixed-seed synthetic input.

Dataset generation uses a dedicated ``torch.Generator`` and model initialization
uses ``fork_rng`` so neither changes global RNG state. Batches follow the same
seeded permutation in both data-residency modes.

Host residency copies each batch to the device during the step. Device residency
places the dataset on the device before training, so no per-step host-to-device
copy exists. The small model and default batch size keep fixture runs bounded.

Loss extraction happens after the timer stops because ``.item()`` synchronizes
the device. Moving it into the timed region would change ``--no-sync`` semantics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from shardscope import execution
from shardscope.execution import (
    DATA_RESIDENCY_CHOICES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SAMPLES,
    DEFAULT_STEPS,
    DEVICE_CHOICES,
    _mps_is_usable,
    step_csv_text,
    summarize,
)
from shardscope.execution import (
    IMAGE_SIZE as IMAGE_SIZE,
)
from shardscope.execution import (
    NUM_CLASSES as NUM_CLASSES,
)
from shardscope.execution import (
    STEP_CSV_COLUMNS as STEP_CSV_COLUMNS,
)
from shardscope.execution import (
    StepRecord as StepRecord,
)
from shardscope.execution import (
    _batch_index_schedule as _batch_index_schedule,
)
from shardscope.execution import (
    build_model as build_model,
)
from shardscope.execution import (
    make_synthetic_dataset as make_synthetic_dataset,
)
from shardscope.execution import (
    write_step_csv as write_step_csv,
)


def resolve_device(request):
    if request == "auto":
        request = "cuda" if torch.cuda.is_available() else "mps" if _mps_is_usable() else "cpu"
    return execution.resolve_device(request)


def run_training(*, device="auto", **kwargs):
    return execution.run_training(device=str(resolve_device(device)), **kwargs)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI.

    Every run knob defaults to ``None`` rather than to its value, so
    ``_resolve_config`` can tell "the user passed this" from "the user left
    it alone" and overlay only explicit flags on top of ``--config``.
    Defaults themselves live in ``shardscope.config.RunConfig``.
    """
    parser = argparse.ArgumentParser(description="Tiny single-device training fixture.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON run config; explicit flags below override it",
    )
    parser.add_argument("--steps", type=int, default=None, help=f"default {DEFAULT_STEPS}")
    parser.add_argument(
        "--batch-size", type=int, default=None, help=f"default {DEFAULT_BATCH_SIZE}"
    )
    parser.add_argument("--samples", type=int, default=None, help=f"default {DEFAULT_SAMPLES}")
    parser.add_argument("--seed", type=int, default=None, help="default 0")
    parser.add_argument("--lr", type=float, default=None, help=f"default {DEFAULT_LEARNING_RATE}")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default=None, help="default auto")
    parser.add_argument(
        "--data-residency", choices=DATA_RESIDENCY_CHOICES, default=None, help="default host"
    )
    parser.add_argument(
        "--sync",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="synchronize the device around each step (default). --no-sync makes "
        "timings measure kernel launch, not execution",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="write steps.csv, environment.json and summary.json here",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-step lines")
    return parser


def _resolve_config(args):
    """Overlay the explicitly-passed flags on top of ``--config``.

    Returns ``(config, None)`` or ``(None, message)``: a bad config file is
    the user's mistake, so it becomes one actionable line rather than a
    traceback. ``shardscope`` is imported here rather than at module scope
    so this script still runs from a checkout where the package is not
    installed.
    """
    from shardscope.config import RunConfig, load_run_config
    from shardscope.errors import ShardScopeError

    try:
        base = load_run_config(args.config) if args.config is not None else RunConfig()
        return base.merged(
            steps=args.steps,
            batch_size=args.batch_size,
            samples=args.samples,
            seed=args.seed,
            learning_rate=args.lr,
            device=args.device,
            data_residency=args.data_residency,
            synchronize=args.sync,
        ), None
    except ShardScopeError as exc:
        return None, str(exc)


def out_dir_problem(out_dir: Path) -> str | None:
    """Say why ``out_dir`` cannot receive a run's artifacts, or return ``None``.r."""
    if out_dir.exists() and not out_dir.is_dir():
        return (
            f"--out-dir {out_dir} exists and is not a directory. Point it at a new "
            "or existing directory."
        )

    ancestor = next((path for path in (out_dir, *out_dir.parents) if path.exists()), None)
    if ancestor is None:  # pragma: no cover - the filesystem root always exists
        return f"--out-dir {out_dir} has no existing parent directory to create it under."
    if not ancestor.is_dir():
        return f"--out-dir {out_dir} cannot be created: {ancestor} is a file, not a directory."
    if not os.access(ancestor, os.W_OK):
        return f"--out-dir {out_dir} is not writable: no write permission on {ancestor}."
    return None


def _write_run_artifacts(out_dir: Path, args, config, records, stats, device) -> None:
    """Write the artifact triple, each file landing whole or not at all."""
    from shardscope.artifacts import replace_file_contents
    from shardscope.env import capture_environment

    environment = capture_environment().with_extra(
        steps=config.steps,
        batch_size=config.batch_size,
        samples=config.samples,
        seed=config.seed,
        learning_rate=config.learning_rate,
        device_requested=config.device,
        device_resolved=device.type,
        data_residency=config.data_residency,
        synchronized=config.synchronize,
        config_file=str(args.config) if args.config is not None else None,
        profiled=False,
        source="examples/train_single.py",
    )
    documents = (
        ("steps.csv", step_csv_text(records)),
        ("environment.json", environment.to_json() + "\n"),
        ("summary.json", json.dumps(stats, indent=2, sort_keys=True) + "\n"),
    )
    for name, text in documents:
        replace_file_contents(out_dir / name, text)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config, error = _resolve_config(args)
    if error is not None:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.out_dir is not None:
        problem = out_dir_problem(args.out_dir)
        if problem is not None:
            print(f"error: {problem}", file=sys.stderr)
            return 2

    records = run_training(
        steps=config.steps,
        batch_size=config.batch_size,
        device=config.device,
        seed=config.seed,
        data_residency=config.data_residency,
        synchronize=config.synchronize,
        n_samples=config.samples,
        learning_rate=config.learning_rate,
    )

    if not args.quiet:
        for record in records:
            print(f"step {record.step:4d}  loss {record.loss:8.4f}  {record.step_time_ms:8.3f} ms")

    stats = summarize(records)
    device = resolve_device(config.device)
    print(
        f"\ndevice={device.type} residency={config.data_residency} "
        f"synchronized={config.synchronize} seed={config.seed}"
    )
    print(
        f"steps={int(stats['steps'])} median={stats['median_step_ms']:.3f} ms "
        f"p95={stats['p95_step_ms']:.3f} ms "
        f"min={stats['min_step_ms']:.3f} ms max={stats['max_step_ms']:.3f} ms"
    )
    if not config.synchronize:
        print("warning: --no-sync timings measure launch, not execution")

    if args.out_dir is not None:
        try:
            _write_run_artifacts(args.out_dir, args, config, records, stats, device)
        except OSError as exc:
            print(
                f"error: the run finished but its artifacts could not be written to "
                f"{args.out_dir}: {exc.strerror or exc}. This run is not recorded. The "
                "directory may hold an incomplete artifact set, or a mix of this run's "
                "files and an earlier recording's; do not report from it. Repeat the run.",
                file=sys.stderr,
            )
            return 2
        print(f"wrote {args.out_dir}/steps.csv, environment.json, summary.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
