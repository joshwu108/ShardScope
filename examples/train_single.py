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
import csv
import io
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

NUM_CLASSES = 10
IMAGE_SIZE = 28
DEFAULT_SAMPLES = 2048
DEFAULT_BATCH_SIZE = 64
DEFAULT_STEPS = 100
DEFAULT_LEARNING_RATE = 0.01

DATA_RESIDENCY_CHOICES = ("host", "device")
DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")

STEP_CSV_COLUMNS = (
    "step",
    "loss",
    "step_time_ms",
    "data_residency",
    "synchronized",
)


@dataclass(frozen=True)
class StepRecord:
    """One training iteration's measurement.

    ``data_residency`` and ``synchronized`` travel with the number because a
    step time is only comparable against another step time measured the same
    way.
    """

    step: int
    loss: float
    step_time_ms: float
    data_residency: str
    synchronized: bool


def make_synthetic_dataset(
    n_samples: int,
    seed: int,
    *,
    image_size: int = IMAGE_SIZE,
    num_classes: int = NUM_CLASSES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(images, labels)`` drawn from a seed-local generator.

    Synthetic input exercises the training path without an accuracy target.
    Shape, dtype and reproducibility determine fixture behavior.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(
        n_samples, 1, image_size, image_size, generator=generator, dtype=torch.float32
    )
    labels = torch.randint(0, num_classes, (n_samples,), generator=generator, dtype=torch.int64)
    return images, labels


def build_model(*, seed: int, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Build the small CNN, initialized deterministically for ``seed``.

    ``fork_rng`` keeps the initialization from consuming global randomness.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, num_classes),
        )


def _mps_is_usable() -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_available())
    except Exception:  # pragma: no cover - platform-specific
        return False


def resolve_device(request: str) -> torch.device:
    """Turn a device request into a concrete device, falling back honestly."""
    if request not in DEVICE_CHOICES:
        raise ValueError(f"unknown device {request!r}; choose from {DEVICE_CHOICES}")
    if request == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_usable():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(request)


def _synchronize(device: torch.device) -> None:
    """Drain the device queue so a wall-clock reading means what it says."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _batch_index_schedule(n_samples: int, batch_size: int, steps: int, seed: int) -> torch.Tensor:
    """Indices for every step, as one ``(steps, batch_size)`` tensor.

    One seeded permutation, wrapped around as needed. Computing it up front
    keeps batch selection identical across residency modes and devices.
    """
    generator = torch.Generator().manual_seed(seed + 1)
    order = torch.randperm(n_samples, generator=generator)
    needed = steps * batch_size
    repeats = -(-needed // n_samples)  # ceiling division
    flat = order.repeat(repeats)[:needed]
    return flat.view(steps, batch_size)


def run_training(
    *,
    steps: int,
    batch_size: int,
    device: str = "auto",
    seed: int = 0,
    data_residency: str = "host",
    synchronize: bool = True,
    n_samples: int = DEFAULT_SAMPLES,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> tuple[StepRecord, ...]:
    """Train for ``steps`` iterations and return one record per step."""
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if data_residency not in DATA_RESIDENCY_CHOICES:
        raise ValueError(
            f"data_residency must be one of {DATA_RESIDENCY_CHOICES}, got {data_residency!r}"
        )

    # fetch device from device str
    target = resolve_device(device)
    images, labels = make_synthetic_dataset(n_samples=n_samples, seed=seed)
    if data_residency == "device":
        images = images.to(target)
        labels = labels.to(target)

    # build small neural network
    model = build_model(seed=seed).to(target)

    # updates the model's weights
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # measures how wrong preds are to labels
    loss_fn = nn.CrossEntropyLoss()

    # schedule which samples belong to each training batch
    schedule = _batch_index_schedule(n_samples, batch_size, steps, seed)

    records: list[StepRecord] = []
    for step in range(steps):
        indices = schedule[step]

        # wait for previous work to finish
        if synchronize:
            _synchronize(target)
        started = time.perf_counter()

        # fetch next images
        batch_images = images[indices]
        batch_labels = labels[indices]
        if data_residency == "host":
            batch_images = batch_images.to(target)
            batch_labels = batch_labels.to(target)

        optimizer.zero_grad(set_to_none=True) # clear gradients
        logits = model(batch_images) #forward pass
        loss = loss_fn(logits, batch_labels) # calculate loss
        loss.backward() # backward step
        optimizer.step() #adjust weights using gradients

        if synchronize:
            _synchronize(target)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        # Read the loss only after the timer stops: .item() blocks on the
        # device, which would make --no-sync silently synchronize.
        records.append(
            StepRecord(
                step=step,
                loss=float(loss.detach().item()),
                step_time_ms=elapsed_ms,
                data_residency=data_residency,
                synchronized=synchronize,
            )
        )

    return tuple(records)


def step_csv_text(records: Sequence[StepRecord]) -> str:
    """Render per-step records as CSV text.

    ``\n`` line endings, explicitly: the csv module defaults to ``\r\n``,
    and ``shardscope.artifacts.steps_csv_text`` -- the reader's rendering,
    which ``shardscope report --format csv`` emits -- writes ``\n``. The two
    have to agree byte for byte or "the CSV view is the recorded file" is
    only true after newline translation.
    """
    if not records:
        raise ValueError("refusing to write a step CSV with no records")

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(STEP_CSV_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(asdict(record))
    return buffer.getvalue()


def write_step_csv(path: str | Path, records: Sequence[StepRecord]) -> None:
    """Write per-step records to ``path``, replacing any existing file whole."""
    from shardscope.artifacts import replace_file_contents

    replace_file_contents(path, step_csv_text(records))


def summarize(records: Sequence[StepRecord]) -> dict[str, float]:
    """Plain descriptive statistics over a run's step times.

    Nearest-rank percentile, so every reported figure is an actual observed
    step time rather than an interpolation between two of them.

    These come from a *timed* run, not from the warm unprofiled benchmark
    harness (benchmark methodology not implemented), and must not be quoted as throughput
    results.
    """
    if not records:
        raise ValueError("refusing to summarize an empty record set")

    times = sorted(record.step_time_ms for record in records)
    count = len(times)
    midpoint = count // 2
    median = times[midpoint] if count % 2 else (times[midpoint - 1] + times[midpoint]) / 2
    p95_index = min(count - 1, math.ceil(0.95 * count) - 1)
    return {
        "steps": float(count),
        "median_step_ms": median,
        "p95_step_ms": times[p95_index],
        "min_step_ms": times[0],
        "max_step_ms": times[-1],
    }


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
    """Say why ``out_dir`` cannot receive a run's artifacts, or return ``None``.r.
    """
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
    """Write the artifact triple, each file landing whole or not at all.
    """
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
    # build the arguments and flags to be used by the CLI
    args = build_parser().parse_args(argv)

    # fetch the config
    config, error = _resolve_config(args)
    if error is not None:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.out_dir is not None:
        problem = out_dir_problem(args.out_dir)
        if problem is not None:
            print(f"error: {problem}", file=sys.stderr)
            return 2

    # training pipeline
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
