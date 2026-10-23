"""Tiny single-device training loop -- the fixture every ShardScope run profiles.

Requirement SS-01: a tiny, reproducible PyTorch training example with
fixed-seed synthetic input.

Authorship: written by the AI agent at the student's request, under the
"Project plumbing" row of PRD v2.0 section 13 ("Project scaffold, examples,
CLI, parsers, reports, fixtures"). It contains no custom kernels and no
device hot-path optimization; those remain student-owned. Nothing here may be
presented as the student's own GPU work.

Design notes that later weeks depend on:

* ``make_synthetic_dataset`` draws from its own ``torch.Generator`` and
  ``build_model`` initializes inside ``fork_rng``, so neither disturbs the
  global RNG. A fixture that silently reseeds would make every later
  before/after comparison suspect.
* Batches come from one seeded permutation with wrap-around, so the *same*
  examples are seen in the same order regardless of where the data lives.
  That is what makes the two ``--data-residency`` modes comparable.
* ``.item()`` on the loss forces a device synchronize, so it is called only
  after the step timer stops. Otherwise ``--no-sync`` would secretly
  synchronize and measure the same thing as ``--sync``.

Open approval gates (PRD v2.0 section 13 approval gate) -- current values are
defaults, not decisions:

* GATE 1.1 model shape and batch size: the defaults below (a 9,098-parameter
  CNN, batch 64) are chosen to keep steps and traces small. Replace them if
  the student wants a different operating point.
* GATE 1.2 data residency: deliberately unresolved. ``host`` copies each
  batch to the device per step, so a host-to-device transfer exists and can
  be measured; ``device`` pre-places the whole dataset, so no such transfer
  exists and ``h2d_copy_ms`` is structurally absent. Both run; the student
  picks after measuring.
"""

from __future__ import annotations

import argparse
import csv
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

    The data is meaningless by design (PRD v2.0 section 5, SS-01 asks for
    seeded synthetic input and no accuracy score); only its shape, dtype and
    reproducibility matter.
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

    target = resolve_device(device)
    images, labels = make_synthetic_dataset(n_samples=n_samples, seed=seed)
    if data_residency == "device":
        images = images.to(target)
        labels = labels.to(target)

    model = build_model(seed=seed).to(target)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    schedule = _batch_index_schedule(n_samples, batch_size, steps, seed)

    records: list[StepRecord] = []
    for step in range(steps):
        indices = schedule[step]
        if synchronize:
            _synchronize(target)
        started = time.perf_counter()

        batch_images = images[indices]
        batch_labels = labels[indices]
        if data_residency == "host":
            batch_images = batch_images.to(target)
            batch_labels = batch_labels.to(target)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_images)
        loss = loss_fn(logits, batch_labels)
        loss.backward()
        optimizer.step()

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


def write_step_csv(path: str | Path, records: Sequence[StepRecord]) -> None:
    """Write per-step records to ``path`` (requirement SS-03, CSV half)."""
    if not records:
        raise ValueError("refusing to write a step CSV with no records")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(STEP_CSV_COLUMNS))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def summarize(records: Sequence[StepRecord]) -> dict[str, float]:
    """Plain descriptive statistics over a run's step times.

    Nearest-rank percentile, so every reported figure is an actual observed
    step time rather than an interpolation between two of them.

    These come from a *timed* run, not from the warm unprofiled benchmark
    Week 3 adds, and must not be quoted as throughput results.
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
        help="JSON run config (PRD v2.0 section 11); explicit flags below override it",
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
    """Say why ``out_dir`` cannot receive a run's artifacts, or return ``None``.

    Called before the first step. The artifacts are written after the last
    one, so without this check a mistyped path throws away every measured
    step and reports it as a traceback out of ``mkdir``.

    Nothing is created here. A directory made up front would be left behind
    empty if the run then failed, and an empty run directory reads as a run
    that measured nothing.
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
    from shardscope.env import capture_environment

    write_step_csv(out_dir / "steps.csv", records)
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
    (out_dir / "environment.json").write_text(environment.to_json() + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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
            # The run itself succeeded; say plainly that its artifacts did
            # not survive, so the numbers above are not mistaken for a
            # recorded run.
            print(
                f"error: the run finished but its artifacts could not be written to "
                f"{args.out_dir}: {exc.strerror or exc}. The measurements above were "
                "not saved.",
                file=sys.stderr,
            )
            return 2
        print(f"wrote {args.out_dir}/steps.csv, environment.json, summary.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
