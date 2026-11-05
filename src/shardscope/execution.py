"""Shared deterministic fixture execution; instrumentation is optional."""

from __future__ import annotations

import csv
import io
import math
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


class Uninstrumented:
    """Hooks that do not attach a profiler or synchronize the device."""

    def training_context(self):
        return nullcontext()

    def step_context(self, step):
        return nullcontext()

    def phase(self, name):
        return nullcontext()

    def before_step(self, step, device):
        pass

    def after_step(self, step, device):
        pass


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
    """One training iteration's measurement."""
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
    """Return ``(images, labels)`` drawn from a seed-local generator."""
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
    if isinstance(request, str) and request.startswith("cuda:"):
        return torch.device(request)
    if request not in DEVICE_CHOICES:
        raise ValueError(f"unknown device {request!r}; choose from {DEVICE_CHOICES}")
    if request == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_usable():
            return torch.device("mps")
        return torch.device("cpu")
    if request == "cuda" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(request)


def _synchronize(device: torch.device) -> None:
    """Drain the device queue so a wall-clock reading means what it says."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
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
    instrumentation=None,
    collect_losses: bool = True,
    rank: int = 0,
    world_size: int = 1,
    model_wrapper=None,
    setup_guard=None,
    batch_iterator=None,
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

    if type(world_size) is not int or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("rank/world_size must describe an existing rank")
    if batch_iterator is not None and data_residency != "host":
        raise ValueError("an external host batch iterator requires data_residency=host")
    target = resolve_device(device)
    with setup_guard() if setup_guard is not None else nullcontext():
        images, labels = make_synthetic_dataset(n_samples=n_samples, seed=seed)
        if world_size > 1:
            # Disjoint strided shards of the same deterministic global dataset.
            images, labels = images[rank::world_size], labels[rank::world_size]
            n_samples = len(labels)
            if not n_samples:
                raise ValueError("samples must provide at least one sample per rank")
        if data_residency == "device":
            images = images.to(target)
            labels = labels.to(target)

        model = build_model(seed=seed).to(target)
        loss_fn = nn.CrossEntropyLoss()
        schedule = _batch_index_schedule(n_samples, batch_size, steps, seed)
    if model_wrapper is not None:
        model = model_wrapper(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    hook = instrumentation or Uninstrumented()
    records: list[StepRecord] = []
    with hook.training_context():
        for step in range(steps):
            hook.before_step(step, target)
            indices = schedule[step]
            if synchronize:
                _synchronize(target)
            started = time.perf_counter()
            with hook.step_context(step):
                if batch_iterator is None:
                    batch_images = images[indices]
                    batch_labels = labels[indices]
                else:
                    with hook.phase("data_wait"):
                        batch_images, batch_labels = next(batch_iterator)
                if data_residency == "host":
                    with hook.phase("h2d"):
                        batch_images = batch_images.to(target)
                        batch_labels = batch_labels.to(target)

                optimizer.zero_grad(set_to_none=True)
                with hook.phase("forward"):
                    logits = model(batch_images)
                    loss = loss_fn(logits, batch_labels)
                with hook.phase("backward"):
                    loss.backward()
                with hook.phase("optimizer"):
                    optimizer.step()

            if synchronize:
                _synchronize(target)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            # Loss extraction stays outside step timing. Benchmarks omit it
            # entirely so it cannot synchronize inside a throughput window.
            records.append(
                StepRecord(
                    step=step,
                    loss=float(loss.detach().item()) if collect_losses else float("nan"),
                    step_time_ms=elapsed_ms,
                    data_residency=data_residency,
                    synchronized=synchronize,
                )
            )
            hook.after_step(step, target)

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

    These describe per-step fixture timings. Completed-work benchmark windows
    are measured separately and provide the throughput denominator.
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
