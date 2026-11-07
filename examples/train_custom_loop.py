"""Profile a training loop you own with ``shardscope.profile_loop``.

This is the reference integration. Three unrelated architectures (an MLP on
flat vectors, a CNN on images, a Transformer encoder on token sequences) share
one loop with a ``DataLoader``, optional bfloat16 autocast and optional
``DistributedDataParallel``. Nothing is special-cased per model.

Usage:
    Single process::

        python examples/train_custom_loop.py --model transformer --device cuda \\
            --steps 12 --wait 1 --warmup 1 --active 4 --repeat 2 --out runs/transformer

    Under ``torchrun`` (each rank writes ``<out>/rank-NNNNN``)::

        torchrun --standalone --nproc-per-node=2 examples/train_custom_loop.py \\
            --ddp --model cnn --device cuda --steps 8 --out runs/cnn-ddp

    Then::

        shardscope analyze --run runs/transformer --threshold h2d_share_threshold=0.2
        shardscope findings --run runs/transformer

To adapt this to your own script, keep the ``with prof.step(...)`` /
``rec.phase(...)`` structure in ``train`` and replace ``make_dataset`` and
``build_model``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import shardscope
from shardscope.distributed import Launch, rank_directory
from shardscope.errors import ShardScopeError
from shardscope.execution import resolve_device

NUM_CLASSES = 5
MODEL_CHOICES = ("mlp", "cnn", "transformer")
VOCABULARY = 50
SEQUENCE = 12


class TokenClassifier(nn.Module):
    """A small Transformer encoder that classifies a token sequence."""

    def __init__(self, vocabulary: int, width: int, classes: int):
        super().__init__()
        self.embed = nn.Embedding(vocabulary, width)
        layer = nn.TransformerEncoderLayer(
            width, nhead=4, dim_feedforward=2 * width, dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(width, classes)

    def forward(self, tokens):
        return self.head(self.encoder(self.embed(tokens)).mean(dim=1))


def build_model(kind: str, *, seed: int) -> nn.Module:
    """One of three unrelated architectures, initialized deterministically."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if kind == "mlp":
            return nn.Sequential(
                nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 5)
            )
        if kind == "cnn":
            return nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, NUM_CLASSES),
            )
        if kind == "transformer":
            return TokenClassifier(VOCABULARY, 32, NUM_CLASSES)
    raise ValueError(f"unknown model {kind!r}; choose from {', '.join(MODEL_CHOICES)}")


def make_dataset(kind: str, samples: int, *, seed: int) -> TensorDataset:
    """Synthetic inputs shaped for ``kind``; a seed-local generator, no global RNG use."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, NUM_CLASSES, (samples,), generator=generator)
    if kind == "mlp":
        inputs = torch.randn(samples, 64, generator=generator)
    elif kind == "cnn":
        inputs = torch.randn(samples, 3, 16, 16, generator=generator)
    elif kind == "transformer":
        inputs = torch.randint(0, VOCABULARY, (samples, SEQUENCE), generator=generator)
    else:
        raise ValueError(f"unknown model {kind!r}; choose from {', '.join(MODEL_CHOICES)}")
    return TensorDataset(inputs, labels)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=MODEL_CHOICES, default="mlp")
    parser.add_argument("--out", type=Path, required=True, help="new or empty run directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--steps", type=int, default=12, help="training steps to run")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--wait", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--active", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--no-sync", action="store_true", help="do not drain the device per step")
    parser.add_argument("--amp", action="store_true", help="autocast the forward pass to bfloat16")
    parser.add_argument("--ddp", action="store_true", help="wrap in DDP; launch with torchrun")
    parser.add_argument("--quiet", action="store_true")
    return parser


def train(args: argparse.Namespace) -> Path:
    """Run the training loop under the profiler.

    Returns:
        The published run directory (the rank directory under ``--ddp``).

    Raises:
        ShardScopeError: Invalid arguments, an unsatisfied capture schedule, or
            a loop misuse; the CLI maps it to exit status 2.
    """
    for name in ("steps", "batch_size", "samples"):
        if getattr(args, name) < 1:
            raise ShardScopeError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ShardScopeError("--num-workers must be nonnegative")
    rank, local_rank, world_size = 0, 0, 1
    launch = None
    if args.ddp:
        launch = Launch.from_environment()
        rank, local_rank, world_size = launch.rank, launch.local_rank, launch.world_size
        if resolve_device(args.device).type != "cuda":
            raise ShardScopeError("--ddp requires a CUDA device")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = resolve_device(args.device)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ShardScopeError("CUDA device requested but CUDA is unavailable")
    capture = dict(
        wait=args.wait,
        warmup=args.warmup,
        active=args.active,
        repeat=args.repeat,
        activities=["cpu", "cuda"] if device.type == "cuda" else ["cpu"],
    )

    dataset = make_dataset(args.model, args.samples, seed=args.seed)
    if world_size > 1:
        # Disjoint strided shards of one deterministic dataset, like the fixture.
        dataset = TensorDataset(*(tensor[rank::world_size] for tensor in dataset.tensors))
        if not len(dataset):
            raise ShardScopeError("--samples must provide at least one sample per rank")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(args.model, seed=args.seed).to(device)
    workload = shardscope.describe_module(model) | dict(
        example_model=args.model,
        batch_size=args.batch_size,
        samples_per_rank=len(dataset),
        amp="bfloat16" if args.amp else None,
        loader_workers=args.num_workers,
    )
    if launch is not None:
        import torch.distributed as dist

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
    out = rank_directory(args.out, rank) if launch is not None else args.out
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    try:
        with shardscope.profile_loop(
            out,
            capture=capture,
            device=device,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            synchronize=not args.no_sync,
            workload=workload,
        ) as prof:
            batches = iter(loader)
            for step in range(args.steps):
                with prof.step(step) as rec:
                    with rec.phase("data_wait"):
                        try:
                            inputs, targets = next(batches)
                        except StopIteration:
                            batches = iter(loader)
                            inputs, targets = next(batches)
                    with rec.phase("h2d"):
                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                    with rec.phase("forward"):
                        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
                            logits = model(inputs)
                        loss = loss_fn(logits.float(), targets)
                    with rec.phase("backward"):
                        loss.backward()
                    with rec.phase("optimizer"):
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    rec.loss(loss)
    finally:
        if launch is not None:
            import torch.distributed as dist

            dist.destroy_process_group()
    return out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        out = train(args)
    except ShardScopeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
