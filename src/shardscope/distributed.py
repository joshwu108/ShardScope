"""Conventional torchrun/NCCL setup and per-rank artifact execution.

Run under torchrun's process supervision. Rank-local failures propagate as
nonzero exits so the launcher terminates peers; collectives use a finite timeout.
Rank directories publish independently, without a distributed transaction.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from shardscope.capture import profile_run, validate_capture, write_recording
from shardscope.config import from_mapping, load_run_config
from shardscope.errors import ConfigError, ShardScopeError
from shardscope.execution import run_training
from shardscope.publication import staged_run


@dataclass(frozen=True)
class Launch:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int

    @classmethod
    def from_environment(cls, environ=None):
        env = os.environ if environ is None else environ
        try:
            values = {key: int(env[key.upper()]) for key in cls.__dataclass_fields__}
            launch = cls(**values)
            if not env["MASTER_ADDR"] or not 0 < int(env["MASTER_PORT"]) < 65536:
                raise ValueError("invalid rendezvous address/port")
        except (KeyError, ValueError) as exc:
            raise ConfigError(
                f"launch with torchrun: missing/invalid rank or rendezvous: {exc}"
            ) from exc
        if (
            launch.world_size < 2
            or not 0 <= launch.rank < launch.world_size
            or not 0 <= launch.local_rank < launch.local_world_size
            or not 1 <= launch.local_world_size <= launch.world_size
        ):
            raise ConfigError(
                "torchrun rank/world-size configuration is inconsistent; require >=2 ranks"
            )
        return launch


def rank_directory(root, rank):
    return Path(root) / f"rank-{rank:05d}"


def agree(error, *, world_size):
    """Exchange readiness before entering the next common collective phase."""
    errors = [None] * world_size
    dist.all_gather_object(errors, error)
    if any(item is not None for item in errors):
        raise ConfigError(f"distributed setup refused: {errors}")


@contextmanager
def coordinated_setup(world_size):
    error = None
    try:
        yield
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    agree(error, world_size=world_size)


def execute(config, root, launch, *, profiled=False, config_file=None):
    if not torch.cuda.is_available() or torch.cuda.device_count() < launch.local_world_size:
        raise ConfigError("DDP requires CUDA and one visible device per local process")
    if not dist.is_available() or not dist.is_nccl_available():
        raise ConfigError("DDP requires an installed NCCL backend")
    torch.cuda.set_device(launch.local_rank)
    initialized = False
    created_root = False
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=launch.rank,
            world_size=launch.world_size,
            timeout=timedelta(seconds=60),
        )
        initialized = True
        with coordinated_setup(launch.world_size):
            config = from_mapping(config.to_dict())
            if config.device != "cuda":
                raise ConfigError("DDP config must explicitly select device=cuda")
            if config.samples < launch.world_size:
                raise ConfigError("samples must provide at least one sample per rank")
            if profiled:
                validate_capture(config)
            elif config.capture is not None:
                raise ConfigError("unprofiled DDP refuses capture settings; use --profile")
        signatures = [None] * launch.world_size
        signature = dict(
            config=config.to_dict(), root=str(Path(root).absolute()), profiled=profiled
        )
        dist.all_gather_object(signatures, signature)
        if any(x != signature for x in signatures):
            raise ConfigError(
                "all ranks must use identical workload, capture and output configuration"
            )
        with coordinated_setup(launch.world_size):
            if launch.rank == 0:
                root = Path(root)
                if root.is_symlink() or (
                    root.exists() and (not root.is_dir() or any(root.iterdir()))
                ):
                    raise ConfigError("DDP output must be a new or empty directory")
                root.mkdir(parents=True, exist_ok=True)
                created_root = True
        destination = rank_directory(root, launch.rank)

        def wrap(model):
            return DistributedDataParallel(
                model, device_ids=[launch.local_rank], output_device=launch.local_rank
            )

        def guard():
            return coordinated_setup(launch.world_size)

        if profiled:
            profile_run(
                config,
                destination,
                config_file=config_file,
                rank=launch.rank,
                local_rank=launch.local_rank,
                world_size=launch.world_size,
                model_wrapper=wrap,
                setup_guard=guard,
            )
        else:
            with staged_run(destination) as stage:
                records = run_training(
                    steps=config.steps,
                    batch_size=config.batch_size,
                    device=f"cuda:{launch.local_rank}",
                    seed=config.seed,
                    n_samples=config.samples,
                    learning_rate=config.learning_rate,
                    data_residency=config.data_residency,
                    synchronize=config.synchronize,
                    rank=launch.rank,
                    world_size=launch.world_size,
                    model_wrapper=wrap,
                    setup_guard=guard,
                )
                write_recording(
                    stage,
                    config,
                    records,
                    torch.device("cuda", launch.local_rank),
                    profiled=False,
                    config_file=config_file,
                    rank=launch.rank,
                    local_rank=launch.local_rank,
                    world_size=launch.world_size,
                )
        return destination
    finally:
        if initialized:
            dist.destroy_process_group()
        if created_root:
            with suppress(OSError):
                Path(root).rmdir()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Conventional CUDA DDP fixture; launch with torchrun"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args(argv)
    try:
        launch = Launch.from_environment()
        config = load_run_config(args.config)
        destination = execute(
            config, args.out, launch, profiled=args.profile, config_file=str(args.config)
        )
        print(f"rank {launch.rank}: wrote {destination}; work is per-rank; launch={asdict(launch)}")
        return 0
    except (ShardScopeError, OSError, RuntimeError) as exc:
        print(
            f"error: DDP failed: {exc}. Rank artifacts may be incomplete; inspect each rank.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
