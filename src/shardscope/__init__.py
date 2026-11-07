"""ShardScope: a diagnostic profiler for PyTorch training steps.

ShardScope records what the CPU, data loader, GPUs and GPU-to-GPU
communication do during a bounded window of training steps, then reports
where each step loses time with trace evidence attached.

Design policy (see planning/ for rationale):
  * Profiled diagnostic runs and unprofiled throughput benchmarks are
    separate commands and are never mixed in one report.
  * Overlapping CUDA work means durations do not sum to step time. Nothing
    in this package derives a "percent wasted" figure from naive sums.
  * When an event class is unavailable in the running PyTorch/CUDA/NCCL
    build, it is reported as NOT_OBSERVED rather than given a duration.
"""

from shardscope.env import RunEnvironment, capture_environment

__all__ = [
    "RunEnvironment",
    "capture_environment",
    "describe_module",
    "profile_loop",
    "__version__",
]

#: Loop-integration entry points live in ``shardscope.loop``, which imports
#: torch. They are resolved on first use so that importing the package, and
#: the CLI commands that only read artifacts, stay torch-free.
_LAZY = {"profile_loop": "shardscope.loop", "describe_module": "shardscope.loop"}


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module 'shardscope' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


__version__ = "0.0.1"
