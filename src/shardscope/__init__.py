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

__all__ = ["RunEnvironment", "capture_environment", "__version__"]

__version__ = "0.0.1"
