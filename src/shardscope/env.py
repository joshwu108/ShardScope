"""Capture reproducibility metadata for training runs.

Each run records interpreter, library and device details so its measurements can
be interpreted in context. CUDA is never imported eagerly; environment capture
also works on CPU-only hosts and in CI.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field, replace
from typing import Any

#: Value used anywhere a quantity could not be observed in this environment.
#: Keep unobservable activity distinct from a measured zero.
NOT_OBSERVED = "not_observed"


@dataclass(frozen=True)
class RunEnvironment:
    """Immutable snapshot of the machine and library versions behind a run."""

    python_version: str
    platform: str
    machine: str
    torch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    cudnn_version: int | None
    nccl_version: str | None
    device_count: int
    device_names: tuple[str, ...]
    mps_available: bool
    extra: dict[str, Any] = field(default_factory=dict)

    def with_extra(self, **items: Any) -> RunEnvironment:
        """Return a new snapshot with additional free-form fields recorded."""
        return replace(self, extra={**self.extra, **items})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @property
    def accelerator(self) -> str:
        """Coarse label for which backend a run can actually use."""
        if self.cuda_available and self.device_count > 0:
            return "cuda"
        if self.mps_available:
            return "mps"
        return "cpu"

    def supports_distributed(self) -> bool:
        """True only when a real multi-CUDA-device DDP run is possible here.

        DDP/NCCL claims in any report must be gated on this returning True;
        Device availability alone does not establish that a distributed run occurred.
        """
        return distributed_capable(
            cuda_available=self.cuda_available, device_count=self.device_count
        )


def distributed_capable(*, cuda_available: bool, device_count: int) -> bool:
    """Whether a host with this CUDA inventory could run a real DDP job.

    Kept as a free function so a *recorded* environment -- an
    ``environment.json`` loaded back from a finished run -- is judged by the
    same rule as the live machine, instead of each caller re-spelling
    ">= 2 devices". A report that disagreed with ``shardscope env`` about
    what a host can do is exactly how a distributed claim gets made for a
    run that never happened.
    """
    return bool(cuda_available) and device_count >= 2


def _torch_module():
    """Import torch lazily, returning None when it is not installed."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def capture_environment() -> RunEnvironment:
    """Collect the current environment snapshot.

    Missing or unavailable facts are recorded as ``None``/``False``/empty --
    never as a plausible-looking default.
    """
    torch = _torch_module()
    if torch is None:
        return RunEnvironment(
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            machine=platform.machine(),
            torch_version=None,
            cuda_available=False,
            cuda_version=None,
            cudnn_version=None,
            nccl_version=None,
            device_count=0,
            device_names=(),
            mps_available=False,
        )

    cuda_available = bool(torch.cuda.is_available())
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_names = tuple(torch.cuda.get_device_name(i) for i in range(device_count))

    return RunEnvironment(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_version=getattr(torch.version, "cuda", None),
        cudnn_version=_cudnn_version(torch, cuda_available),
        nccl_version=_nccl_version(torch, cuda_available),
        device_count=device_count,
        device_names=device_names,
        mps_available=_mps_available(torch),
    )


def _cudnn_version(torch, cuda_available: bool) -> int | None:
    if not cuda_available:
        return None
    try:
        return torch.backends.cudnn.version()
    except Exception:  # pragma: no cover - driver-specific
        return None


def _nccl_version(torch, cuda_available: bool) -> str | None:
    if not cuda_available:
        return None
    try:
        return ".".join(str(part) for part in torch.cuda.nccl.version())
    except Exception:  # pragma: no cover - build-specific
        return None


def _mps_available(torch) -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_available())
    except Exception:  # pragma: no cover - platform-specific
        return False
