"""Profile your own PyTorch training loop.

``shardscope profile`` drives the built-in fixture model. Use this module to
capture a training loop you own instead: any model, optimizer, data loader,
precision policy or ``DistributedDataParallel`` wrapper. You keep the loop and
mark its steps and phases; ShardScope records a run directory that
``shardscope validate``, ``analyze``, ``findings`` and ``report`` consume
unchanged.

Example:
    >>> import shardscope
    >>> capture = {"wait": 1, "warmup": 1, "active": 4, "repeat": 1,
    ...            "activities": ["cpu", "cuda"]}
    >>> with shardscope.profile_loop("runs/my-model", capture=capture,
    ...                              device="cuda",
    ...                              workload=shardscope.describe_module(model)) as prof:
    ...     batches = iter(loader)
    ...     for step in range(steps):
    ...         with prof.step(step) as rec:
    ...             with rec.phase("data_wait"):
    ...                 inputs, targets = next(batches)
    ...             with rec.phase("h2d"):
    ...                 inputs, targets = inputs.to("cuda"), targets.to("cuda")
    ...             with rec.phase("forward"):
    ...                 loss = loss_fn(model(inputs), targets)
    ...             with rec.phase("backward"):
    ...                 loss.backward()
    ...             with rec.phase("optimizer"):
    ...                 optimizer.step()
    ...                 optimizer.zero_grad(set_to_none=True)
    ...             rec.loss(loss)

Semantics:
    * ``shardscope/step`` is the host range of one training step; phase markers
      are host ranges inside it. The analysis maps trace events to training
      steps through the identity marker, never through ``ProfilerStep#`` names.
    * A phase you do not enter is reported ``not_recorded``, never as zero.
    * A phase must be entered on the thread that entered its step. Another
      thread's range could not be attributed to the step, so it is refused.
    * Any failure (an exception in your loop, a schedule the loop did not
      satisfy, an API misuse) unwinds the staging directory. The destination
      is either a complete run or absent.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Mapping
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import Any

import torch

from shardscope.capture import CaptureHooks, rank_provenance, validate_capture
from shardscope.config import DATA_RESIDENCY_CHOICES, CaptureConfig, RunConfig
from shardscope.env import capture_environment
from shardscope.errors import ConfigError
from shardscope.execution import StepRecord, _synchronize, resolve_device, step_csv_text, summarize
from shardscope.parse import HOST_PHASES
from shardscope.publication import staged_run

__all__ = [
    "PHASES",
    "SOURCE",
    "LoopProfiler",
    "LoopStep",
    "LoopUsageError",
    "describe_module",
    "profile_loop",
]

#: Phase names :meth:`LoopStep.phase` accepts, derived from the host phases
#: the analysis measures: ``data_wait``, ``h2d``, ``forward``, ``backward``,
#: ``optimizer``. The step marker itself is owned by the step handle.
PHASES: tuple[str, ...] = tuple(
    marker.removeprefix("shardscope/") for marker in HOST_PHASES if marker != "shardscope/step"
)

#: Value of ``environment.json`` ``extra.source`` for runs this module writes.
SOURCE = "shardscope.loop"


class LoopUsageError(ConfigError):
    """Raised when a loop uses the profiler outside its contract.

    Examples: an unknown phase name, a phase entered from another thread,
    nested steps, a non-increasing step index, ``loss()`` after the step
    closed, or a run in which no step was wrapped. Subclasses ``ConfigError``
    so the CLI exits with status 2.
    """


def describe_module(module: torch.nn.Module) -> dict[str, Any]:
    """Describe any ``nn.Module`` as JSON-serializable provenance.

    Intended for the ``workload`` argument of :func:`profile_loop`. Counts are
    exact parameter element counts; nothing is estimated.

    Args:
        module: The model (or wrapper) to describe.

    Returns:
        A dict with ``module_class``, ``parameters``, ``trainable_parameters``,
        ``parameter_dtypes``, ``parameter_devices`` and ``submodules``.

    Raises:
        TypeError: If ``module`` is not an ``nn.Module``.
    """
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"describe_module expects an nn.Module, got {type(module).__name__}")
    parameters = list(module.parameters())
    return {
        "module_class": f"{type(module).__module__}.{type(module).__qualname__}",
        "parameters": sum(p.numel() for p in parameters),
        "trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad),
        "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
        "parameter_devices": sorted({str(p.device) for p in parameters}),
        "submodules": sum(1 for _ in module.modules()) - 1,
    }


def _check_workload(workload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return ``workload`` as a plain dict, or raise if it cannot be persisted as JSON."""
    if workload is None:
        return {}
    if not isinstance(workload, Mapping):
        raise ConfigError("workload must be a mapping of JSON-serializable provenance")
    payload = dict(workload)
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"workload must be JSON-serializable: {exc}") from exc
    if any(not isinstance(key, str) for key in payload):
        raise ConfigError("workload keys must be strings")
    return payload


def _check_ranks(rank: int, local_rank: int, world_size: int) -> None:
    if type(world_size) is not int or world_size < 1 or type(rank) is not int:
        raise ConfigError("rank/world_size must be integers describing an existing rank")
    if not 0 <= rank < world_size:
        raise ConfigError(f"rank {rank} is outside world_size {world_size}")
    if type(local_rank) is not int or local_rank < 0:
        raise ConfigError("local_rank must be a nonnegative integer")


def _checked_loss(value: Any) -> Any:
    """Validate a loss value; tensors are detached and read later, after the step timer."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise LoopUsageError(
                f"loss must be a single value; got a tensor of shape {tuple(value.shape)}"
            )
        return value.detach()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LoopUsageError(
            f"loss must be a number or a one-element tensor, got {type(value).__name__}"
        )
    return value


def _loss_value(value: Any) -> float:
    """Convert a validated loss to the float written to ``steps.csv`` (NaN if none)."""
    if value is None:
        return float("nan")
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


class LoopStep:
    """Handle for one training step, used as ``with prof.step(n) as rec:``.

    Timing runs from ``__enter__`` to ``__exit__``, with an optional device
    synchronization at both ends (see ``synchronize`` on :func:`profile_loop`).
    The loss is read after the timer stops because ``.item()`` on a CUDA
    tensor synchronizes.

    Attributes:
        index: The caller's training-step number.
    """

    def __init__(self, profiler: LoopProfiler, index: int):
        self._profiler = profiler
        self.index = index
        self._stack: ExitStack | None = None
        self._thread: int | None = None
        self._started: float | None = None
        self._loss: Any = None
        self._closed = False

    # -- context manager --------------------------------------------------

    def __enter__(self) -> LoopStep:
        self._profiler._open_step(self)
        self._thread = threading.get_ident()
        self._stack = ExitStack()
        try:
            self._stack.enter_context(self._profiler._hooks.step_context(self.index))
        except BaseException:
            self._profiler._close_step(self)
            raise
        if self._profiler.synchronize:
            _synchronize(self._profiler.device)
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self._stack is not None
        self._closed = True
        try:
            self._stack.__exit__(exc_type, exc, tb)
        finally:
            self._profiler._close_step(self)
        if exc_type is not None:
            return False
        if self._profiler.synchronize:
            _synchronize(self._profiler.device)
        elapsed_ms = (time.perf_counter() - self._started) * 1000.0
        self._profiler._record(
            StepRecord(
                step=self.index,
                loss=_loss_value(self._loss),
                step_time_ms=elapsed_ms,
                data_residency=self._profiler.data_residency,
                synchronized=self._profiler.synchronize,
            )
        )
        # Advances the torch.profiler schedule; exports a trace at cycle end.
        self._profiler._hooks.after_step(self.index, self._profiler.device)
        return False

    # -- public API -------------------------------------------------------

    def phase(self, name: str):
        """Open a ``shardscope/<name>`` host range inside this step.

        Entering the same phase more than once in a step is fine (gradient
        accumulation, for example); the analysis unions the ranges.

        Args:
            name: One of :data:`PHASES`.

        Returns:
            A context manager wrapping ``torch.profiler.record_function``.

        Raises:
            LoopUsageError: Unknown phase name. Entering the returned context
                outside the step, or from another thread, also raises.
        """
        if name not in PHASES:
            raise LoopUsageError(f"unknown phase {name!r}; choose from {', '.join(PHASES)}")
        return _Phase(self, name)

    def loss(self, value: Any) -> None:
        """Record this step's loss for ``steps.csv``.

        Args:
            value: A number or a one-element tensor. Tensors are detached now
                and read only after the step timer stops.

        Raises:
            LoopUsageError: Wrong type or shape, or the step already closed.
        """
        if self._closed:
            raise LoopUsageError(
                f"step {self.index} has already finished; loss() must be called inside it"
            )
        self._loss = _checked_loss(value)

    def _check_phase_entry(self, name: str) -> None:
        if self._closed or self._stack is None:
            raise LoopUsageError(f"phase {name!r} entered outside its step {self.index}")
        if threading.get_ident() != self._thread:
            raise LoopUsageError(
                f"phase {name!r} entered from a different thread than step {self.index}; a "
                "phase range is attributed to a step only on the step's own host thread"
            )


class _Phase:
    """Context manager returned by :meth:`LoopStep.phase`; checks placement on entry."""

    def __init__(self, step: LoopStep, name: str):
        self._step, self._name = step, name
        self._range = None

    def __enter__(self):
        self._step._check_phase_entry(self._name)
        self._range = self._step._profiler._hooks.phase(self._name)
        self._range.__enter__()
        self._step._profiler._phases_entered.add(self._name)
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._range.__exit__(exc_type, exc, tb)


class LoopProfiler:
    """Bounded capture of a caller-owned training loop.

    Create one with :func:`profile_loop` and use it as a context manager. On a
    clean exit the run is published atomically to ``out``; on any failure the
    destination stays absent. One instance records one run.

    Attributes:
        out: Destination run directory.
        capture: The validated :class:`~shardscope.config.CaptureConfig`.
        device: The resolved ``torch.device`` used for synchronization.
        rank, local_rank, world_size: Rank provenance written to the run.
        synchronize: Whether each step drains the device at both timer ends.
        data_residency: ``"host"`` or ``"device"`` as declared by the caller.
        workload: The caller's provenance dict (``extra.workload``).
    """

    def __init__(
        self,
        out: str | Path,
        *,
        capture: CaptureConfig | Mapping[str, Any],
        device: str | torch.device = "auto",
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
        synchronize: bool = True,
        data_residency: str = "host",
        workload: Mapping[str, Any] | None = None,
        profiler_factory=None,
    ):
        if isinstance(capture, CaptureConfig):
            settings = capture
        else:
            settings = CaptureConfig.from_mapping(
                dict(capture) if isinstance(capture, Mapping) else capture
            )
        if type(synchronize) is not bool:
            raise ConfigError("synchronize must be true or false")
        if data_residency not in DATA_RESIDENCY_CHOICES:
            raise ConfigError(
                f"data_residency must be one of {DATA_RESIDENCY_CHOICES}, got {data_residency!r}"
            )
        _check_ranks(rank, local_rank, world_size)
        requested = str(device)
        resolved = resolve_device(requested)
        # validate_capture() takes the fixture config's device vocabulary
        # (auto/cpu/cuda/mps). A concrete "cuda:N" validates as "cuda"; the
        # concrete device is kept for synchronization. `steps` is sized to the
        # schedule so the fixture default cannot cap a longer capture.
        probe = RunConfig(
            steps=max(1, settings.required_steps),
            device=resolved.type if resolved.type in ("cpu", "cuda", "mps") else requested,
            capture=settings,
        )
        _, _, self._activities = validate_capture(probe)
        self.out = Path(out)
        self.capture = settings
        self.device = resolved
        self.device_requested = requested
        self.rank, self.local_rank, self.world_size = rank, local_rank, world_size
        self.synchronize = synchronize
        self.data_residency = data_residency
        self.workload = _check_workload(workload)
        self._profiler_factory = profiler_factory
        self._records: list[StepRecord] = []
        self._phases_entered: set[str] = set()
        self._active: LoopStep | None = None
        self._stage_cm = None
        self._stage: Path | None = None
        self._profiler_cm = None
        self._hooks: CaptureHooks | None = None
        self._state = "new"
        self._owner_thread: int | None = None

    # -- context manager --------------------------------------------------

    def __enter__(self) -> LoopProfiler:
        if self._state != "new":
            raise LoopUsageError("a LoopProfiler records one run; create a new one for another")
        self._state = "open"
        self._owner_thread = threading.get_ident()
        self._stage_cm = staged_run(self.out)
        self._stage = self._stage_cm.__enter__()
        try:
            self._hooks = CaptureHooks(
                self.capture, self._activities, self._stage, self.rank, self._profiler_factory
            )
            self._profiler_cm = self._hooks.training_context()
            self._profiler_cm.__enter__()
        except BaseException:
            self._state = "failed"
            self._stage_cm.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            # Report the caller's exception. Stopping the profiler may itself
            # refuse an unfinished cycle; that noise is dropped.
            self._state = "failed"
            try:
                with suppress(Exception):
                    self._profiler_cm.__exit__(exc_type, exc, tb)
            finally:
                self._stage_cm.__exit__(exc_type, exc, tb)
            return False
        try:
            # Stop the profiler first: a cycle still open at loop end exports
            # on stop, and the completeness check must see that export.
            try:
                self._profiler_cm.__exit__(None, None, None)
            except ConfigError as exc:
                # The export hook refuses a cycle with no mapped steps. For a
                # loop that means the schedule ran short; say so.
                raise self._incomplete_error() from exc
            self._finish()
        except BaseException:
            self._state = "failed"
            self._stage_cm.__exit__(*sys.exc_info())
            raise
        self._state = "closed"
        return self._stage_cm.__exit__(None, None, None)

    # -- public API -------------------------------------------------------

    def step(self, index: int) -> LoopStep:
        """Return the handle for training step ``index``.

        Args:
            index: Your own training-step number. Nonnegative, strictly
                increasing across the run, not necessarily consecutive. A step
                you do not wrap is simply not recorded.

        Returns:
            A :class:`LoopStep` to use as a context manager.

        Raises:
            LoopUsageError: Outside the ``with`` block, bad index, or called
                from a thread other than the one that entered the profiler.
        """
        if self._state != "open":
            raise LoopUsageError("step() is only valid inside the profiler's with-block")
        if type(index) is not int or index < 0:
            raise LoopUsageError(f"training step index must be a nonnegative int, got {index!r}")
        if self._records and index <= self._records[-1].step:
            raise LoopUsageError(
                f"training step {index} does not follow the last recorded step "
                f"{self._records[-1].step}; indices must strictly increase"
            )
        if threading.get_ident() != self._owner_thread:
            raise LoopUsageError(
                "step() must be called on the thread that entered the profiler; the step "
                "boundary and the profiler schedule live on one host thread"
            )
        return LoopStep(self, index)

    @property
    def recorded_steps(self) -> tuple[int, ...]:
        """Training-step indices recorded so far."""
        return tuple(record.step for record in self._records)

    # -- internals --------------------------------------------------------

    def _open_step(self, step: LoopStep) -> None:
        if self._state != "open":
            raise LoopUsageError("step entered outside the profiler's with-block")
        if self._active is not None:
            raise LoopUsageError(
                f"step {step.index} entered while step {self._active.index} is still open; "
                "steps do not nest"
            )
        if self._records and step.index <= self._records[-1].step:
            raise LoopUsageError(
                f"training step {step.index} does not follow the last recorded step "
                f"{self._records[-1].step}; indices must strictly increase"
            )
        self._active = step

    def _close_step(self, step: LoopStep) -> None:
        if self._active is step:
            self._active = None

    def _record(self, record: StepRecord) -> None:
        self._records.append(record)

    def _incomplete_error(self) -> ConfigError:
        """The error for a run whose schedule was not satisfied."""
        settings, hooks = self.capture, self._hooks
        if not self._records:
            return LoopUsageError(
                "no training step was wrapped with prof.step(); nothing to record"
            )
        return ConfigError(
            f"profiler schedule needs {settings.required_steps} wrapped training steps "
            f"((wait + warmup + active) * repeat) but the loop wrapped {len(self._records)}; "
            f"{len(hooks.trace_files)} of {settings.repeat} capture cycle(s) exported. "
            "This run is not recorded."
        )

    def _finish(self) -> None:
        """Check capture completeness and write steps.csv, environment.json, summary.json."""
        hooks, settings = self._hooks, self.capture
        wrapped = len(self._records)
        if (
            not self._records
            or len(hooks.trace_files) != settings.repeat
            or hooks.pending
            or len(hooks.step_map) != settings.active * settings.repeat
        ):
            raise self._incomplete_error()
        capture = {
            **{
                key: getattr(settings, key)
                for key in (
                    "wait",
                    "warmup",
                    "active",
                    "repeat",
                    "record_shapes",
                    "profile_memory",
                    "with_stack",
                    "with_flops",
                )
            },
            "activities": list(settings.activities),
            "device_resolved": str(self.device),
            "trace_files": list(hooks.trace_files),
            "step_map": list(hooks.step_map),
            "step_map_schema": 1,
            "integration": "shardscope.loop.profile_loop",
            "phases_entered": sorted(self._phases_entered),
            "data_wait": (
                "caller-entered shardscope/data_wait host range"
                if "data_wait" in self._phases_entered
                else "not_recorded: the loop entered no data_wait phase"
            ),
        }
        extra = rank_provenance(self.rank, self.local_rank, self.world_size)
        extra.update(
            device_requested=self.device_requested,
            device_resolved=str(self.device),
            synchronized=self.synchronize,
            data_residency=self.data_residency,
            profiled=True,
            source=SOURCE,
            work_scope="per_rank",
            steps=wrapped,
            default_dtype=str(torch.get_default_dtype()),
            workload=self.workload,
            capture=capture,
        )
        environment = capture_environment().with_extra(**extra)
        stage = self._stage
        (stage / "steps.csv").write_text(step_csv_text(self._records))
        (stage / "environment.json").write_text(environment.to_json() + "\n")
        (stage / "summary.json").write_text(json.dumps(summarize(self._records), indent=2) + "\n")


def profile_loop(
    out: str | Path,
    *,
    capture: CaptureConfig | Mapping[str, Any],
    device: str | torch.device = "auto",
    rank: int = 0,
    local_rank: int = 0,
    world_size: int = 1,
    synchronize: bool = True,
    data_residency: str = "host",
    workload: Mapping[str, Any] | None = None,
) -> LoopProfiler:
    """Profile a training loop you own into the run directory ``out``.

    Args:
        out: New or empty directory for the run. Published atomically on
            success; left absent on failure.
        capture: Bounded profiler schedule, the same keys ``shardscope
            profile`` takes: ``wait``, ``warmup``, ``active``, ``repeat``,
            ``activities`` (``["cpu"]`` or ``["cpu", "cuda"]``) and the opt-in
            ``record_shapes``, ``profile_memory``, ``with_stack``,
            ``with_flops`` flags. The loop must wrap at least
            ``(wait + warmup + active) * repeat`` steps.
        device: Where per-step synchronization happens (``"cuda"``,
            ``"cuda:N"``, ``"cpu"`` or ``"auto"``). It moves nothing. A CUDA
            device requires the ``cuda`` activity. MPS capture is unsupported.
        rank: Global rank for ``torchrun`` launches; give each rank its own
            ``out`` (see ``shardscope.distributed.rank_directory``).
        local_rank: Local rank, recorded as provenance.
        world_size: Number of ranks; ``rank`` must be below it.
        synchronize: Drain the device before the step timer starts and after
            it stops. Recorded per row in ``steps.csv``.
        data_residency: ``"host"`` (default) or ``"device"``. ``"device"``
            tells the analysis there is no per-step host-to-device batch copy.
        workload: JSON-serializable provenance stored under
            ``environment.json`` ``extra.workload``; see :func:`describe_module`.

    Returns:
        A :class:`LoopProfiler` to use as a context manager.

    Raises:
        ConfigError: Invalid capture settings, device, residency, ranks or
            workload. Raised before any training runs.
    """
    return LoopProfiler(
        out,
        capture=capture,
        device=device,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        synchronize=synchronize,
        data_residency=data_residency,
        workload=workload,
    )
