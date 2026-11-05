"""Unprofiled completed-work windows and descriptive benchmark comparisons."""

from __future__ import annotations

import json
import math
import os
import socket
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from shardscope.config import from_mapping
from shardscope.env import capture_environment
from shardscope.errors import ConfigError
from shardscope.execution import Uninstrumented, _synchronize, resolve_device, run_training
from shardscope.publication import staged_run

SCHEMA_VERSION = 1
TIMING_POLICY = "completion_boundaries_no_step_sync_no_loss_extraction_v1"


@dataclass(frozen=True)
class BenchmarkSettings:
    warmup: int
    repeat: int
    measured_steps: int

    def __post_init__(self):
        for name in ("warmup", "repeat", "measured_steps"):
            value = getattr(self, name)
            minimum = 0 if name == "warmup" else 1
            if type(value) is not int or value < minimum:
                raise ConfigError(f"benchmark {name} must be an integer >= {minimum}")


class Window(Uninstrumented):
    def __init__(self, settings):
        self.settings = settings
        self.started = self.elapsed = None

    def before_step(self, step, device):
        if step == self.settings.warmup:
            _synchronize(device)
            self.started = time.perf_counter()

    def after_step(self, step, device):
        if step == self.settings.warmup + self.settings.measured_steps - 1:
            _synchronize(device)
            self.elapsed = time.perf_counter() - self.started


def benchmark_run(config, settings, destination, *, config_file=None):
    config = from_mapping(config.to_dict())
    if config.capture is not None:
        raise ConfigError("bench refuses capture settings; supply an unprofiled workload config")
    device = resolve_device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigError("CUDA is unavailable")
    workload = dict(
        model="synthetic-cnn-v1",
        input_shape=[1, 28, 28],
        dtype=str(torch.get_default_dtype()).removeprefix("torch."),
        batch_size=config.batch_size,
        samples=config.samples,
        seed=config.seed,
        learning_rate=config.learning_rate,
        data_residency=config.data_residency,
        device_resolved=str(device),
        work_scope="single_process",
    )
    timing = dict(
        policy=TIMING_POLICY,
        unit="seconds",
        boundary_synchronization=True,
        step_synchronization=False,
        loss_extraction=False,
        scope="iterations including batch selection, transfers and Python bookkeeping",
        repeat_state="fresh model/data/optimizer in same process; same seed",
        **asdict(settings),
    )
    environment = (
        capture_environment()
        .with_extra(
            profiled=False,
            artifact_schema=1,
            source="shardscope.benchmark",
            config_file=config_file,
            requested_run_config=config.to_dict(),
            resolved_run_config=config.merged(
                steps=settings.warmup + settings.measured_steps, synchronize=False
            ).to_dict(),
            hostname=socket.gethostname(),
            pid=os.getpid(),
            timing=timing,
            torch_num_threads=torch.get_num_threads(),
            torch_num_interop_threads=torch.get_num_interop_threads(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            default_dtype=str(torch.get_default_dtype()),
        )
        .to_dict()
    )
    observations = []
    with staged_run(destination) as stage:
        for repeat in range(settings.repeat):
            window = Window(settings)
            run_training(
                steps=settings.warmup + settings.measured_steps,
                batch_size=config.batch_size,
                n_samples=config.samples,
                learning_rate=config.learning_rate,
                seed=config.seed,
                device=str(device),
                data_residency=config.data_residency,
                synchronize=False,
                collect_losses=False,
                instrumentation=window,
            )
            elapsed = window.elapsed
            if elapsed is None or not math.isfinite(elapsed) or elapsed <= 0:
                raise ConfigError("benchmark completion window must be finite and positive")
            observations.append(
                dict(
                    repeat=repeat,
                    elapsed_seconds=elapsed,
                    completed_steps=settings.measured_steps,
                    completed_examples=settings.measured_steps * config.batch_size,
                    steps_per_second=settings.measured_steps / elapsed,
                    examples_per_second=settings.measured_steps * config.batch_size / elapsed,
                )
            )
        payload = dict(
            schema_version=SCHEMA_VERSION,
            kind="shardscope.benchmark",
            profiled=False,
            workload=workload,
            timing=timing,
            environment=environment,
            units=dict(steps_per_second="steps/s", examples_per_second="examples/s"),
            observations=observations,
            significance=dict(status="unconfigured"),
        )
        (stage / "benchmark.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return json.loads(json.dumps(payload))


def load_benchmark(path):
    path = Path(path)
    if path.is_dir():
        path = path / "benchmark.json"
    try:
        payload = json.loads(path.read_text())
        validate_benchmark(payload)
        return payload
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ConfigError(f"invalid benchmark {path}: {exc}") from exc


def validate_benchmark(payload):
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["kind"] != "shardscope.benchmark"
    ):
        raise ValueError("unsupported benchmark schema")
    if payload["profiled"] is not False or payload["environment"]["extra"]["profiled"] is not False:
        raise ValueError("benchmark must explicitly be unprofiled")
    if payload["units"] != {"steps_per_second": "steps/s", "examples_per_second": "examples/s"}:
        raise ValueError("unsupported throughput units")
    timing = payload["timing"]
    if (
        timing["policy"] != TIMING_POLICY
        or timing["unit"] != "seconds"
        or timing["boundary_synchronization"] is not True
        or timing["step_synchronization"] is not False
        or timing["loss_extraction"] is not False
    ):
        raise ValueError("incompatible timing policy")
    settings = BenchmarkSettings(**{k: timing[k] for k in ("warmup", "repeat", "measured_steps")})
    if len(payload["observations"]) != settings.repeat:
        raise ValueError("repeat count disagrees with raw observations")
    batch = payload["workload"]["batch_size"]
    if type(batch) is not int or batch <= 0:
        raise ValueError("invalid batch size")
    for index, item in enumerate(payload["observations"]):
        seconds = item["elapsed_seconds"]
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("elapsed_seconds must be finite and positive")
        if (
            type(item["repeat"]) is not int
            or type(item["completed_steps"]) is not int
            or item["repeat"] != index
            or item["completed_steps"] != settings.measured_steps
        ):
            raise ValueError("raw observation step/repeat mismatch")
        if (
            type(item["completed_examples"]) is not int
            or item["completed_examples"] != settings.measured_steps * batch
        ):
            raise ValueError("raw observation example count mismatch")
        for key, count in [
            ("steps_per_second", settings.measured_steps),
            ("examples_per_second", settings.measured_steps * batch),
        ]:
            rate = item[key]
            if (
                type(rate) not in (int, float)
                or not math.isfinite(rate)
                or not math.isclose(rate, count / seconds, rel_tol=1e-9)
            ):
                raise ValueError(f"{key} disagrees with completed work / elapsed_seconds")


def compare_benchmarks(baseline, candidate):
    validate_benchmark(baseline)
    validate_benchmark(candidate)
    if baseline["workload"] != candidate["workload"]:
        raise ConfigError("cannot compare incompatible workloads")
    left, right = [dict(x["timing"]) for x in (baseline, candidate)]
    left.pop("repeat")
    right.pop("repeat")
    if left != right:
        raise ConfigError("cannot compare incompatible timing policies")
    metrics = {}
    for key in ("steps_per_second", "examples_per_second"):
        a, b = [[item[key] for item in p["observations"]] for p in (baseline, candidate)]
        am, bm = statistics.median(a), statistics.median(b)
        metrics[key] = dict(
            baseline_median=am,
            candidate_median=bm,
            baseline_min=min(a),
            baseline_max=max(a),
            candidate_min=min(b),
            candidate_max=max(b),
            descriptive_delta_percent=(bm / am - 1) * 100,
        )
    return dict(
        schema_version=1,
        kind="shardscope.benchmark_comparison",
        metrics=metrics,
        baseline_environment=baseline["environment"],
        candidate_environment=candidate["environment"],
        significance=dict(status="unconfigured", decision=None),
        interpretation=(
            "Descriptive observations only; no confirmed speedup or significance decision."
        ),
    )
