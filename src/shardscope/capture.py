"""Bounded profiler capture with explicit event-to-training-step evidence."""

from __future__ import annotations

import json
import os
import socket
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch

from shardscope.config import from_mapping
from shardscope.env import capture_environment
from shardscope.errors import ConfigError
from shardscope.execution import (
    Uninstrumented,
    resolve_device,
    run_training,
    step_csv_text,
    summarize,
)
from shardscope.publication import staged_run


def validate_capture(config):
    """Simple validations for the capture settings"""
    config = from_mapping(config.to_dict())
    if config.capture is None:
        raise ConfigError("profile requires an explicit capture config contract")
    device = resolve_device(config.device)
    if device.type == "mps":
        raise ConfigError("MPS capture is unsupported; select cpu or cuda explicitly")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigError("CUDA device requested but CUDA is unavailable")
    labels = {
        "cpu": torch.profiler.ProfilerActivity.CPU,
        "cuda": torch.profiler.ProfilerActivity.CUDA,
    }
    supported = torch.profiler.supported_activities()
    for label in config.capture.activities:
        if labels[label] not in supported:
            raise ConfigError(f"profiler activity {label!r} is unavailable")
        if label == "cuda" and (device.type != "cuda" or not torch.cuda.is_available()):
            raise ConfigError("cuda activity requires an available CUDA training device")
    if device.type == "cuda" and "cuda" not in config.capture.activities:
        raise ConfigError("CUDA training capture requires cuda activity")
    return config, device, [labels[x] for x in config.capture.activities]


def map_trace(path, expected, rank):
    """Join explicit identity ranges to actual source-array marker indices.

    ProfilerStep names are retained only as evidence, never interpreted as
    training indices. Identity markers are nested inside shardscope/step.
    """
    events = json.loads(Path(path).read_text())["traceEvents"]
    mapping = []
    for step in expected:
        identities = [
            (i, e)
            for i, e in enumerate(events)
            if e.get("ph") == "X" and e.get("name") == f"shardscope/training_step/{step}"
        ]
        if len(identities) != 1:
            raise ConfigError(f"{path.name}: expected one identity marker for training step {step}")
        identity_index, identity = identities[0]

        def contains(event, identity=identity):
            return (
                event.get("ph") == "X"
                and event.get("pid") == identity.get("pid")
                and event.get("tid") == identity.get("tid")
                and event["ts"] <= identity["ts"]
                and event["ts"] + event["dur"] >= identity["ts"] + identity["dur"]
            )

        parents = [
            (i, e)
            for i, e in enumerate(events)
            if e.get("name") == "shardscope/step" and contains(e)
        ]
        profiler = [
            (i, e)
            for i, e in enumerate(events)
            if str(e.get("name", "")).startswith("ProfilerStep#") and contains(e)
        ]
        if len(parents) != 1 or len(profiler) != 1:
            raise ConfigError(f"{path.name}: ambiguous or missing parent marker for step {step}")
        mapping.append(
            dict(
                rank=rank,
                training_step=step,
                trace_file=path.name,
                step_event_index=parents[0][0],
                identity_event_index=identity_index,
                profiler_step_event_index=profiler[0][0],
                profiler_step_name=profiler[0][1]["name"],
            )
        )
    actual = [
        e
        for e in events
        if e.get("ph") == "X" and str(e.get("name", "")).startswith("shardscope/training_step/")
    ]
    if len(actual) != len(mapping):
        raise ConfigError(f"{path.name}: unexpected training identity markers")
    return mapping


class CaptureHooks(Uninstrumented):
    """Simple Class to store extra methods the training loop calls"""

    def __init__(self, settings, activities, stage, rank=0, profiler_factory=None):
        self.settings, self.stage, self.rank = settings, stage, rank
        self.trace_files, self.step_map, self.pending = [], [], []
        self.schedule = torch.profiler.schedule(
            **{k: getattr(settings, k) for k in ("wait", "warmup", "active", "repeat")}
        )
        factory = profiler_factory or torch.profiler.profile
        self.profiler = factory(
            activities=activities,
            schedule=self.schedule,
            on_trace_ready=self.export,
            **{
                k: getattr(settings, k)
                for k in ("record_shapes", "profile_memory", "with_stack", "with_flops")
            },
        )

    def training_context(self):
        return self.profiler

    def phase(self, name):
        return torch.profiler.record_function(f"shardscope/{name}")

    @contextmanager
    def step_context(self, step):
        action = self.profiler.current_action
        if action in (
            torch.profiler.ProfilerAction.RECORD,
            torch.profiler.ProfilerAction.RECORD_AND_SAVE,
        ):
            self.pending.append(step)
        with self.phase("step"), self.phase(f"training_step/{step}"):
            yield

    def after_step(self, step, device):
        self.profiler.step()

    def export(self, profiler):
        name = f"trace-{len(self.trace_files):05d}.json"
        path = self.stage / name
        profiler.export_chrome_trace(str(path))
        mapped = map_trace(path, self.pending, self.rank)
        if not mapped:
            raise ConfigError("capture cycle exported no training step markers")
        self.step_map.extend(mapped)
        self.trace_files.append(name)
        self.pending.clear()


def rank_provenance(rank=0, local_rank=0, world_size=1):
    return dict(
        artifact_schema=1,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        hostname=socket.gethostname(),
        pid=os.getpid(),
    )


def write_recording(
    stage,
    config,
    records,
    device,
    *,
    profiled,
    capture=None,
    config_file=None,
    rank=0,
    local_rank=0,
    world_size=1,
):
    extra = config.to_dict()
    extra.pop("capture", None)
    extra.update(rank_provenance(rank, local_rank, world_size))
    extra.update(
        device_requested=config.device,
        device_resolved=str(device),
        synchronized=config.synchronize,
        config_file=config_file,
        profiled=profiled,
        source="shardscope.capture" if profiled else "shardscope.ddp",
        work_scope="per_rank",
        global_batch_size=config.batch_size * world_size,
        fixture_model="synthetic-cnn-v1",
        fixture_dtype=str(torch.get_default_dtype()),
        input_shape=[1, 28, 28],
        samples_scope="global",
        rank_samples=len(range(rank, config.samples, world_size)),
        data_partition={"kind": "strided", "start": rank, "stride": world_size},
    )
    if capture is not None:
        extra["capture"] = capture
    environment = capture_environment().with_extra(**extra)
    (stage / "steps.csv").write_text(step_csv_text(records))
    (stage / "environment.json").write_text(environment.to_json() + "\n")
    (stage / "summary.json").write_text(json.dumps(summarize(records), indent=2) + "\n")


def profile_run(
    config,
    destination,
    *,
    config_file=None,
    rank=0,
    local_rank=0,
    world_size=1,
    model_wrapper=None,
    profiler_factory=None,
    setup_guard=None,
    batch_iterator=None,
    batch_source_metadata=None,
):
    config, device, activities = validate_capture(config)
    with staged_run(destination) as stage:
        hooks = CaptureHooks(config.capture, activities, stage, rank, profiler_factory)
        records = run_training(
            steps=config.steps,
            batch_size=config.batch_size,
            device=str(device),
            seed=config.seed,
            data_residency=config.data_residency,
            synchronize=config.synchronize,
            n_samples=config.samples,
            learning_rate=config.learning_rate,
            instrumentation=hooks,
            rank=rank,
            world_size=world_size,
            model_wrapper=model_wrapper,
            setup_guard=setup_guard,
            batch_iterator=batch_iterator,
        )
        if (
            len(hooks.trace_files) != config.capture.repeat
            or hooks.pending
            or len(hooks.step_map) != config.capture.active * config.capture.repeat
        ):
            raise ConfigError("profiler did not export every configured capture cycle")
        capture = asdict(config.capture)
        capture.update(
            device_resolved=str(device),
            trace_files=hooks.trace_files,
            step_map=hooks.step_map,
            step_map_schema=1,
            data_wait=(
                "host iterator next(); opt-in batch source"
                if batch_iterator is not None
                else "not_recorded: synthetic tensor indexing; no loader iterator"
            ),
        )
        if batch_source_metadata is not None:
            capture["batch_source"] = dict(batch_source_metadata)
        write_recording(
            stage,
            config,
            records,
            device,
            profiled=True,
            capture=capture,
            config_file=config_file,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
        )
    return Path(destination)
