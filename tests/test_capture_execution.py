"""Capture lifecycle software tests; synthetic doubles are not GPU evidence."""

import json
from contextlib import nullcontext

import pytest
import torch

from shardscope import capture
from shardscope.artifacts import load_run
from shardscope.config import CaptureConfig, RunConfig
from shardscope.errors import ConfigError
from shardscope.execution import Uninstrumented, run_training
from shardscope.publication import staged_run


def config(wait=1, warmup=1, active=2, repeat=2, **kw):
    return RunConfig(
        steps=(wait + warmup + active) * repeat + 1,
        batch_size=2,
        samples=8,
        device="cpu",
        capture=CaptureConfig(wait, warmup, active, repeat, ("cpu",)),
        **kw,
    )


class SyntheticProfiler:
    """Controllable lifecycle using the installed schedule and invented events."""

    def __init__(self, *, schedule, on_trace_ready, **kwargs):
        self.schedule, self.callback = schedule, on_trace_ready
        self.step_num = 0
        self.current_action = schedule(0)
        self.steps = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def step(self):
        if self.current_action in (
            torch.profiler.ProfilerAction.RECORD,
            torch.profiler.ProfilerAction.RECORD_AND_SAVE,
        ):
            self.steps.append(self.step_num)
        save = self.current_action == torch.profiler.ProfilerAction.RECORD_AND_SAVE
        self.step_num += 1
        self.current_action = self.schedule(self.step_num)
        if save:
            self.callback(self)
            self.steps.clear()

    def export_chrome_trace(self, path):
        events = []
        for step in self.steps:
            for name, offset, dur in [
                (f"ProfilerStep#{900 + step}", 0, 100),
                ("shardscope/step", 1, 90),
                (f"shardscope/training_step/{step}", 2, 80),
            ]:
                events.append(
                    dict(name=name, ph="X", pid=1, tid=1, ts=step * 100 + offset, dur=dur)
                )
        with open(path, "w") as handle:
            json.dump({"synthetic": True, "traceEvents": events}, handle)


@pytest.mark.parametrize(
    "wait,warmup,active,repeat", [(0, 0, 1, 1), (1, 1, 1, 2), (0, 1, 2, 3), (2, 0, 1, 2)]
)
def test_cycles_map_explicit_steps(tmp_path, wait, warmup, active, repeat):
    cfg = config(wait, warmup, active, repeat)
    dest = tmp_path / "run"
    capture.profile_run(cfg, dest, profiler_factory=SyntheticProfiler)
    run = load_run(dest)
    extra = run.environment["extra"]
    emitted = extra["capture"]
    cycle = wait + warmup + active
    expected = [c * cycle + wait + warmup + i for c in range(repeat) for i in range(active)]
    assert [x["training_step"] for x in emitted["step_map"]] == expected
    assert emitted["trace_files"] == [f"trace-{i:05d}.json" for i in range(repeat)]
    assert all(
        x["profiler_step_name"] == f"ProfilerStep#{900 + x['training_step']}"
        for x in emitted["step_map"]
    )
    assert extra["profiled"] is True
    assert extra["rank"] == 0 and extra["world_size"] == 1
    assert extra["artifact_schema"] == 1
    assert len(run.steps) == cfg.steps


def test_export_failure_leaves_no_run(tmp_path):
    class Failing(SyntheticProfiler):
        def export_chrome_trace(self, path):
            super().export_chrome_trace(path)
            raise OSError("export failed")

    dest = tmp_path / "failed"
    with pytest.raises(OSError, match="export failed"):
        capture.profile_run(config(), dest, profiler_factory=Failing)
    assert list(tmp_path.iterdir()) == []


def test_training_failure_cleanup(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("training failed")

    monkeypatch.setattr(capture, "run_training", fail)
    with pytest.raises(RuntimeError):
        capture.profile_run(config(), tmp_path / "run", profiler_factory=SyntheticProfiler)
    assert list(tmp_path.iterdir()) == []


def test_unavailable_activity_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.profiler, "supported_activities", lambda: set())
    with pytest.raises(ConfigError, match="unavailable"):
        capture.profile_run(config(), tmp_path / "run")
    assert not list(tmp_path.iterdir())


def test_missing_capture_and_mps_refused():
    with pytest.raises(ConfigError, match="explicit capture"):
        capture.validate_capture(RunConfig(device="cpu"))
    from dataclasses import replace

    with pytest.raises(ConfigError, match="MPS"):
        capture.validate_capture(replace(config(), device="mps"))


def test_nonempty_output_preserved(tmp_path):
    dest = tmp_path / "existing"
    dest.mkdir()
    (dest / "other").write_text("keep")
    with pytest.raises(ConfigError, match="nonempty"), staged_run(dest):
        pytest.fail("entered")
    assert (dest / "other").read_text() == "keep"


def test_racing_writer_preserved(tmp_path):
    dest = tmp_path / "run"
    with pytest.raises(OSError), staged_run(dest) as stage:
        (stage / "complete").write_text("new")
        (dest / "racing").write_text("keep")
    assert [p.name for p in dest.iterdir()] == ["racing"]
    assert [p.name for p in tmp_path.iterdir()] == ["run"]


def test_replaced_destination_preserved(tmp_path):
    dest = tmp_path / "run"
    with pytest.raises(ConfigError, match="changed"), staged_run(dest):
        dest.rename(tmp_path / "reserved")
        dest.mkdir()
    assert dest.is_dir()


def test_existing_empty_directory_published(tmp_path):
    dest = tmp_path / "run"
    dest.mkdir()
    with staged_run(dest) as stage:
        (stage / "done").write_text("yes")
    assert (dest / "done").read_text() == "yes"


def test_no_profiler_in_regular_training(monkeypatch):
    def fail(*a, **kw):
        pytest.fail("profiler attached")

    monkeypatch.setattr(torch.profiler, "profile", fail)
    monkeypatch.setattr(torch.profiler, "record_function", fail)
    records = run_training(steps=2, batch_size=2, device="cpu", n_samples=8)
    assert len(records) == 2


def test_instrumentation_preserves_loss_and_rng():
    args = dict(steps=2, batch_size=2, device="cpu", n_samples=8, synchronize=False)
    before = torch.get_rng_state()
    first = run_training(**args)

    class Hooks(Uninstrumented):
        def phase(self, name):
            return nullcontext()

    second = run_training(**args, instrumentation=Hooks())
    assert torch.equal(before, torch.get_rng_state())
    assert [r.loss for r in first] == [r.loss for r in second]
    assert not any(r.synchronized for r in second)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_real_cuda_capture(tmp_path):
    from dataclasses import replace

    cfg = config(1, 1, 2, 2)
    cfg = replace(cfg, device="cuda", capture=replace(cfg.capture, activities=("cpu", "cuda")))
    capture.profile_run(cfg, tmp_path / "cuda")
    run = load_run(tmp_path / "cuda")
    assert len(run.environment["extra"]["capture"]["step_map"]) == 4
    for name in run.environment["extra"]["capture"]["trace_files"]:
        events = json.loads((tmp_path / "cuda" / name).read_text())["traceEvents"]
        assert any(e.get("cat") == "kernel" for e in events)


@pytest.mark.parametrize("wait,warmup,active,repeat", [(0, 0, 1, 2), (1, 1, 2, 2)])
def test_installed_profiler_lifecycle_software_only(tmp_path, wait, warmup, active, repeat):
    # Structural lifecycle verification only; no CPU timing acceptance claims.
    capture.profile_run(config(wait, warmup, active, repeat), tmp_path / "actual")
    extra = load_run(tmp_path / "actual").environment["extra"]
    assert len(extra["capture"]["step_map"]) == active * repeat
    assert len(extra["capture"]["trace_files"]) == repeat


def test_exclusive_publication_refuses_race_after_reservation_removal(tmp_path, monkeypatch):
    from shardscope import publication

    original = publication.publish_exclusive
    destination = tmp_path / "race"

    def race(source, dest):
        dest.mkdir()
        original(source, dest)

    monkeypatch.setattr(publication, "publish_exclusive", race)
    with pytest.raises(FileExistsError), publication.staged_run(destination) as stage:
        (stage / "completed").write_text("not published")
    assert destination.is_dir() and not list(destination.iterdir())
    assert list(tmp_path.iterdir()) == [destination]


def test_opt_in_iterator_markers_and_provenance(tmp_path):
    import profile_delayed

    cfg = config(0, 1, 1, 1)
    capture.profile_run(
        cfg,
        tmp_path / "iterator",
        batch_iterator=profile_delayed.delayed_batches(cfg, 0),
        batch_source_metadata=dict(kind="synthetic_delayed_host_iterator", delay_ms=0),
    )
    run = load_run(tmp_path / "iterator")
    recorded = run.environment["extra"]["capture"]
    assert recorded["batch_source"]["delay_ms"] == 0
    events = json.loads((tmp_path / "iterator" / recorded["trace_files"][0]).read_text())[
        "traceEvents"
    ]
    names = {event.get("name") for event in events}
    assert {
        f"shardscope/{name}"
        for name in ("step", "data_wait", "h2d", "forward", "backward", "optimizer")
    } <= names


def test_synthetic_shared_interface_fixture(tmp_path):
    from pathlib import Path

    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "synthetic_capture_interface.json").read_text()
    )
    assert fixture["synthetic"] is True
    settings = fixture["environment_extra"]["capture"]
    actual = []
    for name in settings["trace_files"]:
        path = tmp_path / name
        path.write_text(json.dumps(fixture["traces"][name]))
        expected = [x["training_step"] for x in settings["step_map"] if x["trace_file"] == name]
        actual.extend(capture.map_trace(path, expected, rank=0))
    assert actual == settings["step_map"]


def test_artifact_write_failure_cleanup(tmp_path, monkeypatch):
    def fail(*a, **kw):
        raise OSError("artifact write failed")

    monkeypatch.setattr(capture, "write_recording", fail)
    with pytest.raises(OSError, match="artifact write"):
        capture.profile_run(config(), tmp_path / "run", profiler_factory=SyntheticProfiler)
    assert not list(tmp_path.iterdir())


def test_ambiguous_identity_is_refused(tmp_path):
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps({"traceEvents": []}))
    with pytest.raises(ConfigError, match="identity marker"):
        capture.map_trace(path, [0], 0)


def test_no_sync_does_not_add_synchronization(monkeypatch, tmp_path):
    from shardscope import execution

    def fail(*a, **kw):
        pytest.fail("explicit synchronize in --no-sync capture")

    monkeypatch.setattr(execution, "_synchronize", fail)
    cfg = config(synchronize=False)
    capture.profile_run(cfg, tmp_path / "run", profiler_factory=SyntheticProfiler)
    assert not any(x.synchronized for x in load_run(tmp_path / "run").steps)


@pytest.mark.parametrize("synchronized", [True, False])
def test_loss_extraction_and_sync_stay_outside_timer(monkeypatch, synchronized):
    from types import SimpleNamespace

    from shardscope import execution

    events = []
    ticks = iter([1.0, 2.0])

    def clock():
        value = next(ticks)
        events.append(f"timer-{value}")
        return value

    original_item = torch.Tensor.item

    def item(tensor, *a, **kw):
        events.append("loss-item")
        return original_item(tensor, *a, **kw)

    monkeypatch.setattr(execution, "time", SimpleNamespace(perf_counter=clock))
    monkeypatch.setattr(execution, "_synchronize", lambda d: events.append("sync"))
    monkeypatch.setattr(torch.Tensor, "item", item)
    records = execution.run_training(
        steps=1, batch_size=2, n_samples=8, device="cpu", synchronize=synchronized
    )
    expected = (["sync"] if synchronized else []) + ["timer-1.0"]
    expected += (["sync"] if synchronized else []) + ["timer-2.0", "loss-item"]
    assert events == expected
    assert records[0].step_time_ms == 1000
