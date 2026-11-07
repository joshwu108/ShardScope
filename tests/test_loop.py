"""Loop-integration API: software contract on CPU; GPU acceptance is in test_loop_gpu."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import shardscope
from shardscope import cli
from shardscope.analysis import analyze_run, validate_analysis, write_analysis
from shardscope.artifacts import load_run
from shardscope.errors import ConfigError
from shardscope.findings import render_findings_text, summarize_findings
from shardscope.loop import PHASES, SOURCE, LoopProfiler, LoopUsageError, describe_module
from shardscope.parse import HOST_PHASES
from shardscope.report import render_html, render_json
from shardscope.rules import RULE_IDS, RuleConfig

CAPTURE = dict(wait=1, warmup=1, active=2, repeat=2, activities=["cpu"])


def tiny_model(seed=0):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))


def tiny_loader(samples=64, batch=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(samples, 8, generator=generator)
    targets = torch.randint(0, 3, (samples,), generator=generator)
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch)


def run_loop(out, steps=8, *, phases=PHASES, capture=CAPTURE, loss="tensor", **settings):
    """Train the tiny model for ``steps`` steps under the API; returns the profiler."""
    model, loader = tiny_model(), tiny_loader()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    with shardscope.profile_loop(out, capture=capture, device="cpu", **settings) as prof:
        batches = iter(loader)
        for step in range(steps):
            with prof.step(step) as rec:
                with rec.phase("data_wait") if "data_wait" in phases else _nothing():
                    inputs, targets = next(batches)
                with rec.phase("h2d") if "h2d" in phases else _nothing():
                    inputs, targets = inputs.to("cpu"), targets.to("cpu")
                with rec.phase("forward") if "forward" in phases else _nothing():
                    value = loss_fn(model(inputs), targets)
                with rec.phase("backward") if "backward" in phases else _nothing():
                    value.backward()
                with rec.phase("optimizer") if "optimizer" in phases else _nothing():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if loss == "tensor":
                    rec.loss(value)
                elif loss == "float":
                    rec.loss(float(value.detach()))
    return prof


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --- end to end -------------------------------------------------------------------


def test_loop_run_is_a_complete_shardscope_run(tmp_path):
    out = tmp_path / "loop"
    prof = run_loop(out)
    assert prof.recorded_steps == tuple(range(8))
    run = load_run(out)
    assert [row.step for row in run.steps] == list(range(8))
    assert all(math.isfinite(row.loss) for row in run.steps)
    assert all(row.synchronized and row.data_residency == "host" for row in run.steps)
    extra = run.environment["extra"]
    assert extra["source"] == SOURCE and extra["profiled"] is True
    assert (extra["rank"], extra["local_rank"], extra["world_size"]) == (0, 0, 1)
    assert extra["steps"] == 8 and extra["workload"] == {}
    capture = extra["capture"]
    assert capture["trace_files"] == ["trace-00000.json", "trace-00001.json"]
    assert [m["training_step"] for m in capture["step_map"]] == [2, 3, 6, 7]
    assert capture["step_map_schema"] == 1
    assert capture["phases_entered"] == sorted(PHASES)
    assert capture["data_wait"].startswith("caller-entered")
    assert capture["integration"] == "shardscope.loop.profile_loop"
    assert run.summary["steps"] == 8.0
    for name in capture["trace_files"]:
        names = {e.get("name") for e in json.loads((out / name).read_text())["traceEvents"]}
        assert set(HOST_PHASES) <= names


def test_loop_run_flows_through_analyze_validate_findings_and_reports(tmp_path, capsys):
    out = tmp_path / "loop"
    run_loop(out)
    config = RuleConfig(0.0, h2d_share_threshold=0.0, step_variability_ratio=1.0)
    payload = analyze_run(out, config=config)
    mapped = {row["training_step"]: row for row in payload["steps"] if row["boundary"]}
    assert sorted(mapped) == [2, 3, 6, 7]
    for row in mapped.values():
        for metric in HOST_PHASES.values():
            measured = row["measurements"][metric]
            assert measured["status"] == "recorded" and measured["basis"] == "host_range"
            assert measured["evidence"] and measured["value_ms"] >= 0
    for row in payload["steps"]:
        if row["boundary"] is None:
            assert row["measurements"]["host_step_ms"]["status"] == "not_recorded"
    assert payload["mapping_source"] == "capture" and payload["rules"] == list(RULE_IDS)
    triggered = {f["rule_id"] for f in payload["findings"] if f["status"].startswith("possible")}
    # The host rules were configured and fire at these thresholds; the device
    # rules were left unconfigured and must say so rather than compare.
    assert triggered == {"host_loader_wait", "host_h2d_share", "step_duration_variability"}
    assert {
        f["status"]
        for f in payload["findings"]
        if f["rule_id"] in ("device_transfer_overlap", "collective_compute_overlap")
    } == {"unconfigured"}
    write_analysis(out, payload)
    run = load_run(out)
    validate_analysis(run.analysis, directory=out, environment=run.environment)
    summary = summarize_findings(run)
    assert summary["mapped_steps"] == 4 and summary["total_steps"] == 8
    text = render_findings_text(summary)
    assert "host_loader_wait [loader_wait_threshold_ms=0]" in text
    assert "step_duration_variability" in text
    html = render_html(run)
    assert SOURCE in html and "host_h2d_share" in html
    assert json.loads(render_json(run))["environment"]["extra"]["source"] == SOURCE
    assert cli.main(["validate", "--run", str(out)]) == 0
    assert "valid run" in capsys.readouterr().out
    assert cli.main(["findings", "--run", str(out), "--rule", "host_h2d_share"]) == 0
    assert "possible_host_copy_wait" in capsys.readouterr().out


def test_package_import_stays_torch_free_and_resolves_lazily():
    code = (
        "import sys, shardscope; assert 'torch' not in sys.modules; "
        "shardscope.profile_loop; shardscope.describe_module; assert 'torch' in sys.modules; "
        "print(sorted(shardscope.__all__))"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "profile_loop" in completed.stdout
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        shardscope.nope  # noqa: B018


# --- nothing recorded on failure ------------------------------------------------------


def test_short_schedule_records_nothing_and_names_the_shortfall(tmp_path):
    out = tmp_path / "short"
    with pytest.raises(ConfigError, match="needs 8 wrapped training steps .* wrapped 5; 1 of 2"):
        run_loop(out, steps=5)
    assert not out.exists() and list(tmp_path.iterdir()) == []


def test_schedule_cut_during_warmup_records_nothing(tmp_path):
    out = tmp_path / "short"
    with pytest.raises(ConfigError, match="wrapped 2; 0 of 2"):
        run_loop(out, steps=2)
    assert not out.exists()


def test_loop_exception_records_nothing(tmp_path):
    out = tmp_path / "boom"
    with (
        pytest.raises(RuntimeError, match="user loop"),
        shardscope.profile_loop(out, capture=CAPTURE, device="cpu") as prof,
    ):
        with prof.step(0):
            pass
        with prof.step(1), pytest.raises(KeyError):
            raise KeyError("recovered inside the step")
        raise RuntimeError("user loop failed")
    assert not out.exists() and list(tmp_path.iterdir()) == []


def test_nothing_wrapped_is_refused(tmp_path):
    out = tmp_path / "empty"
    with (
        pytest.raises(LoopUsageError, match="no training step was wrapped"),
        shardscope.profile_loop(out, capture=dict(CAPTURE, wait=0, warmup=0), device="cpu"),
    ):
        pass
    assert not out.exists()


def test_nonempty_destination_is_refused_before_training(tmp_path):
    out = tmp_path / "taken"
    out.mkdir()
    (out / "keep").write_text("mine")
    with pytest.raises(ConfigError, match="nonempty"):
        run_loop(out)
    assert (out / "keep").read_text() == "mine"


# --- step and phase contract -----------------------------------------------------------


def test_step_indices_must_be_nonnegative_strictly_increasing_ints(tmp_path):
    with shardscope.profile_loop(tmp_path / "x", capture=CAPTURE, device="cpu") as prof:
        with prof.step(3):
            pass
        for bad, message in ((3, "strictly increase"), (2, "strictly increase")):
            with pytest.raises(LoopUsageError, match=message):
                prof.step(bad)
        with pytest.raises(LoopUsageError, match="nonnegative int"):
            prof.step(-1)
        with pytest.raises(LoopUsageError, match="nonnegative int"):
            prof.step(4.0)
        with pytest.raises(LoopUsageError, match="nonnegative int"):
            prof.step(True)
        for step in (5, 6, 7, 9, 10, 11, 12):
            with prof.step(step):
                pass
    assert [row.step for row in load_run(tmp_path / "x").steps] == [3, 5, 6, 7, 9, 10, 11, 12]


def test_steps_do_not_nest_and_need_the_open_profiler(tmp_path):
    profiler = shardscope.profile_loop(tmp_path / "x", capture=CAPTURE, device="cpu")
    with pytest.raises(LoopUsageError, match="inside the profiler"):
        profiler.step(0)
    with pytest.raises(ConfigError, match="wrapped 1"), profiler as prof:
        with prof.step(0) as rec:
            with pytest.raises(LoopUsageError, match="do not nest"), prof.step(1):
                pass
            assert prof.recorded_steps == ()
            rec.loss(1.5)
        assert prof.recorded_steps == (0,)
        with pytest.raises(LoopUsageError, match="already finished"):
            rec.loss(2.0)
    with pytest.raises(LoopUsageError, match="inside the profiler"):
        profiler.step(2)
    with pytest.raises(LoopUsageError, match="records one run"), profiler:
        pass


def test_phase_names_threads_and_placement_are_checked(tmp_path):
    with (
        pytest.raises(ConfigError, match="wrapped 1"),
        shardscope.profile_loop(tmp_path / "x", capture=CAPTURE, device="cpu") as prof,
    ):
        with prof.step(0) as rec:
            with pytest.raises(LoopUsageError, match="unknown phase 'step'"):
                rec.phase("step")
            with pytest.raises(LoopUsageError, match="unknown phase 'training_step/0'"):
                rec.phase("training_step/0")
            failures = []

            def worker():
                try:
                    with rec.phase("forward"):
                        pass
                except LoopUsageError as exc:
                    failures.append(str(exc))

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            assert failures and "different thread" in failures[0]
            with rec.phase("forward"):
                pass
            with rec.phase("forward"):  # repeated phases are allowed and unioned
                pass
        with pytest.raises(LoopUsageError, match="outside its step 0"), rec.phase("backward"):
            pass
        failures.clear()
        thread = threading.Thread(target=lambda: failures.append(_try_step(prof)))
        thread.start()
        thread.join()
        assert "thread that entered the profiler" in failures[0]
    assert not (tmp_path / "x").exists()


def _try_step(prof):
    try:
        prof.step(5)
    except LoopUsageError as exc:
        return str(exc)
    return "no error"


def test_repeated_phase_is_unioned_with_a_warning(tmp_path):
    out = tmp_path / "twice"
    with shardscope.profile_loop(out, capture=dict(CAPTURE, repeat=1), device="cpu") as prof:
        for step in range(4):
            with prof.step(step) as rec:
                with rec.phase("forward"):
                    pass
                with rec.phase("forward"):
                    pass
    payload = analyze_run(out)
    assert any("multiple shardscope/forward ranges" in w for w in payload["warnings"])
    mapped = [row for row in payload["steps"] if row["boundary"]]
    assert all(len(row["measurements"]["host_forward_ms"]["evidence"]) == 2 for row in mapped)
    assert all(row["measurements"]["host_h2d_ms"]["status"] == "not_recorded" for row in mapped)


def test_unentered_phases_are_not_recorded_not_zero(tmp_path):
    out = tmp_path / "partial"
    run_loop(out, phases=("forward", "backward"))
    capture = load_run(out).environment["extra"]["capture"]
    assert capture["phases_entered"] == ["backward", "forward"]
    assert capture["data_wait"].startswith("not_recorded")
    payload = analyze_run(out, config=RuleConfig(0.0, h2d_share_threshold=0.0))
    mapped = [row for row in payload["steps"] if row["boundary"]]
    for row in mapped:
        assert row["measurements"]["host_forward_ms"]["status"] == "recorded"
        assert row["measurements"]["host_data_wait_ms"]["status"] == "not_recorded"
        assert row["measurements"]["host_data_wait_ms"]["value_ms"] is None
    statuses = {(f["rule_id"], f["status"]) for f in payload["findings"]}
    assert ("host_loader_wait", "unavailable") in statuses
    assert ("host_h2d_share", "unavailable") in statuses
    assert ("host_loader_wait", "possible_input_wait") not in statuses


# --- loss and settings -------------------------------------------------------------


def test_loss_forms(tmp_path):
    run_loop(tmp_path / "tensor", loss="tensor")
    run_loop(tmp_path / "float", loss="float")
    run_loop(tmp_path / "none", loss=None)
    assert all(math.isfinite(r.loss) for r in load_run(tmp_path / "tensor").steps)
    assert all(math.isfinite(r.loss) for r in load_run(tmp_path / "float").steps)
    assert all(math.isnan(r.loss) for r in load_run(tmp_path / "none").steps)
    with (
        pytest.raises(ConfigError, match="wrapped 1"),
        shardscope.profile_loop(tmp_path / "bad", capture=CAPTURE, device="cpu") as prof,
        prof.step(0) as rec,
    ):
        with pytest.raises(LoopUsageError, match="single value"):
            rec.loss(torch.zeros(2))
        with pytest.raises(LoopUsageError, match="number or a one-element tensor"):
            rec.loss("0.5")
        with pytest.raises(LoopUsageError, match="number or a one-element tensor"):
            rec.loss(True)
        rec.loss(torch.tensor([[0.25]]))
        rec.loss(3)


def test_loss_tensor_is_read_after_the_timer_stops(tmp_path, monkeypatch):
    reads = []
    original = torch.Tensor.item

    def item(self):
        reads.append(float(original(self)))
        return original(self)

    monkeypatch.setattr(torch.Tensor, "item", item)
    with (
        pytest.raises(ConfigError, match="wrapped 1"),
        shardscope.profile_loop(tmp_path / "x", capture=CAPTURE, device="cpu") as prof,
    ):
        with prof.step(0) as rec:
            rec.loss(torch.tensor(0.75))
            assert reads == []
        assert reads == [0.75]


def test_synchronize_and_residency_declarations_are_recorded(tmp_path):
    out = tmp_path / "declared"
    run_loop(out, synchronize=False, data_residency="device", workload={"model": "tiny"})
    run = load_run(out)
    assert all(not row.synchronized and row.data_residency == "device" for row in run.steps)
    extra = run.environment["extra"]
    assert extra["synchronized"] is False and extra["data_residency"] == "device"
    assert extra["workload"] == {"model": "tiny"}
    payload = analyze_run(out)
    for row in payload["steps"]:
        assert row["measurements"]["device_h2d_ms"]["status"] == "not_observed"
        assert "data_residency=device" in row["measurements"]["device_h2d_ms"]["reason"]


@pytest.mark.parametrize(
    "settings,message",
    [
        (dict(data_residency="gpu"), "data_residency must be one of"),
        (dict(synchronize=1), "synchronize must be true or false"),
        (dict(rank=1), "outside world_size"),
        (dict(rank=0, world_size=0), "rank/world_size"),
        (dict(rank=1.0, world_size=2), "rank/world_size"),
        (dict(local_rank=-1), "local_rank"),
        (dict(workload=["not", "a", "mapping"]), "workload must be a mapping"),
        (dict(workload={"nan": float("nan")}), "JSON-serializable"),
        (dict(workload={"obj": object()}), "JSON-serializable"),
        (dict(capture=dict(CAPTURE, activities=["cuda"])), "must include cpu"),
        (dict(capture=dict(CAPTURE, wait=-1)), "capture.wait"),
        (dict(capture="nope"), "capture must be an object"),
        (dict(device="mps"), "MPS capture is unsupported"),
    ],
)
def test_invalid_settings_fail_before_any_training(tmp_path, settings, message):
    kwargs = dict(capture=CAPTURE, device="cpu")
    kwargs.update(settings)
    with pytest.raises(ConfigError, match=message):
        LoopProfiler(tmp_path / "never", **kwargs)
    assert list(tmp_path.iterdir()) == []


def test_cuda_request_without_cuda_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ConfigError, match="CUDA"):
        LoopProfiler(
            tmp_path / "x", capture=dict(CAPTURE, activities=["cpu", "cuda"]), device="cuda"
        )
    with pytest.raises(ConfigError, match="CUDA"):
        LoopProfiler(tmp_path / "x", capture=CAPTURE, device="cuda:0")


def test_large_schedule_is_not_capped_by_fixture_defaults(tmp_path):
    # RunConfig's own default of 100 steps must not limit a loop's schedule.
    capture = dict(wait=0, warmup=0, active=150, repeat=1, activities=["cpu"])
    profiler = LoopProfiler(tmp_path / "big", capture=capture, device="cpu")
    assert profiler.capture.required_steps == 150


def test_rank_provenance_for_a_torchrun_style_launch(tmp_path):
    out = tmp_path / "rank-00001"
    run_loop(out, rank=1, local_rank=1, world_size=2)
    extra = load_run(out).environment["extra"]
    assert (extra["rank"], extra["local_rank"], extra["world_size"]) == (1, 1, 2)
    assert all(m["rank"] == 1 for m in extra["capture"]["step_map"])
    payload = analyze_run(out)
    assert payload["rank"] == 1 and payload["rank_basis"] == "environment.extra.rank"
    with pytest.raises(Exception, match="disagrees"):
        analyze_run(out, rank=0)


def test_describe_module_is_exact_and_json_serializable():
    model = tiny_model()
    described = describe_module(model)
    assert described["parameters"] == 8 * 16 + 16 + 16 * 3 + 3
    assert described["trainable_parameters"] == described["parameters"]
    assert described["parameter_dtypes"] == ["torch.float32"]
    assert described["parameter_devices"] == ["cpu"]
    assert described["submodules"] == 3
    assert described["module_class"].endswith("Sequential")
    json.dumps(described)
    model[0].weight.requires_grad_(False)
    assert describe_module(model)["trainable_parameters"] == described["parameters"] - 8 * 16
    with pytest.raises(TypeError, match="nn.Module"):
        describe_module(object())


# --- the example script ---------------------------------------------------------------


@pytest.mark.parametrize("model", ["mlp", "cnn", "transformer"])
def test_example_trains_each_model_into_an_analyzable_run(tmp_path, model, capsys):
    import train_custom_loop

    out = tmp_path / model
    argv = ["--model", model, "--device", "cpu", "--steps", "5", "--out", str(out)]
    argv += ["--wait", "0", "--warmup", "1", "--active", "2", "--repeat", "1", "--batch-size", "4"]
    assert train_custom_loop.main(argv) == 0
    assert capsys.readouterr().out.strip() == f"wrote {out}"
    run = load_run(out)
    workload = run.environment["extra"]["workload"]
    assert workload["example_model"] == model and workload["parameters"] > 0
    assert workload["amp"] is None and workload["samples_per_rank"] == 256
    assert len(run.steps) == 5 and all(math.isfinite(row.loss) for row in run.steps)
    payload = analyze_run(out, config=RuleConfig(step_variability_ratio=1.0))
    mapped = [row for row in payload["steps"] if row["boundary"]]
    assert [row["training_step"] for row in mapped] == [1, 2]
    for row in mapped:
        for metric in HOST_PHASES.values():
            assert row["measurements"][metric]["status"] == "recorded"
    assert any(f["status"] == "possible_step_variability" for f in payload["findings"])


def test_example_amp_no_sync_and_loader_wraparound(tmp_path):
    import train_custom_loop

    out = tmp_path / "amp"
    argv = [
        "--model",
        "mlp",
        "--device",
        "cpu",
        "--steps",
        "6",
        "--samples",
        "8",
        "--out",
        str(out),
    ]
    argv += ["--wait", "0", "--warmup", "0", "--active", "3", "--repeat", "2", "--batch-size", "4"]
    argv += ["--amp", "--no-sync", "--quiet"]
    assert train_custom_loop.main(argv) == 0
    run = load_run(out)
    assert all(not row.synchronized for row in run.steps)
    assert run.environment["extra"]["workload"]["amp"] == "bfloat16"
    assert len(run.environment["extra"]["capture"]["step_map"]) == 6


def test_example_reports_invalid_input_with_exit_two(tmp_path, capsys):
    import train_custom_loop

    base = ["--model", "mlp", "--device", "cpu", "--out", str(tmp_path / "never")]
    assert train_custom_loop.main([*base, "--steps", "0"]) == 2
    assert "--steps must be positive" in capsys.readouterr().err
    assert train_custom_loop.main([*base, "--steps", "4", "--num-workers", "-1"]) == 2
    assert "--num-workers" in capsys.readouterr().err
    assert train_custom_loop.main([*base, "--steps", "2"]) == 2
    assert "needs 4 wrapped training steps" in capsys.readouterr().err
    assert train_custom_loop.main([*base, "--steps", "4", "--ddp"]) == 2
    assert "torchrun" in capsys.readouterr().err
    assert not (tmp_path / "never").exists()
    with pytest.raises(ValueError, match="unknown model"):
        train_custom_loop.build_model("rnn", seed=0)
    with pytest.raises(ValueError, match="unknown model"):
        train_custom_loop.make_dataset("rnn", 4, seed=0)


def test_example_refuses_cuda_without_cuda(tmp_path, monkeypatch, capsys):
    import train_custom_loop

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    argv = ["--model", "mlp", "--device", "cuda", "--steps", "4", "--out", str(tmp_path / "x")]
    assert train_custom_loop.main(argv) == 2
    assert "CUDA is unavailable" in capsys.readouterr().err


def test_example_script_runs_as_a_process(tmp_path):
    out = tmp_path / "proc"
    argv = [sys.executable, "examples/train_custom_loop.py", "--model", "mlp", "--device", "cpu"]
    argv += ["--steps", "4", "--out", str(out), "--wait", "0", "--warmup", "1", "--active", "2"]
    completed = subprocess.run(
        argv, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == f"wrote {out}"
    assert load_run(out).environment["extra"]["source"] == SOURCE
