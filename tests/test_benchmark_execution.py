"""Software validation of unprofiled windows and descriptive output."""

import copy
import json

import pytest
import torch

from shardscope import benchmark
from shardscope.cli import main
from shardscope.config import RunConfig
from shardscope.errors import ConfigError


def test_completed_window_excludes_setup_warmup_and_loss(monkeypatch, tmp_path):
    ticks = iter([20.0, 22.0])
    sync = []
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(benchmark, "_synchronize", lambda d: sync.append(str(d)))

    def execution(**kw):
        assert kw["synchronize"] is False and kw["collect_losses"] is False
        for step in range(kw["steps"]):
            kw["instrumentation"].before_step(step, "cpu")
            kw["instrumentation"].after_step(step, "cpu")

    monkeypatch.setattr(benchmark, "run_training", execution)
    result = benchmark.benchmark_run(
        RunConfig(device="cpu", batch_size=4),
        benchmark.BenchmarkSettings(3, 1, 8),
        tmp_path / "bench",
    )
    item = result["observations"][0]
    assert item["elapsed_seconds"] == 2
    assert item["steps_per_second"] == 4
    assert item["examples_per_second"] == 16
    assert sync == ["cpu", "cpu"]
    assert benchmark.load_benchmark(tmp_path / "bench") == result


def test_actual_benchmark_has_no_profiler_or_loss_extraction(monkeypatch, tmp_path):
    def fail(*a, **kw):
        pytest.fail("profiler or scalar extraction in benchmark")

    monkeypatch.setattr(torch.profiler, "profile", fail)
    monkeypatch.setattr(torch.profiler, "record_function", fail)
    monkeypatch.setattr(torch.Tensor, "item", fail)
    cfg = RunConfig(device="cpu", samples=8, batch_size=2)
    result = benchmark.benchmark_run(cfg, benchmark.BenchmarkSettings(1, 2, 2), tmp_path / "b")
    assert len(result["observations"]) == 2
    assert result["environment"]["extra"]["profiled"] is False
    assert result["significance"]["status"] == "unconfigured"
    compared = benchmark.compare_benchmarks(result, result)
    assert compared["metrics"]["steps_per_second"]["descriptive_delta_percent"] == 0
    assert compared["significance"]["decision"] is None


@pytest.fixture
def result(tmp_path):
    return benchmark.benchmark_run(
        RunConfig(device="cpu", samples=8, batch_size=2),
        benchmark.BenchmarkSettings(0, 1, 1),
        tmp_path / "b",
    )


@pytest.mark.parametrize("key,value", [("warmup", -1), ("repeat", 0), ("measured_steps", True)])
def test_settings_explicit_and_valid(key, value):
    args = dict(warmup=0, repeat=1, measured_steps=1)
    args[key] = value
    with pytest.raises(ConfigError):
        benchmark.BenchmarkSettings(**args)


@pytest.mark.parametrize(
    "field,value",
    [("batch_size", 32), ("seed", 9), ("data_residency", "device"), ("model", "other")],
)
def test_incompatible_workloads_refused(result, field, value):
    other = copy.deepcopy(result)
    other["workload"][field] = value
    with pytest.raises((ConfigError, ValueError)):
        benchmark.compare_benchmarks(result, other)


@pytest.mark.parametrize(
    "field,value", [("profiled", True), ("schema_version", 0), ("units", {}), ("observations", [])]
)
def test_invalid_input_refused(result, tmp_path, field, value):
    result[field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(result))
    with pytest.raises(ConfigError):
        benchmark.load_benchmark(path)


def test_raw_rates_verified(result):
    result["observations"][0]["steps_per_second"] *= 2
    with pytest.raises(ValueError, match="disagrees"):
        benchmark.validate_benchmark(result)


def test_cli_bench_and_compare(tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text('{"device":"cpu","batch_size":2,"samples":8}')
    out = tmp_path / "bench"
    assert (
        main(
            [
                "bench",
                "--config",
                str(path),
                "--out",
                str(out),
                "--warmup",
                "0",
                "--repeat",
                "1",
                "--measured-steps",
                "1",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["compare", "--baseline", str(out), "--candidate", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["significance"]["status"] == "unconfigured"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_cuda_completed_work_benchmark(tmp_path):
    result = benchmark.benchmark_run(
        RunConfig(device="cuda", batch_size=2, samples=8),
        benchmark.BenchmarkSettings(1, 2, 2),
        tmp_path / "cuda",
    )
    assert all(x["elapsed_seconds"] > 0 for x in result["observations"])
    assert result["profiled"] is False


def test_compare_rejects_different_window_policy(result):
    candidate = copy.deepcopy(result)
    candidate["timing"]["scope"] = "other scope"
    with pytest.raises(ConfigError, match="timing policies"):
        benchmark.compare_benchmarks(result, candidate)


def test_compare_rejects_profiled_input(result):
    candidate = copy.deepcopy(result)
    candidate["environment"]["extra"]["profiled"] = True
    with pytest.raises(ValueError, match="unprofiled"):
        benchmark.compare_benchmarks(result, candidate)


def test_bench_refuses_capture_config(tmp_path):
    from shardscope.config import CaptureConfig

    cfg = RunConfig(device="cpu", capture=CaptureConfig(0, 1, 1, 1, ("cpu",)))
    with pytest.raises(ConfigError, match="refuses capture"):
        benchmark.benchmark_run(cfg, benchmark.BenchmarkSettings(1, 1, 1), tmp_path / "run")
    assert not list(tmp_path.iterdir())


def test_failure_cleans_benchmark_stage(tmp_path, monkeypatch):
    def fail(**kw):
        raise RuntimeError("execution failure")

    monkeypatch.setattr(benchmark, "run_training", fail)
    with pytest.raises(RuntimeError):
        benchmark.benchmark_run(
            RunConfig(device="cpu"), benchmark.BenchmarkSettings(1, 1, 1), tmp_path / "run"
        )
    assert not list(tmp_path.iterdir())
