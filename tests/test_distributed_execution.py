"""DDP launch/configuration doubles and separate genuine CUDA acceptance."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from shardscope import distributed as ddp
from shardscope.artifacts import load_run
from shardscope.capture import write_recording
from shardscope.config import RunConfig
from shardscope.errors import ConfigError
from shardscope.execution import StepRecord


def environment(**changes):
    env = dict(
        RANK="1",
        LOCAL_RANK="1",
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
        MASTER_ADDR="localhost",
        MASTER_PORT="29500",
    )
    env.update(changes)
    return env


def test_explicit_torchrun_launch():
    assert ddp.Launch.from_environment(environment()) == ddp.Launch(1, 1, 2, 2)
    assert ddp.rank_directory("out", 12) == Path("out/rank-00012")


@pytest.mark.parametrize(
    "changes",
    [
        dict(RANK="2"),
        dict(WORLD_SIZE="1"),
        dict(LOCAL_RANK="-1"),
        dict(LOCAL_WORLD_SIZE="0"),
        dict(RANK="bad"),
        dict(MASTER_PORT="70000"),
    ],
)
def test_invalid_launch(changes):
    with pytest.raises(ConfigError):
        ddp.Launch.from_environment(environment(**changes))


def test_missing_environment():
    with pytest.raises(ConfigError, match="torchrun"):
        ddp.Launch.from_environment({})


@pytest.fixture
def backend(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: calls.append(("device", rank)))
    monkeypatch.setattr(ddp.dist, "is_available", lambda: True)
    monkeypatch.setattr(ddp.dist, "is_nccl_available", lambda: True)
    monkeypatch.setattr(ddp.dist, "init_process_group", lambda **kw: calls.append(("init", kw)))
    monkeypatch.setattr(ddp.dist, "destroy_process_group", lambda: calls.append(("destroy",)))

    def gather(output, value):
        output[:] = [value] * len(output)

    monkeypatch.setattr(ddp.dist, "all_gather_object", gather)
    return calls


def test_setup_failure_cleans_process_group(tmp_path, backend):
    with pytest.raises(ConfigError, match="explicitly select"):
        ddp.execute(RunConfig(device="cpu"), tmp_path / "run", ddp.Launch(0, 0, 2, 2))
    assert backend[0] == ("device", 0)
    assert backend[1][1]["backend"] == "nccl"
    assert backend[1][1]["timeout"].total_seconds() == 60
    assert backend[-1] == ("destroy",)
    assert list(tmp_path.iterdir()) == []


def test_peer_error_prevents_collective_training(monkeypatch, backend):
    def gather(output, value):
        output[:] = [None, "peer setup failed"]

    monkeypatch.setattr(ddp.dist, "all_gather_object", gather)
    with pytest.raises(ConfigError, match="peer setup failed"), ddp.coordinated_setup(2):
        pass


def test_rank_setup_and_independent_output(monkeypatch, tmp_path, backend):
    def training(**kwargs):
        assert kwargs["rank"] == 0 and kwargs["world_size"] == 2
        assert kwargs["device"] == "cuda:0"
        with kwargs["setup_guard"]():
            pass
        assert kwargs["model_wrapper"]("model") == "wrapped"
        return ()

    def wrap(model, **kwargs):
        assert model == "model" and kwargs == dict(device_ids=[0], output_device=0)
        return "wrapped"

    def write(stage, config, records, device, **kwargs):
        assert kwargs["profiled"] is False and kwargs["world_size"] == 2
        (stage / "synthetic.txt").write_text("software fixture")

    monkeypatch.setattr(ddp, "run_training", training)
    monkeypatch.setattr(ddp, "DistributedDataParallel", wrap)
    monkeypatch.setattr(ddp, "write_recording", write)
    root = tmp_path / "run"
    dest = ddp.execute(RunConfig(device="cuda"), root, ddp.Launch(0, 0, 2, 2))
    assert dest == root / "rank-00000"
    assert (dest / "synthetic.txt").is_file()
    assert backend[-1] == ("destroy",)


def test_rank_provenance_is_loadable_software_fixture(tmp_path):
    cfg = RunConfig(device="cpu", steps=1, samples=8, batch_size=2)
    records = (StepRecord(0, 1, 2, "host", True),)
    write_recording(
        tmp_path,
        cfg,
        records,
        torch.device("cpu"),
        profiled=False,
        rank=1,
        local_rank=1,
        world_size=2,
    )
    extra = load_run(tmp_path).environment["extra"]
    assert extra["rank"] == 1 and extra["local_rank"] == 1
    assert extra["world_size"] == 2 and extra["global_batch_size"] == 4
    assert extra["work_scope"] == "per_rank" and extra["profiled"] is False
    assert extra["hostname"] and extra["pid"] > 0


def test_data_partition_is_disjoint_deterministic(monkeypatch):
    from shardscope import execution

    seen = []
    original = execution._batch_index_schedule

    def schedule(n_samples, *args):
        seen.append(n_samples)
        return original(n_samples, *args)

    monkeypatch.setattr(execution, "_batch_index_schedule", schedule)
    args = dict(steps=2, batch_size=2, n_samples=9, device="cpu", world_size=2)
    first = execution.run_training(**args, rank=0)
    second = execution.run_training(**args, rank=1)
    assert seen == [5, 4]
    assert [r.loss for r in first] != [r.loss for r in second]
    assert [r.loss for r in execution.run_training(**args, rank=1)] == [r.loss for r in second]


@pytest.mark.gpu
@pytest.mark.distributed
@pytest.mark.parametrize("profiled", [False, True])
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires >=2 real CUDA devices; CPU DDP is not acceptance",
)
@pytest.mark.skipif(not ddp.dist.is_nccl_available(), reason="NCCL backend is unavailable")
def test_real_torchrun_ddp(tmp_path, profiled):
    config = dict(device="cuda", steps=6, batch_size=2, samples=16)
    if profiled:
        config["capture"] = dict(wait=1, warmup=1, active=1, repeat=2, activities=["cpu", "cuda"])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    out = tmp_path / "ranks"
    argv = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "examples/train_ddp.py",
        "--config",
        str(path),
        "--out",
        str(out),
    ]
    if profiled:
        argv.append("--profile")
    completed = subprocess.run(
        argv,
        cwd=Path(__file__).resolve().parents[1],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for rank in range(2):
        run = load_run(ddp.rank_directory(out, rank))
        extra = run.environment["extra"]
        assert extra["rank"] == rank and extra["world_size"] == 2
        assert extra["profiled"] is profiled
        assert len(run.steps) == 6
        if profiled:
            assert len(extra["capture"]["trace_files"]) == 2
            assert all(x["rank"] == rank for x in extra["capture"]["step_map"])
