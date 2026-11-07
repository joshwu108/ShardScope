"""Hardware acceptance for the loop API: real CUDA traces, real kernels, real DDP.

These tests are evidence that caller-owned loops on a GPU produce traces the
analysis can map and attribute. They skip, not pass, without the hardware.

Captures run in a fresh interpreter. In the PyTorch build this was developed
against, a profiler session that requests only the CPU activity disables CUDA
kernel collection for every later session in the same process, and a shared
pytest process has usually run CPU-only captures first. The analysis of the
resulting run happens in-process as usual.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from shardscope import distributed as ddp
from shardscope.analysis import analyze_run, validate_analysis, write_analysis
from shardscope.artifacts import load_run
from shardscope.findings import render_findings_text, summarize_findings
from shardscope.loop import SOURCE
from shardscope.parse import HOST_PHASES
from shardscope.rules import RULE_IDS, RuleConfig

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable"),
]

ROOT = Path(__file__).resolve().parents[1]


def _fresh_process(argv, *, timeout=180):
    """Run ``argv`` from the repository root in a new interpreter; fail loudly."""
    completed = subprocess.run(
        argv, cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True, timeout=timeout
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


def _kernels(out, trace_files):
    count = 0
    for name in trace_files:
        events = json.loads((out / name).read_text())["traceEvents"]
        count += sum(1 for e in events if e.get("cat") == "kernel")
    return count


@pytest.mark.parametrize("model", ["mlp", "cnn", "transformer"])
def test_example_models_on_cuda_produce_mapped_and_attributed_traces(tmp_path, model):
    out = tmp_path / model
    argv = [sys.executable, "examples/train_custom_loop.py", "--model", model, "--device", "cuda"]
    argv += ["--steps", "8", "--out", str(out), "--quiet", "--batch-size", "8"]
    argv += ["--wait", "1", "--warmup", "1", "--active", "2", "--repeat", "2"]
    if model == "transformer":
        argv.append("--amp")
    _fresh_process(argv)
    run = load_run(out)
    extra = run.environment["extra"]
    assert extra["source"] == SOURCE and extra["device_resolved"].startswith("cuda")
    assert extra["capture"]["activities"] == ["cpu", "cuda"]
    assert extra["workload"]["parameter_devices"] == [extra["device_resolved"]]
    assert _kernels(out, extra["capture"]["trace_files"]) > 0

    config = RuleConfig(0.0, h2d_share_threshold=0.0, step_variability_ratio=1.0)
    payload = analyze_run(out, config=config)
    mapped = [row for row in payload["steps"] if row["boundary"]]
    assert [row["training_step"] for row in mapped] == [2, 3, 6, 7]
    for row in mapped:
        for metric in HOST_PHASES.values():
            assert row["measurements"][metric]["status"] == "recorded", metric
        attributed = row["measurements"]["device_attributed_kernel_ms"]
        assert attributed["status"] == "recorded" and attributed["value_ms"] > 0
        assert attributed["coverage"]["completeness"] in ("partial", "complete")
    # Every rule compared something on a real capture; none was unavailable
    # on a mapped step, and the triggered statuses are the configured ones.
    by_rule = {}
    for finding in payload["findings"]:
        by_rule.setdefault(finding["rule_id"], set()).add(finding["status"])
    assert set(by_rule) == set(RULE_IDS)
    assert by_rule["host_loader_wait"] == {"possible_input_wait", "unavailable"}
    assert by_rule["host_h2d_share"] == {"possible_host_copy_wait", "unavailable"}
    assert by_rule["step_duration_variability"] == {"possible_step_variability"}
    write_analysis(out, payload)
    run = load_run(out)
    validate_analysis(run.analysis, directory=out, environment=run.environment)
    text = render_findings_text(summarize_findings(run))
    assert "4 of 8 training step(s) mapped" in text


DIRECT_API_SCRIPT = """
import sys
import torch
from torch import nn

import shardscope

out = sys.argv[1]
device = torch.device("cuda", torch.cuda.current_device())
model = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 4)).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
loss_fn = nn.CrossEntropyLoss()
generator = torch.Generator().manual_seed(1)
inputs = torch.randn(64, 32, generator=generator).pin_memory()
targets = torch.randint(0, 4, (64,), generator=generator).pin_memory()
capture = dict(wait=0, warmup=1, active=3, repeat=1, activities=["cpu", "cuda"])
with shardscope.profile_loop(
    out, capture=capture, device=device, workload=shardscope.describe_module(model)
) as prof:
    for step in range(4):
        with prof.step(step) as rec:
            rows = slice(step * 16, step * 16 + 16)
            with rec.phase("h2d"):
                x = inputs[rows].to(device, non_blocking=True)
                y = targets[rows].to(device, non_blocking=True)
            with rec.phase("forward"):
                loss = loss_fn(model(x), y)
            with rec.phase("backward"):
                loss.backward()
            with rec.phase("optimizer"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            rec.loss(loss)
print(prof.recorded_steps)
"""


def test_direct_api_on_cuda_with_a_pinned_non_blocking_loader(tmp_path):
    out = tmp_path / "direct"
    completed = _fresh_process([sys.executable, "-c", textwrap.dedent(DIRECT_API_SCRIPT), str(out)])
    assert "(0, 1, 2, 3)" in completed.stdout
    run = load_run(out)
    extra = run.environment["extra"]
    device = extra["device_resolved"]
    assert extra["capture"]["phases_entered"] == ["backward", "forward", "h2d", "optimizer"]
    assert extra["capture"]["data_wait"].startswith("not_recorded")
    assert extra["workload"]["parameter_devices"] == [device]
    assert _kernels(out, extra["capture"]["trace_files"]) > 0
    payload = analyze_run(out, config=RuleConfig(1.0, h2d_share_threshold=0.5))
    mapped = [row for row in payload["steps"] if row["boundary"]]
    assert [row["training_step"] for row in mapped] == [1, 2, 3]
    for row in mapped:
        assert row["measurements"]["host_data_wait_ms"]["status"] == "not_recorded"
        assert row["measurements"]["host_h2d_ms"]["status"] == "recorded"
        assert row["measurements"]["device_attributed_kernel_ms"]["status"] == "recorded"
    loader_wait = [f for f in payload["findings"] if f["rule_id"] == "host_loader_wait"]
    assert {f["status"] for f in loader_wait} == {"unavailable"}


@pytest.mark.distributed
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires >=2 real CUDA devices; CPU DDP is not acceptance",
)
@pytest.mark.skipif(not ddp.dist.is_nccl_available(), reason="NCCL backend is unavailable")
def test_example_under_torchrun_writes_one_analyzable_run_per_rank(tmp_path):
    out = tmp_path / "ddp"
    argv = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2"]
    argv += ["examples/train_custom_loop.py", "--ddp", "--model", "cnn", "--device", "cuda"]
    argv += ["--steps", "6", "--wait", "1", "--warmup", "1", "--active", "1", "--repeat", "2"]
    argv += ["--batch-size", "4", "--out", str(out), "--quiet"]
    _fresh_process(argv)
    for rank in range(2):
        directory = ddp.rank_directory(out, rank)
        run = load_run(directory)
        extra = run.environment["extra"]
        assert extra["rank"] == rank and extra["world_size"] == 2 and extra["local_rank"] == rank
        assert extra["source"] == SOURCE and extra["device_resolved"] == f"cuda:{rank}"
        # The workload describes the caller's module, not the DDP wrapper.
        assert extra["workload"]["module_class"].endswith("Sequential")
        assert extra["workload"]["samples_per_rank"] == 128
        assert len(run.steps) == 6
        assert [m["training_step"] for m in extra["capture"]["step_map"]] == [2, 5]
        assert all(m["rank"] == rank for m in extra["capture"]["step_map"])
        payload = analyze_run(directory, config=RuleConfig(step_variability_ratio=1.0))
        assert payload["rank"] == rank
        mapped = [row for row in payload["steps"] if row["boundary"]]
        assert len(mapped) == 2
        for row in mapped:
            assert row["measurements"]["host_backward_ms"]["status"] == "recorded"
            assert row["measurements"]["device_attributed_kernel_ms"]["status"] == "recorded"
