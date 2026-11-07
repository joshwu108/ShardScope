"""Real-GPU acceptance for kernel attribution: skipped without a CUDA device.

The synthetic coverage lives in ``tests/test_kernel_attribution.py``. This test
captures a short real run, analyzes it, and checks the persisted attribution
against the trace bytes. It asserts structure and disclosure, not timing
values: what the device took is whatever it took.
"""

from __future__ import annotations

import json

import pytest
import torch

from shardscope import capture
from shardscope.analysis import analyze_run, write_analysis
from shardscope.artifacts import load_run
from shardscope.attribution import ATTRIBUTED_KERNEL_METRIC, is_kernel_event, is_supported_launch
from shardscope.config import CaptureConfig, RunConfig
from shardscope.report import render_html
from shardscope.validate import validate_run


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_real_capture_attributes_kernels_and_discloses_coverage(tmp_path):
    cfg = RunConfig(
        steps=5,
        batch_size=2,
        samples=8,
        device="cuda",
        capture=CaptureConfig(1, 1, 2, 1, ("cpu", "cuda")),
    )
    capture.profile_run(cfg, tmp_path / "cuda")
    payload = analyze_run(tmp_path / "cuda")
    write_analysis(tmp_path / "cuda", payload)
    run = load_run(tmp_path / "cuda")
    assert validate_run(tmp_path / "cuda").ok

    (name,) = run.environment["extra"]["capture"]["trace_files"]
    events = json.loads((tmp_path / "cuda" / name).read_text())["traceEvents"]
    result = payload["kernel_attribution"]["traces"][name]
    kernels = [i for i, e in enumerate(events) if is_kernel_event(e)]
    assert result["kernel_events"]["total"] == len(kernels)
    assert result["kernel_events"]["attributed"] > 0
    assert set(kernels) == {
        r["kernel_event_index"]
        for r in result["attributed"] + result["unmatched"] + result["ambiguous"]
    }
    for record in result["attributed"]:
        kernel = events[record["kernel_event_index"]]
        launch = events[record["launch_event_index"]]
        boundary = events[record["boundary_event_index"]]
        assert is_kernel_event(kernel) and is_supported_launch(launch)
        assert (
            kernel["args"]["correlation"] == launch["args"]["correlation"] == record["correlation"]
        )
        assert (launch["pid"], launch["tid"]) == (boundary["pid"], boundary["tid"])
        assert boundary["ts"] <= launch["ts"]
        assert launch["ts"] + launch["dur"] <= boundary["ts"] + boundary["dur"]

    mapped = [row for row in payload["steps"] if row["boundary"] is not None]
    assert mapped
    for row in mapped:
        metric = row["measurements"][ATTRIBUTED_KERNEL_METRIC]
        assert metric["status"] in ("recorded", "not_recorded")
        if metric["status"] == "recorded":
            assert metric["value_ms"] > 0
            assert metric["coverage"]["completeness"] in ("partial", "complete")
            assert all(is_kernel_event(events[ref["event_index"]]) for ref in metric["evidence"])
        if result["kernel_events"]["unmatched"]:
            assert "unattributed" in metric["reason"]
    rendered = render_html(run)
    assert "<h2>Kernel attribution</h2>" in rendered
    assert "cuda_runtime:cudaLaunchKernel" in rendered
