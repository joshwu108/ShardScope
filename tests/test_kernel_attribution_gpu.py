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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_real_capture_attributes_batch_copies_and_classifies_no_collective(tmp_path):
    """Single-GPU acceptance for transfer attribution and kernel classes.

    The fixture copies each host batch inside ``shardscope/h2d`` on the main
    thread, so at least one ``gpu_memcpy`` per mapped step must resolve to the
    step through its ``cudaMemcpyAsync`` call. With one device and no process
    group, no kernel may be classified as a collective.
    """
    from shardscope.attribution import (
        ATTRIBUTED_TRANSFER_METRIC,
        TRANSFER_OVERLAP_METRIC,
        is_supported_copy,
        is_transfer_event,
    )
    from shardscope.rules import RuleConfig

    cfg = RunConfig(
        steps=5,
        batch_size=2,
        samples=8,
        device="cuda",
        capture=CaptureConfig(1, 1, 2, 1, ("cpu", "cuda")),
    )
    capture.profile_run(cfg, tmp_path / "cuda")
    payload = analyze_run(tmp_path / "cuda", config=RuleConfig(transfer_exposed_share=0.0))
    write_analysis(tmp_path / "cuda", payload)
    run = load_run(tmp_path / "cuda")
    assert validate_run(tmp_path / "cuda").ok

    (name,) = run.environment["extra"]["capture"]["trace_files"]
    events = json.loads((tmp_path / "cuda" / name).read_text())["traceEvents"]
    result = payload["kernel_attribution"]["traces"][name]
    transfers = [i for i, e in enumerate(events) if is_transfer_event(e)]
    assert result["transfer_events"]["total"] == len(transfers)
    assert result["transfer_events"]["attributed"] > 0
    assert set(transfers) == {
        r["transfer_event_index"]
        for r in result["transfers_attributed"]
        + result["transfers_unmatched"]
        + result["transfers_ambiguous"]
    }
    for record in result["transfers_attributed"]:
        copy_event = events[record["transfer_event_index"]]
        call = events[record["copy_event_index"]]
        boundary = events[record["boundary_event_index"]]
        assert is_transfer_event(copy_event) and is_supported_copy(call)
        assert copy_event["args"]["correlation"] == call["args"]["correlation"]
        assert (call["pid"], call["tid"]) == (boundary["pid"], boundary["tid"])
        assert (
            boundary["ts"]
            <= call["ts"]
            <= call["ts"] + call["dur"]
            <= boundary["ts"] + boundary["dur"]
        )
        assert record["direction"] in ("h2d", "d2h", "other", "unknown")
    assert any(r["direction"] == "h2d" for r in result["transfers_attributed"])
    # Every kernel has a class and none is a collective on one device.
    classes = {r["class"] for r in result["attributed"] + result["unmatched"] + result["ambiguous"]}
    assert classes == {"compute"}
    for entry in result["device_activity"]:
        assert entry["collective_kernel_events"] == 0
        assert entry["collective_kernel_ms"] is None
        assert entry["compute_kernel_events"] > 0 and entry["compute_kernel_ms"] > 0

    mapped = [row for row in payload["steps"] if row["boundary"] is not None]
    assert mapped
    recorded = 0
    for row in mapped:
        transfer = row["measurements"][ATTRIBUTED_TRANSFER_METRIC]
        overlap = row["measurements"][TRANSFER_OVERLAP_METRIC]
        assert transfer["status"] in ("recorded", "not_recorded")
        if transfer["status"] == "recorded":
            recorded += 1
            assert transfer["value_ms"] > 0 and transfer["basis"] == "device_interval_union"
            assert all(
                is_transfer_event(events[ref["event_index"]]) for ref in transfer["evidence"]
            )
            assert transfer["coverage"]["directions"]["h2d"]["events"] > 0
            # Compute kernels were collected, so the overlap is defined.
            assert overlap["status"] == "recorded"
            assert 0 <= overlap["value_ms"] <= transfer["value_ms"] + 1e-9
        else:
            assert overlap["status"] == "not_recorded"
    assert recorded > 0, "no mapped step recorded an attributed transfer"
    findings = [f for f in payload["findings"] if f["rule_id"] == "device_transfer_overlap"]
    assert {
        f["status"] for f in findings if f["steps"][0] in {r["training_step"] for r in mapped}
    } <= {
        "possible_exposed_transfer",
        "unavailable",
    }
    assert all(
        0 <= f["comparison"]["value"] <= 1 for f in findings if f["comparison"]["value"] is not None
    )
    rendered = render_html(run)
    assert "Device transfer attribution" in rendered
    assert "cuda_runtime:cudaMemcpyAsync" in rendered
