"""Tests for the reproducibility snapshot."""

from __future__ import annotations

import json

from shardscope.env import NOT_OBSERVED, RunEnvironment, capture_environment


def _env(**overrides) -> RunEnvironment:
    base = {
        "python_version": "3.12.9",
        "platform": "macOS-15",
        "machine": "arm64",
        "torch_version": "2.6.0",
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_version": None,
        "nccl_version": None,
        "device_count": 0,
        "device_names": (),
        "mps_available": False,
    }
    return RunEnvironment(**{**base, **overrides})


def test_capture_environment_reports_this_machine():
    env = capture_environment()
    assert env.python_version
    assert env.machine
    assert env.device_count >= 0


def test_accelerator_prefers_cuda_then_mps_then_cpu():
    assert _env(cuda_available=True, device_count=1).accelerator == "cuda"
    assert _env(mps_available=True).accelerator == "mps"
    assert _env().accelerator == "cpu"


def test_cuda_available_without_devices_is_not_cuda():
    # A driver that reports availability but exposes no device must not be
    # treated as a usable accelerator.
    assert _env(cuda_available=True, device_count=0).accelerator == "cpu"


def test_supports_distributed_requires_two_cuda_devices():
    assert not _env().supports_distributed()
    assert not _env(mps_available=True).supports_distributed()
    assert not _env(cuda_available=True, device_count=1).supports_distributed()
    assert _env(cuda_available=True, device_count=2).supports_distributed()


def test_with_extra_does_not_mutate_the_original():
    original = _env()
    enriched = original.with_extra(seed=1234)

    assert original.extra == {}
    assert enriched.extra == {"seed": 1234}
    assert enriched is not original


def test_to_json_round_trips():
    payload = json.loads(_env(device_names=("A100",)).to_json())
    assert payload["device_names"] == ["A100"]
    assert payload["torch_version"] == "2.6.0"


def test_not_observed_is_a_sentinel_string_not_a_number():
    # Unobservable quantities must remain distinct from measured zero.
    assert isinstance(NOT_OBSERVED, str)
    assert NOT_OBSERVED != 0
