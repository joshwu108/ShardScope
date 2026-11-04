"""Explicit capture schedules; no profiler or accelerator execution."""

import pytest

from shardscope.config import CaptureConfig, from_mapping
from shardscope.errors import ConfigError


def settings(**changes):
    return dict(wait=1, warmup=1, active=2, repeat=2, activities=["cpu"], **changes)


def test_resolved_config_roundtrip():
    config = from_mapping({"steps": 8, "capture": settings()})
    assert config.capture.required_steps == 8
    assert from_mapping(config.to_dict()) == config
    assert not config.capture.with_stack
    assert not config.capture.profile_memory


@pytest.mark.parametrize("key", ["wait", "warmup", "active", "repeat", "activities"])
def test_explicit_required_fields(key):
    value = settings()
    del value[key]
    with pytest.raises(ConfigError):
        CaptureConfig.from_mapping(value)


@pytest.mark.parametrize(
    "key,value",
    [
        ("wait", -1),
        ("warmup", True),
        ("active", 0),
        ("repeat", 0),
        ("activities", []),
        ("activities", ["cuda"]),
        ("activities", ["cpu", "cpu"]),
        ("activities", ["mps"]),
        ("with_stack", 1),
        ("other", 1),
    ],
)
def test_invalid_settings(key, value):
    payload = settings()
    payload[key] = value
    with pytest.raises(ConfigError):
        CaptureConfig.from_mapping(payload)


def test_schedule_must_fit_all_cycles():
    with pytest.raises(ConfigError, match="requires 8.*available 7"):
        from_mapping({"steps": 7, "capture": settings()})
    assert from_mapping({"steps": 9, "capture": settings()}).capture.repeat == 2
