"""Shared test setup.

``examples/`` holds runnable scripts rather than an installed package, so it
is put on ``sys.path`` here. The scripts are written to do no work at import
time (see examples/train_single.py) precisely so tests can drive a handful of
steps in-process instead of spawning subprocesses.

The ``make_run`` fixture below builds real run directories on disk rather
than mocking the filesystem: what the artifact tests exercise *is* file
handling, and a mock would agree with whatever the loader happens to do.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"

if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))


#: Header + three rows in exactly the shape examples/train_single.py writes.
GOOD_STEPS_CSV = (
    "step,loss,step_time_ms,data_residency,synchronized\n"
    "0,2.3827712535858154,17.921874998137355,host,True\n"
    "1,2.294814109802246,9.41929197870195,host,True\n"
    "2,2.2615203857421875,8.802208000794053,host,True\n"
)

#: A RunEnvironment dump plus the ``extra`` block the fixture records.
GOOD_ENVIRONMENT = {
    "cuda_available": False,
    "cuda_version": None,
    "cudnn_version": None,
    "device_count": 0,
    "device_names": [],
    "extra": {
        "batch_size": 64,
        "config_file": None,
        "data_residency": "host",
        "device_requested": "cpu",
        "device_resolved": "cpu",
        "learning_rate": 0.01,
        "profiled": False,
        "samples": 2048,
        "seed": 0,
        "source": "examples/train_single.py",
        "steps": 3,
        "synchronized": True,
    },
    "machine": "arm64",
    "mps_available": True,
    "nccl_version": None,
    "platform": "macOS-27.0-arm64-arm-64bit",
    "python_version": "3.12.14",
    "torch_version": "2.14.1",
}

GOOD_SUMMARY = {
    "max_step_ms": 17.921874998137355,
    "median_step_ms": 9.41929197870195,
    "min_step_ms": 8.802208000794053,
    "p95_step_ms": 17.921874998137355,
    "steps": 3.0,
}


def _dump(payload) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


@pytest.fixture
def make_run(tmp_path):
    """Build a run directory on disk and return its path.

    ``steps_csv``/``environment``/``summary`` default to a valid artifact
    triple. Pass ``None`` to leave that artifact out, or a string to write
    something malformed verbatim.
    """

    def build(
        name="run",
        *,
        steps_csv=GOOD_STEPS_CSV,
        environment=GOOD_ENVIRONMENT,
        summary=GOOD_SUMMARY,
        extra_files=(),
    ):
        run_dir = tmp_path / name
        run_dir.mkdir(parents=True, exist_ok=True)
        if steps_csv is not None:
            (run_dir / "steps.csv").write_text(steps_csv)
        if environment is not None:
            (run_dir / "environment.json").write_text(_dump(environment))
        if summary is not None:
            (run_dir / "summary.json").write_text(_dump(summary))
        for filename in extra_files:
            (run_dir / filename).write_text("{}\n")
        return run_dir

    return build
