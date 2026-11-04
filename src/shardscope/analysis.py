"""Explicit offline analysis: python -m shardscope.analysis --run DIR.

The producer-map adapter is pending. --step-map names a consumer mapping file
inside the run: {"mappings": [{"trace_file": "trace-00000.json",
"event_index": 0, "training_step": 12}]}. Indices address original arrays.
Rank is supplied explicitly or read from environment.json.extra.rank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from shardscope.errors import ArtifactError
from shardscope.parse import (
    DEVICE_METRICS,
    HOST_PHASES,
    StepMapping,
    analyze_traces,
    finite_number,
    measurement,
    parse_trace,
    read_json_bytes,
)
from shardscope.rules import RULE_ID, RuleConfig, evaluate_rules

ANALYSIS_JSON = "analysis.json"


def _filename(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and Path(value).name == value
        and value not in (".", "..")
    )


def _bytes(path: Path) -> bytes:
    try:
        if not path.is_file():
            raise ArtifactError(f"{path}: expected a regular source file")
        return path.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"cannot read {path}: {exc}") from exc


def _digest(path: Path) -> str:
    return hashlib.sha256(_bytes(path)).hexdigest()


def load_step_map(path: Path) -> tuple[StepMapping, ...]:
    """Read the consumer map, without guessing a producer's mapping layout."""
    payload = read_json_bytes(_bytes(path), source=str(path))
    if (
        not isinstance(payload, dict)
        or set(payload) != {"mappings"}
        or not isinstance(payload["mappings"], list)
    ):
        raise ArtifactError(f"{path}: expected object with mappings array")
    result = []
    for index, item in enumerate(payload["mappings"]):
        if not isinstance(item, dict) or set(item) != {
            "trace_file",
            "event_index",
            "training_step",
        }:
            raise ArtifactError(
                f"{path} mappings[{index}]: expected trace_file, event_index, training_step"
            )
        if not _filename(item["trace_file"]):
            raise ArtifactError(f"{path} mappings[{index}]: trace_file must be a local filename")
        result.append(StepMapping(**item))
    return tuple(result)


def analyze_run(
    path: str | Path,
    *,
    rank: int | None = None,
    step_map_file: str | None = None,
    trace_files: tuple[str, ...] | None = None,
    config: RuleConfig | None = None,
) -> dict:
    """Build a sidecar in memory; this function never writes source artifacts."""
    from shardscope.artifacts import load_run

    directory = Path(path)
    if trace_files is None:
        try:
            trace_files = tuple(
                sorted(
                    p.name
                    for p in directory.iterdir()
                    if p.name.startswith("trace")
                    and (p.name.endswith(".json") or p.name.endswith(".json.gz"))
                )
            )
        except OSError as exc:
            raise ArtifactError(f"cannot list run {directory}: {exc}") from exc
    names = ["steps.csv", "environment.json", *trace_files]
    if (directory / "summary.json").exists():
        names.append("summary.json")
    if step_map_file is not None:
        names.append(step_map_file)
    if any(not _filename(name) or name == ANALYSIS_JSON for name in names):
        raise ArtifactError("analysis sources must be local filenames, excluding analysis.json")
    if len(set(trace_files)) != len(trace_files):
        raise ArtifactError("duplicate trace filenames")
    sources = {name: _digest(directory / name) for name in sorted(set(names))}
    run = load_run(directory)
    recorded_rank = run.extra.get("rank")
    if rank is None:
        rank = recorded_rank
    if type(rank) is not int or rank < 0:
        raise ArtifactError(
            "analysis requires --rank or explicit nonnegative environment.extra.rank"
        )
    if recorded_rank is not None and (type(recorded_rank) is not int or recorded_rank != rank):
        raise ArtifactError("explicit rank disagrees with environment.extra.rank")
    traces = tuple(parse_trace(directory / name) for name in trace_files)
    mappings = load_step_map(directory / step_map_file) if step_map_file else ()
    unavailable = {}
    if run.environment.get("cuda_available") is False or run.environment.get("device_count") == 0:
        unavailable = dict.fromkeys(DEVICE_METRICS, "recorded environment has no CUDA device")
    if all(row.data_residency == "device" for row in run.steps):
        unavailable["device_h2d_ms"] = "recorded data_residency=device; no per-step batch transfer"
    count = run.environment.get("device_count")
    if type(count) is int and count < 2:
        unavailable["device_collective_ms"] = "recorded environment has fewer than two CUDA devices"
    steps, warnings = analyze_traces(traces, rank=rank, step_map=mappings, unavailable=unavailable)
    valid_steps = {row.step for row in run.steps}
    if any(item["training_step"] not in valid_steps for item in steps):
        raise ArtifactError("step map references training step absent from steps.csv")
    mapped_steps = {item["training_step"] for item in steps}
    for step in sorted(valid_steps - mapped_steps):
        metrics = {
            metric: measurement(None, reason="no explicit mapped step boundary")
            for metric in (*HOST_PHASES.values(), *DEVICE_METRICS)
        }
        for metric, reason in unavailable.items():
            metrics[metric] = measurement(None, reason=reason, status="not_observed")
        steps.append({"training_step": step, "boundary": None, "measurements": metrics})
    steps.sort(key=lambda item: item["training_step"])
    warnings.extend(
        [
            "thresholds are caller configuration, not hardware-calibrated defaults",
            "producer step-map JSON integration pending; consumer map used only when explicit",
        ]
    )
    payload = {
        "schema_version": 1,
        "rank": rank,
        "source": sources,
        "steps": steps,
        "findings": evaluate_rules(steps, config),
        "warnings": warnings,
        "rule_config": config.to_dict() if config else None,
        "mapping_file": step_map_file,
        "rank_basis": "environment.extra.rank"
        if recorded_rank is not None
        else "explicit_argument",
    }
    validate_analysis(payload, directory=directory, environment=run.environment, steps=valid_steps)
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(f"analysis.json: {message}")


def validate_analysis(
    payload: Any,
    *,
    directory: Path | None = None,
    environment: dict | None = None,
    steps: set[int] | None = None,
) -> dict:
    """Check schema and, with a directory, digests and evidence targets.

    No metrics or statistics are derived here. A serialized report can be
    checked structurally without its source directory; rendering requires the
    directory for freshness and evidence verification.
    """
    _require(isinstance(payload, dict), "expected object")
    _require(
        type(payload.get("schema_version")) is int and payload["schema_version"] == 1,
        "unsupported schema_version; expected 1",
    )
    rank = payload.get("rank")
    _require(type(rank) is int and rank >= 0, "rank must be nonnegative integer")
    if environment is not None:
        extra = environment.get("extra", {})
        recorded = extra.get("rank") if isinstance(extra, dict) else None
        if recorded is not None:
            _require(type(recorded) is int and recorded == rank, "rank disagrees with environment")
    _require(
        payload.get("rank_basis") in ("environment.extra.rank", "explicit_argument"),
        "missing or unsupported rank_basis",
    )
    source = payload.get("source")
    _require(
        isinstance(source, dict) and {"steps.csv", "environment.json"} <= source.keys(),
        "source must include steps.csv and environment.json digests",
    )
    for name, digest in source.items():
        _require(_filename(name) and name != ANALYSIS_JSON, "source needs local artifact filenames")
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest),
            f"invalid SHA-256 for {name}",
        )
        if directory is not None:
            _require(
                _digest(directory / name) == digest,
                f"stale source {name}; re-run explicit analysis",
            )
    warnings = payload.get("warnings")
    _require(
        isinstance(warnings, list) and all(isinstance(w, str) for w in warnings),
        "warnings must be string array",
    )
    mapping = payload.get("mapping_file")
    _require(
        mapping is None or (isinstance(mapping, str) and mapping in source), "invalid mapping_file"
    )
    traces = {}

    def reference(ref):
        _require(isinstance(ref, dict), "evidence reference must be object")
        name, index = ref.get("file"), ref.get("event_index")
        _require(
            isinstance(name, str)
            and name in source
            and name not in ("steps.csv", "environment.json", "summary.json", mapping),
            "evidence file absent from trace sources",
        )
        _require(type(index) is int and index >= 0, "invalid evidence event_index")
        if directory is None:
            return None
        if name not in traces:
            traces[name] = parse_trace(directory / name)
        trace = traces[name]
        trace.reference(index)
        return trace.events[index]

    def references(refs):
        _require(isinstance(refs, list), "evidence must be array")
        return [reference(ref) for ref in refs]

    items = payload.get("steps")
    _require(isinstance(items, list), "steps must be array")
    seen = set()
    by_step = {}
    for row in items:
        _require(isinstance(row, dict), "step must be object")
        step = row.get("training_step")
        _require(
            type(step) is int and step >= 0 and step not in seen,
            "invalid or duplicate training_step",
        )
        _require(steps is None or step in steps, f"training_step {step} absent from steps.csv")
        seen.add(step)
        by_step[step] = row
        boundary_ref = row.get("boundary")
        boundary = reference(boundary_ref) if boundary_ref is not None else None
        if boundary is not None:
            _require(
                boundary["ph"] == "X"
                and (
                    boundary.get("name") == "shardscope/step"
                    or boundary.get("name", "").startswith("ProfilerStep#")
                ),
                "boundary is not complete step marker",
            )
        metrics = row.get("measurements")
        _require(isinstance(metrics, dict), "measurements must be object")
        _require(
            set(HOST_PHASES.values()) | set(DEVICE_METRICS) <= metrics.keys(),
            "missing measurement fields",
        )
        for name, metric in metrics.items():
            _require(isinstance(metric, dict), f"{name}: measurement must be object")
            _require(
                {"value_ms", "status", "basis", "reason", "evidence", "scope"} <= metric.keys(),
                f"{name}: incomplete measurement",
            )
            status, value, basis = metric["status"], metric["value_ms"], metric["basis"]
            _require(
                status in ("recorded", "not_recorded", "not_observed"), f"{name}: invalid status"
            )
            _require(
                isinstance(metric["reason"], str) and isinstance(metric["scope"], str),
                f"{name}: reason and scope must be strings",
            )
            events = references(metric["evidence"])
            if status == "recorded":
                _require(
                    finite_number(value) and value >= 0,
                    f"{name}: value_ms must be finite nonnegative",
                )
                _require(
                    bool(events) and boundary_ref is not None,
                    f"{name}: recorded metric needs evidence and boundary",
                )
                _require(
                    basis in ("host_range", "device_interval_union"),
                    f"{name}: invalid measurement basis",
                )
                if name in HOST_PHASES.values():
                    _require(basis == "host_range", f"{name}: needs host_range basis")
                    marker = next(k for k, v in HOST_PHASES.items() if v == name)
                    for ref, event in zip(metric["evidence"], events, strict=True):
                        if event is None:
                            continue
                        _require(
                            event["ph"] == "X" and event.get("name") == marker,
                            f"{name}: evidence is not the expected complete host marker",
                        )
                        _require(
                            ref["file"] == boundary_ref["file"]
                            and event["pid"] == boundary["pid"]
                            and event["tid"] == boundary["tid"]
                            and boundary["ts"] <= event["ts"]
                            and event["ts"] + event["dur"] <= boundary["ts"] + boundary["dur"],
                            f"{name}: evidence is outside its host boundary",
                        )
                if name in DEVICE_METRICS:
                    # No validated device attribution implementation exists yet.
                    _require(
                        False, f"{name}: recorded device attribution unsupported by this reader"
                    )
            else:
                _require(
                    value is None and basis is None and not events and bool(metric["reason"]),
                    f"{name}: unavailable measurement needs nulls, reason and no evidence",
                )
    _require(steps is None or seen == steps, "analysis must retain all steps.csv identities")
    config = payload.get("rule_config")
    if config is not None:
        _require(
            isinstance(config, dict) and set(config) == {"loader_wait_threshold_ms"},
            "invalid rule_config",
        )
        RuleConfig(**config)
    findings = payload.get("findings")
    _require(isinstance(findings, list), "findings must be array")
    # This checks persisted comparisons against their recorded inputs; it never
    # estimates a missing measurement or supplies configuration.
    _require(
        findings == evaluate_rules(items, RuleConfig(**config) if config else None),
        f"{RULE_ID}: findings disagree with measurements/configuration",
    )
    for finding in findings:
        references(finding["evidence"])
    return payload


def write_analysis(path: str | Path, payload: dict, *, overwrite: bool = False) -> Path:
    """Publish a complete sidecar atomically; default publication never clobbers.

    Source digests are rechecked before publication. This is a snapshot check,
    not a filesystem transaction with a concurrently running capture producer.
    """
    from shardscope.artifacts import load_run

    directory = Path(path)
    target = directory / ANALYSIS_JSON
    if target.is_symlink():
        raise ArtifactError(f"refusing analysis output symlink {target}")
    if target.exists() and not overwrite:
        raise ArtifactError(f"{target} exists; pass --overwrite to replace explicit analysis")
    run = load_run(directory)
    validate_analysis(
        payload,
        directory=directory,
        environment=run.environment,
        steps={row.step for row in run.steps},
    )
    if target.exists():
        for name in payload["source"]:
            if target.samefile(directory / name):
                raise ArtifactError(f"refusing analysis/source collision with {name}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=".analysis-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
    except OSError as exc:
        raise ArtifactError(f"cannot publish {target}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--rank", type=int)
    parser.add_argument(
        "--step-map", help="consumer map filename within run; producer adapter pending"
    )
    parser.add_argument(
        "--trace", action="append", help="trace filename within run; repeat for cycles"
    )
    parser.add_argument("--loader-wait-threshold-ms", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if (args.run / ANALYSIS_JSON).exists() and not args.overwrite:
            raise ArtifactError(f"{args.run / ANALYSIS_JSON} exists; pass --overwrite")
        config = (
            RuleConfig(args.loader_wait_threshold_ms)
            if args.loader_wait_threshold_ms is not None
            else None
        )
        payload = analyze_run(
            args.run,
            rank=args.rank,
            step_map_file=args.step_map,
            trace_files=tuple(args.trace) if args.trace else None,
            config=config,
        )
        print(write_analysis(args.run, payload, overwrite=args.overwrite))
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
