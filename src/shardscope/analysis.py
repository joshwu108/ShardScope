"""Explicit offline analysis: python -m shardscope.analysis --run DIR.

An explicit --step-map selects a consumer mapping file inside the run.
Otherwise analysis uses the capture mapping recorded in environment.json.
Indices address original event arrays. Rank is supplied explicitly or read
from environment.json.extra.rank.
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

from shardscope.attribution import (
    ATTRIBUTED_KERNEL_METRIC,
    ATTRIBUTION_BASIS,
    attribute_traces,
    is_kernel_event,
    validate_attribution,
)
from shardscope.errors import ArtifactError
from shardscope.parse import (
    DEVICE_METRICS,
    HOST_PHASES,
    StepMapping,
    analyze_traces,
    finite_number,
    identity_step,
    is_device_annotation,
    is_step_boundary,
    measurement,
    parse_trace,
    read_json_bytes,
)
from shardscope.rules import (
    LEGACY_RULES,
    RULE_IDS,
    RuleConfig,
    evaluate_rules,
    parse_threshold,
)

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


def load_capture_step_map(
    capture: dict | None,
    *,
    rank: int,
) -> tuple[StepMapping, ...]:
    if capture is None:
        return ()
    if not isinstance(capture, dict):
        raise ArtifactError("extra.capture must be an object")

    schema = capture.get("step_map_schema")
    if type(schema) is not int or schema != 1:
        raise ArtifactError("unsupported capture step_map_schema")

    steps = capture.get("step_map")
    if not isinstance(steps, list):
        raise ArtifactError("Unsupported step_map type")

    results = []
    for index, step in enumerate(steps):
        location = f"capture.step_map[{index}]"

        if not isinstance(step, dict):
            raise ArtifactError(f"{location}: expected an object")

        step_rank = step.get("rank")
        if type(step_rank) is not int or step_rank != rank:
            raise ArtifactError(f"{location}: rank must match run {rank}")
        trace_file = step.get("trace_file")
        event_index = step.get("step_event_index")
        training_step = step.get("training_step")

        if not _filename(trace_file):
            raise ArtifactError(f"{location}: trace_file must be a local filename")
        if type(event_index) is not int or event_index < 0:
            raise ArtifactError(f"{location}: step_event_index must be nonnegative")
        if type(training_step) is not int or training_step < 0:
            raise ArtifactError(f"{location}: training_step must be nonnegative")
        results.append(
            StepMapping(trace_file=trace_file, event_index=event_index, training_step=training_step)
        )
    return tuple(results)


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
    run = load_run(directory, include_analysis=False)
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

    capture = run.extra.get("capture")
    if step_map_file is not None:
        mapping_source = "file"
        mappings = load_step_map(directory / step_map_file)
    elif capture is not None:
        mapping_source = "capture"
        mappings = load_capture_step_map(capture, rank=rank)
    else:
        mapping_source = "none"
        mappings = ()
    unavailable = {}
    if run.environment.get("cuda_available") is False or run.environment.get("device_count") == 0:
        unavailable = dict.fromkeys(
            (*DEVICE_METRICS, ATTRIBUTED_KERNEL_METRIC), "recorded environment has no CUDA device"
        )
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
    # Kernel attribution runs after host analysis so it can only use validated
    # boundaries; it adds device_attributed_kernel_ms to every step row.
    attribution, attribution_warnings = attribute_traces(
        traces, steps, rank=rank, unavailable=unavailable.get(ATTRIBUTED_KERNEL_METRIC)
    )
    warnings.extend(attribution_warnings)
    warnings.append("thresholds are caller configuration, not hardware-calibrated defaults")
    payload = {
        "schema_version": 1,
        "rank": rank,
        "source": sources,
        "steps": steps,
        "rules": list(RULE_IDS),
        "findings": evaluate_rules(steps, config, rules=RULE_IDS),
        "kernel_attribution": attribution,
        "warnings": warnings,
        "rule_config": config.to_dict() if config else None,
        "mapping_file": step_map_file,
        "mapping_source": mapping_source,
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
    if environment is not None and payload["rank_basis"] == "environment.extra.rank":
        _require(recorded is not None, "rank_basis claims missing environment rank")
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
    # Older sidecars used only explicit mapping files and have no source label.
    mapping_source = payload.get("mapping_source", "file" if mapping is not None else "none")
    _require(mapping_source in ("file", "capture", "none"), "invalid mapping_source")
    if mapping_source == "file":
        _require(mapping is not None, "file mapping_source requires mapping_file provenance")
    else:
        _require(mapping is None, "mapping_file conflicts with mapping_source")
    entries = None
    if mapping_source == "file" and directory is not None:
        entries = load_step_map(directory / mapping)
    elif mapping_source == "capture":
        mapping_environment = environment
        if directory is not None:
            mapping_environment = read_json_bytes(
                _bytes(directory / "environment.json"), source=str(directory / "environment.json")
            )
        if mapping_environment is not None:
            _require(
                isinstance(mapping_environment, dict), "capture mapping requires environment object"
            )
            extra = mapping_environment.get("extra")
            _require(isinstance(extra, dict), "capture mapping requires environment.extra")
            capture = extra.get("capture")
            _require(isinstance(capture, dict), "capture mapping metadata is missing")
            entries = load_capture_step_map(capture, rank=rank)
    mapped = None
    if entries is not None:
        mapped = {}
        targets = set()
        for entry in entries:
            _require(
                type(entry.training_step) is int and entry.training_step >= 0,
                "invalid mapped training step",
            )
            _require(
                type(entry.event_index) is int and entry.event_index >= 0,
                "invalid mapped event index",
            )
            _require(entry.training_step not in mapped, "duplicate mapped training step")
            _require(
                entry.trace_file in source
                and entry.trace_file
                not in ("steps.csv", "environment.json", "summary.json", mapping),
                "step map names a file absent from trace sources",
            )
            target = (entry.trace_file, entry.event_index)
            _require(target not in targets, "same boundary mapped to multiple training steps")
            targets.add(target)
            mapped[entry.training_step] = {
                "file": entry.trace_file,
                "event_index": entry.event_index,
            }
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
    for row in items:
        _require(isinstance(row, dict), "step must be object")
        step = row.get("training_step")
        _require(
            type(step) is int and step >= 0 and step not in seen,
            "invalid or duplicate training_step",
        )
        _require(steps is None or step in steps, f"training_step {step} absent from steps.csv")
        seen.add(step)
        _require("boundary" in row, "step missing boundary field")
        boundary_ref = row.get("boundary")
        _require(
            boundary_ref is None or mapping_source in ("file", "capture"),
            "mapped boundary requires mapping_file provenance or capture mapping provenance",
        )
        boundary = reference(boundary_ref) if boundary_ref is not None else None
        if boundary is not None:
            _require(
                is_step_boundary(boundary),
                "boundary is not complete host step marker"
                + (" (device annotation)" if is_device_annotation(boundary) else ""),
            )
        if mapped is not None:
            _require(
                boundary_ref == mapped.get(step),
                f"training_step {step}: boundary disagrees with explicit step map",
            )
        if boundary is not None and directory is not None:
            identity = identity_step(traces[boundary_ref["file"]], boundary_ref["event_index"])
            _require(
                identity is None or identity == step,
                f"training_step {step}: boundary's identity marker names step {identity}",
            )
        metrics = row.get("measurements")
        _require(isinstance(metrics, dict), "measurements must be object")
        # Sidecars written before kernel attribution existed lack the
        # attributed-kernel metric; they still load, and never gain a value.
        required = set(HOST_PHASES.values()) | set(DEVICE_METRICS)
        _require(
            required <= metrics.keys() <= required | {ATTRIBUTED_KERNEL_METRIC},
            "missing or unsupported measurement fields",
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
                            event["ph"] == "X"
                            and event.get("name") == marker
                            and not is_device_annotation(event),
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
                    # Complete device activity metrics have no validated
                    # attribution implementation; only correlated kernel
                    # ranges are attributed, under their own metric.
                    _require(
                        False, f"{name}: recorded device attribution unsupported by this reader"
                    )
                if name == ATTRIBUTED_KERNEL_METRIC:
                    _require(basis == ATTRIBUTION_BASIS, f"{name}: needs {ATTRIBUTION_BASIS} basis")
                    _require(
                        all(ref["file"] == boundary_ref["file"] for ref in metric["evidence"]),
                        f"{name}: evidence must come from the step's own trace",
                    )
                    for event in events:
                        _require(
                            event is None or is_kernel_event(event),
                            f"{name}: evidence is not a device kernel event",
                        )
            else:
                _require(
                    value is None and basis is None and not events and bool(metric["reason"]),
                    f"{name}: unavailable measurement needs nulls, reason and no evidence",
                )
    _require(steps is None or seen == steps, "analysis must retain all steps.csv identities")
    _require(
        [row["training_step"] for row in items] == sorted(seen),
        "analysis steps must be ordered by training_step",
    )
    if mapped is not None:
        _require(mapped.keys() <= seen, "step map contains training steps absent from analysis")
    config = payload.get("rule_config")
    rule_config = None
    if config is not None:
        _require(isinstance(config, dict) and bool(config), "invalid rule_config")
        rule_config = RuleConfig.from_mapping(config)
    # Sidecars written before the rule registry carry no ``rules`` key and were
    # evaluated with the single legacy rule; they keep loading unchanged.
    rules = payload.get("rules", list(LEGACY_RULES))
    _require(
        isinstance(rules, list)
        and all(isinstance(rule, str) and rule in RULE_IDS for rule in rules)
        and len(set(rules)) == len(rules),
        f"rules must list distinct known rule ids from: {', '.join(RULE_IDS)}",
    )
    findings = payload.get("findings")
    _require(isinstance(findings, list), "findings must be array")
    # This checks persisted comparisons against their recorded inputs; it never
    # estimates a missing measurement or supplies configuration.
    _require(
        findings == evaluate_rules(items, rule_config, rules=rules),
        f"{', '.join(rules)}: findings disagree with measurements/configuration",
    )
    for finding in findings:
        references(finding["evidence"])

    # Kernel attribution relationships: every attributed kernel must resolve,
    # through its launch, to the boundary this sidecar assigns to its step.
    # With a source directory the block is recomputed from the trace bytes.
    boundaries: dict[str, list[tuple[int, int]]] = {}
    for row in items:
        if row["boundary"] is not None:
            boundaries.setdefault(row["boundary"]["file"], []).append(
                (row["training_step"], row["boundary"]["event_index"])
            )
    trace_names = {
        name
        for name in source
        if name not in ("steps.csv", "environment.json", "summary.json", mapping)
    }

    def load_trace(name):
        if directory is None:
            return None
        if name not in traces:
            traces[name] = parse_trace(directory / name)
        return traces[name]

    validate_attribution(
        payload, trace_names=trace_names, boundaries=boundaries, load_trace=load_trace
    )
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
    run = load_run(directory, include_analysis=False)
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
    except (OSError, ValueError, TypeError) as exc:
        raise ArtifactError(f"cannot publish {target}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Share the module interface with a future console subcommand."""
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--rank", type=int)
    parser.add_argument(
        "--step-map", help="consumer map filename within run; overrides embedded capture mapping"
    )
    parser.add_argument(
        "--trace", action="append", help="trace filename within run; repeat for cycles"
    )
    parser.add_argument(
        "--loader-wait-threshold-ms",
        type=float,
        help="enable host_loader_wait; same as --threshold loader_wait_threshold_ms=MS",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        metavar="KEY=VALUE",
        help=(
            "configure one rule threshold; repeat for several. Keys and ranges: "
            "shardscope rules. A rule with no threshold reports unconfigured"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")


def rule_config_from_arguments(args: argparse.Namespace) -> RuleConfig | None:
    """Combine the legacy loader-wait flag with ``--threshold KEY=VALUE`` items.

    Returns ``None`` when nothing was configured, so the sidecar records
    ``rule_config: null`` and every rule reports ``unconfigured``. Supplying the
    same key twice with different values is a mistake, not a precedence rule.
    """
    values: dict[str, int | float] = {}
    items = list(getattr(args, "threshold", None) or [])
    if getattr(args, "loader_wait_threshold_ms", None) is not None:
        values["loader_wait_threshold_ms"] = args.loader_wait_threshold_ms
    for item in items:
        key, value = parse_threshold(item)
        if key in values and values[key] != value:
            raise ArtifactError(
                f"threshold {key} given twice with different values ({values[key]!r} and {value!r})"
            )
        values[key] = value
    return RuleConfig(**values) if values else None


def run_arguments(args: argparse.Namespace) -> int:
    """Execute explicitly requested analysis from validated command arguments."""
    try:
        if (args.run / ANALYSIS_JSON).exists() and not args.overwrite:
            raise ArtifactError(f"{args.run / ANALYSIS_JSON} exists; pass --overwrite")
        config = rule_config_from_arguments(args)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    return run_arguments(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
