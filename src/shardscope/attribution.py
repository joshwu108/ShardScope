"""Kernel-to-step attribution through host CUDA kernel-launch correlation IDs.

Kineto records a host-side ``cuda_runtime`` range for each CUDA API call and a
device-side ``kernel`` range for each kernel execution; the two share
``args.correlation``. A kernel is attributed to a training step only when:

1. its correlation ID resolves to exactly one host CUDA API event in the same
   trace (IDs are looked up per trace; they are never assumed globally unique),
2. that host event is a supported kernel-launch form (``cudaLaunchKernel`` or
   ``cudaLaunchKernelExC``, the forms observed in real captures), and
3. the launch range lies entirely inside exactly one explicitly mapped
   ``shardscope/step`` boundary on the same host process and thread.

Nothing is attributed by GPU timestamp containment: a kernel that runs inside a
step's wall-clock window but whose launch cannot be resolved stays unmatched.
Launches on a host thread that carries no mapped boundary (the autograd engine
thread, for instance) are reported, not guessed at, so attribution is usually
incomplete and every result says so.

"Attributed kernel time" is the per-device interval union of the kernel ranges
attributed to one step within one trace. It is not utilization, not total step
latency and not total device busy time; copies, memsets, collectives and
``gpu_user_annotation`` mirrors are not kernels and are never counted.
"""

from __future__ import annotations

from typing import Any

from shardscope.errors import ArtifactError
from shardscope.parse import (
    Trace,
    contains_range,
    finite_number,
    interval_union_us,
    measurement,
    summarize_indices,
)

__all__ = [
    "ATTRIBUTED_KERNEL_METRIC",
    "ATTRIBUTION_BASIS",
    "ATTRIBUTION_SCHEMA",
    "HOST_API_CATEGORIES",
    "KERNEL_CATEGORY",
    "SUPPORTED_LAUNCHES",
    "attribute_trace",
    "attribute_traces",
    "classify_kernels",
    "correlation_id",
    "is_host_api_event",
    "is_kernel_event",
    "is_supported_launch",
    "step_measurement",
    "validate_attribution",
]

#: Per-step measurement carrying the attributed kernel interval union.
ATTRIBUTED_KERNEL_METRIC = "device_attributed_kernel_ms"
ATTRIBUTION_BASIS = "device_interval_union"
ATTRIBUTION_SCHEMA = 1
METHOD = "host_launch_correlation"

#: Kineto categories of host-side CUDA API ranges that carry correlation IDs.
HOST_API_CATEGORIES = ("cuda_runtime", "cuda_driver")
#: The only launch forms attributed; (category, name) pairs seen in real traces.
SUPPORTED_LAUNCHES = (
    ("cuda_runtime", "cudaLaunchKernel"),
    ("cuda_runtime", "cudaLaunchKernelExC"),
)
#: Kineto's category for device kernel execution; copies, memsets and
#: gpu_user_annotation mirrors have other categories and are not kernels.
KERNEL_CATEGORY = "kernel"

SCOPE_STATEMENT = (
    "attributed kernel time: per-device interval union of kernel ranges tied to a "
    "training step through host kernel-launch correlation ids within one trace; not "
    "utilization, not total step latency, not total device busy time; copies, memsets, "
    "collectives and gpu_user_annotation events are not kernels and are not counted"
)

LAUNCH_KINDS = (
    "attributed",
    "outside_mapped_step",
    "ambiguous_boundary",
    "duplicate_correlation",
    "missing_correlation",
)
UNMATCHED_KINDS = (
    "missing_correlation",
    "no_host_api_event",
    "unsupported_host_event",
    "launch_outside_mapped_step",
    "missing_device",
)
AMBIGUOUS_KINDS = ("conflicting_host_events", "launch_in_several_boundaries")


def correlation_id(event: dict[str, Any]) -> int | None:
    """``args.correlation`` as a nonnegative integer, or ``None`` when absent/invalid."""
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    value = args.get("correlation")
    return value if type(value) is int and value >= 0 else None


def _device_id(event: dict[str, Any]) -> int | None:
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    value = args.get("device")
    return value if type(value) is int and value >= 0 else None


def _stream_id(event: dict[str, Any]) -> Any:
    args = event.get("args")
    return args.get("stream") if isinstance(args, dict) else None


def is_host_api_event(event: dict[str, Any]) -> bool:
    """A complete host-side CUDA runtime/driver API range, by category only."""
    return event.get("ph") == "X" and event.get("cat") in HOST_API_CATEGORIES


def is_supported_launch(event: dict[str, Any]) -> bool:
    return is_host_api_event(event) and (event.get("cat"), event.get("name")) in SUPPORTED_LAUNCHES


def is_kernel_event(event: dict[str, Any]) -> bool:
    """A complete device kernel range, by category only."""
    return event.get("ph") == "X" and event.get("cat") == KERNEL_CATEGORY


def _thread(event: dict[str, Any]) -> str:
    return f"pid={event.get('pid')}; tid={event.get('tid')}"


def classify_kernels(trace: Trace, boundaries: list[tuple[int, int]]) -> dict[str, Any]:
    """Resolve every kernel and launch in one trace without timing anything.

    ``boundaries`` holds ``(training_step, boundary_event_index)`` pairs that
    the caller has already validated against the explicit step map. The result
    is plain JSON data (lists, dicts, ints, strings) so a persisted copy
    compares equal to a recomputation; validation re-derives exactly this
    classification and never re-times a kernel.
    """
    events = trace.events
    by_step = dict(boundaries)
    if len(by_step) != len(boundaries):
        raise ArtifactError(f"{trace.path.name}: duplicate training step among boundaries")
    boundary_events = [(step, index, events[index]) for step, index in boundaries]

    # 1. Correlation lookup over every host CUDA API range in this trace.
    by_correlation: dict[int, list[int]] = {}
    for index, event in enumerate(events):
        if is_host_api_event(event):
            value = correlation_id(event)
            if value is not None:
                by_correlation.setdefault(value, []).append(index)

    # 2. Classify each supported launch against the mapped boundaries.
    launch_kind: dict[int, tuple[str, Any]] = {}
    launch_lists: dict[str, list[int]] = {kind: [] for kind in LAUNCH_KINDS}
    lookup: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if not is_supported_launch(event):
            continue
        value = correlation_id(event)
        if value is None:
            kind: tuple[str, Any] = ("missing_correlation", None)
        elif len(by_correlation[value]) > 1:
            kind = ("duplicate_correlation", by_correlation[value])
        else:
            owners = [
                (step, bindex)
                for step, bindex, boundary in boundary_events
                if contains_range(boundary, event)
            ]
            if not owners:
                kind = ("outside_mapped_step", None)
            elif len(owners) > 1:
                kind = ("ambiguous_boundary", sorted(step for step, _ in owners))
            else:
                kind = ("attributed", owners[0])
                step, bindex = owners[0]
                lookup.append(
                    {
                        "correlation": value,
                        "training_step": step,
                        "launch_event_index": index,
                        "boundary_event_index": bindex,
                    }
                )
        launch_kind[index] = kind
        launch_lists[kind[0]].append(index)

    # 3. Resolve every kernel through the lookup; never through timestamps.
    attributed: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if not is_kernel_event(event):
            continue
        value = correlation_id(event)
        record = {"kernel_event_index": index, "correlation": value}
        if value is None:
            unmatched.append(
                {
                    **record,
                    "kind": "missing_correlation",
                    "reason": "kernel event has no "
                    "integer args.correlation; it cannot be tied to a host launch",
                }
            )
            continue
        hosts = by_correlation.get(value, [])
        if not hosts:
            unmatched.append(
                {
                    **record,
                    "kind": "no_host_api_event",
                    "reason": f"no host CUDA API event in this trace carries correlation {value}",
                }
            )
            continue
        if len(hosts) > 1:
            ambiguous.append(
                {
                    **record,
                    "kind": "conflicting_host_events",
                    "host_event_indices": list(hosts),
                    "reason": f"correlation {value} appears on several host CUDA API events "
                    f"({summarize_indices(hosts)}); no candidate is chosen",
                }
            )
            continue
        host_index = hosts[0]
        host = events[host_index]
        if not is_supported_launch(host):
            unmatched.append(
                {
                    **record,
                    "host_event_index": host_index,
                    "kind": "unsupported_host_event",
                    "reason": f"correlated host event[{host_index}] is "
                    f"{host.get('cat')}:{host.get('name')}, not a supported kernel-launch form",
                }
            )
            continue
        kind, detail = launch_kind[host_index]
        if kind == "outside_mapped_step":
            unmatched.append(
                {
                    **record,
                    "launch_event_index": host_index,
                    "kind": "launch_outside_mapped_step",
                    "reason": f"launch event[{host_index}] ({_thread(host)}) is not inside any "
                    "mapped step boundary on its host thread",
                }
            )
            continue
        if kind == "ambiguous_boundary":
            ambiguous.append(
                {
                    **record,
                    "launch_event_index": host_index,
                    "kind": "launch_in_several_boundaries",
                    "candidate_training_steps": detail,
                    "reason": f"launch event[{host_index}] lies inside several mapped step "
                    f"boundaries (training steps {detail}); no step is chosen",
                }
            )
            continue
        if kind != "attributed":  # pragma: no cover - duplicates were caught above
            raise ArtifactError(f"{trace.path.name}: unexpected launch outcome {kind}")
        step, bindex = detail
        device = _device_id(event)
        if device is None:
            unmatched.append(
                {
                    **record,
                    "launch_event_index": host_index,
                    "kind": "missing_device",
                    "reason": "kernel event has no integer args.device; its range cannot be "
                    "placed on one device clock",
                }
            )
            continue
        stream = _stream_id(event)
        attributed.append(
            {
                **record,
                "launch_event_index": host_index,
                "training_step": step,
                "boundary_event_index": bindex,
                "device": device,
                "stream": stream,
            }
        )

    total = len(attributed) + len(unmatched) + len(ambiguous)
    return {
        "kernel_events": {
            "total": total,
            "attributed": len(attributed),
            "unmatched": len(unmatched),
            "ambiguous": len(ambiguous),
        },
        "launch_events": {kind: len(indices) for kind, indices in launch_lists.items()},
        "launch_lookup": lookup,
        "excluded_launches": [
            {"kind": kind, "launch_event_indices": indices}
            for kind, indices in launch_lists.items()
            if kind != "attributed" and indices
        ],
        "attributed": attributed,
        "unmatched": unmatched,
        "ambiguous": ambiguous,
    }


def _step_groups(classified: dict[str, Any]) -> dict[int, dict[int, list[dict[str, Any]]]]:
    """Attributed records grouped by training step, then device, in trace order."""
    groups: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for record in classified["attributed"]:
        groups.setdefault(record["training_step"], {}).setdefault(record["device"], []).append(
            record
        )
    return groups


def attribute_trace(trace: Trace, boundaries: list[tuple[int, int]]) -> dict[str, Any]:
    """Classify one trace's kernels, then union the attributed ranges per step and device.

    Devices are never combined: each ``per_device`` entry is one device clock.
    """
    classified = classify_kernels(trace, boundaries)
    by_step = dict(boundaries)
    steps = []
    for step, devices in sorted(_step_groups(classified).items()):
        per_device = []
        for device, records in sorted(devices.items()):
            kernels = [trace.events[r["kernel_event_index"]] for r in records]
            per_device.append(
                {
                    "device": device,
                    "attributed_kernel_ms": interval_union_us(
                        (k["ts"], k["ts"] + k["dur"]) for k in kernels
                    )
                    / 1000,
                    "kernel_event_indices": [r["kernel_event_index"] for r in records],
                    "launch_event_indices": sorted({r["launch_event_index"] for r in records}),
                    "streams": sorted({str(r["stream"]) for r in records}),
                }
            )
        steps.append(
            {"training_step": step, "boundary_event_index": by_step[step], "per_device": per_device}
        )
    return {**classified, "steps": steps}


def _trace_warnings(name: str, result: dict[str, Any], trace: Trace) -> list[str]:
    counts = result["kernel_events"]
    summary = (
        f"{name}: kernel attribution: {counts['attributed']}/{counts['total']} kernel event(s) "
        "attributed through host launch correlation ids"
    )
    if counts["unmatched"]:
        summary += f"; {counts['unmatched']} unmatched"
    if counts["ambiguous"]:
        summary += f"; {counts['ambiguous']} ambiguous"
    lines = [summary]
    for label, records in (("unmatched", result["unmatched"]), ("ambiguous", result["ambiguous"])):
        by_kind: dict[str, list[int]] = {}
        for record in records:
            by_kind.setdefault(record["kind"], []).append(record["kernel_event_index"])
        for kind, indices in by_kind.items():
            lines.append(
                f"{name}: kernel events {label} ({kind}); not attributed and not timed: "
                + summarize_indices(indices)
            )
    for item in result["excluded_launches"]:
        indices = item["launch_event_indices"]
        detail = ""
        if item["kind"] == "outside_mapped_step":
            threads = sorted({_thread(trace.events[i]) for i in indices})
            detail = f" on host thread(s) {', '.join(threads)}"
        lines.append(
            f"{name}: supported kernel-launch events {item['kind']}{detail}; kernels they "
            "launched stay unmatched: " + summarize_indices(indices)
        )
    return lines


def step_measurement(
    *,
    rank: int,
    training_step: int,
    trace_name: str,
    result: dict[str, Any],
    unavailable: str | None = None,
) -> dict[str, Any]:
    """Build the per-step measurement from one trace's attribution result.

    The value is published only when every attributed kernel of the step sits
    on one device; several devices are kept apart in the attribution block and
    the measurement says so instead of adding clocks together.
    """
    scope = f"rank={rank}; training_step={training_step}; trace={trace_name}"
    counts = result["kernel_events"]
    unattributed = counts["unmatched"] + counts["ambiguous"]
    entry = next((s for s in result["steps"] if s["training_step"] == training_step), None)
    per_device = entry["per_device"] if entry else []
    coverage = {
        "step_kernel_events": sum(len(d["kernel_event_indices"]) for d in per_device),
        "step_launch_events": sum(len(d["launch_event_indices"]) for d in per_device),
        "devices": [d["device"] for d in per_device],
        "trace_kernel_events": counts["total"],
        "trace_attributed_kernel_events": counts["attributed"],
        "trace_unattributed_kernel_events": unattributed,
        "completeness": (
            "none" if not per_device else ("complete" if unattributed == 0 else "partial")
        ),
    }
    if unavailable:
        value = measurement(None, status="not_observed", reason=unavailable, scope=scope)
    elif not per_device:
        value = measurement(
            None,
            reason="no kernel event attributed to this step through host launch correlation; "
            + (
                f"{unattributed} of {counts['total']} kernel event(s) in this trace unattributed"
                if unattributed
                else f"{counts['total']} kernel event(s) in this trace"
            ),
            scope=scope,
        )
    elif len(per_device) > 1:
        value = measurement(
            None,
            reason=f"attributed kernels span devices {coverage['devices']}; per-device interval "
            "unions are kept separately in kernel_attribution and are not combined",
            scope=scope,
        )
    else:
        (device,) = per_device
        value = measurement(
            device["attributed_kernel_ms"],
            basis=ATTRIBUTION_BASIS,
            evidence=[
                {"file": trace_name, "event_index": index}
                for index in device["kernel_event_indices"]
            ],
            scope=f"{scope}; device={device['device']}",
            reason=SCOPE_STATEMENT
            + (
                f"; attribution incomplete: {unattributed} of {counts['total']} kernel "
                "event(s) in this trace unattributed"
                if unattributed
                else "; every kernel event in this trace was attributed to a mapped step"
            ),
        )
    value["coverage"] = coverage
    return value


def attribute_traces(
    traces: tuple[Trace, ...],
    rows: list[dict[str, Any]],
    *,
    rank: int,
    unavailable: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Attribute every trace and attach the per-step measurement to ``rows``.

    ``rows`` are the analysis step rows (with validated ``boundary``
    references) and are updated in place. Returns the persisted attribution
    block and warnings.
    """
    by_name = {trace.path.name: trace for trace in traces}
    boundaries: dict[str, list[tuple[int, int]]] = {name: [] for name in by_name}
    for row in rows:
        if row["boundary"] is not None:
            boundaries[row["boundary"]["file"]].append(
                (row["training_step"], row["boundary"]["event_index"])
            )
    results = {name: attribute_trace(trace, boundaries[name]) for name, trace in by_name.items()}
    warnings = []
    for name in sorted(results):
        warnings.extend(_trace_warnings(name, results[name], by_name[name]))
    for row in rows:
        if row["boundary"] is None:
            value = measurement(
                None,
                status="not_observed" if unavailable else None,
                reason=unavailable or "no explicit mapped step boundary",
            )
        else:
            name = row["boundary"]["file"]
            value = step_measurement(
                rank=rank,
                training_step=row["training_step"],
                trace_name=name,
                result=results[name],
                unavailable=unavailable,
            )
        row["measurements"][ATTRIBUTED_KERNEL_METRIC] = value
    warnings.append(
        SCOPE_STATEMENT + "; transfer, collective and busy-time metrics remain unavailable"
    )
    block = {
        "schema": ATTRIBUTION_SCHEMA,
        "method": METHOD,
        "supported_launch_events": [f"{cat}:{name}" for cat, name in SUPPORTED_LAUNCHES],
        "kernel_category": KERNEL_CATEGORY,
        "scope": SCOPE_STATEMENT,
        "traces": results,
    }
    return block, warnings


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(f"analysis.json kernel_attribution: {message}")


def _indices(value: Any, label: str) -> None:
    _require(
        isinstance(value, list) and all(type(i) is int and i >= 0 for i in value),
        f"{label} must be a list of nonnegative event indices",
    )


def validate_attribution(
    payload: dict[str, Any],
    *,
    trace_names: set[str],
    boundaries: dict[str, list[tuple[int, int]]],
    load_trace,
) -> None:
    """Check the persisted block and its relationship to the step measurements.

    ``load_trace(name)`` returns a parsed :class:`Trace` or ``None`` when no
    source directory is available; with sources, the block is recomputed from
    the trace and the stored copy must match exactly. Without sources the
    structure and the measurement/evidence relationships are checked.
    """
    block = payload.get("kernel_attribution")
    rows = payload["steps"]
    present = [row for row in rows if ATTRIBUTED_KERNEL_METRIC in row["measurements"]]
    if block is None:
        # Sidecars written before attribution existed carry neither the block
        # nor the metric; a metric without its evidence block is refused.
        _require(not present, "device_attributed_kernel_ms requires a kernel_attribution block")
        return
    _require(isinstance(block, dict), "expected object")
    _require(block.get("schema") == ATTRIBUTION_SCHEMA, "unsupported schema")
    _require(block.get("method") == METHOD, "unsupported method")
    _require(
        block.get("supported_launch_events") == [f"{c}:{n}" for c, n in SUPPORTED_LAUNCHES]
        and block.get("kernel_category") == KERNEL_CATEGORY,
        "supported event forms disagree with this reader",
    )
    traces = block.get("traces")
    _require(isinstance(traces, dict) and set(traces) <= trace_names, "traces must name sources")
    boundary_steps = {
        (name, index): step for name, items in boundaries.items() for step, index in items
    }
    for name, result in traces.items():
        _require(isinstance(result, dict), f"{name}: expected object")
        _require(
            {
                "kernel_events",
                "launch_events",
                "launch_lookup",
                "excluded_launches",
                "attributed",
                "unmatched",
                "ambiguous",
                "steps",
            }
            <= result.keys(),
            f"{name}: incomplete trace result",
        )
        for key in ("launch_lookup", "excluded_launches", "attributed", "unmatched", "ambiguous"):
            _require(
                isinstance(result[key], list) and all(isinstance(r, dict) for r in result[key]),
                f"{name}: {key} must be a list of records",
            )
        attributed_by_kernel: dict[int, dict] = {}
        for record in result["attributed"]:
            for key in ("kernel_event_index", "launch_event_index", "boundary_event_index"):
                _require(
                    type(record.get(key)) is int and record[key] >= 0,
                    f"{name}: attributed record needs integer {key}",
                )
            _require(
                type(record.get("correlation")) is int and type(record.get("device")) is int,
                f"{name}: attributed record needs integer correlation and device",
            )
            _require(
                boundary_steps.get((name, record["boundary_event_index"]))
                == record.get("training_step"),
                f"{name}: attributed kernel event[{record.get('kernel_event_index')}] names a "
                "boundary that is not the mapped boundary of its training step",
            )
            _require(
                record["kernel_event_index"] not in attributed_by_kernel,
                f"{name}: kernel event[{record['kernel_event_index']}] attributed twice",
            )
            attributed_by_kernel[record["kernel_event_index"]] = record
        for record in result["unmatched"] + result["ambiguous"]:
            _require(
                type(record.get("kernel_event_index")) is int
                and isinstance(record.get("reason"), str)
                and record["reason"]
                and record.get("kind") in UNMATCHED_KINDS + AMBIGUOUS_KINDS,
                f"{name}: unmatched/ambiguous records need kernel_event_index, kind and reason",
            )
            _require(
                record["kernel_event_index"] not in attributed_by_kernel,
                f"{name}: kernel event[{record['kernel_event_index']}] both attributed and not",
            )
        _require(isinstance(result["steps"], list), f"{name}: steps must be a list")
        for entry in result["steps"]:
            _require(
                isinstance(entry, dict) and isinstance(entry.get("per_device"), list),
                f"{name}: step entry must be an object with per_device",
            )
            for device in entry["per_device"]:
                _require(isinstance(device, dict), f"{name}: per_device entries must be objects")
                _indices(device.get("kernel_event_indices"), f"{name}: kernel_event_indices")
                _indices(device.get("launch_event_indices"), f"{name}: launch_event_indices")
                _require(
                    finite_number(device.get("attributed_kernel_ms"))
                    and device["attributed_kernel_ms"] >= 0,
                    f"{name}: attributed_kernel_ms must be finite and nonnegative",
                )
                for index in device["kernel_event_indices"]:
                    record = attributed_by_kernel.get(index)
                    _require(
                        record is not None
                        and record["training_step"] == entry.get("training_step")
                        and record["device"] == device.get("device"),
                        f"{name}: step {entry.get('training_step')} device "
                        f"{device.get('device')} cites kernel event[{index}] that is not "
                        "attributed to it",
                    )
        trace = load_trace(name)
        if trace is not None:
            expected = classify_kernels(trace, boundaries.get(name, []))
            _require(
                {key: result[key] for key in expected} == expected,
                f"{name}: stored attribution disagrees with the trace and step map",
            )
            grouped = _step_groups(expected)
            stored = {
                entry["training_step"]: {device["device"]: device for device in entry["per_device"]}
                for entry in result["steps"]
            }
            _require(
                stored.keys() == grouped.keys()
                and all(stored[step].keys() == grouped[step].keys() for step in grouped)
                and all(
                    stored[step][device]["kernel_event_indices"]
                    == [r["kernel_event_index"] for r in records]
                    and stored[step][device]["launch_event_indices"]
                    == sorted({r["launch_event_index"] for r in records})
                    for step, devices in grouped.items()
                    for device, records in devices.items()
                ),
                f"{name}: per-step kernel groups disagree with the attributed records",
            )

    # Each step measurement must be the one the block supports.
    for row in rows:
        value = row["measurements"].get(ATTRIBUTED_KERNEL_METRIC)
        if value is None:
            continue
        if row["boundary"] is None:
            _require(
                value["status"] != "recorded", f"training_step {row['training_step']}: no boundary"
            )
            continue
        coverage = value.get("coverage")
        _require(
            isinstance(coverage, dict)
            and coverage.get("completeness") in ("none", "partial", "complete"),
            f"training_step {row['training_step']}: measurement needs coverage",
        )
        name = row["boundary"]["file"]
        result = traces.get(name)
        _require(result is not None, f"training_step {row['training_step']}: trace {name} missing")
        entry = next(
            (s for s in result["steps"] if s.get("training_step") == row["training_step"]), None
        )
        per_device = entry["per_device"] if entry else []
        if value["status"] == "recorded":
            _require(
                value["basis"] == ATTRIBUTION_BASIS,
                f"training_step {row['training_step']}: needs {ATTRIBUTION_BASIS} basis",
            )
            _require(
                len(per_device) == 1
                and value["value_ms"] == per_device[0]["attributed_kernel_ms"]
                and [ref["event_index"] for ref in value["evidence"]]
                == per_device[0]["kernel_event_indices"]
                and all(ref["file"] == name for ref in value["evidence"])
                and value["scope"].endswith(f"device={per_device[0]['device']}"),
                f"training_step {row['training_step']}: recorded attributed kernel time "
                "disagrees with the attribution block",
            )
        else:
            _require(
                len(per_device) != 1 or value["status"] == "not_observed",
                f"training_step {row['training_step']}: attributed kernels exist but the "
                "measurement withholds them without a not_observed reason",
            )
