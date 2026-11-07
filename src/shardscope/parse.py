"""Structural Chrome/Kineto reading with stable source-array evidence.

Chrome timestamps and durations are microseconds; displayTimeUnit only controls
viewer formatting. Unknown events and metadata remain intact. Reading a trace
neither assigns a rank nor establishes asynchronous device ownership.

Warnings describe what was retained without being measured. Repetitive
informational forms (metadata, flow and instant events; device-side copies of
host markers) are summarized once per trace with representative original
indices, so a malformed host marker or a mapping problem is not buried among
hundreds of identical lines.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shardscope.errors import ArtifactError


@dataclass(frozen=True)
class Trace:
    """Original events, metadata and digest of the exact bytes."""

    path: Path
    events: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]
    sha256: str
    warnings: tuple[str, ...]

    def reference(self, index: int) -> dict[str, Any]:
        if type(index) is not int or not 0 <= index < len(self.events):
            raise ArtifactError(f"{self.path}: event_index {index!r} is out of range")
        return {"file": self.path.name, "event_index": index}


#: Kineto's category for the device-side mirror of a ``record_function`` range.
#: The same ``shardscope/*`` names appear on a GPU stream (pid 0, tid = stream)
#: with device timestamps; they are not host phase markers.
DEVICE_ANNOTATION_CATEGORY = "gpu_user_annotation"

#: Human labels for Chrome event forms that carry no measurable host range.
_FORM_LABELS = {
    "M": "metadata",
    "s": "flow-start",
    "t": "flow-step",
    "f": "flow-end",
    "i": "instant",
    "I": "instant",
    "B": "duration-begin",
    "E": "duration-end",
    "b": "async-begin",
    "n": "async-instant",
    "e": "async-end",
    "C": "counter",
}

#: Representative original indices quoted in an aggregated warning.
INDEX_SAMPLE = 5


def is_device_annotation(event: dict[str, Any]) -> bool:
    """True for a Kineto GPU-side user annotation; classified by category only.

    Nothing is inferred from the name, the pid/tid or timestamp containment:
    a device annotation is one the producer labelled as such.
    """
    return event.get("cat") == DEVICE_ANNOTATION_CATEGORY


def summarize_indices(indices, *, limit: int = INDEX_SAMPLE) -> str:
    """Spell a count plus up to ``limit`` representative original event indices."""
    indices = list(indices)
    shown = ", ".join(f"event[{index}]" for index in indices[:limit])
    rest = len(indices) - min(len(indices), limit)
    if rest:
        shown += f", ... {rest} more"
    return f"{len(indices)} event(s): {shown}"


def finite_number(value: Any) -> bool:
    """JSON numbers exclude booleans and non-finite extensions."""
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def read_json_bytes(raw: bytes, *, source: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ArtifactError(f"{source}: malformed UTF-8 JSON: {exc}") from exc


def parse_trace(path: str | Path) -> Trace:
    """Read object-with-traceEvents and legacy event-array exports.

    Complete X ranges are measurable. Synchronous B/E ranges are structurally
    checked but retained as unsupported for timing. Other forms stay raw.
    No activity is inferred from names, filenames or device inventory.
    """
    target = Path(path)
    try:
        if not target.is_file():
            raise ArtifactError(f"cannot read trace {target}: expected a regular file")
        raw = target.read_bytes()
        decoded = gzip.decompress(raw) if target.suffix.lower() == ".gz" else raw
    except (OSError, EOFError, zlib.error) as exc:
        raise ArtifactError(f"cannot read trace {target}: {exc}") from exc
    payload = read_json_bytes(decoded, source=str(target))
    metadata = {}
    if isinstance(payload, dict):
        metadata = {k: v for k, v in payload.items() if k != "traceEvents"}
        payload = payload.get("traceEvents")
        # Chrome fixes ts/dur to microseconds. Reject an explicit conflicting extension.
        for key in ("timestampUnit", "durationUnit"):
            if key in metadata and metadata[key] != "us":
                raise ArtifactError(f"{target}: unsupported {key}={metadata[key]!r}; expected us")
    if not isinstance(payload, list):
        raise ArtifactError(f"{target}: expected event array or object with traceEvents array")
    warnings = []
    stacks: dict[tuple[Any, Any], list[tuple[int, dict]]] = {}
    last_range_time = {}
    retained: dict[str, list[int]] = {}
    for index, event in enumerate(payload):
        where = f"{target} event[{index}]"
        if not isinstance(event, dict) or not isinstance(event.get("ph"), str) or not event["ph"]:
            raise ArtifactError(f"{where}: expected event object with nonempty string ph")
        phase = event["ph"]
        for key in ("ts", "dur", "tts", "tdur"):
            if key in event and (
                not finite_number(event[key]) or (key in ("dur", "tdur") and event[key] < 0)
            ):
                raise ArtifactError(
                    f"{where}: {key} must be finite numeric microseconds; duration >= 0"
                )
        for key in ("name", "cat"):
            if key in event and not isinstance(event[key], str):
                raise ArtifactError(f"{where}: {key} must be a string")
        if "args" in event and not isinstance(event["args"], dict):
            raise ArtifactError(f"{where}: args must be an object")
        if phase in ("X", "B", "E"):
            required = ("ts", "pid", "tid") + (("name",) if phase != "E" else ())
            for key in required:
                if key not in event:
                    raise ArtifactError(f"{where}: {phase} range missing {key}")
            for key in ("pid", "tid"):
                if type(event[key]) not in (str, int):
                    raise ArtifactError(f"{where}: {key} must be string or integer")
            if phase == "X":
                if "dur" not in event:
                    raise ArtifactError(f"{where}: X range missing dur")
                if not finite_number(event["ts"] + event["dur"]):
                    raise ArtifactError(f"{where}: range endpoint is not finite")
            else:
                thread = (event["pid"], event["tid"])
                stack = stacks.setdefault(thread, [])
                if thread in last_range_time and event["ts"] < last_range_time[thread]:
                    raise ArtifactError(f"{where}: reversed B/E range timestamp order")
                last_range_time[thread] = event["ts"]
                if phase == "B":
                    stack.append((index, event))
                elif not stack:
                    raise ArtifactError(f"{where}: unmatched B/E range")
                else:
                    _, begin = stack.pop()
                    if "name" in event and event["name"] != begin["name"]:
                        raise ArtifactError(f"{where}: mismatched B/E range name")
        if phase != "X":
            retained.setdefault(phase, []).append(index)
    for stack in stacks.values():
        if stack:
            raise ArtifactError(f"{target} event[{stack[-1][0]}]: unclosed B/E range")
    for phase, indices in retained.items():
        label = _FORM_LABELS.get(phase, "unknown-form")
        warnings.append(
            f"{label} ({phase!r}) events retained raw; unsupported for host timing: "
            + summarize_indices(indices)
        )
    if not payload:
        warnings.append("empty trace: no events recorded")
    return Trace(target, tuple(payload), metadata, hashlib.sha256(raw).hexdigest(), tuple(warnings))


HOST_PHASES = {
    "shardscope/step": "host_step_ms",
    "shardscope/data_wait": "host_data_wait_ms",
    "shardscope/h2d": "host_h2d_ms",
    "shardscope/forward": "host_forward_ms",
    "shardscope/backward": "host_backward_ms",
    "shardscope/optimizer": "host_optimizer_ms",
}
DEVICE_METRICS = ("device_h2d_ms", "device_compute_ms", "device_collective_ms", "device_busy_ms")

#: Why each device metric stays unavailable when the environment could have
#: produced it. Only correlated kernel ranges are attributed, and that subset
#: is published separately as ``device_attributed_kernel_ms``; none of these
#: four complete-activity metrics can be derived from it.
DEVICE_METRIC_REASONS = {
    "device_compute_ms": (
        "complete per-step device compute time is not measured; "
        "device_attributed_kernel_ms carries only the kernel ranges attributed through "
        "host launch correlation"
    ),
    "device_h2d_ms": (
        "complete per-step device copy time is not measured; "
        "device_attributed_transfer_ms carries only the gpu_memcpy ranges attributed through "
        "host copy-call correlation, with direction in its coverage"
    ),
    "device_collective_ms": (
        "per-step collective time is not attributed: collective kernels are launched from a "
        "host thread without a mapped step boundary; kernel_attribution carries window-level "
        "per-device collective unions and their overlap with compute kernels"
    ),
}
DEFAULT_DEVICE_REASON = "no validated device-to-step correlation or activity classification"


@dataclass(frozen=True)
class StepMapping:
    """Producer-supplied identity of an actual complete step-marker event.

    This consumer interface is independent of the producer's pending JSON map.
    Indices are positions in the original trace array, never filtered indices.
    """

    trace_file: str
    event_index: int
    training_step: int


def interval_union_us(intervals) -> float:
    """Length of an interval union in one caller-specified clock and scope.

    Never pass intervals from different traces, hosts or unaligned clocks.
    The caller is responsible for device/stream scope and valid attribution.
    """
    checked = []
    for start, end in intervals:
        if not finite_number(start) or not finite_number(end) or end < start:
            raise ArtifactError("interval union requires finite ordered microsecond endpoints")
        checked.append((start, end))
    total = 0.0
    right = None
    for start, end in sorted(checked):
        total += end - start if right is None else max(0, end - max(start, right))
        right = end if right is None else max(right, end)
    if not finite_number(total):
        raise ArtifactError("interval union overflow")
    return total


def _merged_intervals(intervals) -> list[tuple[float, float]]:
    """Sorted, non-overlapping intervals covering the same points as ``intervals``."""
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not finite_number(start) or not finite_number(end) or end < start:
            raise ArtifactError("interval union requires finite ordered microsecond endpoints")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def interval_intersection_us(first, second) -> float:
    """Length of ``(union of first) & (union of second)`` in one clock and scope.

    The same caveat as :func:`interval_union_us` applies: both interval sets
    must come from the same trace and device clock, and the caller has
    established what each set means. Overlapping members inside one set are
    unioned first, so a copy that two kernels straddle is counted once.
    """
    left, right = _merged_intervals(first), _merged_intervals(second)
    total = 0.0
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            total += end - start
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    if not finite_number(total):
        raise ArtifactError("interval intersection overflow")
    return total


IDENTITY_PREFIX = "shardscope/training_step/"
STEP_MARKER = "shardscope/step"
PROFILER_STEP_PREFIX = "ProfilerStep#"


def is_step_boundary(event: dict[str, Any]) -> bool:
    """A complete host step marker: the only event form a step map may name."""
    name = event.get("name", "")
    return (
        event.get("ph") == "X"
        and not is_device_annotation(event)
        and (name == STEP_MARKER or name.startswith(PROFILER_STEP_PREFIX))
    )


def contains_range(boundary: dict[str, Any], event: dict[str, Any]) -> bool:
    """Same host thread and the whole range inside the boundary; no slack."""
    return (
        event["pid"] == boundary["pid"]
        and event["tid"] == boundary["tid"]
        and boundary["ts"] <= event["ts"]
        and event["ts"] + event["dur"] <= boundary["ts"] + boundary["dur"]
    )


def identity_step(trace: Trace, boundary_index: int) -> int | None:
    """The training step named by the identity marker nested in a boundary.

    The capture producer nests ``shardscope/training_step/<N>`` inside each
    ``shardscope/step``; a mapping that disagrees with that marker is a
    mapping problem, not a measurement. Returns ``None`` when the boundary
    holds no identity marker, so a trace without identity markers is still
    analyzable from its explicit map alone. A malformed or ambiguous identity
    is an error rather than a guess.
    """
    boundary = trace.events[boundary_index]
    found: dict[int, int] = {}
    for index, event in enumerate(trace.events):
        name = event.get("name", "")
        if (
            event.get("ph") != "X"
            or not name.startswith(IDENTITY_PREFIX)
            or is_device_annotation(event)
            or not contains_range(boundary, event)
        ):
            continue
        suffix = name[len(IDENTITY_PREFIX) :]
        if not suffix.isdigit():
            raise ArtifactError(
                f"{trace.path.name} event[{index}]: malformed identity marker {name!r}"
            )
        found.setdefault(int(suffix), index)
    if len(found) > 1:
        raise ArtifactError(
            f"{trace.path.name} event[{boundary_index}]: boundary contains identity markers "
            f"for several training steps: {sorted(found)}"
        )
    return next(iter(found), None)


def measurement(value, *, basis=None, reason="", evidence=(), status=None, scope="") -> dict:
    return {
        "value_ms": value,
        "status": status or ("recorded" if value is not None else "not_recorded"),
        "basis": basis,
        "reason": reason,
        "evidence": list(evidence),
        "scope": scope,
    }


def analyze_traces(
    traces: tuple[Trace, ...],
    *,
    rank: int,
    step_map: tuple[StepMapping, ...],
    unavailable: dict[str, str] | None = None,
) -> tuple[list[dict], list[str]]:
    """Measure host markers using explicit rank and training-step identities.

    Host phase ownership requires a unique containing mapped boundary on the
    same thread. Nested phase ranges use a union, not a duration sum. Device
    events stay raw: host containment does not prove asynchronous ownership.
    """
    if type(rank) is not int or rank < 0:
        raise ArtifactError("rank must be an explicit nonnegative integer")
    by_name = {trace.path.name: trace for trace in traces}
    if len(by_name) != len(traces):
        raise ArtifactError("trace filenames must be unique within one rank")
    warnings = [f"{trace.path.name}: {warning}" for trace in traces for warning in trace.warnings]
    boundaries = []
    seen_steps = set()
    seen_boundaries = set()
    for mapping in step_map:
        if not isinstance(mapping, StepMapping):
            raise ArtifactError("step_map must contain StepMapping objects")
        target = (mapping.trace_file, mapping.event_index)
        if not isinstance(mapping.trace_file, str) or type(mapping.event_index) is not int:
            raise ArtifactError("step mapping requires filename and integer event index")
        if target in seen_boundaries:
            raise ArtifactError("same boundary mapped to multiple training steps")
        seen_boundaries.add(target)
        if type(mapping.training_step) is not int or mapping.training_step < 0:
            raise ArtifactError("training_step must be a nonnegative integer")
        if mapping.training_step in seen_steps:
            raise ArtifactError(f"duplicate mapped training step {mapping.training_step}")
        seen_steps.add(mapping.training_step)
        trace = by_name.get(mapping.trace_file)
        if trace is None:
            raise ArtifactError(f"step map names unknown trace {mapping.trace_file!r}")
        trace.reference(mapping.event_index)
        event = trace.events[mapping.event_index]
        if not is_step_boundary(event):
            kind = (
                f"device annotation ({DEVICE_ANNOTATION_CATEGORY})"
                if is_device_annotation(event)
                else "not a complete step marker"
            )
            raise ArtifactError(
                f"{mapping.trace_file} event[{mapping.event_index}]: map needs complete host "
                f"step marker; {kind}"
            )
        identity = identity_step(trace, mapping.event_index)
        if identity is not None and identity != mapping.training_step:
            raise ArtifactError(
                f"{mapping.trace_file} event[{mapping.event_index}]: identity marker names "
                f"training step {identity} but the map says {mapping.training_step}"
            )
        boundaries.append((mapping, trace, event, identity))

    # Classify every host-marker-named event once per trace. Each event keeps
    # its original index; nothing is filtered out of the trace itself.
    classified: dict[str, dict[str, Any]] = {}
    for trace in traces:
        device: dict[str, list[int]] = {}
        unsupported: dict[str, list[tuple[int, str]]] = {}
        host: list[tuple[int, dict, list]] = []
        local = [b for b in boundaries if b[1] is trace]
        for index, event in enumerate(trace.events):
            marker = event.get("name")
            if marker not in HOST_PHASES:
                continue
            if is_device_annotation(event):
                device.setdefault(marker, []).append(index)
            elif event["ph"] != "X":
                unsupported.setdefault(marker, []).append((index, event["ph"]))
            else:
                owners = [m for m, _, b, _ in local if contains_range(b, event)]
                host.append((index, event, owners))
        classified[trace.path.name] = {"device": device, "unsupported": unsupported, "host": host}

    rows = []
    for mapping, trace, boundary, _identity in boundaries:
        metrics = {}
        scope = (
            f"rank={rank}; trace={trace.path.name}; pid={boundary['pid']}; tid={boundary['tid']}"
        )
        kinds = classified[trace.path.name]
        for marker, metric in HOST_PHASES.items():
            ranges = []
            refs = []
            ambiguous = False
            for index, event, owners in kinds["host"]:
                if event["name"] != marker or mapping not in owners:
                    continue
                if len(owners) != 1:
                    ambiguous = True
                    continue
                ranges.append((event["ts"], event["ts"] + event["dur"]))
                refs.append(trace.reference(index))
            if len(refs) > 1:
                warnings.append(
                    f"step {mapping.training_step}: multiple {marker} ranges; using interval union"
                )
            if marker in kinds["unsupported"]:
                # Unsupported marker forms can hide additional duration.
                # Do not publish a partial union as the complete phase.
                metrics[metric] = measurement(
                    None,
                    reason="unsupported host marker form prevents complete phase timing",
                    scope=scope,
                )
            elif ambiguous:
                metrics[metric] = measurement(
                    None, reason="ambiguous overlapping mapped host boundaries", scope=scope
                )
            elif refs:
                metrics[metric] = measurement(
                    interval_union_us(ranges) / 1000,
                    basis="host_range",
                    evidence=refs,
                    scope=scope,
                    reason="union of complete host ranges; not device execution time",
                )
            else:
                metrics[metric] = measurement(
                    None, reason=f"missing complete {marker} marker", scope=scope
                )
        for metric in DEVICE_METRICS:
            reason = (unavailable or {}).get(metric)
            metrics[metric] = measurement(
                None,
                status="not_observed" if reason else "not_recorded",
                reason=reason or DEVICE_METRIC_REASONS.get(metric, DEFAULT_DEVICE_REASON),
                scope=f"rank={rank}; training_step={mapping.training_step}",
            )
        rows.append(
            {
                "training_step": mapping.training_step,
                "boundary": trace.reference(mapping.event_index),
                "measurements": metrics,
            }
        )

    # One summary line per trace and marker for each retained-but-unmeasured
    # class, with representative original indices. These are the lines a
    # reader acts on; the raw events stay in the source trace.
    for trace in traces:
        name = trace.path.name
        kinds = classified[name]
        for marker, items in kinds["unsupported"].items():
            forms = sorted({form for _, form in items})
            warnings.append(
                f"{name}: {marker} in unsupported host marker form {forms}; "
                f"{HOST_PHASES[marker]} withheld for every step mapped in this trace: "
                + summarize_indices(index for index, _ in items)
            )
        unmapped: dict[str, list[int]] = {}
        ambiguous: dict[str, list[int]] = {}
        for index, event, owners in kinds["host"]:
            if not owners:
                unmapped.setdefault(event["name"], []).append(index)
            elif len(owners) > 1:
                ambiguous.setdefault(event["name"], []).append(index)
        for marker, indices in unmapped.items():
            warnings.append(
                f"{name}: {marker} host marker outside every mapped step boundary "
                "(unmapped; not measured): " + summarize_indices(indices)
            )
        for marker, indices in ambiguous.items():
            warnings.append(
                f"{name}: {marker} host marker inside several mapped step boundaries "
                "(ambiguous; not measured): " + summarize_indices(indices)
            )
        for marker, indices in kinds["device"].items():
            warnings.append(
                f"{name}: {marker} device annotation ({DEVICE_ANNOTATION_CATEGORY}) retained "
                "raw; not a host phase and not attributed to a step: " + summarize_indices(indices)
            )
        without_identity = [
            m.event_index for m, t, _, identity in boundaries if t is trace and identity is None
        ]
        if without_identity:
            warnings.append(
                f"{name}: mapped step boundaries without a nested {IDENTITY_PREFIX}<N> identity "
                "marker; training steps taken from the explicit map alone: "
                + summarize_indices(without_identity)
            )
    if not step_map:
        warnings.append("no explicit step mapping; training-step measurements unavailable")
    warnings.append(
        "device metrics unavailable without complete validated attribution; only kernel and "
        "gpu_memcpy ranges tied to a step by host launch/copy-call correlation are attributed "
        "(device_attributed_kernel_ms, device_attributed_transfer_ms); raw events preserved in "
        "source traces"
    )
    return sorted(rows, key=lambda row: row["training_step"]), warnings
