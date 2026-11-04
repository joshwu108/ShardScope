"""Structural Chrome/Kineto reading with stable source-array evidence.

Chrome timestamps and durations are microseconds; displayTimeUnit only controls
viewer formatting. Unknown events and metadata remain intact. Reading a trace
neither assigns a rank nor establishes asynchronous device ownership.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shardscope.errors import ArtifactError


@dataclass(frozen=True)
class Trace:
    """Original events, metadata and digest of the exact (possibly gzip) bytes."""

    path: Path
    events: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]
    sha256: str
    warnings: tuple[str, ...]

    def reference(self, index: int) -> dict[str, Any]:
        if type(index) is not int or not 0 <= index < len(self.events):
            raise ArtifactError(f"{self.path}: event_index {index!r} is out of range")
        return {"file": self.path.name, "event_index": index}


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
        raw = target.read_bytes()
        decoded = gzip.decompress(raw) if target.suffix.lower() == ".gz" else raw
    except (OSError, EOFError) as exc:
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
    stacks: dict[tuple[Any, Any], list[dict]] = {}
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
                stack = stacks.setdefault((event["pid"], event["tid"]), [])
                if phase == "B":
                    stack.append(event)
                elif not stack or event["ts"] < stack.pop()["ts"]:
                    raise ArtifactError(f"{where}: unmatched or reversed B/E range")
        if phase != "X":
            warnings.append(f"event[{index}]: {phase!r} form retained; unsupported for host timing")
    if any(stacks.values()):
        raise ArtifactError(f"{target}: unclosed B/E range")
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
    for mapping in step_map:
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
        name = event.get("name", "")
        if event["ph"] != "X" or not (
            name == "shardscope/step" or name.startswith("ProfilerStep#")
        ):
            raise ArtifactError(
                f"{mapping.trace_file} event[{mapping.event_index}]: map needs complete step marker"
            )
        boundaries.append((mapping, trace, event))
    rows = []
    assigned = set()
    for mapping, trace, boundary in boundaries:
        metrics = {}
        scope = (
            f"rank={rank}; trace={trace.path.name}; pid={boundary['pid']}; tid={boundary['tid']}"
        )
        for marker, metric in HOST_PHASES.items():
            ranges = []
            refs = []
            ambiguous = False
            for index, event in enumerate(trace.events):
                if event.get("name") != marker:
                    continue
                if event["ph"] != "X":
                    warnings.append(
                        f"{trace.path.name} event[{index}]: unsupported host marker form"
                    )
                    continue
                owners = [
                    m
                    for m, t, b in boundaries
                    if t is trace
                    and event["pid"] == b["pid"]
                    and event["tid"] == b["tid"]
                    and b["ts"] <= event["ts"]
                    and event["ts"] + event["dur"] <= b["ts"] + b["dur"]
                ]
                if mapping not in owners:
                    continue
                if len(owners) != 1:
                    ambiguous = True
                    continue
                assigned.add((trace.path.name, index))
                ranges.append((event["ts"], event["ts"] + event["dur"]))
                refs.append(trace.reference(index))
            if len(refs) > 1:
                warnings.append(
                    f"step {mapping.training_step}: multiple {marker} ranges; using interval union"
                )
            if ambiguous:
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
                reason=reason
                or "no validated device-to-step correlation or activity classification",
                scope=f"rank={rank}; training_step={mapping.training_step}",
            )
        rows.append(
            {
                "training_step": mapping.training_step,
                "boundary": trace.reference(mapping.event_index),
                "measurements": metrics,
            }
        )
    for trace in traces:
        for index, event in enumerate(trace.events):
            if event.get("name") in HOST_PHASES and (trace.path.name, index) not in assigned:
                warnings.append(
                    f"{trace.path.name} event[{index}]: host marker unmapped or ambiguous"
                )
    if not step_map:
        warnings.append("no explicit step mapping; training-step measurements unavailable")
    warnings.append(
        "device metrics unavailable without validated attribution; "
        "raw events preserved in source traces"
    )
    return sorted(rows, key=lambda row: row["training_step"]), warnings
