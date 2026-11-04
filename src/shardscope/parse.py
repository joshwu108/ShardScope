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
