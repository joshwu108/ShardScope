"""Device-event attribution to training steps through host CUDA correlation IDs.

Kineto records a host-side ``cuda_runtime``/``cuda_driver`` range for each CUDA
API call and a device-side range for what it caused: a ``kernel`` for a kernel
launch, a ``gpu_memcpy`` for a copy call. Host and device sides share
``args.correlation``. A device event is attributed to a training step only when:

1. its correlation ID resolves to exactly one host CUDA API event in the same
   trace (IDs are looked up per trace; they are never assumed globally unique),
2. that host event is a supported issuing form: ``cudaLaunchKernel``,
   ``cudaLaunchKernelExC`` or ``cuLaunchKernelEx`` for kernels,
   ``cudaMemcpyAsync`` for transfers (the forms observed in real captures), and
3. the host range lies entirely inside exactly one explicitly mapped
   ``shardscope/step`` boundary on the same host process and thread.

Nothing is attributed by GPU timestamp containment. Host calls made on a thread
that carries no mapped boundary (the autograd engine thread launches backward
kernels, DDP bucket copies and NCCL collectives) are reported, not guessed at,
so attribution is usually incomplete and every result says so.

Kernels are also classified, independently of step attribution, as ``compute``,
``collective`` or ``collective_candidate``. A collective needs two signals: the
NCCL device-kernel name prefix and a launch nested inside a ``c10d::`` collective
op on the same host thread. One signal alone makes a candidate, which belongs to
neither set. Per device and per trace, the compute union, the collective union
and their intersection are recorded; these are window-level figures on one
device clock and are not attributed to steps.

Three per-step measurements come out of this, each a per-device interval union
or intersection within one trace:

* ``device_attributed_kernel_ms``: union of the kernels attributed to the step.
* ``device_attributed_transfer_ms``: union of the ``gpu_memcpy`` ranges
  attributed to the step, with direction read from Kineto's label.
* ``device_transfer_compute_overlap_ms``: length of that transfer union that
  coincides with any compute-class kernel on the same device in the trace,
  whichever step launched it.

None of them is utilization, total step latency, total device busy time or
bandwidth; memsets and ``gpu_user_annotation`` mirrors are counted nowhere.
"""

from __future__ import annotations

import re
from typing import Any

from shardscope.errors import ArtifactError
from shardscope.parse import (
    Trace,
    contains_range,
    finite_number,
    interval_intersection_us,
    interval_union_us,
    measurement,
    summarize_indices,
)

__all__ = [
    "ATTRIBUTED_KERNEL_METRIC",
    "ATTRIBUTED_TRANSFER_METRIC",
    "ATTRIBUTION_BASIS",
    "ATTRIBUTION_METRICS",
    "ATTRIBUTION_SCHEMA",
    "COLLECTIVE_KERNEL_PREFIXES",
    "COLLECTIVE_OP_PREFIX",
    "HOST_API_CATEGORIES",
    "KERNEL_CATEGORY",
    "KERNEL_CLASSES",
    "OVERLAP_BASIS",
    "SUPPORTED_COPIES",
    "SUPPORTED_LAUNCHES",
    "TRANSFER_CATEGORY",
    "TRANSFER_DIRECTIONS",
    "TRANSFER_OVERLAP_METRIC",
    "attribute_trace",
    "attribute_traces",
    "classify_device_events",
    "correlation_id",
    "is_host_api_event",
    "is_kernel_event",
    "is_supported_copy",
    "is_supported_launch",
    "is_transfer_event",
    "step_measurements",
    "transfer_direction",
    "validate_attribution",
]

#: Per-step measurements carrying attributed device ranges.
ATTRIBUTED_KERNEL_METRIC = "device_attributed_kernel_ms"
ATTRIBUTED_TRANSFER_METRIC = "device_attributed_transfer_ms"
TRANSFER_OVERLAP_METRIC = "device_transfer_compute_overlap_ms"
ATTRIBUTION_METRICS = (
    ATTRIBUTED_KERNEL_METRIC,
    ATTRIBUTED_TRANSFER_METRIC,
    TRANSFER_OVERLAP_METRIC,
)
ATTRIBUTION_BASIS = "device_interval_union"
OVERLAP_BASIS = "device_interval_intersection"
#: Schema 1 blocks (kernels only, runtime launches only) still validate.
ATTRIBUTION_SCHEMA = 2
LEGACY_SCHEMAS = (1,)
METHOD = "host_launch_correlation"

#: Kineto categories of host-side CUDA API ranges that carry correlation IDs.
HOST_API_CATEGORIES = ("cuda_runtime", "cuda_driver")
#: Kernel-launch forms attributed, as (category, name); the first two are the
#: runtime launches PyTorch kernels use, the driver form is what NCCL uses.
SUPPORTED_LAUNCHES = (
    ("cuda_runtime", "cudaLaunchKernel"),
    ("cuda_runtime", "cudaLaunchKernelExC"),
    ("cuda_driver", "cuLaunchKernelEx"),
)
#: What a schema 1 block was computed with; needed to recheck such a block.
_LEGACY_LAUNCHES = SUPPORTED_LAUNCHES[:2]
#: Copy-call forms attributed. PyTorch issues every CUDA copy through
#: cudaMemcpyAsync (synchronous copies add a stream synchronization).
SUPPORTED_COPIES = (("cuda_runtime", "cudaMemcpyAsync"),)
#: Kineto categories. Memsets and gpu_user_annotation mirrors are neither.
KERNEL_CATEGORY = "kernel"
TRANSFER_CATEGORY = "gpu_memcpy"

#: NCCL generates device kernels as ``ncclDevKernel_<Coll>[_<Op>]_<Type>_<Algo>``
#: (older releases: ``ncclKernel_...``). The name alone is one signal.
COLLECTIVE_KERNEL_PREFIXES = ("ncclDevKernel_", "ncclKernel_")
#: torch.distributed dispatches collectives through ``c10d::`` ops (``cpu_op``
#: ranges such as ``c10d::allreduce_``); a launch nested in one is the other.
COLLECTIVE_OP_CATEGORY = "cpu_op"
COLLECTIVE_OP_PREFIX = "c10d::"
KERNEL_CLASSES = ("compute", "collective", "collective_candidate")

#: Kineto names a copy ``Memcpy <kind> (<src> -> <dst>)`` with CUPTI's kind
#: strings (HtoD, DtoH, DtoD, HtoH, PtoP, HtoA, ...). Only HtoD and DtoH are
#: given a direction; every other parsed kind is ``other``, anything else
#: ``unknown``.
_MEMCPY_NAME = re.compile(r"^Memcpy (\S+) \((.+) -> (.+)\)$")
TRANSFER_DIRECTIONS = ("h2d", "d2h", "other", "unknown")
_DIRECTION_BY_KIND = {"HtoD": "h2d", "DtoH": "d2h"}

KERNEL_SCOPE_STATEMENT = (
    "attributed kernel time: per-device interval union of kernel ranges tied to a "
    "training step through host kernel-launch correlation ids within one trace; not "
    "utilization, not total step latency, not total device busy time; copies, memsets and "
    "gpu_user_annotation events are not kernels and are not counted; collective kernels "
    "are kernels and are classified separately in kernel_attribution"
)
TRANSFER_SCOPE_STATEMENT = (
    "attributed transfer time: per-device interval union of gpu_memcpy ranges tied to a "
    "training step through host copy-call correlation ids within one trace, with direction "
    "read from Kineto's memcpy label; not bandwidth, not host copy-call time (host_h2d_ms), "
    "not total device busy time; memsets are not transfers"
)
OVERLAP_SCOPE_STATEMENT = (
    "transfer/compute overlap: length of the step's attributed transfer union that "
    "coincides with any compute-class kernel on the same device in this trace, whichever "
    "step launched it; collective kernels are not compute; the remainder of the transfer "
    "union is device copy time with no compute kernel running; not utilization"
)
COLLECTIVE_SCOPE_STATEMENT = (
    "collective/compute overlap: per device and per trace, the interval union of kernels "
    "classified collective, the union of compute-class kernels and their intersection, on "
    "one device clock over the whole captured window; not attributed to training steps, "
    "not network transfer time, not a statement about which rank waited"
)

ISSUE_KINDS = (
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

#: Keys of a trace result that classification alone determines; validation
#: recomputes exactly these and never re-times anything.
_CLASSIFICATION_KEYS = (
    "kernel_events",
    "launch_events",
    "launch_lookup",
    "excluded_launches",
    "attributed",
    "unmatched",
    "ambiguous",
    "kernel_classes",
    "collective_candidates",
    "unplaced_kernel_event_indices",
    "transfer_events",
    "copy_events",
    "copy_lookup",
    "excluded_copies",
    "transfers_attributed",
    "transfers_unmatched",
    "transfers_ambiguous",
    "devices",
)
_LEGACY_CLASSIFICATION_KEYS = _CLASSIFICATION_KEYS[:7]


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


def _bytes(event: dict[str, Any]) -> int | None:
    args = event.get("args")
    value = args.get("bytes") if isinstance(args, dict) else None
    return value if type(value) is int and value >= 0 else None


def is_host_api_event(event: dict[str, Any]) -> bool:
    """A complete host-side CUDA runtime/driver API range, by category only."""
    return event.get("ph") == "X" and event.get("cat") in HOST_API_CATEGORIES


def is_supported_launch(event: dict[str, Any]) -> bool:
    return is_host_api_event(event) and (event.get("cat"), event.get("name")) in SUPPORTED_LAUNCHES


def is_supported_copy(event: dict[str, Any]) -> bool:
    return is_host_api_event(event) and (event.get("cat"), event.get("name")) in SUPPORTED_COPIES


def is_kernel_event(event: dict[str, Any]) -> bool:
    """A complete device kernel range, by category only."""
    return event.get("ph") == "X" and event.get("cat") == KERNEL_CATEGORY


def is_transfer_event(event: dict[str, Any]) -> bool:
    """A complete device copy range, by category only; memsets are not copies."""
    return event.get("ph") == "X" and event.get("cat") == TRANSFER_CATEGORY


def transfer_direction(event: dict[str, Any]) -> tuple[str, str | None]:
    """``(direction, kineto_kind)`` from the producer's memcpy label.

    Direction is ``h2d``, ``d2h``, ``other`` for any other parsed CUPTI kind
    (DtoD, HtoH, PtoP, array copies) or ``unknown`` when the label is not in
    Kineto's ``Memcpy <kind> (<src> -> <dst>)`` form.
    """
    match = _MEMCPY_NAME.match(event.get("name") or "")
    if match is None:
        return "unknown", None
    kind = match.group(1)
    return _DIRECTION_BY_KIND.get(kind, "other"), kind


def _thread(event: dict[str, Any]) -> str:
    return f"pid={event.get('pid')}; tid={event.get('tid')}"


def _resolve_issuers(events, boundary_events, by_correlation, supported):
    """Classify every supported host issuing call against the mapped boundaries.

    Returns ``(kind_by_index, indices_by_kind, lookup)``: the outcome of each
    call, the calls grouped by outcome and the attributed lookup records.
    """
    kinds: dict[int, tuple[str, Any]] = {}
    lists: dict[str, list[int]] = {kind: [] for kind in ISSUE_KINDS}
    lookup: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if not supported(event):
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
                        "host_event_index": index,
                        "boundary_event_index": bindex,
                    }
                )
        kinds[index] = kind
        lists[kind[0]].append(index)
    return kinds, lists, lookup


def _resolve_device_event(index, event, events, by_correlation, supported, issuer_kinds, noun):
    """Resolve one device event through its correlation ID; never through timestamps.

    Returns ``(outcome, record)`` with outcome ``attributed``, ``unmatched`` or
    ``ambiguous``. ``noun`` names the issuing call in reasons (``launch`` or
    ``copy call``); ``index_key`` carries the host event under both names.
    """
    value = correlation_id(event)
    record: dict[str, Any] = {"correlation": value}
    if value is None:
        return "unmatched", {
            **record,
            "kind": "missing_correlation",
            "reason": "device event has no integer args.correlation; it cannot be tied to a "
            f"host {noun}",
        }
    hosts = by_correlation.get(value, [])
    if not hosts:
        return "unmatched", {
            **record,
            "kind": "no_host_api_event",
            "reason": f"no host CUDA API event in this trace carries correlation {value}",
        }
    if len(hosts) > 1:
        return "ambiguous", {
            **record,
            "kind": "conflicting_host_events",
            "host_event_indices": list(hosts),
            "reason": f"correlation {value} appears on several host CUDA API events "
            f"({summarize_indices(hosts)}); no candidate is chosen",
        }
    host_index = hosts[0]
    host = events[host_index]
    if not supported(host):
        return "unmatched", {
            **record,
            "host_event_index": host_index,
            "kind": "unsupported_host_event",
            "reason": f"correlated host event[{host_index}] is "
            f"{host.get('cat')}:{host.get('name')}, not a supported {noun} form",
        }
    kind, detail = issuer_kinds[host_index]
    if kind == "outside_mapped_step":
        return "unmatched", {
            **record,
            "host_event_index": host_index,
            "kind": "launch_outside_mapped_step",
            "reason": f"{noun} event[{host_index}] ({_thread(host)}) is not inside any "
            "mapped step boundary on its host thread",
        }
    if kind == "ambiguous_boundary":
        return "ambiguous", {
            **record,
            "host_event_index": host_index,
            "kind": "launch_in_several_boundaries",
            "candidate_training_steps": detail,
            "reason": f"{noun} event[{host_index}] lies inside several mapped step "
            f"boundaries (training steps {detail}); no step is chosen",
        }
    if kind != "attributed":  # pragma: no cover - duplicates were caught above
        raise ArtifactError(f"unexpected {noun} outcome {kind}")
    step, bindex = detail
    device = _device_id(event)
    if device is None:
        return "unmatched", {
            **record,
            "host_event_index": host_index,
            "kind": "missing_device",
            "reason": "device event has no integer args.device; its range cannot be placed on "
            "one device clock",
        }
    return "attributed", {
        **record,
        "host_event_index": host_index,
        "training_step": step,
        "boundary_event_index": bindex,
        "device": device,
        "stream": _stream_id(event),
    }


def _rename(record: dict[str, Any], key: str) -> dict[str, Any]:
    """Spell the issuing host event as ``launch_event_index`` or ``copy_event_index``."""
    if "host_event_index" in record and record.get("kind") != "unsupported_host_event":
        record = {key if k == "host_event_index" else k: v for k, v in record.items()}
    return record


def _kernel_class(event, host_index, events, collective_ops) -> tuple[str, str | None]:
    """``compute``, ``collective`` or ``collective_candidate`` with the reason for a candidate.

    ``host_index`` is the unique correlated host event, or ``None``. The op
    signal requires that host range to lie inside a ``c10d::`` op on the same
    thread; a kernel without a resolvable host event cannot show it.
    """
    name = event.get("name") or ""
    named = name.startswith(COLLECTIVE_KERNEL_PREFIXES)
    inside_op = host_index is not None and any(
        contains_range(op, events[host_index]) for op in collective_ops
    )
    if named and inside_op:
        return "collective", None
    if named:
        return "collective_candidate", (
            "NCCL kernel name without a correlated launch inside a c10d:: collective op; "
            "excluded from both the compute and the collective set"
        )
    if inside_op:
        return "collective_candidate", (
            "launched inside a c10d:: collective op but not named as an NCCL device kernel; "
            "excluded from both the compute and the collective set"
        )
    return "compute", None


def classify_device_events(
    trace: Trace, boundaries: list[tuple[int, int]], *, launches=SUPPORTED_LAUNCHES
) -> dict[str, Any]:
    """Resolve every kernel, copy, launch and copy call in one trace without timing.

    ``boundaries`` holds ``(training_step, boundary_event_index)`` pairs that
    the caller has already validated against the explicit step map. The result
    is plain JSON data (lists, dicts, ints, strings) so a persisted copy
    compares equal to a recomputation; validation re-derives exactly this
    classification and never re-times an event. ``launches`` is the supported
    launch set, overridable only to recheck a schema 1 block.
    """
    events = trace.events
    by_step = dict(boundaries)
    if len(by_step) != len(boundaries):
        raise ArtifactError(f"{trace.path.name}: duplicate training step among boundaries")
    boundary_events = [(step, index, events[index]) for step, index in boundaries]

    def supported_launch(event):
        return is_host_api_event(event) and (event.get("cat"), event.get("name")) in launches

    # 1. One correlation lookup over every host CUDA API range in this trace.
    by_correlation: dict[int, list[int]] = {}
    for index, event in enumerate(events):
        if is_host_api_event(event):
            value = correlation_id(event)
            if value is not None:
                by_correlation.setdefault(value, []).append(index)
    collective_ops = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == COLLECTIVE_OP_CATEGORY
        and (event.get("name") or "").startswith(COLLECTIVE_OP_PREFIX)
    ]

    # 2. Classify each supported launch and copy call against the boundaries.
    launch_kind, launch_lists, launch_lookup = _resolve_issuers(
        events, boundary_events, by_correlation, supported_launch
    )
    copy_kind, copy_lists, copy_lookup = _resolve_issuers(
        events, boundary_events, by_correlation, is_supported_copy
    )

    # 3. Resolve every kernel and copy through the lookup.
    attributed: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    classes = dict.fromkeys(KERNEL_CLASSES, 0)
    candidates: list[dict[str, Any]] = []
    unplaced: list[int] = []
    per_device: dict[int, dict[str, list[int]]] = {}
    for index, event in enumerate(events):
        if not is_kernel_event(event):
            continue
        outcome, record = _resolve_device_event(
            index, event, events, by_correlation, supported_launch, launch_kind, "launch"
        )
        value = correlation_id(event)
        hosts = by_correlation.get(value, []) if value is not None else []
        kernel_class, why = _kernel_class(
            event, hosts[0] if len(hosts) == 1 else None, events, collective_ops
        )
        classes[kernel_class] += 1
        if why is not None:
            candidates.append({"kernel_event_index": index, "reason": why})
        record = _rename({"kernel_event_index": index, **record}, "launch_event_index")
        record["class"] = kernel_class
        {"attributed": attributed, "unmatched": unmatched, "ambiguous": ambiguous}[outcome].append(
            record
        )
        device = _device_id(event)
        if device is None:
            unplaced.append(index)
        else:
            lists = per_device.setdefault(
                device, {"compute": [], "collective": [], "collective_candidate": []}
            )
            lists[kernel_class].append(index)

    transfers_attributed: list[dict[str, Any]] = []
    transfers_unmatched: list[dict[str, Any]] = []
    transfers_ambiguous: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if not is_transfer_event(event):
            continue
        outcome, record = _resolve_device_event(
            index, event, events, by_correlation, is_supported_copy, copy_kind, "copy call"
        )
        direction, kind = transfer_direction(event)
        record = _rename({"transfer_event_index": index, **record}, "copy_event_index")
        record.update(direction=direction, memcpy_kind=kind, bytes=_bytes(event))
        {
            "attributed": transfers_attributed,
            "unmatched": transfers_unmatched,
            "ambiguous": transfers_ambiguous,
        }[outcome].append(record)

    def counts(found, missing, several):
        return {
            "total": len(found) + len(missing) + len(several),
            "attributed": len(found),
            "unmatched": len(missing),
            "ambiguous": len(several),
        }

    def excluded(lists, key):
        return [
            {"kind": kind, key: indices}
            for kind, indices in lists.items()
            if kind != "attributed" and indices
        ]

    return {
        "kernel_events": counts(attributed, unmatched, ambiguous),
        "launch_events": {kind: len(indices) for kind, indices in launch_lists.items()},
        "launch_lookup": [_rename(item, "launch_event_index") for item in launch_lookup],
        "excluded_launches": excluded(launch_lists, "launch_event_indices"),
        "attributed": attributed,
        "unmatched": unmatched,
        "ambiguous": ambiguous,
        "kernel_classes": classes,
        "collective_candidates": candidates,
        "unplaced_kernel_event_indices": unplaced,
        "transfer_events": counts(transfers_attributed, transfers_unmatched, transfers_ambiguous),
        "copy_events": {kind: len(indices) for kind, indices in copy_lists.items()},
        "copy_lookup": [_rename(item, "copy_event_index") for item in copy_lookup],
        "excluded_copies": excluded(copy_lists, "copy_event_indices"),
        "transfers_attributed": transfers_attributed,
        "transfers_unmatched": transfers_unmatched,
        "transfers_ambiguous": transfers_ambiguous,
        "devices": [
            {
                "device": device,
                "compute_kernel_event_indices": lists["compute"],
                "collective_kernel_event_indices": lists["collective"],
                "candidate_kernel_event_indices": lists["collective_candidate"],
            }
            for device, lists in sorted(per_device.items())
        ],
    }


def _step_groups(classified: dict[str, Any]) -> dict[int, dict[int, dict[str, list]]]:
    """Attributed records grouped by training step, then device, in trace order."""
    groups: dict[int, dict[int, dict[str, list]]] = {}
    for label, key in (("kernels", "attributed"), ("transfers", "transfers_attributed")):
        for record in classified[key]:
            device = groups.setdefault(record["training_step"], {}).setdefault(
                record["device"], {"kernels": [], "transfers": []}
            )
            device[label].append(record)
    return groups


def _range(event: dict[str, Any]) -> tuple[float, float]:
    return event["ts"], event["ts"] + event["dur"]


def _union_ms(events) -> float:
    return interval_union_us(_range(e) for e in events) / 1000


def attribute_trace(trace: Trace, boundaries: list[tuple[int, int]]) -> dict[str, Any]:
    """Classify one trace, then union and intersect the attributed ranges.

    Devices are never combined: every figure belongs to one device clock.
    ``device_activity`` carries the window-level compute/collective unions and
    their intersection per device; ``steps`` carries the per-step unions.
    """
    classified = classify_device_events(trace, boundaries)
    events = trace.events
    by_step = dict(boundaries)
    compute_by_device = {
        entry["device"]: [events[i] for i in entry["compute_kernel_event_indices"]]
        for entry in classified["devices"]
    }
    activity = []
    for entry in classified["devices"]:
        compute = compute_by_device[entry["device"]]
        collective = [events[i] for i in entry["collective_kernel_event_indices"]]
        overlap = interval_intersection_us(map(_range, collective), map(_range, compute)) / 1000
        activity.append(
            {
                "device": entry["device"],
                "compute_kernel_events": len(compute),
                "collective_kernel_events": len(collective),
                "compute_kernel_ms": _union_ms(compute) if compute else None,
                "collective_kernel_ms": _union_ms(collective) if collective else None,
                "collective_compute_overlap_ms": overlap if collective else None,
                "overlapping_compute_kernel_event_indices": [
                    i
                    for i in entry["compute_kernel_event_indices"]
                    if interval_intersection_us([_range(events[i])], map(_range, collective))
                ]
                if collective
                else [],
            }
        )

    steps = []
    for step, devices in sorted(_step_groups(classified).items()):
        per_device = []
        for device, groups in sorted(devices.items()):
            kernels = [events[r["kernel_event_index"]] for r in groups["kernels"]]
            transfers = [events[r["transfer_event_index"]] for r in groups["transfers"]]
            classes = dict.fromkeys(KERNEL_CLASSES, 0)
            for record in groups["kernels"]:
                classes[record["class"]] += 1
            directions = {}
            for direction in TRANSFER_DIRECTIONS:
                chosen = [r for r in groups["transfers"] if r["direction"] == direction]
                sizes = [r["bytes"] for r in chosen]
                directions[direction] = {
                    "events": len(chosen),
                    "ms": _union_ms(events[r["transfer_event_index"]] for r in chosen)
                    if chosen
                    else None,
                    # Byte counts are summed only when every copy recorded one;
                    # no bandwidth is derived from them.
                    "bytes": sum(sizes) if chosen and all(s is not None for s in sizes) else None,
                }
            compute = compute_by_device.get(device, [])
            overlapping = (
                [
                    i
                    for i in next(
                        e["compute_kernel_event_indices"]
                        for e in classified["devices"]
                        if e["device"] == device
                    )
                    if interval_intersection_us([_range(events[i])], map(_range, transfers))
                ]
                if transfers and compute
                else []
            )
            per_device.append(
                {
                    "device": device,
                    "attributed_kernel_ms": _union_ms(kernels) if kernels else None,
                    "kernel_event_indices": [r["kernel_event_index"] for r in groups["kernels"]],
                    "launch_event_indices": sorted(
                        {r["launch_event_index"] for r in groups["kernels"]}
                    ),
                    "streams": sorted({str(r["stream"]) for r in groups["kernels"]}),
                    "kernel_classes": classes,
                    "attributed_transfer_ms": _union_ms(transfers) if transfers else None,
                    "transfer_event_indices": [
                        r["transfer_event_index"] for r in groups["transfers"]
                    ],
                    "copy_event_indices": sorted(
                        {r["copy_event_index"] for r in groups["transfers"]}
                    ),
                    "transfer_directions": directions,
                    # Compute kernels from any step count: the question is
                    # whether the device computed while it copied.
                    "transfer_compute_overlap_ms": (
                        interval_intersection_us(map(_range, transfers), map(_range, compute))
                        / 1000
                        if transfers and compute
                        else None
                    ),
                    "overlapping_compute_kernel_event_indices": overlapping,
                }
            )
        steps.append(
            {"training_step": step, "boundary_event_index": by_step[step], "per_device": per_device}
        )
    return {**classified, "device_activity": activity, "steps": steps}


def _trace_warnings(name: str, result: dict[str, Any], trace: Trace) -> list[str]:
    lines = []
    for label, noun, counts_key, found, several, excluded_key, index_key in (
        (
            "kernel",
            "kernel-launch",
            "kernel_events",
            result["unmatched"],
            result["ambiguous"],
            "excluded_launches",
            "launch_event_indices",
        ),
        (
            "transfer",
            "copy-call",
            "transfer_events",
            result["transfers_unmatched"],
            result["transfers_ambiguous"],
            "excluded_copies",
            "copy_event_indices",
        ),
    ):
        counts = result[counts_key]
        summary = (
            f"{name}: {label} attribution: {counts['attributed']}/{counts['total']} {label} "
            f"event(s) attributed through host {noun} correlation ids"
        )
        if counts["unmatched"]:
            summary += f"; {counts['unmatched']} unmatched"
        if counts["ambiguous"]:
            summary += f"; {counts['ambiguous']} ambiguous"
        lines.append(summary)
        key = f"{label}_event_index"
        for outcome, records in (("unmatched", found), ("ambiguous", several)):
            by_kind: dict[str, list[int]] = {}
            for record in records:
                by_kind.setdefault(record["kind"], []).append(record[key])
            for kind, indices in by_kind.items():
                lines.append(
                    f"{name}: {label} events {outcome} ({kind}); not attributed and not timed: "
                    + summarize_indices(indices)
                )
        for item in result[excluded_key]:
            indices = item[index_key]
            detail = ""
            if item["kind"] == "outside_mapped_step":
                threads = sorted({_thread(trace.events[i]) for i in indices})
                detail = f" on host thread(s) {', '.join(threads)}"
            lines.append(
                f"{name}: supported {noun} events {item['kind']}{detail}; {label}s they issued "
                "stay unmatched: " + summarize_indices(indices)
            )
    classes = result["kernel_classes"]
    if classes["collective"] or classes["collective_candidate"]:
        lines.append(
            f"{name}: kernel classes: {classes['compute']} compute, {classes['collective']} "
            f"collective (NCCL device kernel launched inside a c10d:: op), "
            f"{classes['collective_candidate']} collective candidate(s) in neither set"
            + (
                ": "
                + summarize_indices(
                    c["kernel_event_index"] for c in result["collective_candidates"]
                )
                if classes["collective_candidate"]
                else ""
            )
        )
    for entry in result["device_activity"]:
        if entry["collective_kernel_events"]:
            lines.append(
                f"{name}: device {entry['device']}: collective kernel union "
                f"{entry['collective_kernel_ms']} ms over the captured window, of which "
                f"{entry['collective_compute_overlap_ms']} ms coincided with compute kernels; "
                "window-level, not attributed to steps"
            )
    return lines


def _coverage(counts: dict[str, int], per_device: list, events_key: str, hosts_key: str) -> dict:
    unattributed = counts["unmatched"] + counts["ambiguous"]
    return {
        f"step_{events_key}": sum(len(d[f"{events_key[:-1]}_indices"]) for d in per_device),
        f"step_{hosts_key}": sum(len(d[f"{hosts_key[:-1]}_indices"]) for d in per_device),
        "devices": [d["device"] for d in per_device],
        f"trace_{events_key}": counts["total"],
        f"trace_attributed_{events_key}": counts["attributed"],
        f"trace_unattributed_{events_key}": unattributed,
        "completeness": (
            "none" if not per_device else ("complete" if unattributed == 0 else "partial")
        ),
    }


def _incomplete(counts: dict[str, int], label: str) -> str:
    unattributed = counts["unmatched"] + counts["ambiguous"]
    if unattributed:
        return (
            f"; attribution incomplete: {unattributed} of {counts['total']} {label} event(s) in "
            "this trace unattributed"
        )
    return f"; every {label} event in this trace was attributed to a mapped step"


def step_measurements(
    *,
    rank: int,
    training_step: int,
    trace_name: str,
    result: dict[str, Any],
    unavailable: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the three per-step measurements from one trace's attribution result.

    A value is published only when every attributed range of its kind sits on
    one device; several devices are kept apart in the attribution block and
    the measurement says so instead of adding clocks together. ``unavailable``
    maps a metric to the reason it is ``not_observed`` in this environment.
    """
    unavailable = unavailable or {}
    scope = f"rank={rank}; training_step={training_step}; trace={trace_name}"
    entry = next((s for s in result["steps"] if s["training_step"] == training_step), None)
    per_device = entry["per_device"] if entry else []
    kernel_devices = [d for d in per_device if d["kernel_event_indices"]]
    transfer_devices = [d for d in per_device if d["transfer_event_indices"]]
    kernel_counts, transfer_counts = result["kernel_events"], result["transfer_events"]
    out: dict[str, dict[str, Any]] = {}

    def withheld(metric, devices, coverage, none_reason):
        if metric in unavailable:
            value = measurement(
                None, status="not_observed", reason=unavailable[metric], scope=scope
            )
        elif not devices:
            value = measurement(None, reason=none_reason, scope=scope)
        else:
            value = measurement(
                None,
                reason=f"attributed ranges span devices {coverage['devices']}; per-device "
                "figures are kept separately in kernel_attribution and are not combined",
                scope=scope,
            )
        value["coverage"] = coverage
        return value

    def refs(indices):
        return [{"file": trace_name, "event_index": index} for index in indices]

    # Attributed kernel time.
    coverage = _coverage(kernel_counts, kernel_devices, "kernel_events", "launch_events")
    unattributed = kernel_counts["unmatched"] + kernel_counts["ambiguous"]
    if ATTRIBUTED_KERNEL_METRIC not in unavailable and len(kernel_devices) == 1:
        (device,) = kernel_devices
        value = measurement(
            device["attributed_kernel_ms"],
            basis=ATTRIBUTION_BASIS,
            evidence=refs(device["kernel_event_indices"]),
            scope=f"{scope}; device={device['device']}",
            reason=KERNEL_SCOPE_STATEMENT + _incomplete(kernel_counts, "kernel"),
        )
        value["coverage"] = coverage
    else:
        value = withheld(
            ATTRIBUTED_KERNEL_METRIC,
            kernel_devices,
            coverage,
            "no kernel event attributed to this step through host launch correlation; "
            + (
                f"{unattributed} of {kernel_counts['total']} kernel event(s) in this trace "
                "unattributed"
                if unattributed
                else f"{kernel_counts['total']} kernel event(s) in this trace"
            ),
        )
    out[ATTRIBUTED_KERNEL_METRIC] = value

    # Attributed transfer time, with direction breakdown in the coverage.
    coverage = _coverage(transfer_counts, transfer_devices, "transfer_events", "copy_events")
    coverage["directions"] = (
        transfer_devices[0]["transfer_directions"]
        if len(transfer_devices) == 1
        else {d: {"events": 0, "ms": None, "bytes": None} for d in TRANSFER_DIRECTIONS}
    )
    unattributed = transfer_counts["unmatched"] + transfer_counts["ambiguous"]
    if ATTRIBUTED_TRANSFER_METRIC not in unavailable and len(transfer_devices) == 1:
        (device,) = transfer_devices
        value = measurement(
            device["attributed_transfer_ms"],
            basis=ATTRIBUTION_BASIS,
            evidence=refs(device["transfer_event_indices"]),
            scope=f"{scope}; device={device['device']}",
            reason=TRANSFER_SCOPE_STATEMENT + _incomplete(transfer_counts, "transfer"),
        )
        value["coverage"] = coverage
    else:
        value = withheld(
            ATTRIBUTED_TRANSFER_METRIC,
            transfer_devices,
            coverage,
            "no gpu_memcpy event attributed to this step through host copy-call correlation; "
            + (
                f"{unattributed} of {transfer_counts['total']} transfer event(s) in this trace "
                "unattributed"
                if unattributed
                else f"{transfer_counts['total']} transfer event(s) in this trace"
            ),
        )
    out[ATTRIBUTED_TRANSFER_METRIC] = value

    # Transfer/compute overlap: needs the single-device transfer union and at
    # least one compute-class kernel on that device in the trace.
    transfer = out[ATTRIBUTED_TRANSFER_METRIC]
    activity = None
    if len(transfer_devices) == 1:
        activity = next(
            (a for a in result["device_activity"] if a["device"] == transfer_devices[0]["device"]),
            None,
        )
    coverage = {
        "devices": coverage["devices"],
        "transfer_ms": transfer["value_ms"],
        "overlap_ms": None,
        "exposed_ms": None,
        "trace_compute_kernel_events": activity["compute_kernel_events"] if activity else None,
        "trace_compute_kernel_ms": activity["compute_kernel_ms"] if activity else None,
        "trace_kernel_events": kernel_counts["total"],
        "completeness": coverage["completeness"],
    }
    if TRANSFER_OVERLAP_METRIC in unavailable:
        value = measurement(
            None, status="not_observed", reason=unavailable[TRANSFER_OVERLAP_METRIC], scope=scope
        )
    elif transfer["status"] != "recorded":
        value = measurement(
            None,
            reason="requires a recorded single-device attributed transfer union; "
            + transfer["reason"],
            scope=scope,
        )
    elif activity is None or not activity["compute_kernel_events"]:
        value = measurement(
            None,
            reason="no compute-class kernel event on this device in this trace; an overlap of "
            "zero would not distinguish idle compute from missing kernel collection",
            scope=transfer["scope"],
        )
    else:
        (device,) = transfer_devices
        overlap = device["transfer_compute_overlap_ms"]
        coverage.update(overlap_ms=overlap, exposed_ms=transfer["value_ms"] - overlap)
        value = measurement(
            overlap,
            basis=OVERLAP_BASIS,
            evidence=refs(
                device["transfer_event_indices"]
                + device["overlapping_compute_kernel_event_indices"]
            ),
            scope=transfer["scope"],
            reason=OVERLAP_SCOPE_STATEMENT + _incomplete(transfer_counts, "transfer"),
        )
    value["coverage"] = coverage
    out[TRANSFER_OVERLAP_METRIC] = value
    return out


def attribute_traces(
    traces: tuple[Trace, ...],
    rows: list[dict[str, Any]],
    *,
    rank: int,
    unavailable: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Attribute every trace and attach the per-step measurements to ``rows``.

    ``rows`` are the analysis step rows (with validated ``boundary``
    references) and are updated in place. Returns the persisted attribution
    block and warnings.
    """
    unavailable = unavailable or {}
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
            for metric in ATTRIBUTION_METRICS:
                row["measurements"][metric] = measurement(
                    None,
                    status="not_observed" if metric in unavailable else None,
                    reason=unavailable.get(metric, "no explicit mapped step boundary"),
                )
        else:
            name = row["boundary"]["file"]
            row["measurements"].update(
                step_measurements(
                    rank=rank,
                    training_step=row["training_step"],
                    trace_name=name,
                    result=results[name],
                    unavailable=unavailable,
                )
            )
    warnings.append(
        KERNEL_SCOPE_STATEMENT
        + "; complete per-step transfer, collective and busy-time metrics remain unavailable; "
        "attributed transfers, transfer/compute overlap and window-level collective/compute "
        "overlap are reported under their own names"
    )
    block = {
        "schema": ATTRIBUTION_SCHEMA,
        "method": METHOD,
        "supported_launch_events": [f"{cat}:{name}" for cat, name in SUPPORTED_LAUNCHES],
        "supported_copy_events": [f"{cat}:{name}" for cat, name in SUPPORTED_COPIES],
        "kernel_category": KERNEL_CATEGORY,
        "transfer_category": TRANSFER_CATEGORY,
        "collective_classification": {
            "kernel_name_prefixes": list(COLLECTIVE_KERNEL_PREFIXES),
            "host_op_prefix": COLLECTIVE_OP_PREFIX,
            "rule": "both signals required; one signal makes a candidate in neither set",
        },
        "scope": KERNEL_SCOPE_STATEMENT,
        "transfer_scope": TRANSFER_SCOPE_STATEMENT,
        "overlap_scope": OVERLAP_SCOPE_STATEMENT,
        "collective_scope": COLLECTIVE_SCOPE_STATEMENT,
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


def _optional_ms(value: Any) -> bool:
    return value is None or (finite_number(value) and value >= 0)


def _strip(records, keys):
    return [{k: v for k, v in r.items() if k not in keys} for r in records]


def _check_records(name, result, boundary_steps, *, event_key, host_key, found, missing, several):
    """Shape and relationship checks shared by kernel and transfer records."""
    by_event: dict[int, dict] = {}
    for record in result[found]:
        for key in (event_key, host_key, "boundary_event_index"):
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
            f"{name}: attributed {event_key[:-12]} event[{record.get(event_key)}] names a "
            "boundary that is not the mapped boundary of its training step",
        )
        _require(
            record[event_key] not in by_event,
            f"{name}: {event_key[:-12]} event[{record[event_key]}] attributed twice",
        )
        by_event[record[event_key]] = record
    for record in result[missing] + result[several]:
        _require(
            type(record.get(event_key)) is int
            and isinstance(record.get("reason"), str)
            and record["reason"]
            and record.get("kind") in UNMATCHED_KINDS + AMBIGUOUS_KINDS,
            f"{name}: unmatched/ambiguous records need {event_key}, kind and reason",
        )
        _require(
            record[event_key] not in by_event,
            f"{name}: {event_key[:-12]} event[{record[event_key]}] both attributed and not",
        )
    return by_event


def validate_attribution(
    payload: dict[str, Any],
    *,
    trace_names: set[str],
    boundaries: dict[str, list[tuple[int, int]]],
    load_trace,
) -> None:
    """Check the persisted block and its relationship to the step measurements.

    ``load_trace(name)`` returns a parsed :class:`Trace` or ``None`` when no
    source directory is available; with sources, the classification is
    recomputed from the trace and the stored copy must match exactly. Nothing
    is re-timed. Without sources the structure and the measurement/evidence
    relationships are checked. A schema 1 block (kernels only) is rechecked
    with the launch forms it was computed with and may not carry transfer or
    overlap measurements.
    """
    block = payload.get("kernel_attribution")
    rows = payload["steps"]
    present = {
        metric for row in rows for metric in ATTRIBUTION_METRICS if metric in row["measurements"]
    }
    if block is None:
        # Sidecars written before attribution existed carry neither the block
        # nor the metrics; a metric without its evidence block is refused.
        _require(
            not present,
            f"attributed device metric {min(present, default='')} requires a kernel_attribution "
            "block",
        )
        return
    _require(isinstance(block, dict), "expected object")
    schema = block.get("schema")
    _require(schema == ATTRIBUTION_SCHEMA or schema in LEGACY_SCHEMAS, "unsupported schema")
    legacy = schema != ATTRIBUTION_SCHEMA
    launches = _LEGACY_LAUNCHES if legacy else SUPPORTED_LAUNCHES
    _require(block.get("method") == METHOD, "unsupported method")
    _require(
        block.get("supported_launch_events") == [f"{c}:{n}" for c, n in launches]
        and block.get("kernel_category") == KERNEL_CATEGORY
        and (
            legacy
            or (
                block.get("supported_copy_events") == [f"{c}:{n}" for c, n in SUPPORTED_COPIES]
                and block.get("transfer_category") == TRANSFER_CATEGORY
            )
        ),
        "supported event forms disagree with this reader",
    )
    if legacy:
        _require(
            present <= {ATTRIBUTED_KERNEL_METRIC},
            "a schema 1 block supports device_attributed_kernel_ms only",
        )
    else:
        _require(
            not present or present == set(ATTRIBUTION_METRICS),
            "a schema 2 block carries all three attributed device metrics or none",
        )
    traces = block.get("traces")
    _require(isinstance(traces, dict) and set(traces) <= trace_names, "traces must name sources")
    boundary_steps = {
        (name, index): step for name, items in boundaries.items() for step, index in items
    }
    keys = _LEGACY_CLASSIFICATION_KEYS if legacy else _CLASSIFICATION_KEYS
    for name, result in traces.items():
        _require(isinstance(result, dict), f"{name}: expected object")
        _require(
            set(keys) | {"steps"} | (set() if legacy else {"device_activity"}) <= result.keys(),
            f"{name}: incomplete trace result",
        )
        for key in keys:
            if key in (
                "kernel_events",
                "launch_events",
                "kernel_classes",
                "transfer_events",
                "copy_events",
            ):
                _require(isinstance(result[key], dict), f"{name}: {key} must be an object")
            elif key == "unplaced_kernel_event_indices":
                _indices(result[key], f"{name}: {key}")
            else:
                _require(
                    isinstance(result[key], list) and all(isinstance(r, dict) for r in result[key]),
                    f"{name}: {key} must be a list of records",
                )
        kernels = _check_records(
            name,
            result,
            boundary_steps,
            event_key="kernel_event_index",
            host_key="launch_event_index",
            found="attributed",
            missing="unmatched",
            several="ambiguous",
        )
        transfers = (
            _check_records(
                name,
                result,
                boundary_steps,
                event_key="transfer_event_index",
                host_key="copy_event_index",
                found="transfers_attributed",
                missing="transfers_unmatched",
                several="transfers_ambiguous",
            )
            if not legacy
            else {}
        )
        if not legacy:
            for record in result["transfers_attributed"]:
                _require(
                    record.get("direction") in TRANSFER_DIRECTIONS,
                    f"{name}: transfer event[{record['transfer_event_index']}] has an unknown "
                    "direction label",
                )
            for record in result["attributed"] + result["unmatched"] + result["ambiguous"]:
                _require(
                    record.get("class") in KERNEL_CLASSES,
                    f"{name}: kernel event[{record.get('kernel_event_index')}] needs a class",
                )
            classes = {r["kernel_event_index"]: r["class"] for r in result["attributed"]}
            for record in result["unmatched"] + result["ambiguous"]:
                classes[record["kernel_event_index"]] = record["class"]
            devices = {}
            for entry in result["devices"]:
                _require(
                    isinstance(entry, dict) and type(entry.get("device")) is int,
                    f"{name}: devices entries need an integer device",
                )
                for key, label in (
                    ("compute_kernel_event_indices", "compute"),
                    ("collective_kernel_event_indices", "collective"),
                    ("candidate_kernel_event_indices", "collective_candidate"),
                ):
                    _indices(entry.get(key), f"{name}: {key}")
                    for index in entry[key]:
                        _require(
                            classes.get(index) == label,
                            f"{name}: device {entry['device']} lists kernel event[{index}] as "
                            f"{label} but its record says otherwise",
                        )
                devices[entry["device"]] = entry
            _require(isinstance(result["device_activity"], list), f"{name}: device_activity")
            for entry in result["device_activity"]:
                _require(
                    isinstance(entry, dict) and entry.get("device") in devices,
                    f"{name}: device_activity names a device without classified kernels",
                )
                listed = devices[entry["device"]]
                _require(
                    entry.get("compute_kernel_events")
                    == len(listed["compute_kernel_event_indices"])
                    and entry.get("collective_kernel_events")
                    == len(listed["collective_kernel_event_indices"])
                    and _optional_ms(entry.get("compute_kernel_ms"))
                    and _optional_ms(entry.get("collective_kernel_ms"))
                    and _optional_ms(entry.get("collective_compute_overlap_ms"))
                    and (
                        entry.get("collective_compute_overlap_ms") is None
                        or entry["collective_compute_overlap_ms"]
                        <= entry["collective_kernel_ms"] + 1e-9
                    ),
                    f"{name}: device {entry['device']} activity disagrees with its kernel lists",
                )
                _indices(
                    entry.get("overlapping_compute_kernel_event_indices"),
                    f"{name}: overlapping_compute_kernel_event_indices",
                )
                # An intersection is positive exactly when some compute kernel
                # intersects the collective union; the list of those kernels is
                # classification data and must agree with the figure.
                _require(
                    bool(entry.get("collective_compute_overlap_ms"))
                    == bool(entry["overlapping_compute_kernel_event_indices"]),
                    f"{name}: device {entry['device']} collective overlap disagrees with its "
                    "overlapping kernel list",
                )
                _require(
                    set(entry["overlapping_compute_kernel_event_indices"])
                    <= set(listed["compute_kernel_event_indices"]),
                    f"{name}: device {entry['device']} overlap cites a non-compute kernel",
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
                    _optional_ms(device.get("attributed_kernel_ms"))
                    and (
                        legacy
                        or (device["attributed_kernel_ms"] is None)
                        == (not device["kernel_event_indices"])
                    ),
                    f"{name}: attributed_kernel_ms must be finite and nonnegative",
                )
                for index in device["kernel_event_indices"]:
                    record = kernels.get(index)
                    _require(
                        record is not None
                        and record["training_step"] == entry.get("training_step")
                        and record["device"] == device.get("device"),
                        f"{name}: step {entry.get('training_step')} device "
                        f"{device.get('device')} cites kernel event[{index}] that is not "
                        "attributed to it",
                    )
                if legacy:
                    continue
                _indices(device.get("transfer_event_indices"), f"{name}: transfer_event_indices")
                _indices(device.get("copy_event_indices"), f"{name}: copy_event_indices")
                _require(
                    _optional_ms(device.get("attributed_transfer_ms"))
                    and (device["attributed_transfer_ms"] is None)
                    == (not device["transfer_event_indices"])
                    and _optional_ms(device.get("transfer_compute_overlap_ms"))
                    and (
                        device["transfer_compute_overlap_ms"] is None
                        or device["transfer_compute_overlap_ms"]
                        <= device["attributed_transfer_ms"] + 1e-9
                    ),
                    f"{name}: attributed_transfer_ms and transfer_compute_overlap_ms must be "
                    "finite, nonnegative and consistent",
                )
                _require(
                    bool(device["transfer_compute_overlap_ms"])
                    == bool(device.get("overlapping_compute_kernel_event_indices")),
                    f"{name}: step {entry.get('training_step')} transfer overlap disagrees with "
                    "its overlapping kernel list",
                )
                for index in device["transfer_event_indices"]:
                    record = transfers.get(index)
                    _require(
                        record is not None
                        and record["training_step"] == entry.get("training_step")
                        and record["device"] == device.get("device"),
                        f"{name}: step {entry.get('training_step')} device "
                        f"{device.get('device')} cites transfer event[{index}] that is not "
                        "attributed to it",
                    )
                _indices(
                    device.get("overlapping_compute_kernel_event_indices"),
                    f"{name}: overlapping_compute_kernel_event_indices",
                )
                _require(
                    all(
                        classes.get(i) == "compute"
                        for i in device["overlapping_compute_kernel_event_indices"]
                    ),
                    f"{name}: step {entry.get('training_step')} transfer overlap cites a "
                    "non-compute kernel",
                )
        trace = load_trace(name)
        if trace is not None:
            if legacy:
                expected = classify_device_events(
                    trace, boundaries.get(name, []), launches=launches
                )
                stored = {
                    key: _strip(result[key], ("class",))
                    if key in ("attributed", "unmatched", "ambiguous")
                    else result[key]
                    for key in keys
                }
                computed = {
                    key: _strip(expected[key], ("class",))
                    if key in ("attributed", "unmatched", "ambiguous")
                    else expected[key]
                    for key in keys
                }
            else:
                expected = classify_device_events(trace, boundaries.get(name, []))
                stored = {key: result[key] for key in keys}
                computed = {key: expected[key] for key in keys}
            _require(
                stored == computed,
                f"{name}: stored attribution disagrees with the trace and step map",
            )
            grouped = _step_groups(expected)
            stored_steps = {
                entry["training_step"]: {device["device"]: device for device in entry["per_device"]}
                for entry in result["steps"]
            }
            _require(
                stored_steps.keys() == grouped.keys()
                and all(stored_steps[step].keys() == grouped[step].keys() for step in grouped)
                and all(
                    stored_steps[step][device]["kernel_event_indices"]
                    == [r["kernel_event_index"] for r in records["kernels"]]
                    and stored_steps[step][device]["launch_event_indices"]
                    == sorted({r["launch_event_index"] for r in records["kernels"]})
                    and (
                        legacy
                        or (
                            stored_steps[step][device]["transfer_event_indices"]
                            == [r["transfer_event_index"] for r in records["transfers"]]
                            and stored_steps[step][device]["copy_event_indices"]
                            == sorted({r["copy_event_index"] for r in records["transfers"]})
                        )
                    )
                    for step, devices in grouped.items()
                    for device, records in devices.items()
                ),
                f"{name}: per-step kernel groups disagree with the attributed records",
            )

    # Each step measurement must be the one the block supports.
    for row in rows:
        for metric in ATTRIBUTION_METRICS:
            value = row["measurements"].get(metric)
            if value is None:
                continue
            step = row["training_step"]
            if row["boundary"] is None:
                _require(value["status"] != "recorded", f"training_step {step}: no boundary")
                continue
            coverage = value.get("coverage")
            _require(
                isinstance(coverage, dict)
                and coverage.get("completeness") in ("none", "partial", "complete"),
                f"training_step {step}: {metric} needs coverage",
            )
            name = row["boundary"]["file"]
            result = traces.get(name)
            _require(result is not None, f"training_step {step}: trace {name} missing")
            entry = next((s for s in result["steps"] if s.get("training_step") == step), None)
            per_device = entry["per_device"] if entry else []
            if metric == ATTRIBUTED_KERNEL_METRIC:
                devices = [d for d in per_device if d["kernel_event_indices"]]
                value_key, index_key, basis = (
                    "attributed_kernel_ms",
                    "kernel_event_indices",
                    ATTRIBUTION_BASIS,
                )
            elif metric == ATTRIBUTED_TRANSFER_METRIC:
                devices = [d for d in per_device if d["transfer_event_indices"]]
                value_key, index_key, basis = (
                    "attributed_transfer_ms",
                    "transfer_event_indices",
                    ATTRIBUTION_BASIS,
                )
            else:
                devices = [
                    d
                    for d in per_device
                    if d["transfer_event_indices"] and d["transfer_compute_overlap_ms"] is not None
                ]
                value_key, index_key, basis = (
                    "transfer_compute_overlap_ms",
                    "transfer_event_indices",
                    OVERLAP_BASIS,
                )
            if value["status"] == "recorded":
                _require(value["basis"] == basis, f"training_step {step}: {metric} needs {basis}")
                expected_indices = devices[0][index_key] if len(devices) == 1 else None
                if metric == TRANSFER_OVERLAP_METRIC and expected_indices is not None:
                    expected_indices = (
                        expected_indices + devices[0]["overlapping_compute_kernel_event_indices"]
                    )
                _require(
                    len(devices) == 1
                    and value["value_ms"] == devices[0][value_key]
                    and [ref["event_index"] for ref in value["evidence"]] == expected_indices
                    and all(ref["file"] == name for ref in value["evidence"])
                    and value["scope"].endswith(f"device={devices[0]['device']}"),
                    f"training_step {step}: recorded {metric} disagrees with the attribution block",
                )
            else:
                _require(
                    len(devices) != 1
                    or value["status"] == "not_observed"
                    or (
                        metric == TRANSFER_OVERLAP_METRIC
                        and row["measurements"][ATTRIBUTED_TRANSFER_METRIC]["status"] != "recorded"
                    ),
                    f"training_step {step}: attributed ranges exist but {metric} withholds them "
                    "without a not_observed reason",
                )
