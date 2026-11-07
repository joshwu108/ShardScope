"""Describe report-field availability using recorded artifacts.

``recorded`` means the run contains the field. ``not_recorded`` means capture
or analysis did not produce it. ``not_observed`` means the recorded environment
or data residency could not produce the activity.

Absent measurements never become zero. This module reports availability only;
it does not infer bottlenecks, apply thresholds, or assign confidence levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shardscope.artifacts import (
    ENVIRONMENT_JSON,
    STEPS_CSV,
    SUMMARY_JSON,
    RunArtifacts,
    checked_analysis,
)
from shardscope.attribution import (
    ATTRIBUTED_KERNEL_METRIC,
    ATTRIBUTED_TRANSFER_METRIC,
    ATTRIBUTION_SCHEMA,
    TRANSFER_OVERLAP_METRIC,
)
from shardscope.env import NOT_OBSERVED, distributed_capable
from shardscope.rules import TRIGGERED_STATUSES

__all__ = [
    "NOT_OBSERVED",
    "NOT_RECORDED",
    "RECORDED",
    "REPORT_FIELDS",
    "FieldStatus",
    "field_statuses",
]

#: The run wrote the field down and the report shows what it wrote.
RECORDED = "recorded"

#: Nothing captured the field. Not zero, and not a statement about the host.
NOT_RECORDED = "not_recorded"

#: Stable display order for report-field availability.
REPORT_FIELDS = (
    "step_time_ms",
    "data_wait_ms",
    "h2d_copy_ms",
    "gpu_activity",
    "collective_activity",
    "per_rank_step_ms",
    "suspected_bottleneck",
    "confidence_or_visibility",
    "evidence_refs",
    "profiled",
    "run_manifest",
)

#: Suffixes that mark a file as trace evidence rather than a summary.
_TRACE_SUFFIXES = (".trace.json", ".trace.json.gz")
_TRACE_PREFIX = "trace"


@dataclass(frozen=True)
class FieldStatus:
    """One report field's availability in one run, with the reason."""

    field: str
    state: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"field": self.field, "state": self.state, "detail": self.detail}


def field_statuses(run: RunArtifacts) -> tuple[FieldStatus, ...]:
    """Availability of each report field in display order."""
    analysis = checked_analysis(run)
    statuses = {field: _STATUS_BUILDERS[field](run) for field in REPORT_FIELDS}
    if analysis is None:
        return tuple(statuses.values())
    for name in sorted({name for row in analysis["steps"] for name in row["measurements"]}):
        values = [
            row["measurements"][name] for row in analysis["steps"] if name in row["measurements"]
        ]
        recorded = sum(value["status"] == RECORDED for value in values)
        state = (
            RECORDED
            if recorded
            else (
                NOT_OBSERVED if all(v["status"] == NOT_OBSERVED for v in values) else NOT_RECORDED
            )
        )
        if name == ATTRIBUTED_KERNEL_METRIC:
            kind = _attribution_coverage(values)
        elif name == ATTRIBUTED_TRANSFER_METRIC:
            kind = _transfer_coverage(values)
        elif name == TRANSFER_OVERLAP_METRIC:
            kind = (
                "transfer/compute overlap: intersection of the step's attributed gpu_memcpy "
                "union with the compute-class kernel union on the same device and clock; "
                "unavailable, not zero, when no compute kernel was collected on the device. "
            )
        elif name.startswith("host_"):
            kind = "host ranges, not GPU execution or idle time. "
        else:
            kind = "device timing requires validated device evidence. "
        detail = (
            f"analysis.json: {recorded}/{len(values)} step(s) recorded in ms; "
            + kind
            + " ".join(dict.fromkeys(v["reason"] for v in values))
        )
        statuses[name] = FieldStatus(name, state, detail)
    for old, explicit in (
        ("data_wait_ms", "host_data_wait_ms"),
        ("h2d_copy_ms", "host_h2d_ms"),
        ("gpu_activity", "device_busy_ms"),
        ("collective_activity", "device_collective_ms"),
    ):
        if explicit in statuses:
            value = statuses[explicit]
            statuses[old] = FieldStatus(
                old, value.state, f"See explicit {explicit}: {value.detail}"
            )
    if ATTRIBUTED_TRANSFER_METRIC in statuses and "h2d_copy_ms" in statuses:
        current = statuses["h2d_copy_ms"]
        statuses["h2d_copy_ms"] = FieldStatus(
            current.field,
            current.state,
            current.detail
            + f" Device copy ranges attributed per step exist as {ATTRIBUTED_TRANSFER_METRIC} "
            "(direction in its coverage); they are device activity, not host copy-call time.",
        )
    block = analysis.get("kernel_attribution")
    if isinstance(block, dict) and block.get("schema") == ATTRIBUTION_SCHEMA:
        statuses["collective_activity"] = _collective_from_attribution(block)
    if ATTRIBUTED_KERNEL_METRIC in statuses and "gpu_activity" in statuses:
        # Total busy time stays unavailable; the attributed subset is a
        # separate, partial field and is pointed to rather than substituted.
        current = statuses["gpu_activity"]
        statuses["gpu_activity"] = FieldStatus(
            current.field,
            current.state,
            current.detail
            + f" A partial per-step kernel figure exists as {ATTRIBUTED_KERNEL_METRIC}; "
            "it is not total device busy time.",
        )
    refs = [
        ref
        for row in analysis["steps"]
        for value in row["measurements"].values()
        for ref in value["evidence"]
    ]
    statuses["evidence_refs"] = FieldStatus(
        "evidence_refs",
        RECORDED if refs else NOT_RECORDED,
        "analysis.json evidence uses source filename and original event-array index; "
        "source digests and targets checked.",
    )
    statuses["per_rank_step_ms"] = FieldStatus(
        "per_rank_step_ms",
        RECORDED,
        f"steps.csv durations for explicit rank {analysis['rank']}; "
        "no cross-machine clock alignment.",
    )
    possible = any(f["status"] in TRIGGERED_STATUSES for f in analysis["findings"])
    statuses["suspected_bottleneck"] = FieldStatus(
        "suspected_bottleneck",
        RECORDED if possible else NOT_RECORDED,
        (
            "Diagnostic rules are unconfigured; no threshold was supplied."
            if analysis["rule_config"] is None
            else "Persisted configured comparisons over host ranges; a triggered status "
            "(for example possible input waiting) does not establish GPU starvation. "
            "No calibrated confidence or confirmed root cause. See: shardscope findings."
        ),
    )
    return tuple(statuses.values())


def _attribution_coverage(values: list[dict[str, Any]]) -> str:
    """Spell attribution coverage from the persisted per-step coverage blocks."""
    by_trace: dict[str, tuple[int, int]] = {}
    for value in values:
        coverage = value.get("coverage")
        scope = value.get("scope", "")
        trace = next((part[6:] for part in scope.split("; ") if part.startswith("trace=")), None)
        if isinstance(coverage, dict) and trace is not None:
            by_trace[trace] = (
                coverage.get("trace_attributed_kernel_events", 0),
                coverage.get("trace_kernel_events", 0),
            )
    attributed = sum(a for a, _ in by_trace.values())
    total = sum(t for _, t in by_trace.values())
    return (
        "attributed kernel time: per-device interval union of kernel ranges tied to a step by "
        "host launch correlation; not utilization, step latency or total device busy time. "
        f"Coverage: {attributed}/{total} kernel event(s) attributed across "
        f"{len(by_trace)} analyzed trace(s). "
    )


def _transfer_coverage(values: list[dict[str, Any]]) -> str:
    by_trace: dict[str, tuple[int, int]] = {}
    directions: dict[str, int] = {}
    for value in values:
        coverage = value.get("coverage")
        scope = value.get("scope", "")
        trace = next((part[6:] for part in scope.split("; ") if part.startswith("trace=")), None)
        if isinstance(coverage, dict) and trace is not None:
            by_trace[trace] = (
                coverage.get("trace_attributed_transfer_events", 0),
                coverage.get("trace_transfer_events", 0),
            )
            for direction, entry in (coverage.get("directions") or {}).items():
                directions[direction] = directions.get(direction, 0) + entry.get("events", 0)
    attributed = sum(a for a, _ in by_trace.values())
    total = sum(t for _, t in by_trace.values())
    spelled = ", ".join(f"{k} {v}" for k, v in directions.items() if v) or "none"
    return (
        "attributed transfer time: per-device interval union of gpu_memcpy ranges tied to a "
        "step by host copy-call correlation; device activity, not host copy-call time, not "
        f"bandwidth. Coverage: {attributed}/{total} gpu_memcpy event(s) attributed across "
        f"{len(by_trace)} analyzed trace(s); attributed directions: {spelled}. "
    )


def _collective_from_attribution(block: dict[str, Any]) -> FieldStatus:
    """Window-level collective classification from the persisted attribution block."""
    kernels = 0
    candidates = 0
    devices = set()
    for name, result in block["traces"].items():
        for entry in result["devices"]:
            kernels += len(entry["collective_kernel_event_indices"])
            candidates += len(entry["candidate_kernel_event_indices"])
            if entry["collective_kernel_event_indices"]:
                devices.add((name, entry["device"]))
    if not kernels:
        return FieldStatus(
            "collective_activity",
            NOT_RECORDED,
            "analysis.json classified no kernel as a collective (NCCL device-kernel name plus a "
            f"launch inside a c10d:: op); {candidates} candidate(s) carried one signal only. "
            "Per-step collective time is not attributed in any case.",
        )
    return FieldStatus(
        "collective_activity",
        RECORDED,
        f"analysis.json: {kernels} collective kernel(s) classified on {len(devices)} "
        f"trace/device pair(s), with {candidates} candidate(s) excluded. Recorded as "
        "window-level per-device unions and their intersection with compute kernels in "
        "kernel_attribution.device_activity; not attributed to steps, and a collective "
        "kernel's duration includes waiting for peer ranks, so it is not network time.",
    )


def _step_time_ms(run: RunArtifacts) -> FieldStatus:
    policies = sorted({row.synchronized for row in run.steps})
    policy = (
        f"synchronized={policies[0]}"
        if len(policies) == 1
        else "rows disagree about the synchronize policy: " + ", ".join(map(str, policies))
    )
    return FieldStatus(
        "step_time_ms",
        RECORDED,
        f"{len(run.steps)} row(s) in {STEPS_CSV}, measured with {policy}. An "
        "unsynchronized timing measures kernel launch rather than execution, so "
        "it is only comparable with another run measured the same way.",
    )


def _data_wait_ms(run: RunArtifacts) -> FieldStatus:
    return FieldStatus(
        "data_wait_ms",
        NOT_RECORDED,
        "No persisted host loader-wait measurement exists in this run. The measurement contract "
        "uses explicit mapped host ranges; run offline analysis with a producer mapping.",
    )


def _h2d_copy_ms(run: RunArtifacts) -> FieldStatus:
    residencies = sorted({row.data_residency for row in run.steps})
    if residencies == ["device"]:
        return FieldStatus(
            "h2d_copy_ms",
            NOT_OBSERVED,
            "This run used data_residency=device, which pre-places the whole "
            "dataset on the device, so no per-step host-to-device copy exists to "
            "measure. This data residency makes the field structurally absent, "
            "rather than a measured duration of zero.",
        )
    if len(residencies) > 1:
        return FieldStatus(
            "h2d_copy_ms",
            NOT_RECORDED,
            "Rows in this run disagree about data residency "
            f"({', '.join(residencies)}), so a single copy figure would describe "
            "neither. Steps measured under different residency are not "
            "comparable; the per-row value is in " + STEPS_CSV + ".",
        )
    return FieldStatus(
        "h2d_copy_ms",
        NOT_RECORDED,
        "This run used data_residency=host, so a per-step host-to-device copy "
        "does happen, but nothing measured it separately from the step: the "
        "run has no persisted host or device copy measurement. Offline analysis requires "
        "explicit phase markers and step mapping.",
    )


def _gpu_activity(run: RunArtifacts) -> FieldStatus:
    inventory, problem = _device_inventory(run)
    if inventory is None:
        return FieldStatus("gpu_activity", NOT_RECORDED, _no_inventory_detail(problem))

    available, count = inventory
    if not available or count == 0:
        return FieldStatus(
            "gpu_activity",
            NOT_OBSERVED,
            f"This run's {ENVIRONMENT_JSON} records cuda_available="
            f"{_json_bool(available)} and {count} CUDA device(s), so no CUDA kernel "
            "activity could have been observed while it ran. Reporting a number "
            "here would describe a device that was not used.",
        )
    return FieldStatus(
        "gpu_activity",
        NOT_RECORDED,
        f"This run recorded {count} CUDA device(s), so kernel activity was "
        "observable in principle, but no persisted device timing is attached. Device events "
        "require validated activity classification and step attribution.",
    )


def _collective_activity(run: RunArtifacts) -> FieldStatus:
    inventory, problem = _device_inventory(run)
    if inventory is None:
        return FieldStatus("collective_activity", NOT_RECORDED, _no_inventory_detail(problem))

    available, count = inventory
    if not distributed_capable(cuda_available=available, device_count=count):
        return FieldStatus(
            "collective_activity",
            NOT_OBSERVED,
            f"This run's {ENVIRONMENT_JSON} records {count} CUDA device(s); a real "
            "DDP/NCCL job needs at least two. No collective could have been "
            "observed. Distributed measurements require "
            "a host that actually ran the distributed workload.",
        )
    return FieldStatus(
        "collective_activity",
        NOT_RECORDED,
        f"This run recorded {count} CUDA device(s), but no persisted collective measurement "
        "is attached. Device inventory does not establish collective execution. "
        "Overlapping collectives must never be summed into step time.",
    )


def _per_rank_step_ms(run: RunArtifacts) -> FieldStatus:
    return FieldStatus(
        "per_rank_step_ms",
        NOT_RECORDED,
        f"{STEPS_CSV} carries no rank column and no validated analysis supplies explicit rank "
        "identity. No cross-rank measurement is inferred from directory names; compare the "
        "ranks of one launch with 'shardscope ranks --root DIR', which writes ranks.json "
        "beside the rank directories.",
    )


def _suspected_bottleneck(run: RunArtifacts) -> FieldStatus:
    return FieldStatus(
        "suspected_bottleneck",
        NOT_RECORDED,
        "No persisted diagnosis is attached to this run. Explicit thresholds and resolving "
        "host-wait evidence are required; this report does not run analysis.",
    )


def _confidence_or_visibility(run: RunArtifacts) -> FieldStatus:
    return FieldStatus(
        "confidence_or_visibility",
        NOT_RECORDED,
        "No calibrated confidence label is attached to this run. Availability and explicit "
        "configured comparisons do not establish a confidence score or confirmed cause.",
    )


def _evidence_refs(run: RunArtifacts) -> FieldStatus:
    traces = [name for name in run.artifact_names if _looks_like_trace(name)]
    if traces:
        return FieldStatus(
            "evidence_refs",
            NOT_RECORDED,
            "Trace file(s) are present in the run directory -- "
            f"{', '.join(traces)} -- but this report does not read them for new timings. "
            "No validated analysis.json is attached. Listing a trace does not imply "
            "it has been analyzed.",
        )
    return FieldStatus(
        "evidence_refs",
        NOT_RECORDED,
        "No trace artifact is present in this run directory, so no claim in this "
        f"report can cite one. Files found: {', '.join(run.artifact_names) or 'none'}.",
    )


def _profiled(run: RunArtifacts) -> FieldStatus:
    status = run.profiled
    if status is None:
        return FieldStatus(
            "profiled",
            NOT_RECORDED,
            f"extra.profiled is absent from {ENVIRONMENT_JSON}, so whether the "
            "profiler was attached is unknown for this run. These timings cannot "
            "be classified as diagnostic or throughput measurements.",
        )
    if status:
        return FieldStatus(
            "profiled",
            RECORDED,
            "extra.profiled is true: these diagnostic timings include profiler "
            "overhead and cannot be quoted as throughput results.",
        )
    return FieldStatus(
        "profiled",
        RECORDED,
        "extra.profiled is false: no profiler was attached. These step times still "
        "come from a timed fixture run, not from the warm repeated benchmark "
        "harness (benchmark methodology not implemented), so they are not throughput results "
        "either.",
    )


def _run_manifest(run: RunArtifacts) -> FieldStatus:
    present = [
        name for name in (ENVIRONMENT_JSON, SUMMARY_JSON, STEPS_CSV) if name in run.artifact_names
    ]
    return FieldStatus(
        "run_manifest",
        RECORDED,
        f"Provenance for this run is {ENVIRONMENT_JSON} (a RunEnvironment dump plus "
        f"the run's settings). Artifacts found: {', '.join(present)}.",
    )


_STATUS_BUILDERS = {
    "step_time_ms": _step_time_ms,
    "data_wait_ms": _data_wait_ms,
    "h2d_copy_ms": _h2d_copy_ms,
    "gpu_activity": _gpu_activity,
    "collective_activity": _collective_activity,
    "per_rank_step_ms": _per_rank_step_ms,
    "suspected_bottleneck": _suspected_bottleneck,
    "confidence_or_visibility": _confidence_or_visibility,
    "evidence_refs": _evidence_refs,
    "profiled": _profiled,
    "run_manifest": _run_manifest,
}


def _device_inventory(run: RunArtifacts) -> tuple[tuple[bool, int] | None, str]:
    """Read ``(cuda_available, device_count)`` out of the recorded provenance.

    Returns ``(None, problem)`` when either key is absent or unreadable.
    Defaulting them to ``false``/``0`` would make the report *state* that the
    run saw no CUDA device -- a fact the artifact never recorded. That is the
    same mistake as turning an absent measurement into zero, so an
    unreadable inventory stays unreadable and says which key is at fault.
    """
    environment = run.environment
    available: Any = environment.get("cuda_available")
    count: Any = environment.get("device_count")

    if available is None:
        return None, f"'cuda_available' is absent from {ENVIRONMENT_JSON}"
    if not isinstance(available, bool):
        return None, f"'cuda_available' is {available!r}, not true or false"
    if count is None:
        return None, f"'device_count' is absent from {ENVIRONMENT_JSON}"
    if isinstance(count, bool) or not isinstance(count, int):
        return None, f"'device_count' is {count!r}, not an integer"
    if count < 0:
        return None, f"'device_count' is {count}, which is not a device count"
    return (available, count), ""


def _no_inventory_detail(problem: str) -> str:
    return (
        f"This run did not record a readable CUDA device inventory: {problem}. "
        "Whether this activity could have been observed therefore cannot be "
        "judged from the run's own provenance, and nothing captured it either "
        "(no persisted device measurement is attached). Treating the "
        "absent inventory as zero devices would state a fact the run never "
        "recorded."
    )


def _json_bool(value: bool) -> str:
    """Spell a boolean the way the JSON artifact does, so quoting it matches."""
    return "true" if value else "false"


def _looks_like_trace(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith(_TRACE_PREFIX) or lowered.endswith(_TRACE_SUFFIXES)
