"""Which of the PRD's report fields a loaded run actually contains.

PRD v2.0 section 5 lists the core report fields: ``step_time_ms``,
``data_wait_ms``, ``h2d_copy_ms``, ``gpu_activity``,
``collective_activity``, ``per_rank_step_ms``, ``suspected_bottleneck``,
``confidence_or_visibility``, ``evidence_refs``, ``profiled`` and
``run_manifest``. Today's artifact triple carries three of them. This
module answers, per field and *from the artifacts alone*, which of three
things is true:

``recorded``
    the run wrote it down, and the report shows the recorded value;
``not_recorded``
    nothing captured it -- usually because the capture or rule that would
    produce it is unimplemented, and in several cases because the decision
    defining it is still open (the detail names the GATE);
``not_observed``
    the run's own environment could not have produced it -- no CUDA device
    for kernel activity, fewer than two for collectives, no per-step
    host-to-device copy under device residency.

The distinction is the point. "Nobody measured this yet" and "this machine
cannot measure this" are different facts, and neither one is zero (PRD v2.0
section 8 "Guardrails"). Nothing here interprets a measurement, applies a
threshold or attributes an event: those are owner decisions (GATE 4.1, 4.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shardscope.artifacts import ENVIRONMENT_JSON, STEPS_CSV, SUMMARY_JSON, RunArtifacts
from shardscope.env import NOT_OBSERVED, distributed_capable

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

#: PRD v2.0 section 5 "Core report fields", in the order the PRD lists them.
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
    """Status for every PRD v2.0 section 5 core report field, in PRD order."""
    return tuple(_STATUS_BUILDERS[field](run) for field in REPORT_FIELDS)


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
        "No loader-wait measurement exists in this run. The bounded profiler "
        "capture that would produce one is unimplemented (GATE 2.1 schedule, "
        "GATE 2.2 activity set, GATE 2.3 marker names), and what the field means "
        "operationally -- time inside the load_data marker, or time the host "
        "blocks on the loader iterator -- is GATE 3.2, an open owner decision.",
    )


def _h2d_copy_ms(run: RunArtifacts) -> FieldStatus:
    residencies = sorted({row.data_residency for row in run.steps})
    if residencies == ["device"]:
        return FieldStatus(
            "h2d_copy_ms",
            NOT_OBSERVED,
            "This run used data_residency=device, which pre-places the whole "
            "dataset on the device, so no per-step host-to-device copy exists to "
            "measure. The field is structurally absent rather than small "
            "(GATE 1.2 keeps both residency modes available on purpose).",
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
        "profiler capture is unimplemented (GATE 2.1-2.3) and the per-step row "
        "schema that would carry it is GATE 3.1.",
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
        "observable in principle, but no trace was captured: the bounded capture "
        "is unimplemented (GATE 2.1-2.3) and its trace layout is GATE 2.4.",
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
            "observed, and PRD v2.0 section 8 forbids reporting distributed "
            "results from a host that did not run them.",
        )
    return FieldStatus(
        "collective_activity",
        NOT_RECORDED,
        f"This run recorded {count} CUDA device(s), but it is a single-device run "
        "with no collective capture: DDP tracing is unimplemented and hardware-gated "
        "(GATE 5.1 synchronization strategy, GATE 5.2 straggler "
        "injection). Overlapping collectives must never be summed into step time.",
    )


def _per_rank_step_ms(run: RunArtifacts) -> FieldStatus:
    return FieldStatus(
        "per_rank_step_ms",
        NOT_RECORDED,
        f"{STEPS_CSV} carries no rank column: this is a single-device run. "
        "Per-rank alignment arrives with the DDP block, which is unimplemented and "
        "hardware-gated.",
    )


def _suspected_bottleneck(run: RunArtifacts) -> FieldStatus:
    return FieldStatus(
        "suspected_bottleneck",
        NOT_RECORDED,
        "No diagnosis is attached to this run. The rule engine is unimplemented, "
        "and its thresholds are GATE 4.1 -- an owner decision. This report "
        "presents recorded measurements only; it does not attribute a cause.",
    )


def _confidence_or_visibility(run: RunArtifacts) -> FieldStatus:
    return FieldStatus(
        "confidence_or_visibility",
        NOT_RECORDED,
        "No confidence label is attached to this run. The vocabulary a rule may "
        "emit, and the evidence each level requires, is GATE 4.2 -- a "
        "an owner decision this report must not pre-empt by inventing one.",
    )


def _evidence_refs(run: RunArtifacts) -> FieldStatus:
    traces = [name for name in run.artifact_names if _looks_like_trace(name)]
    if traces:
        return FieldStatus(
            "evidence_refs",
            NOT_RECORDED,
            "Trace file(s) are present in the run directory -- "
            f"{', '.join(traces)} -- but this report does not read them: the trace "
            "parser is unimplemented and the event schema it would key off is "
            "GATE 2.3/3.1. They are listed so the evidence is not lost, not "
            "interpreted.",
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
            "profiler was attached is unknown for this run. Requirement SS-04 "
            "separates diagnostic from throughput figures, and an unknown status "
            "cannot be read as either one.",
        )
    if status:
        return FieldStatus(
            "profiled",
            RECORDED,
            "extra.profiled is true: these are diagnostic figures carrying "
            "profiler overhead. Requirement SS-04 forbids quoting them as "
            "throughput results.",
        )
    return FieldStatus(
        "profiled",
        RECORDED,
        "extra.profiled is false: no profiler was attached. These step times still "
        "come from a timed fixture run, not from the warm repeated benchmark "
        "harness (GATE 7.1, unimplemented), so they are not throughput results "
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
        f"the run's settings). Artifacts found: {', '.join(present)}. PRD v2.0 "
        "section 11 names run.json for this role; the producer writes "
        f"{ENVIRONMENT_JSON} today, and this report reads what exists rather than "
        "renaming it.",
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
        "(the bounded capture is unimplemented, GATE 2.1-2.3). Treating the "
        "absent inventory as zero devices would state a fact the run never "
        "recorded."
    )


def _json_bool(value: bool) -> str:
    """Spell a boolean the way the JSON artifact does, so quoting it matches."""
    return "true" if value else "false"


def _looks_like_trace(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith(_TRACE_PREFIX) or lowered.endswith(_TRACE_SUFFIXES)
