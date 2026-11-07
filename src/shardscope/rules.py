"""Diagnostic rules: explicit threshold comparisons over persisted measurements.

Each rule reads validated measurements from an analysis payload and compares
them against thresholds the caller chose. Nothing is calibrated, defaulted,
scored or ranked:

* A missing measurement is never zero; a rule that lacks an input says so.
* ``below_threshold`` means only that the configured comparison did not fire.
  It does not establish that the workload is healthy.
* A triggered status is a restrained observation with a next place to look,
  not a root cause, a confidence score or an expected speedup.

Two registries describe the rules. :data:`RULES` are evaluated by
``shardscope analyze`` over one rank's ``analysis.json``; :data:`RANK_RULES`
are evaluated by ``shardscope ranks`` over the per-rank recordings of one DDP
launch. ``shardscope rules`` prints both, so the same text answers "what can be
evaluated?" and "what did this finding mean?".

Adding a rule:
    1. Add a :class:`RuleSpec` to :data:`RULES` (or :data:`RANK_RULES`) and a
       config field to :class:`RuleConfig` (validated in ``__post_init__``).
    2. Write an evaluator ``(steps, config, context) -> list[finding]`` and
       register it in ``_EVALUATORS``. Use :func:`_recorded` to accept only
       recorded, finite, evidenced measurements with the right basis.
    3. Cover triggered, below-threshold, exact-boundary and missing-evidence
       cases in ``tests/test_rules.py``.

Compatibility: ``host_loader_wait`` emits the finding shape it always has, and
a sidecar written before the registry existed (no ``rules`` key) is
re-evaluated with :data:`LEGACY_RULES`, so saved runs keep loading.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

from shardscope.attribution import (
    ATTRIBUTED_TRANSFER_METRIC,
    ATTRIBUTION_BASIS,
    ATTRIBUTION_SCHEMA,
    OVERLAP_BASIS,
    TRANSFER_OVERLAP_METRIC,
)
from shardscope.env import NOT_OBSERVED
from shardscope.errors import ArtifactError
from shardscope.parse import finite_number

__all__ = [
    "LEGACY_RULES",
    "NOT_OBSERVED",
    "RANK_RULES",
    "RANK_RULE_IDS",
    "RULE_ID",
    "RULE_IDS",
    "RULES",
    "TRIGGERED_STATUSES",
    "RuleConfig",
    "RuleSpec",
    "describe_rules",
    "evaluate_rank_rules",
    "evaluate_rules",
    "parse_threshold",
    "rule_keys",
]

#: The original rule; kept as a module constant because callers import it.
RULE_ID = "host_loader_wait"
H2D_SHARE_RULE = "host_h2d_share"
VARIABILITY_RULE = "step_duration_variability"
TRANSFER_OVERLAP_RULE = "device_transfer_overlap"
COLLECTIVE_OVERLAP_RULE = "collective_compute_overlap"
RANK_IMBALANCE_RULE = "rank_step_imbalance"

#: Sidecars without a ``rules`` key were written when this was the whole set.
LEGACY_RULES: tuple[str, ...] = (RULE_ID,)

#: The definitional floor for a spread statistic: a median of one value has no
#: spread. This is a requirement of the arithmetic, not a calibrated setting;
#: callers raise it with ``step_variability_min_steps``.
MIN_STEPS_FLOOR = 2


@dataclass(frozen=True)
class RuleSpec:
    """Static description of one rule, as printed by ``shardscope rules``.

    Attributes:
        rule_id: Stable identifier persisted in findings.
        description: One-paragraph plain description of the comparison.
        scope: What is compared, over which steps, on which rank.
        required_measurements: Measurement names the rule needs recorded.
        config: One dict per threshold key: ``key``, ``unit``, ``type``,
            ``range``, ``compares``.
        statuses: Status name to meaning.
        interpretation: Text attached to a triggered finding.
        next_investigation: Where to look next when the rule fires.
        granularity: ``"per training step"``, ``"per analysis"`` or similar.
        evaluated_by: The command that evaluates the rule.
    """

    rule_id: str
    description: str
    scope: str
    required_measurements: tuple[str, ...]
    config: tuple[dict[str, Any], ...]
    statuses: dict[str, str]
    interpretation: str
    next_investigation: str
    granularity: str
    evaluated_by: str = "shardscope analyze"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "granularity": self.granularity,
            "evaluated_by": self.evaluated_by,
            "scope": self.scope,
            "required_measurements": list(self.required_measurements),
            "config": [dict(item) for item in self.config],
            "statuses": dict(self.statuses),
            "interpretation": self.interpretation,
            "next_investigation": self.next_investigation,
        }


_COMMON_STATUSES = {
    "unconfigured": "no threshold was supplied for this rule; nothing was compared",
    "unavailable": "a required measurement is not recorded for the step; not measured zero",
    "below_threshold": "the configured comparison did not fire; this does not establish health",
}
_SHARE_RANGE = "finite, 0 <= value <= 1"

RULES: tuple[RuleSpec, ...] = (
    RuleSpec(
        rule_id=RULE_ID,
        granularity="per training step",
        description=(
            "Host time blocked waiting for the next batch (the shardscope/data_wait host "
            "range) meets a caller-chosen number of milliseconds."
        ),
        scope=(
            "one rank; each mapped training step with a complete shardscope/data_wait host "
            "marker inside its step boundary on the same thread"
        ),
        required_measurements=("host_data_wait_ms",),
        config=(
            {
                "key": "loader_wait_threshold_ms",
                "unit": "ms",
                "type": "number",
                "range": "finite, >= 0",
                "compares": "host_data_wait_ms >= loader_wait_threshold_ms",
            },
        ),
        statuses={
            **_COMMON_STATUSES,
            "possible_input_wait": "host wait meets the threshold; possible input waiting",
        },
        interpretation=(
            "A host wait can suggest the input pipeline did not have the next batch ready. "
            "It is host time, not GPU idle time, and does not establish GPU starvation."
        ),
        next_investigation=(
            "Inspect the loader: worker count, prefetching, per-sample transform cost, "
            "and whether the wait is uniform across steps or periodic."
        ),
    ),
    RuleSpec(
        rule_id=H2D_SHARE_RULE,
        granularity="per training step",
        description=(
            "Host time spent inside host-to-device copy calls (shardscope/h2d) as a share "
            "of the host step range (shardscope/step) meets a caller-chosen fraction."
        ),
        scope=(
            "one rank; each mapped training step with both host_h2d_ms and host_step_ms "
            "recorded from the same trace, process and thread. Numerator and denominator "
            "are host interval unions; the h2d ranges lie inside the step range, so the "
            "share is in [0, 1]"
        ),
        required_measurements=("host_h2d_ms", "host_step_ms"),
        config=(
            {
                "key": "h2d_share_threshold",
                "unit": "fraction of host_step_ms",
                "type": "number",
                "range": _SHARE_RANGE,
                "compares": "host_h2d_ms / host_step_ms >= h2d_share_threshold",
            },
        ),
        statuses={
            **_COMMON_STATUSES,
            "ambiguous": (
                "inputs are recorded but the share is undefined or inconsistent (zero-length "
                "step range, different scopes, or a copy union longer than its step)"
            ),
            "possible_host_copy_wait": (
                "the host spent at least the configured share of the step inside copy calls"
            ),
        },
        interpretation=(
            "A large share means the host thread was inside copy calls for much of the step. "
            "That is host call time: a synchronous pageable copy blocks the host for the "
            "transfer, a pinned non_blocking copy returns at once. It is not device transfer "
            "time, bandwidth, or an overlap measurement."
        ),
        next_investigation=(
            "Check whether batches are pinned and copied with non_blocking=True, and read "
            "the gpu_memcpy events for these steps in the source trace."
        ),
    ),
    RuleSpec(
        rule_id=VARIABILITY_RULE,
        granularity="one finding per analysis (all mapped steps on the rank)",
        description=(
            "The slowest mapped step's host step range divided by the median host step "
            "range, across the recorded steps on this rank, meets a caller-chosen ratio."
        ),
        scope=(
            "one rank; every mapped training step with host_step_ms recorded. Durations are "
            "host ranges from one process and may come from several profiler cycles. Steps "
            "are only compared when they recorded the same set of host phases, as a check "
            "that they did equivalent logical work; the rule cannot see microbatch counts"
        ),
        required_measurements=("host_step_ms",),
        config=(
            {
                "key": "step_variability_ratio",
                "unit": "ratio (max host_step_ms / median host_step_ms)",
                "type": "number",
                "range": "finite, >= 1",
                "compares": "max(host_step_ms) / median(host_step_ms) >= step_variability_ratio",
            },
            {
                "key": "step_variability_min_steps",
                "unit": "count of recorded steps",
                "type": "integer",
                "range": f">= {MIN_STEPS_FLOOR}; optional, {MIN_STEPS_FLOOR} when omitted",
                "compares": "recorded steps >= step_variability_min_steps, else insufficient",
            },
        ),
        statuses={
            "unconfigured": _COMMON_STATUSES["unconfigured"],
            "unavailable": "no mapped step recorded host_step_ms",
            "insufficient": "fewer recorded steps than required; no statistic is formed",
            "ambiguous": (
                "recorded steps differ in which host phases they recorded, so they are not "
                "known to be equivalent work; nothing is compared"
            ),
            "below_threshold": _COMMON_STATUSES["below_threshold"],
            "possible_step_variability": (
                "the slowest step is at least the configured ratio of the median"
            ),
        },
        interpretation=(
            "A high ratio means at least one step took much longer than the typical step on "
            "this rank. The ratio names the slowest step; it does not say why it was slow, "
            "and it does not compare ranks."
        ),
        next_investigation=(
            "Look at the slowest step's phase measurements and warnings, and compare its "
            "host_data_wait_ms and host_h2d_ms with the median step's."
        ),
    ),
    RuleSpec(
        rule_id=TRANSFER_OVERLAP_RULE,
        granularity="per training step",
        description=(
            "The part of a step's attributed device transfer time during which no compute "
            "kernel ran on the same device, as a share of that transfer time, meets a "
            "caller-chosen fraction."
        ),
        scope=(
            "one rank; each mapped training step with device_attributed_transfer_ms and "
            "device_transfer_compute_overlap_ms recorded on one device in one trace. The "
            "transfer union holds the gpu_memcpy ranges whose copy call lay inside the step "
            "boundary on the launching thread (any direction; the coverage block splits "
            "h2d, d2h and other); the overlap is its intersection with every compute-class "
            "kernel on that device in the trace, whichever step launched it. Copies issued "
            "from other host threads are not in the union and are listed in "
            "kernel_attribution"
        ),
        required_measurements=(ATTRIBUTED_TRANSFER_METRIC, TRANSFER_OVERLAP_METRIC),
        config=(
            {
                "key": "transfer_exposed_share",
                "unit": "fraction of device_attributed_transfer_ms",
                "type": "number",
                "range": _SHARE_RANGE,
                "compares": (
                    "(device_attributed_transfer_ms - device_transfer_compute_overlap_ms) / "
                    "device_attributed_transfer_ms >= transfer_exposed_share"
                ),
            },
        ),
        statuses={
            **_COMMON_STATUSES,
            "ambiguous": (
                "inputs are recorded but the share is undefined or inconsistent (zero-length "
                "transfer union, different scopes, or an overlap longer than the union)"
            ),
            "possible_exposed_transfer": (
                "at least the configured share of the step's attributed copy time ran with no "
                "compute kernel on the device"
            ),
        },
        interpretation=(
            "Exposed transfer time is device copy time during which no compute kernel in this "
            "trace ran on the same device. It is device activity on one clock, not host "
            "copy-call time (host_h2d_ms), not bandwidth and not a share of the step. Copies "
            "on the compute stream serialize with kernels, so a pageable copy of the batch at "
            "step start is typically fully exposed; a partial attribution coverage means other "
            "copies in the trace were not placed in any step."
        ),
        next_investigation=(
            "Check whether the batch is pinned and copied with non_blocking=True on a side "
            "stream, whether the next batch could be prefetched during compute, and compare "
            "host_h2d_ms with device_attributed_transfer_ms for these steps."
        ),
    ),
    RuleSpec(
        rule_id=COLLECTIVE_OVERLAP_RULE,
        granularity="one finding per analyzed trace and device with collective kernels",
        description=(
            "The part of a device's collective kernel time during which no compute kernel ran "
            "on the same device, as a share of that collective time over the captured "
            "window, meets a caller-chosen fraction."
        ),
        scope=(
            "one rank; each analyzed trace and device whose kernel_attribution device_activity "
            "lists kernels classified collective (NCCL device-kernel name plus a launch "
            "nested in a c10d:: op). Collective and compute unions are taken over the whole "
            "trace window on one device clock; they are not attributed to training steps, "
            "because collectives are launched from a host thread without a mapped boundary. "
            "Kernels with only one of the two signals are candidates in neither set"
        ),
        required_measurements=(
            "kernel_attribution.traces[*].device_activity[*].collective_kernel_ms",
            "kernel_attribution.traces[*].device_activity[*].collective_compute_overlap_ms",
        ),
        config=(
            {
                "key": "collective_exposed_share",
                "unit": "fraction of collective_kernel_ms",
                "type": "number",
                "range": _SHARE_RANGE,
                "compares": (
                    "(collective_kernel_ms - collective_compute_overlap_ms) / "
                    "collective_kernel_ms >= collective_exposed_share"
                ),
            },
        ),
        statuses={
            "unconfigured": _COMMON_STATUSES["unconfigured"],
            "unavailable": (
                "no analyzed trace classified a collective kernel, or the sidecar predates "
                "collective classification; not measured zero"
            ),
            "ambiguous": (
                "a collective union of zero length or an overlap longer than the union; the "
                "share is undefined"
            ),
            "below_threshold": _COMMON_STATUSES["below_threshold"],
            "possible_exposed_collective": (
                "at least the configured share of the device's collective kernel time ran with "
                "no compute kernel on the device"
            ),
        },
        interpretation=(
            "Exposed collective time is NCCL kernel time on this device during which no "
            "compute kernel ran, over the captured window as a whole. A collective kernel's "
            "duration includes waiting for peer ranks, so it is not network transfer time and "
            "does not blame the network or name a straggler; it is not attributed to steps "
            "and is not a communication tail measurement."
        ),
        next_investigation=(
            "Compare the same window on the other ranks, check whether the allreduce runs "
            "during backward compute (DDP bucket size, gradient_as_bucket_view) and read the "
            "collective kernel ranges in the source trace next to the compute kernels."
        ),
    ),
)

#: Cross-rank rules; evaluated by ``shardscope ranks`` over one DDP root.
RANK_RULES: tuple[RuleSpec, ...] = (
    RuleSpec(
        rule_id=RANK_IMBALANCE_RULE,
        granularity="per training step across every rank of one launch",
        evaluated_by="shardscope ranks",
        description=(
            "For one training step, the longest recorded step time among the ranks divided "
            "by the shortest meets a caller-chosen ratio."
        ),
        scope=(
            "one torchrun launch; every rank directory under the root, aligned by "
            "training-step index. Durations are each rank's own steps.csv step_time_ms, "
            "measured on that rank's host clock with device synchronization at both ends; "
            "no cross-machine clock alignment is assumed and no timestamp is compared. Ranks "
            "must record compatible configuration and provenance (same source, workload, "
            "schedule, world size, versions); a step is compared only when every rank "
            "recorded it synchronized, under the same data residency and, when every rank "
            "has a validated analysis.json, with the same set of host phases"
        ),
        required_measurements=("steps.csv step_time_ms on every rank",),
        config=(
            {
                "key": "rank_imbalance_ratio",
                "unit": "ratio (max step_time_ms / min step_time_ms over ranks)",
                "type": "number",
                "range": "finite, >= 1",
                "compares": "max(step_time_ms) / min(step_time_ms) >= rank_imbalance_ratio",
            },
        ),
        statuses={
            "unconfigured": _COMMON_STATUSES["unconfigured"],
            "incompatible": (
                "the rank directories do not record compatible configuration or provenance; "
                "nothing is compared"
            ),
            "incomplete": (
                "a rank directory is missing, or the step was not recorded on every rank; "
                "nothing is compared for it"
            ),
            "ambiguous": (
                "the step is recorded on every rank but not as equivalent synchronized work "
                "(unsynchronized rows, differing residency or host phase sets, zero minimum)"
            ),
            "below_threshold": _COMMON_STATUSES["below_threshold"],
            "possible_rank_imbalance": (
                "the ranks' recorded step times for this step differ by at least the ratio"
            ),
        },
        interpretation=(
            "The ranks took different amounts of time for the same training step. A longer "
            "step time does not identify a straggler: with synchronized DDP steps a fast rank "
            "spends the difference waiting inside the collective, so the rank with the "
            "longest recorded step may be the one that waited. The ratio compares durations "
            "only, never timestamps across machines."
        ),
        next_investigation=(
            "Analyze each rank's traces and compare the per-step host phases "
            "(host_backward_ms, host_data_wait_ms) and the window-level collective overlap "
            "across ranks; check the input partition and per-rank batch sizes."
        ),
    ),
)

RULE_IDS: tuple[str, ...] = tuple(spec.rule_id for spec in RULES)
RANK_RULE_IDS: tuple[str, ...] = tuple(spec.rule_id for spec in RANK_RULES)
_BY_ID = {spec.rule_id: spec for spec in (*RULES, *RANK_RULES)}

#: Statuses that mean "the configured comparison fired", per rule.
TRIGGERED_STATUSES: frozenset[str] = frozenset(
    {
        "possible_input_wait",
        "possible_host_copy_wait",
        "possible_step_variability",
        "possible_exposed_transfer",
        "possible_exposed_collective",
        "possible_rank_imbalance",
    }
)


def rule_keys(rule_ids: Sequence[str]) -> tuple[str, ...]:
    """Threshold keys owned by ``rule_ids``, in registry order."""
    return tuple(item["key"] for rule in rule_ids for item in _BY_ID[rule].config)


def _share(value: Any, name: str) -> None:
    if value is not None and (not finite_number(value) or not 0 <= value <= 1):
        raise ArtifactError(f"{name} must be a finite fraction between 0 and 1")


def _ratio(value: Any, name: str) -> None:
    if value is not None and (not finite_number(value) or value < 1):
        raise ArtifactError(f"{name} must be finite and at least 1")


@dataclass(frozen=True)
class RuleConfig:
    """Caller-selected thresholds. There are no defaults.

    Every field is optional; a rule whose keys are all ``None`` reports
    ``unconfigured``. :meth:`to_dict` writes only the keys that were supplied,
    so a sidecar configured for one rule round-trips unchanged.

    Attributes:
        loader_wait_threshold_ms: ``host_loader_wait`` fires at
            ``host_data_wait_ms >= value``. Finite, >= 0.
        h2d_share_threshold: ``host_h2d_share`` fires at
            ``host_h2d_ms / host_step_ms >= value``. Fraction in [0, 1].
        step_variability_ratio: ``step_duration_variability`` fires at
            ``max(host_step_ms) / median(host_step_ms) >= value``. >= 1.
        step_variability_min_steps: Minimum recorded steps before the ratio is
            computed. Integer >= 2; requires ``step_variability_ratio``.
        transfer_exposed_share: ``device_transfer_overlap`` fires at
            ``exposed / device_attributed_transfer_ms >= value``. Fraction.
        collective_exposed_share: ``collective_compute_overlap`` fires at
            ``exposed / collective_kernel_ms >= value``. Fraction.
        rank_imbalance_ratio: ``rank_step_imbalance`` fires at
            ``max(step_time_ms) / min(step_time_ms) >= value``. >= 1.

    Raises:
        ArtifactError: A value outside its documented range.
    """

    loader_wait_threshold_ms: float | None = None
    h2d_share_threshold: float | None = None
    step_variability_ratio: float | None = None
    step_variability_min_steps: int | None = None
    transfer_exposed_share: float | None = None
    collective_exposed_share: float | None = None
    rank_imbalance_ratio: float | None = None

    def __post_init__(self):
        value = self.loader_wait_threshold_ms
        if value is not None and (not finite_number(value) or value < 0):
            raise ArtifactError("loader_wait_threshold_ms must be finite and nonnegative")
        _share(self.h2d_share_threshold, "h2d_share_threshold")
        _ratio(self.step_variability_ratio, "step_variability_ratio")
        value = self.step_variability_min_steps
        if value is not None and (type(value) is not int or value < MIN_STEPS_FLOOR):
            raise ArtifactError(
                f"step_variability_min_steps must be an integer of at least {MIN_STEPS_FLOOR}"
            )
        if value is not None and self.step_variability_ratio is None:
            raise ArtifactError(
                "step_variability_min_steps requires step_variability_ratio; a minimum step "
                "count alone configures no comparison"
            )
        _share(self.transfer_exposed_share, "transfer_exposed_share")
        _share(self.collective_exposed_share, "collective_exposed_share")
        _ratio(self.rank_imbalance_ratio, "rank_imbalance_ratio")

    @classmethod
    def keys(cls) -> tuple[str, ...]:
        """All threshold keys, in declaration order."""
        return tuple(field.name for field in fields(cls))

    @classmethod
    def from_mapping(cls, payload: Any) -> RuleConfig:
        """Build from a persisted ``rule_config`` object.

        Args:
            payload: The JSON object stored in ``analysis.json``.

        Raises:
            ArtifactError: Not an object, unknown keys, or out-of-range values.
        """
        if not isinstance(payload, dict):
            raise ArtifactError("rule_config must be an object")
        unknown = sorted(set(payload) - set(cls.keys()))
        if unknown:
            raise ArtifactError(
                f"rule_config has unsupported key(s): {', '.join(unknown)}. "
                f"Known keys: {', '.join(cls.keys())}"
            )
        return cls(**payload)

    def to_dict(self) -> dict:
        """The supplied keys only, for persistence in ``analysis.json``."""
        return {
            name: getattr(self, name) for name in self.keys() if getattr(self, name) is not None
        }

    def restrict(self, rule_ids: Sequence[str], *, command: str) -> RuleConfig:
        """Refuse keys that ``command`` does not evaluate.

        ``analyze`` and ``ranks`` share one key vocabulary; a cross-rank
        threshold given to ``analyze`` would be persisted and never compared,
        which is a configuration mistake rather than a precedence rule.
        """
        allowed = set(rule_keys(rule_ids))
        stray = [key for key in self.to_dict() if key not in allowed]
        if stray:
            owners = sorted(
                _BY_ID[rule].evaluated_by for rule in _BY_ID if set(rule_keys([rule])) & set(stray)
            )
            raise ArtifactError(
                f"threshold key(s) {', '.join(stray)} are not evaluated by {command}; "
                f"they belong to {', '.join(owners)} (see: shardscope rules)"
            )
        return self


def parse_threshold(text: str) -> tuple[str, int | float]:
    """Parse one ``--threshold KEY=VALUE`` command-line item.

    Integers stay integers (``step_variability_min_steps`` needs one); anything
    else must spell a plain decimal number. Range checks belong to
    :class:`RuleConfig`, so an out-of-range value gets the same message
    whether it arrived here or in a persisted sidecar.

    Args:
        text: For example ``"h2d_share_threshold=0.3"``.

    Returns:
        ``(key, value)`` with ``value`` an ``int`` or ``float``.

    Raises:
        ArtifactError: Missing ``=``, unknown key, or a value that is not a
            plain finite number.
    """
    key, separator, raw = text.partition("=")
    key, raw = key.strip(), raw.strip()
    if not separator or not key or not raw:
        raise ArtifactError(
            f"threshold {text!r} must be KEY=VALUE; keys: {', '.join(RuleConfig.keys())}"
        )
    if key not in set(RuleConfig.keys()):
        raise ArtifactError(
            f"unknown threshold key {key!r}; keys: {', '.join(RuleConfig.keys())} "
            "(see: shardscope rules)"
        )
    value: int | float
    try:
        if "_" in raw or not raw.isascii():
            raise ValueError(raw)
        value = int(raw) if raw.lstrip("+-").isdigit() else float(raw)
        if not finite_number(value):
            raise ValueError(raw)
    except ValueError as exc:
        raise ArtifactError(
            f"threshold {key}={raw!r}: value must be a plain finite number"
        ) from exc
    return key, value


def describe_rules() -> list[dict[str, Any]]:
    """Return both registries as JSON data: per-rank rules first, then cross-rank.

    This is the payload of ``shardscope rules --format json``.
    """
    return [spec.to_dict() for spec in (*RULES, *RANK_RULES)]


def _recorded(measured: Any, *, basis: str = "host_range") -> bool:
    """A measurement this module will compare: recorded, finite, evidenced, right basis."""
    return bool(
        isinstance(measured, dict)
        and measured.get("status") == "recorded"
        and finite_number(measured.get("value_ms"))
        and measured["value_ms"] >= 0
        and measured.get("evidence")
        and measured.get("basis") == basis
    )


def _unconfigured(rule_id: str, text: str) -> dict:
    return {
        "rule_id": rule_id,
        "status": "unconfigured",
        "steps": [],
        "inputs": [],
        "comparison": None,
        "evidence": [],
        "interpretation": text,
    }


def _distinct(refs: list[dict]) -> list[dict]:
    seen: list[dict] = []
    for ref in refs:
        if ref not in seen:
            seen.append(ref)
    return seen


def _input(step: dict, metric: str) -> dict:
    return {
        "training_step": step["training_step"],
        "metric": metric,
        "measurement": step["measurements"].get(metric),
    }


# --- host_loader_wait (legacy shape; do not alter) --------------------------------


def _loader_wait(steps: list[dict], config: RuleConfig, context: dict) -> list[dict]:
    if config.loader_wait_threshold_ms is None:
        return [
            _unconfigured(RULE_ID, "No explicit loader-wait threshold configured; no diagnosis.")
        ]
    threshold = config.loader_wait_threshold_ms
    outputs = []
    for step in steps:
        measured = step["measurements"].get("host_data_wait_ms")
        ready = _recorded(measured)
        value = measured["value_ms"] if ready else None
        exceeds = value >= threshold if ready else None
        outputs.append(
            {
                "rule_id": RULE_ID,
                "status": "possible_input_wait"
                if exceeds
                else ("below_threshold" if ready else "unavailable"),
                "steps": [step["training_step"]],
                "inputs": [_input(step, "host_data_wait_ms")],
                "comparison": {
                    "operator": ">=",
                    "threshold_ms": threshold,
                    "value_ms": value,
                    "result": exceeds,
                },
                "evidence": measured["evidence"] if ready else [],
                "interpretation": (
                    "Host wait meets the configured threshold: possible input waiting; "
                    "does not establish GPU starvation or a confirmed root cause."
                    if exceeds
                    else (
                        "Host wait is below the configured threshold."
                        if ready
                        else "Host loader-wait evidence unavailable; not measured zero."
                    )
                ),
            }
        )
    return outputs


# --- shared share-of-union evaluation ------------------------------------------------


def _exposed_share(
    *,
    spec: RuleSpec,
    step: dict,
    numerator_metric: str,
    denominator_metric: str,
    threshold: float,
    triggered: str,
    statistic: str,
    mode: str,
    bases: tuple[str, str],
    describe: str,
) -> dict:
    """Compare a per-step share built from two recorded measurements of one step.

    ``mode`` is ``"share"`` (numerator / denominator) or ``"exposed"``
    ((denominator - numerator) / denominator). Both inputs must be recorded,
    carry their expected basis and share one scope; otherwise the finding is
    ``unavailable`` or ``ambiguous`` and compares nothing.
    """
    top = step["measurements"].get(numerator_metric)
    whole = step["measurements"].get(denominator_metric)
    missing = [
        name
        for name, measured, basis in (
            (numerator_metric, top, bases[0]),
            (denominator_metric, whole, bases[1]),
        )
        if not _recorded(measured, basis=basis)
    ]
    status, share, problem = "unavailable", None, None
    if not missing:
        if top.get("scope") != whole.get("scope"):
            problem = f"{numerator_metric} and {denominator_metric} come from different scopes"
        elif whole["value_ms"] == 0:
            problem = f"zero-length {describe}; a share of zero time is undefined"
        elif top["value_ms"] > whole["value_ms"]:
            problem = f"{numerator_metric} is longer than its {describe}"
        if problem is not None:
            status = "ambiguous"
        else:
            share = (
                top["value_ms"] / whole["value_ms"]
                if mode == "share"
                else (whole["value_ms"] - top["value_ms"]) / whole["value_ms"]
            )
            status = triggered if share >= threshold else "below_threshold"
    result = None if share is None else share >= threshold
    if status == triggered:
        interpretation = (
            f"{statistic} is {share:.3f} and meets the configured share. "
            + spec.interpretation
            + " Next: "
            + spec.next_investigation
        )
    elif status == "below_threshold":
        interpretation = (
            f"{statistic} is below the configured threshold; this does not establish health."
        )
    elif status == "ambiguous":
        interpretation = f"Share not compared: {problem}."
    else:
        interpretation = (
            f"Required measurement(s) not recorded: {', '.join(missing)}; not measured zero."
        )
    numerator = (
        None
        if numerator_metric in missing
        else (
            top["value_ms"]
            if mode == "share"
            else (None if denominator_metric in missing else whole["value_ms"] - top["value_ms"])
        )
    )
    return {
        "rule_id": spec.rule_id,
        "status": status,
        "steps": [step["training_step"]],
        "inputs": [_input(step, numerator_metric), _input(step, denominator_metric)],
        "comparison": {
            "operator": ">=",
            "statistic": statistic,
            "threshold": threshold,
            "value": share,
            "numerator_ms": numerator,
            "denominator_ms": whole["value_ms"] if denominator_metric not in missing else None,
            "result": result,
        },
        # The overlap's evidence repeats the transfer events; cite each once.
        "evidence": _distinct(list(top["evidence"]) + list(whole["evidence"]))
        if status in (triggered, "below_threshold")
        else [],
        "interpretation": interpretation,
    }


# --- host_h2d_share ----------------------------------------------------------------


def _h2d_share(steps: list[dict], config: RuleConfig, context: dict) -> list[dict]:
    spec = _BY_ID[H2D_SHARE_RULE]
    if config.h2d_share_threshold is None:
        return [
            _unconfigured(
                H2D_SHARE_RULE, "No explicit h2d share threshold configured; no diagnosis."
            )
        ]
    outputs = []
    for step in steps:
        finding = _exposed_share(
            spec=spec,
            step=step,
            numerator_metric="host_h2d_ms",
            denominator_metric="host_step_ms",
            threshold=config.h2d_share_threshold,
            triggered="possible_host_copy_wait",
            statistic="host_h2d_ms / host_step_ms",
            mode="share",
            bases=("host_range", "host_range"),
            describe="host step range",
        )
        # The original wording of this rule's text is kept for persisted sidecars.
        if finding["status"] == "possible_host_copy_wait":
            finding["interpretation"] = (
                f"Host time inside host-to-device copy calls is "
                f"{finding['comparison']['value']:.3f} of the host step range and meets the "
                "configured share. " + spec.interpretation + " Next: " + spec.next_investigation
            )
        elif finding["status"] == "below_threshold":
            finding["interpretation"] = (
                "Host copy-call share of the host step is below the configured threshold; "
                "this does not establish that transfers are efficient."
            )
        elif finding["status"] == "ambiguous":
            finding["interpretation"] = finding["interpretation"].replace(
                "host_h2d_ms is longer than its host step range",
                "host copy union is longer than its host step range",
            )
        outputs.append(finding)
    return outputs


# --- device_transfer_overlap -----------------------------------------------------------


def _transfer_overlap(steps: list[dict], config: RuleConfig, context: dict) -> list[dict]:
    spec = _BY_ID[TRANSFER_OVERLAP_RULE]
    if config.transfer_exposed_share is None:
        return [
            _unconfigured(
                TRANSFER_OVERLAP_RULE,
                "No explicit transfer exposed-share threshold configured; no diagnosis.",
            )
        ]
    outputs = []
    for step in steps:
        finding = _exposed_share(
            spec=spec,
            step=step,
            numerator_metric=TRANSFER_OVERLAP_METRIC,
            denominator_metric=ATTRIBUTED_TRANSFER_METRIC,
            threshold=config.transfer_exposed_share,
            triggered="possible_exposed_transfer",
            statistic=(
                f"({ATTRIBUTED_TRANSFER_METRIC} - {TRANSFER_OVERLAP_METRIC}) / "
                f"{ATTRIBUTED_TRANSFER_METRIC}"
            ),
            mode="exposed",
            bases=(OVERLAP_BASIS, ATTRIBUTION_BASIS),
            describe="attributed transfer union",
        )
        coverage = (step["measurements"].get(ATTRIBUTED_TRANSFER_METRIC) or {}).get("coverage")
        if isinstance(coverage, dict):
            finding["comparison"]["directions"] = coverage.get("directions")
            finding["comparison"]["transfer_completeness"] = coverage.get("completeness")
        outputs.append(finding)
    return outputs


# --- collective_compute_overlap --------------------------------------------------------


def _collective_overlap(steps: list[dict], config: RuleConfig, context: dict) -> list[dict]:
    spec = _BY_ID[COLLECTIVE_OVERLAP_RULE]
    if config.collective_exposed_share is None:
        return [
            _unconfigured(
                COLLECTIVE_OVERLAP_RULE,
                "No explicit collective exposed-share threshold configured; no diagnosis.",
            )
        ]
    threshold = config.collective_exposed_share
    block = context.get("attribution")
    statistic = "(collective_kernel_ms - collective_compute_overlap_ms) / collective_kernel_ms"

    def finding(status, *, steps_in=(), inputs=(), comparison=None, evidence=(), text):
        return {
            "rule_id": COLLECTIVE_OVERLAP_RULE,
            "status": status,
            "steps": list(steps_in),
            "inputs": list(inputs),
            "comparison": comparison,
            "evidence": list(evidence),
            "interpretation": text,
        }

    if not isinstance(block, dict) or block.get("schema") != ATTRIBUTION_SCHEMA:
        return [
            finding(
                "unavailable",
                comparison={"operator": ">=", "statistic": statistic, "threshold": threshold},
                text=(
                    "This analysis carries no collective classification (kernel_attribution "
                    f"schema {ATTRIBUTION_SCHEMA}); re-run shardscope analyze --overwrite. Not "
                    "measured zero."
                ),
            )
        ]
    mapped: dict[str, list[int]] = {}
    for step in steps:
        if step.get("boundary") is not None:
            mapped.setdefault(step["boundary"]["file"], []).append(step["training_step"])
    outputs = []
    for name in sorted(block["traces"]):
        for activity in block["traces"][name].get("device_activity", []):
            if not activity["collective_kernel_events"]:
                continue
            collective = activity["collective_kernel_ms"]
            overlap = activity["collective_compute_overlap_ms"]
            device = activity["device"]
            inputs = [
                {"trace": name, "device": device, "metric": key, "value_ms": activity[key]}
                for key in ("collective_kernel_ms", "collective_compute_overlap_ms")
            ]
            comparison = {
                "operator": ">=",
                "statistic": statistic,
                "threshold": threshold,
                "value": None,
                "numerator_ms": None,
                "denominator_ms": collective,
                "result": None,
                "trace": name,
                "device": device,
                "collective_kernel_events": activity["collective_kernel_events"],
                "compute_kernel_events": activity["compute_kernel_events"],
            }
            evidence = [
                {"file": name, "event_index": index}
                for index in block["traces"][name]["devices"][
                    [d["device"] for d in block["traces"][name]["devices"]].index(device)
                ]["collective_kernel_event_indices"]
            ]
            if not finite_number(collective) or collective == 0:
                outputs.append(
                    finding(
                        "ambiguous",
                        steps_in=mapped.get(name, []),
                        inputs=inputs,
                        comparison=comparison,
                        text="Share not compared: zero-length collective union on device "
                        f"{device} of {name}; a share of zero time is undefined.",
                    )
                )
                continue
            if overlap > collective:
                outputs.append(
                    finding(
                        "ambiguous",
                        steps_in=mapped.get(name, []),
                        inputs=inputs,
                        comparison=comparison,
                        text="Share not compared: overlap longer than the collective union on "
                        f"device {device} of {name}.",
                    )
                )
                continue
            exposed = collective - overlap
            share = exposed / collective
            comparison.update(value=share, numerator_ms=exposed, result=share >= threshold)
            if share >= threshold:
                text = (
                    f"On device {device} of {name}, {share:.3f} of the collective kernel union "
                    f"({activity['collective_kernel_events']} kernel(s), {collective:.4g} ms) ran "
                    "with no compute kernel on the device and meets the configured share. "
                    + spec.interpretation
                    + " Next: "
                    + spec.next_investigation
                )
                status = "possible_exposed_collective"
            else:
                text = (
                    f"On device {device} of {name}, the exposed share of the collective kernel "
                    "union is below the configured threshold; this does not establish that "
                    "communication is hidden."
                )
                status = "below_threshold"
            outputs.append(
                finding(
                    status,
                    steps_in=mapped.get(name, []),
                    inputs=inputs,
                    comparison=comparison,
                    evidence=evidence,
                    text=text,
                )
            )
    if not outputs:
        outputs.append(
            finding(
                "unavailable",
                comparison={"operator": ">=", "statistic": statistic, "threshold": threshold},
                text=(
                    "No kernel in any analyzed trace was classified as a collective (NCCL "
                    "device-kernel name plus a launch inside a c10d:: op); nothing is compared. "
                    "Not measured zero."
                ),
            )
        )
    return outputs


# --- step_duration_variability -----------------------------------------------------


def _phase_signature(step: dict) -> list[str]:
    """Which host phases this step recorded; the cheap equivalence check."""
    return sorted(
        name
        for name, measured in step["measurements"].items()
        if name.startswith("host_") and _recorded(measured)
    )


def _variability(steps: list[dict], config: RuleConfig, context: dict) -> list[dict]:
    spec = _BY_ID[VARIABILITY_RULE]
    if config.step_variability_ratio is None:
        return [
            _unconfigured(
                VARIABILITY_RULE,
                "No explicit step variability ratio configured; no diagnosis.",
            )
        ]
    threshold = config.step_variability_ratio
    minimum = (
        config.step_variability_min_steps
        if config.step_variability_min_steps is not None
        else MIN_STEPS_FLOOR
    )
    recorded = [s for s in steps if _recorded(s["measurements"].get("host_step_ms"))]
    signatures = {tuple(_phase_signature(s)) for s in recorded}
    comparison: dict[str, Any] = {
        "operator": ">=",
        "statistic": "max(host_step_ms) / median(host_step_ms)",
        "threshold": threshold,
        "value": None,
        "result": None,
        "recorded_steps": len(recorded),
        "required_steps": minimum,
        "min_ms": None,
        "median_ms": None,
        "max_ms": None,
        "slowest_training_step": None,
    }
    inputs = [_input(s, "host_step_ms") for s in recorded]
    evidence: list = []
    if not recorded:
        status = "unavailable"
        interpretation = "No mapped step recorded host_step_ms; nothing to compare."
    elif len(recorded) < minimum:
        status = "insufficient"
        interpretation = (
            f"Only {len(recorded)} step(s) recorded host_step_ms; {minimum} required before a "
            "spread statistic is formed."
        )
    elif len(signatures) > 1:
        status = "ambiguous"
        interpretation = (
            "Recorded steps differ in which host phases they recorded "
            f"({len(signatures)} distinct phase sets), so they are not known to be "
            "equivalent logical work; no spread is compared."
        )
        comparison["phase_sets"] = [list(signature) for signature in sorted(signatures)]
    else:
        values = [s["measurements"]["host_step_ms"]["value_ms"] for s in recorded]
        median = statistics.median(values)
        slowest = max(recorded, key=lambda s: s["measurements"]["host_step_ms"]["value_ms"])
        comparison.update(
            min_ms=min(values),
            median_ms=median,
            max_ms=max(values),
            slowest_training_step=slowest["training_step"],
        )
        if median == 0:
            status = "ambiguous"
            interpretation = "Median host step range is zero; the ratio is undefined."
        else:
            ratio = max(values) / median
            comparison.update(value=ratio, result=ratio >= threshold)
            evidence = [
                ref for s in recorded for ref in s["measurements"]["host_step_ms"]["evidence"]
            ]
            if ratio >= threshold:
                status = "possible_step_variability"
                interpretation = (
                    f"Training step {slowest['training_step']} took {ratio:.3f} times the "
                    f"median host step range over {len(recorded)} recorded step(s), meeting "
                    "the configured ratio. "
                    + spec.interpretation
                    + " Next: "
                    + spec.next_investigation
                )
            else:
                status = "below_threshold"
                interpretation = (
                    f"Slowest-to-median host step ratio over {len(recorded)} recorded step(s) "
                    "is below the configured ratio; this does not establish steady steps."
                )
    return [
        {
            "rule_id": VARIABILITY_RULE,
            "status": status,
            "steps": [s["training_step"] for s in recorded],
            "inputs": inputs,
            "comparison": comparison,
            "evidence": evidence,
            "interpretation": interpretation,
        }
    ]


_EVALUATORS = {
    RULE_ID: _loader_wait,
    H2D_SHARE_RULE: _h2d_share,
    VARIABILITY_RULE: _variability,
    TRANSFER_OVERLAP_RULE: _transfer_overlap,
    COLLECTIVE_OVERLAP_RULE: _collective_overlap,
}


def evaluate_rules(
    steps: list[dict],
    config: RuleConfig | None,
    *,
    rules: Sequence[str] = RULE_IDS,
    attribution: dict | None = None,
) -> list[dict]:
    """Evaluate ``rules`` over one rank's analysis and return their findings.

    Args:
        steps: Analysis step rows (``training_step``, ``boundary``,
            ``measurements``), with evidence already validated by the caller.
        config: Thresholds; ``None`` evaluates every rule as ``unconfigured``.
        rules: Registry subset to evaluate, in output order. A legacy sidecar
            passes :data:`LEGACY_RULES` so its persisted findings compare equal.
        attribution: The sidecar's ``kernel_attribution`` block, read by the
            window-level collective rule; ``None`` for sidecars without one.

    Returns:
        One finding dict per rule per step (or per analysis, trace or device
        for whole-run rules), concatenated in ``rules`` order. Each has
        ``rule_id``, ``status``, ``steps``, ``inputs``, ``comparison``,
        ``evidence`` and ``interpretation``.

    Raises:
        ArtifactError: An unknown or repeated rule id.
    """
    unknown = [rule for rule in rules if rule not in _EVALUATORS]
    if unknown or len(set(rules)) != len(rules):
        raise ArtifactError(
            f"unknown or repeated rule id(s): {', '.join(unknown) or 'repeated'}; "
            f"known: {', '.join(RULE_IDS)}"
        )
    effective = config if config is not None else RuleConfig()
    context = {"attribution": attribution}
    findings: list[dict] = []
    for rule in rules:
        findings.extend(_EVALUATORS[rule](steps, effective, context))
    return findings


# --- rank_step_imbalance (cross-rank; evaluated by shardscope ranks) --------------------


def evaluate_rank_rules(comparison: dict, config: RuleConfig | None) -> list[dict]:
    """Evaluate the cross-rank rules over a rank comparison table.

    Args:
        comparison: The measurement part of a ``ranks.json`` payload:
            ``world_size``, ``ranks``, ``missing_ranks``, ``compatibility``
            (with ``problems``) and ``steps`` rows carrying per-rank
            ``step_time_ms`` plus the derived ``min_ms``, ``max_ms``,
            ``min_rank``, ``max_rank``, ``spread_ms``, ``ratio``, ``coverage``
            and ``problems``.
        config: Thresholds; ``None`` reports ``unconfigured``.

    Returns:
        Findings in the per-rank finding shape. ``incompatible`` and
        ``incomplete`` (missing rank directories) are one finding each; a
        compared launch yields one finding per training step.
    """
    spec = _BY_ID[RANK_IMBALANCE_RULE]
    effective = config if config is not None else RuleConfig()
    if effective.rank_imbalance_ratio is None:
        return [
            _unconfigured(
                RANK_IMBALANCE_RULE, "No explicit rank imbalance ratio configured; no diagnosis."
            )
        ]
    threshold = effective.rank_imbalance_ratio
    statistic = "max(step_time_ms) / min(step_time_ms)"

    def finding(status, *, steps=(), inputs=(), comparison=None, text):
        return {
            "rule_id": RANK_IMBALANCE_RULE,
            "status": status,
            "steps": list(steps),
            "inputs": list(inputs),
            "comparison": comparison,
            "evidence": [],
            "interpretation": text,
        }

    problems = comparison["compatibility"]["problems"]
    if problems:
        return [
            finding(
                "incompatible",
                comparison={"operator": ">=", "statistic": statistic, "threshold": threshold},
                text="Rank recordings are not comparable: " + "; ".join(problems) + ".",
            )
        ]
    if comparison["missing_ranks"]:
        return [
            finding(
                "incomplete",
                comparison={
                    "operator": ">=",
                    "statistic": statistic,
                    "threshold": threshold,
                    "world_size": comparison["world_size"],
                    "missing_ranks": list(comparison["missing_ranks"]),
                },
                text=(
                    f"Rank directories {comparison['missing_ranks']} of world size "
                    f"{comparison['world_size']} are missing; a straggler could be the missing "
                    "rank, so nothing is compared."
                ),
            )
        ]
    outputs = []
    for row in comparison["steps"]:
        inputs = [
            {"rank": item["rank"], "metric": "step_time_ms", "value_ms": item["step_time_ms"]}
            for item in row["per_rank"]
        ]
        compared = {
            "operator": ">=",
            "statistic": statistic,
            "threshold": threshold,
            "value": row["ratio"],
            "result": None if row["ratio"] is None else row["ratio"] >= threshold,
            "min_ms": row["min_ms"],
            "max_ms": row["max_ms"],
            "min_rank": row["min_rank"],
            "max_rank": row["max_rank"],
            "spread_ms": row["spread_ms"],
            "ranks_recorded": len(row["per_rank"]),
            "work_equivalence": row["work_equivalence"],
        }
        if row["coverage"] != "complete":
            status = "incomplete"
            text = (
                f"Training step {row['training_step']} is not recorded on rank(s) "
                f"{row['missing_ranks']}; nothing is compared."
            )
        elif row["problems"]:
            status = "ambiguous"
            text = "Not compared: " + "; ".join(row["problems"]) + "."
        elif row["ratio"] >= threshold:
            status = "possible_rank_imbalance"
            text = (
                f"Training step {row['training_step']}: rank {row['max_rank']} recorded "
                f"{row['max_ms']:.4g} ms and rank {row['min_rank']} {row['min_ms']:.4g} ms, a "
                f"ratio of {row['ratio']:.3f} that meets the configured ratio. "
                + spec.interpretation
                + " Next: "
                + spec.next_investigation
            )
        else:
            status = "below_threshold"
            text = (
                f"Training step {row['training_step']}: the ranks' step times differ by a ratio "
                "below the configured threshold; this does not establish balanced work."
            )
        outputs.append(
            finding(
                status, steps=[row["training_step"]], inputs=inputs, comparison=compared, text=text
            )
        )
    return outputs
