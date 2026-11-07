"""Diagnostic rules: explicit threshold comparisons over persisted measurements.

Each rule reads validated per-step measurements from an analysis payload and
compares them against thresholds the caller chose. Nothing is calibrated,
defaulted, scored or ranked:

* A missing measurement is never zero; a rule that lacks an input says so.
* ``below_threshold`` means only that the configured comparison did not fire.
  It does not establish that the workload is healthy.
* A triggered status is a restrained observation with a next place to look,
  not a root cause, a confidence score or an expected speedup.

The registry :data:`RULES` is the single description of each rule: stable id,
required measurements, scope, configuration keys with units and ranges, the
statuses it can emit and the next investigation. ``shardscope rules`` prints
it, so the same text answers "what can analyze evaluate?" and "what did this
finding mean?".

Adding a rule:
    1. Add a :class:`RuleSpec` to :data:`RULES` and a config field to
       :class:`RuleConfig` (validated in ``__post_init__``).
    2. Write an evaluator ``(steps, config) -> list[finding]`` and register it
       in ``_EVALUATORS``. Use :func:`_recorded` to accept only recorded,
       finite, evidenced measurements with the right basis.
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

from shardscope.env import NOT_OBSERVED
from shardscope.errors import ArtifactError
from shardscope.parse import finite_number

__all__ = [
    "LEGACY_RULES",
    "NOT_OBSERVED",
    "RULE_ID",
    "RULE_IDS",
    "RULES",
    "TRIGGERED_STATUSES",
    "RuleConfig",
    "RuleSpec",
    "describe_rules",
    "evaluate_rules",
    "parse_threshold",
]

#: The original rule; kept as a module constant because callers import it.
RULE_ID = "host_loader_wait"
H2D_SHARE_RULE = "host_h2d_share"
VARIABILITY_RULE = "step_duration_variability"

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
        granularity: ``"per training step"`` or ``"per analysis"``.
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "granularity": self.granularity,
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
                "range": "finite, 0 <= value <= 1",
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
)

RULE_IDS: tuple[str, ...] = tuple(spec.rule_id for spec in RULES)
_BY_ID = {spec.rule_id: spec for spec in RULES}

#: Statuses that mean "the configured comparison fired", per rule.
TRIGGERED_STATUSES: frozenset[str] = frozenset(
    {"possible_input_wait", "possible_host_copy_wait", "possible_step_variability"}
)


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

    Raises:
        ArtifactError: A value outside its documented range.
    """

    loader_wait_threshold_ms: float | None = None
    h2d_share_threshold: float | None = None
    step_variability_ratio: float | None = None
    step_variability_min_steps: int | None = None

    def __post_init__(self):
        value = self.loader_wait_threshold_ms
        if value is not None and (not finite_number(value) or value < 0):
            raise ArtifactError("loader_wait_threshold_ms must be finite and nonnegative")
        value = self.h2d_share_threshold
        if value is not None and (not finite_number(value) or not 0 <= value <= 1):
            raise ArtifactError("h2d_share_threshold must be a finite fraction between 0 and 1")
        value = self.step_variability_ratio
        if value is not None and (not finite_number(value) or value < 1):
            raise ArtifactError("step_variability_ratio must be finite and at least 1")
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
    """Return the registry as JSON data, in evaluation order.

    This is the payload of ``shardscope rules --format json``.
    """
    return [spec.to_dict() for spec in RULES]


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


def _input(step: dict, metric: str) -> dict:
    return {
        "training_step": step["training_step"],
        "metric": metric,
        "measurement": step["measurements"].get(metric),
    }


# --- host_loader_wait (legacy shape; do not alter) --------------------------------


def _loader_wait(steps: list[dict], config: RuleConfig) -> list[dict]:
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


# --- host_h2d_share ----------------------------------------------------------------


def _h2d_share(steps: list[dict], config: RuleConfig) -> list[dict]:
    spec = _BY_ID[H2D_SHARE_RULE]
    if config.h2d_share_threshold is None:
        return [
            _unconfigured(
                H2D_SHARE_RULE, "No explicit h2d share threshold configured; no diagnosis."
            )
        ]
    threshold = config.h2d_share_threshold
    outputs = []
    for step in steps:
        h2d = step["measurements"].get("host_h2d_ms")
        whole = step["measurements"].get("host_step_ms")
        missing = [
            name
            for name, measured in (("host_h2d_ms", h2d), ("host_step_ms", whole))
            if not _recorded(measured)
        ]
        status, share, problem = "unavailable", None, None
        if not missing:
            if h2d.get("scope") != whole.get("scope"):
                problem = "host_h2d_ms and host_step_ms come from different scopes"
            elif whole["value_ms"] == 0:
                problem = "zero-length host step range; a share of zero time is undefined"
            elif h2d["value_ms"] > whole["value_ms"]:
                problem = "host copy union is longer than its host step range"
            if problem is not None:
                status = "ambiguous"
            else:
                share = h2d["value_ms"] / whole["value_ms"]
                status = "possible_host_copy_wait" if share >= threshold else "below_threshold"
        result = None if share is None else share >= threshold
        if status == "possible_host_copy_wait":
            interpretation = (
                f"Host time inside host-to-device copy calls is {share:.3f} of the host step "
                "range and meets the configured share. "
                + spec.interpretation
                + " Next: "
                + spec.next_investigation
            )
        elif status == "below_threshold":
            interpretation = (
                "Host copy-call share of the host step is below the configured threshold; "
                "this does not establish that transfers are efficient."
            )
        elif status == "ambiguous":
            interpretation = f"Share not compared: {problem}."
        else:
            interpretation = (
                f"Required measurement(s) not recorded: {', '.join(missing)}; not measured zero."
            )
        outputs.append(
            {
                "rule_id": H2D_SHARE_RULE,
                "status": status,
                "steps": [step["training_step"]],
                "inputs": [_input(step, "host_h2d_ms"), _input(step, "host_step_ms")],
                "comparison": {
                    "operator": ">=",
                    "statistic": "host_h2d_ms / host_step_ms",
                    "threshold": threshold,
                    "value": share,
                    "numerator_ms": h2d["value_ms"] if "host_h2d_ms" not in missing else None,
                    "denominator_ms": whole["value_ms"] if "host_step_ms" not in missing else None,
                    "result": result,
                },
                "evidence": (list(h2d["evidence"]) + list(whole["evidence"]))
                if status in ("possible_host_copy_wait", "below_threshold")
                else [],
                "interpretation": interpretation,
            }
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


def _variability(steps: list[dict], config: RuleConfig) -> list[dict]:
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
}


def evaluate_rules(
    steps: list[dict],
    config: RuleConfig | None,
    *,
    rules: Sequence[str] = RULE_IDS,
) -> list[dict]:
    """Evaluate ``rules`` over per-step measurements and return their findings.

    Args:
        steps: Analysis step rows (``training_step``, ``measurements``), with
            evidence already validated by the caller.
        config: Thresholds; ``None`` evaluates every rule as ``unconfigured``.
        rules: Registry subset to evaluate, in output order. A legacy sidecar
            passes :data:`LEGACY_RULES` so its persisted findings compare equal.

    Returns:
        One finding dict per rule per step (or per analysis for whole-run
        rules), concatenated in ``rules`` order. Each has ``rule_id``,
        ``status``, ``steps``, ``inputs``, ``comparison``, ``evidence`` and
        ``interpretation``.

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
    findings: list[dict] = []
    for rule in rules:
        findings.extend(_EVALUATORS[rule](steps, effective))
    return findings
