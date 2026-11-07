"""Summaries of the findings stored in ``analysis.json`` or ``ranks.json``.

Backs ``shardscope findings``. A run is loaded exactly as ``report`` loads it
(the sidecar is validated against its sources first, so a stale diagnosis is
refused) and the persisted findings are regrouped per rule: status counts,
which steps fired, the compared values and the evidence references. With
``--ranks`` the same is done for a launch's persisted cross-rank comparison.

Nothing here evaluates a rule, reads a trace for new timing or supplies a
threshold. To change what a finding says, re-run ``shardscope analyze``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from shardscope.artifacts import RunArtifacts, checked_analysis
from shardscope.errors import ArtifactError
from shardscope.rules import LEGACY_RULES, RANK_RULES, RULES, TRIGGERED_STATUSES, RuleConfig

__all__ = [
    "known_statuses",
    "render_findings_text",
    "render_rank_findings_text",
    "summarize_findings",
    "summarize_rank_findings",
]

_SPECS = {spec.rule_id: spec for spec in (*RULES, *RANK_RULES)}


def known_statuses() -> tuple[str, ...]:
    """Every status any registered rule can emit, for argument validation."""
    return tuple(sorted({status for spec in (*RULES, *RANK_RULES) for status in spec.statuses}))


def _step_ranges(steps: Iterable[int]) -> str:
    """``4-7, 12-15`` for ``[4,5,6,7,12,13,14,15]``; ``none`` for nothing."""
    ordered = sorted(set(steps))
    if not ordered:
        return "none"
    parts = []
    start = previous = ordered[0]
    for step in ordered[1:]:
        if step == previous + 1:
            previous = step
            continue
        parts.append(f"{start}-{previous}" if start != previous else str(start))
        start = previous = step
    parts.append(f"{start}-{previous}" if start != previous else str(start))
    return ", ".join(parts)


def _number(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _compared(finding: dict[str, Any]) -> str:
    """``statistic value >= threshold`` from whichever keys the comparison uses."""
    comparison = finding.get("comparison") or {}
    statistic = comparison.get("statistic")
    if statistic is None:
        inputs = finding.get("inputs") or []
        statistic = inputs[0]["metric"] if inputs else "value"
    value = comparison.get("value", comparison.get("value_ms"))
    threshold = comparison.get("threshold", comparison.get("threshold_ms"))
    unit = " ms" if "value_ms" in comparison else ""
    operator = comparison.get("operator", "?")
    return f"{statistic} {_number(value)}{unit} {operator} {_number(threshold)}{unit}"


def _refs(evidence: Sequence[dict[str, Any]], *, limit: int = 4) -> str:
    shown = [f"{ref['file']} event[{ref['event_index']}]" for ref in evidence[:limit]]
    rest = len(evidence) - len(shown)
    return (", ".join(shown) + (f", ... {rest} more" if rest > 0 else "")) or "none"


def summarize_findings(
    run: RunArtifacts,
    *,
    rule_ids: Sequence[str] = (),
    statuses: Sequence[str] = (),
) -> dict[str, Any]:
    """Group a validated sidecar's findings per rule.

    Args:
        run: A loaded run; its ``analysis.json`` is validated before use.
        rule_ids: Restrict the rules shown. Empty means all registered rules,
            including ones the sidecar predates (reported ``evaluated: false``).
        statuses: Restrict the findings listed under each rule. Per-status
            counts still describe every persisted finding, so a filtered view
            says what it left out.

    Returns:
        A JSON-serializable dict: run identity, ``rules_evaluated``,
        ``rule_config``, mapped/total step counts, one entry per rule with
        ``counts``, ``triggered_steps`` and ``findings``, and a ``note`` on how
        to read statuses.

    Raises:
        ArtifactError: No ``analysis.json``, a stale or malformed sidecar, or an
            unknown rule id or status filter.
    """
    analysis = checked_analysis(run)
    if analysis is None:
        raise ArtifactError(
            f"{run.path} has no analysis.json; run 'shardscope analyze --run {run.path} "
            "--threshold KEY=VALUE' first. Findings are read from the persisted sidecar only."
        )
    _check_filters(rule_ids, statuses)
    evaluated = list(analysis.get("rules", LEGACY_RULES))
    config = analysis.get("rule_config")
    rule_config = RuleConfig.from_mapping(config) if config is not None else RuleConfig()
    by_rule: dict[str, list[dict[str, Any]]] = {rule: [] for rule in evaluated}
    for finding in analysis["findings"]:
        by_rule.setdefault(finding["rule_id"], []).append(finding)

    rules = [
        _group(
            spec,
            by_rule.get(spec.rule_id, []),
            rule_config,
            statuses,
            evaluated=spec.rule_id in evaluated,
        )
        for spec in RULES
        if not rule_ids or spec.rule_id in rule_ids
    ]
    mapped = sum(1 for row in analysis["steps"] if row["boundary"] is not None)
    return {
        "run": str(run.path),
        "run_id": run.run_id,
        "rank": analysis["rank"],
        "rules_evaluated": evaluated,
        "rule_config": config,
        "mapped_steps": mapped,
        "total_steps": len(analysis["steps"]),
        "source": analysis["source"],
        "rules": rules,
        "note": (
            "Statuses are configured comparisons over persisted measurements; a triggered "
            "status is a possible explanation to investigate, not a confirmed cause, and a "
            "below_threshold status does not establish health. Missing measurements are "
            "not zero."
        ),
    }


def _check_filters(rule_ids: Sequence[str], statuses: Sequence[str]) -> None:
    unknown = sorted(set(rule_ids) - set(_SPECS))
    if unknown:
        raise ArtifactError(f"unknown rule id(s): {', '.join(unknown)}; known: {', '.join(_SPECS)}")
    unknown = sorted(set(statuses) - set(known_statuses()))
    if unknown:
        raise ArtifactError(
            f"unknown status(es): {', '.join(unknown)}; known: {', '.join(known_statuses())}"
        )


def _group(spec, findings, rule_config, statuses, *, evaluated):
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding["status"]] = counts.get(finding["status"], 0) + 1
    settings = {
        item["key"]: getattr(rule_config, item["key"])
        for item in spec.config
        if getattr(rule_config, item["key"]) is not None
    }
    return {
        "rule_id": spec.rule_id,
        "evaluated": evaluated,
        "configured": bool(findings) and findings[0]["status"] != "unconfigured",
        "config": settings,
        "granularity": spec.granularity,
        "description": spec.description,
        "counts": counts,
        "triggered_steps": sorted(
            {step for f in findings if f["status"] in TRIGGERED_STATUSES for step in f["steps"]}
        ),
        "next_investigation": spec.next_investigation,
        "findings": [f for f in findings if not statuses or f["status"] in statuses],
    }


def summarize_rank_findings(
    root: str | Path,
    *,
    rule_ids: Sequence[str] = (),
    statuses: Sequence[str] = (),
) -> dict[str, Any]:
    """Group a validated ``ranks.json``'s findings per cross-rank rule.

    The comparison is validated against the rank directories first (digests
    and a full recomputation), exactly as ``summarize_findings`` validates a
    run's sidecar. Nothing is evaluated here.
    """
    from shardscope.ranks import load_ranks

    _check_filters(rule_ids, statuses)
    payload = load_ranks(root)
    config = payload.get("rule_config")
    rule_config = RuleConfig.from_mapping(config) if config is not None else RuleConfig()
    by_rule: dict[str, list[dict[str, Any]]] = {}
    for finding in payload["findings"]:
        by_rule.setdefault(finding["rule_id"], []).append(finding)
    rules = [
        _group(spec, by_rule.get(spec.rule_id, []), rule_config, statuses, evaluated=True)
        for spec in RANK_RULES
        if not rule_ids or spec.rule_id in rule_ids
    ]
    compared = sum(1 for row in payload["steps"] if row["ratio"] is not None)
    return {
        "root": str(Path(root)),
        "world_size": payload["world_size"],
        "ranks_recorded": [item["rank"] for item in payload["ranks"]],
        "missing_ranks": payload["missing_ranks"],
        "compatibility_problems": payload["compatibility"]["problems"],
        "rules_evaluated": payload["rules"],
        "rule_config": config,
        "compared_steps": compared,
        "total_steps": len(payload["steps"]),
        "rules": rules,
        "note": (
            "Statuses are configured comparisons over each rank's recorded step durations; a "
            "triggered status says the ranks' step times differ, not which rank caused it, and "
            "a below_threshold status does not establish balanced work. Durations are compared, "
            "never timestamps across machines."
        ),
    }


def render_rank_findings_text(summary: dict[str, Any]) -> str:
    """Render :func:`summarize_rank_findings` output as plain text."""
    lines = [
        f"cross-rank diagnostics for {summary['root']} (ranks.json, world size "
        f"{summary['world_size']}, ranks recorded {summary['ranks_recorded']}): "
        f"{summary['compared_steps']} of {summary['total_steps']} training step(s) compared",
        "rules evaluated: " + ", ".join(summary["rules_evaluated"]),
    ]
    config = summary["rule_config"]
    lines.append(
        "thresholds: "
        + (", ".join(f"{k}={_number(v)}" for k, v in config.items()) if config else "none")
    )
    if summary["missing_ranks"]:
        lines.append(f"missing rank directories: {summary['missing_ranks']}")
    for problem in summary["compatibility_problems"]:
        lines.append(f"incompatible: {problem}")
    _render_rule_blocks(lines, summary["rules"], command="shardscope ranks")
    lines.append("")
    lines.append(summary["note"])
    return "\n".join(lines) + "\n"


def _render_rule_blocks(lines: list[str], rules: list[dict[str, Any]], *, command: str) -> None:
    for rule in rules:
        lines.append("")
        if not rule["evaluated"]:
            lines.append(
                f"{rule['rule_id']}: not evaluated (this analysis.json predates the rule; "
                "re-run shardscope analyze --overwrite to evaluate it)"
            )
            continue
        if not rule["configured"]:
            keys = ", ".join(item["key"] for item in _SPECS[rule["rule_id"]].config)
            lines.append(f"{rule['rule_id']}: unconfigured (set --threshold {keys} on {command})")
            continue
        settings = ", ".join(f"{k}={_number(v)}" for k, v in rule["config"].items())
        counts = ", ".join(f"{n} {status}" for status, n in sorted(rule["counts"].items()))
        lines.append(f"{rule['rule_id']} [{settings}]: {counts}")
        by_status: dict[str, list[dict[str, Any]]] = {}
        for finding in rule["findings"]:
            by_status.setdefault(finding["status"], []).append(finding)
        for status in sorted(by_status, key=lambda s: (s not in TRIGGERED_STATUSES, s)):
            findings = by_status[status]
            steps = [step for f in findings for step in f["steps"]]
            lines.append(f"  {status}: steps {_step_ranges(steps)}")
            if status in TRIGGERED_STATUSES:
                for finding in findings:
                    lines.append(
                        f"    steps {_step_ranges(finding['steps'])}: {_compared(finding)}; "
                        f"evidence {_refs(finding['evidence'])}"
                    )
                lines.append(f"    {findings[0]['interpretation']}")
            else:
                reasons = list(dict.fromkeys(f["interpretation"] for f in findings))
                more = " (and other reasons)" if len(reasons) > 1 else ""
                lines.append(f"    {reasons[0]}{more}")


def render_findings_text(summary: dict[str, Any]) -> str:
    """Render :func:`summarize_findings` output as plain text.

    One block per rule; triggered findings list their compared values and
    evidence references, other statuses show their recorded interpretation.

    Args:
        summary: The dict returned by :func:`summarize_findings`.

    Returns:
        Text ending in a newline, ready for stdout.
    """
    lines = [
        f"diagnostics for {summary['run']} (analysis.json, rank {summary['rank']}): "
        f"{summary['mapped_steps']} of {summary['total_steps']} training step(s) mapped",
        "rules evaluated: " + ", ".join(summary["rules_evaluated"]),
    ]
    config = summary["rule_config"]
    lines.append(
        "thresholds: "
        + (", ".join(f"{k}={_number(v)}" for k, v in config.items()) if config else "none")
    )
    _render_rule_blocks(lines, summary["rules"], command="shardscope analyze")
    lines.append("")
    lines.append(summary["note"])
    lines.append(
        "Evidence references are original event-array indices in the named trace files; "
        "source SHA-256 digests are in analysis.json."
    )
    return "\n".join(lines) + "\n"
