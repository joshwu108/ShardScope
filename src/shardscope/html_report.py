"""Render a run as a self-contained HTML document.

All interpolated provenance, paths and filenames are HTML-escaped. No scripts,
external stylesheets or network requests are required to open the result.
Identical artifacts and generation timestamps produce identical output.

Rendering preserves recorded values and availability states without deriving
statistics or adding overlapping activity durations.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from shardscope.artifacts import ENVIRONMENT_JSON, STEPS_CSV, SUMMARY_JSON, RunArtifacts
from shardscope.attribution import ATTRIBUTED_KERNEL_METRIC
from shardscope.availability import NOT_OBSERVED, NOT_RECORDED, RECORDED, field_statuses
from shardscope.errors import ReportError
from shardscope.parse import summarize_indices

__all__ = ["DEFAULT_MAX_ROWS", "render_html"]

#: Rows shown inline before the table is truncated. The full data is always
#: in steps.csv, and a truncated table says exactly how much it is hiding.
DEFAULT_MAX_ROWS = 200

_STATE_LABELS = {
    RECORDED: "recorded",
    NOT_RECORDED: "not recorded",
    NOT_OBSERVED: NOT_OBSERVED,
}

_STYLE = """
body { font-family: -apple-system, system-ui, sans-serif; line-height: 1.45;
       margin: 2rem auto; max-width: 60rem; padding: 0 1rem; color: #16191d; }
h1 { font-size: 1.6rem; margin-bottom: 0.2rem; }
h2 { font-size: 1.15rem; margin-top: 2.2rem; border-bottom: 1px solid #d8dde3;
     padding-bottom: 0.3rem; }
p.sub { color: #5a6472; margin-top: 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { border: 1px solid #d8dde3; padding: 0.35rem 0.5rem; text-align: left;
         vertical-align: top; }
th { background: #f3f5f8; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
code, td.mono { font-family: ui-monospace, Menlo, monospace; font-size: 0.85rem; }
.callout { border-left: 4px solid #5a6472; background: #f3f5f8; padding: 0.7rem 1rem;
           margin: 1rem 0; }
.state-recorded { color: #1b5e20; font-weight: 600; }
.state-absent { color: #8a4b00; font-weight: 600; }
ul.notes { font-size: 0.9rem; color: #343b45; }
""".strip()


def render_html(
    run: RunArtifacts,
    *,
    generated_at: str | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> str:
    """Return a complete HTML report for ``run``.

    ``generated_at`` is interpolated verbatim (escaped) when given and
    omitted otherwise, which keeps the output deterministic for tests and
    for diffing two reports of the same run.
    """
    if max_rows <= 0:
        raise ReportError(
            f"max_rows must be positive, got {max_rows}. A report with no step rows "
            "would read as a run that measured nothing."
        )

    title = f"ShardScope run report: {run.run_id}"
    body = "\n".join(
        (
            _header(run, title=title, generated_at=generated_at),
            _profiling_callout(run),
            _availability_section(run),
            _summary_section(run),
            _analysis_section(run),
            _steps_section(run, max_rows=max_rows),
            _provenance_section(run),
            _notes_section(),
        )
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{_esc(title)}</title>\n<style>\n{_STYLE}\n</style>\n</head>\n<body>\n"
        f"{body}\n</body>\n</html>\n"
    )


def _header(run: RunArtifacts, *, title: str, generated_at: str | None) -> str:
    lines = [f"<h1>{_esc(title)}</h1>"]
    subtitle = f"run directory <code>{_esc(str(run.path))}</code>"
    if generated_at is not None:
        subtitle += f" &middot; report generated {_esc(generated_at)}"
    lines.append(f'<p class="sub">{subtitle}</p>')
    return "\n".join(lines)


def _profiling_callout(run: RunArtifacts) -> str:
    """Explain whether recorded timings include profiling overhead."""
    status = run.profiled
    if status is None:
        text = (
            f"<strong>Profiling status was not recorded.</strong> This run's "
            f"{_esc(ENVIRONMENT_JSON)} has no <code>extra.profiled</code> flag, so "
            "whether a profiler was attached is unknown. Treat these step times "
            "as unclassified; their profiling overhead is unknown."
        )
    elif status:
        text = (
            "<strong>Profiled diagnostic run.</strong> The step times below carry "
            "profiler overhead. They are diagnostic evidence and "
            "must not be quoted as throughput results."
        )
    else:
        text = (
            "<strong>Unprofiled run.</strong> No profiler was attached. The step "
            "times below still come from a timed fixture run rather than from the "
            "warm, repeated benchmark harness, which is not implemented. "
            "These fixture timings are not throughput results."
        )
    return f'<div class="callout">{text}</div>'


def _availability_section(run: RunArtifacts) -> str:
    rows = []
    for status in field_statuses(run):
        css = "state-recorded" if status.state == RECORDED else "state-absent"
        label = _STATE_LABELS.get(status.state, status.state)
        rows.append(
            f'<tr><td class="mono">{_esc(status.field)}</td>'
            f'<td><span class="{css}">{_esc(label)}</span></td>'
            f"<td>{_esc(status.detail)}</td></tr>"
        )
    return "\n".join(
        (
            "<h2>What this run records</h2>",
            '<p class="sub">Measurement availability for this run. An absent '
            "field is absent, with its reason; it is never shown as zero.</p>",
            _table(("field", "status", "why"), rows),
        )
    )


def _summary_section(run: RunArtifacts) -> str:
    heading = "<h2>Recorded statistics</h2>"
    if run.summary is None:
        return "\n".join(
            (
                heading,
                f'<p class="sub">{_esc(SUMMARY_JSON)} is not present in this run '
                "directory, so no recorded statistics are shown. This report does "
                "not compute replacements for a statistic the run did not write.</p>",
            )
        )

    rows = [
        f'<tr><td class="mono">{_esc(key)}</td><td class="num">{_esc(_cell(value))}</td></tr>'
        for key, value in sorted(run.summary.items())
    ]
    note = (
        f'<p class="sub">As written by the run in {_esc(SUMMARY_JSON)}, verbatim. '
        "These describe the timed run above, not a warm unprofiled benchmark.</p>"
    )
    return "\n".join((heading, note, _table(("statistic", "value"), rows)))


def _steps_section(run: RunArtifacts, *, max_rows: int) -> str:
    shown = run.steps[:max_rows]
    rows = [
        "<tr>"
        f'<td class="num">{_esc(_cell(row.step))}</td>'
        f'<td class="num">{_esc(_cell(row.loss))}</td>'
        f'<td class="num">{_esc(_cell(row.step_time_ms))}</td>'
        f"<td>{_esc(row.data_residency)}</td>"
        # str() rather than the JSON spelling used in the provenance tables:
        # this table mirrors steps.csv, which writes True/False.
        f"<td>{_esc(str(row.synchronized))}</td>"
        "</tr>"
        for row in shown
    ]
    lines = ["<h2>Per-step records</h2>", _counts_paragraph(run)]
    if len(shown) != len(run.steps):
        lines.append(
            f'<p class="sub">Showing the first {len(shown)} of {len(run.steps)} '
            f"recorded steps; the complete per-step data is in "
            f"<code>{_esc(STEPS_CSV)}</code>.</p>"
        )
    lines.append(_table(("step", "loss", "step_time_ms", "data_residency", "synchronized"), rows))
    return "\n".join(lines)


def _counts_paragraph(run: RunArtifacts) -> str:
    """Show both recorded counts side by side, and flag a disagreement.

    The row count and ``summary.json``'s ``steps`` are written separately by
    the producer. Showing both -- and naming a difference instead of picking
    a winner -- keeps a damaged or hand-edited run directory visible.
    """
    recorded = run.summary_step_count
    text = f"{len(run.steps)} row(s) in <code>{_esc(STEPS_CSV)}</code>"
    if recorded is None:
        text += f"; no readable <code>steps</code> count in <code>{_esc(SUMMARY_JSON)}</code>"
    else:
        text += f"; <code>{_esc(SUMMARY_JSON)}</code> records steps={_esc(str(recorded))}"
        if recorded != len(run.steps):
            text += (
                " &mdash; <strong>these disagree</strong>. The two artifacts were "
                "written separately, so one of them does not describe this run. "
                "Neither is adjusted here."
            )
    return f'<p class="sub">{text}.</p>'


def _provenance_section(run: RunArtifacts) -> str:
    top = {key: value for key, value in run.environment.items() if key != "extra"}
    rows = [
        f'<tr><td class="mono">{_esc(key)}</td><td class="mono">{_esc(_cell(value))}</td></tr>'
        for key, value in sorted(top.items())
    ]
    extra_rows = [
        f'<tr><td class="mono">{_esc(key)}</td><td class="mono">{_esc(_cell(value))}</td></tr>'
        for key, value in sorted(run.extra.items())
    ]
    lines = [
        "<h2>Provenance</h2>",
        f'<p class="sub">From {_esc(ENVIRONMENT_JSON)}, preserved as recorded.</p>',
        _table(("key", "value"), rows),
    ]
    if extra_rows:
        lines.append("<h2>Run settings</h2>")
        lines.append(
            f'<p class="sub">The <code>extra</code> block of {_esc(ENVIRONMENT_JSON)}: '
            "the knobs this run was started with.</p>"
        )
        lines.append(_table(("key", "value"), extra_rows))
    if run.artifact_names:
        files = ", ".join(f"<code>{_esc(name)}</code>" for name in run.artifact_names)
        lines.append(f'<p class="sub">Files in the run directory: {files}.</p>')
    return "\n".join(lines)


def _notes_section() -> str:
    """The project's measurement guardrails, restated inside the artifact."""
    notes = (
        "Profiled diagnostic runs and unprofiled throughput benchmarks are "
        "different measurements. A profiled step time is never a throughput "
        "result.",
        "GPU kernels, host-to-device copies and collectives overlap. Their "
        "durations do not sum to step time, so this report does not sum them and "
        "derives no &quot;percent wasted&quot; figure from them.",
        "A field this run could not observe is shown as <code>not_observed</code> "
        "with the reason; a field nobody captured is shown as not recorded. "
        "Neither is rendered as zero.",
        "Diagnostic comparisons appear only when persisted analysis supplies configuration "
        "and evidence. Thresholds are not calibrated confidence or confirmed root causes; "
        "host waiting does not establish GPU starvation.",
    )
    items = "\n".join(f"<li>{note}</li>" for note in notes)
    return "\n".join(("<h2>How to read this report</h2>", f'<ul class="notes">{items}</ul>'))


def _table(headers: Iterable[str], rows: list[str]) -> str:
    head = "".join(f"<th>{_esc(header)}</th>" for header in headers)
    body = "\n".join(rows) if rows else '<tr><td colspan="9">(nothing recorded)</td></tr>'
    return f"<table>\n<thead><tr>{head}</tr></thead>\n<tbody>\n{body}\n</tbody>\n</table>"


def _esc(value: str) -> str:
    return html.escape(value, quote=True)


def _cell(value: Any) -> str:
    """Render one artifact value as text without changing what it says.

    ``repr`` for floats, so a reader sees the recorded digits rather than a
    rounded version of them; JSON spelling for ``None`` and booleans, so the
    report and the artifact agree; compact JSON for anything nested.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)  # pragma: no cover - JSON cannot produce another type


def _analysis_section(run: RunArtifacts) -> str:
    """Display only persisted values; evidence anchors identify source indices."""
    analysis = run.analysis
    if analysis is None:
        return ""
    refs = {}

    def evidence_links(evidence):
        links = []
        for ref in evidence:
            key = (ref["file"], ref["event_index"])
            refs.setdefault(key, f"evidence-{len(refs)}")
            label = f"{ref['file']} event[{ref['event_index']}]"
            links.append(f'<a href="#{refs[key]}">{_esc(label)}</a>')
        return ", ".join(links) or "none"

    rows = []
    for step in analysis["steps"]:
        for name, metric in step["measurements"].items():
            value = "unavailable" if metric["value_ms"] is None else _cell(metric["value_ms"])
            rows.append(
                f"<tr><td>{step['training_step']}</td><td>{_esc(name)}</td>"
                f"<td>{_esc(value)}</td><td>{_esc(metric['status'])}</td>"
                f"<td>{_esc(_cell(metric['basis']))}</td><td>{_esc(metric['scope'])}</td>"
                f"<td>{_esc(metric['reason'])}</td><td>{evidence_links(metric['evidence'])}</td></tr>"
            )
    lines = [
        "<h2>Persisted trace analysis</h2>",
        f"<p>analysis.json schema 1; explicit rank {analysis['rank']}. "
        "Host and device durations are distinct, in milliseconds. "
        "Host copy calls are not device transfer time; host wait is not GPU idle time.</p>",
        _table(
            (
                "training step",
                "metric",
                "ms",
                "availability",
                "basis",
                "scope",
                "reason",
                "evidence",
            ),
            rows,
        ),
    ]
    lines.extend(_attribution_section(analysis, evidence_links))
    lines.append("<h2>Configured diagnostic comparisons</h2>")
    for finding in analysis["findings"]:
        lines.append(
            f"<p>{_esc(finding['rule_id'])}: {_esc(finding['status'])}; "
            f"steps {_esc(_cell(finding['steps']))}. {_esc(finding['interpretation'])}</p>"
        )
        comparison = {"inputs": finding["inputs"], "comparison": finding["comparison"]}
        lines.append(f"<pre>{_esc(json.dumps(comparison, indent=2))}</pre>")
        lines.append(f"<p>{evidence_links(finding['evidence'])}</p>")
    warnings = analysis["warnings"]
    lines.append(f"<h2>Analysis warnings ({len(warnings)})</h2>")
    lines.append(
        '<p class="sub">As persisted in analysis.json. Repetitive retained-but-unmeasured '
        "event forms are summarized per trace with representative original event indices; "
        "the raw events remain in the source traces.</p>"
    )
    lines.append("<ul>")
    lines.extend(f"<li>{_esc(warning)}</li>" for warning in warnings)
    lines.append("</ul><h2>Analysis evidence</h2><ul>")
    for (name, index), anchor in refs.items():
        source = str((run.path / name).resolve())
        url = Path(source).as_uri()
        lines.append(
            f'<li id="{anchor}"><a href="{_esc(url)}">{_esc(name)}</a>: '
            f"original event-array index {index}; SHA-256 {_esc(analysis['source'][name])}</li>"
        )
    lines.append("</ul>")
    return "\n".join(lines)


def _attribution_section(analysis: Mapping[str, Any], evidence_links) -> list[str]:
    """Persisted kernel attribution: coverage per trace, timing per step.

    Everything shown is read from ``analysis.json``; counts, unions and
    outcomes are the persisted ones. Unmatched and ambiguous kernels are
    summarized per reason with representative original indices.
    """
    block = analysis.get("kernel_attribution")
    lines = ["<h2>Kernel attribution</h2>"]
    if block is None:
        lines.append(
            '<p class="sub">This analysis.json predates kernel attribution; no kernel event is '
            "tied to a step and no attributed kernel time is shown.</p>"
        )
        return lines
    lines.append(f"<p>{_esc(block['scope'])}. Supported launch forms: ")
    lines[-1] += ", ".join(f"<code>{_esc(x)}</code>" for x in block["supported_launch_events"])
    lines[-1] += (
        f"; kernel category <code>{_esc(block['kernel_category'])}</code>. A launch is tied to "
        "a step only when it lies entirely inside that step's mapped host boundary on the same "
        "process and thread; launches on other host threads are listed, not guessed.</p>"
    )
    trace_rows = []
    outcome_rows = []
    for name in sorted(block["traces"]):
        result = block["traces"][name]
        kernels, launches = result["kernel_events"], result["launch_events"]
        trace_rows.append(
            f'<tr><td class="mono">{_esc(name)}</td>'
            f'<td class="num">{kernels["total"]}</td>'
            f'<td class="num">{kernels["attributed"]}</td>'
            f'<td class="num">{kernels["unmatched"]}</td>'
            f'<td class="num">{kernels["ambiguous"]}</td>'
            f'<td class="num">{launches["attributed"]}</td>'
            f'<td class="num">{launches["outside_mapped_step"]}</td>'
            f'<td class="num">{launches["ambiguous_boundary"]}</td>'
            f'<td class="num">{launches["duplicate_correlation"]}</td>'
            f'<td class="num">{launches["missing_correlation"]}</td></tr>'
        )
        by_kind: dict[tuple[str, str], list[int]] = {}
        for label in ("unmatched", "ambiguous"):
            for record in result[label]:
                by_kind.setdefault((label, record["kind"]), []).append(record["kernel_event_index"])
        for (label, kind), indices in sorted(by_kind.items()):
            outcome_rows.append(
                f'<tr><td class="mono">{_esc(name)}</td><td>{_esc(label)}</td>'
                f'<td class="mono">{_esc(kind)}</td>'
                f"<td>{_esc(summarize_indices(indices))}</td></tr>"
            )
    lines.append(
        _table(
            (
                "trace",
                "kernel events",
                "attributed",
                "unmatched",
                "ambiguous",
                "launches attributed",
                "launches outside mapped steps",
                "launches in ambiguous boundaries",
                "duplicate correlation ids",
                "launches without correlation",
            ),
            trace_rows,
        )
    )
    step_rows = []
    for step in analysis["steps"]:
        metric = step["measurements"].get(ATTRIBUTED_KERNEL_METRIC)
        if metric is None or step["boundary"] is None:
            continue
        coverage = metric.get("coverage", {})
        value = "unavailable" if metric["value_ms"] is None else _cell(metric["value_ms"])
        step_rows.append(
            f'<tr><td class="num">{step["training_step"]}</td>'
            f'<td class="mono">{_esc(step["boundary"]["file"])}</td>'
            f"<td>{_esc(_cell(coverage.get('devices')))}</td>"
            f'<td class="num">{_esc(value)}</td><td>{_esc(metric["status"])}</td>'
            f'<td class="num">{_esc(_cell(coverage.get("step_kernel_events")))}</td>'
            f'<td class="num">{_esc(_cell(coverage.get("step_launch_events")))}</td>'
            f"<td>{_esc(_cell(coverage.get('completeness')))}; "
            f"{_esc(_cell(coverage.get('trace_attributed_kernel_events')))}/"
            f"{_esc(_cell(coverage.get('trace_kernel_events')))} kernel events in trace</td>"
            f"<td>{evidence_links(metric['evidence'])}</td></tr>"
        )
    lines.append(
        '<p class="sub">Per mapped step: the interval union of attributed kernel ranges on one '
        "device. A partial coverage means kernels in the same trace could not be tied to any "
        "step, so the figure is a lower bound on attributed kernel execution, not a complete "
        "compute measurement.</p>"
    )
    lines.append(
        _table(
            (
                "training step",
                "trace",
                "device",
                "attributed kernel ms",
                "availability",
                "kernel events",
                "launch events",
                "coverage",
                "kernel evidence",
            ),
            step_rows,
        )
    )
    if outcome_rows:
        lines.append(
            '<p class="sub">Kernel events that were not attributed, by outcome and reason, with '
            "representative original event indices; each record is in analysis.json.</p>"
        )
        lines.append(_table(("trace", "outcome", "reason", "kernel events"), outcome_rows))
    return lines
