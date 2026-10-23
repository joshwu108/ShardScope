"""Render a loaded run as a single self-contained HTML document.

Formatting only. Every number in the output came from an artifact on disk:
nothing here derives a statistic, converts an absent measurement into a
value, or adds overlapping durations together (PRD v2.0 section 8
"Guardrails"). Where a field is absent, the document says so and says why --
that text comes from :mod:`shardscope.availability`.

Three properties this module is written to keep:

* **Escaped.** Everything interpolated into the document goes through
  :func:`html.escape`, including provenance read out of ``environment.json``
  and file names read off disk. A trace, a config path or a device name can
  carry arbitrary text from the profiled program, and a report is often the
  artifact that gets shared.
* **Self-contained.** No scripts, no stylesheets, no network references. The
  file opens from disk with nothing fetched.
* **Deterministic.** The same run and the same ``generated_at`` render
  byte-identically, so two reports can be diffed.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterable, Mapping
from typing import Any

from shardscope.artifacts import ENVIRONMENT_JSON, STEPS_CSV, SUMMARY_JSON, RunArtifacts
from shardscope.availability import NOT_OBSERVED, NOT_RECORDED, RECORDED, field_statuses
from shardscope.errors import ReportError

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
    """State what kind of numbers follow, in the words of requirement SS-04."""
    status = run.profiled
    if status is None:
        text = (
            f"<strong>Profiling status was not recorded.</strong> This run's "
            f"{_esc(ENVIRONMENT_JSON)} has no <code>extra.profiled</code> flag, so "
            "whether a profiler was attached is unknown. Requirement SS-04 keeps "
            "diagnostic and throughput figures apart, and an unknown status cannot "
            "be read as either; treat the step times below as unclassified."
        )
    elif status:
        text = (
            "<strong>Profiled diagnostic run.</strong> The step times below carry "
            "profiler overhead. Requirement SS-04: they are diagnostic evidence and "
            "must not be quoted as throughput results."
        )
    else:
        text = (
            "<strong>Unprofiled run.</strong> No profiler was attached. The step "
            "times below still come from a timed fixture run rather than from the "
            "warm, repeated benchmark harness (GATE 7.1, unimplemented), so under "
            "requirement SS-04 they are not throughput results either."
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
            '<p class="sub">The core report fields of PRD v2.0 section 5. An absent '
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
        f'<p class="sub">From {_esc(ENVIRONMENT_JSON)} (PRD v2.0 section 4, run '
        "contract), preserved as recorded.</p>",
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
        "result (requirement SS-04).",
        "GPU kernels, host-to-device copies and collectives overlap. Their "
        "durations do not sum to step time, so this report does not sum them and "
        "derives no &quot;percent wasted&quot; figure from them.",
        "A field this run could not observe is shown as <code>not_observed</code> "
        "with the reason; a field nobody captured is shown as not recorded. "
        "Neither is rendered as zero.",
        "No diagnosis, threshold or confidence label appears in this report. "
        "Those are open decisions (GATE 4.1 thresholds, GATE 4.2 confidence "
        "vocabulary), and inventing one here would put words in the author's mouth.",
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
