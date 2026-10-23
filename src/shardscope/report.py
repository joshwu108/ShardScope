"""Report rendering: HTML / JSON / CSV views over a loaded run.

Rendering is pure formatting over already-loaded artifacts. It adds no
analysis, computes no statistic the run did not record, and labels every
figure with whether the run it came from was profiled, so a diagnostic
number is never mistaken for a throughput result (requirement SS-04).

The three views are deliberately different shapes of the same facts:

``html``
    the readable document -- provenance, what the run does and does not
    record, the statistics it wrote, and its per-step rows;
``json``
    the same run plus the per-field availability statuses, round-trippable
    back into a :class:`~shardscope.artifacts.RunArtifacts`;
``csv``
    the per-step rows exactly as the producer wrote them.

Scope boundary: all three render the artifact triple
``examples/train_single.py`` writes. The profiled per-step schema is GATE
3.1/3.2 and has no loader yet, so it has no renderer here either.
"""

from __future__ import annotations

import json
from pathlib import Path

from shardscope.artifacts import RunArtifacts, steps_csv_text
from shardscope.availability import field_statuses
from shardscope.errors import ReportError
from shardscope.html_report import DEFAULT_MAX_ROWS, render_html

__all__ = [
    "DEFAULT_MAX_ROWS",
    "REPORT_FORMATS",
    "format_for_path",
    "render",
    "render_html",
    "render_json",
    "render_steps_csv",
    "write_report",
]

#: Output formats, in the order ``--help`` lists them.
REPORT_FORMATS = ("html", "json", "csv")

#: Output suffix -> format. ``.htm`` is accepted because people type it.
_SUFFIX_FORMATS = {".html": "html", ".htm": "html", ".json": "json", ".csv": "csv"}


def render_json(run: RunArtifacts, *, indent: int = 2) -> str:
    """Serialize ``run`` plus its per-field availability statuses.

    ``RunArtifacts.from_dict`` reads this back, so a JSON report is a
    loadable artifact rather than a dead end. An absent measurement stays
    ``null``: a consumer must be able to tell "not recorded" from a value.

    Caveat, deliberately not papered over: a diverged run can record a loss
    of ``nan`` or ``inf``, and :func:`json.dumps` spells those as the
    non-standard tokens ``NaN``/``Infinity``. Python reads them back, strict
    parsers elsewhere will not. The alternative -- writing ``null`` -- would
    turn a recorded measurement into a missing one, which is worse.
    """
    payload = {
        **run.to_dict(),
        "availability": [status.to_dict() for status in field_statuses(run)],
    }
    return json.dumps(payload, indent=indent, sort_keys=True) + "\n"


def render_steps_csv(run: RunArtifacts) -> str:
    """Return the per-step rows as CSV text, byte-identical to the producer.

    Shares its writer with :func:`~shardscope.artifacts.write_steps_csv`, so
    rendering a run to CSV and copying its per-step file cannot disagree
    about quoting.
    """
    return steps_csv_text(run.steps)


def render(
    run: RunArtifacts,
    fmt: str,
    *,
    generated_at: str | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> str:
    """Render ``run`` in ``fmt``, raising :class:`ReportError` for an unknown one.

    ``max_rows`` only truncates the HTML table -- the json and csv views
    always carry every row -- but a non-positive value is refused whatever
    the format, because the flag is accepted whatever the format and
    silently ignoring it for two of three would be a trap.
    """
    if max_rows <= 0:
        raise ReportError(
            f"max_rows must be positive, got {max_rows}. A report with no step rows "
            "would read as a run that measured nothing."
        )
    if fmt == "html":
        return render_html(run, generated_at=generated_at, max_rows=max_rows)
    if fmt == "json":
        return render_json(run)
    if fmt == "csv":
        return render_steps_csv(run)
    raise ReportError(f"unknown report format {fmt!r}. Choose one of: {', '.join(REPORT_FORMATS)}.")


def format_for_path(path: str | Path) -> str:
    """Infer the report format from an output path's suffix."""
    suffix = Path(path).suffix.lower()
    try:
        return _SUFFIX_FORMATS[suffix]
    except KeyError as exc:
        known = ", ".join(sorted(_SUFFIX_FORMATS))
        raise ReportError(
            f"cannot tell the report format from {Path(path).name!r}: unknown suffix "
            f"{suffix or '(none)'}. Use one of {known}, or pass --format explicitly "
            f"({', '.join(REPORT_FORMATS)})."
        ) from exc


def write_report(
    run: RunArtifacts,
    out: str | Path,
    *,
    fmt: str,
    generated_at: str | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> Path:
    """Render ``run`` and write it to ``out``, returning the path written.

    The document is rendered in full before anything touches the
    filesystem, so a rendering failure leaves no half-written report (and no
    directory) behind to be mistaken for a real one.
    """
    text = render(run, fmt, generated_at=generated_at, max_rows=max_rows)
    target = Path(out)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # newline="" and an explicit encoding: the CSV view promises bytes
        # identical to the producer's file, which rules out newline
        # translation and a locale-dependent encoding.
        with target.open("w", newline="", encoding="utf-8") as handle:
            handle.write(text)
    except OSError as exc:
        # A directory in the way, or a file where a directory belongs, is the
        # user's mistake and deserves a sentence rather than a traceback.
        raise ReportError(f"cannot write the report to {target}: {exc.strerror or exc}.") from exc
    return target
