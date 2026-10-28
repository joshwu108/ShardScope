"""Render HTML, JSON and CSV views of recorded run artifacts.

Rendering adds no measurements or statistics. Profiling status remains attached
to the recorded values so diagnostic timings cannot be mistaken for throughput.

HTML includes provenance, field availability, recorded summaries and step rows.
JSON preserves a loadable run with availability metadata. CSV preserves the
producer's per-step representation. Unsupported profiled schemas are rejected
by the artifact reader before rendering.
"""

from __future__ import annotations

import json
from pathlib import Path

from shardscope.artifacts import (
    ENVIRONMENT_JSON,
    STEPS_CSV,
    SUMMARY_JSON,
    RunArtifacts,
    replace_file_contents,
    steps_csv_text,
)
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

    Three things happen in a fixed order, and the order is the point:

    1. The document is rendered in full. A rendering failure therefore
       leaves no half-written report, and no directory, behind.
    2. ``out`` is checked against the run's own artifacts. ``--out
       runs/x/steps.csv`` would replace the evidence with a rendering of it,
       so that is refused before anything is opened -- through symlinks and
       relative aliases too, since those are the same bytes on disk.
    3. The text replaces ``out`` whole, via a sibling temporary file. A
       write that fails partway leaves a previously valid report intact
       rather than truncated (:func:`~shardscope.artifacts.replace_file_contents`
       states exactly what that does and does not guarantee).
    """
    text = render(run, fmt, generated_at=generated_at, max_rows=max_rows)
    target = Path(out)
    _refuse_source_collision(run, target)
    try:
        replace_file_contents(target, text)
    except OSError as exc:
        # A directory in the way, or a file where a directory belongs, is the
        # user's mistake and deserves a sentence rather than a traceback.
        raise ReportError(f"cannot write the report to {target}: {exc.strerror or exc}.") from exc
    return target


def _refuse_source_collision(run: RunArtifacts, target: Path) -> None:
    """Refuse an output path that is one of the run's recorded artifact files.

    Candidates are the files the run listed when it was loaded plus the
    three the layout always names, so a run rebuilt from a JSON report with
    no file list is still protected. Only an existing file can collide; a
    new file inside the run directory is the caller's choice and becomes
    part of the directory listing on the next load.
    """
    if not target.exists():
        return
    names = {*run.artifact_names, STEPS_CSV, ENVIRONMENT_JSON, SUMMARY_JSON}
    for name in sorted(names):
        candidate = run.path / name
        try:
            collides = candidate.is_file() and target.samefile(candidate)
        except OSError as exc:
            raise ReportError(
                f"cannot check whether --out {target} is an artifact of the run being "
                f"reported ({candidate}): {exc.strerror or exc}. Choose an output path "
                "outside the run directory."
            ) from exc
        if collides:
            raise ReportError(
                f"refusing to write the report over {candidate}: --out names an artifact "
                f"of the run being reported ({run.path}). That file is part of the evidence "
                "the report describes; writing the report over it would replace the evidence "
                "with a description of it. Choose an output path outside the run directory."
            )
