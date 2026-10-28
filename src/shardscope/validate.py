"""Check that a run directory is a recording that can be reported.

``shardscope validate --run DIR`` is the artifact reader exposed on its own:
it loads the directory exactly as ``report`` does and says whether that
worked, what the run contains, and what it does not. It renders nothing,
writes nothing and never touches the directory it reads.

It accepts what ``report`` accepts, with one deliberate difference. A step
count that disagrees between ``summary.json`` and ``steps.csv`` is *shown*
by a report, because a report describes what is on disk; here it is a
failure, because the question being answered is "can this directory be
trusted as a recording of one run", and two artifacts that describe
different runs mean it cannot. Neither number is adjusted either way.

Two kinds of finding, kept apart on purpose:

``errors``
    the directory cannot be vouched for -- it did not load, or its
    artifacts disagree. Exit 2. There is at most one, because loading stops
    at the first defect.
``warnings``
    the directory loaded and can be reported, but something a reader would
    want to know is missing or mixed: no readable recorded step count, an
    unrecorded profiling status, rows measured under different policies.
    Exit 0. A warning is never promoted to a default value.

The result also carries the per-field availability statuses from
:mod:`shardscope.availability`, so automation can see which report fields
this run records without rendering a report. It carries no statistic: a
validation result that quoted a median would be a report by another name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shardscope.artifacts import ENVIRONMENT_JSON, STEPS_CSV, SUMMARY_JSON, load_run
from shardscope.availability import (
    NOT_OBSERVED,
    NOT_RECORDED,
    RECORDED,
    FieldStatus,
    field_statuses,
)
from shardscope.errors import ArtifactError

__all__ = ["ValidationResult", "validate_run"]


@dataclass(frozen=True)
class ValidationResult:
    """What ``validate`` found, in one shape for both the text and JSON views.

    ``rows``, ``step_range``, ``profiled``, ``summary_steps`` and
    ``artifact_names`` are ``None``/empty when the run did not load; a
    consumer must read them as unknown, not as zero or as agreement.
    """

    path: str
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    rows: int | None
    step_range: tuple[int, int] | None
    synchronized: tuple[bool, ...]
    data_residency: tuple[str, ...]
    profiled: bool | None
    summary_present: bool | None
    summary_steps: int | None
    artifact_names: tuple[str, ...]
    availability: tuple[FieldStatus, ...]

    @property
    def summary_agrees_with_rows(self) -> bool | None:
        """True/False when both counts are readable, else ``None`` (unknown)."""
        if self.rows is None or self.summary_steps is None:
            return None
        return self.summary_steps == self.rows

    def to_dict(self) -> dict[str, Any]:
        steps = None
        if self.rows is not None and self.step_range is not None:
            steps = {
                "rows": self.rows,
                "first_step": self.step_range[0],
                "last_step": self.step_range[1],
                "synchronized": list(self.synchronized),
                "data_residency": list(self.data_residency),
            }
        summary: dict[str, Any] | None = None
        if self.summary_present is not None:
            summary = {"present": self.summary_present}
            if self.summary_present:
                summary["steps"] = self.summary_steps
                summary["agrees_with_rows"] = self.summary_agrees_with_rows
        return {
            "path": self.path,
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "steps": steps,
            "summary": summary,
            "profiled": self.profiled,
            "artifact_names": list(self.artifact_names),
            "availability": [status.to_dict() for status in self.availability],
        }

    def to_json(self) -> str:
        """Strict JSON: ``allow_nan=False`` so a damaged artifact value can
        never leak into the document as a ``NaN`` token. Nothing in the
        result is a float, so this is a guard rather than a choice."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"

    def to_text(self) -> str:
        """The human view: a verdict line and one line per artifact."""
        if not self.ok:
            return "\n".join(f"error: {error}" for error in self.errors) + "\n"

        assert self.rows is not None and self.step_range is not None  # ok implies loaded
        first, last = self.step_range
        lines = [
            f"valid run: {self.path}",
            f"  {STEPS_CSV:<17} {self.rows} row(s), steps {first}..{last}, "
            f"synchronized={_spell(self.synchronized)}, "
            f"data_residency={_spell(self.data_residency)}",
            f"  {ENVIRONMENT_JSON:<17} present; extra.profiled={_spell_optional(self.profiled)}",
            f"  {SUMMARY_JSON:<17} {self._summary_text()}",
            f"  {'files':<17} {', '.join(self.artifact_names) or 'none listed'}",
            f"  {'fields':<17} {self._availability_text()}",
        ]
        if self.warnings:
            lines.extend(f"  {'warning':<17} {warning}" for warning in self.warnings)
        else:
            lines.append(f"  {'warnings':<17} none")
        return "\n".join(lines) + "\n"

    def _summary_text(self) -> str:
        if not self.summary_present:
            return "absent (optional; the run recorded no statistics, and none are computed here)"
        if self.summary_steps is None:
            return "present; no readable integer 'steps' count"
        return f"present; steps={self.summary_steps}, agrees with {STEPS_CSV}"

    def _availability_text(self) -> str:
        counts = {RECORDED: 0, NOT_OBSERVED: 0, NOT_RECORDED: 0}
        for status in self.availability:
            counts[status.state] = counts.get(status.state, 0) + 1
        return (
            f"{counts[RECORDED]} recorded, {counts[NOT_OBSERVED]} {NOT_OBSERVED}, "
            f"{counts[NOT_RECORDED]} not recorded (shardscope report explains each)"
        )


def validate_run(path: str | Path) -> ValidationResult:
    """Load ``path`` the way ``report`` does and describe what was found.

    Never raises :class:`ArtifactError`: a directory that will not load is a
    result with ``ok=False`` and the loader's message as its one error, so
    the JSON view can carry it. Anything that is not an ``ArtifactError`` is
    a bug and propagates.
    """
    try:
        run = load_run(path)
    except ArtifactError as exc:
        return ValidationResult(
            path=str(path),
            ok=False,
            errors=(str(exc),),
            warnings=(),
            rows=None,
            step_range=None,
            synchronized=(),
            data_residency=(),
            profiled=None,
            summary_present=None,
            summary_steps=None,
            artifact_names=(),
            availability=(),
        )

    errors: list[str] = []
    warnings: list[str] = []
    rows = len(run.steps)
    recorded = run.summary_step_count

    if run.summary is not None and recorded is None:
        warnings.append(
            f"{SUMMARY_JSON} has no readable integer 'steps' count (recorded value: "
            f"{run.summary.get('steps')!r}), so the {rows} row(s) in {STEPS_CSV} cannot be "
            "checked against it."
        )
    elif recorded is not None and recorded != rows:
        errors.append(
            f"{run.path}: {SUMMARY_JSON} records steps={recorded} but {STEPS_CSV} has {rows} "
            "row(s). The two artifacts were written separately, so one of them does not "
            "describe this run; neither is adjusted. Treat the directory as unrecorded and "
            "repeat the run."
        )

    if run.profiled is None:
        warnings.append(
            f"{ENVIRONMENT_JSON} does not record extra.profiled, so whether a profiler was "
            "attached is unknown. The step times cannot be classed as diagnostic or as "
            "throughput figures (measurement classification)."
        )

    synchronized = _distinct(row.synchronized for row in run.steps)
    residency = _distinct(row.data_residency for row in run.steps)
    if len(synchronized) > 1:
        warnings.append(
            f"{STEPS_CSV} rows disagree about synchronized ({_spell(synchronized)}); an "
            "unsynchronized timing measures launch rather than execution, so these rows "
            "are not comparable with each other."
        )
    if len(residency) > 1:
        warnings.append(
            f"{STEPS_CSV} rows disagree about data_residency ({_spell(residency)}); steps "
            "measured under different residency are not comparable with each other."
        )

    return ValidationResult(
        path=str(run.path),
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        rows=rows,
        step_range=(run.steps[0].step, run.steps[-1].step),
        synchronized=synchronized,
        data_residency=residency,
        profiled=run.profiled,
        summary_present=run.summary is not None,
        summary_steps=recorded,
        artifact_names=run.artifact_names,
        availability=field_statuses(run),
    )


def _distinct(values) -> tuple:
    """Distinct values in first-seen order; the order rows were recorded in."""
    seen: dict = {}
    for value in values:
        seen.setdefault(value, None)
    return tuple(seen)


def _spell(values) -> str:
    """Spell values the way the JSON artifacts do, joined with commas."""
    return ", ".join(_spell_optional(value) for value in values)


def _spell_optional(value: Any) -> str:
    if value is None:
        return "not recorded"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
