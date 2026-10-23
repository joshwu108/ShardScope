"""Discover, load and validate the artifacts a ShardScope run writes.

A run directory is the unit every later stage consumes:

    runs/<run_id>/steps.csv          per-step records (requirement SS-03)
    runs/<run_id>/environment.json   provenance (PRD v2.0 section 4)
    runs/<run_id>/summary.json       descriptive statistics, as produced

Scope boundary, deliberately narrow: this module understands exactly the
artifact triple ``examples/train_single.py`` already writes. The *profiled*
per-step schema -- ``data_wait_ms``, ``h2d_copy_ms`` and friends -- is an
open student decision (GATE 3.1, GATE 3.2), so there is no loader for it
here. A CSV carrying those columns is **refused**, not silently narrowed to
the columns this module knows: quietly dropping a measured column is how a
report comes to describe a run that did not happen.

Two rules the validation follows:

* Missing is missing. An absent ``summary.json`` loads as ``None`` and an
  unrecorded ``profiled`` flag reads as ``None``; neither becomes ``0`` or
  ``False`` (PRD v2.0 section 8 "Guardrails").
* Provenance is preserved verbatim. ``environment.json`` is kept as the
  mapping it was written as, including keys this version has never heard of,
  so a report cannot silently shed the metadata that makes two runs
  comparable.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shardscope.config import DATA_RESIDENCY_CHOICES
from shardscope.errors import ArtifactError

__all__ = [
    "ENVIRONMENT_JSON",
    "STEPS_CSV",
    "STEP_COLUMNS",
    "SUMMARY_JSON",
    "RunArtifacts",
    "StepRow",
    "discover_runs",
    "load_run",
    "load_steps_csv",
    "looks_like_run",
    "steps_csv_text",
    "write_steps_csv",
]

#: Artifact filenames, as written by examples/train_single.py.
STEPS_CSV = "steps.csv"
ENVIRONMENT_JSON = "environment.json"
SUMMARY_JSON = "summary.json"

#: The per-step CSV contract. ``tests/test_artifacts.py`` pins this equal to
#: ``train_single.STEP_CSV_COLUMNS`` so the two cannot drift apart.
STEP_COLUMNS = (
    "step",
    "loss",
    "step_time_ms",
    "data_residency",
    "synchronized",
)

_TRUE_VALUES = frozenset({"true", "1"})
_FALSE_VALUES = frozenset({"false", "0"})

#: Named in the error for an unrecognized column, so the reader learns that
#: the column belongs to a schema nobody has approved yet rather than
#: thinking this loader is simply out of date.
_UNKNOWN_COLUMN_HINT = (
    "This loader understands only the Week 1 fixture schema. The profiled "
    "per-step schema (data_wait_ms, h2d_copy_ms, ...) is still an open "
    "decision -- GATE 3.1 (which fields are measured versus derived) and "
    "GATE 3.2 (what data_wait_ms means operationally) -- so there is no "
    "reader for it yet, and dropping the column would misreport the run"
)


@dataclass(frozen=True)
class StepRow:
    """One per-step record as recorded on disk.

    ``data_residency`` and ``synchronized`` travel with the number because a
    step time is only comparable against another step time measured the same
    way; they are loaded, never defaulted.
    """

    step: int
    loss: float
    step_time_ms: float
    data_residency: str
    synchronized: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "loss": self.loss,
            "step_time_ms": self.step_time_ms,
            "data_residency": self.data_residency,
            "synchronized": self.synchronized,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StepRow:
        missing = [name for name in STEP_COLUMNS if name not in payload]
        if missing:
            raise ArtifactError(
                f"step record is missing field(s): {', '.join(missing)}. "
                f"Expected: {', '.join(STEP_COLUMNS)}."
            )
        return cls(
            step=payload["step"],
            loss=payload["loss"],
            step_time_ms=payload["step_time_ms"],
            data_residency=payload["data_residency"],
            synchronized=payload["synchronized"],
        )


@dataclass(frozen=True)
class RunArtifacts:
    """Everything one run directory recorded, loaded and validated.

    Frozen because a loaded run is evidence: a renderer that could mutate
    the provenance it is about to print is a renderer that can disagree with
    the files on disk.
    """

    path: Path
    steps: tuple[StepRow, ...]
    environment: dict[str, Any]
    summary: dict[str, Any] | None
    artifact_names: tuple[str, ...]

    @property
    def run_id(self) -> str:
        """The run directory's name -- the handle a report is filed under."""
        return self.path.name

    @property
    def extra(self) -> dict[str, Any]:
        """The ``extra`` block of the environment dump, or ``{}`` if absent.

        Empty means "nothing was recorded there", and callers must treat a
        missing key as unknown rather than as a default value.
        """
        block = self.environment.get("extra")
        return block if isinstance(block, dict) else {}

    @property
    def profiled(self) -> bool | None:
        """Whether this run was profiled; ``None`` when it was not recorded.

        ``None`` is not ``False``. Requirement SS-04 turns on the difference:
        a profiled step time must never be quoted as throughput, and a run
        whose profiling status was never written down cannot be declared
        safe to quote.
        """
        return self.extra.get("profiled")

    @property
    def summary_step_count(self) -> int | None:
        """``steps`` as recorded in ``summary.json``, or ``None`` if absent."""
        if self.summary is None:
            return None
        value = self.summary.get("steps")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return int(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "path": str(self.path),
            "environment": self.environment,
            "summary": self.summary,
            "steps": [row.to_dict() for row in self.steps],
            "artifact_names": list(self.artifact_names),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunArtifacts:
        """Rebuild a run from :meth:`to_dict` output (the JSON report view)."""
        if not isinstance(payload, Mapping):
            raise ArtifactError(
                f"not a serialized run: expected a JSON object, got {type(payload).__name__}."
            )
        missing = [key for key in ("path", "environment", "steps") if key not in payload]
        if missing:
            raise ArtifactError(
                f"not a serialized run: missing key(s) {', '.join(missing)}. "
                "Expected the output of RunArtifacts.to_dict()."
            )
        environment = _validated_environment(payload["environment"], source="<serialized run>")
        summary = payload.get("summary")
        if summary is not None:
            summary = _require_json_object(summary, source="<serialized run>", label="summary")
        return cls(
            path=Path(payload["path"]),
            steps=tuple(StepRow.from_dict(row) for row in payload["steps"]),
            environment=environment,
            summary=summary,
            artifact_names=tuple(payload.get("artifact_names", ())),
        )


def looks_like_run(path: str | Path) -> bool:
    """True when ``path`` is a directory holding a per-step CSV.

    ``steps.csv`` is the marker because it is the one artifact a run cannot
    be reported without.
    """
    target = Path(path)
    return target.is_dir() and (target / STEPS_CSV).is_file()


def discover_runs(root: str | Path) -> tuple[Path, ...]:
    """Return the run directories directly inside ``root``, in name order.

    One level deep and nothing more: that matches how runs are laid out
    (``runs/<run_id>/``) and keeps discovery from wandering into trace
    exports or unrelated trees. A ``root`` that is itself a run is not its
    own child -- use :func:`looks_like_run` for that question.
    """
    target = Path(root)
    if not target.exists():
        raise ArtifactError(
            f"no such directory: {target}. Pass a directory containing run "
            f"directories (each with a {STEPS_CSV})."
        )
    if not target.is_dir():
        raise ArtifactError(f"{target} is not a directory, so it cannot contain runs.")
    return tuple(sorted(child for child in target.iterdir() if looks_like_run(child)))


def load_run(path: str | Path) -> RunArtifacts:
    """Load and validate one run directory.

    Raises :class:`ArtifactError` -- naming the file and, where it applies,
    the row or key -- for a missing directory, a missing required artifact,
    malformed JSON or CSV, or a value that does not match the recorded
    schema.
    """
    run_dir = Path(path)
    if not run_dir.exists():
        raise ArtifactError(
            f"no such run directory: {run_dir}. A run directory holds {STEPS_CSV} "
            f"and {ENVIRONMENT_JSON} (see examples/train_single.py --out-dir)."
        )
    if not run_dir.is_dir():
        raise ArtifactError(f"{run_dir} is not a directory; pass the run directory itself.")

    steps_path = run_dir / STEPS_CSV
    if not steps_path.is_file():
        raise ArtifactError(
            f"{run_dir} has no {STEPS_CSV}, so it is not a run directory.{_discovery_hint(run_dir)}"
        )

    environment_path = run_dir / ENVIRONMENT_JSON
    if not environment_path.is_file():
        raise ArtifactError(
            f"{run_dir} has no {ENVIRONMENT_JSON}. That file is the run's provenance "
            "(PRD v2.0 section 4): without it there is no record of the machine, "
            "versions or settings behind these numbers, and nothing can be judged "
            "comparable to anything. Re-run with --out-dir rather than reporting a "
            "run whose provenance was lost."
        )

    environment = _validated_environment(
        _load_json_object(environment_path, label="environment"), source=str(environment_path)
    )

    summary_path = run_dir / SUMMARY_JSON
    summary = _load_json_object(summary_path, label="summary") if summary_path.is_file() else None

    return RunArtifacts(
        path=run_dir,
        steps=load_steps_csv(steps_path),
        environment=environment,
        summary=summary,
        artifact_names=tuple(sorted(item.name for item in run_dir.iterdir() if item.is_file())),
    )


def _discovery_hint(run_dir: Path) -> str:
    """Name the runs inside ``run_dir``, for when a parent was passed."""
    found = discover_runs(run_dir)
    if not found:
        return ""
    names = ", ".join(path.name for path in found)
    return (
        f" It does contain {len(found)} run director{'y' if len(found) == 1 else 'ies'}: {names}."
    )


def load_steps_csv(path: str | Path) -> tuple[StepRow, ...]:
    """Read and validate a per-step CSV into immutable rows."""
    target = Path(path)
    try:
        # Universal newlines on read, so a CSV written on Windows loads too;
        # splitlines() below then feeds csv.reader one record per line.
        text = target.read_text()
    except FileNotFoundError as exc:
        raise ArtifactError(
            f"no such per-step CSV: {target}. Columns expected: {', '.join(STEP_COLUMNS)}."
        ) from exc
    except IsADirectoryError as exc:
        raise ArtifactError(f"{target} is a directory, not a {STEPS_CSV} file.") from exc
    except OSError as exc:  # pragma: no cover - permissions/IO specific
        raise ArtifactError(f"cannot read {target}: {exc}") from exc

    reader = csv.reader(text.splitlines())
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ArtifactError(
            f"{target} is empty: no header row. Expected: {', '.join(STEP_COLUMNS)}."
        ) from exc

    _validate_header(header, target)

    rows: list[StepRow] = []
    previous_step: int | None = None
    for offset, values in enumerate(reader, start=2):
        if not values:
            continue
        row = _parse_row(values, target=target, row_number=offset)
        if previous_step is not None:
            if row.step == previous_step:
                raise ArtifactError(
                    f"{target} row {offset}: step {row.step} repeats the previous row. "
                    "Each row must describe a different training step."
                )
            if row.step < previous_step:
                raise ArtifactError(
                    f"{target} row {offset}: step {row.step} follows step {previous_step}. "
                    "Steps must be increasing; a bounded capture window may skip "
                    "steps but never reorder them."
                )
        previous_step = row.step
        rows.append(row)

    if not rows:
        raise ArtifactError(
            f"{target} has a header but no step rows. A run with no measurements is a "
            "failed run, not an empty report."
        )
    return tuple(rows)


def _validate_header(header: Sequence[str], target: Path) -> None:
    columns = [name.strip() for name in header]
    duplicates = sorted({name for name in columns if columns.count(name) > 1})
    if duplicates:
        raise ArtifactError(
            f"{target}: duplicate column(s) in the header: {', '.join(duplicates)}."
        )

    missing = [name for name in STEP_COLUMNS if name not in columns]
    unknown = [name for name in columns if name not in STEP_COLUMNS]
    if missing:
        raise ArtifactError(
            f"{target}: missing required column(s): {', '.join(missing)}. "
            f"Expected: {', '.join(STEP_COLUMNS)}."
        )
    if unknown:
        raise ArtifactError(
            f"{target}: unexpected column(s): {', '.join(unknown)}. {_UNKNOWN_COLUMN_HINT}."
        )
    if columns != list(STEP_COLUMNS):
        raise ArtifactError(
            f"{target}: columns are out of order: {', '.join(columns)}. "
            f"Expected: {', '.join(STEP_COLUMNS)}."
        )


def _parse_row(values: Sequence[str], *, target: Path, row_number: int) -> StepRow:
    where = f"{target} row {row_number}"
    if len(values) < len(STEP_COLUMNS):
        raise ArtifactError(
            f"{where}: too few fields ({len(values)} of {len(STEP_COLUMNS)}). "
            f"Expected: {', '.join(STEP_COLUMNS)}."
        )
    if len(values) > len(STEP_COLUMNS):
        raise ArtifactError(
            f"{where}: too many fields ({len(values)} of {len(STEP_COLUMNS)}). "
            f"Expected: {', '.join(STEP_COLUMNS)}."
        )

    # Lengths were checked above, so strict= would never fire; it is set
    # explicitly because a silent zip truncation is exactly the bug the
    # length checks exist to prevent.
    cells = dict(zip(STEP_COLUMNS, (value.strip() for value in values), strict=True))
    return StepRow(
        step=_parse_int(cells["step"], where=where, column="step"),
        # A diverged run really can record nan or inf as its loss. That is a
        # measurement, and hiding it would hide the divergence.
        loss=_parse_float(cells["loss"], where=where, column="loss", finite=False),
        step_time_ms=_parse_float(
            cells["step_time_ms"], where=where, column="step_time_ms", non_negative=True
        ),
        data_residency=_parse_choice(
            cells["data_residency"],
            where=where,
            column="data_residency",
            choices=DATA_RESIDENCY_CHOICES,
        ),
        synchronized=_parse_bool(cells["synchronized"], where=where, column="synchronized"),
    )


def _parse_int(value: str, *, where: str, column: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ArtifactError(
            f"{where}: column {column!r} must be an integer, got {value!r}."
        ) from exc


def _parse_float(
    value: str, *, where: str, column: str, finite: bool = True, non_negative: bool = False
) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ArtifactError(f"{where}: column {column!r} must be a number, got {value!r}.") from exc
    if finite and not math.isfinite(number):
        raise ArtifactError(f"{where}: column {column!r} must be finite, got {value!r}.")
    if non_negative and number < 0:
        raise ArtifactError(
            f"{where}: column {column!r} must not be negative, got {value!r}. "
            "A negative duration means the file is damaged, not that the step was fast."
        )
    return number


def _parse_choice(value: str, *, where: str, column: str, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise ArtifactError(
            f"{where}: column {column!r} must be one of {', '.join(choices)}; got {value!r}."
        )
    return value


def _parse_bool(value: str, *, where: str, column: str) -> bool:
    lowered = value.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ArtifactError(f"{where}: column {column!r} must be true or false, got {value!r}.")


def steps_csv_text(rows: Sequence[StepRow]) -> str:
    """Render ``rows`` as per-step CSV text, byte-compatible with the producer.

    Round-trips: a CSV written by ``examples/train_single.py``, loaded and
    rendered again by this function is identical to the original. The CSV
    module does the quoting, so a value that ever grows a comma or a quote
    is escaped by the same code that escapes it on the way in.
    """
    if not rows:
        raise ArtifactError(
            "refusing to render a per-step CSV with no rows. An empty CSV would "
            "read as a run that measured nothing."
        )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(STEP_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row.to_dict())
    return buffer.getvalue()


def write_steps_csv(path: str | Path, rows: Sequence[StepRow]) -> Path:
    """Write ``rows`` to ``path`` as a per-step CSV; see :func:`steps_csv_text`."""
    if not rows:
        raise ArtifactError(
            f"refusing to write a per-step CSV with no rows: {path}. "
            "An empty CSV would read as a run that measured nothing."
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(steps_csv_text(rows))
    return target


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        text = path.read_text()
    except OSError as exc:  # pragma: no cover - permissions/IO specific
        raise ArtifactError(f"cannot read {path}: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path} is not valid JSON: {exc}") from exc
    return _require_json_object(payload, source=str(path), label=label)


def _require_json_object(payload: Any, *, source: str, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ArtifactError(
            f"{source}: the {label} must be a JSON object, got {type(payload).__name__}."
        )
    return payload


def _validated_environment(payload: Any, *, source: str) -> dict[str, Any]:
    """Check the shape a renderer relies on, keeping every key verbatim."""
    environment = _require_json_object(payload, source=source, label="environment")

    block = environment.get("extra")
    if block is not None and not isinstance(block, dict):
        raise ArtifactError(
            f"{source}: 'extra' must be a JSON object of run settings, got {type(block).__name__}."
        )

    if isinstance(block, dict) and "profiled" in block:
        profiled = block["profiled"]
        if not isinstance(profiled, bool):
            raise ArtifactError(
                f"{source}: extra.profiled must be true or false, got {profiled!r}. "
                "Requirement SS-04 depends on this flag, so a value that cannot be "
                "read is refused rather than guessed."
            )

    names = environment.get("device_names")
    if names is not None and not isinstance(names, list):
        raise ArtifactError(
            f"{source}: 'device_names' must be a JSON array, got {type(names).__name__}."
        )
    return environment
