"""Discover, load and validate the artifacts a ShardScope run writes.

A run directory is the unit every later stage consumes:

    runs/<run_id>/steps.csv          per-step records (requirement SS-03)
    runs/<run_id>/environment.json   provenance (PRD v2.0 section 4)
    runs/<run_id>/summary.json       descriptive statistics, as produced

Scope boundary, deliberately narrow: this module understands exactly the
artifact triple ``examples/train_single.py`` already writes. The *profiled*
per-step schema -- ``data_wait_ms``, ``h2d_copy_ms`` and friends -- is an
open owner decision (GATE 3.1, GATE 3.2), so there is no loader for it
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

import contextlib
import csv
import io
import json
import math
import os
import secrets
from collections.abc import Iterable, Mapping, Sequence
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
    "replace_file_contents",
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
    "This loader understands only the schema examples/train_single.py writes. The profiled "
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
    def from_dict(cls, payload: Any, *, where: str = "step record") -> StepRow:
        """Build a row from an already-typed mapping, validating it.

        The CSV loader and this one are two doors into the same record type,
        so they enforce the same rules: a JSON report is an input like any
        other, and a consumer that trusted it less than the CSV would be a
        hole in the validation boundary. ``where`` names the record in error
        messages (``steps[4]`` for a serialized run).
        """
        if not isinstance(payload, Mapping):
            raise ArtifactError(
                f"{where}: a step record must be a JSON object, got {type(payload).__name__}."
            )
        missing = [name for name in STEP_COLUMNS if name not in payload]
        if missing:
            raise ArtifactError(
                f"{where}: missing field(s): {', '.join(missing)}. "
                f"Expected: {', '.join(STEP_COLUMNS)}."
            )
        return _checked_step(
            step=payload["step"],
            loss=payload["loss"],
            step_time_ms=payload["step_time_ms"],
            data_residency=payload["data_residency"],
            synchronized=payload["synchronized"],
            where=where,
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
        """``steps`` as recorded in ``summary.json``, or ``None`` when unreadable.

        The producer writes the count as a float (``3.0``), so an
        integer-valued float is read as the integer it spells. Anything else
        -- a string, a boolean, a fractional or non-finite number -- is
        unknown rather than coerced: ``int(3.5)`` would let a damaged summary
        agree with a three-row CSV, and ``int(nan)`` raises. The summary
        itself is kept verbatim either way; this reports damage, it does not
        repair it.
        """
        if self.summary is None:
            return None
        value = self.summary.get("steps")
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
        return None

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
        """Rebuild a run from :meth:`to_dict` output (the JSON report view).

        ``path``, ``environment`` and ``steps`` are required. ``summary`` and
        ``artifact_names`` may be absent: the first is an optional artifact
        and stays ``None``, and the second is a directory listing rather
        than a measurement -- an empty one renders as "no file list shown",
        not as "the directory was empty".
        """
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
        source = "<serialized run>"
        path = payload["path"]
        if not isinstance(path, str) or not path:
            raise ArtifactError(
                f"{source}: 'path' must be a non-empty string naming the run directory, "
                f"got {path!r}."
            )
        environment = _validated_environment(payload["environment"], source=source)
        summary = payload.get("summary")
        if summary is not None:
            summary = _require_json_object(summary, source=source, label="summary")
        return cls(
            path=Path(path),
            steps=_collect_rows(_serialized_rows(payload["steps"], source=source), source=source),
            environment=environment,
            summary=summary,
            artifact_names=_validated_artifact_names(payload.get("artifact_names", ()), source),
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
    return tuple(sorted(child for child in _entries(target) if looks_like_run(child)))


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
        artifact_names=tuple(sorted(item.name for item in _entries(run_dir) if item.is_file())),
    )


def _entries(target: Path) -> tuple[Path, ...]:
    """List a directory, turning an OS-level failure into an actionable line.

    Reading a directory can fail for ordinary reasons (permissions, a stale
    mount). Letting that surface as a traceback from inside the construction
    of *another* error message would hide which run was being read.
    """
    try:
        return tuple(target.iterdir())
    except OSError as exc:
        raise ArtifactError(
            f"cannot read the contents of {target}: {exc.strerror or exc}."
        ) from exc


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
        # splitlines() below then feeds csv.reader one record per line. UTF-8
        # explicitly: a device name or config path can be non-ASCII, and the
        # locale default is not UTF-8 everywhere on Python 3.10-3.12.
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ArtifactError(
            f"no such per-step CSV: {target}. Columns expected: {', '.join(STEP_COLUMNS)}."
        ) from exc
    except IsADirectoryError as exc:
        raise ArtifactError(f"{target} is a directory, not a {STEPS_CSV} file.") from exc
    except UnicodeDecodeError as exc:
        raise ArtifactError(
            f"{target} is not valid UTF-8 ({exc.reason} at byte {exc.start}). "
            "Per-step CSVs are written as UTF-8."
        ) from exc
    except OSError as exc:  # pragma: no cover - permissions/IO specific
        raise ArtifactError(f"cannot read {target}: {exc}") from exc

    reader = csv.reader(text.splitlines())
    try:
        header = next(_csv_records(reader, target))
    except StopIteration as exc:
        raise ArtifactError(
            f"{target} is empty: no header row. Expected: {', '.join(STEP_COLUMNS)}."
        ) from exc

    _validate_header(header, target)

    parsed = (
        (f"{target} row {offset}", _parse_row(values, target=target, row_number=offset))
        for offset, values in enumerate(_csv_records(reader, target), start=2)
        if values
    )
    return _collect_rows(parsed, source=str(target))


def _csv_records(reader: Iterable[Sequence[str]], target: Path) -> Iterable[Sequence[str]]:
    """Yield CSV records, turning the csv module's own failures into ours.

    ``csv.reader`` raises :class:`csv.Error` for a field over its size limit
    (128 KiB by default). That exception is not a :class:`ShardScopeError`,
    so left alone it reaches the user as a traceback instead of naming the
    file and line.
    """
    iterator = iter(reader)
    while True:
        try:
            record = next(iterator)
        except StopIteration:
            return
        except csv.Error as exc:
            line = getattr(reader, "line_num", "?")
            raise ArtifactError(
                f"{target} line {line}: cannot be parsed as CSV ({exc}). A per-step CSV "
                f"is plain comma-separated text with the columns {', '.join(STEP_COLUMNS)}; "
                "this file is damaged or is not a per-step CSV."
            ) from exc
        yield record


def _serialized_rows(payload: Any, *, source: str) -> Iterable[tuple[str, StepRow]]:
    """Validate the ``steps`` array of a serialized run, record by record."""
    if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
        raise ArtifactError(
            f"{source}: 'steps' must be a JSON array of step records, got {type(payload).__name__}."
        )
    for index, item in enumerate(payload):
        where = f"{source} steps[{index}]"
        yield where, StepRow.from_dict(item, where=where)


def _collect_rows(items: Iterable[tuple[str, StepRow]], *, source: str) -> tuple[StepRow, ...]:
    """Apply the sequence invariants every loader shares.

    At least one row, and strictly increasing step indices. Both loaders --
    the CSV file and the serialized JSON run -- go through here so the two
    cannot drift apart on what a valid sequence of steps is. ``items`` pairs
    each row with the location to name if it breaks the sequence.
    """
    rows: list[StepRow] = []
    previous: int | None = None
    for where, row in items:
        if previous is not None:
            if row.step == previous:
                raise ArtifactError(
                    f"{where}: step {row.step} repeats the previous row. "
                    "Each row must describe a different training step."
                )
            if row.step < previous:
                raise ArtifactError(
                    f"{where}: step {row.step} follows step {previous}. "
                    "Steps must be increasing; a bounded capture window may skip "
                    "steps but never reorder them."
                )
        previous = row.step
        rows.append(row)

    if not rows:
        raise ArtifactError(
            f"{source} has no step rows. A run with no measurements is a "
            "failed run, not an empty report."
        )
    return tuple(rows)


def _validated_artifact_names(payload: Any, source: str) -> tuple[str, ...]:
    """The run directory's file list: a JSON array of strings, possibly empty.

    A string is itself iterable, so without this check ``"steps.csv"`` would
    quietly become nine one-character file names, and a non-string item
    would crash the renderer rather than be refused here.
    """
    if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
        raise ArtifactError(
            f"{source}: 'artifact_names' must be a JSON array of file names, "
            f"got {type(payload).__name__}."
        )
    names = tuple(payload)
    offenders = [name for name in names if not isinstance(name, str)]
    if offenders:
        raise ArtifactError(
            f"{source}: 'artifact_names' must contain only strings, got {offenders[0]!r}."
        )
    return names


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
    # Strings first become values (type errors name the column), then go
    # through the same checker the JSON path uses for range and choice.
    return _checked_step(
        step=_parse_int(cells["step"], where=where, column="step"),
        loss=_parse_float(cells["loss"], where=where, column="loss"),
        step_time_ms=_parse_float(cells["step_time_ms"], where=where, column="step_time_ms"),
        data_residency=cells["data_residency"],
        synchronized=_parse_bool(cells["synchronized"], where=where, column="synchronized"),
        where=where,
    )


def _checked_step(
    *,
    step: Any,
    loss: Any,
    step_time_ms: Any,
    data_residency: Any,
    synchronized: Any,
    where: str,
) -> StepRow:
    """Validate already-typed field values and build the row.

    The single place the per-step invariants live: integer step index, a loss
    that may be non-finite (a diverged run really records nan, and hiding it
    would hide the divergence), a finite non-negative duration, a known
    residency mode, and a real boolean synchronize policy.
    """
    _require_int(step, where=where, column="step")
    _require_number(loss, where=where, column="loss", finite=False)
    _require_number(step_time_ms, where=where, column="step_time_ms", non_negative=True)
    if data_residency not in DATA_RESIDENCY_CHOICES:
        raise ArtifactError(
            f"{where}: column 'data_residency' must be one of "
            f"{', '.join(DATA_RESIDENCY_CHOICES)}; got {data_residency!r}."
        )
    if not isinstance(synchronized, bool):
        raise ArtifactError(
            f"{where}: column 'synchronized' must be true or false, got {synchronized!r}."
        )
    return StepRow(
        step=step,
        loss=float(loss),
        step_time_ms=float(step_time_ms),
        data_residency=data_residency,
        synchronized=synchronized,
    )


def _require_int(value: Any, *, where: str, column: str) -> None:
    # bool is a subclass of int; a step index of `true` is a mistake, not a 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactError(f"{where}: column {column!r} must be an integer, got {value!r}.")


def _require_number(
    value: Any, *, where: str, column: str, finite: bool = True, non_negative: bool = False
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError(f"{where}: column {column!r} must be a number, got {value!r}.")
    _require_representable(value, where=where, column=column)
    if finite and not math.isfinite(value):
        raise ArtifactError(f"{where}: column {column!r} must be finite, got {value!r}.")
    if non_negative and value < 0:
        raise ArtifactError(
            f"{where}: column {column!r} must not be negative, got {value!r}. "
            "A negative duration means the file is damaged, not that the step was fast."
        )


def _require_representable(value: int | float, *, where: str, column: str) -> None:
    """Refuse an integer no float can hold.

    JSON integers are unbounded, and both ``float(10**400)`` and
    ``math.isfinite(10**400)`` raise ``OverflowError``. A value like that is
    damage, not a measurement, and it has to be caught before any float
    arithmetic touches it.
    """
    try:
        float(value)
    except OverflowError as exc:
        shown = repr(value)
        if len(shown) > 24:
            shown = f"{shown[:24]}... ({len(shown)} digits)"
        raise ArtifactError(
            f"{where}: column {column!r} is too large to represent as a number: {shown}."
        ) from exc


def _parse_int(value: str, *, where: str, column: str) -> int:
    try:
        _require_plain_spelling(value)
        return int(value)
    except ValueError as exc:
        raise ArtifactError(
            f"{where}: column {column!r} must be an integer, got {value!r}."
        ) from exc


def _parse_float(value: str, *, where: str, column: str) -> float:
    """Convert one CSV cell to a float; range checks live in _checked_step."""
    try:
        _require_plain_spelling(value)
        return float(value)
    except ValueError as exc:
        raise ArtifactError(f"{where}: column {column!r} must be a number, got {value!r}.") from exc


def _require_plain_spelling(value: str) -> None:
    """Reject numeric spellings only Python accepts.

    ``int()`` and ``float()`` take ``1_0`` and non-ASCII digits. The
    producer never writes either, so a cell like that is a hand edit, and
    loading it as ``10`` would quietly accept a file that is not ours.
    """
    if "_" in value or not value.isascii():
        raise ValueError(f"not a plain ASCII number: {value!r}")


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
    """Write ``rows`` to ``path`` as a per-step CSV; see :func:`steps_csv_text`.

    Opened with ``newline=""`` so the ``\n`` line endings
    :func:`steps_csv_text` went to the trouble of forcing are not translated
    to CRLF on Windows -- that would quietly break the byte-identical round
    trip with the producer.
    """
    if not rows:
        raise ArtifactError(
            f"refusing to write a per-step CSV with no rows: {path}. "
            "An empty CSV would read as a run that measured nothing."
        )
    return replace_file_contents(path, steps_csv_text(rows))


#: Attempts at an unused sibling name before giving up. A collision needs
#: another writer to have picked the same 64-bit token in the same directory.
_TEMP_NAME_ATTEMPTS = 8


def replace_file_contents(path: str | Path, text: str) -> Path:
    """Write ``text`` to ``path`` so no reader ever sees a partial file.

    The text goes to a temporary file in the same directory, that file is
    closed, and ``os.replace`` swaps it into place. What that does and does
    not promise:

    * A failed write -- a full disk, a permission that changed -- leaves a
      previously valid ``path`` exactly as it was and removes the temporary
      file. A concurrent reader sees the old contents or the new ones.
    * It is a single-file guarantee. Writing three artifacts is three
      replacements, not one transaction.
    * Nothing is fsync'd: the swap is atomic against other readers, not
      against power loss.
    * The destination gets the permissions of a file newly created under
      the current umask, not those of the file it replaced. The temporary
      file is opened with ``os.open`` rather than :mod:`tempfile` for that
      reason: ``tempfile`` creates ``0600`` files, which would turn a shared
      report into an owner-only one.
    * ``newline=""`` and UTF-8, so the bytes on disk are the rendered text.
    * A symlink at ``path`` is replaced by a regular file; whatever it pointed
      at is left alone. Writing *through* a link is how an output path
      quietly lands on a file the caller never named. For the same reason a
      hard link to the old file keeps the old contents.

    Missing parent directories are created. ``OSError`` propagates: the
    caller knows what the file was for and can say so.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = _open_sibling_temp(target)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
    return target


def _open_sibling_temp(target: Path) -> tuple[int, Path]:
    """Create an exclusive, umask-respecting temporary file next to ``target``."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for _ in range(_TEMP_NAME_ATTEMPTS):
        candidate = target.with_name(f".{target.name}.{secrets.token_hex(8)}.tmp")
        try:
            return os.open(candidate, flags, 0o666), candidate
        except FileExistsError:
            continue
    raise OSError(  # pragma: no cover - needs eight 64-bit token collisions
        f"could not create a temporary file next to {target} after {_TEMP_NAME_ATTEMPTS} attempts"
    )


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactError(
            f"{path} is not valid UTF-8 ({exc.reason} at byte {exc.start}). "
            "Artifacts are written as UTF-8; re-produce the run rather than "
            "reporting a file that cannot be decoded."
        ) from exc
    except OSError as exc:  # pragma: no cover - permissions/IO specific
        raise ArtifactError(f"cannot read {path}: {exc}") from exc
    try:
        payload = json.loads(text)
    except ValueError as exc:
        # JSONDecodeError is one ValueError; an integer literal past
        # CPython's 4300-digit limit is another, raised as a plain
        # ValueError with no position.
        raise ArtifactError(f"{path} is not valid JSON: {exc}") from exc
    except RecursionError as exc:
        raise ArtifactError(
            f"{path} is nested too deeply to be read. A {label} artifact is a flat "
            "JSON object; a document this deep is damaged or is not an artifact."
        ) from exc
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
