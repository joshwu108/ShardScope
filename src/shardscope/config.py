"""Load and validate single-device training configuration.

A JSON config records the knobs needed to reproduce a fixture run. Explicit CLI
flags override file values; omitted flags preserve them. Invalid keys, types,
and ranges fail before training starts.

Capture settings require explicit bounded schedules; optional overhead is opt-in.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from difflib import get_close_matches
from pathlib import Path
from typing import Any

from shardscope.errors import ConfigError

__all__ = [
    "DATA_RESIDENCY_CHOICES",
    "DEVICE_CHOICES",
    "SEED_MAX",
    "SEED_MIN",
    "RunConfig",
    "load_run_config",
]

DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")
DATA_RESIDENCY_CHOICES = ("host", "device")
SEED_MIN = -(2**63)
SEED_MAX = 2**64 - 2
_COMMENT_PREFIX = "_"


@dataclass(frozen=True)
class CaptureConfig:
    """An explicit finite profiler schedule, with opt-in overhead features."""

    wait: int
    warmup: int
    active: int
    repeat: int
    activities: tuple[str, ...]
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    with_flops: bool = False

    def __post_init__(self):
        for name in ("wait", "warmup", "active", "repeat"):
            value = getattr(self, name)
            minimum = 1 if name in ("active", "repeat") else 0
            if type(value) is not int or value < minimum:
                raise ConfigError(f"capture.{name} must be an integer >= {minimum}")
        if not isinstance(self.activities, (list, tuple)) or not self.activities:
            raise ConfigError("capture.activities must be a nonempty list of cpu/cuda labels")
        if any(x not in ("cpu", "cuda") for x in self.activities):
            raise ConfigError("capture.activities supports only cpu and cuda")
        if len(set(self.activities)) != len(self.activities):
            raise ConfigError("capture.activities must not contain duplicates")
        # Host markers are required for an explicit trace-to-training step map.
        if "cpu" not in self.activities:
            raise ConfigError("capture.activities must include cpu for step markers")
        object.__setattr__(self, "activities", tuple(self.activities))
        for name in ("record_shapes", "profile_memory", "with_stack", "with_flops"):
            if type(getattr(self, name)) is not bool:
                raise ConfigError(f"capture.{name} must be true or false")

    @property
    def required_steps(self) -> int:
        return (self.wait + self.warmup + self.active) * self.repeat

    def validate_steps(self, steps: int) -> None:
        if self.required_steps > steps:
            raise ConfigError(
                f"capture schedule requires {self.required_steps} training steps; available {steps}"
            )

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict):
            raise ConfigError("capture must be an object")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ConfigError(f"invalid capture settings: {exc}") from exc


@dataclass(frozen=True)
class RunConfig:
    """Validated knobs for one single-device training run.

    Frozen: a config is provenance. Once a run has started, the object that
    described it must not be mutable, or the artifact it is written into
    could disagree with what ran.

    Defaults match ``examples/train_single.py`` and can be overridden per run.
    """

    steps: int = 100
    batch_size: int = 64
    samples: int = 2048
    seed: int = 0
    learning_rate: float = 0.01
    device: str = "auto"
    data_residency: str = "host"
    synchronize: bool = True
    capture: CaptureConfig | None = None

    def merged(self, **changes: Any) -> RunConfig:
        """Return a new config with the non-``None`` ``changes`` applied.

        ``None`` means "not specified", which is what argparse yields for a
        flag the user omitted. That lets a caller overlay explicit CLI flags
        on top of a config file without having to know which flags were
        given.
        """
        supplied = {key: value for key, value in changes.items() if value is not None}
        unknown = sorted(set(supplied) - set(field_names()))
        if unknown:
            raise ConfigError(
                f"cannot override unknown config field(s): {', '.join(unknown)}. "
                f"Known fields: {', '.join(field_names())}."
            )
        return _validate(replace(self, **supplied), source="<override>")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def field_names() -> tuple[str, ...]:
    """Config field names, in declaration order."""
    return tuple(RunConfig.__dataclass_fields__)


def load_run_config(path: str | Path) -> RunConfig:
    """Read and validate a JSON run config.

    Raises :class:`ConfigError` with an actionable message for a missing
    file, a file that is not valid UTF-8, malformed JSON, an unknown key, or
    a value of the wrong type or out of range.
    """
    target = Path(path)
    try:
        # UTF-8 explicitly, and not the locale default: a config can carry a
        # non-ASCII comment or path, and the same bytes must load the same
        # way on every machine that replays the run. Matches how
        # shardscope.artifacts reads a run's files.
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(
            f"no such config file: {target}. Write a JSON object with any of: "
            f"{', '.join(field_names())}."
        ) from exc
    except IsADirectoryError as exc:
        raise ConfigError(f"{target} is a directory, not a config file.") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"{target} is not valid UTF-8 ({exc.reason} at byte {exc.start}). "
            "Config files are read as UTF-8; re-save the file in that encoding."
        ) from exc
    except OSError as exc:  # pragma: no cover - permissions/IO specific
        raise ConfigError(f"cannot read config file {target}: {exc}") from exc

    try:
        payload = json.loads(text)
    except ValueError as exc:
        # JSONDecodeError is one ValueError; an integer literal past
        # CPython's 4300-digit limit is another, raised with no position.
        raise ConfigError(f"{target} is not valid JSON: {exc}") from exc
    except RecursionError as exc:
        raise ConfigError(
            f"{target} is nested too deeply to be read. A config is a flat JSON object "
            f"with any of: {', '.join(field_names())}."
        ) from exc

    return from_mapping(payload, source=str(target))


def from_mapping(payload: Any, *, source: str = "<mapping>") -> RunConfig:
    """Validate an already-parsed mapping into a :class:`RunConfig`."""
    if not isinstance(payload, dict):
        raise ConfigError(
            f'{source}: top level must be a JSON object (e.g. {{"steps": 100}}), '
            f"got {type(payload).__name__}."
        )

    known = field_names()
    values: dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(key, str):  # pragma: no cover - json keys are always str
            raise ConfigError(f"{source}: config keys must be strings, got {key!r}.")
        if key.startswith(_COMMENT_PREFIX):
            continue
        if key not in known:
            raise ConfigError(
                f"{source}: unknown config key {key!r}.{_suggest(key, known)} "
                f"Known keys: {', '.join(known)}."
            )
        values[key] = value

    return _validate(RunConfig(**{**RunConfig().to_dict(), **values}), source=source)


def _suggest(key: str, known: tuple[str, ...]) -> str:
    matches = get_close_matches(key, known, n=1, cutoff=0.6)
    return f" Did you mean {matches[0]!r}?" if matches else ""


def _validate(config: RunConfig, *, source: str) -> RunConfig:
    """Type- and range-check every field, naming ``source`` in any failure."""
    for name in ("steps", "batch_size", "samples"):
        _require_positive_int(config, name, source)

    _require_int(config, "seed", source)
    _require_seed_in_range(config, source)
    _require_positive_finite_number(config, "learning_rate", source)
    _require_choice(config, "device", DEVICE_CHOICES, source)
    _require_choice(config, "data_residency", DATA_RESIDENCY_CHOICES, source)
    _require_bool(config, "synchronize", source)
    capture = config.capture
    if capture is not None:
        if not isinstance(capture, CaptureConfig):
            capture = CaptureConfig.from_mapping(capture)
        capture.validate_steps(config.steps)
        config = replace(config, capture=capture)
    return config


def _value(config: RunConfig, name: str) -> Any:
    return getattr(config, name)


def _describe(value: Any) -> str:
    return f"{type(value).__name__} ({value!r})"


def _require_int(config: RunConfig, name: str, source: str) -> None:
    value = _value(config, name)
    # bool is a subclass of int; `"steps": true` is a mistake, not a 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"{source}: config key {name!r} must be an integer, got {_describe(value)}."
        )


def _require_positive_int(config: RunConfig, name: str, source: str) -> None:
    _require_int(config, name, source)
    value = _value(config, name)
    if value <= 0:
        raise ConfigError(f"{source}: config key {name!r} must be positive, got {value}.")


def _require_seed_in_range(config: RunConfig, source: str) -> None:
    value = _value(config, "seed")
    if not SEED_MIN <= value <= SEED_MAX:
        raise ConfigError(
            f"{source}: config key 'seed' must be between {SEED_MIN} and {SEED_MAX} "
            f"(a 64-bit generator seed), got {value}."
        )


def _require_positive_finite_number(config: RunConfig, name: str, source: str) -> None:
    value = _value(config, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"{source}: config key {name!r} must be a number, got {_describe(value)}."
        )
    if not math.isfinite(value):
        raise ConfigError(f"{source}: config key {name!r} must be finite, got {value}.")
    # Validate the range without choosing a tuning policy.
    if value <= 0:
        raise ConfigError(f"{source}: config key {name!r} must be greater than 0, got {value}.")


def _require_choice(config: RunConfig, name: str, choices: tuple[str, ...], source: str) -> None:
    value = _value(config, name)
    if value not in choices:
        raise ConfigError(
            f"{source}: config key {name!r} must be one of {', '.join(choices)}; got {value!r}."
        )


def _require_bool(config: RunConfig, name: str, source: str) -> None:
    value = _value(config, name)
    if not isinstance(value, bool):
        raise ConfigError(
            f"{source}: config key {name!r} must be true or false, got {_describe(value)}."
        )
