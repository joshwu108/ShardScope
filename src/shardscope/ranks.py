"""Cross-rank step comparison: python -m shardscope.ranks --root DIR.

A torchrun launch records one run directory per rank under a common root
(``rank-00000``, ``rank-00001``, ...). This module reads every rank's
``steps.csv`` and ``environment.json``, checks that the ranks recorded
compatible work, aligns their rows by training step and compares the recorded
step durations. The result is persisted as ``ranks.json`` next to the rank
directories and is the only input to ``shardscope findings --ranks``.

What is compared: each rank's own ``step_time_ms``, measured on that rank's
host clock between device synchronizations. Durations are compared, never
timestamps, so no clock alignment between machines is assumed. What is not
derived: a straggler. With synchronized DDP steps every rank waits in the
collective for the slowest one, so the rank with the longest recorded step is
not thereby the one that caused the wait.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from shardscope.errors import ArtifactError
from shardscope.parse import finite_number
from shardscope.rules import RANK_RULE_IDS, RuleConfig, evaluate_rank_rules, parse_threshold

__all__ = [
    "COMPARED_METRIC",
    "RANKS_JSON",
    "RANKS_SCHEMA",
    "analyze_ranks",
    "load_ranks",
    "validate_ranks",
    "write_ranks",
]

RANKS_JSON = "ranks.json"
RANKS_SCHEMA = 1
COMPARED_METRIC = "step_time_ms"
_RANK_DIRECTORY = re.compile(r"^rank-(\d{5})$")

#: environment.json values that must agree before ranks are compared. They
#: pin the workload (what each step does) and the measurement contract (how
#: step_time_ms was taken). A disagreement is a different experiment.
REQUIRED_AGREEMENT = (
    "cuda_available",
    "torch_version",
    "cuda_version",
    "nccl_version",
    "extra.source",
    "extra.artifact_schema",
    "extra.world_size",
    "extra.work_scope",
    "extra.steps",
    "extra.batch_size",
    "extra.global_batch_size",
    "extra.samples",
    "extra.samples_scope",
    "extra.seed",
    "extra.fixture_model",
    "extra.fixture_dtype",
    "extra.input_shape",
    "extra.learning_rate",
    "extra.data_residency",
    "extra.synchronize",
    "extra.synchronized",
    "extra.profiled",
)
#: Values that legitimately differ per rank or per node and are recorded, not compared.
EXPECTED_DIFFERENCES = (
    "machine",
    "platform",
    "python_version",
    "cudnn_version",
    "device_count",
    "device_names",
    "mps_available",
    "extra.rank",
    "extra.local_rank",
    "extra.pid",
    "extra.hostname",
    "extra.device",
    "extra.device_requested",
    "extra.device_resolved",
    "extra.data_partition",
    "extra.rank_samples",
    "extra.capture",
    "extra.config_file",
)

_MISSING = object()


def _lookup(environment: dict[str, Any], key: str) -> Any:
    if key.startswith("extra."):
        extra = environment.get("extra")
        return extra.get(key[6:], _MISSING) if isinstance(extra, dict) else _MISSING
    return environment.get(key, _MISSING)


def _digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ArtifactError(f"cannot read {path}: {exc}") from exc


def _discover(root: Path) -> dict[int, Path]:
    if not root.is_dir():
        raise ArtifactError(f"{root} is not a directory of rank-NNNNN run directories")
    found = {}
    for entry in sorted(root.iterdir()):
        match = _RANK_DIRECTORY.match(entry.name)
        if match and entry.is_dir():
            found[int(match.group(1))] = entry
    if not found:
        raise ArtifactError(
            f"{root} holds no rank-NNNNN directory; pass the root a torchrun launch wrote "
            "(examples/train_ddp.py --out-dir)"
        )
    return found


def _host_phases(analysis: dict[str, Any] | None) -> dict[int, list[str] | None] | None:
    """Per training step, the host phases a validated analysis recorded; None if unmapped."""
    if analysis is None:
        return None
    phases: dict[int, list[str] | None] = {}
    for row in analysis["steps"]:
        if row["boundary"] is None:
            phases[row["training_step"]] = None
        else:
            phases[row["training_step"]] = sorted(
                name
                for name, metric in row["measurements"].items()
                if name.startswith("host_") and metric["status"] == "recorded"
            )
    return phases


def _compare_rows(
    rows_by_rank: dict[int, dict[int, Any]],
    phases_by_rank: dict[int, dict[int, list[str] | None] | None],
    world_size: int,
) -> list[dict[str, Any]]:
    present = sorted(rows_by_rank)
    every_step = sorted({step for rows in rows_by_rank.values() for step in rows})
    with_analysis = all(phases_by_rank[rank] is not None for rank in present)
    table = []
    for step in every_step:
        per_rank = []
        for rank in present:
            row = rows_by_rank[rank].get(step)
            if row is None:
                continue
            phases = phases_by_rank[rank]
            per_rank.append(
                {
                    "rank": rank,
                    COMPARED_METRIC: row.step_time_ms,
                    "synchronized": row.synchronized,
                    "data_residency": row.data_residency,
                    "host_phases": phases.get(step) if phases is not None else None,
                }
            )
        recorded = {item["rank"] for item in per_rank}
        missing = [rank for rank in range(world_size) if rank not in recorded]
        problems = []
        unsynchronized = [i["rank"] for i in per_rank if not i["synchronized"]]
        if unsynchronized:
            problems.append(
                f"{COMPARED_METRIC} was measured without device synchronization on rank(s) "
                f"{unsynchronized}; it times kernel launch, not execution"
            )
        if len({i["data_residency"] for i in per_rank}) > 1:
            problems.append("ranks recorded different data_residency for this step")
        equivalence = "configuration"
        if with_analysis and not missing:
            sets = {tuple(i["host_phases"]) for i in per_rank if i["host_phases"] is not None}
            unmapped = [i["rank"] for i in per_rank if i["host_phases"] is None]
            if unmapped:
                equivalence = "configuration; step unmapped in analysis.json on rank(s) " + str(
                    unmapped
                )
            elif len(sets) > 1:
                problems.append(
                    "ranks recorded different host phase sets for this step "
                    f"({len(sets)} distinct), so it is not known to be equivalent work"
                )
                equivalence = "configuration; host phases differ"
            else:
                equivalence = "configuration and host phases"
        values = [i[COMPARED_METRIC] for i in per_rank]
        low = min(per_rank, key=lambda i: i[COMPARED_METRIC]) if per_rank else None
        high = max(per_rank, key=lambda i: i[COMPARED_METRIC]) if per_rank else None
        ratio = None
        if per_rank and not missing and not problems:
            if low[COMPARED_METRIC] == 0:
                problems.append(f"minimum {COMPARED_METRIC} is zero; the ratio is undefined")
            else:
                ratio = high[COMPARED_METRIC] / low[COMPARED_METRIC]
        table.append(
            {
                "training_step": step,
                "per_rank": per_rank,
                "coverage": "complete" if not missing else "partial",
                "missing_ranks": missing,
                "problems": problems,
                "work_equivalence": equivalence,
                "min_ms": min(values) if values else None,
                "max_ms": max(values) if values else None,
                "min_rank": low["rank"] if low else None,
                "max_rank": high["rank"] if high else None,
                "spread_ms": (max(values) - min(values)) if values else None,
                "ratio": ratio,
            }
        )
    return table


def analyze_ranks(root: str | Path, *, config: RuleConfig | None = None) -> dict[str, Any]:
    """Build the cross-rank comparison for the rank directories under ``root``.

    Every rank directory is loaded as a run (its ``analysis.json``, when
    present, is validated against its sources first, so a stale per-rank
    analysis refuses the comparison rather than feeding it). Nothing under
    ``root`` is written or modified.

    Args:
        root: Directory holding ``rank-NNNNN`` run directories.
        config: Cross-rank thresholds; ``None`` leaves the rule unconfigured.

    Returns:
        The ``ranks.json`` payload: rank inventory with source digests,
        compatibility verdict, the per-step comparison table, the evaluated
        rule ids, configuration and findings.

    Raises:
        ArtifactError: No rank directory, a rank directory that is not a
            valid run, a rank whose environment does not name its rank and
            world size, or a threshold key the cross-rank rule does not take.
    """
    from shardscope.artifacts import load_run

    root = Path(root)
    if config is not None:
        config.restrict(RANK_RULE_IDS, command="shardscope ranks")
    directories = _discover(root)
    runs = {}
    for rank, directory in directories.items():
        run = load_run(directory)
        recorded = run.extra.get("rank")
        if recorded != rank:
            raise ArtifactError(
                f"{directory}: environment.extra.rank is {recorded!r}, not the directory's "
                f"rank {rank}; the directory name is not evidence of identity"
            )
        runs[rank] = run
    world_sizes = {run.extra.get("world_size") for run in runs.values()}
    if len(world_sizes) != 1 or type(next(iter(world_sizes))) is not int:
        raise ArtifactError(
            f"{root}: ranks disagree about environment.extra.world_size or do not record it "
            f"({sorted(map(repr, world_sizes))}); not one DDP launch"
        )
    world_size = next(iter(world_sizes))
    if world_size < 2 or any(rank >= world_size for rank in runs):
        raise ArtifactError(
            f"{root}: world_size {world_size} does not cover rank directories {sorted(runs)}"
        )
    missing_ranks = [rank for rank in range(world_size) if rank not in runs]

    inventory = []
    for rank in sorted(runs):
        run = runs[rank]
        sources = {
            name: _digest(run.path / name)
            for name in ("steps.csv", "environment.json", "analysis.json")
            if (run.path / name).is_file()
        }
        inventory.append(
            {
                "rank": rank,
                "directory": run.path.name,
                "source": sources,
                "rows": len(run.steps),
                "profiled": run.profiled,
                "analysis_present": run.analysis is not None,
                "hostname": run.extra.get("hostname"),
                "device_resolved": run.extra.get("device_resolved"),
            }
        )

    problems = []
    unrecorded = []
    differing: dict[str, dict[str, Any]] = {}
    for key in REQUIRED_AGREEMENT:
        values = {rank: _lookup(runs[rank].environment, key) for rank in sorted(runs)}
        absent = [rank for rank, value in values.items() if value is _MISSING]
        if len(absent) == len(values):
            # A producer that never records the key cannot disagree about it;
            # the omission is reported so the reader knows what was not checked.
            unrecorded.append(key)
            continue
        if absent:
            problems.append(f"{key} is not recorded on rank(s) {absent}")
            continue
        distinct = {json.dumps(value, sort_keys=True) for value in values.values()}
        if len(distinct) > 1:
            shown = ", ".join(
                f"rank {r}={json.dumps(v, sort_keys=True)}" for r, v in values.items()
            )
            problems.append(f"{key} differs across ranks: {shown}")
    for key in EXPECTED_DIFFERENCES:
        values = {rank: _lookup(runs[rank].environment, key) for rank in sorted(runs)}
        shown = {str(r): v for r, v in values.items() if v is not _MISSING}
        if len({json.dumps(v, sort_keys=True) for v in shown.values()}) > 1 and key not in (
            "extra.capture",
        ):
            differing[key] = shown
    profiled = {run.profiled for run in runs.values()}
    if len(profiled) > 1:
        problems.append(
            "some ranks were profiled and others were not; step times are not comparable"
        )

    rows_by_rank = {rank: {row.step: row for row in run.steps} for rank, run in runs.items()}
    phases_by_rank = {rank: _host_phases(run.analysis) for rank, run in runs.items()}
    steps = _compare_rows(rows_by_rank, phases_by_rank, world_size)

    warnings = [
        f"{COMPARED_METRIC} is each rank's own host-clock duration between device "
        "synchronizations; durations are compared, timestamps never are, and no clock "
        "alignment between machines is assumed",
        "a longer recorded step on one rank does not identify it as the straggler: with "
        "synchronized DDP steps the faster ranks spend the difference waiting inside the "
        "collective",
        "steps.csv rows are the producer's training steps; microbatch or gradient-accumulation "
        "structure inside a step is not recorded and is not compared",
    ]
    if missing_ranks:
        warnings.append(
            f"rank directories {missing_ranks} of world size {world_size} are absent; the "
            "comparison covers the recorded ranks only"
        )
    if not all(run.analysis is not None for run in runs.values()):
        warnings.append(
            "not every rank has a validated analysis.json; work equivalence rests on recorded "
            "configuration only (run shardscope analyze per rank to compare host phase sets)"
        )
    payload = {
        "schema_version": RANKS_SCHEMA,
        "root": str(root),
        "world_size": world_size,
        "compared_metric": COMPARED_METRIC,
        "ranks": inventory,
        "missing_ranks": missing_ranks,
        "compatibility": {
            "required_agreement": list(REQUIRED_AGREEMENT),
            "expected_differences": list(EXPECTED_DIFFERENCES),
            "unrecorded": unrecorded,
            "problems": problems,
            "differing": differing,
        },
        "steps": steps,
        "rules": list(RANK_RULE_IDS),
        "rule_config": config.to_dict() if config else None,
        "findings": [],
        "warnings": warnings,
    }
    payload["findings"] = evaluate_rank_rules(payload, config)
    # Digests were just taken, so only structure and findings are checked here;
    # readers recompute the whole comparison through ``root``.
    validate_ranks(payload)
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(f"{RANKS_JSON}: {message}")


def validate_ranks(payload: Any, *, root: str | Path | None = None) -> dict[str, Any]:
    """Check a ``ranks.json`` payload; with ``root``, against the rank directories.

    Structure, the derived per-step figures and the findings are checked
    without sources. With ``root`` the per-rank source digests must still
    match and the whole comparison is recomputed from the rank directories
    and must agree exactly, so a stale or hand-edited comparison is refused.
    """
    _require(isinstance(payload, dict), "expected object")
    _require(
        payload.get("schema_version") == RANKS_SCHEMA,
        f"unsupported schema; expected {RANKS_SCHEMA}",
    )
    _require(payload.get("compared_metric") == COMPARED_METRIC, "unsupported compared_metric")
    world_size = payload.get("world_size")
    _require(type(world_size) is int and world_size >= 2, "world_size must be an integer >= 2")
    ranks = payload.get("ranks")
    _require(isinstance(ranks, list) and ranks, "ranks must be a non-empty list")
    seen = set()
    for item in ranks:
        _require(isinstance(item, dict) and type(item.get("rank")) is int, "rank entry needs rank")
        _require(
            0 <= item["rank"] < world_size and item["rank"] not in seen,
            "rank out of range or repeated",
        )
        seen.add(item["rank"])
        _require(
            item.get("directory") == f"rank-{item['rank']:05d}", "rank directory name mismatch"
        )
        source = item.get("source")
        _require(
            isinstance(source, dict) and {"steps.csv", "environment.json"} <= source.keys(),
            "rank source digests missing",
        )
        for name, digest in source.items():
            _require(
                isinstance(digest, str)
                and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest),
                f"invalid SHA-256 for rank {item['rank']} {name}",
            )
    _require(
        payload.get("missing_ranks") == [r for r in range(world_size) if r not in seen],
        "missing_ranks disagrees with ranks",
    )
    compatibility = payload.get("compatibility")
    _require(
        isinstance(compatibility, dict)
        and isinstance(compatibility.get("problems"), list)
        and all(isinstance(p, str) for p in compatibility["problems"]),
        "compatibility.problems must be a list of strings",
    )
    steps = payload.get("steps")
    _require(isinstance(steps, list), "steps must be a list")
    previous = -1
    for row in steps:
        _require(
            isinstance(row, dict) and type(row.get("training_step")) is int,
            "step row needs training_step",
        )
        _require(row["training_step"] > previous, "steps must be ordered by training_step")
        previous = row["training_step"]
        per_rank = row.get("per_rank")
        _require(isinstance(per_rank, list) and per_rank, "step row needs per_rank values")
        values = []
        recorded = set()
        for item in per_rank:
            _require(
                isinstance(item, dict)
                and item.get("rank") in seen
                and finite_number(item.get(COMPARED_METRIC))
                and item[COMPARED_METRIC] >= 0
                and isinstance(item.get("synchronized"), bool),
                f"training_step {row['training_step']}: invalid per-rank value",
            )
            _require(item["rank"] not in recorded, "rank repeated within a step")
            recorded.add(item["rank"])
            values.append(item)
        missing = [r for r in range(world_size) if r not in recorded]
        _require(row.get("missing_ranks") == missing, "step missing_ranks disagrees with per_rank")
        _require(
            row.get("coverage") == ("complete" if not missing else "partial"),
            "coverage label disagrees",
        )
        low = min(values, key=lambda i: i[COMPARED_METRIC])
        high = max(values, key=lambda i: i[COMPARED_METRIC])
        _require(
            row.get("min_ms") == low[COMPARED_METRIC]
            and row.get("max_ms") == high[COMPARED_METRIC]
            and row.get("spread_ms") == high[COMPARED_METRIC] - low[COMPARED_METRIC]
            and row.get("min_rank") in recorded
            and row.get("max_rank") in recorded,
            f"training_step {row['training_step']}: derived extrema disagree with per-rank values",
        )
        _require(isinstance(row.get("problems"), list), "step problems must be a list")
        if row.get("ratio") is not None:
            _require(
                not missing
                and not row["problems"]
                and low[COMPARED_METRIC] > 0
                and row["ratio"] == high[COMPARED_METRIC] / low[COMPARED_METRIC],
                f"training_step {row['training_step']}: ratio disagrees with per-rank values",
            )
        else:
            _require(
                bool(missing) or bool(row["problems"]),
                f"training_step {row['training_step']}: a compared step must carry its ratio",
            )
    _require(payload.get("rules") == list(RANK_RULE_IDS), "rules must list the cross-rank rule ids")
    config = payload.get("rule_config")
    rule_config = None
    if config is not None:
        _require(isinstance(config, dict) and bool(config), "invalid rule_config")
        rule_config = RuleConfig.from_mapping(config).restrict(
            RANK_RULE_IDS, command="shardscope ranks"
        )
    _require(
        payload.get("findings") == evaluate_rank_rules(payload, rule_config),
        "findings disagree with the comparison table or configuration",
    )
    _require(
        isinstance(payload.get("warnings"), list)
        and all(isinstance(w, str) for w in payload["warnings"]),
        "warnings must be a list of strings",
    )
    if root is not None:
        root = Path(root)
        for item in ranks:
            for name, digest in item["source"].items():
                _require(
                    _digest(root / item["directory"] / name) == digest,
                    f"stale source rank {item['rank']} {name}; re-run shardscope ranks --overwrite",
                )
        fresh = analyze_ranks(root, config=rule_config)
        _require(
            {k: v for k, v in payload.items() if k != "root"}
            == {k: v for k, v in fresh.items() if k != "root"},
            "comparison disagrees with the rank directories; re-run shardscope ranks --overwrite",
        )
    return payload


def load_ranks(root: str | Path) -> dict[str, Any]:
    """Read and validate ``root/ranks.json`` against the rank directories."""
    from shardscope.parse import read_json_bytes

    root = Path(root)
    target = root / RANKS_JSON
    if not target.is_file():
        raise ArtifactError(
            f"{root} has no {RANKS_JSON}; run 'shardscope ranks --root {root} --threshold "
            "rank_imbalance_ratio=R' first. Cross-rank findings are read from the persisted "
            "comparison only."
        )
    try:
        payload = read_json_bytes(target.read_bytes(), source=str(target))
    except OSError as exc:
        raise ArtifactError(f"cannot read {target}: {exc}") from exc
    return validate_ranks(payload, root=root)


def write_ranks(root: str | Path, payload: dict[str, Any], *, overwrite: bool = False) -> Path:
    """Publish ``ranks.json`` atomically; never clobbers without ``overwrite``."""
    root = Path(root)
    target = root / RANKS_JSON
    if target.is_symlink():
        raise ArtifactError(f"refusing output symlink {target}")
    if target.exists() and not overwrite:
        raise ArtifactError(f"{target} exists; pass --overwrite to replace the comparison")
    validate_ranks(payload, root=root)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=root, prefix=".ranks-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
    except (OSError, ValueError, TypeError) as exc:
        raise ArtifactError(f"cannot publish {target}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root", required=True, type=Path, help="directory holding rank-NNNNN run directories"
    )
    parser.add_argument(
        "--threshold",
        action="append",
        metavar="KEY=VALUE",
        help=(
            "configure the cross-rank rule (rank_imbalance_ratio=R, R >= 1); without it the "
            "comparison is written and the rule reports unconfigured"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")


def rule_config_from_arguments(args: argparse.Namespace) -> RuleConfig | None:
    values: dict[str, int | float] = {}
    for item in args.threshold or []:
        key, value = parse_threshold(item)
        if key in values and values[key] != value:
            raise ArtifactError(f"threshold {key} given twice with different values")
        values[key] = value
    return RuleConfig(**values) if values else None


def run_arguments(args: argparse.Namespace) -> int:
    try:
        if (args.root / RANKS_JSON).exists() and not args.overwrite:
            raise ArtifactError(f"{args.root / RANKS_JSON} exists; pass --overwrite")
        payload = analyze_ranks(args.root, config=rule_config_from_arguments(args))
        print(write_ranks(args.root, payload, overwrite=args.overwrite))
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    return run_arguments(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
