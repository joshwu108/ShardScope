"""Cross-rank step comparison: compatibility, alignment, the rule, persistence, CLI."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from conftest import GOOD_ENVIRONMENT
from shardscope.cli import main
from shardscope.errors import ArtifactError
from shardscope.ranks import (
    RANKS_JSON,
    _compare_rows,
    analyze_ranks,
    load_ranks,
    validate_ranks,
    write_ranks,
)
from shardscope.rules import RuleConfig, evaluate_rank_rules

HEADER = "step,loss,step_time_ms,data_residency,synchronized\n"


def rank_environment(rank, *, world_size=2, **extra):
    environment = copy.deepcopy(GOOD_ENVIRONMENT)
    environment.update(
        cuda_available=True, cuda_version="13.2", nccl_version="2.30.7", device_count=2
    )
    environment["extra"].update(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        pid=1000 + rank,
        hostname="node",
        source="shardscope.capture",
        work_scope="per_rank",
        global_batch_size=128,
        data_partition={"rank": rank},
        device_resolved=f"cuda:{rank}",
        profiled=False,
    )
    environment["extra"].update(extra)
    return environment


def rows(times, *, synchronized=True, residency="host", steps=None):
    steps = range(len(times)) if steps is None else steps
    return HEADER + "".join(
        f"{step},1.0,{time},{residency},{synchronized}\n"
        for step, time in zip(steps, times, strict=True)
    )


def make_root(tmp_path, per_rank, *, world_size=None, name="launch"):
    """``per_rank`` maps rank -> (steps_csv text, environment dict)."""
    root = tmp_path / name
    root.mkdir()
    world_size = world_size or len(per_rank)
    for rank, (csv, environment) in per_rank.items():
        directory = root / f"rank-{rank:05d}"
        directory.mkdir()
        (directory / "steps.csv").write_text(csv)
        (directory / "environment.json").write_text(json.dumps(environment))
    return root


def two_ranks(tmp_path, a, b, **kwargs):
    return make_root(
        tmp_path,
        {0: (rows(a), rank_environment(0)), 1: (rows(b), rank_environment(1))},
        **kwargs,
    )


def digests(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# --- comparison table ----------------------------------------------------------------


def test_compatible_ranks_are_aligned_by_training_step_and_compared_as_durations(tmp_path):
    root = two_ranks(tmp_path, [10.0, 20.0, 5.0], [10.0, 8.0, 15.0])
    before = digests(root)
    payload = analyze_ranks(root, config=RuleConfig(rank_imbalance_ratio=2.5))
    assert digests(root) == before, "rank directories are never modified"
    assert payload["world_size"] == 2 and payload["missing_ranks"] == []
    assert payload["compatibility"]["problems"] == []
    assert "extra.fixture_model" in payload["compatibility"]["unrecorded"]
    assert payload["compatibility"]["differing"]["extra.pid"] == {"0": 1000, "1": 1001}
    assert [r["rows"] for r in payload["ranks"]] == [3, 3]
    table = payload["steps"]
    assert [row["ratio"] for row in table] == [1.0, 2.5, 3.0]
    assert [(row["min_rank"], row["max_rank"]) for row in table] == [(0, 0), (1, 0), (0, 1)]
    assert table[1]["spread_ms"] == 12.0 and table[1]["coverage"] == "complete"
    assert table[0]["work_equivalence"] == "configuration"
    assert all(row["per_rank"][0]["host_phases"] is None for row in table)
    statuses = [f["status"] for f in payload["findings"]]
    assert statuses == ["below_threshold", "possible_rank_imbalance", "possible_rank_imbalance"]
    hit = payload["findings"][1]
    assert hit["comparison"]["value"] == 2.5 and hit["comparison"]["result"] is True
    assert hit["comparison"]["max_rank"] == 0 and hit["comparison"]["min_rank"] == 1
    assert hit["inputs"] == [
        {"rank": 0, "metric": "step_time_ms", "value_ms": 20.0},
        {"rank": 1, "metric": "step_time_ms", "value_ms": 8.0},
    ]
    assert "does not identify a straggler" in hit["interpretation"]
    assert hit["evidence"] == []
    assert any("timestamps never" in w for w in payload["warnings"])


def test_missing_rank_directory_and_unrecorded_steps_are_incomplete_not_filled(tmp_path):
    root = make_root(
        tmp_path,
        {
            0: (rows([10.0, 20.0, 30.0]), rank_environment(0, world_size=3)),
            2: (rows([10.0, 10.0], steps=[0, 2]), rank_environment(2, world_size=3)),
        },
    )
    payload = analyze_ranks(root, config=RuleConfig(rank_imbalance_ratio=1.5))
    assert payload["missing_ranks"] == [1]
    assert [row["coverage"] for row in payload["steps"]] == ["partial"] * 3
    assert payload["steps"][1]["missing_ranks"] == [1, 2]
    assert all(row["ratio"] is None for row in payload["steps"])
    (finding,) = payload["findings"]
    assert finding["status"] == "incomplete" and finding["comparison"]["missing_ranks"] == [1]
    assert "a straggler could be the missing rank" in finding["interpretation"]
    # With every rank directory present, an individual unrecorded step is incomplete.
    root = make_root(
        tmp_path,
        {
            0: (rows([10.0, 20.0, 30.0]), rank_environment(0)),
            1: (rows([10.0, 10.0], steps=[0, 2]), rank_environment(1)),
        },
        name="partial",
    )
    payload = analyze_ranks(root, config=RuleConfig(rank_imbalance_ratio=1.5))
    assert [f["status"] for f in payload["findings"]] == [
        "below_threshold",
        "incomplete",
        "possible_rank_imbalance",
    ]
    assert payload["findings"][1]["steps"] == [1]


def test_incompatible_recordings_compare_nothing(tmp_path):
    root = make_root(
        tmp_path,
        {
            0: (rows([10.0]), rank_environment(0)),
            1: (rows([10.0]), rank_environment(1, batch_size=32)),
        },
    )
    payload = analyze_ranks(root, config=RuleConfig(rank_imbalance_ratio=1.0))
    (problem,) = payload["compatibility"]["problems"]
    assert problem.startswith("extra.batch_size differs across ranks")
    (finding,) = payload["findings"]
    assert finding["status"] == "incompatible" and problem in finding["interpretation"]
    # A key recorded on one rank only is also a disagreement.
    root = make_root(
        tmp_path,
        {
            0: (rows([10.0]), rank_environment(0, fixture_model="cnn")),
            1: (rows([10.0]), rank_environment(1)),
        },
        name="half",
    )
    payload = analyze_ranks(root)
    assert payload["compatibility"]["problems"] == [
        "extra.fixture_model is not recorded on rank(s) [1]"
    ]
    assert payload["findings"][0]["status"] == "unconfigured"


def test_unsynchronized_rows_mixed_residency_and_zero_minimum_are_ambiguous(tmp_path):
    root = make_root(
        tmp_path,
        {
            0: (rows([10.0, 10.0, 0.0]), rank_environment(0)),
            1: (
                HEADER + "0,1.0,10.0,host,False\n1,1.0,10.0,device,True\n2,1.0,5.0,host,True\n",
                rank_environment(1),
            ),
        },
    )
    payload = analyze_ranks(root, config=RuleConfig(rank_imbalance_ratio=1.0))
    problems = [row["problems"] for row in payload["steps"]]
    assert "without device synchronization on rank(s) [1]" in problems[0][0]
    assert "different data_residency" in problems[1][0]
    assert "minimum step_time_ms is zero" in problems[2][0]
    assert [f["status"] for f in payload["findings"]] == ["ambiguous"] * 3


def test_host_phase_sets_from_validated_analyses_gate_work_equivalence():
    rows_by_rank = {0: {0: Row(10.0), 1: Row(10.0)}, 1: {0: Row(12.0), 1: Row(12.0)}}
    phases = {
        0: {0: ["host_backward_ms", "host_step_ms"], 1: ["host_step_ms"]},
        1: {0: ["host_backward_ms", "host_step_ms"], 1: ["host_backward_ms", "host_step_ms"]},
    }
    first, second = _compare_rows(rows_by_rank, phases, 2)
    assert first["work_equivalence"] == "configuration and host phases" and first["ratio"] == 1.2
    assert second["ratio"] is None and "different host phase sets" in second["problems"][0]
    # One rank without analysis: configuration only, still compared.
    (only,) = _compare_rows({0: {0: Row(10.0)}, 1: {0: Row(10.0)}}, {0: phases[0], 1: None}, 2)
    assert only["work_equivalence"] == "configuration" and only["ratio"] == 1.0
    # A step unmapped in one analysis cannot be checked and says so.
    (unmapped,) = _compare_rows(
        {0: {0: Row(10.0)}, 1: {0: Row(10.0)}}, {0: {0: None}, 1: {0: ["host_step_ms"]}}, 2
    )
    assert "unmapped" in unmapped["work_equivalence"] and unmapped["ratio"] == 1.0


class Row:
    def __init__(self, step_time_ms, *, synchronized=True, data_residency="host"):
        self.step_time_ms = step_time_ms
        self.synchronized = synchronized
        self.data_residency = data_residency


# --- refusals -------------------------------------------------------------------------


def test_rank_identity_comes_from_environment_not_directory_names(tmp_path):
    root = make_root(
        tmp_path,
        {0: (rows([1.0]), rank_environment(0)), 1: (rows([1.0]), rank_environment(0))},
    )
    with pytest.raises(ArtifactError, match="environment.extra.rank is 0, not the directory"):
        analyze_ranks(root)


def test_disagreeing_or_missing_world_size_is_refused(tmp_path):
    root = make_root(
        tmp_path,
        {
            0: (rows([1.0]), rank_environment(0, world_size=2)),
            1: (rows([1.0]), rank_environment(1, world_size=4)),
        },
    )
    with pytest.raises(ArtifactError, match="world_size"):
        analyze_ranks(root)
    single = tmp_path / "single"
    single.mkdir()
    (single / "rank-00000").mkdir()
    (single / "rank-00000" / "steps.csv").write_text(rows([1.0]))
    environment = rank_environment(0)
    del environment["extra"]["world_size"]
    (single / "rank-00000" / "environment.json").write_text(json.dumps(environment))
    with pytest.raises(ArtifactError, match="world_size"):
        analyze_ranks(single)
    with pytest.raises(ArtifactError, match="no rank-NNNNN directory"):
        analyze_ranks(tmp_path)


def test_per_rank_threshold_keys_are_refused_by_ranks(tmp_path):
    root = two_ranks(tmp_path, [1.0], [1.0])
    with pytest.raises(ArtifactError, match="shardscope analyze"):
        analyze_ranks(root, config=RuleConfig(h2d_share_threshold=0.1))
    with pytest.raises(ArtifactError):
        evaluate_rank_rules({}, RuleConfig(rank_imbalance_ratio=0.5))


# --- persistence --------------------------------------------------------------------


def test_ranks_json_round_trips_and_tampering_or_staleness_is_refused(tmp_path):
    root = two_ranks(tmp_path, [10.0, 20.0], [10.0, 8.0])
    payload = analyze_ranks(root, config=RuleConfig(rank_imbalance_ratio=2.0))
    target = write_ranks(root, payload)
    assert target == root / RANKS_JSON
    with pytest.raises(ArtifactError, match="--overwrite"):
        write_ranks(root, payload)
    assert load_ranks(root) == payload
    assert validate_ranks(copy.deepcopy(payload)) == payload

    forged = copy.deepcopy(payload)
    forged["steps"][1]["ratio"] = 1.0
    with pytest.raises(ArtifactError, match="ratio disagrees"):
        validate_ranks(forged)
    forged = copy.deepcopy(payload)
    forged["findings"][1]["status"] = "below_threshold"
    with pytest.raises(ArtifactError, match="findings disagree"):
        validate_ranks(forged)
    forged = copy.deepcopy(payload)
    forged["steps"][1]["per_rank"][1]["step_time_ms"] = 19.0
    with pytest.raises(ArtifactError, match="disagree"):
        validate_ranks(forged)
    forged = copy.deepcopy(payload)
    forged["rule_config"] = {"rank_imbalance_ratio": 3.0}
    with pytest.raises(ArtifactError, match="findings disagree"):
        validate_ranks(forged)
    forged = copy.deepcopy(payload)
    forged["rule_config"] = {"h2d_share_threshold": 0.5}
    with pytest.raises(ArtifactError, match="not evaluated by shardscope ranks"):
        validate_ranks(forged)
    # Consistent in itself but not with the directories: a changed steps.csv.
    (root / "rank-00001" / "steps.csv").write_text(rows([10.0, 9.0]))
    with pytest.raises(ArtifactError, match="stale source rank 1 steps.csv"):
        load_ranks(root)
    with pytest.raises(ArtifactError, match="stale"):
        write_ranks(root, payload, overwrite=True)


def test_unconfigured_comparison_persists_and_findings_say_so(tmp_path):
    root = two_ranks(tmp_path, [10.0], [30.0])
    payload = analyze_ranks(root)
    assert payload["rule_config"] is None
    assert payload["steps"][0]["ratio"] == 3.0
    assert [f["status"] for f in payload["findings"]] == ["unconfigured"]
    write_ranks(root, payload)
    assert load_ranks(root) == payload


# --- CLI --------------------------------------------------------------------------


def test_ranks_and_findings_commands(tmp_path, capsys):
    root = two_ranks(tmp_path, [10.0, 20.0, 30.0], [10.0, 8.0, 30.0])
    before = digests(root)
    assert main(["ranks", "--root", str(root), "--threshold", "rank_imbalance_ratio=2"]) == 0
    assert capsys.readouterr().out.strip() == str(root / RANKS_JSON)
    assert digests(root) == {**before, RANKS_JSON: digests(root)[RANKS_JSON]}
    assert main(["ranks", "--root", str(root)]) == 2
    assert "pass --overwrite" in capsys.readouterr().err
    assert (
        main(
            ["ranks", "--root", str(root), "--threshold", "h2d_share_threshold=0.1", "--overwrite"]
        )
        == 2
    )
    assert "not evaluated by shardscope ranks" in capsys.readouterr().err

    assert main(["findings", "--ranks", str(root)]) == 0
    text = capsys.readouterr().out
    assert "cross-rank diagnostics" in text and "world size 2" in text
    assert "3 of 3 training step(s) compared" in text
    assert (
        "rank_step_imbalance [rank_imbalance_ratio=2]: 2 below_threshold, 1 possible_rank_imbalance"
        in text
    )
    assert "possible_rank_imbalance: steps 1" in text
    assert "max(step_time_ms) / min(step_time_ms) 2.5 >= 2" in text
    assert "never timestamps" in text

    assert (
        main(
            [
                "findings",
                "--ranks",
                str(root),
                "--format",
                "json",
                "--status",
                "possible_rank_imbalance",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["ranks_recorded"] == [0, 1] and summary["compared_steps"] == 3
    (rule,) = summary["rules"]
    assert rule["rule_id"] == "rank_step_imbalance" and rule["triggered_steps"] == [1]
    assert [f["steps"] for f in rule["findings"]] == [[1]]
    assert rule["counts"] == {"below_threshold": 2, "possible_rank_imbalance": 1}

    with pytest.raises(SystemExit):
        main(["findings", "--run", str(root / "rank-00000"), "--ranks", str(root)])
    assert main(["findings", "--ranks", str(root), "--rule", "host_loader_wait"]) == 0
    assert "rank_step_imbalance" not in capsys.readouterr().out.split("\n", 3)[3]
    missing = tmp_path / "nothing"
    missing.mkdir()
    (missing / "rank-00000").mkdir()
    assert main(["findings", "--ranks", str(missing)]) == 2
    assert "no ranks.json" in capsys.readouterr().err
