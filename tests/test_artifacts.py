"""Run-artifact discovery, loading, validation and serialization.

These tests cover the artifact triple ``examples/train_single.py`` already
writes -- ``steps.csv``, ``environment.json``, ``summary.json`` -- and
nothing else. The *profiled* per-step schema (``data_wait_ms``,
``h2d_copy_ms``, ...) is the profiled metric schema and deliberately has no loader here;
the point of ``test_an_unknown_column_is_refused_not_dropped`` is that a
file carrying those columns is refused rather than silently narrowed.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from conftest import GOOD_ENVIRONMENT, GOOD_STEPS_CSV
from shardscope.artifacts import (
    ENVIRONMENT_JSON,
    STEP_COLUMNS,
    STEPS_CSV,
    SUMMARY_JSON,
    RunArtifacts,
    StepRow,
    discover_runs,
    load_run,
    load_steps_csv,
    looks_like_run,
    replace_file_contents,
    steps_csv_text,
    write_steps_csv,
)
from shardscope.errors import ArtifactError, ShardScopeError

HEADER = ",".join(STEP_COLUMNS)


def _csv(*rows: str) -> str:
    return "\n".join((HEADER, *rows)) + "\n"


class TestLoadStepsCsv:
    def test_reads_the_fixture_shape_with_real_types(self, make_run):
        rows = load_steps_csv(make_run() / STEPS_CSV)

        assert len(rows) == 3
        assert rows[0] == StepRow(
            step=0,
            loss=2.3827712535858154,
            step_time_ms=17.921874998137355,
            data_residency="host",
            synchronized=True,
        )
        assert isinstance(rows[0].step, int)
        assert isinstance(rows[0].synchronized, bool)

    def test_a_bounded_capture_window_may_start_anywhere(self, tmp_path):
        # A profiled window will not start at step 0; requiring 0..N would
        # reject a legitimate bounded capture (bounded capture schedule).
        path = tmp_path / STEPS_CSV
        path.write_text(_csv("10,1.0,2.0,host,True", "11,1.0,2.0,host,True"))

        assert [row.step for row in load_steps_csv(path)] == [10, 11]

    def test_a_trailing_blank_line_is_tolerated(self, tmp_path):
        path = tmp_path / STEPS_CSV
        path.write_text(_csv("0,1.0,2.0,host,True") + "\n")

        assert len(load_steps_csv(path)) == 1

    def test_an_undecodable_file_is_refused_with_a_reason(self, tmp_path):
        path = tmp_path / STEPS_CSV
        path.write_bytes(
            b"step,loss,step_time_ms,data_residency,synchronized\n0,1.0,2.0,h\xffst,True\n"
        )

        with pytest.raises(ArtifactError, match="UTF-8"):
            load_steps_csv(path)

    def test_a_missing_file_names_the_path(self, tmp_path):
        with pytest.raises(ArtifactError, match=str(tmp_path / STEPS_CSV)):
            load_steps_csv(tmp_path / STEPS_CSV)

    def test_a_directory_is_not_a_csv(self, tmp_path):
        with pytest.raises(ArtifactError, match="directory"):
            load_steps_csv(tmp_path)

    def test_an_empty_file_is_refused(self, tmp_path):
        path = tmp_path / STEPS_CSV
        path.write_text("")

        with pytest.raises(ArtifactError, match="no header"):
            load_steps_csv(path)

    def test_a_header_with_no_rows_is_refused(self, tmp_path):
        # Zero rows is not "a run with no measurements"; it is a broken file,
        # and the fixture refuses to write one.
        path = tmp_path / STEPS_CSV
        path.write_text(HEADER + "\n")

        with pytest.raises(ArtifactError, match="no step rows"):
            load_steps_csv(path)

    def test_a_missing_column_is_named(self, tmp_path):
        path = tmp_path / STEPS_CSV
        path.write_text("step,loss,step_time_ms,data_residency\n0,1.0,2.0,host\n")

        with pytest.raises(ArtifactError, match="synchronized"):
            load_steps_csv(path)

    def test_an_unknown_column_is_refused_not_dropped(self, tmp_path):
        # A profiled CSV must not load as if the profiled fields were absent.
        path = tmp_path / STEPS_CSV
        path.write_text(f"{HEADER},data_wait_ms\n0,1.0,2.0,host,True,3.0\n")

        with pytest.raises(ArtifactError) as excinfo:
            load_steps_csv(path)

        assert "data_wait_ms" in str(excinfo.value)
        assert "metric schema" in str(excinfo.value)

    def test_a_duplicated_column_is_refused(self, tmp_path):
        path = tmp_path / STEPS_CSV
        path.write_text(f"{HEADER},loss\n0,1.0,2.0,host,True,9.0\n")

        with pytest.raises(ArtifactError, match="duplicate"):
            load_steps_csv(path)

    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            ("0,1.0,2.0,host", "too few"),
            ("0,1.0,2.0,host,True,extra", "too many"),
            ("x,1.0,2.0,host,True", "integer"),
            ("0.5,1.0,2.0,host,True", "integer"),
            (",1.0,2.0,host,True", "integer"),
            ("0,nope,2.0,host,True", "number"),
            ("0,1.0,,host,True", "number"),
            ("0,1.0,nan,host,True", "finite"),
            ("0,1.0,inf,host,True", "finite"),
            ("0,1.0,-2.0,host,True", "negative"),
            ("0,1.0,2.0,elsewhere,True", "data_residency"),
            ("0,1.0,2.0,host,maybe", "true or false"),
        ],
    )
    def test_a_malformed_row_is_refused_with_its_row_number(self, tmp_path, row, expected):
        path = tmp_path / STEPS_CSV
        path.write_text(_csv("0,1.0,2.0,host,True", row))

        with pytest.raises(ArtifactError) as excinfo:
            load_steps_csv(path)

        message = str(excinfo.value)
        assert expected in message
        assert "row 3" in message, message

    def test_columns_out_of_order_are_refused(self, tmp_path):
        # The order is part of the contract with the producer; a silently
        # accepted reordering would pair every value with the wrong column.
        path = tmp_path / STEPS_CSV
        path.write_text("loss,step,step_time_ms,data_residency,synchronized\n1.0,0,2.0,host,True\n")

        with pytest.raises(ArtifactError, match="out of order"):
            load_steps_csv(path)

    def test_a_crlf_file_loads(self, tmp_path):
        path = tmp_path / STEPS_CSV
        path.write_bytes(_csv("0,1.0,2.0,host,True").replace("\n", "\r\n").encode())

        assert len(load_steps_csv(path)) == 1

    def test_a_repeated_step_index_is_refused(self, tmp_path):
        path = tmp_path / STEPS_CSV
        path.write_text(_csv("0,1.0,2.0,host,True", "0,1.0,2.0,host,True"))

        with pytest.raises(ArtifactError, match="repeat"):
            load_steps_csv(path)

    def test_a_decreasing_step_index_is_refused(self, tmp_path):
        path = tmp_path / STEPS_CSV
        path.write_text(_csv("5,1.0,2.0,host,True", "4,1.0,2.0,host,True"))

        with pytest.raises(ArtifactError, match="increasing"):
            load_steps_csv(path)

    def test_a_loss_may_be_non_finite(self, tmp_path):
        # A diverged run really can record nan; that is a measurement, not a
        # malformed file, and dropping it would hide the divergence.
        path = tmp_path / STEPS_CSV
        path.write_text(_csv("0,nan,2.0,host,True"))

        assert math.isnan(load_steps_csv(path)[0].loss)


class TestLoadRun:
    def test_loads_the_artifact_triple(self, make_run):
        run = load_run(make_run("baseline"))

        assert run.run_id == "baseline"
        assert len(run.steps) == 3
        assert run.summary["median_step_ms"] == 9.41929197870195

    def test_provenance_is_preserved_verbatim_including_unknown_keys(self, make_run):
        environment = {
            **GOOD_ENVIRONMENT,
            "future_field": {"nested": [1, 2]},
            "extra": {**GOOD_ENVIRONMENT["extra"], "custom_knob": "kept"},
        }

        run = load_run(make_run(environment=environment))

        assert run.environment == environment
        assert run.environment["extra"]["custom_knob"] == "kept"

    def test_an_absent_summary_stays_absent(self, make_run):
        run = load_run(make_run(summary=None))

        # Not {}, not zeros: a statistic nobody recorded has no value.
        assert run.summary is None

    def test_a_missing_steps_csv_names_the_file_and_the_directory(self, make_run):
        run_dir = make_run(steps_csv=None)

        with pytest.raises(ArtifactError) as excinfo:
            load_run(run_dir)

        assert STEPS_CSV in str(excinfo.value)
        assert str(run_dir) in str(excinfo.value)

    def test_a_missing_environment_is_refused_as_lost_provenance(self, make_run):
        # A run without provenance cannot be judged
        # comparable to anything, so it must not render as a report.
        with pytest.raises(ArtifactError) as excinfo:
            load_run(make_run(environment=None))

        assert ENVIRONMENT_JSON in str(excinfo.value)
        assert "provenance" in str(excinfo.value)

    def test_a_missing_directory_is_refused(self, tmp_path):
        with pytest.raises(ArtifactError, match="no such run directory"):
            load_run(tmp_path / "absent")

    def test_a_file_is_not_a_run_directory(self, tmp_path):
        target = tmp_path / "file.txt"
        target.write_text("hi")

        with pytest.raises(ArtifactError, match="not a directory"):
            load_run(target)

    def test_pointing_at_a_parent_of_runs_lists_the_runs_it_found(self, make_run, tmp_path):
        make_run("alpha")
        make_run("beta")

        with pytest.raises(ArtifactError) as excinfo:
            load_run(tmp_path)

        message = str(excinfo.value)
        assert "alpha" in message
        assert "beta" in message

    @pytest.mark.parametrize("artifact", [ENVIRONMENT_JSON, SUMMARY_JSON])
    def test_malformed_json_names_the_file_and_the_position(self, make_run, artifact):
        run_dir = make_run(**{artifact.removesuffix(".json"): "{not json"})

        with pytest.raises(ArtifactError) as excinfo:
            load_run(run_dir)

        assert artifact in str(excinfo.value)
        assert "line" in str(excinfo.value)

    @pytest.mark.parametrize("payload", ["[]\n", '"text"\n', "null\n"])
    def test_a_non_object_environment_is_refused(self, make_run, payload):
        with pytest.raises(ArtifactError, match="JSON object"):
            load_run(make_run(environment=payload))

    def test_an_undecodable_json_artifact_is_refused_with_a_reason(self, make_run):
        run_dir = make_run()
        (run_dir / ENVIRONMENT_JSON).write_bytes(b'{"platform": "\xff"}')

        with pytest.raises(ArtifactError, match="UTF-8"):
            load_run(run_dir)

    def test_a_non_object_extra_block_is_refused(self, make_run):
        with pytest.raises(ArtifactError, match="extra"):
            load_run(make_run(environment={**GOOD_ENVIRONMENT, "extra": [1, 2]}))

    def test_a_non_object_summary_is_refused(self, make_run):
        with pytest.raises(ArtifactError, match="JSON object"):
            load_run(make_run(summary="[1, 2]\n"))

    def test_the_profiling_status_is_read_not_assumed(self, make_run):
        profiled = {**GOOD_ENVIRONMENT, "extra": {**GOOD_ENVIRONMENT["extra"], "profiled": True}}

        assert load_run(make_run(environment=profiled)).profiled is True
        assert load_run(make_run()).profiled is False

    def test_an_unrecorded_profiling_status_is_unknown_not_false(self, make_run):
        extra = {
            key: value for key, value in GOOD_ENVIRONMENT["extra"].items() if key != "profiled"
        }

        run = load_run(make_run(environment={**GOOD_ENVIRONMENT, "extra": extra}))

        assert run.profiled is None

    def test_a_non_boolean_profiling_status_is_refused(self, make_run):
        extra = {**GOOD_ENVIRONMENT["extra"], "profiled": "yes"}

        with pytest.raises(ArtifactError, match="profiled"):
            load_run(make_run(environment={**GOOD_ENVIRONMENT, "extra": extra}))

    def test_sidecar_files_are_listed_as_present_artifacts(self, make_run):
        run = load_run(make_run(extra_files=("trace_rank_0.json",)))

        assert "trace_rank_0.json" in run.artifact_names
        assert STEPS_CSV in run.artifact_names

    def test_the_recorded_step_count_is_read_from_the_summary(self, make_run):
        assert load_run(make_run()).summary_step_count == 3

    @pytest.mark.parametrize("summary", [{}, {"steps": "three"}, {"steps": True}])
    def test_an_unreadable_recorded_step_count_is_unknown_not_guessed(self, make_run, summary):
        # The report shows the row count and this number side by side; an
        # unreadable value must read as unknown rather than as agreement.
        assert load_run(make_run(summary=summary)).summary_step_count is None

    def test_an_absent_summary_has_no_recorded_step_count(self, make_run):
        assert load_run(make_run(summary=None)).summary_step_count is None

    def test_a_non_array_device_list_is_refused(self, make_run):
        with pytest.raises(ArtifactError, match="device_names"):
            load_run(make_run(environment={**GOOD_ENVIRONMENT, "device_names": "gpu0"}))

    def test_an_environment_without_an_extra_block_still_loads(self, make_run):
        environment = {key: value for key, value in GOOD_ENVIRONMENT.items() if key != "extra"}

        run = load_run(make_run(environment=environment))

        assert run.extra == {}
        assert run.profiled is None

    def test_an_artifact_error_is_a_shardscope_error(self, tmp_path):
        # The CLI turns ShardScopeError into one actionable line; an artifact
        # failure outside that hierarchy would print a traceback instead.
        with pytest.raises(ShardScopeError):
            load_run(tmp_path / "absent")


class TestDiscoverRuns:
    def test_finds_run_directories_in_name_order(self, make_run, tmp_path):
        make_run("gamma")
        make_run("alpha")
        make_run("beta")

        assert [path.name for path in discover_runs(tmp_path)] == ["alpha", "beta", "gamma"]

    def test_a_directory_without_steps_is_not_a_run(self, make_run, tmp_path):
        make_run("real")
        (tmp_path / "notes").mkdir()

        assert [path.name for path in discover_runs(tmp_path)] == ["real"]

    def test_files_are_ignored(self, make_run, tmp_path):
        make_run("real")
        (tmp_path / "steps.csv").write_text(GOOD_STEPS_CSV)

        assert [path.name for path in discover_runs(tmp_path)] == ["real"]

    def test_an_empty_root_discovers_nothing(self, tmp_path):
        assert discover_runs(tmp_path) == ()

    def test_a_missing_root_is_refused(self, tmp_path):
        with pytest.raises(ArtifactError, match="no such"):
            discover_runs(tmp_path / "absent")

    def test_an_unreadable_root_is_an_actionable_error(self, tmp_path, monkeypatch):
        def deny(self):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(Path, "iterdir", deny)

        with pytest.raises(ArtifactError, match="cannot read the contents"):
            discover_runs(tmp_path)

    def test_a_file_root_is_refused(self, tmp_path):
        target = tmp_path / "file.txt"
        target.write_text("hi")

        with pytest.raises(ArtifactError, match="not a directory"):
            discover_runs(target)

    def test_a_run_root_is_recognized_but_not_its_own_child(self, make_run):
        run_dir = make_run()

        assert looks_like_run(run_dir) is True
        assert discover_runs(run_dir) == ()


class TestSerialization:
    def test_csv_round_trips_byte_for_byte(self, make_run, tmp_path):
        rows = load_steps_csv(make_run() / STEPS_CSV)
        out = tmp_path / "out" / STEPS_CSV

        write_steps_csv(out, rows)

        assert out.read_text() == GOOD_STEPS_CSV

    def test_full_precision_survives_a_round_trip(self, tmp_path):
        rows = (
            StepRow(
                step=0,
                loss=2.3827712535858154,
                step_time_ms=17.921874998137355,
                data_residency="host",
                synchronized=True,
            ),
        )
        path = tmp_path / STEPS_CSV

        write_steps_csv(path, rows)

        assert load_steps_csv(path) == rows

    def test_the_writer_does_not_translate_newlines(self, make_run, tmp_path):
        # steps_csv_text goes to the trouble of forcing \n; writing it back
        # through a translating text stream would undo that on Windows and
        # break the byte-identical round trip the docstrings promise.
        rows = load_steps_csv(make_run() / STEPS_CSV)
        out = tmp_path / STEPS_CSV

        write_steps_csv(out, rows)

        assert b"\r\n" not in out.read_bytes()

    def test_writing_no_rows_is_refused(self, tmp_path):
        with pytest.raises(ArtifactError, match="no rows"):
            write_steps_csv(tmp_path / STEPS_CSV, ())

    def test_a_run_round_trips_through_json(self, make_run):
        run = load_run(make_run("baseline", extra_files=("trace_rank_0.json",)))

        restored = RunArtifacts.from_dict(json.loads(json.dumps(run.to_dict())))

        assert restored == run

    def test_an_unsynchronized_run_round_trips(self, tmp_path, make_run):
        # --no-sync runs measure launch, not execution. The flag has to
        # survive the round trip or the two kinds of number become
        # indistinguishable.
        source = make_run(steps_csv=_csv("0,1.0,2.0,device,False"))
        rows = load_steps_csv(source / STEPS_CSV)

        assert rows[0].synchronized is False
        assert rows[0].data_residency == "device"

        out = tmp_path / "copy.csv"
        write_steps_csv(out, rows)

        assert load_steps_csv(out) == rows

    def test_a_run_without_a_summary_round_trips(self, make_run):
        run = load_run(make_run(summary=None))

        restored = RunArtifacts.from_dict(json.loads(json.dumps(run.to_dict())))

        assert restored == run
        assert restored.summary is None

    def test_the_json_view_keeps_an_absent_summary_null(self, make_run):
        payload = load_run(make_run(summary=None)).to_dict()

        assert payload["summary"] is None

    def test_from_dict_refuses_a_payload_that_is_not_a_run(self):
        with pytest.raises(ArtifactError, match="missing key"):
            RunArtifacts.from_dict({"nope": 1})

    def test_from_dict_refuses_a_non_object_payload(self):
        with pytest.raises(ArtifactError, match="JSON object"):
            RunArtifacts.from_dict([1, 2])

    def test_from_dict_refuses_a_step_record_with_a_missing_field(self, make_run):
        payload = load_run(make_run()).to_dict()
        del payload["steps"][1]["step_time_ms"]

        with pytest.raises(ArtifactError, match="step_time_ms"):
            RunArtifacts.from_dict(payload)

    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [
            ("step", True, "integer"),
            ("step", "0", "integer"),
            ("loss", "1.0", "number"),
            ("step_time_ms", -5, "negative"),
            ("step_time_ms", float("inf"), "finite"),
            ("step_time_ms", "2.0", "number"),
            ("data_residency", "nowhere", "data_residency"),
            ("synchronized", "false", "true or false"),
            ("synchronized", 1, "true or false"),
        ],
    )
    def test_from_dict_enforces_the_same_invariants_as_the_csv_loader(
        self, make_run, field, value, expected
    ):
        # Two loaders of the same record type must not trust their input to
        # different degrees: a JSON report is an input like any other.
        payload = load_run(make_run()).to_dict()
        payload["steps"][0][field] = value

        with pytest.raises(ArtifactError) as excinfo:
            RunArtifacts.from_dict(payload)

        assert expected in str(excinfo.value)

    def test_from_dict_accepts_an_integer_where_a_float_belongs(self, make_run):
        # JSON writes 2.0 as 2; refusing that would reject our own output.
        payload = load_run(make_run()).to_dict()
        payload["steps"][0]["step_time_ms"] = 2

        assert RunArtifacts.from_dict(payload).steps[0].step_time_ms == 2

    def test_from_dict_keeps_a_non_finite_loss(self, make_run):
        payload = load_run(make_run()).to_dict()
        payload["steps"][0]["loss"] = float("nan")

        assert math.isnan(RunArtifacts.from_dict(payload).steps[0].loss)

    def test_from_dict_refuses_a_non_object_summary(self, make_run):
        payload = {**load_run(make_run()).to_dict(), "summary": [1, 2]}

        with pytest.raises(ArtifactError, match="JSON object"):
            RunArtifacts.from_dict(payload)


class TestProducerParity:
    def test_the_column_contract_matches_the_producer(self):
        # The fixture writes these columns; if it ever changes them, this
        # loader must change with it rather than silently reject every run.
        import train_single

        assert STEP_COLUMNS == train_single.STEP_CSV_COLUMNS

    def test_a_real_fixture_run_loads(self, tmp_path):
        import train_single

        out_dir = tmp_path / "fixture-run"
        assert (
            train_single.main(
                ["--steps", "3", "--device", "cpu", "--quiet", "--out-dir", str(out_dir)]
            )
            == 0
        )

        run = load_run(out_dir)

        assert len(run.steps) == 3
        assert run.profiled is False
        assert run.environment["extra"]["source"] == "examples/train_single.py"
        assert set(run.summary) == {
            "steps",
            "median_step_ms",
            "p95_step_ms",
            "min_step_ms",
            "max_step_ms",
        }


class TestRecordedStepCount:
    """``summary.json``'s ``steps`` is read, never coerced.

    The report shows this number next to the row count and flags a
    disagreement. A damaged value that is quietly coerced can therefore
    manufacture agreement (3.5 truncating to 3) or crash the renderer (nan
    and inf cannot become an int at all). Either way the summary on disk is
    left exactly as written: the loader reports damage, it does not repair
    it.
    """

    @pytest.mark.parametrize(
        "payload", ['{"steps": NaN}', '{"steps": Infinity}', '{"steps": 1e999}']
    )
    def test_a_non_finite_count_is_unknown_and_the_summary_is_kept_verbatim(
        self, make_run, payload
    ):
        run = load_run(make_run(summary=payload))

        assert run.summary_step_count is None
        assert not math.isfinite(run.summary["steps"])

    def test_a_fractional_count_is_unknown_not_truncated(self, make_run):
        # 3.5 rows is not a count. Truncating it to 3 would make a damaged
        # summary agree with a three-row CSV.
        run = load_run(make_run(summary={"steps": 3.5}))

        assert run.summary_step_count is None
        assert run.summary["steps"] == 3.5

    def test_an_integer_valued_float_is_read_as_the_producer_writes_it(self, make_run):
        # examples/train_single.py records steps as float(count), so 3.0 is
        # the ordinary, valid spelling and must keep loading as 3.
        count = load_run(make_run(summary={"steps": 3.0})).summary_step_count

        assert count == 3
        assert isinstance(count, int)

    def test_a_negative_count_is_still_shown_so_the_disagreement_is_visible(self, make_run):
        assert load_run(make_run(summary={"steps": -3})).summary_step_count == -3


class TestFromDictInvariants:
    """``from_dict`` is a loader, and must refuse what the CSV loader refuses.

    Every failure below used to surface as a raw ``TypeError``,
    ``OverflowError`` or a silently accepted run. The CLI only turns
    :class:`ArtifactError` into an actionable line, so anything else reads
    as a crash -- and a silently accepted run is worse than a crash.
    """

    @pytest.fixture
    def payload(self, make_run):
        return load_run(make_run()).to_dict()

    @pytest.mark.parametrize("path", [5, None, "", ["a"]], ids=["int", "null", "empty", "list"])
    def test_the_path_must_be_a_non_empty_string(self, payload, path):
        with pytest.raises(ArtifactError, match="'path'"):
            RunArtifacts.from_dict({**payload, "path": path})

    @pytest.mark.parametrize("steps", ["abc", {}, 5, None], ids=["str", "dict", "int", "null"])
    def test_the_steps_must_be_a_json_array(self, payload, steps):
        with pytest.raises(ArtifactError, match="'steps'.*JSON array"):
            RunArtifacts.from_dict({**payload, "steps": steps})

    def test_an_empty_step_list_is_refused_like_an_empty_csv(self, payload):
        with pytest.raises(ArtifactError, match="no step rows"):
            RunArtifacts.from_dict({**payload, "steps": []})

    def test_a_step_record_that_is_not_an_object_is_named_by_position(self, payload):
        with pytest.raises(ArtifactError, match=r"steps\[1\].*JSON object"):
            RunArtifacts.from_dict({**payload, "steps": [payload["steps"][0], 5]})

    def test_a_missing_field_is_named_by_position(self, payload):
        del payload["steps"][2]["loss"]

        with pytest.raises(ArtifactError, match=r"steps\[2\].*loss"):
            RunArtifacts.from_dict(payload)

    def test_a_repeated_step_index_is_refused(self, payload):
        first = payload["steps"][0]

        with pytest.raises(ArtifactError, match=r"steps\[1\].*repeat"):
            RunArtifacts.from_dict({**payload, "steps": [first, dict(first)]})

    def test_a_decreasing_step_index_is_refused(self, payload):
        reordered = [payload["steps"][1], payload["steps"][0]]

        with pytest.raises(ArtifactError, match=r"steps\[1\].*increasing"):
            RunArtifacts.from_dict({**payload, "steps": reordered})

    @pytest.mark.parametrize("column", ["loss", "step_time_ms"])
    def test_a_number_too_large_to_represent_is_refused_not_crashed_on(self, payload, column):
        # JSON integers are unbounded; float() on 10**400 raises
        # OverflowError, which the CLI would print as a traceback.
        payload["steps"][0][column] = 10**400

        with pytest.raises(ArtifactError, match=f"'{column}'.*too large"):
            RunArtifacts.from_dict(payload)

    def test_a_large_step_index_is_an_integer_and_loads(self, payload):
        payload["steps"] = [{**payload["steps"][0], "step": 10**30}]

        assert RunArtifacts.from_dict(payload).steps[0].step == 10**30

    @pytest.mark.parametrize(
        "names", [5, "steps.csv", [1], [None]], ids=["int", "str", "int-item", "null-item"]
    )
    def test_the_artifact_names_must_be_an_array_of_strings(self, payload, names):
        # A string is iterable, so "steps.csv" would silently become nine
        # one-character file names; a non-string item crashes the renderer.
        with pytest.raises(ArtifactError, match="'artifact_names'"):
            RunArtifacts.from_dict({**payload, "artifact_names": names})

    def test_a_step_record_that_is_not_a_mapping_is_refused(self):
        with pytest.raises(ArtifactError, match="JSON object"):
            StepRow.from_dict(5)


class TestReaderLimits:
    def test_an_oversized_field_is_refused_with_the_file_and_line(self, tmp_path):
        # csv.reader enforces a field size limit (128 KiB by default) and
        # raises its own csv.Error, which is not a ShardScopeError and so
        # would reach the user as a traceback.
        path = tmp_path / STEPS_CSV
        path.write_text(_csv("0,1.0,2.0,host,True", f"1,1.0,2.0,{'x' * 200_000},True"))

        with pytest.raises(ArtifactError) as excinfo:
            load_steps_csv(path)

        message = str(excinfo.value)
        assert str(path) in message
        assert "line 3" in message
        assert "field larger" in message


class TestReplaceFileContents:
    """A single-file write that never leaves a half-written destination.

    The text goes to a sibling temporary file, which then replaces the
    destination. A reader sees the old file or the new one, never a prefix
    of the new one, and a failed write leaves the old file alone.
    """

    def test_writes_the_exact_bytes_without_newline_translation(self, tmp_path):
        target = tmp_path / "out" / "steps.csv"
        text = "a,b\n1,2\né\n"

        assert replace_file_contents(target, text) == target
        assert target.read_bytes() == text.encode("utf-8")

    def test_replaces_an_existing_file_and_leaves_nothing_else_behind(self, tmp_path):
        target = tmp_path / "report.html"
        target.write_text("old")

        replace_file_contents(target, "new")

        assert target.read_text() == "new"
        assert [path.name for path in tmp_path.iterdir()] == ["report.html"]

    def test_a_failed_replacement_preserves_the_previous_file(self, tmp_path, monkeypatch):
        target = tmp_path / "report.html"
        target.write_text("previous, valid")

        def no_space(_src, _dst):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(os, "replace", no_space)

        with pytest.raises(OSError, match="No space"):
            replace_file_contents(target, "half-written replacement")

        assert target.read_text() == "previous, valid"
        assert [path.name for path in tmp_path.iterdir()] == ["report.html"]

    def test_a_failed_write_preserves_the_previous_file(self, tmp_path, monkeypatch):
        # The failure strikes while the bytes are going out, before any
        # replacement: the common full-disk case.
        target = tmp_path / "report.html"
        target.write_text("previous, valid")
        real_fdopen = os.fdopen

        class Failing:
            def __init__(self, handle):
                self._handle = handle

            def write(self, _text):
                raise OSError(28, "No space left on device")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self._handle.close()

        monkeypatch.setattr(os, "fdopen", lambda fd, *a, **k: Failing(real_fdopen(fd, *a, **k)))

        with pytest.raises(OSError, match="No space"):
            replace_file_contents(target, "replacement")

        assert target.read_text() == "previous, valid"
        assert [path.name for path in tmp_path.iterdir()] == ["report.html"]

    def test_the_temporary_file_is_a_sibling_of_the_destination(self, tmp_path, monkeypatch):
        # A temp file in another directory could be on another filesystem,
        # where os.replace is not a rename and not atomic.
        seen = {}
        real_replace = os.replace

        def spy(src, dst):
            seen["src"] = Path(src)
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", spy)
        target = tmp_path / "nested" / "report.html"

        replace_file_contents(target, "x")

        assert seen["src"].parent == target.parent
        assert not seen["src"].exists()

    def test_the_destination_is_readable_by_others_under_the_default_umask(self, tmp_path):
        # tempfile's helpers create 0600 files; replacing a report with one
        # would quietly make shared reports unreadable to other users. The destination
        # should look like any file created under the caller's umask.
        if os.name != "posix":
            pytest.skip("permission bits are POSIX-specific")
        previous = os.umask(0o022)
        try:
            target = tmp_path / "report.html"
            replace_file_contents(target, "x")
        finally:
            os.umask(previous)

        assert target.stat().st_mode & 0o777 == 0o644

    def test_a_directory_in_the_way_is_an_os_error_and_leaves_no_temp_file(self, tmp_path):
        target = tmp_path / "report.html"
        target.mkdir()

        with pytest.raises(OSError):
            replace_file_contents(target, "x")

        assert target.is_dir()
        assert [path.name for path in tmp_path.iterdir()] == ["report.html"]

    def test_a_taken_temporary_name_is_skipped(self, tmp_path, monkeypatch):
        import secrets

        target = tmp_path / "report.html"
        taken = target.with_name(".report.html.0000000000000000.tmp")
        taken.write_text("someone else's")
        tokens = iter(["0000000000000000", "1111111111111111"])
        monkeypatch.setattr(secrets, "token_hex", lambda _n: next(tokens))

        replace_file_contents(target, "mine")

        assert target.read_text() == "mine"
        assert taken.read_text() == "someone else's"


class TestProducerBytes:
    def test_the_recorded_csv_is_byte_identical_to_the_readers_rendering(self, tmp_path):
        # The promise steps_csv_text makes. Before the producer switched to
        # "\\n" it held only after universal-newline translation.
        import train_single

        out_dir = tmp_path / "fixture-run"
        train_single.main(["--steps", "2", "--device", "cpu", "--quiet", "--out-dir", str(out_dir)])

        run = load_run(out_dir)

        assert (out_dir / STEPS_CSV).read_bytes() == steps_csv_text(run.steps).encode("utf-8")


class TestJsonParserLimits:
    """Failures json.loads raises that are not JSONDecodeError.

    CPython refuses an integer literal over 4300 digits with a plain
    ``ValueError`` and a deeply nested document with ``RecursionError``.
    Neither is a ShardScopeError, so left alone both reach the user as a
    traceback instead of naming the damaged file.
    """

    OVERSIZED = '{"steps": ' + "9" * 5000 + "}"
    DEEP = "[" * 100_000 + "]" * 100_000

    @pytest.mark.parametrize("artifact", [ENVIRONMENT_JSON, SUMMARY_JSON])
    @pytest.mark.parametrize("payload", [OVERSIZED, DEEP], ids=["oversized-int", "deep-nesting"])
    def test_an_unparseable_json_artifact_names_the_file(self, make_run, artifact, payload):
        run_dir = make_run(**{artifact.removesuffix(".json"): payload})

        with pytest.raises(ArtifactError) as excinfo:
            load_run(run_dir)

        assert artifact in str(excinfo.value)


class TestNumericSpellings:
    @pytest.mark.parametrize(
        "row",
        [
            "1_0,1.0,2.0,host,True",
            "0,1_0.5,2.0,host,True",
            "0,1.0,1_0,host,True",
            "١,1.0,2.0,host,True",
            "0,1.0,١٠,host,True",
        ],
        ids=["int-underscore", "loss-underscore", "time-underscore", "int-arabic", "time-arabic"],
    )
    def test_python_only_numeric_spellings_are_refused(self, tmp_path, row):
        # int() and float() accept underscores and non-ASCII digits. The
        # producer never writes them, so a cell like 1_0 is a hand edit, and
        # loading it as 10 would quietly accept a file that is not ours.
        path = tmp_path / STEPS_CSV
        path.write_text(_csv("0,1.0,2.0,host,True", row) if not row.startswith("0,") else _csv(row))

        with pytest.raises(ArtifactError) as excinfo:
            load_steps_csv(path)

        assert "row" in str(excinfo.value)
