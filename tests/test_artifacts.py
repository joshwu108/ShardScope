"""Run-artifact discovery, loading, validation and serialization.

These tests cover the artifact triple ``examples/train_single.py`` already
writes -- ``steps.csv``, ``environment.json``, ``summary.json`` -- and
nothing else. The *profiled* per-step schema (``data_wait_ms``,
``h2d_copy_ms``, ...) is GATE 3.1/3.2 and deliberately has no loader here;
the point of ``test_an_unknown_column_is_refused_not_dropped`` is that a
file carrying those columns is refused rather than silently narrowed.
"""

from __future__ import annotations

import json
import math
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
        # reject a legitimate bounded capture (GATE 2.1).
        path = tmp_path / STEPS_CSV
        path.write_text(_csv("10,1.0,2.0,host,True", "11,1.0,2.0,host,True"))

        assert [row.step for row in load_steps_csv(path)] == [10, 11]

    def test_a_trailing_blank_line_is_tolerated(self, tmp_path):
        path = tmp_path / STEPS_CSV
        path.write_text(_csv("0,1.0,2.0,host,True") + "\n")

        assert len(load_steps_csv(path)) == 1

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
        assert "GATE 3.1" in str(excinfo.value)

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
        # PRD v2.0 section 4: a run without its provenance cannot be judged
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
