"""Report rendering: availability, escaping, round trips, no-fabrication.

Two invariants carry most of these tests:

* A report labels profiled diagnostic figures separately from unprofiled
  throughput figures (requirement SS-04), and says so in words.
* A measurement nobody recorded is rendered as absent, with the reason --
  never as 0, never as an empty cell that reads like 0 (PRD v2.0 section 8
  "Guardrails").
"""

from __future__ import annotations

import json

import pytest

from conftest import GOOD_ENVIRONMENT, GOOD_STEPS_CSV, GOOD_SUMMARY
from shardscope.artifacts import STEP_COLUMNS, STEPS_CSV, RunArtifacts, load_run
from shardscope.availability import (
    NOT_OBSERVED,
    NOT_RECORDED,
    RECORDED,
    REPORT_FIELDS,
    field_statuses,
)
from shardscope.errors import ReportError, ShardScopeError
from shardscope.report import (
    REPORT_FORMATS,
    format_for_path,
    render,
    render_html,
    render_json,
    render_steps_csv,
    write_report,
)

HEADER = ",".join(STEP_COLUMNS)


def _csv(*rows: str) -> str:
    return "\n".join((HEADER, *rows)) + "\n"


def _status(run, field):
    return next(status for status in field_statuses(run) if status.field == field)


def _with_extra(**items):
    return {**GOOD_ENVIRONMENT, "extra": {**GOOD_ENVIRONMENT["extra"], **items}}


class TestFieldAvailability:
    def test_every_prd_core_report_field_gets_a_status(self, make_run):
        run = load_run(make_run())

        assert tuple(status.field for status in field_statuses(run)) == REPORT_FIELDS

    def test_step_time_is_recorded_from_the_csv(self, make_run):
        status = _status(load_run(make_run()), "step_time_ms")

        assert status.state == RECORDED
        assert STEPS_CSV in status.detail

    @pytest.mark.parametrize(
        "field",
        [
            "data_wait_ms",
            "h2d_copy_ms",
            "gpu_activity",
            "collective_activity",
            "per_rank_step_ms",
            "suspected_bottleneck",
            "confidence_or_visibility",
            "evidence_refs",
        ],
    )
    def test_an_unmeasured_field_is_absent_with_a_reason(self, make_run, field):
        status = _status(load_run(make_run()), field)

        assert status.state in (NOT_RECORDED, NOT_OBSERVED)
        assert len(status.detail) > 20, status.detail

    @pytest.mark.parametrize("field", ["data_wait_ms", "suspected_bottleneck"])
    def test_a_field_blocked_on_a_decision_names_the_gate(self, make_run, field):
        # "Unimplemented" and "waiting on a decision the student owns" are
        # different states; the report must not blur them.
        assert "GATE" in _status(load_run(make_run()), field).detail

    def test_gpu_activity_is_not_observed_when_the_run_recorded_no_cuda_device(self, make_run):
        status = _status(load_run(make_run()), "gpu_activity")

        assert status.state == NOT_OBSERVED
        assert "0" in status.detail and "CUDA" in status.detail

    def test_gpu_activity_on_a_cuda_host_is_merely_unrecorded(self, make_run):
        environment = {
            **GOOD_ENVIRONMENT,
            "cuda_available": True,
            "cuda_version": "12.4",
            "device_count": 1,
            "device_names": ["NVIDIA A100"],
        }

        status = _status(load_run(make_run(environment=environment)), "gpu_activity")

        # The host could have produced kernel activity; nothing captured it.
        assert status.state == NOT_RECORDED
        assert "GATE 2" in status.detail

    def test_collectives_are_not_observed_on_a_host_that_cannot_run_ddp(self, make_run):
        environment = {
            **GOOD_ENVIRONMENT,
            "cuda_available": True,
            "device_count": 1,
            "device_names": ["NVIDIA A100"],
        }

        status = _status(load_run(make_run(environment=environment)), "collective_activity")

        assert status.state == NOT_OBSERVED
        assert "1" in status.detail

    def test_collectives_on_a_two_gpu_host_are_merely_unrecorded(self, make_run):
        environment = {
            **GOOD_ENVIRONMENT,
            "cuda_available": True,
            "device_count": 2,
            "device_names": ["NVIDIA A100", "NVIDIA A100"],
        }

        status = _status(load_run(make_run(environment=environment)), "collective_activity")

        assert status.state == NOT_RECORDED

    def test_the_ddp_capability_rule_is_the_one_env_py_defines(self, make_run):
        # Two spellings of "can this host run DDP" is how a report comes to
        # claim a distributed result from a host that cannot run one.
        from shardscope.env import distributed_capable

        assert distributed_capable(cuda_available=True, device_count=2) is True
        assert distributed_capable(cuda_available=True, device_count=1) is False
        assert distributed_capable(cuda_available=False, device_count=4) is False

    def test_host_residency_reports_an_unmeasured_copy(self, make_run):
        status = _status(load_run(make_run()), "h2d_copy_ms")

        assert status.state == NOT_RECORDED
        assert "host" in status.detail

    def test_device_residency_reports_the_copy_as_structurally_absent(self, make_run):
        run = load_run(make_run(steps_csv=_csv("0,1.0,2.0,device,True")))

        status = _status(run, "h2d_copy_ms")

        assert status.state == NOT_OBSERVED
        assert "GATE 1.2" in status.detail
        # Structurally absent is not zero milliseconds.
        assert "0 ms" not in status.detail

    def test_mixed_residency_rows_are_reported_rather_than_averaged(self, make_run):
        run = load_run(make_run(steps_csv=_csv("0,1.0,2.0,host,True", "1,1.0,2.0,device,True")))

        detail = _status(run, "h2d_copy_ms").detail

        assert "device" in detail and "host" in detail

    def test_the_profiling_status_is_recorded_when_the_run_recorded_it(self, make_run):
        assert _status(load_run(make_run()), "profiled").state == RECORDED

    def test_an_unrecorded_profiling_status_is_absent_not_false(self, make_run):
        extra = {k: v for k, v in GOOD_ENVIRONMENT["extra"].items() if k != "profiled"}
        run = load_run(make_run(environment={**GOOD_ENVIRONMENT, "extra": extra}))

        status = _status(run, "profiled")

        assert status.state == NOT_RECORDED
        assert "unprofiled" not in status.detail.lower().replace("not unprofiled", "")

    def test_a_trace_sidecar_is_listed_as_evidence_without_being_interpreted(self, make_run):
        run = load_run(make_run(extra_files=("trace_rank_0.json",)))

        status = _status(run, "evidence_refs")

        assert "trace_rank_0.json" in status.detail
        assert status.state == NOT_RECORDED

    @pytest.mark.parametrize("field", ["gpu_activity", "collective_activity"])
    @pytest.mark.parametrize(
        "environment",
        [
            {key: value for key, value in GOOD_ENVIRONMENT.items() if key != "device_count"},
            {key: value for key, value in GOOD_ENVIRONMENT.items() if key != "cuda_available"},
            {**GOOD_ENVIRONMENT, "cuda_available": True, "device_count": "two"},
            {**GOOD_ENVIRONMENT, "cuda_available": "yes"},
        ],
        ids=["no-count", "no-flag", "count-not-an-integer", "flag-not-a-boolean"],
    )
    def test_an_unrecorded_device_inventory_is_not_reported_as_no_device(
        self, make_run, environment, field
    ):
        # Saying "this run recorded 0 CUDA devices" about provenance that
        # recorded no such thing is a fabricated fact, exactly like turning an
        # absent measurement into zero.
        status = _status(load_run(make_run(environment=environment)), field)

        assert status.state == NOT_RECORDED
        assert "0 CUDA device" not in status.detail
        assert "inventory" in status.detail

    def test_a_negative_device_count_is_not_a_device_count(self, make_run):
        environment = {**GOOD_ENVIRONMENT, "cuda_available": True, "device_count": -1}

        status = _status(load_run(make_run(environment=environment)), "gpu_activity")

        assert status.state == NOT_RECORDED

    def test_an_unrecorded_device_inventory_names_the_offending_key(self, make_run):
        environment = {**GOOD_ENVIRONMENT, "device_count": "two"}

        detail = _status(load_run(make_run(environment=environment)), "gpu_activity").detail

        assert "device_count" in detail

    def test_the_run_manifest_names_the_provenance_files_found(self, make_run):
        status = _status(load_run(make_run()), "run_manifest")

        assert status.state == RECORDED
        assert "environment.json" in status.detail


class TestHtmlReport:
    def test_renders_a_self_contained_document(self, make_run):
        html = render_html(load_run(make_run("baseline")))

        assert html.startswith("<!DOCTYPE html>")
        assert html.rstrip().endswith("</html>")
        assert "baseline" in html
        # No scripts and no network fetches: a report is read from disk, and
        # a trace can carry arbitrary strings from the profiled program.
        assert "<script" not in html.lower()
        assert "http://" not in html and "https://" not in html

    def test_text_from_the_run_is_escaped(self, make_run):
        hostile = "<script>alert('xss')</script>"
        environment = {
            **_with_extra(config_file=f'cfg "{hostile}".json', source=hostile),
            "platform": hostile,
            "device_names": [hostile],
        }

        html = render_html(load_run(make_run(f"run-{hostile}", environment=environment)))

        assert hostile not in html
        assert "&lt;script&gt;" in html
        assert "alert(&#x27;xss&#x27;)" in html or "alert('xss')" not in html

    def test_non_ascii_provenance_survives_the_round_trip(self, make_run):
        # Device names and config paths come from the machine that ran the
        # job; a non-UTF-8 read would corrupt them or crash on them.
        name = "NVIDIA A100 \u2014 b\u00fcro-\u30b5\u30fc\u30d0\u30fc"
        environment = {**GOOD_ENVIRONMENT, "device_names": [name], "platform": name}

        run = load_run(make_run(environment=environment))
        html = render_html(run)

        assert run.environment["platform"] == name
        assert name in html

    def test_an_unprofiled_run_is_labelled_and_fenced_off_from_throughput(self, make_run):
        html = render_html(load_run(make_run()))

        assert "unprofiled" in html.lower()
        assert "throughput" in html.lower()

    def test_a_profiled_run_says_its_numbers_are_diagnostic(self, make_run):
        html = render_html(load_run(make_run(environment=_with_extra(profiled=True))))

        assert "profiled" in html.lower()
        assert "SS-04" in html

    def test_an_unknown_profiling_status_is_not_rendered_as_unprofiled(self, make_run):
        extra = {k: v for k, v in GOOD_ENVIRONMENT["extra"].items() if k != "profiled"}

        html = render_html(load_run(make_run(environment={**GOOD_ENVIRONMENT, "extra": extra})))

        assert "profiling status was not recorded" in html.lower()

    def test_recorded_statistics_are_shown_exactly_as_produced(self, make_run):
        html = render_html(load_run(make_run()))

        for key, value in GOOD_SUMMARY.items():
            assert key in html
            assert repr(value) in html

    def test_an_absent_summary_is_declared_absent_not_zeroed(self, make_run):
        html = render_html(load_run(make_run(summary=None)))

        assert "summary.json" in html
        assert "not present" in html
        assert "median_step_ms" not in html
        assert "0.0 ms" not in html

    def test_the_provenance_block_is_rendered_in_full(self, make_run):
        html = render_html(load_run(make_run()))

        for key in ("torch_version", "python_version", "machine", "cuda_available"):
            assert key in html
        for key in ("seed", "batch_size", "data_residency", "source"):
            assert key in html

    def test_every_step_row_is_rendered_when_it_fits(self, make_run):
        html = render_html(load_run(make_run()))

        assert "2.3827712535858154" in html
        assert "17.921874998137355" in html

    def test_a_long_run_is_truncated_with_an_explicit_count(self, make_run):
        rows = [f"{index},1.0,{index}.5,host,True" for index in range(5)]
        run = load_run(make_run(steps_csv=_csv(*rows)))

        html = render_html(run, max_rows=2)

        assert "2 of 5" in html
        assert STEPS_CSV in html
        assert "4.5" not in html

    def test_a_non_positive_row_cap_is_refused(self, make_run):
        with pytest.raises(ReportError, match="max_rows"):
            render_html(load_run(make_run()), max_rows=0)

    def test_the_row_count_and_the_recorded_count_are_shown_side_by_side(self, make_run):
        html = render_html(load_run(make_run()))

        assert "3" in html
        assert "steps.csv" in html

    def test_a_disagreement_between_artifacts_is_surfaced_not_reconciled(self, make_run):
        run = load_run(make_run(summary={**GOOD_SUMMARY, "steps": 99.0}))

        html = render_html(run)

        assert "disagree" in html

    def test_an_unknown_recorded_count_is_not_called_a_disagreement(self, make_run):
        html = render_html(load_run(make_run(summary={})))

        assert "disagree" not in html

    def test_a_run_without_a_settings_block_still_renders(self, make_run):
        environment = {key: value for key, value in GOOD_ENVIRONMENT.items() if key != "extra"}

        html = render_html(load_run(make_run(environment=environment)))

        assert "Run settings" not in html
        assert "Provenance" in html

    def test_the_guardrails_are_stated_in_the_report_itself(self, make_run):
        html = render_html(load_run(make_run()))

        assert "do not sum" in html.lower()
        assert "not_observed" in html

    def test_the_step_table_mirrors_the_csv_spelling_of_the_sync_policy(self, make_run):
        html = render_html(load_run(make_run()))

        assert "<td>True</td>" in html

    def test_a_run_rebuilt_from_json_without_a_file_list_still_renders(self, make_run):
        payload = load_run(make_run()).to_dict()
        payload["artifact_names"] = []

        html = render_html(RunArtifacts.from_dict(payload))

        assert "Files in the run directory" not in html
        assert "Per-step records" in html

    def test_rendering_is_deterministic(self, make_run):
        run = load_run(make_run())

        assert render_html(run, generated_at="2026-10-23T00:00:00Z") == render_html(
            run, generated_at="2026-10-23T00:00:00Z"
        )

    def test_the_generation_timestamp_is_shown_when_given(self, make_run):
        html = render_html(load_run(make_run()), generated_at="2026-10-23T00:00:00Z")

        assert "2026-10-23T00:00:00Z" in html

    def test_an_escaped_timestamp_cannot_inject_markup(self, make_run):
        html = render_html(load_run(make_run()), generated_at="<b>now</b>")

        assert "<b>now</b>" not in html


class TestJsonAndCsvViews:
    def test_the_json_view_round_trips_back_into_a_run(self, make_run):
        run = load_run(make_run())

        restored = RunArtifacts.from_dict(json.loads(render_json(run)))

        assert restored == run

    def test_the_json_view_keeps_absent_measurements_null(self, make_run):
        payload = json.loads(render_json(load_run(make_run(summary=None))))

        assert payload["summary"] is None

    def test_a_non_finite_loss_survives_the_json_view(self, make_run):
        # A diverged run can record nan. json.dumps spells it as the
        # non-standard token NaN; this round trip is what the docstring
        # promises, and the caveat is documented there.
        run = load_run(make_run(steps_csv=_csv("0,nan,2.0,host,True")))

        restored = RunArtifacts.from_dict(json.loads(render_json(run)))

        assert restored == run or str(restored.steps[0].loss) == "nan"

    def test_the_json_view_carries_the_availability_statuses(self, make_run):
        payload = json.loads(render_json(load_run(make_run())))

        states = {entry["field"]: entry["state"] for entry in payload["availability"]}

        assert states["step_time_ms"] == RECORDED
        assert states["data_wait_ms"] == NOT_RECORDED

    def test_the_csv_view_round_trips_the_producer_file(self, make_run):
        assert render_steps_csv(load_run(make_run())) == GOOD_STEPS_CSV

    def test_the_csv_renderer_refuses_an_empty_run(self):
        from shardscope.artifacts import steps_csv_text
        from shardscope.errors import ArtifactError

        with pytest.raises(ArtifactError, match="no rows"):
            steps_csv_text(())


class TestWriteReport:
    @pytest.mark.parametrize("fmt", REPORT_FORMATS)
    def test_writes_each_format_and_creates_parent_directories(self, make_run, tmp_path, fmt):
        out = tmp_path / "reports" / f"baseline.{fmt}"

        written = write_report(load_run(make_run()), out, fmt=fmt)

        assert written == out
        assert out.read_text().strip() != ""

    def test_a_failed_render_leaves_no_partial_file(self, make_run, tmp_path):
        out = tmp_path / "reports" / "baseline.html"

        with pytest.raises(ReportError):
            write_report(load_run(make_run()), out, fmt="html", max_rows=0)

        assert not out.exists()
        assert not out.parent.exists()

    def test_a_directory_in_the_way_of_the_output_is_reported(self, make_run, tmp_path):
        out = tmp_path / "report.html"
        out.mkdir()

        with pytest.raises(ReportError) as excinfo:
            write_report(load_run(make_run()), out, fmt="html")

        assert str(out) in str(excinfo.value)

    def test_an_unwritable_parent_path_is_reported(self, make_run, tmp_path):
        blocker = tmp_path / "notes.txt"
        blocker.write_text("not a directory")

        with pytest.raises(ReportError, match="cannot write"):
            write_report(load_run(make_run()), blocker / "report.html", fmt="html")

    def test_an_unknown_format_is_refused(self, make_run, tmp_path):
        with pytest.raises(ReportError, match="format"):
            write_report(load_run(make_run()), tmp_path / "r.html", fmt="pdf")

    @pytest.mark.parametrize("fmt", REPORT_FORMATS)
    def test_a_non_positive_row_cap_is_refused_for_every_format(self, make_run, fmt):
        # The flag is accepted by the CLI whatever the format, so accepting a
        # nonsensical value for two of three formats is a trap.
        with pytest.raises(ReportError, match="max_rows"):
            render(load_run(make_run()), fmt, max_rows=0)

    def test_render_dispatches_on_format(self, make_run):
        run = load_run(make_run())

        assert render(run, "csv") == render_steps_csv(run)
        assert render(run, "json") == render_json(run)

    @pytest.mark.parametrize(
        ("name", "fmt"),
        [("r.html", "html"), ("r.htm", "html"), ("r.json", "json"), ("r.csv", "csv")],
    )
    def test_a_format_can_be_inferred_from_the_output_suffix(self, tmp_path, name, fmt):
        assert format_for_path(tmp_path / name) == fmt

    def test_an_unknown_suffix_is_refused_with_the_choices(self, tmp_path):
        with pytest.raises(ReportError) as excinfo:
            format_for_path(tmp_path / "report.pdf")

        assert "--format" in str(excinfo.value)
        for fmt in REPORT_FORMATS:
            assert fmt in str(excinfo.value)

    def test_a_report_error_is_a_shardscope_error(self, make_run, tmp_path):
        with pytest.raises(ShardScopeError):
            write_report(load_run(make_run()), tmp_path / "r.html", fmt="pdf")


class TestDamagedRecordedCounts:
    @pytest.mark.parametrize("payload", ['{"steps": NaN}', '{"steps": 3.5}', '{"steps": 1e999}'])
    def test_an_unreadable_count_renders_as_unknown_not_as_a_crash_or_agreement(
        self, make_run, payload
    ):
        # int(nan) raises and int(3.5) is 3: the first crashed the renderer,
        # the second made a damaged summary agree with a three-row CSV.
        html = render_html(load_run(make_run(summary=payload)))

        assert "no readable <code>steps</code> count" in html
        assert "disagree" not in html

    def test_the_damaged_value_is_still_shown_in_the_statistics_table(self, make_run):
        # Unknown as a *count*, but the recorded value itself is evidence of
        # the damage and stays visible verbatim.
        html = render_html(load_run(make_run(summary={"steps": 3.5})))

        assert "3.5" in html


class TestWriteReportProtectsTheEvidence:
    """``--out`` must never replace an artifact of the run being reported.

    ``report --run runs/x --out runs/x/steps.csv`` would render the run and
    then write the rendering over the evidence it came from. The check runs
    before the output is opened and follows symlinks and relative aliases,
    because those resolve to the same bytes on disk.
    """

    @pytest.mark.parametrize(
        ("name", "fmt"),
        [("steps.csv", "csv"), ("environment.json", "json"), ("summary.json", "json")],
    )
    def test_refuses_to_overwrite_a_source_artifact(self, make_run, name, fmt):
        run_dir = make_run()
        before = (run_dir / name).read_bytes()

        with pytest.raises(ReportError, match="artifact of the run"):
            write_report(load_run(run_dir), run_dir / name, fmt=fmt)

        assert (run_dir / name).read_bytes() == before

    def test_refuses_a_relative_alias_of_a_source_artifact(self, make_run):
        run_dir = make_run()
        alias = run_dir / ".." / run_dir.name / "steps.csv"

        with pytest.raises(ReportError, match="artifact of the run"):
            write_report(load_run(run_dir), alias, fmt="csv")

    def test_refuses_a_symlink_to_a_source_artifact(self, make_run, tmp_path):
        run_dir = make_run()
        link = tmp_path / "elsewhere.csv"
        link.symlink_to(run_dir / "steps.csv")

        with pytest.raises(ReportError, match="artifact of the run"):
            write_report(load_run(run_dir), link, fmt="html")

        assert (run_dir / "steps.csv").read_text() == GOOD_STEPS_CSV

    def test_refuses_a_sidecar_artifact_too(self, make_run):
        run_dir = make_run(extra_files=("trace_rank_0.json",))

        with pytest.raises(ReportError, match="artifact of the run"):
            write_report(load_run(run_dir), run_dir / "trace_rank_0.json", fmt="json")

    def test_a_new_file_inside_the_run_directory_is_not_a_collision(self, make_run):
        # Only existing evidence is protected; where a report goes is the
        # caller's choice.
        run_dir = make_run()

        written = write_report(load_run(run_dir), run_dir / "report.html", fmt="html")

        assert written.read_text().startswith("<!DOCTYPE html>")

    def test_a_failed_replacement_preserves_the_previous_report(
        self, make_run, tmp_path, monkeypatch
    ):
        import os

        out = tmp_path / "report.html"
        out.write_text("<!DOCTYPE html>previous, complete report")

        def no_space(_src, _dst):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(os, "replace", no_space)

        with pytest.raises(ReportError, match="cannot write"):
            write_report(load_run(make_run()), out, fmt="html")

        assert out.read_text() == "<!DOCTYPE html>previous, complete report"
        assert [path.name for path in tmp_path.iterdir() if path.is_file()] == ["report.html"]

    def test_the_csv_report_is_written_with_lf_endings_and_utf8(self, make_run, tmp_path):
        name = "büro-サーバー"
        run = load_run(make_run(environment={**GOOD_ENVIRONMENT, "platform": name}))

        as_csv = write_report(run, tmp_path / "r.csv", fmt="csv")
        as_html = write_report(run, tmp_path / "r.html", fmt="html")

        assert b"\r\n" not in as_csv.read_bytes()
        assert as_csv.read_bytes() == GOOD_STEPS_CSV.encode("utf-8")
        assert name.encode("utf-8") in as_html.read_bytes()

    def test_an_unanswerable_collision_check_refuses_rather_than_guessing(
        self, make_run, tmp_path, monkeypatch
    ):
        # If the filesystem will not say whether two paths are the same file,
        # writing anyway could be the overwrite the check exists to prevent.
        from pathlib import Path

        out = tmp_path / "report.html"
        out.write_text("existing")

        def denied(self, _other):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(Path, "samefile", denied)

        with pytest.raises(ReportError, match="cannot check"):
            write_report(load_run(make_run()), out, fmt="html")

        assert out.read_text() == "existing"
