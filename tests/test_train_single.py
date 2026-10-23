"""Tests for the Week 1 training fixture (requirement SS-01).

What SS-01 actually demands, and therefore what these tests pin down:
a tiny, reproducible PyTorch training example with fixed-seed synthetic
input, able to run ~100 iterations and report per-step wall-clock time.

Reproducibility is the load-bearing property. Every later ShardScope claim
is a comparison between two runs, and a comparison is meaningless if the
fixture itself drifts between runs.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math

import pytest
import torch

import train_single


def test_synthetic_dataset_has_the_declared_shape_and_dtypes():
    images, labels = train_single.make_synthetic_dataset(n_samples=16, seed=0)

    assert images.shape == (16, 1, 28, 28)
    assert images.dtype == torch.float32
    assert labels.shape == (16,)
    assert labels.dtype == torch.int64
    assert int(labels.min()) >= 0
    assert int(labels.max()) < train_single.NUM_CLASSES


def test_synthetic_dataset_is_identical_for_the_same_seed():
    first = train_single.make_synthetic_dataset(n_samples=8, seed=1234)
    second = train_single.make_synthetic_dataset(n_samples=8, seed=1234)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_synthetic_dataset_differs_for_a_different_seed():
    images_a, _ = train_single.make_synthetic_dataset(n_samples=8, seed=1)
    images_b, _ = train_single.make_synthetic_dataset(n_samples=8, seed=2)

    assert not torch.equal(images_a, images_b)


def test_dataset_generation_does_not_disturb_global_rng_state():
    # A fixture that reseeds the global generator would silently change any
    # run that draws randomness after it.
    torch.manual_seed(99)
    before = torch.rand(4)

    torch.manual_seed(99)
    train_single.make_synthetic_dataset(n_samples=8, seed=7)
    after = torch.rand(4)

    assert torch.equal(before, after)


def test_model_maps_a_batch_to_class_logits():
    model = train_single.build_model(seed=0)
    batch = torch.zeros(5, 1, 28, 28)

    logits = model(batch)

    assert logits.shape == (5, train_single.NUM_CLASSES)


def test_model_is_small_enough_to_profile_quickly():
    # PRD v2.0 section 5, SS-01: a *small* training example. Keeping this
    # bounded keeps trace files and step times in the profiler's range.
    model = train_single.build_model(seed=0)
    parameters = sum(p.numel() for p in model.parameters())

    assert 0 < parameters < 100_000


def test_build_model_is_deterministic_for_a_seed():
    first = train_single.build_model(seed=5)
    second = train_single.build_model(seed=5)

    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(a, b)


def test_resolve_device_honours_an_explicit_cpu_request():
    assert train_single.resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_falls_back_to_cpu_without_an_accelerator(monkeypatch):
    monkeypatch.setattr(train_single, "_mps_is_usable", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert train_single.resolve_device("auto").type == "cpu"


def test_resolve_device_auto_prefers_mps_when_usable(monkeypatch):
    monkeypatch.setattr(train_single, "_mps_is_usable", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert train_single.resolve_device("auto").type == "mps"


def test_resolve_device_rejects_an_unknown_request():
    with pytest.raises(ValueError, match="unknown device"):
        train_single.resolve_device("tpu")


def test_run_training_returns_one_record_per_step():
    records = train_single.run_training(steps=5, batch_size=8, device="cpu", seed=0)

    assert len(records) == 5
    assert [r.step for r in records] == [0, 1, 2, 3, 4]


def test_every_step_reports_a_positive_wall_clock_duration():
    records = train_single.run_training(steps=3, batch_size=8, device="cpu", seed=0)

    assert all(r.step_time_ms > 0 for r in records)


def test_every_step_reports_a_finite_loss():
    records = train_single.run_training(steps=3, batch_size=8, device="cpu", seed=0)

    assert all(math.isfinite(r.loss) for r in records)


def test_identical_seeds_produce_identical_loss_trajectories():
    kwargs = {"steps": 4, "batch_size": 8, "device": "cpu", "seed": 42}
    first = train_single.run_training(**kwargs)
    second = train_single.run_training(**kwargs)

    assert [r.loss for r in first] == [r.loss for r in second]


def test_different_seeds_produce_different_loss_trajectories():
    first = train_single.run_training(steps=4, batch_size=8, device="cpu", seed=1)
    second = train_single.run_training(steps=4, batch_size=8, device="cpu", seed=2)

    assert [r.loss for r in first] != [r.loss for r in second]


@pytest.mark.parametrize("residency", ["host", "device"])
def test_both_data_residency_modes_run(residency):
    # GATE 1.2 is deliberately left open: both are implemented so the choice
    # can be settled by measurement rather than by assumption.
    records = train_single.run_training(
        steps=3, batch_size=8, device="cpu", seed=0, data_residency=residency
    )

    assert len(records) == 3
    assert all(r.data_residency == residency for r in records)


def test_data_residency_does_not_change_the_loss_trajectory():
    # Where the batch lives is a transfer-cost decision, not a maths
    # decision. If this ever fails, the two paths disagree numerically and
    # no before/after transfer comparison would be valid.
    host = train_single.run_training(
        steps=4, batch_size=8, device="cpu", seed=3, data_residency="host"
    )
    device = train_single.run_training(
        steps=4, batch_size=8, device="cpu", seed=3, data_residency="device"
    )

    assert [r.loss for r in host] == [r.loss for r in device]


def test_unknown_data_residency_is_rejected():
    with pytest.raises(ValueError, match="data_residency"):
        train_single.run_training(
            steps=1, batch_size=8, device="cpu", seed=0, data_residency="elsewhere"
        )


@pytest.mark.parametrize("steps", [0, -1])
def test_non_positive_step_counts_are_rejected(steps):
    with pytest.raises(ValueError, match="steps"):
        train_single.run_training(steps=steps, batch_size=8, device="cpu", seed=0)


def test_non_positive_batch_sizes_are_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        train_single.run_training(steps=1, batch_size=0, device="cpu", seed=0)


def test_step_record_is_immutable():
    record = train_single.run_training(steps=1, batch_size=8, device="cpu", seed=0)[0]

    with pytest.raises(dataclasses.FrozenInstanceError):
        record.step_time_ms = 0.0


def test_records_carry_the_synchronize_policy_that_produced_them():
    # Step timings are only comparable between runs that synchronized the
    # same way, so the policy travels with the measurement.
    synced = train_single.run_training(
        steps=2, batch_size=8, device="cpu", seed=0, synchronize=True
    )
    unsynced = train_single.run_training(
        steps=2, batch_size=8, device="cpu", seed=0, synchronize=False
    )

    assert all(r.synchronized for r in synced)
    assert not any(r.synchronized for r in unsynced)


def test_write_step_csv_emits_the_declared_header_and_one_row_per_step(tmp_path):
    records = train_single.run_training(steps=3, batch_size=8, device="cpu", seed=0)
    target = tmp_path / "steps.csv"

    train_single.write_step_csv(target, records)

    with target.open(newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == list(train_single.STEP_CSV_COLUMNS)
    assert len(rows) == 4
    assert rows[1][0] == "0"


def test_write_step_csv_creates_missing_parent_directories(tmp_path):
    records = train_single.run_training(steps=1, batch_size=8, device="cpu", seed=0)
    target = tmp_path / "nested" / "run" / "steps.csv"

    train_single.write_step_csv(target, records)

    assert target.is_file()


def test_write_step_csv_refuses_an_empty_record_set(tmp_path):
    with pytest.raises(ValueError, match="no records"):
        train_single.write_step_csv(tmp_path / "steps.csv", ())


class TestSummarize:
    """Descriptive statistics over a run's step times."""

    @staticmethod
    def _records(times):
        return tuple(
            train_single.StepRecord(
                step=i,
                loss=0.0,
                step_time_ms=t,
                data_residency="host",
                synchronized=True,
            )
            for i, t in enumerate(times)
        )

    def test_reports_the_step_count(self):
        stats = train_single.summarize(self._records([5.0, 1.0, 3.0]))

        assert stats["steps"] == 3

    def test_median_of_an_odd_number_of_steps_is_the_middle_value(self):
        stats = train_single.summarize(self._records([5.0, 1.0, 3.0]))

        assert stats["median_step_ms"] == 3.0

    def test_median_of_an_even_number_of_steps_averages_the_middle_pair(self):
        stats = train_single.summarize(self._records([1.0, 2.0, 3.0, 4.0]))

        assert stats["median_step_ms"] == 2.5

    def test_reports_the_extremes(self):
        stats = train_single.summarize(self._records([5.0, 1.0, 3.0]))

        assert stats["min_step_ms"] == 1.0
        assert stats["max_step_ms"] == 5.0

    def test_p95_of_twenty_steps_is_the_nineteenth_value(self):
        # Nearest-rank percentile: ceil(0.95 * 20) = 19th of 20 sorted values.
        stats = train_single.summarize(self._records([float(i) for i in range(1, 21)]))

        assert stats["p95_step_ms"] == 19.0

    def test_p95_of_a_single_step_is_that_step(self):
        stats = train_single.summarize(self._records([7.5]))

        assert stats["p95_step_ms"] == 7.5

    def test_p95_is_never_below_the_median(self):
        stats = train_single.summarize(self._records([float(i) for i in range(50)]))

        assert stats["p95_step_ms"] >= stats["median_step_ms"]

    def test_refuses_an_empty_record_set(self):
        with pytest.raises(ValueError, match="empty"):
            train_single.summarize(())


class TestMain:
    """The runnable entry point."""

    def test_returns_zero_on_a_successful_run(self, capsys):
        assert train_single.main(["--steps", "3", "--device", "cpu", "--quiet"]) == 0

    def test_prints_one_line_per_step_by_default(self, capsys):
        train_single.main(["--steps", "3", "--device", "cpu"])
        out = capsys.readouterr().out

        assert out.count("step ") == 3

    def test_quiet_suppresses_per_step_lines_but_keeps_the_summary(self, capsys):
        train_single.main(["--steps", "3", "--device", "cpu", "--quiet"])
        out = capsys.readouterr().out

        assert "step " not in out
        assert "median=" in out

    def test_summary_names_the_resolved_device_and_policies(self, capsys):
        train_single.main(["--steps", "2", "--device", "cpu", "--quiet"])
        out = capsys.readouterr().out

        assert "device=cpu" in out
        assert "residency=host" in out
        assert "synchronized=True" in out

    def test_no_sync_runs_are_labelled_as_launch_timings(self, capsys):
        train_single.main(["--steps", "2", "--device", "cpu", "--quiet", "--no-sync"])
        out = capsys.readouterr().out

        # Without this warning a reader could mistake launch time for
        # execution time, which is the single easiest way to publish a
        # bogus speedup.
        assert "measure launch, not execution" in out
        assert "synchronized=False" in out

    def test_out_dir_receives_the_three_run_artifacts(self, tmp_path, capsys):
        train_single.main(
            ["--steps", "3", "--device", "cpu", "--quiet", "--out-dir", str(tmp_path)]
        )

        assert (tmp_path / "steps.csv").is_file()
        assert (tmp_path / "environment.json").is_file()
        assert (tmp_path / "summary.json").is_file()

    def test_written_environment_records_that_the_run_was_not_profiled(self, tmp_path, capsys):
        # Requirement SS-04: an artifact must say which side of the
        # profiled/unprofiled split it came from.
        train_single.main(
            ["--steps", "2", "--device", "cpu", "--quiet", "--out-dir", str(tmp_path)]
        )
        payload = json.loads((tmp_path / "environment.json").read_text())

        assert payload["extra"]["profiled"] is False
        assert payload["extra"]["source"] == "examples/train_single.py"

    def test_written_environment_records_the_knobs_needed_to_repeat_the_run(self, tmp_path, capsys):
        train_single.main(
            [
                "--steps",
                "2",
                "--batch-size",
                "4",
                "--seed",
                "11",
                "--device",
                "cpu",
                "--quiet",
                "--out-dir",
                str(tmp_path),
            ]
        )
        extra = json.loads((tmp_path / "environment.json").read_text())["extra"]

        assert extra["steps"] == 2
        assert extra["batch_size"] == 4
        assert extra["seed"] == 11
        assert extra["device_resolved"] == "cpu"
        assert extra["synchronized"] is True

    def test_written_summary_matches_the_csv_row_count(self, tmp_path, capsys):
        train_single.main(
            ["--steps", "4", "--device", "cpu", "--quiet", "--out-dir", str(tmp_path)]
        )
        summary = json.loads((tmp_path / "summary.json").read_text())

        with (tmp_path / "steps.csv").open(newline="") as handle:
            rows = list(csv.reader(handle))

        assert summary["steps"] == 4
        assert len(rows) == 5

    def test_an_invalid_device_choice_is_rejected_by_the_parser(self):
        with pytest.raises(SystemExit):
            train_single.main(["--device", "tpu"])


class TestConfigFile:
    """``--config`` (PRD v2.0 section 11) and its precedence over flags."""

    @staticmethod
    def _config(tmp_path, payload) -> str:
        target = tmp_path / "run.json"
        target.write_text(json.dumps(payload))
        return str(target)

    def test_a_config_file_supplies_the_run_knobs(self, tmp_path, capsys):
        path = self._config(tmp_path, {"steps": 2, "device": "cpu", "data_residency": "device"})

        assert train_single.main(["--config", path, "--quiet"]) == 0
        out = capsys.readouterr().out

        assert "residency=device" in out
        assert "steps=2" in out

    def test_an_explicit_flag_overrides_the_config_file(self, tmp_path, capsys):
        path = self._config(tmp_path, {"steps": 2, "seed": 1, "device": "cpu"})

        train_single.main(["--config", path, "--seed", "99", "--quiet"])

        assert "seed=99" in capsys.readouterr().out

    def test_omitted_flags_do_not_override_the_config_file(self, tmp_path, capsys):
        # argparse yields None for an unpassed flag; that must not be read
        # as "the user asked for the default".
        path = self._config(tmp_path, {"steps": 2, "seed": 7, "device": "cpu"})

        train_single.main(["--config", path, "--quiet"])

        assert "seed=7" in capsys.readouterr().out

    def test_no_sync_overrides_a_synchronizing_config(self, tmp_path, capsys):
        path = self._config(tmp_path, {"steps": 2, "device": "cpu", "synchronize": True})

        train_single.main(["--config", path, "--no-sync", "--quiet"])

        assert "synchronized=False" in capsys.readouterr().out

    def test_sync_overrides_a_non_synchronizing_config(self, tmp_path, capsys):
        # Without an explicit --sync there would be no way to re-enable
        # synchronization from the CLI once a config file disabled it.
        path = self._config(tmp_path, {"steps": 2, "device": "cpu", "synchronize": False})

        train_single.main(["--config", path, "--sync", "--quiet"])

        assert "synchronized=True" in capsys.readouterr().out

    def test_a_missing_config_file_exits_two_with_an_actionable_message(self, tmp_path, capsys):
        assert train_single.main(["--config", str(tmp_path / "absent.json")]) == 2
        assert "no such config file" in capsys.readouterr().err

    def test_an_invalid_config_value_exits_two_without_training(self, tmp_path, capsys):
        path = self._config(tmp_path, {"steps": 0})

        assert train_single.main(["--config", path]) == 2
        captured = capsys.readouterr()

        assert "must be positive" in captured.err
        assert "step " not in captured.out

    def test_a_misspelled_config_key_is_reported_with_a_suggestion(self, tmp_path, capsys):
        path = self._config(tmp_path, {"batchsize": 4})

        assert train_single.main(["--config", path]) == 2
        assert "Did you mean 'batch_size'?" in capsys.readouterr().err

    def test_the_written_environment_records_the_config_file_used(self, tmp_path, capsys):
        # Provenance: a replayed run must be traceable back to the file that
        # described it (PRD v2.0 section 4, "Run contract").
        path = self._config(tmp_path, {"steps": 2, "device": "cpu"})
        out_dir = tmp_path / "run"

        train_single.main(["--config", path, "--quiet", "--out-dir", str(out_dir)])
        extra = json.loads((out_dir / "environment.json").read_text())["extra"]

        assert extra["config_file"] == path
        assert extra["steps"] == 2

    def test_the_written_environment_records_no_config_file_when_none_was_used(
        self, tmp_path, capsys
    ):
        train_single.main(
            ["--steps", "2", "--device", "cpu", "--quiet", "--out-dir", str(tmp_path)]
        )
        extra = json.loads((tmp_path / "environment.json").read_text())["extra"]

        assert extra["config_file"] is None


def test_non_positive_sample_counts_are_rejected():
    with pytest.raises(ValueError, match="n_samples"):
        train_single.make_synthetic_dataset(n_samples=0, seed=0)


def test_mps_usability_probe_returns_a_boolean():
    # Guards against the probe leaking a torch object or raising on a
    # platform without the backend.
    assert isinstance(train_single._mps_is_usable(), bool)
