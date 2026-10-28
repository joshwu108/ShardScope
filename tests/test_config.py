"""Configuration loading, validation and CLI precedence tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shardscope.config import (
    DATA_RESIDENCY_CHOICES,
    DEVICE_CHOICES,
    RunConfig,
    field_names,
    from_mapping,
    load_run_config,
)
from shardscope.errors import ConfigError, ShardScopeError

import train_single


def _write(tmp_path, payload) -> str:
    target = tmp_path / "run.json"
    target.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return str(target)


class TestDefaults:
    def test_defaults_match_the_training_fixture(self):
        # The fixture is the thing that actually runs; if these drift, a
        # config-driven run and a flag-driven run stop being comparable.
        config = RunConfig()

        assert config.steps == train_single.DEFAULT_STEPS
        assert config.batch_size == train_single.DEFAULT_BATCH_SIZE
        assert config.samples == train_single.DEFAULT_SAMPLES
        assert config.learning_rate == train_single.DEFAULT_LEARNING_RATE

    def test_choices_match_the_training_fixture(self):
        assert set(DEVICE_CHOICES) == set(train_single.DEVICE_CHOICES)
        assert set(DATA_RESIDENCY_CHOICES) == set(train_single.DATA_RESIDENCY_CHOICES)

    def test_config_is_immutable(self):
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            RunConfig().steps = 1

    def test_merged_returns_a_new_object_and_leaves_the_original_alone(self):
        original = RunConfig()

        updated = original.merged(steps=7)

        assert updated.steps == 7
        assert original.steps == RunConfig().steps
        assert updated is not original

    def test_merged_ignores_unspecified_overrides(self):
        # argparse yields None for a flag the user did not pass.
        config = RunConfig(steps=5).merged(steps=None, batch_size=2)

        assert config.steps == 5
        assert config.batch_size == 2

    def test_merged_validates_the_result(self):
        with pytest.raises(ConfigError, match="must be positive"):
            RunConfig().merged(steps=0)

    def test_merged_rejects_an_unknown_field(self):
        with pytest.raises(ConfigError, match="unknown config field"):
            RunConfig().merged(stepz=1)


class TestLoading:
    def test_an_empty_object_yields_the_defaults(self, tmp_path):
        assert load_run_config(_write(tmp_path, {})) == RunConfig()

    def test_values_from_the_file_are_applied(self, tmp_path):
        path = _write(
            tmp_path,
            {
                "steps": 10,
                "batch_size": 4,
                "samples": 32,
                "seed": 11,
                "learning_rate": 0.5,
                "device": "cpu",
                "data_residency": "device",
                "synchronize": False,
            },
        )

        config = load_run_config(path)

        assert config == RunConfig(
            steps=10,
            batch_size=4,
            samples=32,
            seed=11,
            learning_rate=0.5,
            device="cpu",
            data_residency="device",
            synchronize=False,
        )

    def test_unspecified_keys_keep_their_defaults(self, tmp_path):
        config = load_run_config(_write(tmp_path, {"steps": 3}))

        assert config.steps == 3
        assert config.batch_size == RunConfig().batch_size

    def test_underscore_prefixed_keys_are_treated_as_comments(self, tmp_path):
        path = _write(tmp_path, {"_comment": "loader sweep, arm 2", "steps": 3})

        assert load_run_config(path).steps == 3

    def test_round_trips_through_to_dict(self, tmp_path):
        original = RunConfig(steps=9, device="cpu", synchronize=False)

        assert load_run_config(_write(tmp_path, original.to_dict())) == original

    def test_accepts_a_path_object(self, tmp_path):
        target = tmp_path / "run.json"
        target.write_text(json.dumps({"steps": 2}))

        assert load_run_config(target).steps == 2


class TestActionableErrors:
    def test_every_error_is_a_shardscope_error(self, tmp_path):
        # The CLI catches ShardScopeError to print one actionable line.
        with pytest.raises(ShardScopeError):
            load_run_config(tmp_path / "absent.json")

    def test_missing_file_names_the_path_and_the_known_keys(self, tmp_path):
        with pytest.raises(ConfigError) as caught:
            load_run_config(tmp_path / "absent.json")

        message = str(caught.value)
        assert "no such config file" in message
        assert "absent.json" in message
        assert "data_residency" in message

    def test_a_directory_is_rejected_clearly(self, tmp_path):
        with pytest.raises(ConfigError, match="is a directory"):
            load_run_config(tmp_path)

    def test_an_undecodable_file_is_rejected_by_name(self, tmp_path):
        # A config is read the same way run artifacts are, so it must fail
        # the same way: one actionable line naming the file and the byte,
        # not a UnicodeDecodeError traceback out of pathlib.
        target = tmp_path / "latin1.json"
        target.write_bytes(b'{"_note": "caf\xe9", "steps": 3}')

        with pytest.raises(ConfigError) as caught:
            load_run_config(target)

        message = str(caught.value)
        assert "latin1.json" in message
        assert "not valid UTF-8" in message

    def test_a_config_with_non_ascii_text_loads(self, tmp_path):
        # The other half of the same fix: a UTF-8 config must load whatever
        # the machine's locale encoding happens to be.
        target = tmp_path / "utf8.json"
        target.write_bytes('{"_note": "caf\u00e9 \u2014 na\u00efve", "steps": 3}'.encode())

        assert load_run_config(target).steps == 3

    def test_malformed_json_reports_the_parser_position(self, tmp_path):
        with pytest.raises(ConfigError) as caught:
            load_run_config(_write(tmp_path, '{"steps": 3,}'))

        message = str(caught.value)
        assert "is not valid JSON" in message
        assert "line 1" in message

    @pytest.mark.parametrize("payload", ["[]", '"steps"', "3", "null"])
    def test_a_non_object_top_level_is_rejected(self, tmp_path, payload):
        with pytest.raises(ConfigError, match="top level must be a JSON object"):
            load_run_config(_write(tmp_path, payload))

    def test_an_unknown_key_suggests_the_closest_real_key(self, tmp_path):
        with pytest.raises(ConfigError) as caught:
            load_run_config(_write(tmp_path, {"batchsize": 4}))

        message = str(caught.value)
        assert "unknown config key 'batchsize'" in message
        assert "Did you mean 'batch_size'?" in message

    def test_an_unrecognisable_key_still_lists_the_known_keys(self, tmp_path):
        with pytest.raises(ConfigError) as caught:
            load_run_config(_write(tmp_path, {"zzzzzzzz": 4}))

        message = str(caught.value)
        assert "Did you mean" not in message
        assert all(name in message for name in field_names())

    @pytest.mark.parametrize("key", ["steps", "batch_size", "samples"])
    def test_a_non_integer_count_is_rejected(self, tmp_path, key):
        with pytest.raises(ConfigError, match="must be an integer"):
            load_run_config(_write(tmp_path, {key: "100"}))

    @pytest.mark.parametrize("key", ["steps", "batch_size", "samples"])
    @pytest.mark.parametrize("value", [0, -1])
    def test_a_non_positive_count_is_rejected(self, tmp_path, key, value):
        with pytest.raises(ConfigError, match="must be positive"):
            load_run_config(_write(tmp_path, {key: value}))

    def test_a_boolean_is_not_accepted_as_an_integer(self, tmp_path):
        # bool subclasses int in Python, so `"steps": true` would otherwise
        # quietly become a one-step run.
        with pytest.raises(ConfigError, match="must be an integer"):
            load_run_config(_write(tmp_path, {"steps": True}))

    def test_a_float_step_count_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="must be an integer"):
            load_run_config(_write(tmp_path, {"steps": 10.5}))

    def test_a_non_integer_seed_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="'seed' must be an integer"):
            load_run_config(_write(tmp_path, {"seed": 1.5}))

    def test_a_non_numeric_learning_rate_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="must be a number"):
            load_run_config(_write(tmp_path, {"learning_rate": "fast"}))

    @pytest.mark.parametrize("value", [0, -0.1])
    def test_a_non_positive_learning_rate_is_rejected(self, tmp_path, value):
        with pytest.raises(ConfigError, match="greater than 0"):
            load_run_config(_write(tmp_path, {"learning_rate": value}))

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_a_non_finite_learning_rate_is_rejected(self, tmp_path, literal):
        # json.loads accepts these non-standard literals, so the guard has
        # to be ours.
        with pytest.raises(ConfigError, match="must be finite|greater than 0"):
            load_run_config(_write(tmp_path, f'{{"learning_rate": {literal}}}'))

    def test_an_unknown_device_lists_the_supported_ones(self, tmp_path):
        with pytest.raises(ConfigError) as caught:
            load_run_config(_write(tmp_path, {"device": "gpu"}))

        message = str(caught.value)
        assert "'device' must be one of" in message
        assert all(choice in message for choice in DEVICE_CHOICES)

    def test_an_unknown_data_residency_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="'data_residency' must be one of"):
            load_run_config(_write(tmp_path, {"data_residency": "elsewhere"}))

    @pytest.mark.parametrize("value", [1, 0, "true"])
    def test_a_non_boolean_synchronize_is_rejected(self, tmp_path, value):
        # "synchronize": 1 must not pass: the synchronize policy decides
        # whether a step time measures execution or launch, so a sloppy
        # value here silently changes what every number means.
        with pytest.raises(ConfigError, match="must be true or false"):
            load_run_config(_write(tmp_path, {"synchronize": value}))

    def test_from_mapping_names_a_caller_supplied_source(self):
        with pytest.raises(ConfigError, match="inline-config: unknown config key"):
            from_mapping({"nope": 1}, source="inline-config")


class TestShippedExampleConfigs:
    """Every config shipped under examples/configs/ must actually load."""

    CONFIG_DIR = Path(__file__).resolve().parents[1] / "examples" / "configs"

    def test_the_directory_is_not_empty(self):
        assert sorted(self.CONFIG_DIR.glob("*.json"))

    @pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.json")), ids=lambda p: p.name)
    def test_a_shipped_config_validates(self, path):
        # A documented example that does not load is worse than no example.
        assert isinstance(load_run_config(path), RunConfig)

    def test_the_cpu_baseline_matches_the_fixture_defaults(self):
        # It is published as "the defaults"; if it drifts from them the
        # README is lying about what a baseline run does.
        config = load_run_config(self.CONFIG_DIR / "cpu_baseline.json")

        assert config == RunConfig(device="cpu")


class TestJsonParserLimits:
    @pytest.mark.parametrize(
        "payload",
        ['{"steps": ' + "9" * 5000 + "}", "[" * 100_000 + "]" * 100_000],
        ids=["oversized-int", "deep-nesting"],
    )
    def test_an_unparseable_config_is_a_config_error_naming_the_file(self, tmp_path, payload):
        # json.loads raises a plain ValueError past 4300 digits and a
        # RecursionError on deep nesting; neither is a JSONDecodeError, and
        # neither must reach the user as a traceback.
        path = tmp_path / "run.json"
        path.write_text(payload, encoding="utf-8")

        with pytest.raises(ConfigError) as excinfo:
            load_run_config(path)

        assert str(path) in str(excinfo.value)
