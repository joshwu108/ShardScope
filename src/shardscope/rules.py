"""Inspectable host-wait comparisons with explicit, uncalibrated configuration."""

from __future__ import annotations

from dataclasses import dataclass

from shardscope.env import NOT_OBSERVED
from shardscope.errors import ArtifactError
from shardscope.parse import finite_number

__all__ = ["NOT_OBSERVED", "RuleConfig", "evaluate_rules"]

RULE_ID = "host_loader_wait"


@dataclass(frozen=True)
class RuleConfig:
    """A caller-selected threshold in host milliseconds, never a default."""

    loader_wait_threshold_ms: float

    def __post_init__(self):
        if not finite_number(self.loader_wait_threshold_ms) or self.loader_wait_threshold_ms < 0:
            raise ArtifactError("loader_wait_threshold_ms must be finite and nonnegative")

    def to_dict(self) -> dict:
        return {"loader_wait_threshold_ms": self.loader_wait_threshold_ms}


def evaluate_rules(steps: list[dict], config: RuleConfig | None) -> list[dict]:
    """Emit comparisons for clean, delayed, missing and unconfigured inputs.

    The caller supplies validated measurements with resolving evidence. A host
    wait can suggest input waiting, but cannot establish GPU starvation.
    """
    if config is None:
        return [
            {
                "rule_id": RULE_ID,
                "status": "unconfigured",
                "steps": [],
                "inputs": [],
                "comparison": None,
                "evidence": [],
                "interpretation": "No explicit loader-wait threshold configured; no diagnosis.",
            }
        ]
    outputs = []
    for step in steps:
        measured = step["measurements"].get("host_data_wait_ms")
        ready = bool(
            measured
            and measured.get("status") == "recorded"
            and finite_number(measured.get("value_ms"))
            and measured["value_ms"] >= 0
            and measured.get("evidence")
            and measured.get("basis") == "host_range"
        )
        value = measured["value_ms"] if ready else None
        exceeds = value >= config.loader_wait_threshold_ms if ready else None
        outputs.append(
            {
                "rule_id": RULE_ID,
                "status": "possible_input_wait"
                if exceeds
                else ("below_threshold" if ready else "unavailable"),
                "steps": [step["training_step"]],
                "inputs": [
                    {
                        "training_step": step["training_step"],
                        "metric": "host_data_wait_ms",
                        "measurement": measured,
                    }
                ],
                "comparison": {
                    "operator": ">=",
                    "threshold_ms": config.loader_wait_threshold_ms,
                    "value_ms": value,
                    "result": exceeds,
                },
                "evidence": measured["evidence"] if ready else [],
                "interpretation": (
                    "Host wait meets the configured threshold: possible input waiting; "
                    "does not establish GPU starvation or a confirmed root cause."
                    if exceeds
                    else (
                        "Host wait is below the configured threshold."
                        if ready
                        else "Host loader-wait evidence unavailable; not measured zero."
                    )
                ),
            }
        )
    return outputs
