"""Opt-in host iterator delay fixture, separate from baseline tensor indexing.

The data_wait marker covers next(iterator), including the requested host delay
and batch preparation. It measures host blocking, not GPU idle time.
"""

import argparse
import math
import time
from pathlib import Path

from shardscope.capture import profile_run
from shardscope.config import load_run_config
from shardscope.execution import _batch_index_schedule, make_synthetic_dataset


def delayed_batches(config, delay_ms):
    images, labels = make_synthetic_dataset(config.samples, config.seed)
    indices = _batch_index_schedule(config.samples, config.batch_size, config.steps, config.seed)

    def batches():
        for row in indices:
            time.sleep(delay_ms / 1000)
            yield images[row], labels[row]

    return batches()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--delay-ms", type=float, required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.delay_ms) or args.delay_ms < 0:
        parser.error("--delay-ms must be finite and non-negative")
    config = load_run_config(args.config)
    if config.data_residency != "host":
        parser.error("this iterator fixture requires data_residency=host")
    profile_run(
        config,
        args.out,
        config_file=str(args.config),
        batch_iterator=delayed_batches(config, args.delay_ms),
        batch_source_metadata=dict(
            kind="synthetic_delayed_host_iterator",
            delay_ms=args.delay_ms,
            scope="next(iterator): delay and tensor indexing",
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
