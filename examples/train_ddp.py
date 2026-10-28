"""Distributed training fixture for torchrun.

Not implemented. Requires at least two CUDA devices and a defined gradient
synchronization and overlap strategy. DDP/NCCL measurements must come from
an actual distributed run on compatible hardware.
"""

from __future__ import annotations


def main() -> int:
    raise NotImplementedError(
        "DDP requires >=2 CUDA devices and a defined synchronization strategy."
    )


if __name__ == "__main__":
    raise SystemExit(main())
