"""Conventional torchrun/DDP fixture with separate rank artifacts.

Example: torchrun --standalone --nproc-per-node=2 examples/train_ddp.py
--config CONFIG --out DIR [--profile]. CUDA/NCCL execution requires hardware
acceptance independently from the software configuration tests.
"""

from shardscope.distributed import main

if __name__ == "__main__":
    raise SystemExit(main())
