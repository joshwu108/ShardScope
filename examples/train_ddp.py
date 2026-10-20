"""Two-process DDP version of the training loop, launched with torchrun.

Status: STUB (Week 5 deliverable, requirements SS-07 / SS-08).

Ownership (PRD section 12.1): DDP synchronization and overlap choices are
student-owned. The agent may build the launch harness, per-rank trace
collection and the synthetic straggler fixture, but must not choose the
gradient-synchronization strategy or edit the GPU hot path.

Blocked until a host with >=2 CUDA devices is available. Do not report
DDP/NCCL numbers produced anywhere else (PRD section 8).
"""

from __future__ import annotations


def main() -> int:
    raise NotImplementedError("Week 5: requires >=2 CUDA devices and student-approved DDP design.")


if __name__ == "__main__":
    raise SystemExit(main())
