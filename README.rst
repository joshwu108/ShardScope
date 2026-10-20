==========
ShardScope
==========

Distributed training bottleneck profiler for PyTorch.

ShardScope records what the CPU, data loader, GPUs, and GPU-to-GPU
communication do during a bounded window of training steps, then reports
where each step loses time, which rank is waiting, and what change might
help. You verify any suggestion by rerunning a separate, unprofiled
throughput benchmark.

It is a diagnostic scanner, not an autotuner.

Status
======

Early. Implemented today:

* ``shardscope env`` -- reproducibility snapshot (Python, PyTorch, CUDA,
  cuDNN, NCCL, device inventory, whether this host can run DDP at all).
* ``examples/train_single.py`` -- the tiny fixed-seed training fixture that
  later profiling runs measure. Runs on CPU, MPS or CUDA; emits per-step
  CSV plus an environment and summary sidecar.

Not implemented yet: ``shardscope profile``, ``shardscope report``,
``shardscope bench``. They exit non-zero rather than pretending to work.

The training fixture
====================

.. code-block:: console

   $ python examples/train_single.py --steps 100 --device auto --out-dir runs/baseline

Two switches exist because they change what the number *means*, not just how
fast it is:

``--no-sync``
  Stops synchronizing the device around each step. Timings then measure
  kernel launch rather than execution, and the run says so in its output.
  On an M4 Max this reads roughly half the synchronized step time -- which
  is how a bogus speedup gets published.

``--data-residency {host,device}``
  ``host`` copies each batch to the device per step, so a host-to-device
  transfer exists and can be measured. ``device`` pre-places the whole
  dataset, so no such transfer exists at all. Both paths train on identical
  batches in an identical order, so their step times are comparable.

Honest-measurement rules
========================

These are project invariants, not style preferences:

* Profiled diagnostic runs and unprofiled throughput benchmarks are separate
  commands. Throughput claims only ever come from unprofiled runs, because
  profiling overhead distorts step time.
* GPU kernels, host-to-device copies, and collectives overlap. Their
  durations do not sum to step time, so ShardScope does not derive
  "percent wasted" figures from naive sums.
* When an event class is not visible in the running PyTorch/CUDA/NCCL build,
  it is reported as ``not_observed`` -- never as a plausible number.
* Multi-GPU and NCCL results are reported only from hosts that actually ran
  them. ``shardscope env`` says whether this host qualifies.

Install
=========

Requires Python 3.10-3.12 (PyTorch wheel coverage). On Apple silicon, make
sure the interpreter is ``arm64``; x86_64 Python has no modern PyTorch wheel.

.. code-block:: console

   $ uv venv --python 3.12 .venv
   $ source .venv/bin/activate
   $ uv pip install -e ".[dev]"
   $ shardscope env

Development
===========

.. code-block:: console

   $ pytest -q                      # unit tests
   $ pytest -q --cov                # coverage, gate at 80%
   $ ruff check . && ruff format .  # lint and format

Tests that need hardware this machine lacks are marked ``gpu`` or
``distributed`` and are skipped rather than faked.

Local commit workflow
=====================

Automated local commits are opt-in and run through a guarded helper. After
cloning, regenerate the git-ignored agent commit skill:

.. code-block:: console

   $ python tools/bootstrap_commit_skill.py
   $ python tools/commit_safe.py --authorize    # then edit the auth file by hand
   $ python tools/commit_safe.py -m "feat: ..." --dry-run

``tools/commit_safe.py`` runs the tests, stages only allowlisted paths,
re-reads the staged set, and refuses Markdown, planning documents, trace
exports, and unreviewed GPU-path changes. It never pushes, merges, rebases,
amends, or force-pushes.

Repository layout
=================

.. code-block:: text

   src/shardscope/     capture, parse, rules, report, cli, env
   examples/           train_single.py, train_ddp.py
   benchmarks/         unprofiled throughput harness and comparisons
   tests/              unit and integration tests
   tools/              tracked bootstrap and safe-commit helpers

Planning documents, design notes, and all Markdown are intentionally
git-ignored and stay local. Public documentation lives in this file.

License
=========

MIT.
