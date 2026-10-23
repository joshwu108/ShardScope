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
* Run configuration files, validated before anything executes.

Declared but **not implemented**: ``shardscope profile``, ``report``,
``bench`` and ``compare``. Their flags appear in ``--help`` so the target
interface is visible, but invoking one exits non-zero and names the decision
it is waiting on. They do not produce partial output or empty run
directories.

``shardscope --help`` lists the current state of every command.

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

Run configuration files
=======================

Any run knob can come from a JSON file instead of a flag, so a recorded run
can be replayed from its own provenance:

.. code-block:: console

   $ cat examples/configs/cpu_baseline.json
   {
     "_comment": "Single-device CPU baseline. ...",
     "steps": 100,
     "batch_size": 64,
     "samples": 2048,
     "seed": 0,
     "learning_rate": 0.01,
     "device": "cpu",
     "data_residency": "host",
     "synchronize": true
   }
   $ python examples/train_single.py --config examples/configs/cpu_baseline.json --quiet

Keys beginning with ``_`` are ignored, so a config can carry notes.
Explicitly passed flags override the file; omitted flags do not. The config
path is recorded in the run's ``environment.json``.

Validation happens before training starts, and says what to fix:

.. code-block:: console

   $ python examples/train_single.py --config bad.json
   error: bad.json: unknown config key 'devcie'. Did you mean 'device'? Known
   keys: steps, batch_size, samples, seed, learning_rate, device,
   data_residency, synchronize.
   $ echo $?
   2

Profiler settings -- the capture schedule, the activity set and the phase
marker names -- are deliberately **not** configurable yet. Those define what
a measurement means, and are not settled.

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

Every check is defined once, in ``tools/dev.py``, and CI invokes the same
script. "It passed locally" and "it passed in CI" therefore mean the same
commands ran.

.. code-block:: console

   $ python tools/dev.py --list     # show every check
   $ python tools/dev.py test       # pytest, no coverage gate
   $ python tools/dev.py cov        # pytest with the 80% coverage gate
   $ python tools/dev.py lint       # ruff check
   $ python tools/dev.py format     # ruff format (rewrites files)
   $ python tools/dev.py ci         # the full CI sequence, in order

``ci`` runs lint, format check, tests with the coverage gate, the commit-skill
bootstrap reproducibility check, the git ignore-policy assertions, and a
CLI/fixture smoke check.

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

   src/shardscope/     cli, env, config, errors; capture/parse/rules/report
   examples/           train_single.py, train_ddp.py
   benchmarks/         unprofiled throughput harness and comparisons
   tests/              unit and integration tests
   tools/              dev.py (all checks), bootstrap and safe-commit helpers

Planning documents, design notes, and all Markdown are intentionally
git-ignored and stay local. Public documentation lives in this file.

License
=========

MIT.
