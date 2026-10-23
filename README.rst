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
* ``shardscope report`` -- renders a recorded run as HTML, JSON or CSV,
  including a field-by-field statement of what the run does *not* contain.

Declared but **not implemented**: ``shardscope profile``, ``bench`` and
``compare``. Their flags appear in ``--help`` so the target interface is
visible, but invoking one exits non-zero and names the decision it is
waiting on. They do not produce partial output or empty run directories.

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

Reporting a run
===============

A run directory is whatever ``--out-dir`` wrote: ``steps.csv``,
``environment.json`` and ``summary.json``. ``shardscope report`` renders it.

.. code-block:: console

   $ python examples/train_single.py --steps 100 --device cpu --out-dir runs/baseline
   $ shardscope report --run runs/baseline --out reports/baseline.html
   wrote reports/baseline.html (html) from runs/baseline

The format follows the output suffix (``.html``, ``.htm``, ``.json``,
``.csv``) and ``--format`` overrides it. Without ``--out`` the report goes to
stdout, which makes ``--format csv`` and ``--format json`` usable in a pipe.
``--max-rows`` caps the inline per-step table; a truncated table says how
many rows it is hiding and that the rest are in ``steps.csv``.

The HTML report is one self-contained file: no scripts, no stylesheets, no
network references. Everything interpolated into it -- provenance strings,
file names, config paths -- is HTML-escaped, because a run can carry
arbitrary text from the program that was profiled and a report is the
artifact that gets shared.

What the report will not do:

* It computes nothing. Every number in it came from an artifact in the run
  directory. ``summary.json`` is shown as produced; if it is absent, the
  report says so instead of recomputing a replacement.
* It never fills a gap with zero. Each core report field from the PRD gets
  one of three states: *recorded*, *not recorded* (nothing captured it, and
  where a decision is pending the text names the open GATE), or
  ``not_observed`` (this run's own environment could not have produced it --
  no CUDA device for kernel activity, fewer than two for collectives, no
  per-step host-to-device copy under ``--data-residency device``).
* It attaches no diagnosis, threshold or confidence label. Those are open
  decisions, and a report that invented one would be putting words in the
  author's mouth.
* It does not reconcile artifacts that disagree. The per-step row count and
  ``summary.json``'s ``steps`` are shown side by side, and a mismatch is
  flagged rather than quietly resolved.

A damaged or unsupported run is refused with one actionable line and exit
code 2, and no partial report is written:

.. code-block:: console

   $ shardscope report --run runs/typo
   error: no such run directory: runs/typo. A run directory holds steps.csv
   and environment.json (see examples/train_single.py --out-dir).
   $ shardscope report --run runs
   error: runs has no steps.csv, so it is not a run directory. It does contain
   2 run directories: w1-cpu, w1-mps.

A per-step CSV carrying columns this version does not know -- the profiled
schema, for instance -- is refused rather than loaded with those columns
dropped, because a silently narrowed file would produce a report describing a
run that did not happen.

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
smoke check that runs the installed CLI, the training fixture, and the
fixture-run-to-HTML-report path end to end.

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

   src/shardscope/     cli, env, config, errors, artifacts, availability,
                       report/html_report; capture/parse/rules skeletons
   examples/           train_single.py, train_ddp.py
   benchmarks/         unprofiled throughput harness and comparisons
   tests/              unit and integration tests
   tools/              dev.py (all checks), bootstrap and safe-commit helpers

Planning documents, design notes, and all Markdown are intentionally
git-ignored and stay local. Public documentation lives in this file.

Run outputs (``runs/``) and rendered reports (``reports/``) are git-ignored
as well: both embed the producing machine's paths, device names and library
versions, and a report is meant to be reviewed before it is shared.

License
=========

MIT.
