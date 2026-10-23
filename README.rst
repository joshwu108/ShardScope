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

Install
=========

Requires Python 3.10-3.12 -- the range PyTorch publishes wheels for, and the
range CI tests (3.10 and 3.12 legs). On Apple silicon the interpreter must be
``arm64``; x86_64 Python has no modern PyTorch wheel. Check with
``python -c "import platform; print(platform.machine())"``.

.. code-block:: console

   $ uv venv --python 3.12 .venv
   $ source .venv/bin/activate
   $ uv pip install -e ".[dev]"

``pip`` works identically if you prefer it: ``python -m venv .venv`` then
``pip install -e ".[dev]"``. The ``dev`` extra adds pytest, pytest-cov, ruff
and docutils, which are what ``tools/dev.py`` needs; without it the test and
lint gates cannot run.

Verify the install -- both commands must agree on the version:

.. code-block:: console

   $ shardscope --version
   shardscope 0.0.1
   $ shardscope env
   ShardScope environment
     python          3.12.14
     platform        macOS-27.0-arm64-arm-64bit
     machine         arm64
     torch           2.14.1
     accelerator     mps
     cuda            not available
     cudnn           not available
     nccl            not available
     cuda devices    0
     ddp capable     False
     note            DDP/NCCL results must not be reported from this host.

The values above are this machine's. ``ddp capable`` is the gate on any
multi-GPU claim: if it says ``False``, no DDP or NCCL result may be reported
from this host.

``python tools/dev.py smoke`` checks the same thing and more, including that
the console script on ``PATH`` and the package this interpreter imports are
the same install.

Quick start
===========

Record a run, then render it. These are two separate commands because a run
is an artifact: it is produced once and read many times.

.. code-block:: console

   $ python examples/train_single.py --steps 100 --device auto --quiet --out-dir runs/baseline
   device=mps residency=host synchronized=True seed=0
   steps=100 median=1.511 ms p95=2.342 ms min=1.294 ms max=355.430 ms
   wrote runs/baseline/steps.csv, environment.json, summary.json
   $ shardscope report --run runs/baseline --out reports/baseline.html
   wrote reports/baseline.html (html) from runs/baseline

Those step times are one machine's, from one timed fixture run. They are not
throughput results and not a benchmark -- see `Honest-measurement rules`_.
Drop ``--quiet`` to see a line per step.

Every command exits ``0`` on success, ``2`` for a mistake you can fix (a bad
path, an invalid config, an unimplemented command), and ``141`` when the
reader of its stdout pipe hangs up.

A refused command writes nothing: bad arguments, an invalid config and an
unusable ``--out-dir`` are all caught before the first training step, and a
report is rendered in full before the output file is opened. The one case
that can still leave a half-written run directory is a write that fails
partway through the three artifacts -- a full disk, say. The fixture says so
and names the directory, and ``shardscope report`` then refuses the
incomplete run rather than describing it.

The training fixture
====================

``examples/train_single.py --help`` lists every knob. Two switches exist
because they change what the number *means*, not just how fast it is:

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
``--max-rows`` caps the per-step table in the HTML view; a truncated table
says how many rows it is hiding and that the rest are in ``steps.csv``. The
``json`` and ``csv`` views always carry every row.

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

Limitations
===========

What this version cannot do, and why. None of these are oversights: each one
waits on a measurement decision that has to be made deliberately, and
guessing would produce a number nobody could defend.

* **No profiler capture.** ``shardscope profile`` is declared but refuses to
  run. The capture schedule, the activity set, and the ``record_function``
  marker names every later rule keys off are unsettled, and they determine
  what a trace *means*.
* **No profiled per-step fields.** ``data_wait_ms``, ``h2d_copy_ms`` and the
  rest have no reader and no renderer. A per-step CSV that carries them is
  refused, not partially loaded.
* **No bottleneck diagnosis, thresholds or confidence labels.** A report
  states that a diagnosis is absent and names the decision it waits on.
* **No throughput benchmark or before/after comparison.** ``bench`` and
  ``compare`` refuse to run: without an agreed warmup count, repeat count
  and variance test, a percentage change would be undefendable.
* **No statistics computed by the report.** It prints what ``summary.json``
  recorded and derives nothing, not even a mean.
* **Single-device only.** ``examples/train_ddp.py`` is a skeleton. Nothing
  here has produced a multi-GPU or NCCL measurement, and ``shardscope env``
  reports whether the host could.
* **No multi-run or sweep reporting.** ``report`` takes one run directory.
* **Reports are local artifacts.** ``runs/`` and ``reports/`` are
  git-ignored: both embed the producing machine's paths, device names and
  library versions.

Troubleshooting
===============

``shardscope: command not found``
  The package is not installed into the active interpreter, or the virtual
  environment is not activated. Re-run ``uv pip install -e ".[dev]"`` after
  ``source .venv/bin/activate``.

``shardscope --version`` disagrees with ``python -c "import shardscope"``
  Another ``shardscope`` is earlier on ``PATH``, or the editable install is
  stale. ``python tools/dev.py smoke`` reports this explicitly; fix it by
  reinstalling and checking ``which -a shardscope``.

``No module named torch`` or a resolution failure during install
  Check the interpreter is 3.10-3.12 and, on Apple silicon, ``arm64``.
  PyTorch publishes no wheel outside that range.

``error: 'profile' is not implemented yet ...``
  Expected. The message names the decision the command waits on. Same for
  ``bench`` and ``compare``.

``error: <dir> has no steps.csv, so it is not a run directory``
  ``--run`` wants one run directory, not the parent that holds several. The
  message lists the run directories it found underneath.

``error: <file>: unexpected column(s): data_wait_ms``
  The CSV uses the profiled schema, which has no reader yet. The message
  names the open decisions. The column is not dropped, because a silently
  narrowed file would produce a report describing a run that did not happen.

``error: --out-dir <path> exists and is not a directory``
  Raised before the first training step, so nothing is lost. Point
  ``--out-dir`` at a new or existing directory.

``error: the run finished but its artifacts could not be written to <dir>``
  The measurements are gone -- the run has to be repeated. Usually a full
  disk or a permission change during the run. The directory may hold a
  partial artifact triple; ``shardscope report`` refuses it rather than
  rendering a run that was not fully recorded.

``error: cannot tell the report format from 'x.pdf'``
  Use an ``.html``, ``.htm``, ``.json`` or ``.csv`` suffix, or pass
  ``--format`` explicitly. PDF is not a supported format.

Exit code 141 from ``shardscope report ... | head``
  Not an error: 128 + SIGPIPE, meaning the reader stopped reading. Nothing
  is printed to stderr. Use ``--out FILE`` if you want the whole report.

A test is skipped rather than run
  Tests needing hardware this machine lacks are marked ``gpu`` or
  ``distributed`` and skip rather than fake a result. ``shardscope env``
  says what the host has. The integration tests skip if the console script
  is not installed.

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

``ci`` runs, in order:
``lint``, ``format-check``, ``cov`` (pytest plus the 80% coverage gate),
``readme-check`` (this file parses with no docutils warnings), the
commit-skill bootstrap reproducibility check, the git ignore-policy
assertions, and ``smoke``.

``smoke`` is the end-to-end gate: it checks the console script's version
against the installed distribution, runs ``shardscope env``, runs the
training fixture with and without ``--config``, and renders a real recorded
run to HTML. ``tests/test_integration.py`` covers the same path as spawned
processes and asserts on the output rather than only the exit codes.

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

Two things that look like failures and are not:

* ``bootstrap_commit_skill.py`` exits ``3`` when the reference committing
  convention named in the PRD is absent from this machine. The skill is
  still written, carrying a banner that says it was not reconciled. CI
  treats ``3`` as the expected state rather than papering over it.
* ``--authorize`` refuses to touch an existing authorization file, and the
  template it creates has ``authorized: false``. Enabling automated commits
  is a hand edit by the repository owner, never an agent's own decision.

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
