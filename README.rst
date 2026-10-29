==========
ShardScope
==========

Training-run inspection tools for PyTorch, with distributed bottleneck
profiling in development.

ShardScope currently records a reproducible single-device training fixture,
validates its artifacts, and renders reports with per-step timings and
execution metadata. CPU development requires no GPU setup.

Profiler capture, trace analysis, bottleneck diagnosis, throughput benchmarks,
and DDP execution are not implemented. The ``profile``, ``bench``, and
``compare`` commands expose their planned options but exit with a dependency
message and produce no artifacts.

Development setup
=================

The package supports Python 3.10-3.12. CI runs on Linux with Python 3.10 and
3.12. Runtime dependencies are PyTorch and NumPy; the ``dev`` extra installs
pytest, coverage, Ruff, and docutils.

From a checkout:

.. code-block:: console

   $ uv venv --python 3.12 .venv
   $ source .venv/bin/activate
   $ uv pip install -e ".[dev]"
   $ shardscope --version
   $ python tools/dev.py smoke

With pip, create the environment using a supported interpreter:

.. code-block:: console

   $ python3.12 -m venv .venv
   $ source .venv/bin/activate
   $ python -m pip install -e ".[dev]"

Keep the virtual environment active when running commands below. On Apple
silicon, use an ARM64 interpreter. ``shardscope env`` reports the detected
backend and library versions; it does not run a hardware validation job.

Record, validate, report
========================

Run the complete artifact workflow on CPU:

.. code-block:: console

   $ python examples/train_single.py --device cpu --steps 20 --quiet --out-dir runs/cpu-check
   $ shardscope validate --run runs/cpu-check
   $ shardscope report --run runs/cpu-check --out reports/cpu-check.html

Open ``reports/cpu-check.html`` in a browser. It contains the recorded
summary, step rows, provenance, and availability of each report field.
HTML reports use inline styles and require no scripts or network access.

Use a fresh run directory for each recording. Fixture timings describe this
workload on this machine; they are not throughput benchmark results.

Commands
========

.. list-table:: Current CLI
   :header-rows: 1
   :widths: 25 55 20

   * - Command
     - Purpose
     - Status
   * - ``env``
     - Capture interpreter, PyTorch, accelerator, and device metadata.
     - Implemented
   * - ``validate --run DIR``
     - Check recorded artifacts and report inconsistencies.
     - Implemented
   * - ``report --run DIR``
     - Render a run as HTML, JSON, or CSV.
     - Implemented
   * - ``profile``
     - Capture a bounded training window with phase markers.
     - Not implemented
   * - ``bench``
     - Measure unprofiled throughput after warmup.
     - Not implemented
   * - ``compare``
     - Compare benchmark results against run-to-run variation.
     - Not implemented

Use ``shardscope --help`` or ``shardscope COMMAND --help`` for options.

.. code-block:: console

   $ shardscope env --format json
   $ shardscope validate --run runs/cpu-check --format json
   $ shardscope report --run runs/cpu-check --out reports/cpu-check.json
   $ shardscope report --run runs/cpu-check --format csv > reports/cpu-check.csv

Report format follows the output suffix: ``.html``, ``.htm``, ``.json``, or
``.csv``. ``--format`` overrides it. Without ``--out``, output goes to stdout
and defaults to HTML. ``--max-rows`` limits the HTML step table; JSON and CSV
retain every row.

Validation exits ``0`` when the run loads and its readable summary count
agrees with the CSV. Missing optional summaries, unknown profiling status,
and mixed measurement policies are reported without inventing replacements.
A recorded count mismatch fails validation; a report still shows both counts
and flags the disagreement.

``validate --format json`` emits ``ok``, ``errors``, ``warnings``, artifact
metadata, and field availability. Validation failures return a JSON document
with ``ok: false`` and exit code ``2``.

Run configuration
=================

The fixture accepts a JSON config and explicit CLI overrides:

.. code-block:: console

   $ python examples/train_single.py --config examples/configs/cpu_baseline.json --steps 20 --quiet

Supported keys and defaults:

.. code-block:: json

   {
     "steps": 100,
     "batch_size": 64,
     "samples": 2048,
     "seed": 0,
     "learning_rate": 0.01,
     "device": "auto",
     "data_residency": "host",
     "synchronize": true
   }

Explicit flags override file values. Keys starting with ``_`` are ignored;
other unknown keys and invalid values are rejected before training starts.
The config path and resolved settings are recorded in ``environment.json``.

``--device`` accepts ``cpu``, ``mps``, ``cuda``, or ``auto``. Automatic
selection prefers CUDA, then MPS, then CPU. Use ``--device cpu`` for local
development that should not depend on an accelerator.

``--data-residency host`` copies each batch to the selected device during
the step. ``device`` places the dataset there before training. Both modes
use the same seeded batch sequence.

Device synchronization is enabled by default. ``--no-sync`` removes the
explicit synchronization around each step; on accelerators, those timings
must not be interpreted as completed device execution time. Reports preserve
the synchronization and residency policies alongside the recorded values.

Artifact contract
=================

.. code-block:: text

   runs/<run-id>/
     steps.csv          required: per-step records
     environment.json   required: execution metadata and run settings
     summary.json       optional for readers: recorded statistics

The fixture writes all three files. Readers accept an absent summary and
preserve that absence. The current CSV schema is:

.. code-block:: text

   step,loss,step_time_ms,data_residency,synchronized

Step indices must increase without duplicates. Durations must be finite and
non-negative. Unsupported columns are rejected rather than dropped.
Provenance retains unrecognized keys so a round trip does not discard metadata.

Reports render recorded statistics without recomputing them. Field availability
uses three states: ``recorded``, ``not_recorded``, and ``not_observed``. Unknown
values remain unknown; missing activity is never substituted with zero.

A diverged run may record a non-finite loss. The JSON report preserves it using
Python's ``NaN``/``Infinity`` extensions, which strict JSON consumers may reject.
Validation JSON contains no floating-point measurements and uses strict JSON.

Output writes replace individual files through temporary siblings. A failed
write preserves an existing destination, and report output cannot overwrite
an existing input artifact, including a path alias to that artifact. The three
run files are written separately: a failure during recording can leave an
incomplete or mixed artifact set. Validation detects missing required files
and count mismatches, but cannot detect every mixture with matching counts.
Repeat failed recordings into a fresh directory.

``runs/``, ``reports/``, datasets, traces, and local planning files are ignored
by Git. Reports include paths and machine metadata; review them before sharing.

Development checks
==================

``tools/dev.py`` defines the checks used locally and by CI:

.. code-block:: console

   $ python tools/dev.py --list
   $ python tools/dev.py test
   $ python tools/dev.py lint
   $ python tools/dev.py format-check
   $ python tools/dev.py format
   $ python tools/dev.py cov
   $ python tools/dev.py ci

``format`` rewrites files; ``format-check`` only checks them. ``cov`` runs the
suite with branch coverage and an 80% minimum. ``ci`` runs lint, formatting,
README parsing, coverage, workflow bootstrap, ignore-policy checks, and the
installed CLI smoke test, in that order.

The smoke check verifies the installed CLI version, captures environment
metadata, runs the CPU fixture with and without a config, validates its
artifacts, and renders HTML. Integration tests exercise commands as separate
processes. Skipped tests identify unimplemented contracts or unavailable
prerequisites.

Changes to artifact formats should update producers, readers, renderers, and
regression tests together. Keep CLI errors actionable and preserve the
separation between missing data, unavailable activity, and measured zero.
Comments should explain behavior, invariants, or non-obvious constraints.

Local commits can use the guarded helper:

.. code-block:: console

   $ python tools/bootstrap_commit_skill.py
   $ python tools/commit_safe.py --authorize
   $ python tools/commit_safe.py --message "docs: update development workflow" --dry-run

Configure the generated, ignored authorization file with the repository-local
Git identity and explicit commit authorization before using the helper. It
runs tests and checks allowed paths before committing. GPU-path changes require
completed review. The helper does not push. Bootstrap exit code ``3`` indicates
an unavailable reference workflow; the local template is still generated.

Development direction
=====================

The artifact and reporting workflow is the current development baseline.
The next work depends on explicit measurement contracts:

1. Define capture scheduling, activity selection, phase markers, and per-rank
   trace layout, then implement bounded profiler capture.
2. Define profiled per-step fields and loader-wait semantics, then implement
   trace parsing and alignment by rank and step.
3. Define diagnostic thresholds and confidence criteria, then add evidence-linked
   rules and an injected loader-stall regression fixture.
4. Define benchmark warmup, repeats, and variance criteria before enabling
   ``bench`` and ``compare``.
5. Implement and validate DDP capture on a host with at least two CUDA devices.
   Synchronization, overlap, and straggler behavior require a defined strategy.

GPU setup and hardware validation remain pending. CPU checks do not establish
CUDA, NCCL, or distributed correctness. An available device inventory alone is
not evidence of a completed distributed run.

Profiled diagnostic timings must remain separate from unprofiled throughput
results. Kernel, copy, and collective durations can overlap and must not be
summed into a fraction of step time. Trace alignment must not assume precise
clock synchronization across hosts. Memory sweeps and autotuning are outside
the current scope.

Repository layout
=================

.. code-block:: text

   src/shardscope/   CLI, configuration, environment, artifacts, validation,
                    reporting; capture, trace parsing, and diagnostic stubs
   examples/        single-device fixture, CPU config, DDP stub
   benchmarks/      throughput and comparison stubs
   tests/           unit and subprocess integration tests
   tools/           development checks, commit tooling, local session launcher

Public documentation lives in ``README.rst``. Markdown working documents and
planning notes remain local.

Troubleshooting
===============

``shardscope`` is missing or reports the wrong version
  Activate ``.venv`` and reinstall with ``uv pip install -e ".[dev]"``.
  Run ``python tools/dev.py smoke`` to check the console script against the
  installed package. Use ``which -a shardscope`` to find competing installs.

A run fails validation
  Pass the individual run directory, not its parent. Check that both required
  files exist. For a recorded count mismatch or a failed recording, rerun the
  fixture into a fresh directory.

``profile``, ``bench``, or ``compare`` exits with a dependency message
  Expected in this version. See `Development direction`_ for the missing
  contracts and implementations.

Exit code ``141`` when piping a report to ``head``
  The reader closed stdout early. Use ``--out FILE`` to save the full report.
  CLI success returns ``0``; invalid input, inconsistent validation results,
  and unavailable commands return ``2``.

License
========

MIT.
