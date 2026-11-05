==========
ShardScope
==========

ShardScope is a profiling and benchmarking toolkit for PyTorch training. It
records per-step timings, captures bounded profiler traces with phase markers,
attributes trace events to training steps, and produces reports that keep
recorded, missing, and unobserved data distinct.

Features
========

- Reproducible single-device training fixture with JSON configuration.
- Bounded ``torch.profiler`` capture with ``shardscope/*`` phase markers and an
  explicit trace-event to training-step map.
- Offline trace analysis that writes an ``analysis.json`` sidecar with
  per-step host phase timings, evidence references, and SHA-256 source digests.
- Threshold-based host loader-wait diagnostics.
- Unprofiled throughput benchmarks with completion-synchronized windows and
  raw per-repeat observations.
- Descriptive benchmark comparison.
- PyTorch DDP execution over NCCL with independent per-rank artifacts.
- Artifact validation and HTML, JSON, and CSV reports.

Requirements
============

- Linux or macOS
- Python 3.10, 3.11, or 3.12
- PyTorch 2.2 or later
- NumPy 1.26 or later
- For CUDA capture: a CUDA build of PyTorch and a visible CUDA device
- For DDP: two or more CUDA devices, NCCL, and ``torchrun``

Installation
============

From a checkout:

.. code-block:: console

   $ uv venv --python 3.12 .venv
   $ source .venv/bin/activate
   $ uv pip install -e ".[dev]"
   $ shardscope --version

With pip:

.. code-block:: console

   $ python3.12 -m venv .venv
   $ source .venv/bin/activate
   $ python -m pip install -e ".[dev]"

The ``dev`` extra installs pytest, coverage, Ruff, and docutils. The
``charts`` extra installs matplotlib. On Apple silicon, use an ARM64
interpreter.

Quick start
===========

Record a CPU run, validate it, and render a report:

.. code-block:: console

   $ python examples/train_single.py --device cpu --steps 20 --quiet --out-dir runs/cpu-check
   $ shardscope validate --run runs/cpu-check
   $ shardscope report --run runs/cpu-check --out reports/cpu-check.html

The HTML report is self-contained: inline styles, no scripts, no network
access.

Commands
========

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - ``shardscope env``
     - Print interpreter, PyTorch, CUDA, cuDNN, NCCL, and device metadata.
   * - ``shardscope validate``
     - Check a run directory and report inconsistencies. Writes nothing.
   * - ``shardscope report``
     - Render a run as HTML, JSON, or CSV.
   * - ``shardscope profile``
     - Run the fixture under a bounded profiler schedule.
   * - ``shardscope bench``
     - Measure unprofiled throughput.
   * - ``shardscope compare``
     - Compare two benchmark results.
   * - ``python -m shardscope.analysis``
     - Analyze a profiled run's traces and write ``analysis.json``.
   * - ``examples/train_ddp.py``
     - Run the fixture under DDP with ``torchrun``.
   * - ``examples/profile_delayed.py``
     - Profile the fixture with an injected host loader delay.

Use ``--help`` on any command for its full option list. All commands exit
``0`` on success and ``2`` on invalid input or a failed check.

Run configuration
=================

The fixture and the ``profile`` and ``bench`` commands read a JSON config.
Supported keys and their defaults:

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

Keys starting with ``_`` are treated as comments. Unknown keys and invalid
values are rejected before training starts. ``seed`` must be in
``[-2**63, 2**64 - 2]``.

``device``
  ``cpu``, ``mps``, ``cuda``, or ``auto``. ``auto`` selects CUDA, then MPS,
  then CPU.

``data_residency``
  ``host`` copies each batch to the device inside the step. ``device`` places
  the dataset on the device before training. Both use the same seeded batch
  sequence.

``synchronize``
  Synchronize the device around each step. With ``false`` (``--no-sync`` on
  the fixture), accelerator step times do not represent completed device
  work.

With ``examples/train_single.py``, command-line flags override config values:

.. code-block:: console

   $ python examples/train_single.py --config examples/configs/cpu_baseline.json --steps 20 --quiet

Profiling
=========

``shardscope profile`` requires a ``capture`` block in the config:

.. code-block:: json

   {
     "steps": 12,
     "device": "cpu",
     "capture": {
       "wait": 1,
       "warmup": 1,
       "active": 2,
       "repeat": 2,
       "activities": ["cpu"]
     }
   }

.. code-block:: console

   $ shardscope profile --config capture.json --out runs/profiled

``wait``, ``warmup``, ``active``, and ``repeat`` map directly to
``torch.profiler.schedule``. ``steps`` must cover
``(wait + warmup + active) * repeat``. ``activities`` must include ``cpu``;
add ``cuda`` when training on a CUDA device. The optional ``record_shapes``,
``profile_memory``, ``with_stack``, and ``with_flops`` flags default to
``false``. MPS capture is not supported.

Each capture cycle writes one Chrome trace (``trace-00000.json``,
``trace-00001.json``, ...). Every training step is wrapped in
``record_function`` ranges:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Marker
     - Range
   * - ``shardscope/step``
     - The full training step.
   * - ``shardscope/training_step/<N>``
     - Identity marker for training step ``N``.
   * - ``shardscope/data_wait``
     - Host time blocked on the next batch.
   * - ``shardscope/h2d``
     - Host time in host-to-device copy calls.
   * - ``shardscope/forward``
     - Forward pass.
   * - ``shardscope/backward``
     - Backward pass.
   * - ``shardscope/optimizer``
     - Optimizer step.

``environment.json`` records the resolved capture settings under
``extra.capture``, including ``step_map``, which links each traced training
step to the source-array index of its ``shardscope/step`` event. Profiler
step names (``ProfilerStep#N``) are kept as evidence and are not used as
training-step indices.

``profile`` refuses a non-empty output directory. Artifacts are written to a
staging directory and published when complete; a failed capture leaves no
output.

The baseline fixture indexes tensors directly and has no loader, so
``data_wait`` is not recorded. To exercise it, use the delayed-iterator
fixture:

.. code-block:: console

   $ python examples/profile_delayed.py --config capture.json --out runs/delayed --delay-ms 10

Trace analysis
==============

Analysis reads a profiled run and writes ``analysis.json`` next to its
artifacts. Steps are attributed through an explicit step-map file in the run
directory:

.. code-block:: json

   {"mappings": [{"trace_file": "trace-00000.json", "event_index": 7, "training_step": 2}]}

Generate it from the map that ``profile`` recorded, then run the analysis:

.. code-block:: console

   $ jq '{mappings: [.extra.capture.step_map[]
          | {trace_file, event_index: .step_event_index, training_step}]}' \
       runs/delayed/environment.json > runs/delayed/step-map.json
   $ python -m shardscope.analysis --run runs/delayed --step-map step-map.json \
       --loader-wait-threshold-ms 5

Options:

``--step-map FILE``
  Step-map filename inside the run directory. Without it, no step is
  attributed and every measurement is ``not_recorded``.

``--trace FILE``
  Trace filename to analyze; repeat for multiple cycles. Defaults to every
  ``trace*.json`` and ``trace*.json.gz`` in the run.

``--rank N``
  Rank to record. Defaults to ``extra.rank`` from ``environment.json``.

``--loader-wait-threshold-ms MS``
  Enables the ``host_loader_wait`` rule. Without it, the rule reports
  ``unconfigured``.

``--overwrite``
  Replace an existing ``analysis.json``.

Each per-step measurement has ``value_ms``, ``status``, ``basis``, ``scope``,
``reason``, and ``evidence`` (trace filename and event index). Host phase
timings are interval unions of complete host ranges (``basis: host_range``).
Device metrics are reported as unavailable.

The ``host_loader_wait`` rule marks a step ``possible_input_wait`` when
``host_data_wait_ms`` meets the threshold, ``below_threshold`` when it does
not, and ``unavailable`` when the step has no recorded wait. A host wait
alone does not establish GPU starvation.

``validate`` and ``report`` load ``analysis.json`` when present. They check
its schema, rank, step references, and source digests; a sidecar whose
sources have changed is rejected as stale.

Benchmarking
============

``bench`` measures throughput without a profiler. The config must not contain
a ``capture`` block.

.. code-block:: console

   $ shardscope bench --config bench.json --out results/baseline \
       --warmup 10 --repeat 5 --measured-steps 50
   $ shardscope bench --config bench.json --out results/candidate \
       --warmup 10 --repeat 5 --measured-steps 50
   $ shardscope compare --baseline results/baseline --candidate results/candidate

Each repeat builds a fresh model, dataset, and optimizer with the same seed,
runs ``--warmup`` steps, then times ``--measured-steps`` steps between two
device synchronizations. Per-step synchronization and loss extraction are
disabled inside the window. ``benchmark.json`` stores every repeat's elapsed
time, completed steps and examples, ``steps/s``, and ``examples/s``, along
with the workload, timing policy, and environment.

``compare`` requires matching workloads and timing policies. It prints JSON
with the median, minimum, and maximum of each metric for both inputs and the
percent difference between medians. It does not perform a significance test.

Timings from ``profile`` include profiler overhead. Use ``bench`` for
throughput figures.

Distributed training
====================

``examples/train_ddp.py`` wraps the fixture in ``DistributedDataParallel``
with the NCCL backend. Launch it with ``torchrun``:

.. code-block:: console

   $ torchrun --standalone --nproc-per-node=2 examples/train_ddp.py \
       --config ddp.json --out runs/ddp [--profile]

The config must set ``"device": "cuda"``. Each process needs its own visible
CUDA device. ``--profile`` requires a ``capture`` block. All ranks must use
the same config and output path, and the output directory must be new or
empty.

Samples are partitioned across ranks with a strided split. Each rank writes
an independently loadable directory:

.. code-block:: text

   runs/ddp/
     rank-00000/
     rank-00001/

Rank directories are published separately; a failure on one rank can leave
others complete. Validate each rank directory individually. Analysis aligns
ranks by training-step index, not wall-clock time.

Validation and reports
======================

.. code-block:: console

   $ shardscope validate --run runs/cpu-check --format json
   $ shardscope report --run runs/cpu-check --out reports/cpu-check.json
   $ shardscope report --run runs/cpu-check --format csv > reports/cpu-check.csv

``validate`` exits ``0`` when the run loads and the step count in
``summary.json`` matches ``steps.csv``. ``--format json`` always writes a
document with ``ok``, ``errors``, ``warnings``, artifact metadata, and field
availability; on failure it contains ``ok: false`` and exits ``2``.

``report`` infers the format from the ``--out`` suffix (``.html``, ``.htm``,
``.json``, ``.csv``); ``--format`` overrides it. Without ``--out``, it writes
HTML to stdout. ``--max-rows`` limits the HTML step table; JSON and CSV
include every row. Reports render recorded values and do not recompute
statistics.

Run directory format
====================

.. code-block:: text

   runs/<run-id>/
     steps.csv           per-step records (required)
     environment.json    environment, run settings, provenance (required)
     summary.json        recorded statistics (optional)
     trace-NNNNN.json    profiler traces (profiled runs)
     analysis.json       analysis sidecar (optional)

``steps.csv`` columns:

.. code-block:: text

   step,loss,step_time_ms,data_residency,synchronized

Step indices must be strictly increasing. Durations must be finite and
non-negative. Unknown columns and a leading UTF-8 byte-order mark are
rejected. Each artifact must be a regular file. Unknown keys in
``environment.json`` are preserved.

Profiled runs record ``artifact_schema``, ``profiled``, ``rank``,
``local_rank``, ``world_size``, ``hostname``, ``pid``, and ``capture`` under
``environment.json`` ``extra``.

Report fields have one of three availability states: ``recorded``,
``not_recorded``, or ``not_observed``. Missing values are never reported as
zero. A diverged run may record a non-finite loss; the JSON report encodes it
as ``NaN`` or ``Infinity``, which strict JSON parsers may reject.

Report and sidecar writes go through a temporary sibling file and an atomic
rename, and never overwrite an input artifact. The fixture writes its three
files separately, so an interrupted recording can leave a partial set; record
into a fresh directory each time.

Reports include file paths and machine metadata. Review them before sharing.

Measurement notes
=================

- ``data_wait`` is host blocking time in the batch iterator, not device idle
  time.
- ``h2d`` is host time in copy calls, not device transfer duration.
- Kernel, copy, and collective durations can overlap; they are not summed
  into a fraction of step time.
- Host phase timings from a profiled run include profiler overhead.
- Cross-host trace alignment uses step identity and assumes no clock
  synchronization.

Known limitations
=================

- Device-side kernel, copy, and collective attribution is not reported.
  Raw device events are preserved in the trace files.
- Analysis requires an explicit step-map file and runs as
  ``python -m shardscope.analysis`` rather than a ``shardscope`` subcommand.
- Capture on MPS is not supported.
- ``examples/train_single.py --device cuda`` on a host without CUDA fails
  with a PyTorch error rather than a configuration error.
- ``compare`` reports descriptive statistics only.

Development
===========

``tools/dev.py`` runs the same checks locally and in CI:

.. code-block:: console

   $ python tools/dev.py --list
   $ python tools/dev.py test
   $ python tools/dev.py lint
   $ python tools/dev.py format
   $ python tools/dev.py cov
   $ python tools/dev.py ci

``ci`` runs lint, format check, README parsing, tests with branch coverage
(80% minimum), workflow bootstrap, ignore-policy checks, and an installed-CLI
smoke test. CI runs on Linux with Python 3.10 and 3.12.

Tests marked ``gpu`` require a CUDA device; tests marked ``distributed``
require two or more CUDA devices and ``torchrun``. They are skipped when the
hardware is unavailable.

Changes to an artifact format should update its producer, readers,
renderers, and tests together.

Repository layout
=================

.. code-block:: text

   src/shardscope/   CLI, configuration, capture, execution, trace parsing,
                     analysis, rules, benchmarking, DDP, validation, reporting
   examples/         single-device, delayed-loader, and DDP fixtures; configs
   benchmarks/       wrappers for the bench and compare commands
   tests/            unit and subprocess integration tests
   tools/            development checks and repository tooling

Troubleshooting
===============

``shardscope`` is missing or reports the wrong version
  Activate ``.venv`` and reinstall with ``uv pip install -e ".[dev]"``. Run
  ``python tools/dev.py smoke`` to check the console script against the
  installed package. ``which -a shardscope`` lists competing installs.

A run fails validation
  Pass the run directory itself, not its parent. Confirm ``steps.csv`` and
  ``environment.json`` exist. For a count mismatch or interrupted recording,
  record again into a fresh directory.

``profile`` refuses the output directory
  Profiled runs require a new or empty ``--out`` directory.

``analysis.json`` is reported as stale
  A source artifact changed after analysis. Re-run the analysis with
  ``--overwrite``.

Exit code ``141`` when piping to ``head``
  The reader closed the pipe early. Use ``--out FILE`` to save the full
  output.

License
=======

MIT.
