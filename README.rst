==========
ShardScope
==========

**Evidence-backed step profiling for PyTorch training loops.**

ShardScope records per-step timings, captures bounded ``torch.profiler``
traces with phase markers, attributes trace events and GPU kernels to
training steps, evaluates explicit diagnostic rules over the result, and
produces reports that keep recorded, missing, and unobserved data distinct.
Every number in a report points back to the trace event it came from.

Use it on your own training loop with ``shardscope.profile_loop`` (see
`Profiling your own training loop`_), or on the built-in fixture for a
reproducible baseline.

Features
========

- Reproducible single-device training fixture with JSON configuration.
- Bounded ``torch.profiler`` capture with ``shardscope/*`` phase markers and an
  explicit trace-event to training-step map.
- A loop integration API (``shardscope.profile_loop``) that records any
  caller-owned PyTorch training loop, whatever the model, optimizer, data
  loader, precision policy or DDP wrapper, into the same run directory format.
- Offline trace analysis that writes an ``analysis.json`` sidecar with
  per-step host phase timings, evidence references, and SHA-256 source digests.
- Kernel-to-step attribution through host CUDA launch correlation IDs, with
  per-step attributed kernel time and explicit coverage of what was not
  attributed.
- Threshold-based diagnostic rules over persisted host ranges (loader wait,
  host-to-device copy share, step-duration variability), with a rule registry
  and a findings view that exposes compared values and evidence.
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

Profile your own loop. Wrap each step and mark its phases; the rest of your
script is unchanged:

.. code-block:: python

   import shardscope

   capture = {"wait": 1, "warmup": 1, "active": 4, "repeat": 1,
              "activities": ["cpu", "cuda"]}
   with shardscope.profile_loop("runs/my-model", capture=capture, device="cuda") as prof:
       for step, (inputs, targets) in enumerate(loader):
           with prof.step(step) as rec:
               with rec.phase("h2d"):
                   inputs, targets = inputs.to("cuda"), targets.to("cuda")
               with rec.phase("forward"):
                   loss = loss_fn(model(inputs), targets)
               with rec.phase("backward"):
                   loss.backward()
               with rec.phase("optimizer"):
                   optimizer.step()
                   optimizer.zero_grad(set_to_none=True)
               rec.loss(loss)

Then analyze, inspect findings, and render a report:

.. code-block:: console

   $ shardscope analyze --run runs/my-model --threshold h2d_share_threshold=0.2
   $ shardscope findings --run runs/my-model
   $ shardscope report --run runs/my-model --out reports/my-model.html

No GPU at hand? Record the built-in CPU fixture instead:

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
   * - ``shardscope rules``
     - List the diagnostic rules ``analyze`` can evaluate, with their evidence
       requirements, thresholds and statuses.
   * - ``shardscope findings``
     - Summarize the findings a run's ``analysis.json`` already holds.
   * - ``examples/train_ddp.py``
     - Run the fixture under DDP with ``torchrun``.
   * - ``examples/profile_delayed.py``
     - Profile the fixture with an injected host loader delay.
   * - ``examples/train_custom_loop.py``
     - Profile a caller-owned loop over an MLP, CNN or Transformer through
       ``shardscope.profile_loop``; ``--ddp`` under ``torchrun``.

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

Profiling your own training loop
================================

``shardscope profile`` drives the built-in fixture. To profile a model and
training loop you own, keep the loop and mark its steps and phases with
``shardscope.profile_loop``:

.. code-block:: python

   import shardscope

   capture = {"wait": 1, "warmup": 1, "active": 4, "repeat": 1,
              "activities": ["cpu", "cuda"]}
   with shardscope.profile_loop(
       "runs/my-model", capture=capture, device="cuda",
       workload=shardscope.describe_module(model),
   ) as prof:
       batches = iter(loader)
       for step in range(steps):
           with prof.step(step) as rec:
               with rec.phase("data_wait"):
                   inputs, targets = next(batches)
               with rec.phase("h2d"):
                   inputs, targets = inputs.to("cuda"), targets.to("cuda")
               with rec.phase("forward"):
                   loss = loss_fn(model(inputs), targets)
               with rec.phase("backward"):
                   loss.backward()
               with rec.phase("optimizer"):
                   optimizer.step()
                   optimizer.zero_grad(set_to_none=True)
               rec.loss(loss)

The result is an ordinary run directory: ``shardscope validate``,
``analyze``, ``findings`` and ``report`` consume it unchanged, and
``environment.json`` records ``extra.source`` as ``shardscope.loop``.

``capture`` takes the same bounded schedule as the ``profile`` command. The
loop must wrap at least ``(wait + warmup + active) * repeat`` steps; a loop
that stops short raises ``ConfigError`` naming the shortfall and records
nothing. ``prof.step(n)`` takes the caller's own training-step index:
nonnegative integers, strictly increasing, not necessarily consecutive; a
step that is not wrapped is simply not recorded. Steps do not nest.

``rec.phase(name)`` opens one of ``data_wait``, ``h2d``, ``forward``,
``backward`` or ``optimizer`` as a ``shardscope/<name>`` host range inside
the step. A phase the loop does not enter is reported ``not_recorded`` by the
analysis, never as zero; entering a phase twice in one step (gradient
accumulation, for example) is allowed and the analysis unions the ranges. A
phase must be entered on the thread that entered the step; another thread's
range could not be attributed to the step, so it is refused rather than
dropped. ``rec.loss(value)`` accepts a number or a one-element tensor, which
is read after the step timer stops because ``.item()`` synchronizes.

Keyword arguments:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Argument
     - Meaning
   * - ``device``
     - Where the per-step synchronization happens (``cuda``, ``cuda:N``,
       ``cpu`` or ``auto``); it moves nothing. A CUDA device requires the
       ``cuda`` profiler activity. MPS capture is unsupported.
   * - ``synchronize``
     - Drain the device before the step timer starts and stops (default
       ``true``). The choice is recorded per row in ``steps.csv``.
   * - ``data_residency``
     - A declaration about the loop: ``"device"`` tells the analysis there is
       no per-step host-to-device batch copy to look for. Default ``"host"``.
   * - ``rank``, ``local_rank``, ``world_size``
     - Rank provenance for ``torchrun`` launches; give each rank its own
       output directory (``shardscope.distributed.rank_directory``).
   * - ``workload``
     - Free JSON provenance kept under ``extra.workload``.
       ``shardscope.describe_module(model)`` supplies exact parameter counts,
       dtypes and devices for any ``nn.Module``.

An exception inside the ``with`` block, an incomplete schedule or a misuse
of the API unwinds the staging directory and leaves the destination absent.
``examples/train_custom_loop.py`` is a complete integration over three
unrelated architectures with a ``DataLoader``, optional ``bfloat16``
autocast and optional ``DistributedDataParallel``:

.. code-block:: console

   $ python examples/train_custom_loop.py --model transformer --device cuda \
       --steps 12 --active 4 --repeat 2 --out runs/transformer
   $ shardscope analyze --run runs/transformer --threshold h2d_share_threshold=0.2
   $ shardscope findings --run runs/transformer
   $ torchrun --standalone --nproc-per-node=2 examples/train_custom_loop.py \
       --ddp --model cnn --device cuda --steps 8 --out runs/cnn-ddp

Trace analysis
==============

Analysis reads a profiled run and writes ``analysis.json`` next to its
artifacts. By default it uses the versioned step mapping recorded in
``environment.json.extra.capture`` and verifies that each entry matches the
run's rank and an actual trace boundary. Run it through the installed command:

.. code-block:: console

   $ shardscope analyze --run runs/delayed --loader-wait-threshold-ms 5

The threshold above is an explicit example, not a calibrated recommendation.
``python -m shardscope.analysis`` supports the same arguments. A supplied
``--step-map`` overrides the embedded mapping. Its file must be inside the run
directory and use this consumer format:

.. code-block:: json

   {"mappings": [{"trace_file": "trace-00000.json", "event_index": 7, "training_step": 2}]}

Missing mappings remain unavailable. Unsupported or malformed mappings fail
analysis; an invalid explicit file never silently falls back to capture metadata.

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
  Enables the ``host_loader_wait`` rule; the same as
  ``--threshold loader_wait_threshold_ms=MS``.

``--threshold KEY=VALUE``
  Configures one rule threshold; repeat for several. Keys, units and ranges
  are listed by ``shardscope rules``. A rule with no threshold reports
  ``unconfigured``. Giving one key twice with different values is an error.

``--overwrite``
  Replace an existing ``analysis.json``.

Each per-step measurement has ``value_ms``, ``status``, ``basis``, ``scope``,
``reason``, and ``evidence`` (trace filename and event index). Host phase
timings are interval unions of complete host ranges (``basis: host_range``).
The complete-activity device metrics (``device_h2d_ms``,
``device_compute_ms``, ``device_collective_ms``, ``device_busy_ms``) are
reported as unavailable. One device figure is measured, under its own name
and contract, as described next.

Kernel attribution
------------------

A CUDA capture records a host ``cuda_runtime`` range for each CUDA API call
and a device ``kernel`` range for each kernel execution; both carry
``args.correlation``. Analysis ties a kernel to a training step only through
that ID, never through GPU timestamp containment in a host step:

1. Correlation IDs are looked up per trace. An ID that appears on several
   host CUDA API events in one trace is a conflict, and nothing is chosen.
2. The host event must be a supported kernel-launch form:
   ``cudaLaunchKernel`` or ``cudaLaunchKernelExC`` in category
   ``cuda_runtime``. Driver-API launches and copy or memset calls are reported
   as unsupported, not guessed at.
3. The launch range must lie entirely inside exactly one mapped
   ``shardscope/step`` boundary on the same process and thread. A launch on a
   host thread that carries no mapped boundary, such as the autograd engine
   thread that launches backward kernels, is reported as outside every mapped
   step. In the recorded A100 captures this leaves roughly 40% of kernel
   events attributed per trace; the remainder are disclosed, not estimated.

Several kernels may reference one unambiguous launch. A kernel that executes
after its step's host boundary has ended is still attributed to the step that
launched it. ``gpu_user_annotation``, ``gpu_memcpy``, and ``gpu_memset``
events are not kernels and are never counted.

The result is persisted under ``kernel_attribution``: per trace, the launch
lookup (correlation ID, training step, launch event index, boundary event
index), every attributed kernel with its launch and device, every unmatched or
ambiguous kernel with a reason, excluded launches by reason, and per-step,
per-device interval unions. Each mapped step also gains the measurement
``device_attributed_kernel_ms`` (``basis: device_interval_union``): the
interval union of the kernel ranges attributed to that step on one device. It
is *attributed kernel time*. It is not utilization, not total step latency,
and not total device busy time. Overlapping kernels are unioned, not summed;
kernels on different devices are kept apart, and a step whose attributed
kernels span several devices withholds the single figure. A step with no
attributed kernel is ``not_recorded``, never zero. The measurement's
``coverage`` block and the report state how many kernel events in the trace
were attributed, so a partial figure reads as a lower bound on attributed
kernel execution rather than a complete compute measurement.

``validate`` and ``report`` re-derive the classification from the trace bytes
and refuse a sidecar whose attributed, unmatched, or ambiguous records, launch
lookup, or per-step kernel groups disagree with it; they never re-time a
kernel. Sidecars written before this feature load unchanged and simply have no
attributed kernel time.

A CUDA capture also contains device-side copies of the markers: Kineto emits
them with category ``gpu_user_annotation`` on a GPU stream. Analysis
classifies them by that category alone, keeps them raw, and never measures
them as host phases or attributes them to a step. A mapping that names one,
or that disagrees with the ``shardscope/training_step/<N>`` identity marker
nested in its boundary, fails analysis. Metadata, flow, and instant events are
retained unmeasured; ``warnings`` summarizes each such class once per trace
with a count and representative original event indices rather than one line
per event.

Diagnostic rules
----------------

Rules compare persisted measurements against thresholds the caller supplies.
No threshold is defaulted or calibrated, and no rule emits a confidence score,
a root cause or an expected speedup. A missing measurement is never zero, and
a ``below_threshold`` status does not establish that the workload is healthy.
``shardscope rules`` prints the registry; ``--format json`` gives the same
data for scripts.

.. list-table::
   :header-rows: 1
   :widths: 28 30 42

   * - Rule
     - Requires
     - Comparison
   * - ``host_loader_wait``
     - ``host_data_wait_ms``
     - ``host_data_wait_ms >= loader_wait_threshold_ms`` (ms, >= 0), per
       step. Triggered status ``possible_input_wait``. A host wait alone does
       not establish GPU starvation.
   * - ``host_h2d_share``
     - ``host_h2d_ms``, ``host_step_ms``
     - ``host_h2d_ms / host_step_ms >= h2d_share_threshold`` (fraction in
       [0, 1]), per step; both are host interval unions from the same thread
       and boundary. Triggered status ``possible_host_copy_wait``. Host time
       inside copy calls is not device transfer time or bandwidth.
   * - ``step_duration_variability``
     - ``host_step_ms``
     - ``max(host_step_ms) / median(host_step_ms) >= step_variability_ratio``
       (ratio >= 1) over the recorded steps of one rank; one finding per
       analysis. ``step_variability_min_steps`` (integer >= 2, optional)
       raises the required step count. Steps that recorded different host
       phase sets are reported ``ambiguous`` instead of compared. Triggered
       status ``possible_step_variability``; it does not compare ranks.

Every rule reports ``unconfigured`` without a threshold and ``unavailable``
when a required measurement is not recorded. ``ambiguous`` means the inputs
are recorded but the comparison is undefined or inconsistent;
``insufficient`` means too few steps for a statistic. Each finding persists
the compared values, the operator, the threshold and the evidence references
(trace filename and original event index). ``analysis.json`` records the
evaluated rule ids under ``rules``; sidecars written before that key existed
are validated against the single ``host_loader_wait`` rule and keep loading.

.. code-block:: console

   $ shardscope analyze --run runs/delayed --threshold loader_wait_threshold_ms=5 \
       --threshold h2d_share_threshold=0.3 --threshold step_variability_ratio=1.5
   $ shardscope findings --run runs/delayed
   $ shardscope findings --run runs/delayed --format json --rule host_h2d_share \
       --status possible_host_copy_wait

``findings`` reads the persisted sidecar (validated against its sources, so a
stale one is refused), evaluates nothing, and exits ``0`` whatever the
findings say; ``2`` means the run or its sidecar would not load. ``--rule``
and ``--status`` filter what is listed while the per-status counts keep
describing every finding.

Candidates not implemented, and the evidence each needs: host synchronization
overhead (a persisted per-step union of host ``cudaDeviceSynchronize`` /
``cudaStreamSynchronize`` ranges inside the step boundary), transfer/compute
and collective/compute overlap and communication tail (validated
``device_h2d_ms``, ``device_collective_ms`` and compute activity on one
device clock), rank duration imbalance (per-rank sidecars aligned by training
step, with equal logical work established), and many short kernels (persisted
per-kernel durations with complete attribution coverage and a host launch-side
measurement). None of these is reported from partial evidence.

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
``environment.json`` ``extra``. ``extra.source`` names the producer:
``shardscope.capture`` for the fixture, ``shardscope.ddp`` for unprofiled DDP
runs and ``shardscope.loop`` for caller-owned loops, which also record
``extra.workload`` and ``extra.capture.phases_entered``.

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
- ``device_attributed_kernel_ms`` covers only kernels whose launch was
  resolved to a mapped step on the launching thread. It is a per-device
  interval union of those kernels, not GPU utilization or device busy time.
- Host phase timings from a profiled run include profiler overhead.
- Cross-host trace alignment uses step identity and assumes no clock
  synchronization.

Known limitations
=================

- Device-side copy, collective, and total busy-time attribution is not
  reported. Kernel attribution covers only launches made from a thread that
  carries a mapped step boundary; kernels launched from the autograd engine
  thread stay unattributed and are listed as such. Raw device events are
  preserved in the trace files.
- Analysis measures host ranges and attributed kernel ranges only. An
  embedded or explicit step mapping is required to attribute either to
  training steps.
- Capture on MPS is not supported.
- With the PyTorch build this was developed against (2.15), a profiler
  session that requests only the ``cpu`` activity disables CUDA kernel
  collection for every later session in the same process. Make the first
  capture in a process a CUDA capture, or run captures in separate processes;
  the analysis reports the missing kernels as unattributed rather than zero.
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

Contributing
============

Contributions are welcome. The short version:

1. Create a branch and make your change with tests. New rules go in
   ``src/shardscope/rules.py`` with cases in ``tests/test_rules.py``; new
   measurements go in ``src/shardscope/parse.py`` with their producer,
   readers and renderers updated together.
2. Run ``python tools/dev.py ci`` before opening a pull request. It is the
   same gate CI runs: lint, format, README parsing, tests with coverage,
   ignore policy and an installed-CLI smoke test.
3. Keep commit messages in the form ``<type>: <description>`` with one of
   ``feat``, ``fix``, ``refactor``, ``docs``, ``test``, ``chore``, ``perf``,
   ``ci``.
4. Report GPU acceptance separately from CPU tests: say which tests ran on
   hardware and which were skipped.

Design rules that reviews hold to: missing measurements are never reported
as zero, overlapping device activity is never summed into a share of step
time, thresholds are caller configuration rather than built-in defaults, and
every reported value cites the trace event it came from. Markdown files are
not tracked in this repository; documentation lives in this README and in
docstrings.

Repository layout
=================

.. code-block:: text

   src/shardscope/   CLI, configuration, capture, loop integration, execution,
                     trace parsing, analysis, rules, benchmarking, DDP,
                     validation, reporting
   examples/         single-device, delayed-loader, DDP and custom-loop
                     examples; configs
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
========

MIT.
