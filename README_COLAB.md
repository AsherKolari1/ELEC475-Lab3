# Lab 3: local WSL first, optional Colab A100

## Daily Workflow

```bash
cd ~/ELEC475/"Lab 3"
source .venv/bin/activate
code .
```

Then ask Codex to implement, test, benchmark or run an experiment. For example:
“Check my pre-lab,” “Benchmark data loading locally,” or “Implement this TODO and test it.”
You do not need to run synchronization or setup commands during normal use.

Codex handles hardware selection, session checks, source synchronization, compatible
remote dependencies, available dataset/checkpoint preparation, execution, controlled
DataLoader tuning, logs, artifact/checkpoint retrieval and result verification as
needed for your request. This is task-driven assistance, not a background service.

**Local hardware is preferred.** Codex asks before allocating/waking a Colab GPU or
running expensive experiments, explains remote benefits and transfer costs, and does
not stop shared Colab work without approval. An appropriate already-running session
can be reused within an authorized task. Missing assets or interactive authentication
are reported explicitly; you are only asked for the missing access/approval.

**This pre-lab has no training step.** Codex verifies the existing models and supplied
weights, preserves instructor TODOs and experiment logging, and reports missing
images/checkpoints or partial checks honestly. Full verification requires all three
course checkpoint files; successful architecture checks alone do not prove it passed.
The previously used `lab3` session was stopped with approval; a new allocation must
be approved before remote execution.

## Advanced reference and troubleshooting

The commands and infrastructure details below are for Codex and optional diagnosis.
Codex performs applicable routine steps itself; this is not a daily checklist. The
existing toolkit, backend overrides, logging and recovery functionality are unchanged.

### Verified assignment and entry points

The local pre-lab PDF says **do not train**: implement the existing models and
verify supplied weights. There is no `train.py`, optimizer, or training loop in
this repository. `colab_config.json` now records the real `experiment_command`:

```bash
python -m models.snoutnet       # architectures, strict weight loading, known prediction
python -m utils.check_dataset  # validate both splits and plot eight nose labels
```

The manager's `verify` task runs that configured command through a logging
wrapper. `scripts/experiment.py` adds bounded inference benchmarks, dataset-layout
checks, and environment inspection; it does not implement a training architecture.
All instructor TODO comments remain in place.

### Permanent adaptive execution policy

1. Default to local WSL for editing, syntax/import checks, unit tests, preprocessing,
   small experiments and debugging. Use the local RTX 3050 for useful short CUDA tests.
2. Use the A100 for demonstrated expensive workloads: long training (once assigned),
   large models/data, sweeps, or local VRAM/time constraints. Availability alone is
   not a reason to use it.
3. `execution_backend: auto|local|colab` in the config sets the default;
   `run --backend ...` overrides it. Auto probes local CUDA and free VRAM and uses
   conservative task/model/batch/step estimates, with optional explicit
   `--estimated-minutes` and `--required-vram-gb`. These are routing hints, not measured
   runtime guarantees. Unknown work stays local. No automatic migration or remote retry.
4. Auto recommendations for Colab require `--approve-remote`. Costly jobs on either
   backend also require user approval, `--approve-expensive` and a finite
   `--budget-seconds`. Agents must ask before supplying approval flags. A forced
   `--backend colab` is an explicit remote choice, not permission for expensive work.
5. Explain remote benefit and source/dependency/data transfer plus artifact-download
   overhead before costly work. Never create, restart, allocate or wake a session
   automatically. This toolkit only reuses an already allocated IDLE session and
   rejects missing/BUSY/unknown states. Allocation requires user approval; Codex handles supported infrastructure steps. Drive OAuth may require interactive user authorization.
6. Never stop sessions with unfinished or unsaved work. Fetch first, inspect status,
   then stop only with user authorization. Report idle paid sessions; don't silently
   leave them running or stop them. See recovery below.
7. Keep environment-specific dependencies separate. Preserve working CUDA torch and
   torchvision. Measure performance before changing worker, batch, pinning or
   persistence defaults; respect CPU/RAM limits. Record backend/reason, hardware,
   effective settings, logs, metrics and artifacts for each managed experiment.

The same policy is permanent in `AGENTS.md`.

### Current inspection (2026-10-19)

- Local: RTX 3050 Laptop GPU, 4 GiB; PyTorch `2.14.1+cu130`, CUDA works outside
  restricted agent sandboxes. The sandbox may report CUDA unavailable; that alone
  must not trigger remote migration. WSL has 16 available CPUs and about 8 GiB RAM;
  available RAM varies.
- Historical `lab3` inspection: A100-SXM4-40GB, 12 CPUs; Python 3.13.15,
  torch `2.11.0+cu130`, torchvision `0.26.0+cu130`; numpy, Pillow, matplotlib,
  torchinfo and tqdm present. After artifacts were retrieved and final source was synced, this session was stopped with user approval. This is historical hardware evidence, not a live allocation; a new allocation requires fresh approval.
- Dataset root is **`data/oxford-iiit-pet-noses`**, containing
  `images-original/images/*.jpg`, `train_noses.txt`, and `test_noses.txt`.
  Local inspection found 7,390 JPEGs, 5,494 train labels, 698 test labels,
  zero missing referenced images, and `(3, 227, 227)` sample tensors.
- At inspection the remote dataset and both local/remote supplied checkpoints were
  absent. The full verification task needs `checkpoints/snoutnet.pth`, `alexnet.pth`,
  `vgg16.pth`. Codex must recheck availability, use authorized accessible copies when available, and clearly report missing files/access. Course login may require user interaction; do not claim verification passed without the weights.
  The PDF's sample-0 target is ground truth `(311, 152)` and prediction
  `(350.3443, 181.7347)` within one pixel.

### Local commands (advanced troubleshooting)

Codex runs these as needed. They are available for diagnosis, not required daily steps. From the project root, activate `.venv` (or invoke its Python explicitly):

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
python scripts/colab_manager.py run --dry-run
python scripts/colab_manager.py run --task layout
python scripts/colab_manager.py run --task environment
# After placing all three supplied checkpoint files:
python scripts/colab_manager.py run --task verify
python scripts/colab_manager.py run --task dataset-check --workers 0 --batch-size 64
```

`--dry-run` probes hardware and prints the policy without synchronizing or allocating
remote resources. Local runs never install dependencies automatically. Existing local
`requirements.txt` is unchanged. Do not reinstall PyTorch just to use the manager.
`verify` retains the instructor's CPU verification behavior; the benchmark supports CUDA.

`--workers`, `--batch-size`, and `--output-root` configure run settings. Environment
variables `ELEC475_NUM_WORKERS`, `ELEC475_BATCH_SIZE`, `ELEC475_OUTPUT_DIR`,
`ELEC475_DATASET_ROOT` also support the dataset checker; pinning/persistence there
use `ELEC475_PIN_MEMORY=1` and `ELEC475_PERSISTENT_WORKERS=1`. The model architecture
test always uses batch 1 as required by the assignment, independently of loader settings.

### Controlled DataLoader benchmarks

The manager defaults to **0 workers and batch size 4**, not the old speculative 4/32.
The benchmark uses the existing SnoutNet and real dataset/resize/normalization, fixed
seed 475, fixed sample order, no shuffle/augmentation/dropout, deterministic operations,
and no optimizer or training. Random weights are explicitly recorded unless
`--checkpoint checkpoints/snoutnet.pth` is supplied. Nothing is downloaded implicitly.

A lightweight initial comparison (0, 2, 4, 8 workers):

```bash
python scripts/colab_manager.py run --backend local --task benchmark --batch-size 2 -- \
  --steps 2 --warmup 1 --device cuda
```

For useful tuning, repeat short trials with more measured batches once authorized;
keep model, seed, batch, steps, transforms and device identical. Forwarded options
follow `--`. Compare one factor at a time:

```bash
# Same baseline, pinned transfers:
python scripts/colab_manager.py run --backend local --task benchmark --batch-size 2 -- \
  --steps 2 --warmup 1 --device cuda --pin-memory
# Test a different batch size separately, without changing assignment defaults:
python scripts/colab_manager.py run --backend local --task benchmark --batch-size 4 -- \
  --steps 2 --warmup 1 --device cuda
```

`--workers-grid 0 2 4 8`, `--model snoutnet|alexnet|vgg16`, `--cpu-threads`,
`--prefetch-factor`, `--pin-memory/--no-pin-memory` and
`--persistent-workers/--no-persistent-workers` are configurable. Workers=0 always
uses no prefetch and no persistence. Pinning is effective only on CUDA. Worker
counts above available CPUs are skipped unless `--allow-oversubscription`; RAM
checks include a configurable `--worker-ram-mb` allowance (384 MiB by default) and
queued tensors, skipping settings estimated to exceed 60% of available RAM.
These are conservative estimates, not a full process memory limit.

`metrics.json` and `benchmark.csv` report:

- time blocked in `next(loader)` (data wait), H2D transfer separately, synchronized
  forward compute time, measured samples/sec, iterator startup time;
- peak allocated/reserved GPU memory, host/terminated-child peak RSS (Linux KiB),
  effective loader settings, fixed prediction checksum, errors and resource skips.

Warmup is excluded. CUDA is synchronized at timing boundaries; throughput includes
loading + transfer + forward time, excluding iterator startup, warmup, saving and
metric serialization. Prefetch can overlap CPU work, so data wait is stall time,
not total transform CPU cost. RSS is a process-lifetime high-water mark, not an
instantaneous total for every worker; GPU utilization is not sampled by this benchmark.
The benchmark is inference throughput, not training throughput. Persistence benefits
require multiple epoch starts and cannot be established by this single-iterator test.
`recommendation.json` identifies only the best measured row; defaults never change
automatically. Tiny trials and warm filesystem caches do not establish a stable winner.
Optional `--save-checkpoint` saves the exact benchmark model state under the run's
`checkpoints/`, clearly identified as an inference snapshot, not newly trained weights.

### Remote source, dependencies and dataset

```bash
python scripts/colab_manager.py doctor
python scripts/colab_manager.py files
python scripts/colab_manager.py sync
python scripts/colab_manager.py setup
```

The CLI is discovered on PATH or `~/.local/bin/colab`. No token/config file contents
are printed. Codex handles source synchronization as needed. Source-only sync excludes `.git`, `.venv`, `.aws`, `.agents`, `.codex`,
`.ssh`, `.config`, data/checkpoints/runs/logs and known secret filenames. Codex reviews the
file list before uploading. SHA-256 checks verify actual remote file bytes, not just
a cached marker; stale previously synchronized files are removed. Data and supplied
checkpoints are never synchronized with source.

Remote setup uses `requirements-colab.txt` and exact constraints for installed
`torch`/`torchvision`. A fresh interpreter verifies imports and CUDA after setup;
unsupported dependency conflicts fail rather than replacing the CUDA stack.
Setup never modifies local dependencies. Existing compatible packages are retained.

Codex checks data availability and prepares the dataset once per new runtime, not per code edit. It restores an accessible authorized Drive archive or transfers existing local data when appropriate, then verifies the actual layout. Missing files/access are reported clearly. Costly transfers require approval; Drive OAuth may require interactive user authorization. The equivalent troubleshooting commands are:

```bash
# Preferred: archive top-level data/ in MyDrive/ELEC475-Lab3/dataset.tar.gz
colab drivemount -s lab3             # run interactively; complete OAuth yourself
python scripts/colab_manager.py data restore
# Alternative: explicit one-time local upload (may be large)
python scripts/colab_manager.py data push
# For >400 MiB compressed, inspect transfer cost then explicitly add --allow-large.
python scripts/colab_manager.py data backup   # refuses to overwrite existing Drive archive
```

Before model runs the manager checks both remote label files, all referenced image
paths, counts and sample tensors. An empty `data/` directory is insufficient.
When available and needed remotely, Codex transfers supplied weights separately once per runtime (not with source sync), checks transfer cost and verifies the files. An equivalent troubleshooting upload is:

```bash
colab upload -s lab3 checkpoints/snoutnet.pth /content/elec475-lab3/checkpoints/snoutnet.pth
```

Codex creates the remote `checkpoints/` directory and handles other required weights as needed. The initial integration tests did not transfer the full dataset or course checkpoints. A missing checkpoint remains a verification blocker.

A cheap transport/dependency smoke test on the existing session, without model compute:

```bash
python scripts/colab_manager.py run --backend colab --task environment
```

Do not run remote benchmarks simply because the session is available. For future
expensive work, discuss benefit, transfer cost and run budget first. An approved
forced remote benchmark uses `--backend colab`; an automatic recommendation uses
`--approve-remote`. Long/memory-heavy work additionally needs
`--approve-expensive --budget-seconds N`. Keep `remote_timeout_seconds` at least
60 seconds greater than the job budget; an expired CLI connection does not prove
remote computation stopped.

### Records, retrieval and safe shutdown

Each run gets `runs/<timestamp>_<backend>_wN_bB/` with `terminal.txt`,
`config_snapshot.json`, child `stdout.txt`, `settings.json`, `summary.json`,
`experiment_status.json` and task metrics. Failed jobs retain logs and nonzero
status; absent weights/data are errors, not successful verification. The wall-clock
budget terminates the child process group, including DataLoader workers.

Remote runs fetch only their run directory into local `artifacts/runs/<run-id>/`,
including any generated checkpoints, metrics or plots saved there. This avoids
re-downloading unrelated instructor checkpoints. Future assigned training must save
periodic resumable checkpoints inside `ELEC475_OUTPUT_DIR/checkpoints`; the current
pre-lab does no training and does not fabricate trained checkpoints. Interrupted
connections require inspection, not a competing run:

```bash
python scripts/colab_manager.py status
python scripts/colab_manager.py fetch --run-id RUN_ID
```

Remote `.colab_state/active-run` and `runs/<id>/RUNNING` markers block new mutations
when work may be unfinished. Fetch is allowed for recovery. Inspect actual remote
process state and save outputs before manually resolving stale markers; never just
delete them to bypass the guard. A fetch proves files transferred at that moment,
not that a running job is finished. `--no-fetch` requires later retrieval; Codex owns recovery in the normal workflow.

After retrieving **all** outputs and confirming no other notebook work is unsaved,
ask/obtain permission to release an idle paid session, then:

```bash
python scripts/colab_manager.py stop --confirm-saved
```

Stop refuses BUSY/unknown status, unfinished managed runs, and remote runs without
local fetch markers. It cannot verify unrelated unsaved notebook state; the explicit
confirmation is required. The toolkit never stops, creates or wakes sessions automatically.

### Initial lightweight validation results

Seven policy/synchronization safety unit tests passed. The local dataset-layout task
passed, and an existing-session environment-only run verified source upload,
CUDA-preserving dependency setup, logging and automatic local artifact retrieval.
No dataset/checkpoint transfers, model training or A100 model benchmarks were launched.

A local RTX 3050 SnoutNet inference smoke test used batch size 2, seed 475,
1 warmup batch and just 2 measured batches per worker setting:

| Workers | Data wait ms/batch | Compute ms/batch | Samples/s | Iterator startup s | Peak allocated MiB |
|---:|---:|---:|---:|---:|---:|
| 0 | 20.16 | 3.47 | 79.95 | 0.00 | 106.6 |
| 2 | 3.09 | 28.23 | 58.09 | 15.50 | 106.6 |
| 4 | 0.28 | 27.72 | 65.96 | 22.76 | 106.6 |
| 8 | 10.30 | 28.65 | 48.25 | 61.20 | 106.6 |

Prediction checksums matched across all settings. The complete smoke job took
117.9 seconds, mostly worker startup/imports; its 120-second budget was respected.
Results and a 22.5 MB explicitly untrained model snapshot are saved in
`runs/20261019_201311_876745_local_w0_b2/`. These tiny measurements support keeping
0 workers for smoke/debug jobs, not a universal throughput claim. Pinning, larger
batches and persistence remain configurable experiments; no unmeasured speedup is
claimed. The newer RAM guard can skip worker settings when available memory is low.
The downloaded Colab environment report is under
`runs/20261019_201559_027344_colab_w0_b4/artifacts/runs/20261019_201559_027344_colab_w0_b4/`.
