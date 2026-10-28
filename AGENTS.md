# Codex daily operating instructions — ELEC 475 Lab 3

## User workflow and Codex ownership

The user's normal setup is only:

```bash
cd ~/ELEC475/"Lab 3"
source .venv/bin/activate
code .
```

After that, the user asks Codex to implement, test, benchmark or run experiments.
Treat those requests as authorization to handle the routine infrastructure needed
for that task. Do not send the user a synchronization/setup checklist or require
manual toolkit commands during normal use. These are agent operating instructions,
not a background service: perform them when handling a user request.

- Read `README_COLAB.md`, `colab_config.json` and the relevant assignment instructions.
- Inspect the actual entry point and prerequisites before execution. Check hardware,
  local dependencies, dataset layout and supplied checkpoints as appropriate.
- Select the execution backend, run bounded checks, prepare available data, handle
  source synchronization and compatible remote dependencies when remote execution
  is justified, execute the requested task, tune DataLoader settings when relevant,
  capture logs/metrics, retrieve generated checkpoints/artifacts and verify results.
- Use the existing manager/experiment tools yourself. Keep their commands in the
  README's advanced section for troubleshooting; do not ask the user to run them
  merely because a routine step is needed.
- Report the result, chosen hardware/reason, relevant measurements, local artifact
  locations and any missing prerequisites or failed checks. Do not claim success
  from a launched command, an empty output directory or a partial verification.

## Permanent adaptive execution policy

- Prefer local WSL for edits, syntax/import checks, unit tests, preprocessing,
  debugging, dataset validation and small experiments. Use the local RTX 3050 for
  bounded CUDA checks when useful. Do not use Colab simply because it is available.
- Honor `execution_backend: auto|local|colab` and explicit user overrides. Probe
  local CUDA/free VRAM and estimate workload cost conservatively. Unknown work
  stays local. A sandbox hiding CUDA is not evidence that Colab is necessary.
- Recommend A100 only for expensive workloads, large models/data, sweeps or
  demonstrated local time/VRAM constraints. Explain the reason, expected benefit,
  source/data transfer overhead and retrieval cost before remote execution.
  Never silently migrate or retry a failed local job remotely.
- Ask before long, costly, overnight or repeated experiments on either backend;
  agree on a finite run budget. Approval flags record actual authorization, never
  substitute for it. Use `--approve-remote` for an authorized auto recommendation;
  expensive runs also require `--approve-expensive --budget-seconds N`.

## Colab GPU Session Management

- Apply the local-first policy above: use Colab only when its performance or
  memory advantage justifies the cost. Reuse a suitable already-running IDLE
  session instead of creating another. Handle status checks, source sync,
  dependencies and execution yourself within the authorized task; routine
  checks/setup do not need repeated user confirmation.
- Obtain explicit user approval before creating, restarting, waking or allocating
  any Colab GPU session.
- The current toolkit only reuses existing allocations; it does not create/wake
  sessions. If one is missing, report that and obtain approval before using a
  supported allocation method. Do not assume an old session still exists.
- Unknown/BUSY status must fail closed. Do not compete with shared work, bypass
  active-run guards or delete stale markers without investigating actual state.
- After each batch of remote experiments, verify that all logs, metrics,
  checkpoints and results are saved to local WSL or persistent storage. Follow
  the artifact retrieval and verification requirements below before shutdown.
- When remote work is complete and the session is idle, recommend stopping it
  and request user approval. Do not leave an idle A100 running unnecessarily.
  If approval is unavailable, explicitly report that it remains active; never
  assume or claim it stopped.
- Never stop a session with active jobs, unfinished work, unsaved files or other
  notebook work without confirmation. Confirm all shared work is complete and
  saved, then shut down only with user approval; idleness alone is not permission.
- For approved overnight experiments, first verify that execution can continue
  independently of VS Code/WSL, save periodic resumable checkpoints to persistent
  storage, and plan for Colab runtime interruptions and recovery. Do not assume
  the current synchronous CLI runner provides detached execution or durable
  checkpoints; arrange and verify those capabilities before starting overnight work.
- After an approved shutdown, check live Colab status to verify the session is
  stopped and report its final status. If verification fails or is inconclusive,
  report that uncertainty rather than claiming shutdown succeeded.
- Closing VS Code or shutting down WSL does not terminate a Colab GPU session.

## Data, dependencies and artifact handling

- The local WSL folder is the source of truth; Colab storage is temporary. Never
  assume remote edits propagate back. Sync eligible source only, verify checksums
  and avoid dataset/checkpoint transfers on every code edit.
- Inspect `data/oxford-iiit-pet-noses/images-original/images`, `train_noses.txt` and
  `test_noses.txt`; verify referenced images and sample tensors, not just a directory.
  Inspect required `checkpoints/*.pth` files before claiming weight verification.
- Prepare data automatically from available authorized local files or an already
  accessible Drive archive: check remote layout, restore or push once per runtime,
  and verify afterward. Transfer supplied checkpoints separately when needed.
  Inspect transfer size/cost first; ask before an expensive transfer. Do not
  fabricate missing assets or replace course data with substitutes.
- If files are unavailable, clearly report exact missing paths and which checks
  remain blocked, continuing independent work. Ask only for the missing access or
  asset, not for routine setup commands. Downloads requiring course login or Drive
  OAuth may need user interaction; explain that specific limitation. Never expose
  authentication URLs/codes, tokens, CLI config, credentials or secrets in logs.
- Preserve working CUDA torch/torchvision. Local dependencies use `requirements.txt`;
  remote setup uses `requirements-colab.txt` with constraints preserving Colab's
  existing CUDA stack, followed by fresh import/CUDA checks. Handle compatible
  dependency preparation yourself; do not blindly reinstall PyTorch.
- Use managed runs for experiments. Record backend/reason, hardware/software,
  effective parameters, source/settings, terminal logs, status and metrics in unique
  local `runs/` folders. Honor `ELEC475_NUM_WORKERS`, `ELEC475_BATCH_SIZE`,
  `ELEC475_DATASET_ROOT` and `ELEC475_OUTPUT_DIR` where supported.
- Append terminal output from Codex-run post-lab Python scripts, tests, debugging,
  verification and training to `runs/postlab/terminal_output.txt`, including the
  command, timestamp, stdout/stderr and exit status. Preserve earlier entries and
  retain any separate managed-run logs. This does not capture commands run in the
  user's independent terminal; do not claim those are logged unless explicitly
  captured or supplied by the user. Continue excluding secrets from logs.
- Also keep timestamped, section-named per-command logs for post-lab work and
  update `POSTLAB_LOGS.md` with links to logs, verification results and figures.
  Preserve previous run artifacts when regenerating outputs.
- Retrieve important remote artifacts automatically, verify checksums and inspect
  logs/checkpoints/metrics before reporting completion or requesting shutdown.
  Recover interrupted retrieval yourself where possible. Never erase unfinished
  work to simplify recovery.
- Future assigned training must save periodic resumable checkpoints inside
  `ELEC475_OUTPUT_DIR/checkpoints`. Inference does not generate trained weights;
  optional benchmark snapshots must remain explicitly identified as untrained.

## Assignment correctness and performance tuning

- This pre-lab explicitly says not to train. There is no `train.py` or training loop.
  `experiment_command` is `python -m models.snoutnet` for architectures, strict
  supplied-weight loading and the known prediction. `python -m utils.check_dataset`
  validates/plots the dataset. Do not invent a trainer, optimizer, loss, architecture
  or future assignment protocol to satisfy a generic request to run experiments.
- Preserve all instructor TODO comments verbatim, even after implementation, and
  existing lab requirements, architectures, preprocessing and experiment logging.
  Avoid unrelated rewrites of assignment code or toolkit functionality.
- Start with lightweight smoke validation. Run relevant syntax/import/unit checks
  locally; do not sync or allocate remote compute for documentation-only changes.
- Tune DataLoader settings yourself when relevant, using controlled short trials
  rather than asking the user to run benchmark commands. Compare 0/2/4/8 workers
  with the same seed, data order, model, batch size, transforms and bounded steps.
  Respect CPU/RAM limits; record skips/failures. Exclude warmup and measure loading
  stalls, H2D transfer, synchronized compute, samples/sec and peak VRAM separately.
- Compare batch size, pinning and persistence separately; persistence benefits need
  multiple iterator/epoch starts. For workers=0 disable persistence and use no
  prefetch; for workers>0 test rather than assume the benefit of pinning,
  persistence or prefetch=2. Do not change optimization/evaluation semantics.
- Choose settings supported by measurements, not the largest available values.
  Record a benchmark table and limitations; tiny cached trials are not definitive.
  Obtain approval before extending short trials into expensive tuning or sweeps.

## Session history (not live status)

On 2026-10-19, the `lab3` A100 environment check completed and outputs were
retrieved locally. The session was then stopped with the user's explicit approval
after final source synchronization. This history is not a substitute for live
status checks during remote work or shutdown verification; obtaining a new
allocation requires fresh approval.
