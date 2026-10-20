#!/usr/bin/env python3
"""WSL-side Colab runner for ELEC 475. Python >= 3.9; no third-party deps.

Nothing here edits your instructor's source files. The existing official `colab`
CLI is the only remote dependency. Read README_COLAB.md before first use.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "colab_config.json"
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
             ".aws", ".agents", ".codex", ".ssh", ".config", ".tox", ".idea", ".vscode", "data", "checkpoints", "logs", "runs", "results", "outputs",
             "wandb", "mlruns", "tensorboard", ".colab_state"}
SKIP_NAMES = {".env", ".env.local", "id_rsa", "id_ed25519", "credentials.json", "token.json",
              "sessions.json", "settings.json", ".DS_Store"}
ALLOWED_EXT = {".py", ".ipynb", ".txt", ".md", ".json", ".toml", ".yaml", ".yml", ".csv", ".sh", ".ini", ".cfg"}
MAX_SOURCE_FILE = 16 * 1024 * 1024
MARKER = "__ELEC475_RESULT__="


def config():
    if not CONFIG.is_file():
        raise RuntimeError(f"Missing {CONFIG}")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    root = cfg.get("remote_root", "/content/elec475-lab3")
    if not root.startswith("/content/") or ".." in Path(root).parts:
        raise ValueError("remote_root must be an absolute path under /content")
    return cfg


def cli(cfg, *args, input_code=None, capture=True, log_file=None):
    binary = shutil.which("colab") or str(Path.home() / ".local/bin/colab")
    if args and args[0] not in ("status", "stop") and not cfg.get("_session_verified"):
        status = cli(cfg, "status", "-s", cfg["session"])
        # IDLE is an allocated, connected kernel; never reconnect/create a runtime.
        if not re.search(r"Status: IDLE\b", status):
            raise RuntimeError("Session must already be allocated and IDLE. Refusing to create, wake, or compete with a busy session. Check `colab status` manually.")
        cfg["_session_verified"] = True
    cmd = [binary, *map(str, args)]
    if log_file is None and capture:
        p = subprocess.run(cmd, input=input_code, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
        if p.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed (exit {p.returncode}):\n{p.stdout[-3500:]}")
        return p.stdout
    if input_code is None:
        stdin = None
    else:
        stdin = subprocess.PIPE
    with subprocess.Popen(cmd, stdin=stdin, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        if input_code is not None:
            process.stdin.write(input_code)
            process.stdin.close()
        lines = []
        handle = open(log_file, "w", encoding="utf-8") if log_file else None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                if handle:
                    handle.write(line)
                    handle.flush()
                lines.append(line)
        finally:
            if handle:
                handle.close()
        rc = process.wait()
    out = "".join(lines)
    if rc:
        raise RuntimeError(f"{' '.join(cmd)} failed (exit {rc}) — see terminal log")
    return out


def remote(cfg, code, *, log_file=None, allow_active=False):
    guard = "from pathlib import Path\n" + "_active=Path(" + repr(cfg["remote_root"]) + ") / '.colab_state/active-run'\n" + "if _active.exists(): raise RuntimeError('Unfinished remote job: inspect/fetch output and resolve active-run marker manually')\n"
    return cli(cfg, "exec", "-s", cfg["session"], "--timeout", str(cfg.get("remote_timeout_seconds", 600)), input_code=("" if allow_active else guard) + code,
               log_file=log_file, capture=log_file is None)


def payload_code(payload, body):
    # Encode JSON once as a Python string literal to avoid JSON true/false syntax errors.
    return "import json\np=json.loads(" + repr(json.dumps(payload)) + ")\n" + body + "\n"


def marked_result(output):
    for line in reversed(output.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    raise RuntimeError("Remote command did not produce a completion marker; check CLI output")


def get_file_list():
    files = []
    configured_output = Path(config().get("output_root", "runs"))
    excluded = SKIP_DIRS | ({configured_output.parts[0]} if not configured_output.is_absolute() else set())
    for directory, dirs, names in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in excluded and not d.startswith(".uv"))
        for name in sorted(names):
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                continue
            if name in SKIP_NAMES or name.startswith(".env") or path.suffix.lower() not in ALLOWED_EXT:
                continue
            if path.stat().st_size > MAX_SOURCE_FILE:
                print(f"[skip] Large source file: {path.relative_to(ROOT)}", file=sys.stderr)
                continue
            files.append(path)
    return files


def sync(cfg):
    files = get_file_list()
    if not files:
        raise RuntimeError("No source files found")
    manifest = {str(path.relative_to(ROOT).as_posix()): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in files}
    archive_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    probe = payload_code({"root": cfg["remote_root"], "manifest": manifest}, '''
from pathlib import Path
import hashlib
root=Path(p["root"])
valid=all((root/name).is_file() and hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in p["manifest"].items())
marker=root/".colab_state/source.sha256"
print("__ELEC475_RESULT__=" + json.dumps({"hash": marker.read_text().strip() if valid and marker.is_file() else None}))
''')
    if marked_result(remote(cfg, probe)).get("hash") == archive_id:
        print(f"[sync] Unchanged ({len(files)} files); skipping upload")
        return
    with tempfile.TemporaryDirectory() as temp:
        archive = Path(temp) / "elec475_source.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for path in files:
                tar.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
        destination = "/content/elec475_source_upload.tar.gz"
        print(f"[sync] Sending {len(files)} source files ({archive.stat().st_size / 1048576:.1f} MiB)")
        cli(cfg, "upload", "-s", cfg["session"], archive, destination, capture=False)
        extracted = payload_code({"root": cfg["remote_root"], "archive": destination,
                                  "hash": archive_id, "files": list(manifest), "manifest": manifest}, '''
import tarfile
from pathlib import Path
root = Path(p["root"]); root.mkdir(parents=True, exist_ok=True)
state = root / ".colab_state"; state.mkdir(exist_ok=True)
old_manifest_path = state / "source_files.json"
try: old_files = json.loads(old_manifest_path.read_text())
except (OSError, ValueError): old_files = []
with tarfile.open(p["archive"], "r:gz") as tar:
    members = tar.getmembers()
    for m in members:
        candidate = (root / m.name).resolve()
        if not candidate.is_relative_to(root.resolve()) or not m.isfile():
            raise RuntimeError("Unexpected archive entry: " + m.name)
    for m in members:
        dest = root / m.name; dest.parent.mkdir(parents=True, exist_ok=True)
        with tar.extractfile(m) as src, dest.open("wb") as target:
            import shutil
            shutil.copyfileobj(src, target)
for name in old_files:
    if name not in p["files"] and name not in ("data", "checkpoints", "runs"):
        old = (root / name).resolve()
        if old.is_relative_to(root.resolve()) and old.is_file(): old.unlink()
import hashlib
for name, expected in p["manifest"].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
        raise RuntimeError("Source checksum mismatch: " + name)
(state / "source_manifest.json").write_text(json.dumps(p["manifest"], indent=2))
old_manifest_path.write_text(json.dumps(p["files"], indent=2))
(state / "source.sha256").write_text(p["hash"])
Path(p["archive"]).unlink(missing_ok=True)
print("__ELEC475_RESULT__=" + json.dumps({"files": len(members)}))
''')
        print(f"[sync] {marked_result(remote(cfg, extracted))['files']} files available at {cfg['remote_root']}")


def setup(cfg):
    """Install only missing non-torch dependencies; constrain the working CUDA stack."""
    req = ROOT / cfg.get("remote_requirements", "requirements-colab.txt")
    result = marked_result(remote(cfg, payload_code({"root": cfg["remote_root"], "requirements": req.name}, r'''
import importlib, importlib.metadata, subprocess, sys, hashlib
from pathlib import Path
import torch, torchvision
assert torch.cuda.is_available(), "Existing Colab CUDA stack unavailable; repair manually"
root=Path(p["root"])
before={name:importlib.metadata.version(name) for name in ("torch","torchvision")}
constraints=root/".colab_state/cuda-constraints.txt"
constraints.parent.mkdir(exist_ok=True)
constraints.write_text("".join(name+"=="+version+"\n" for name,version in before.items()))
req=root/p["requirements"]
sha=hashlib.sha256(req.read_bytes()).hexdigest()
marker=root/".colab_state/requirements.sha256"
if not marker.exists() or marker.read_text()!=sha:
    subprocess.run([sys.executable,"-m","pip","install","-r",str(req),"-c",str(constraints)],check=True)
# Validate fresh imports rather than stale kernel imports after pip.
subprocess.run([sys.executable,"-c","import torch, torchvision, numpy, PIL, matplotlib, torchinfo, tqdm; assert torch.cuda.is_available(); print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0))"],check=True)
after={name:importlib.metadata.version(name) for name in before}
assert before==after, "CUDA stack changed unexpectedly"
marker.write_text(sha)
print("__ELEC475_RESULT__="+json.dumps({"torch_stack":after,"gpu":torch.cuda.get_device_name(0)}))
''')))
    print("[setup] " + str(result))


def data_action(cfg, action, force=False, allow_large=False):
    root = cfg["remote_root"]
    if action == "push":
        data_path = ROOT / cfg.get("local_dataset", "data")
        if not data_path.is_dir():
            raise RuntimeError(f"Dataset folder not found: {data_path}")
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "elec475_dataset.tar.gz"
            print("[data] Archiving local dataset; may take a few minutes", flush=True)
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(data_path, arcname="data")
            mib = archive.stat().st_size / 1048576
            if mib > 400 and not allow_large:
                raise RuntimeError(f"Dataset archive is {mib:.0f} MiB. Prefer Drive restore; "
                                   "add --allow-large to explicitly upload this much data.")
            cli(cfg, "upload", "-s", cfg["session"], archive,
                "/content/elec475_dataset_upload.tar.gz", capture=False)
        archive_remote = "/content/elec475_dataset_upload.tar.gz"
    elif action == "restore":
        archive_remote = cfg.get("drive_dataset_archive", "/content/drive/MyDrive/ELEC475-Lab3/dataset.tar.gz")
    elif action == "backup":
        archive_remote = cfg.get("drive_dataset_archive", "/content/drive/MyDrive/ELEC475-Lab3/dataset.tar.gz")
        code = payload_code({"root": root, "destination": archive_remote}, '''
from pathlib import Path
import tarfile
src=Path(p["root"])/"data"
if not src.is_dir(): raise RuntimeError("Remote data folder absent")
dst=Path(p["destination"])
if not dst.parent.is_dir(): raise RuntimeError("Mount Drive first: colab drivemount -s lab3")
if dst.exists(): raise RuntimeError("Drive archive already exists; refusing to overwrite")
with tarfile.open(dst, "w:gz") as tar: tar.add(src, arcname="data")
print("__ELEC475_RESULT__=" + json.dumps({"saved":str(dst),"bytes":dst.stat().st_size}))
''')
        print("[data] " + str(marked_result(remote(cfg, code))))
        return
    else:
        raise ValueError(action)
    code = payload_code({"root": root, "archive": archive_remote, "force": force}, '''
from pathlib import Path
import tarfile, shutil
root=Path(p["root"]); root.mkdir(parents=True,exist_ok=True)
data=root/"data"
if data.is_dir() and any(data.iterdir()) and not p["force"]:
    print("__ELEC475_RESULT__=" + json.dumps({"message":"Already present; use --force to restore again"}))
else:
    arc=Path(p["archive"])
    if not arc.is_file(): raise RuntimeError("Archive not found: " + str(arc) + ". Check Drive mount or upload.")
    if p["force"] and data.exists(): shutil.rmtree(data)
    with tarfile.open(arc, "r:gz") as tar:
        members=tar.getmembers()
        for m in members:
            if not (m.name == "data" or m.name.startswith("data/")) or not (m.isfile() or m.isdir()) or not (root/m.name).resolve().is_relative_to(root.resolve()):
                raise RuntimeError("Invalid dataset archive member: " + m.name)
        for m in members:
            dest=root/m.name
            if m.isdir(): dest.mkdir(parents=True,exist_ok=True)
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(m) as src, dest.open("wb") as out: shutil.copyfileobj(src,out)
    print("__ELEC475_RESULT__=" + json.dumps({"message":"Dataset ready", "files":sum(1 for x in data.rglob("*") if x.is_file())}))
''')
    print("[data] " + str(marked_result(remote(cfg, code))))


def fetch(cfg, run_id, local_dest=None):
    """Download outputs into runs/<ID>/artifacts, preserving every run's history."""
    out = local_dest or ROOT / cfg.get("output_root", "runs") / run_id
    out.mkdir(parents=True, exist_ok=True)
    archive_remote = f"/content/elec475_results_{run_id}.tar.gz"
    code = payload_code({"root": cfg["remote_root"], "run_id": run_id, "archive": archive_remote}, '''
import tarfile
from pathlib import Path
root=Path(p["root"]); paths=[]
run=root/"runs"/p["run_id"]
if not run.is_dir(): raise RuntimeError("Remote run not found; refusing an empty retrieval")
paths.append(run)
import hashlib
manifest={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in run.rglob("*") if path.is_file() and not path.is_symlink()}
with tarfile.open(p["archive"], "w:gz") as tar:
    for path in paths: tar.add(path, arcname=path.relative_to(root).as_posix())
print("__ELEC475_RESULT__=" + json.dumps({"paths":[x.name for x in paths], "bytes":Path(p["archive"]).stat().st_size, "manifest":manifest}))
''')
    result = marked_result(remote(cfg, code, allow_active=True))
    print(f"[fetch] Downloading {result['bytes'] / 1048576:.1f} MiB: {', '.join(result['paths']) or 'no artifacts yet'}")
    archive_local = out / "artifacts.tar.gz"
    cli(cfg, "download", "-s", cfg["session"], archive_remote, archive_local, capture=False)
    target = out / "artifacts"
    target.mkdir(exist_ok=True)
    with tarfile.open(archive_local, "r:gz") as tar:
        for member in tar.getmembers():
            dest = (target / member.name).resolve()
            if not dest.is_relative_to(target.resolve()) or not (member.isdir() or member.isfile()):
                raise RuntimeError("Unsafe downloaded artifact archive")
            if member.isdir(): dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    for name, expected in result["manifest"].items():
        path = (target / name).resolve()
        if not path.is_relative_to(target.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Artifact checksum mismatch: {name}")
    archive_local.unlink(missing_ok=True)
    (out / "artifact_manifest.json").write_text(json.dumps(result["manifest"], indent=2))
    print(f"[fetch] Saved outputs under {target}")


def local_python():
    candidate = ROOT / ".venv/bin/python"
    return str(candidate) if candidate.exists() else sys.executable


def local_hardware():
    process = subprocess.run([local_python(), "scripts/experiment.py", "--task", "hardware"],
                             cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode:
        return {"cuda_available": False, "probe_error": process.stderr[-1000:]}
    return json.loads(process.stdout.splitlines()[-1])


def choose_backend(cfg, requested, estimated_minutes=None, required_vram_gb=None, workload=None):
    hw = local_hardware()
    workload = workload or {}
    if estimated_minutes is None:
        estimated_minutes = workload.get("rough_minutes")
    if required_vram_gb is None and hw.get("cuda_available"):
        required_vram_gb = workload.get("rough_vram_gb")
    requested = requested or cfg.get("execution_backend", "auto")
    expensive = estimated_minutes is not None and estimated_minutes > cfg.get("auto_policy", {}).get("long_job_minutes", 15)
    memory_short = required_vram_gb is not None and required_vram_gb * 1024**3 > hw.get("free_vram_bytes", 0) * cfg.get("auto_policy", {}).get("local_vram_fraction", 0.8)
    recommended = "colab" if expensive or memory_short else "local"
    if requested != "auto":
        chosen, reason = requested, "explicit backend override"
    elif recommended == "colab":
        chosen, reason = "colab", "declared workload exceeds local time or free VRAM budget"
    else:
        chosen, reason = "local", "small/unknown workload; local WSL preferred; no evidence justifies remote overhead"
    return {"backend": chosen, "reason": reason, "requested": requested, "hardware": hw,
            "estimated_minutes": estimated_minutes, "required_vram_gb": required_vram_gb,
            "costly": expensive or memory_short, "workload_estimate": workload}


def estimate_workload(args, extra, cfg):
    """Conservative routing hints, not promises of measured runtime or allocation."""
    def option(name, default):
        for index, value in reversed(list(enumerate(extra))):
            if value.startswith(name + "="):
                return value.split("=", 1)[1]
            if value == name and index + 1 < len(extra):
                return extra[index + 1]
        return default
    if args.task == "benchmark":
        model = option("--model", "snoutnet")
        batch = args.batch_size or cfg.get("batch_size", 4)
        batches = (int(option("--steps", 4)) + int(option("--warmup", 1))) * 4
        factor = {"snoutnet": 1, "alexnet": 4, "vgg16": 20}.get(model, 20)
        return {"model": model, "batches_upper_hint": batches, "batch_size": batch,
                "rough_minutes": batches * batch * factor * 0.5 / 60,
                "rough_vram_gb": {"snoutnet": 0.05, "alexnet": 0.3, "vgg16": 0.6}.get(model, 0.6) + batch * {"snoutnet": 0.05, "alexnet": 0.03, "vgg16": 0.18}.get(model, 0.18),
                "confidence": "low; conservative inference heuristic; measure before scaling"}
    return {"task": args.task, "rough_minutes": 2 if args.task in ("verify", "dataset-check") else 0.25,
            "confidence": "low; bounded pre-lab task; prefer local"}


def run(cfg, args, extra):
    reserved = {"--task", "--num-workers", "--batch-size", "--output-dir", "--entry-command"}
    if any(value.split("=", 1)[0] in reserved for value in extra):
        raise ValueError("Set task/workers/batch-size before `--`, output location with --output-root, and entry command in colab_config.json")
    decision = choose_backend(cfg, args.backend, args.estimated_minutes, args.required_vram_gb,
                              estimate_workload(args, extra, cfg))
    print("[policy] " + json.dumps(decision), flush=True)
    backend = decision["backend"]
    workers = cfg.get("num_workers", 0) if args.workers is None else args.workers
    batch = cfg.get("batch_size", 4) if args.batch_size is None else args.batch_size
    if workers < 0 or workers > 32 or batch < 1:
        raise ValueError("Invalid worker count or batch size")
    if args.dry_run:
        print("[policy] Recommendation only; no sync, installation, session allocation, or experiment")
        return
    if decision["costly"] and not (args.approve_expensive and args.budget_seconds):
        raise RuntimeError("Costly work requires explicit approval and --approve-expensive --budget-seconds N")
    if backend == "colab":
        print("[policy] Remote benefit: A100 speed/VRAM for declared workload or explicit override. Overhead: source hash/upload, dependency and dataset checks, artifact download. Existing IDLE session only.", flush=True)
        if decision["requested"] == "auto" and not args.approve_remote:
            raise RuntimeError("Recommendation only: approve remote execution with --approve-remote, or force --backend local")
    if args.budget_seconds is not None and args.budget_seconds < 1:
        raise ValueError("Run budget must be positive")
    if backend == "colab" and (args.budget_seconds or cfg.get("smoke_budget_seconds", 180)) + 60 > cfg.get("remote_timeout_seconds", 600):
        raise ValueError("Remote CLI timeout must exceed the experiment budget by at least 60 seconds")
    if backend == "colab":
        sync(cfg)
        if not args.no_setup:
            setup(cfg)
        if cfg.get("require_dataset_before_run", True) and args.task not in ("layout", "environment"):
            check = payload_code({"root": cfg["remote_root"], "dataset_root": cfg["dataset_root"]}, '''
import sys
from pathlib import Path
sys.path.insert(0,p["root"])
from scripts.experiment import dataset_layout
layout=dataset_layout(Path(p["root"])/p["dataset_root"])
print("__ELEC475_RESULT__="+json.dumps(layout))
''')
            print("[dataset] " + json.dumps(marked_result(remote(cfg, check))))
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{backend}_w{workers}_b{batch}"
    local_dir = Path(args.output_root or cfg.get("output_root", "runs")) / run_id
    if not local_dir.is_absolute():
        local_dir = ROOT / local_dir
    local_dir.mkdir(parents=True, exist_ok=False)
    remote_root = cfg["remote_root"] if backend == "colab" else str(ROOT)
    output = f"{remote_root}/runs/{run_id}" if backend == "colab" else str(local_dir)
    command = ["python", "scripts/experiment.py", "--task", args.task,
               "--num-workers", str(workers), "--batch-size", str(batch), "--output-dir", output,
               "--dataset-root", cfg.get("dataset_root", "data/oxford-iiit-pet-noses")]
    command += ["--entry-command", json.dumps(cfg["experiment_command"])]
    if args.task == "benchmark":
        command += ["--workers-grid", "0", "2", "4", "8"]
    command += extra
    settings = {"decision": decision, "command": command, "config": {k:v for k,v in cfg.items() if not k.startswith("_")}, "run_id": run_id,
                "source_manifest": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in get_file_list()}}
    (local_dir / "config_snapshot.json").write_text(json.dumps(settings, indent=2))
    payload = {"root": remote_root, "output": output, "cmd": command, "workers": workers,
               "batch_size": batch, "decision": decision, "settings": settings,
               "timeout": args.budget_seconds or cfg.get("smoke_budget_seconds", 180)}
    # Shared execution wrapper saves logs and status even when the child fails/times out.
    body = r'''
import os, sys, subprocess, time, threading
from pathlib import Path
root=Path(p["root"]); out=Path(p["output"]); out.mkdir(parents=True,exist_ok=True)
(out/"config_snapshot.json").write_text(json.dumps(p["settings"],indent=2))
cmd=p["cmd"]; cmd[0]=sys.executable
reason=p["decision"]["reason"]
env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=str(root),
         ELEC475_BACKEND=p["decision"]["backend"], ELEC475_BACKEND_REASON=reason,
         ELEC475_NUM_WORKERS=str(p["workers"]), ELEC475_BATCH_SIZE=str(p["batch_size"]),
         ELEC475_OUTPUT_DIR=str(out), CUBLAS_WORKSPACE_CONFIG=":4096:8")
start=time.time(); rc=1
lock=None
if p["decision"]["backend"]=="colab":
    lock=root/".colab_state/active-run"; lock.parent.mkdir(exist_ok=True)
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,"w") as handle: handle.write(str(out))
(out/"RUNNING").write_text(str(os.getpid()))
try:
    with (out/"stdout.txt").open("w") as logfile:
        with subprocess.Popen(cmd,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True) as proc:
            def expire():
                import signal
                try: os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError: pass
            timer=threading.Timer(p["timeout"],expire); timer.start()
            try:
                for line in proc.stdout:
                    print(line,end="",flush=True); logfile.write(line); logfile.flush()
                rc=proc.wait()
            finally: timer.cancel()
finally:
    summary={"returncode":rc,"duration_s":time.time()-start,"backend":p["decision"]["backend"],"reason":reason,"command":cmd,"budget_seconds":p["timeout"]}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    (out/"RUNNING").unlink(missing_ok=True)
    if lock: lock.unlink(missing_ok=True)
    print("__ELEC475_RESULT__="+json.dumps(summary),flush=True)
'''
    failure = None
    try:
        if backend == "local":
            process = subprocess.Popen([local_python(), "-u", "-c", payload_code(payload, body)], cwd=ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            lines = []
            with (local_dir / "terminal.txt").open("w") as handle:
                for line in process.stdout:
                    print(line, end="", flush=True); handle.write(line); handle.flush(); lines.append(line)
            rc = process.wait()
            if rc:
                raise RuntimeError(f"Local runner failed: {rc}")
            result = marked_result("".join(lines))
        else:
            result = marked_result(remote(cfg, payload_code(payload, body), log_file=local_dir / "terminal.txt"))
        if result["returncode"]:
            raise RuntimeError(f"Experiment exited {result['returncode']}; see {local_dir}")
    except Exception as exc:
        failure = exc
    finally:
        if backend == "colab" and not args.no_fetch:
            try:
                fetch(cfg, run_id, local_dest=local_dir)
                (local_dir / "FETCHED").write_text(run_id)
            except Exception as exc:
                print(f"Retrieval failed; recover with fetch --run-id {run_id}: {exc}", file=sys.stderr)
                failure = failure or exc
    if failure:
        raise RuntimeError(str(failure))
    print(f"[run] Saved {local_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "status", "sync", "setup"):
        sub.add_parser(name)
    stop_parser = sub.add_parser("stop")
    stop_parser.add_argument("--confirm-saved", action="store_true", help="Confirm all session work is saved/retrieved")
    sub.add_parser("files", help="Preview source files that will be synchronized (no network)")
    dat = sub.add_parser("data", help="Push local dataset, restore/backup Drive archive")
    dat.add_argument("action", choices=["push", "restore", "backup"])
    dat.add_argument("--force", action="store_true")
    dat.add_argument("--allow-large", action="store_true")
    r = sub.add_parser("run", help="Choose local/Colab backend; bounded experiment with logs/artifacts")
    r.add_argument("--backend", choices=["auto", "local", "colab"])
    r.add_argument("--task", choices=["verify", "benchmark", "dataset-check", "layout", "environment"], default="verify")
    r.add_argument("--estimated-minutes", type=float)
    r.add_argument("--required-vram-gb", type=float)
    r.add_argument("--approve-remote", action="store_true")
    r.add_argument("--approve-expensive", action="store_true")
    r.add_argument("--budget-seconds", type=int)
    r.add_argument("--output-root")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--workers", type=int)
    r.add_argument("--batch-size", type=int)
    r.add_argument("--no-setup", action="store_true")
    r.add_argument("--no-fetch", action="store_true")
    f = sub.add_parser("fetch", help="Recover results for a previous run ID")
    f.add_argument("--run-id", required=True)
    args, extra = parser.parse_known_args()
    if args.command != "run" and extra:
        parser.error("Unrecognized arguments: " + " ".join(extra))
    cfg = config()
    if args.command == "doctor":
        print("Project:", ROOT)
        print("Session:", cfg["session"], "Remote:", cfg["remote_root"])
        print("Colab binary:", shutil.which("colab") or str(Path.home() / ".local/bin/colab"))
        print("Requirements:", (ROOT / cfg.get("requirements", "requirements.txt")).exists())
        print("Entry command:", cfg.get("experiment_command"))
        print("Local policy:", json.dumps(choose_backend(cfg, "auto")))
        print("Source files to sync:", len(get_file_list()))
        if shutil.which("colab") or (Path.home() / ".local/bin/colab").exists():
            print(cli(cfg, "status", "-s", cfg["session"]), end="")
    elif args.command == "files":
        for path in get_file_list():
            print(f"{path.stat().st_size:>10}  {path.relative_to(ROOT).as_posix()}")
    elif args.command == "status":
        print(cli(cfg, "status", "-s", cfg["session"]), end="")
    elif args.command == "sync": sync(cfg)
    elif args.command == "setup": sync(cfg); setup(cfg)
    elif args.command == "data": data_action(cfg, args.action, args.force, args.allow_large)
    elif args.command == "run":
        if extra and extra[0] == "--": extra = extra[1:]
        run(cfg, args, extra)
    elif args.command == "fetch":
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", args.run_id):
            raise ValueError("Invalid run ID")
        fetch(cfg, args.run_id)
        (ROOT / cfg.get("output_root", "runs") / args.run_id / "FETCHED").write_text(args.run_id)
    elif args.command == "stop":
        if not args.confirm_saved:
            raise RuntimeError("Inspect/fetch all session outputs first; stop requires --confirm-saved and user authorization")
        status = cli(cfg, "status", "-s", cfg["session"])
        if not re.search(r"Status: IDLE\b", status):
            raise RuntimeError("Refusing to stop a BUSY or unknown session")
        check = payload_code({"root": cfg["remote_root"]}, '''
from pathlib import Path
root=Path(p["root"])
unfinished=list(root.glob("runs/*/RUNNING"))
if unfinished: raise RuntimeError("Unfinished run markers present; inspect and save work manually")
print("__ELEC475_RESULT__="+json.dumps({"runs":[x.name for x in (root/"runs").iterdir() if x.is_dir()] if (root/"runs").exists() else []}))
''')
        result = marked_result(remote(cfg, check))
        for run_id in result["runs"]:
            if not (ROOT / cfg.get("output_root", "runs") / run_id / "FETCHED").exists():
                raise RuntimeError(f"Remote run {run_id} not confirmed fetched; fetch before stopping")
        cli(cfg, "stop", "-s", cfg["session"], capture=False)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
