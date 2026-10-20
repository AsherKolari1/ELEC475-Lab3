"""Bounded pre-lab experiments; reuses instructor models and preprocessing."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temp.replace(path)


def hardware():
    import torch
    result = {'python': sys.version, 'platform': platform.platform(),
              'cpu_count': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count(),
              'torch': torch.__version__, 'cuda_build': torch.version.cuda,
              'cuda_available': torch.cuda.is_available()}
    if result['cuda_available']:
        free, total = torch.cuda.mem_get_info()
        result.update(gpu=torch.cuda.get_device_name(0), vram_bytes=total, free_vram_bytes=free)
    try:
        result['ram_available_bytes'] = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) * 1024
    except (OSError, StopIteration):
        pass
    return result


def dataset_layout(root):
    from utils.dataset import PetNoseDataset
    result = {}
    for split in ('train', 'test'):
        dataset = PetNoseDataset(root=str(root), split=split)
        missing = [name for name, _, _ in dataset.samples if not (dataset.img_dir / name).is_file()]
        image, label, size = dataset[0]
        result[split] = {'samples': len(dataset), 'missing_images': len(missing),
                         'first_image': dataset.samples[0][0], 'shape': list(image.shape),
                         'original_size': size.tolist(), 'label': label.tolist()}
        if missing:
            raise RuntimeError(f'{split}: missing {len(missing)} images, e.g. {missing[:3]}')
    result['jpeg_count'] = sum(1 for _ in (Path(root) / 'images-original/images').glob('*.jpg'))
    return result


def seed_worker(_):
    import numpy as np
    import torch
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def benchmark(args, out):
    import gc
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from models import get_model
    from utils.dataset import PetNoseDataset

    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = PetNoseDataset(root=args.dataset_root, split='train')
    count = (args.warmup + args.steps) * args.batch_size
    if count > len(dataset):
        raise ValueError('Requested benchmark exceeds one dataset pass; reduce steps/batch size')
    dataset = Subset(dataset, list(range(count)))
    model = get_model(args.model, pretrained=False).eval()
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.to(device)
    if args.save_checkpoint:
        checkpoint = out / 'checkpoints/benchmark_initial_state.pth'
        checkpoint.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), checkpoint)
    rows = []
    cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    workers_grid = args.workers_grid if args.workers_grid is not None else [args.num_workers]
    for workers in workers_grid:
        row = {'workers': workers, 'batch_size': args.batch_size,
               'pin_memory': args.pin_memory and device.type == 'cuda',
               'persistent_workers': args.persistent_workers and workers > 0,
               'prefetch_factor': args.prefetch_factor if workers > 0 else None,
               'device': str(device), 'status': 'running'}
        if workers > cpu_count and not args.allow_oversubscription:
            row.update(status='skipped', reason=f'{workers} workers exceeds {cpu_count} available CPUs')
            rows.append(row)
            write_json(out / 'metrics.json', rows)
            continue
        # Conservative budget for queued float32 RGB batches; actual decode/RAM can be higher.
        queue_bytes = args.batch_size * 3 * 227 * 227 * 4 * (1 + workers * args.prefetch_factor)
        worker_ram = workers * args.worker_ram_mb * 1024**2
        row['estimated_loader_ram_bytes'] = queue_bytes + worker_ram
        if queue_bytes + worker_ram > hardware().get('ram_available_bytes', 2**63) * 0.6:
            row.update(status='skipped', reason='estimated worker processes plus queued tensors exceed 60% of available RAM')
            rows.append(row)
            write_json(out / 'metrics.json', rows)
            continue
        generator = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=workers, pin_memory=row['pin_memory'],
                            persistent_workers=row['persistent_workers'],
                            prefetch_factor=row['prefetch_factor'],
                            worker_init_fn=seed_worker, generator=generator)
        iterator = None
        try:
            start = time.perf_counter()
            iterator = iter(loader)
            iterator_start_s = time.perf_counter() - start
            waits, computes, transfers, walls = [], [], [], []
            checksum = 0.0
            def sync():
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
            with torch.inference_mode():
                for index in range(args.warmup + args.steps):
                    sync()
                    if index == args.warmup and device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(device)
                    begin = time.perf_counter()
                    images, _, _ = next(iterator)
                    loaded = time.perf_counter()
                    images = images.to(device, non_blocking=row['pin_memory'])
                    sync()
                    transferred = time.perf_counter()
                    prediction = model(images)
                    sync()
                    computed = time.perf_counter()
                    if index >= args.warmup:
                        waits.append(loaded - begin)
                        transfers.append(transferred - loaded)
                        computes.append(computed - transferred)
                        walls.append(computed - begin)
                        checksum += prediction.sum().item()
            row.update(status='ok', steps=args.steps, samples=args.steps * args.batch_size,
                       data_wait_s=sum(waits), transfer_s=sum(transfers), compute_s=sum(computes),
                       data_wait_ms_per_batch=1000 * sum(waits) / args.steps,
                       compute_ms_per_batch=1000 * sum(computes) / args.steps,
                       samples_per_second=args.steps * args.batch_size / sum(walls),
                       measured_wall_s=sum(walls), iterator_start_s=iterator_start_s,
                       peak_vram_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                       peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0,
                       host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       children_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                       prediction_checksum=checksum)
        except Exception as exc:
            row.update(status='failed', error=str(exc))
            traceback.print_exc()
        finally:
            del iterator, loader
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        rows.append(row)
        write_json(out / 'metrics.json', rows)
        print(json.dumps(row), flush=True)
    keys = sorted({key for row in rows for key in row})
    with (out / 'benchmark.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    valid = [row for row in rows if row['status'] == 'ok']
    write_json(out / 'recommendation.json', {
        'best_measured': max(valid, key=lambda row: row['samples_per_second']) if valid else None,
        'scope': 'Inference only; fixed-order warm-cache short benchmark. Repeat and vary batch/pinning/persistence separately before changing defaults.',
        'checkpoint': str(checkpoint) if checkpoint else None,
        'weights': 'supplied' if args.checkpoint else 'random initialization; not accuracy evaluation'})
    if any(row['status'] == 'failed' for row in rows) or not valid:
        raise RuntimeError('Benchmark incomplete; inspect metrics.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['verify', 'benchmark', 'dataset-check', 'layout', 'hardware', 'environment'], default='verify')
    parser.add_argument('--entry-command', default=json.dumps(['python', '-m', 'models.snoutnet']))
    parser.add_argument('--output-dir', default=os.getenv('ELEC475_OUTPUT_DIR', 'runs/manual'))
    parser.add_argument('--dataset-root', default=os.getenv('ELEC475_DATASET_ROOT', 'data/oxford-iiit-pet-noses'))
    parser.add_argument('--num-workers', type=int, default=int(os.getenv('ELEC475_NUM_WORKERS', '0')))
    parser.add_argument('--workers-grid', nargs='+', type=int)
    parser.add_argument('--batch-size', type=int, default=int(os.getenv('ELEC475_BATCH_SIZE', '4')))
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--seed', type=int, default=475)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--model', choices=['snoutnet', 'alexnet', 'vgg16'], default='snoutnet')
    parser.add_argument('--checkpoint')
    parser.add_argument('--save-checkpoint', action='store_true', help='Save benchmark model state (not a trained checkpoint)')
    parser.add_argument('--pin-memory', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--persistent-workers', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--allow-oversubscription', action='store_true')
    parser.add_argument('--worker-ram-mb', type=int, default=384,
                        help='Conservative per-worker RAM allowance in addition to queued tensors')
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1 or args.batch_size < 1 or args.num_workers < 0 or args.cpu_threads < 1 or args.prefetch_factor < 1 or args.worker_ram_mb < 1 or any(w < 0 for w in args.workers_grid or []):
        parser.error('steps, warmup, batch size, threads and prefetch must be positive; workers nonnegative')
    if args.task == 'hardware':
        print(json.dumps(hardware()))
        return
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    settings.update(hardware=hardware(), backend=os.getenv('ELEC475_BACKEND', 'local'),
                    backend_reason=os.getenv('ELEC475_BACKEND_REASON', 'direct invocation'),
                    checkpoint_manifest={str(path): {'bytes': path.stat().st_size} for path in (ROOT / 'checkpoints').glob('*.pth')})
    write_json(out / 'settings.json', settings)
    start = time.perf_counter()
    status = 'failed'
    try:
        if args.task == 'benchmark':
            benchmark(args, out)
        elif args.task == 'environment':
            import importlib.metadata
            packages = {name: importlib.metadata.version(name) for name in
                        ('torch', 'torchvision', 'numpy', 'Pillow', 'matplotlib', 'torchinfo', 'tqdm')}
            write_json(out / 'metrics.json', {'hardware': hardware(), 'dependencies': packages})
            print((out / 'metrics.json').read_text())
        elif args.task == 'layout':
            write_json(out / 'metrics.json', dataset_layout(args.dataset_root))
            print((out / 'metrics.json').read_text())
        else:
            command = json.loads(args.entry_command) if args.task == 'verify' else ['python', '-m', 'utils.check_dataset']
            command[0] = sys.executable
            env = dict(os.environ, ELEC475_NUM_WORKERS=str(args.num_workers),
                       ELEC475_BATCH_SIZE=str(args.batch_size), ELEC475_OUTPUT_DIR=str(out),
                       ELEC475_DATASET_ROOT=args.dataset_root, OMP_NUM_THREADS=str(args.cpu_threads),
                       MKL_NUM_THREADS=str(args.cpu_threads))
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            if not (out / 'metrics.json').exists():
                write_json(out / 'metrics.json', {'task': args.task, 'passed': True})
        status = 'completed'
    except Exception as exc:
        if not (out / 'metrics.json').exists():
            write_json(out / 'metrics.json', {'status': 'failed', 'error': str(exc)})
        raise
    finally:
        write_json(out / 'experiment_status.json', {'status': status, 'duration_s': time.perf_counter() - start})


if __name__ == '__main__':
    main()
