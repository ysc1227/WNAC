#!/usr/bin/env python3
"""High-VRAM/high-load GPU NaN stress test.

Examples:
  CUDA_VISIBLE_DEVICES=0 python tools/gpu_nan_stress.py --mode single --duration 180
  CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 tools/gpu_nan_stress.py --mode ddp --duration 180
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single", "ddp"], required=True)
    p.add_argument("--duration", type=int, default=180)
    p.add_argument("--target-frac", type=float, default=0.70, help="VRAM fraction to occupy on the tested GPU")
    p.add_argument("--rank0-target-frac", type=float, default=0.70, help="DDP rank0/GPU0 VRAM fraction")
    p.add_argument("--other-rank-target-frac", type=float, default=0.20, help="DDP non-rank0 VRAM fraction")
    p.add_argument("--work-n", type=int, default=8192)
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--smi-interval", type=float, default=10.0)
    p.add_argument("--chunk-gib", type=float, default=1.0)
    return p.parse_args()


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.f = path.open("a", buffering=1)

    def log(self, *xs):
        print(now(), *xs, file=self.f, flush=True)

    def close(self):
        self.f.close()


def run_cmd(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"CMD_ERROR {cmd!r}: {e!r}"


def smi() -> str:
    return run_cmd([
        "nvidia-smi",
        "--query-gpu=index,pci.bus_id,uuid,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ])


def kernel_log_tail() -> str:
    # sudo 없이 접근 가능한 journalctl -k 우선, 실패 시 dmesg 시도.
    cmd = "(journalctl -k --no-pager 2>/dev/null || dmesg 2>/dev/null || true) | grep -Ei 'NVRM|Xid|nvidia|pcie|aer|ecc|thermal' | tail -n 120"
    return run_cmd(["bash", "-lc", cmd])


def init_dist_if_needed(mode: str, logger: Logger) -> tuple[int, int, int]:
    if mode != "ddp":
        return 0, 1, 0
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    logger.log("DDP_INIT", "rank", rank, "world", world, "local_rank", local_rank)
    return rank, world, local_rank


def allocate_ballast(device: torch.device, target_frac: float, chunk_gib: float, logger: Logger):
    props = torch.cuda.get_device_properties(device)
    total_gib = props.total_memory / 1024**3
    target_gib = total_gib * target_frac
    ballast = []
    allocated = 0.0
    logger.log("ALLOC_START", "total_gib", total_gib, "target_frac", target_frac, "target_gib", target_gib)
    while allocated + chunk_gib <= target_gib:
        elems = int(chunk_gib * 1024**3 / 2)  # fp16
        t = torch.empty((elems,), device=device, dtype=torch.float16)
        t.fill_(0.123)
        ballast.append(t)
        allocated += chunk_gib
        if int(allocated * 10) % 50 == 0:
            torch.cuda.synchronize(device)
            logger.log("ALLOC_PROGRESS", "allocated_gib", round(allocated, 3), "mem_alloc_gib", round(torch.cuda.memory_allocated(device)/1024**3, 3))
    rem = target_gib - allocated
    if rem > 0.10:
        elems = int(rem * 1024**3 / 2)
        t = torch.empty((elems,), device=device, dtype=torch.float16)
        t.fill_(0.123)
        ballast.append(t)
        allocated += rem
    torch.cuda.synchronize(device)
    logger.log("ALLOC_DONE", "allocated_gib", round(allocated, 3), "mem_alloc_gib", round(torch.cuda.memory_allocated(device)/1024**3, 3))
    logger.log("SMI_AFTER_ALLOC\n" + smi())
    return ballast


def stress_loop(args: argparse.Namespace, logger: Logger, rank: int, world: int, local_rank: int) -> str:
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    target_frac = args.target_frac
    if args.mode == "ddp":
        target_frac = args.rank0_target_frac if rank == 0 else args.other_rank_target_frac

    logger.log("ENV", "host", socket.gethostname(), "pid", os.getpid(), "visible", os.environ.get("CUDA_VISIBLE_DEVICES"), "rank", rank, "world", world, "local_rank", local_rank)
    logger.log("TORCH", torch.__version__, "torch_cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available(), "device_name", torch.cuda.get_device_name(device))
    logger.log("SMI_START\n" + smi())

    ballast = allocate_ballast(device, target_frac, args.chunk_gib, logger)
    torch.manual_seed(20260604 + rank)
    n = args.work_n
    logger.log("WORK_ALLOC_START", "work_n", n)
    a = torch.randn((n, n), device=device, dtype=torch.float16) / 64
    b = torch.randn((n, n), device=device, dtype=torch.float16) / 64
    ddp_buf = torch.ones((1024,), device=device, dtype=torch.float32) * (rank + 1)
    torch.cuda.synchronize(device)
    logger.log("WORK_ALLOC_DONE", "mem_alloc_gib", round(torch.cuda.memory_allocated(device)/1024**3, 3), "SMI\n" + smi())

    dist = None
    if args.mode == "ddp":
        import torch.distributed as dist_mod
        dist = dist_mod
        dist.barrier()

    start = time.time()
    last_smi = 0.0
    it = 0
    result = "OK"
    try:
        while time.time() - start < args.duration:
            c = a @ b
            a = torch.tanh(c).contiguous()
            idx = it % len(ballast)
            # 실제 VRAM path를 살짝 건드림. 큰 비용 없이 일부/전체 스칼라 연산.
            ballast[idx].mul_(1.0).add_(0.0)
            b = torch.roll(b, shifts=1, dims=0)
            if dist is not None:
                ddp_buf.fill_(rank + 1 + it * 1e-7)
                dist.all_reduce(ddp_buf)
                if not torch.isfinite(ddp_buf).all().item():
                    result = "BAD_DDP_BUF_NAN_INF"
                    logger.log(result, "iter", it, "ddp_buf0", float(ddp_buf[0].detach().cpu()))
                    break
            finite_a = torch.isfinite(a).all()
            finite_ballast = torch.isfinite(ballast[idx][: min(ballast[idx].numel(), 1024 * 1024)]).all()
            mean = a.float().mean()
            maxabs = a.float().abs().max()
            torch.cuda.synchronize(device)
            if (not bool(finite_a.item())) or (not bool(finite_ballast.item())) or (not torch.isfinite(mean).item()) or (not torch.isfinite(maxabs).item()):
                result = "BAD_NAN_INF"
                logger.log(result, "iter", it, "elapsed", round(time.time()-start, 3), "finite_a", bool(finite_a.item()), "finite_ballast_sample", bool(finite_ballast.item()), "mean", float(mean.detach().cpu()) if torch.isfinite(mean).item() else str(mean.detach().cpu()), "maxabs", float(maxabs.detach().cpu()) if torch.isfinite(maxabs).item() else str(maxabs.detach().cpu()))
                break
            elapsed = time.time() - start
            if elapsed - last_smi >= args.smi_interval:
                logger.log("STAT", "iter", it, "elapsed", round(elapsed, 1), "mean", float(mean.detach().cpu()), "maxabs", float(maxabs.detach().cpu()), "mem_alloc_gib", round(torch.cuda.memory_allocated(device)/1024**3, 3))
                logger.log("SMI\n" + smi())
                last_smi = elapsed
            it += 1
    except Exception as e:
        result = "EXCEPTION"
        logger.log("EXCEPTION", repr(e))
        raise
    finally:
        logger.log("RESULT", result, "iters", it, "elapsed", round(time.time()-start, 3), "rank", rank)
        logger.log("KERNEL_LOG_TAIL\n" + kernel_log_tail())
        del a, b, ddp_buf, ballast
        gc.collect()
        torch.cuda.empty_cache()
        logger.log("CLEANUP_SMI\n" + smi())
    return result


def main() -> int:
    args = parse_args()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    rank_env = int(os.environ.get("RANK", "0"))
    tag = f"_{args.tag}" if args.tag else ""
    log_path = Path(args.log_dir) / f"gpu_nan_stress_{args.mode}{tag}_{stamp}_rank{rank_env}.log"
    logger = Logger(log_path)
    try:
        rank, world, local_rank = init_dist_if_needed(args.mode, logger)
        result = stress_loop(args, logger, rank, world, local_rank)
        if args.mode == "ddp":
            import torch.distributed as dist
            code = torch.tensor([0 if result == "OK" else 1], device=f"cuda:{local_rank}", dtype=torch.int32)
            dist.all_reduce(code, op=dist.ReduceOp.SUM)
            logger.log("DDP_RESULT_SUM", int(code.item()))
            dist.barrier()
            dist.destroy_process_group()
            return int(code.item() != 0)
        return 0 if result == "OK" else 1
    finally:
        logger.log("LOG_PATH", str(log_path))
        logger.close()


if __name__ == "__main__":
    sys.exit(main())