import os
from datetime import timedelta
from typing import Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


DEFAULT_DDP_TIMEOUT_MINUTES = 60


def get_ddp_timeout() -> timedelta:
    timeout_minutes = int(os.environ.get("DDP_TIMEOUT_MINUTES", str(DEFAULT_DDP_TIMEOUT_MINUTES)))
    return timedelta(minutes=timeout_minutes)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def setup_runtime(device_arg: str, distributed: bool) -> Tuple[torch.device, bool, int, int, int]:
    wants_distributed = distributed or int(os.environ.get("WORLD_SIZE", "1")) > 1
    if not wants_distributed:
        return resolve_device(device_arg), False, 0, 1, 0

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA.")

    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "Distributed training must be launched with torchrun so that "
            f"{', '.join(missing)} are defined."
        )

    requested_device = resolve_device(device_arg)
    if requested_device.type != "cuda":
        raise RuntimeError("Distributed training is supported only with CUDA devices.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=get_ddp_timeout())
    device = torch.device("cuda", local_rank)
    return device, True, rank, world_size, local_rank


def maybe_parallelize(
    model: nn.Module,
    *,
    device: torch.device,
    use_dataparallel: bool,
    use_distributed: bool,
) -> nn.Module:
    if use_distributed:
        return DistributedDataParallel(model, device_ids=[device.index], output_device=device.index)
    if use_dataparallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"DataParallel enabled across {torch.cuda.device_count()} GPUs.")
        return nn.DataParallel(model)
    return model


def unwrap_state_dict(model: nn.Module):
    if isinstance(model, (nn.DataParallel, DistributedDataParallel)):
        return model.module.state_dict()
    return model.state_dict()


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_sum_and_count(
    total_value: float,
    total_count: int,
    *,
    device: torch.device,
    distributed: bool,
) -> Tuple[float, int]:
    if not distributed:
        return total_value, total_count

    stats = torch.tensor([total_value, float(total_count)], dtype=torch.float64, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item()), int(stats[1].item())


def wait_for_all_ranks(distributed: bool) -> None:
    if distributed:
        dist.barrier()


def cleanup_runtime(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()
