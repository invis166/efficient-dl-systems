import os
from itertools import product

import torch
from torch import nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.benchmark as benchmark

from syncbn import SyncBatchNorm


def run_sbn(world_size, local_rank, hid_dim, batch_size, use_pytorch):
    torch.manual_seed(42)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda')

    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8972'
    dist.init_process_group('nccl', rank=local_rank, world_size=world_size)

    if use_pytorch:
        sbn = nn.SyncBatchNorm(hid_dim, affine=False)
    else:
        sbn = SyncBatchNorm(hid_dim)

    sbn = sbn.to(device)
    input = torch.randn((batch_size, hid_dim), dtype=torch.float32, device=device)
    # warmup run
    sbn(input)

    t = benchmark.Timer(
        stmt='sbn(input)',
        globals={'input': input, 'sbn': sbn},
    )

    return t.timeit(10000).median, torch.cuda.max_memory_allocated()


def profile_sbn(batch_size, hidden_dim, num_workers, use_pytorch):
    ctx = mp.get_context('spawn')
    with ctx.Pool(num_workers) as pool:
        args = [(num_workers, rank, hidden_dim, batch_size, use_pytorch) for rank in range(num_workers)]
        prof_results = pool.starmap(run_sbn, args)

    median_time, max_memory_allocated = prof_results[0]
    print(
        f'{batch_size=}, {hidden_dim=}, {num_workers=}, {use_pytorch=} '
        f'| median time: {median_time:.5f}, max memory allocated: {max_memory_allocated}'
    )


if __name__ == '__main__':
    batch_size_range = [32, 64]
    hidden_size_range = [128, 256, 512, 1024]
    use_pytorch_options = [True, False]

    for batch_size, hidden_size, use_pytorch, in product(batch_size_range, hidden_size_range, use_pytorch_options):
        profile_sbn(batch_size=batch_size, hidden_dim=hidden_size, num_workers=2, use_pytorch=use_pytorch)
