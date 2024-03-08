import random
import os

import pytest
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from syncbn import SyncBatchNorm


def _compute_loss(outputs, global_batch_size, local_rank=None):
    if local_rank is None:
        return outputs[:global_batch_size // 2].sum()

    local_batch_size = outputs.shape[0]
    global_center_idx = global_batch_size // 2 - 1
    batch_start_global_idx = local_batch_size * local_rank
    batch_end_global_idx = local_batch_size * (local_rank + 1) - 1
    if batch_start_global_idx > global_center_idx:
        return (outputs * 0).sum()
    if batch_end_global_idx <= global_center_idx:
        return outputs.sum()
    return outputs[:global_center_idx % local_batch_size + 1].sum()


def init_sbn_process(world_size, local_rank, inputs):
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8972'
    dist.init_process_group('gloo', rank=local_rank, world_size=world_size)

    batch_size = inputs.shape[1]
    hid_dim = inputs.shape[2]
    sbn = SyncBatchNorm(hid_dim, eps=1e-5, momentum=0.1)
    sbn_input = inputs.detach()[local_rank]
    sbn_input.requires_grad_()
    normalized = sbn(sbn_input)
    _compute_loss(normalized, world_size * batch_size, local_rank).backward()

    return normalized.detach(), sbn_input.grad


def _get_input(num_workers, batch_size, hid_dim):
    input = torch.randn([num_workers, batch_size, hid_dim])

    bn_input = input.clone().flatten(0, 1)
    bn_input.requires_grad_()

    sbn_input = input.clone()

    return bn_input, sbn_input


def _get_sbn_output(input, num_workers, hid_dim, batch_size):
    ctx = mp.get_context('spawn')
    with ctx.Pool(num_workers) as pool:
        args = [(num_workers, rank, input) for rank in range(num_workers)]
        results = pool.starmap(init_sbn_process, args)

    sbn_output = torch.vstack([res[0] for res in results])
    sbn_grad = torch.vstack([res[1] for res in results])

    return sbn_output, sbn_grad


@pytest.mark.parametrize(
    'seed,num_workers,batch_size,hid_dim',
    [
        (42, 4, 64, 128),
        (52, 4, 64, 256),
        (53, 1, 32, 512),
        (54, 1, 32, 1024),
    ]
)
def test_batchnorm(seed, num_workers, hid_dim, batch_size):
    # Verify that the implementation of SyncBatchNorm gives the same results (both for outputs
    # and gradients with respect to input) as torch.nn.BatchNorm1d on a variety of inputs.

    # This can help you set up the worker processes. Child processes launched with `spawn` can still run
    # torch.distributed primitives, but you can also communicate their outputs back to the main process to compare them
    # with the outputs of a non-synchronous BatchNorm.
    torch.manual_seed(seed)

    bn_input, sbn_input = _get_input(num_workers, batch_size, hid_dim)

    bn = torch.nn.BatchNorm1d(hid_dim, affine=False, eps=1e-5, momentum=0.1)
    bn_output = bn(bn_input)
    _compute_loss(bn_output, num_workers * batch_size).backward()

    sbn_output, sbn_grad = _get_sbn_output(sbn_input, num_workers, hid_dim, batch_size)

    assert torch.allclose(sbn_output, bn_output, atol=1e-4, rtol=0)
    assert torch.allclose(bn_input.grad, sbn_grad, atol=1e-4, rtol=0)
