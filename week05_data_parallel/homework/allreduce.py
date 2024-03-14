import resource
import os
import random

import torch
import torch.distributed as dist
import time
import torch.multiprocessing as mp



def init_process(rank, world_size, tensor_size, fn, master_port, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    start_time = time.perf_counter()
    initial_tensor, reduced = fn(rank, world_size, tensor_size)
    end_time = time.perf_counter()
    max_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return initial_tensor, reduced, end_time - start_time, max_memory_usage


def butterfly_allreduce(send, rank, size):
    """
    Performs Butterfly All-Reduce over the process group. Modifies the input tensor in place.
    Args:
        send: torch.Tensor to be averaged with other processes.
        rank: Current process rank (in a range from 0 to size)
        size: Number of workers
    """

    buffer_for_chunk = torch.empty((size,), dtype=torch.float)

    send_futures = []

    for i, elem in enumerate(send):
        if i != rank:
            send_futures.append(dist.isend(elem, i))

    recv_futures = []

    for i, elem in enumerate(buffer_for_chunk):
        if i != rank:
            recv_futures.append(dist.irecv(elem, i))
        else:
            elem.copy_(send[i])

    for future in recv_futures:
        future.wait()

    # compute the average
    torch.mean(buffer_for_chunk, dim=0, out=send[rank])

    for i in range(size):
        if i != rank:
            send_futures.append(dist.isend(send[rank], i))

    recv_futures = []

    for i, elem in enumerate(send):
        if i != rank:
            recv_futures.append(dist.irecv(elem, i))

    for future in recv_futures:
        future.wait()
    for future in send_futures:
        future.wait()


def ring_allreduce(send, rank, size):
    """
    Performs Ring All-Reduce over the process group. Modifies the input tensor in place.
    Args:
        send: torch.Tensor to be averaged with other processes.
        rank: Current process rank (in a range from 0 to size)
        size: Number of workers
    """
    chunks = list(torch.chunk(send, size))
    recv_buffer = torch.empty((max(map(len, chunks)),), dtype=torch.float)

    send_chunk = rank
    recv_chunk = (send_chunk - 1) % size
    for round in range(size - 1):
        send_future = dist.isend(chunks[send_chunk], (rank + 1) % size)
        recv_future = dist.irecv(recv_buffer, (rank - 1) % size)
        recv_future.wait()
        send_future.wait()
        chunks[recv_chunk] += recv_buffer

        send_chunk, recv_chunk = recv_chunk, (recv_chunk - 1) % size

    send_chunk = (rank + 1) % size
    recv_chunk = (send_chunk - 1) % size
    for round in range(size - 1):
        send_future = dist.isend(chunks[send_chunk], (rank + 1) % size)
        recv_future = dist.irecv(recv_buffer, (rank - 1) % size)
        recv_future.wait()
        send_future.wait()
        chunks[recv_chunk].copy_(recv_buffer)

        send_chunk, recv_chunk = recv_chunk, (recv_chunk - 1) % size

    send.copy_(torch.hstack(chunks) / size)


def run_butterfly_allreduce(rank, world_size, tensor_size):
    """Simple point-to-point communication."""
    torch.manual_seed(rank)
    tensor = torch.randn((tensor_size,), dtype=torch.float)
    initial_tensor = torch.clone(tensor)
    butterfly_allreduce(tensor, rank, world_size)

    return initial_tensor, tensor


def run_ring_allreduce(rank, world_size, tensor_size):
    """Simple point-to-point communication."""
    torch.manual_seed(rank)
    tensor = torch.randn((tensor_size,), dtype=torch.float)
    initial_tensor = torch.clone(tensor)
    ring_allreduce(tensor, rank, world_size)

    return initial_tensor, tensor


def run_torch_allreduce(rank, world_size, tensor_size):
    """Simple point-to-point communication."""
    torch.manual_seed(rank)
    tensor = torch.randn((tensor_size,), dtype=torch.float)
    initial_tensor = torch.clone(tensor)
    dist.all_reduce(tensor, dist.ReduceOp.SUM)

    return initial_tensor, tensor / world_size


def test_butterfly():
    world_sizes = [1, 2, 4, 8, 16, 32]
    tensor_sizes = [4, 8, 16, 64, 128, 512]

    for world_size, tensor_size in zip(world_sizes, tensor_sizes):
        print(f'testing {world_size=} {tensor_size=}')
        gt, reduced, *_ = _run_allreduce(world_size, tensor_size, run_ring_allreduce)

        assert torch.allclose(gt, reduced, rtol=1e-4)


def _run_allreduce(world_size, tensor_size, fn):
    port = random.randint(25000, 30000)
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=world_size) as pool:
        results = pool.starmap(
            init_process,
            [(rank, world_size, tensor_size, fn, port) for rank in range(world_size)]
        )

    gt = torch.mean(torch.vstack([res[0] for res in results]), axis=0)
    reduced_tensor = results[0][1]
    max_time_consumed = max(res[2] for res in results)
    max_memory_usage = max(res[3] for res in results)

    return gt, reduced_tensor, max_time_consumed, max_memory_usage


def _run_benchmark(world_size, tensor_size, fn, n=10):
    mean_max_time = 0
    mean_max_memory = 0
    for _ in range(n):
        gt, reduced, max_time_consumed, max_memory = _run_allreduce(world_size, tensor_size, fn)
        mean_max_time += max_time_consumed
        mean_max_memory += max_memory

    return gt, reduced, mean_max_time / n, mean_max_memory / n


def compare_custom():
    world_sizes = [1, 2, 4, 8, 16, 32]
    tensor_sizes = [1, 2, 4, 8, 16, 32]

    for world_size, tensor_size in zip(world_sizes, tensor_sizes):
        gt1, ring_reduced, ring_max_time_consumed, ring_max_memory = _run_benchmark(world_size, tensor_size, run_ring_allreduce)
        gt2, butterfly_reduced, butterfly_max_time_consumed, butterfly_max_memory = _run_benchmark(world_size, tensor_size, run_butterfly_allreduce)
        diff_ring = torch.sum((ring_reduced - gt1)**2)
        diff_butterfly = torch.sum((butterfly_reduced - gt2)**2)
        print(f'{world_size=} {tensor_size=} | {ring_max_time_consumed=:.3f}, {butterfly_max_time_consumed=:.3f}, {ring_max_memory=}, {butterfly_max_memory=} {diff_ring=}, {diff_butterfly=}')


def compare_with_torch():
    world_sizes = [1, 2, 4, 8, 16]
    tensor_sizes = [1024, 2048, 4096, 16384, 131072]

    for world_size, tensor_size in zip(world_sizes, tensor_sizes):
        gt1, ring_reduced, ring_max_time_consumed, ring_max_memory = _run_benchmark(world_size, tensor_size, run_ring_allreduce)
        gt2, torch_reduced, torch_max_time_consumed, torch_max_memory = _run_benchmark(world_size, tensor_size, run_torch_allreduce)
        diff_ring = torch.sum((ring_reduced - gt1)**2)
        diff_torch = torch.sum((torch_reduced - gt2)**2)
        print(f'{world_size=} {tensor_size=} | {ring_max_time_consumed=:.3f}, {torch_max_time_consumed=:.3f}, {ring_max_memory=}, {torch_max_memory=} {diff_ring=}, {diff_torch=}')


if __name__ == "__main__":
    test_butterfly()
    compare_custom()
    compare_with_torch()
