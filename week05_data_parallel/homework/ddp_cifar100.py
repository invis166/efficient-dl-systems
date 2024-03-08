import time
import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR100

from syncbn import SyncBatchNorm

torch.set_num_threads(1)


EVAL_RATE = 16


def init_process(local_rank, fn, backend="nccl", **kwargs):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    start_time = time.perf_counter()
    fn(local_rank, size, **kwargs)
    end_time = time.perf_counter()

    print(f'rank {local_rank} | time: {end_time - start_time} | max memory allocated: {torch.cuda.max_memory_allocated()}')


class Net(nn.Module):
    """
    A very simple model with minimal changes from the tutorial, used for the sake of simplicity.
    Feel free to replace it with EffNetV2-XL once you get comfortable injecting SyncBN into models programmatically.
    """

    def __init__(self, impl):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 100)
        if impl == 'pytorch':
            self.bn1 = nn.SyncBatchNorm(128, affine=False)
        else:
            self.bn1 = SyncBatchNorm(128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def _train_model(model: nn.Module, loader, device, grad_accum_steps, distributed_impl, rank, num_epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = len(loader)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = torch.zeros((1,), device=device)
        for step, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            epoch_loss += loss.detach()
            loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                if distributed_impl == 'selfmade':
                    average_gradients(model)
                optimizer.step()
                optimizer.zero_grad()
            if step % EVAL_RATE == 0 and rank == 0:
                acc = (output.argmax(dim=1) == target).float().mean()
                print(f"loss: {epoch_loss / num_batches}, acc: {acc}")


def _eval_model(model: nn.Module, val_loader, device, rank):
    # torch.utils.data.DistributedSampler already restricts the val data
    # to a specific subset based on the process rank, so there is no need in torch.distributed.scatter
    model.eval()
    correct = torch.tensor(0, device=device)
    for data, target in val_loader:
        data = data.to(device)
        target = target.to(device)
        pred = model(data)
        correct += (target == pred.argmax(1)).sum()

    dist.all_reduce(correct, dist.ReduceOp.SUM)

    return correct


def run_training(rank, size, num_epochs, grad_accum_steps, distributed_impl):
    torch.manual_seed(0)

    train_dataset = CIFAR100(
        "./cifar",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        ),
        download=True,
        train=True,
    )
    val_dataset = CIFAR100(
        "./cifar",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        ),
        download=True,
        train=False,
    )
    train_loader = DataLoader(
        train_dataset,
        sampler=DistributedSampler(train_dataset, size, rank),
        batch_size=256,
        pin_memory=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=DistributedSampler(val_dataset, size, rank),
        batch_size=256,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
    )

    model = Net(distributed_impl)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if distributed_impl == 'pytorch':
        model = DistributedDataParallel(model, device_ids=[rank])

    _train_model(model, train_loader, device, grad_accum_steps, distributed_impl, rank, num_epochs)
    n_correct = _eval_model(model, val_loader, device, rank)
    if rank == 0:
        print(f'val acc: {n_correct / len(val_dataset)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--distributed-impl', choices=['pytorch', 'selfmade'])
    parser.add_argument('--num-epochs', default=10)
    parser.add_argument('--grad-accum-steps', default=3)

    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    init_process(
        local_rank,
        fn=run_training,
        backend="nccl",
        num_epochs=int(args.num_epochs),
        grad_accum_steps=int(args.grad_accum_steps),
        distributed_impl=args.distributed_impl,
    )
