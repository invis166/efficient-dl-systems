# Experimental setup

* torch version: 2.1.2
* GPU: 2x nvidia RTX A4000
* number of processes by default is 2 if other is not said

# Training loop benchmark
Selfmade distributed training
| Rank | Time                       | Max Memory Allocated |
|------|----------------------------|----------------------|
| 0    | 29.588 seconds | 170099200 bytes      |
| 1    | 29.632 seconds | 170098688 bytes      |
loss: 3.2136, acc: 0.2734375


pytorch Distributed Data Parallel
| Rank | Time                     | Max Memory Allocated |
|------|--------------------------|----------------------|
| 0    | 29.227 seconds | 173903360 bytes     |
| 1    | 29.279 seconds | 174062592 bytes     |
loss: 3.2149, acc: 0.27734375


how to reproduce pytoch version: `CUDA_VISIBLE_DEVICES=X,Y torchrun --nproc_per_node 2 ddp_cifar100.py --num-epochs 16 --grad-accum-steps 4 --distributed-impl selfmade`

how to reproduce selfmade version: `CUDA_VISIBLE_DEVICES=X,Y torchrun --nproc_per_node 2 ddp_cifar100.py --num-epochs 16 --grad-accum-steps 4 --distributed-impl pytorch`

In practice, those results (especially loss and acc) are not reproducible, each run coverges to a different optimum, alhtough the seed is fixed. Anyway, on a average both implementations converges near to the same result. Performance is the same and pytorch has a slightly larger memory consumtion.

# SyncBatchNorm benchmark
10000 runs for each parameter set

Pytorch BN
| batch_size | hidden_dim | num_workers | median time | max memory allocated |
|------------|------------|-------------|-------------|----------------------|
| 32         | 128        | 2           | 0.00052     | 41984                |
| 32         | 256        | 2           | 0.00047     | 80896                |
| 32         | 512        | 2           | 0.00053     | 158720               |
| 32         | 1024       | 2           | 0.00047     | 314368               |
| 64         | 128        | 2           | 0.00045     | 74752                |
| 64         | 256        | 2           | 0.00046     | 146432               |
| 64         | 512        | 2           | 0.00048     | 289792               |
| 64         | 1024       | 2           | 0.00048     | 576512               |


Selfmade BN
| batch_size | hidden_dim | num_workers | median time | max memory allocated |
|------------|------------|-------------|-------------|----------------------|
| 32         | 128        | 2           | 0.00037     | 101888               |
| 32         | 256        | 2           | 0.00046     | 203264               |
| 32         | 512        | 2           | 0.00037     | 406016               |
| 32         | 1024       | 2           | 0.00040     | 811520               |
| 64         | 128        | 2           | 0.00036     | 200192               |
| 64         | 256        | 2           | 0.00036     | 399872               |
| 64         | 512        | 2           | 0.00039     | 799232               |
| 64         | 1024       | 2           | 0.00038     | 1597952              |

to reproduce: `CUDA_VISIBLE_DEVICES=X,Y python syncbn_benchmark.py`

in general, custom implementation is turned out to be faster but is has larger memory consumption.